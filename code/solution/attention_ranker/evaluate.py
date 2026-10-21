"""官方 validation 比较及冻结后的 test；复用统一计时、线程与分析工具。"""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np

from common import DATA, PROTOCOL, load_inputs, read_jsonl, parse_answer, validate_choices
from solution.evaluate import (THREAD_VARIABLES, run_official, accuracy, bootstrap_interval, confidence_summary)
from solution.group_ranker.features import answer_groups
from solution.attention_ranker.submission import digest, predict_profiled

CODE = Path(__file__).resolve().parents[2]
ROOT, RUNS = CODE.parent, CODE / "solution/runs"
NAMES = ("gated", "attention", "gated_ensemble", "attention_ensemble")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def compare(correct, reference):
    difference = correct.astype(int) - reference.astype(int)
    return {"corrected": int(np.sum(correct & ~reference)), "regressed": int(np.sum(~correct & reference)),
        "net_correct": int(difference.sum()), "difference_pp": float(100*difference.mean()),
        "difference_95ci_pp": bootstrap_interval(difference), "bootstrap_seed": 7506,
        "bootstrap_draws": 2000, "multiple_comparison_adjustment": False}


def run_official_pair(split, name, directory, run, env, freeze=None):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = run / name / "checkpoint"
    prediction = [sys.executable, "predict.py", "--method", "solution/attention_ranker/submission.py",
        "--checkpoint", str(checkpoint.relative_to(CODE)), "--split", split,
        "--output", str((directory / "predictions.json").relative_to(CODE))]
    scoring = [sys.executable, "score_predictions.py", "--split", split,
        "--predictions", str((directory / "predictions.json").relative_to(CODE)),
        "--output", str((directory / "score.json").relative_to(CODE))]
    tick = time.perf_counter()
    processes = [run_official(prediction, env)]
    if freeze:
        write(directory / "prediction_freeze.json", {"created_utc": now(), "freeze_id": freeze["freeze_id"],
            "predictions_sha256": digest(directory / "predictions.json")})
    processes.append(run_official(scoring, env))
    elapsed = time.perf_counter() - tick
    record = {"official": read(directory / "score.json"), "total_wall_seconds": elapsed,
        "processes": processes, "inference_asset_bytes": sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()),
        "checkpoint": str(checkpoint.relative_to(CODE)),
        "measurement": "新预测进程和独立评分进程的外部计时；含启动、导入、哈希、加载、预处理、全部成员、输出与预测冻结；缓存未控制。"}
    if freeze:
        record.update(freeze_id=freeze["freeze_id"], freeze_manifest_sha256=digest(directory / "freeze.json"))
    print(json.dumps({"split": split, "name": name, "official": record["official"], "seconds": elapsed}), flush=True)
    return record


def analyze(split, run, records, folders):
    bank, arrays = load_inputs(split)
    gold = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    assert [(r["id"],r["question"]) for r in bank] == [(r["id"],r["question"]) for r in gold]
    answers = [parse_answer(row["answer"]) for row in gold]
    groups = [answer_groups(row["candidates"]) for row in bank]
    counts = np.array([len(group) for group in groups])
    reference = {"m1_original": ("mlp_text_v1", "m1"), "ensemble_original": ("ensemble_v1", "ensemble"),
                 "linear": ("linear_v1", "linear")}
    files = {name: folder / "predictions.json" for name,folder in folders.items()}
    for name,(experiment,key) in reference.items():
        files[name] = (RUNS / experiment / "test/predictions.json" if split == "test"
                       else RUNS / "ensemble_v1/validation" / f"{key}-predictions.json")
    correctness = {}
    for name,path in files.items():
        prediction = read(path)
        validate_choices(bank,prediction["ids"],prediction["selected"])
        correct = np.array([parse_answer(row["candidates"][j]["text"]) == answer
                           for row,j,answer in zip(bank,prediction["selected"],answers)])
        if name in records:
            assert int(correct.sum()) == records[name]["official"]["correct"]
        correctness[name] = correct
    for name in records:
        choices,profile = predict_profiled(bank,arrays,run/name/"checkpoint",include_scores=True)
        assert choices == read(files[name])["selected"]
        correct = correctness[name]
        scores = profile.pop("group_scores")
        records[name]["analysis"] = {"conflict": accuracy(correct[counts>=2]),
            "by_group_count": {str(g): accuracy(correct[counts==g]) for g in sorted(set(counts))},
            "confidence": confidence_summary(scores,groups,answers,correct,counts>=2), "profile_replay": profile,
            "comparisons": {other:compare(correct,value) for other,value in correctness.items() if other != name}}
        with np.load(run/name/"oof.npz",allow_pickle=False) as archive:
            from solution.group_ranker.training import metrics
            records[name]["train_oof"] = metrics(archive["scores"],archive["labels"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--phase",choices=("validation","test"),required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to(RUNS):
        parser.error("实验目录须在 solution/runs 下")
    plan=read(run/"plan.json")
    historical=read(run/"historical_sha256.json")
    assert all(digest(ROOT/path)==value for path,value in historical.items())
    directories={name:run/name/args.phase for name in NAMES}
    if any(path.exists() for path in directories.values()):
        parser.error("已有评测不覆盖，请使用新的实验目录")
    env={**os.environ,**dict.fromkeys(THREAD_VARIABLES,"3")}
    env.pop("PYTHONPATH",None)
    freezes={}
    if args.phase == "validation":
        command=[sys.executable,"-m","unittest","solution.attention_ranker.test_attention",
                 "solution.group_ranker.test_group_ranker","solution.group_ranker.test_ensemble","-v"]
        check=subprocess.run(command,cwd=CODE,env={**os.environ,**dict.fromkeys(THREAD_VARIABLES,"1")},
                             text=True,capture_output=True,check=True)
        write(run/"checks.json",{"command":command,"returncode":check.returncode,
            "stdout":check.stdout,"stderr":check.stderr,"performed_before_new_test":True})
    else:
        validation=read(run/"validation_summary.json")
        sources=sorted([*Path(__file__).parent.glob("*.py"),*(CODE/"solution/group_ranker").glob("*.py"),
            *[CODE/name for name in ("common.py","predict.py","score_predictions.py","check_data.py",
                "requirements.txt","solution/__init__.py","solution/evaluate.py","solution/requirements-train.txt")],ROOT/"guide/GUIDE.md"])
        with tarfile.open(run/"source.tar","w") as archive:
            for path in sources:
                archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
        metadata=[run/file for file in ("source.tar","plan.json","historical_sha256.json","checks.json",
                                        "training_summary.json","validation_summary.json")]
        metadata += [run/name/file for name in NAMES for file in ("oof.npz", "training.json" if name in ("gated","attention") else "preparation.json")]
        common={str(path.relative_to(ROOT)):digest(path) for path in sources+metadata}
        for name in NAMES:
            output=directories[name]
            output.mkdir()
            assets=[p for p in (run/name/"checkpoint").rglob("*") if p.is_file()]
            freeze={"created_utc":now(),"protocol":PROTOCOL,"name":name,"test_policy":plan["test_policy"],
                "validation_selected":validation["selection"]["name"],"source_archive":"../../source.tar",
                "sha256":{**common,**{str(p.relative_to(ROOT)):digest(p) for p in assets}}}
            import hashlib
            freeze["freeze_id"]=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()
            write(output/"freeze.json",freeze)
            freezes[name]=freeze
    records={name:run_official_pair(args.phase,name,directories[name],run,env,freezes.get(name)) for name in NAMES}
    analyze(args.phase,run,records,directories)
    cpu=next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    report={"created_utc":now(),"split":args.phase,"methods":records,"scope":plan["test_policy"],
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"cpu":cpu,
                       "thread_budget_total":4,"coordinator_threads":1,"child_threads":3},
        "reference_cpu_verified":False,"historical_artifacts_unchanged":True,"historical_artifact_count":len(historical)}
    assert all(digest(ROOT/path)==value for path,value in historical.items())
    if args.phase == "validation":
        original=read(RUNS/"ensemble_v1/validation/summary.json")["methods"]["ensemble"]
        candidates={**records,"ensemble_v1":original}
        eligible=[name for name,r in candidates.items() if r["inference_asset_bytes"]<=16*2**20 and r["total_wall_seconds"]<=30]
        selected=min(eligible,key=lambda name:(-candidates[name]["official"]["accuracy"],
            candidates[name].get("enabled_input_dimensions",2080),candidates[name]["inference_asset_bytes"],
            candidates[name]["total_wall_seconds"],name))
        report["selection"]={"name":selected,"rule":plan["selection_rule"],"all_new_test_scores_not_yet_run":True,
            "checkpoint":str((RUNS/"ensemble_v1/checkpoint" if selected=="ensemble_v1" else run/selected/"checkpoint").relative_to(CODE))}
        write(run/"validation_summary.json",report)
        print(json.dumps({"validation_selected":selected,"validation_counts":{name:r["official"]["correct"] for name,r in records.items()}}),flush=True)
    else:
        for name in NAMES:
            record,folder=records[name],directories[name]
            record["frozen_files_unchanged"]=all(digest(ROOT/path)==value for path,value in freezes[name]["sha256"].items())
            assert record["frozen_files_unchanged"]
            assert digest(folder/"predictions.json")==read(folder/"prediction_freeze.json")["predictions_sha256"]
            write(folder/"summary.json",{**record,"environment":report["environment"],"scope":plan["test_policy"]})
        report.update(validation_selection=validation["selection"],test_used_for_selection=False,
                      training=read(run/"training_summary.json"))
        write(run/"summary.json",report)
        print(json.dumps({"selected":report["validation_selection"],"test_counts":{name:r["official"]["correct"] for name,r in records.items()}}),flush=True)


if __name__ == "__main__":
    main()
