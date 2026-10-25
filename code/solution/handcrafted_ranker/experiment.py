"""Four prespecified feature ablations; select on validation, optionally freeze/test once."""
import os
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np

from common import DATA, load_bank, parse_answer, read_jsonl, validate_choices
from solution.group_ranker.features import answer_groups
from solution.handcrafted_ranker.arithmetic import (
    FEATURE_SETS, GROUP_FEATURE_NAMES, candidate_features, checked_equations,
)

CODE = Path(__file__).resolve().parents[2]
VARIANTS = ("basic", "math", "math_question", "full")
CHILD_ENV = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "3"), "PYTHONDONTWRITEBYTECODE": "1"}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_name():
    return next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")), platform.processor())


def thread_count(pid):
    try:
        return len(list(Path(f"/proc/{pid}/task").iterdir()))
    except FileNotFoundError:
        return 0


def measured_command(command):
    tick = time.perf_counter()
    peak_parent = peak_child = peak_total = 0
    process = subprocess.Popen(command, cwd=CODE, env=CHILD_ENV,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while process.poll() is None:
        parent, child = thread_count(os.getpid()), thread_count(process.pid)
        peak_parent, peak_child = max(peak_parent, parent), max(peak_child, child)
        peak_total = max(peak_total, parent + child)
        time.sleep(0.001)
    stdout, stderr = process.communicate()
    record = {"command": command, "wall_seconds": time.perf_counter() - tick,
              "coordinator_peak_threads": peak_parent, "child_peak_threads": peak_child,
              "total_peak_threads": peak_total, "returncode": process.returncode,
              "stdout": stdout, "stderr": stderr}
    if process.returncode:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    return record


def evaluate(checkpoint, split, output):
    output.mkdir()
    commands = [
        [sys.executable, "predict.py", "--method", "solution/handcrafted_ranker/submission_linear.py",
         "--checkpoint", str(checkpoint), "--split", split, "--output", str(output/"predictions.json")],
        [sys.executable, "score_predictions.py", "--split", split,
         "--predictions", str(output/"predictions.json"), "--output", str(output/"score.json")],
    ]
    tick = time.perf_counter()
    processes = [measured_command(command) for command in commands]
    result = {"split": split, "official": json.loads((output/"score.json").read_text()),
              "total_wall_seconds": time.perf_counter() - tick, "processes": processes,
              "total_peak_threads": max(p["total_peak_threads"] for p in processes),
              "inference_asset_bytes": sum(p.stat().st_size for p in checkpoint.iterdir())}
    write(output/"summary.json", result)
    return result


def accuracy(correct):
    return {"correct": int(correct.sum()), "n": int(correct.size),
            "accuracy": float(100 * correct.mean()) if correct.size else None}


def interval(difference):
    rng = np.random.default_rng(7506)
    sampled = rng.choice(difference, size=(2000, len(difference)), replace=True)
    return (100 * np.quantile(sampled.mean(axis=1), [0.025, 0.975])).tolist()


def diagnostics(split):
    counts = Counter()
    examples = {"valid": [], "invalid": []}
    tick = time.perf_counter()
    for row in load_bank(split):
        features = candidate_features(row["question"], row["candidates"])
        for i, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            steps = checked_equations(text)
            counts["candidates"] += 1
            counts["candidates_with_equals"] += "=" in text
            counts["equals_lines"] += sum("=" in line for line in text.splitlines())
            counts["candidates_with_checked_equations"] += bool(steps)
            counts["checked_equations"] += len(steps)
            counts["invalid_equations"] += sum(not step.valid for step in steps)
            counts["valid_answer_candidates"] += parse_answer(text) is not None
            counts["last_equation_matches"] += bool(features[i, 3])
            counts["trace_reaches"] += bool(features[i, 4])
            for step in steps:
                key = "valid" if step.valid else "invalid"
                if len(examples[key]) < 20:
                    examples[key].append({"id": row["id"], "candidate": i, "line": step.line,
                                          "lhs": str(step.lhs), "rhs": str(step.rhs)})
    return {"counts": dict(counts), "examples": examples, "wall_seconds": time.perf_counter() - tick,
            "notes": "No references read. Unsupported expressions are skipped, not marked invalid."}


def freeze(output, selected, plan):
    # Include project source only; skip runs and local virtual environments.
    paths = list((CODE/"solution").glob("*.py"))
    for package in ("group_ranker", "handcrafted_ranker", "baselines"):
        paths += list((CODE/"solution"/package).glob("*.py"))
    paths += [CODE/"solution/requirements-train.txt", CODE/"solution/handcrafted_ranker/README.md"]
    paths += [CODE/name for name in ("common.py", "predict.py", "score_predictions.py", "requirements.txt")]
    paths += list((selected/"checkpoint").iterdir())
    paths += [output/"plan.json", output/"selection.json"]
    paths = sorted(set(paths))
    hashes = {str(path.relative_to(CODE)): digest(path) for path in paths}
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "selection_basis": plan["selection_policy"], "sha256": hashes,
              "checkpoint": str((selected/"checkpoint").relative_to(CODE)),
              "test_policy": "Only the selected method; no post-test tuning or reselection."}
    record["freeze_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    with tarfile.open(output/"source.tar", "w") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(CODE)))
    record["source_archive_sha256"] = digest(output/"source.tar")
    write(output/"freeze.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test", action="store_true", help="After validation selection, freeze and score only the winner")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    output.mkdir(parents=True)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "variants": list(VARIANTS),
            "seed": 7506, "folds": 5, "l2": [0.0001, 0.001, 0.01],
            "feature_names": list(GROUP_FEATURE_NAMES),
            "enabled_extra_indices": {name: list(FEATURE_SETS[name]) for name in VARIANTS},
            "l2_selection": "Train OOF accuracy, then OOF NLL, then stronger L2",
            "selection_policy": "Validation correct descending; train OOF accuracy descending; train OOF NLL ascending; fewer extra features; variant order. Fixed before training; no test access for selection.",
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "cpu": cpu_name(),
                            "platform": platform.platform(), "coordinator_threads": 1, "inference_child_threads": 3},
            "known_history": "linear_v1 was previously tested (375/467); this is a follow-up experiment, not an untouched test benchmark."}
    write(output/"plan.json", plan)
    command = [sys.executable, "-m", "unittest", "solution.group_ranker.test_group_ranker",
               "solution.handcrafted_ranker.test_handcrafted", "-v"]
    tests = subprocess.run(command, cwd=CODE, env=CHILD_ENV, capture_output=True, text=True)
    write(output/"unit_tests.json", {"command": command, "returncode": tests.returncode,
                                    "stdout": tests.stdout, "stderr": tests.stderr})
    if tests.returncode:
        raise RuntimeError(tests.stderr)
    print("Unit tests passed.", flush=True)

    records = {}
    for name in VARIANTS:
        trial = output/name
        command = [sys.executable, "-m", "solution.handcrafted_ranker.train_linear",
                   "--feature-set", name, "--output", str(trial)]
        tick = time.perf_counter()
        with (output/f"{name}-train.log").open("w") as log:
            subprocess.run(command, cwd=CODE, env=CHILD_ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
        training = json.loads((trial/"training.json").read_text())
        oof = next(entry["oof"] for entry in training["search"] if entry["l2"] == training["selected_l2"])
        result = evaluate(trial/"checkpoint", "validation", trial/"validation")
        records[name] = {"feature_set": name, "train_oof": oof, "selected_l2": training["selected_l2"],
                         "training_wall_seconds": time.perf_counter() - tick - result["total_wall_seconds"],
                         **result}
        print(f"{name}: OOF={oof['accuracy']:.4f}%, validation={result['official']['accuracy']:.4f}%, wall={result['total_wall_seconds']:.3f}s", flush=True)

    # Only validation references are used for paired analysis and model selection.
    bank = load_bank("validation")
    answers = [parse_answer(row["answer"]) for row in read_jsonl(DATA/"gsm8k_validation.jsonl")]
    groups = [answer_groups(row["candidates"]) for row in bank]
    hard = np.array([len(g) >= 2 for g in groups])
    patterns = np.array(["+".join(map(str, sorted(map(len, g.values()), reverse=True))) for g in groups])
    group_counts = np.array([len(g) for g in groups])
    correctness = {}
    frozen_v1 = json.loads((CODE/"solution/runs/linear_v1/validation/linear-predictions.json").read_text())
    for name in VARIANTS:
        prediction = json.loads((output/name/"validation/predictions.json").read_text())
        validate_choices(bank, prediction["ids"], prediction["selected"])
        correct = np.array([parse_answer(row["candidates"][j]["text"]) == answer
                            for row, j, answer in zip(bank, prediction["selected"], answers)])
        if int(correct.sum()) != records[name]["official"]["correct"]:
            raise ValueError("Official/analysis score mismatch")
        correctness[name] = correct
        records[name]["conflict"] = accuracy(correct[hard])
        records[name]["by_group_count"] = {str(g): accuracy(correct[group_counts == g]) for g in sorted(set(group_counts))}
        records[name]["by_vote_pattern"] = {p: accuracy(correct[patterns == p]) for p in sorted(set(patterns))}
        records[name]["same_choices_as_frozen_v1"] = prediction["selected"] == frozen_v1["selected"]
    baseline = correctness["basic"]
    for name, correct in correctness.items():
        records[name]["corrected_vs_basic"] = int(np.sum(~baseline & correct))
        records[name]["regressed_vs_basic"] = int(np.sum(baseline & ~correct))
        records[name]["difference_95ci_pp"] = interval(correct.astype(int) - baseline.astype(int))
        records[name]["conflict_difference_95ci_pp"] = interval(correct[hard].astype(int) - baseline[hard].astype(int))

    selected_name = min(VARIANTS, key=lambda name: (
        -records[name]["official"]["correct"], -records[name]["train_oof"]["accuracy"],
        records[name]["train_oof"]["nll"], len(FEATURE_SETS[name]), VARIANTS.index(name)))
    write(output/"validation_summary.json", {"methods": records, "conflict_n": int(hard.sum())})
    selection = {"selected": selected_name, "policy": plan["selection_policy"],
                 "selected_checkpoint": str((output/selected_name/"checkpoint").relative_to(CODE)),
                 "selected_validation": records[selected_name]["official"]}
    write(output/"selection.json", selection)
    print(f"Selected on validation: {selected_name}", flush=True)
    write(output/"arithmetic_diagnostics.json", {split: diagnostics(split) for split in ("train", "validation")})

    test = None
    if args.test:
        frozen = freeze(output, output/selected_name, plan)
        test = evaluate(output/selected_name/"checkpoint", "test", output/"test")
        test["freeze_id"] = frozen["freeze_id"]
        test["frozen_files_unchanged"] = all(digest(CODE/name) == value for name, value in frozen["sha256"].items())
        test["reference_cpu_verified"] = False
        test["local_under_30_seconds"] = test["total_wall_seconds"] <= 30
        test["assets_within_16_mib"] = test["inference_asset_bytes"] <= 16 * 2**20
        write(output/"test/summary.json", test)
        if not test["frozen_files_unchanged"]:
            raise RuntimeError("Frozen files changed")
        print(f"Frozen test: {test['official']}, wall={test['total_wall_seconds']:.3f}s, peak_threads={test['total_peak_threads']}", flush=True)

    lines = ["# 手工特征消融实验", "", "四个配置和选型规则在训练前写入 plan.json；每个配置使用相同 train 五折，L2 仅由 train OOF 选择。", "",
             "| 配置 | 新增维数 | OOF 冲突题 | validation | validation 冲突题 | 纠正/改错（对基本模型） |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in records.items():
        lines.append(f"| {name} | {len(FEATURE_SETS[name])} | {result['train_oof']['accuracy']:.2f}% | {result['official']['correct']}/798 ({result['official']['accuracy']:.2f}%) | {result['conflict']['accuracy']:.2f}% | {result['corrected_vs_basic']}/{result['regressed_vs_basic']} |")
    lines += ["", f"按预设 validation 规则选择：**{selected_name}**。", "",
              "完整逐题结果、分层指标和成对 bootstrap 区间见 validation_summary.json。算术覆盖率及输入样例见 arithmetic_diagnostics.json。",
              "本轮属于已知公开 test 历史成绩后的后续研究，不能声称测试集完全未见；本轮没有根据 test 调参或重选模型。"]
    if test:
        lines += ["", f"冻结 test：{test['official']['correct']}/{test['official']['n']}，{test['official']['accuracy']:.8f}%；本机完整预测加评分 {test['total_wall_seconds']:.3f} 秒；总线程峰值 {test['total_peak_threads']}；推理资产 {test['inference_asset_bytes']} 字节。",
                  "该时间是本机测量，非 Xeon Platinum 8457C 资格认证。源码和 checkpoint 的冻结哈希见 freeze.json，快照见 source.tar。"]
    (output/"REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
