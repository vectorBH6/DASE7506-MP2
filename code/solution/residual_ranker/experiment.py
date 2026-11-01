"""Four residual trials, a prespecified feature fallback, then one frozen test."""
import os
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[variable]="1"
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import numpy as np

from common import DATA,load_bank,parse_answer,read_jsonl,validate_choices
from solution.group_ranker.features import answer_groups
from solution.handcrafted_ranker.experiment import (
    CODE,CHILD_ENV,accuracy,cpu_name,digest,interval,measured_command,write,
)

OLD=CODE/"solution/runs/handcrafted_v1/math"
NEW_METHOD="solution/residual_ranker/submission.py"
OLD_METHOD="solution/handcrafted_ranker/submission_linear.py"


def evaluate(checkpoint,split,output,method):
    output.mkdir(parents=True)
    commands=[
        [sys.executable,"predict.py","--method",method,"--checkpoint",str(checkpoint),"--split",split,
         "--output",str(output/"predictions.json")],
        [sys.executable,"score_predictions.py","--split",split,"--predictions",str(output/"predictions.json"),
         "--output",str(output/"score.json")],
    ]
    tick=time.perf_counter()
    processes=[measured_command(command) for command in commands]
    result={"official":json.loads((output/"score.json").read_text()),"total_wall_seconds":time.perf_counter()-tick,
            "processes":processes,"total_peak_threads":max(p["total_peak_threads"] for p in processes),
            "inference_asset_bytes":sum(path.stat().st_size for path in checkpoint.iterdir())}
    write(output/"summary.json",result)
    return result


def train(directory,schema,hidden,residual_l2s,linear_l2s,log):
    command=[sys.executable,"-m","solution.residual_ranker.train","--output",str(directory),
             "--schema",schema,"--hidden",str(hidden),"--residual-l2",*[str(x) for x in residual_l2s],
             "--linear-l2",*[str(x) for x in linear_l2s]]
    with log.open("w") as stream:
        subprocess.run(command,cwd=CODE,env=CHILD_ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=json.loads((directory/"training.json").read_text())
    selected=report["search"][report["selected_search_index"]]
    q=24 if schema=="math" else 40
    return {"train_oof":selected["oof"],"hidden":hidden,"schema":schema,"linear_l2":selected["linear_l2"],
            "residual_l2":selected["residual_l2"],"training_seconds":report["training_seconds"],
            "all_optimizers_converged":all(f["converged"] for entry in report["search"] for f in entry["folds"]) and report["final_optimizer"]["converged"],
            "parameter_count":2088+(q+2)*hidden}


def freeze(output,checkpoint,method):
    paths=list((CODE/"solution").glob("*.py"))
    for package in ("group_ranker","handcrafted_ranker","residual_ranker","baselines"):
        paths+=list((CODE/"solution"/package).glob("*.py"))
    paths+=[CODE/"solution/residual_ranker/README.md",CODE/"solution/requirements-train.txt"]
    paths+=[CODE/name for name in ("common.py","predict.py","score_predictions.py","requirements.txt")]
    paths+=list(checkpoint.iterdir())
    paths+=[output/"plan.json",output/"selection.json"]
    paths=sorted(set(paths))
    record={"created_utc":datetime.now(timezone.utc).isoformat(),"method":method,
            "checkpoint":str(checkpoint.relative_to(CODE)),
            "sha256":{str(path.relative_to(CODE)):digest(path) for path in paths},
            "test_policy":"One method selected on validation before test. No tuning or reselection after test."}
    record["freeze_id"]=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    with tarfile.open(output/"source.tar","w") as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(CODE)))
    record["source_archive_sha256"]=digest(output/"source.tar")
    write(output/"freeze.json",record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--test",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():parser.error("New output directory required")
    output.mkdir(parents=True)
    start=time.perf_counter()
    plan={"created_utc":datetime.now(timezone.utc).isoformat(),"seed":7506,"folds":5,
          "initial_grid":{"schema":"math","hidden":[8,16],"residual_l2":[.001,.01],"linear_l2":.01},
          "fallback_trigger":"No initial residual trial has validation correct > incumbent (733). Evaluated before any new test.",
          "fallback":[{"schema":"expanded","hidden":0,"linear_l2":[.003,.01,.03]},
                      {"schema":"expanded","hidden":8,"linear_l2":"OOF winner of expanded linear","residual_l2":[.001,.01]}],
          "selection":"Validation correct descending; train OOF accuracy descending; train OOF NLL ascending; fewer parameters; fixed trial order.",
          "within_trial_selection":"Train OOF accuracy descending, NLL ascending, stronger linear L2, stronger residual L2.",
          "optimizer":{"algorithm":"L-BFGS-B","maxiter":300,"ftol":1e-8,"gtol":1e-5,"warm_start":"train-fold linear model only"},
          "environment":{"python":sys.version.split()[0],"numpy":np.__version__,"cpu":cpu_name()},
          "test_policy":"Previously published test scores are known; current test is not used to decide fallback, hyperparameters or model selection."}
    write(output/"plan.json",plan)
    command=[sys.executable,"-m","unittest","solution.group_ranker.test_group_ranker",
             "solution.handcrafted_ranker.test_handcrafted","solution.handcrafted_ranker.test_optimization",
             "solution.residual_ranker.test_residual","-v"]
    tests=subprocess.run(command,cwd=CODE,env=CHILD_ENV,capture_output=True,text=True)
    write(output/"unit_tests.json",{"command":command,"returncode":tests.returncode,
                                  "stdout":tests.stdout,"stderr":tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stderr)
    print("27 tests passed.",flush=True)
    original=json.loads((OLD/"training.json").read_text())
    oof=next(entry["oof"] for entry in original["search"] if entry["l2"]==original["selected_l2"])
    records={"incumbent":{"train_oof":oof,"hidden":0,"schema":"math","linear_l2":.01,"residual_l2":0,
                           "method":OLD_METHOD,"checkpoint":str((OLD/"checkpoint").relative_to(CODE)),
                           "parameter_count":2088,
                           **evaluate(OLD/"checkpoint","validation",output/"incumbent/validation",OLD_METHOD)}}
    archived=json.loads((OLD/"validation/predictions.json").read_text())["selected"]
    assert json.loads((output/"incumbent/validation/predictions.json").read_text())["selected"]==archived

    def trial(name,schema,hidden,residual_l2s,linear_l2s):
        info=train(output/name,schema,hidden,residual_l2s,linear_l2s,output/f"{name}-train.log")
        records[name]={**info,"method":NEW_METHOD,"checkpoint":str((output/name/"checkpoint").relative_to(CODE)),
                       **evaluate(output/name/"checkpoint","validation",output/name/"validation",NEW_METHOD)}
        print(f"{name}: OOF={info['train_oof']['accuracy']:.4f}%, validation={records[name]['official']['correct']}, converged={info['all_optimizers_converged']}",flush=True)

    for hidden in (8,16):
        for penalty in (.001,.01):
            trial(f"residual_h{hidden}_r{penalty:g}","math",hidden,[penalty],[.01])
    fallback=max(record["official"]["correct"] for name,record in records.items() if name!="incumbent")<=records["incumbent"]["official"]["correct"]
    write(output/"fallback_decision.json",{"triggered":fallback,"basis":"initial residual validation results only"})
    if fallback:
        print("No initial validation gain; running the predefined 16-feature extension.",flush=True)
        trial("expanded_linear","expanded",0,[.01],[.003,.01,.03])
        trial("expanded_residual","expanded",8,[.001,.01],[records["expanded_linear"]["linear_l2"]])

    bank=load_bank("validation")
    answers=[parse_answer(row["answer"]) for row in read_jsonl(DATA/"gsm8k_validation.jsonl")]
    counts=np.array([len(answer_groups(row["candidates"])) for row in bank])
    hard=counts>=2
    correctness={}
    for name,record in records.items():
        prediction=json.loads((output/name/"validation/predictions.json").read_text())
        validate_choices(bank,prediction["ids"],prediction["selected"])
        correct=np.array([parse_answer(row["candidates"][j]["text"])==answer
                          for row,j,answer in zip(bank,prediction["selected"],answers)])
        assert int(correct.sum())==record["official"]["correct"]
        correctness[name]=correct
        record["conflict"]=accuracy(correct[hard])
        record["by_group_count"]={str(g):accuracy(correct[counts==g]) for g in sorted(set(counts))}
    base=correctness["incumbent"]
    for name,correct in correctness.items():
        records[name]["corrected_vs_incumbent"]=int(np.sum(~base & correct))
        records[name]["regressed_vs_incumbent"]=int(np.sum(base & ~correct))
        records[name]["difference_95ci_pp"]=interval(correct.astype(int)-base.astype(int))
    names=tuple(records)
    selected=min(names,key=lambda name:(-records[name]["official"]["correct"],
                 -records[name]["train_oof"]["accuracy"],records[name]["train_oof"]["nll"],
                 records[name]["parameter_count"],names.index(name)))
    chosen=records[selected]
    write(output/"validation_summary.json",{"methods":records})
    selection={"selected":selected,"method":chosen["method"],"checkpoint":chosen["checkpoint"],
               "policy":plan["selection"],"validation":chosen["official"]}
    write(output/"selection.json",selection)
    print(f"Selected before test: {selected}",flush=True)
    test=None
    if args.test:
        checkpoint=CODE/chosen["checkpoint"]
        frozen=freeze(output,checkpoint,chosen["method"])
        test=evaluate(checkpoint,"test",output/"test",chosen["method"])
        test.update(freeze_id=frozen["freeze_id"],
                    frozen_files_unchanged=all(digest(CODE/name)==value for name,value in frozen["sha256"].items()),
                    reference_cpu_verified=False,local_under_30_seconds=test["total_wall_seconds"]<=30,
                    assets_within_16_mib=test["inference_asset_bytes"]<=16*2**20)
        write(output/"test/summary.json",test)
        assert test["frozen_files_unchanged"]
        print(f"Frozen test: {test['official']}, wall={test['total_wall_seconds']:.3f}s",flush=True)
    lines=["# 非线性残差头与手工扩展实验","","试验网格、回退条件和选型规则在训练前写入 plan.json。训练按题五折；只在训练 fold 拟合标准化与线性 warm start。","",
           "| 配置 | OOF 冲突题 | Validation | 冲突题 | 纠正/改错（对原6维模型） |",
           "|---|---:|---:|---:|---:|"]
    for name,record in records.items():
        lines.append(f"| {name} | {record['train_oof']['accuracy']:.2f}% | {record['official']['correct']}/{record['official']['n']} ({record['official']['accuracy']:.2f}%) | {record['conflict']['accuracy']:.2f}% | {record['corrected_vs_incumbent']}/{record['regressed_vs_incumbent']} |")
    lines+=["",f"是否触发手工扩展：{fallback}。按预设规则选定：**{selected}**。",
            f"预测入口：{chosen['method']}。Checkpoint：{chosen['checkpoint']}。",
            "隐藏层/正则搜索记录、优化收敛状态、成对区间见各 training.json 及 validation_summary.json。",
            "公开 test 历史成绩已知；本轮未据 test 调参、扩展特征或重选模型。"]
    if test:
        lines+=["",f"冻结 test：**{test['official']['correct']}/{test['official']['n']}，{test['official']['accuracy']:.8f}%**。",
                f"本机完整预测加评分 {test['total_wall_seconds']:.3f} 秒，线程峰值 {test['total_peak_threads']}，推理资产 {test['inference_asset_bytes']} 字节；参考 CPU 尚未验证。"]
    lines+=["",f"实验外部总用时：{time.perf_counter()-start:.1f} 秒。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
