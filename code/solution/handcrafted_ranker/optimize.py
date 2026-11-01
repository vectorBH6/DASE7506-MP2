"""Bounded staged optimization: train-only searches, validation selection, one frozen test."""
import os
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[variable]="1"

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import DATA,load_bank,parse_answer,read_jsonl,validate_choices
from solution.handcrafted_ranker.arithmetic import FEATURE_SETS
from solution.handcrafted_ranker.experiment import (
    CHILD_ENV,CODE,accuracy,cpu_name,digest,evaluate,freeze,interval,write,
)
from solution.group_ranker.features import answer_groups

INCUMBENT=CODE/"solution/runs/handcrafted_v1/math"
STAGES=("expanded_l2","block_l2","interactions")


def training_result(directory):
    report=json.loads((directory/"training.json").read_text())
    index=report.get("selected_search_index")
    if index is None:
        index=next(i for i,entry in enumerate(report["search"])
                   if entry["l2"]==report["selected_l2"])
    selected=report["search"][index]
    return {"train_oof":selected["oof"],"selected_l2":selected["l2"],
            "handcrafted_l2_multiplier":selected.get("handcrafted_l2_multiplier",1.0),
            "feature_set":report["config"]["feature_set"],
            "training_seconds":report["training_seconds"]}


def train(directory,feature_set,l2s,multipliers,log):
    command=[sys.executable,"-m","solution.handcrafted_ranker.train_linear",
             "--feature-set",feature_set,"--output",str(directory),
             "--l2",*[str(value) for value in l2s],
             "--handcrafted-l2-multiplier",*[str(value) for value in multipliers]]
    with log.open("w") as stream:
        subprocess.run(command,cwd=CODE,env=CHILD_ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return training_result(directory)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--test",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    output.mkdir(parents=True)
    tick=time.perf_counter()
    plan={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "seed":7506,"folds":5,"maxiter":300,
        "incumbent_checkpoint":str((INCUMBENT/"checkpoint").relative_to(CODE)),
        "stages":[
            {"name":"expanded_l2","feature_set":"math","l2":[.01,.03,.1],"multipliers":[1]},
            {"name":"block_l2","feature_set":"math","l2":"OOF winner of expanded_l2","multipliers":[.3,1,3]},
            {"name":"interactions","feature_set":"math_interactions","l2":"OOF winner of expanded_l2","multipliers":[.3,1,3]},
        ],
        "within_stage_selection":"Train OOF accuracy descending, NLL ascending, stronger L2, multiplier nearest 1 in log space, stronger multiplier.",
        "final_selection":"Validation correct descending; train OOF accuracy descending; train OOF NLL ascending; fewer extra features; multiplier nearest 1 in log space; incumbent/stage order.",
        "interaction_definition":{"2072":"group count / 4 * mean invalid equation fraction",
                                  "2073":"group count / 4 * mean last valid equation matches"},
        "penalty_definition":"lambda for old columns; lambda*multiplier for columns 2064:2074; applied in objective after fold-local standardization.",
        "environment":{"python":sys.version.split()[0],"numpy":np.__version__,"cpu":cpu_name(),
                       "coordinator_threads":1,"inference_child_threads_max":3},
        "test_policy":"Existing public test scores are known. No test references or new test predictions used for search/selection; only one selected frozen method is scored.",
    }
    write(output/"plan.json",plan)
    test_command=[sys.executable,"-m","unittest","solution.group_ranker.test_group_ranker",
                  "solution.handcrafted_ranker.test_handcrafted","solution.handcrafted_ranker.test_optimization","-v"]
    unit=subprocess.run(test_command,cwd=CODE,env=CHILD_ENV,capture_output=True,text=True)
    write(output/"unit_tests.json",{"command":test_command,"returncode":unit.returncode,
                                   "stdout":unit.stdout,"stderr":unit.stderr})
    if unit.returncode:
        raise RuntimeError(unit.stderr)
    print("23 unit tests passed.",flush=True)

    records={}
    incumbent_output=output/"incumbent"
    incumbent_output.mkdir()
    records["incumbent"]={**training_result(INCUMBENT),
                         **evaluate(INCUMBENT/"checkpoint","validation",incumbent_output/"validation"),
                         "checkpoint":str((INCUMBENT/"checkpoint").relative_to(CODE))}
    archived=json.loads((INCUMBENT/"validation/predictions.json").read_text())["selected"]
    replay=json.loads((incumbent_output/"validation/predictions.json").read_text())["selected"]
    if archived!=replay:
        raise RuntimeError("Incumbent replay changed")
    print(f"incumbent: validation={records['incumbent']['official']['correct']}",flush=True)

    global_l2=None
    for name in STAGES:
        directory=output/name
        feature_set="math_interactions" if name=="interactions" else "math"
        l2s=[.01,.03,.1] if name=="expanded_l2" else [global_l2]
        multipliers=[1] if name=="expanded_l2" else [.3,1,3]
        info=train(directory,feature_set,l2s,multipliers,output/f"{name}-train.log")
        if name=="expanded_l2":
            global_l2=info["selected_l2"]
        records[name]={**info,**evaluate(directory/"checkpoint","validation",directory/"validation"),
                       "checkpoint":str((directory/"checkpoint").relative_to(CODE))}
        print(f"{name}: L2={info['selected_l2']}, multiplier={info['handcrafted_l2_multiplier']}, OOF={info['train_oof']['accuracy']:.4f}%, validation={records[name]['official']['correct']}",flush=True)

    bank=load_bank("validation")
    answers=[parse_answer(row["answer"]) for row in read_jsonl(DATA/"gsm8k_validation.jsonl")]
    groups=[answer_groups(row["candidates"]) for row in bank]
    group_counts=np.array([len(group) for group in groups])
    hard=group_counts>=2
    patterns=np.array(["+".join(map(str,sorted(map(len,group.values()),reverse=True))) for group in groups])
    correctness={}
    for name,record in records.items():
        prediction=json.loads((output/name/"validation/predictions.json").read_text())
        validate_choices(bank,prediction["ids"],prediction["selected"])
        correct=np.array([parse_answer(row["candidates"][j]["text"])==answer
                          for row,j,answer in zip(bank,prediction["selected"],answers)])
        if int(correct.sum())!=record["official"]["correct"]:
            raise ValueError("Official score mismatch")
        correctness[name]=correct
        record["conflict"]=accuracy(correct[hard])
        record["by_group_count"]={str(g):accuracy(correct[group_counts==g]) for g in sorted(set(group_counts))}
        record["by_vote_pattern"]={p:accuracy(correct[patterns==p]) for p in sorted(set(patterns))}
    baseline=correctness["incumbent"]
    for name,correct in correctness.items():
        records[name]["corrected_vs_incumbent"]=int(np.sum(~baseline & correct))
        records[name]["regressed_vs_incumbent"]=int(np.sum(baseline & ~correct))
        records[name]["difference_95ci_pp"]=interval(correct.astype(int)-baseline.astype(int))
        records[name]["conflict_difference_95ci_pp"]=interval(correct[hard].astype(int)-baseline[hard].astype(int))
    names=tuple(records)
    selected_name=min(names,key=lambda name:(
        -records[name]["official"]["correct"],-records[name]["train_oof"]["accuracy"],
        records[name]["train_oof"]["nll"],len(FEATURE_SETS[records[name]["feature_set"]]),
        abs(np.log(records[name]["handcrafted_l2_multiplier"])),names.index(name)))
    selected=records[selected_name]
    write(output/"validation_summary.json",{"methods":records,"conflict_n":int(hard.sum())})
    selection={"selected":selected_name,"policy":plan["final_selection"],
               "selected_checkpoint":selected["checkpoint"],"validation":selected["official"],
               "selected_l2":selected["selected_l2"],"handcrafted_l2_multiplier":selected["handcrafted_l2_multiplier"],
               "feature_set":selected["feature_set"]}
    write(output/"selection.json",selection)
    print(f"Selected before test: {selected_name}",flush=True)
    selected_directory=(CODE/selected["checkpoint"]).parent
    test=None
    if args.test:
        frozen=freeze(output,selected_directory,{"selection_policy":plan["final_selection"]})
        test=evaluate(selected_directory/"checkpoint","test",output/"test")
        test.update({
            "freeze_id":frozen["freeze_id"],
            "frozen_files_unchanged":all(digest(CODE/name)==value for name,value in frozen["sha256"].items()),
            "reference_cpu_verified":False,"local_under_30_seconds":test["total_wall_seconds"]<=30,
            "assets_within_16_mib":test["inference_asset_bytes"]<=16*2**20,
        })
        write(output/"test/summary.json",test)
        if not test["frozen_files_unchanged"]:
            raise RuntimeError("Frozen source changed")
        print(f"Frozen test: {test['official']}, wall={test['total_wall_seconds']:.3f}s",flush=True)
    lines=["# 有限优化实验","","所有搜索范围和选择规则在训练前写入 plan.json。使用相同 train 五折，标准化按 fold 拟合。每阶段仅将 OOF 选出的一个配置送至 validation。","",
           "| 阶段 | L2 | 手工块倍率 | OOF 冲突题 | Validation | 冲突题 | 纠正/改错（对原6维模型） |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name,record in records.items():
        lines.append(f"| {name} | {record['selected_l2']:g} | {record['handcrafted_l2_multiplier']:g} | {record['train_oof']['accuracy']:.2f}% | {record['official']['correct']}/{record['official']['n']} ({record['official']['accuracy']:.2f}%) | {record['conflict']['accuracy']:.2f}% | {record['corrected_vs_incumbent']}/{record['regressed_vs_incumbent']} |")
    lines+=["",f"按预设 validation 规则选择：**{selected_name}**。",
            f"Checkpoint：{selected['checkpoint']}。","",
            "各阶段训练日志与 training.json 保留全部搜索结果；validation_summary.json 保留分层和成对 bootstrap 区间。历史公开 test 成绩已知，本轮属于后续研究；未根据本轮 test 重选或调参。"]
    if test:
        lines+=["",f"冻结 test：**{test['official']['correct']}/{test['official']['n']}，{test['official']['accuracy']:.8f}%**。",
                f"本机完整预测加评分 {test['total_wall_seconds']:.3f} 秒，总线程峰值 {test['total_peak_threads']}，未压缩推理资产 {test['inference_asset_bytes']} 字节。参考 CPU 尚未验证。"]
    lines+=["",f"本轮外部总用时：{time.perf_counter()-tick:.1f} 秒。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
