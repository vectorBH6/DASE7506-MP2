"""Predefined expanded-feature follow-up, with OOF and convergence eligibility."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[name]="1"
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from common import DATA,load_bank,parse_answer,read_jsonl,validate_choices
from solution.group_ranker.features import answer_groups
from solution.handcrafted_ranker.experiment import CODE,CHILD_ENV,accuracy,interval,write,digest
from solution.residual_ranker.experiment import OLD,OLD_METHOD,NEW_METHOD,evaluate,train,freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--test",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():parser.error("Use a new output directory")
    output.mkdir(parents=True)
    start=time.perf_counter()
    original=json.loads((OLD/"training.json").read_text())
    incumbent_oof=next(r["oof"] for r in original["search"] if r["l2"]==original["selected_l2"])
    plan={"created_utc":datetime.now(timezone.utc).isoformat(),"seed":7506,"folds":5,
          "reason":"Follow-up on prewritten expanded features: weak residual configurations had substantially lower train OOF and hit the iteration cap. Original residual_v1 plan, decision and test remain unchanged.",
          "known_test_history":"Original math=82.22698%; residual_v1=80.51392%. These scores do not determine feature definitions, penalties or current selection.",
          "features":"Same 16 expanded features already implemented before residual_v1 training; no test-error-specific rules.",
          "trials":[{"name":"expanded_linear","hidden":0,"linear_l2":[.003,.01,.03]},
                    {"name":"expanded_residual_r003","hidden":8,"residual_l2":.003,
                     "linear_l2":"OOF selected expanded linear L2"},
                    {"name":"expanded_residual_r01","hidden":8,"residual_l2":.01,
                     "linear_l2":"OOF selected expanded linear L2"}],
          "eligibility":"All fold optimizers and final optimizer converged, and train OOF correct >= incumbent 1736/2114. Incumbent always eligible.",
          "selection":"Among eligible models: validation correct descending; train OOF accuracy descending; train OOF NLL ascending; fewer parameters; fixed order.",
          "test_policy":"Only one selected model is frozen/scored after validation. No tuning after this test."}
    write(output/"plan.json",plan)
    command=[sys.executable,"-m","unittest","solution.group_ranker.test_group_ranker",
             "solution.handcrafted_ranker.test_handcrafted","solution.handcrafted_ranker.test_optimization",
             "solution.residual_ranker.test_residual","-v"]
    tests=subprocess.run(command,cwd=CODE,env=CHILD_ENV,capture_output=True,text=True)
    write(output/"unit_tests.json",{"command":command,"returncode":tests.returncode,
                                  "stdout":tests.stdout,"stderr":tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stderr)
    records={"incumbent":{"train_oof":incumbent_oof,"hidden":0,"parameter_count":2088,
                         "eligible":True,"all_optimizers_converged":True,
                         "method":OLD_METHOD,"checkpoint":str((OLD/"checkpoint").relative_to(CODE)),
                         **evaluate(OLD/"checkpoint","validation",output/"incumbent/validation",OLD_METHOD)}}
    print("27 tests passed; expanded-feature follow-up started.",flush=True)

    def trial(name,hidden,residuals,linears):
        info=train(output/name,"expanded",hidden,residuals,linears,output/f"{name}-train.log")
        info["eligible"]=bool(info["all_optimizers_converged"] and info["train_oof"]["correct"]>=incumbent_oof["correct"])
        records[name]={**info,"method":NEW_METHOD,"checkpoint":str((output/name/"checkpoint").relative_to(CODE)),
                       **evaluate(output/name/"checkpoint","validation",output/name/"validation",NEW_METHOD)}
        print(f"{name}: OOF={info['train_oof']['accuracy']:.4f}%, validation={records[name]['official']['correct']}, eligible={info['eligible']}",flush=True)
    trial("expanded_linear",0,[.01],[.003,.01,.03])
    linear_l2=records["expanded_linear"]["linear_l2"]
    trial("expanded_residual_r003",8,[.003],[linear_l2])
    trial("expanded_residual_r01",8,[.01],[linear_l2])

    bank=load_bank("validation")
    gold=[parse_answer(row["answer"]) for row in read_jsonl(DATA/"gsm8k_validation.jsonl")]
    counts=np.array([len(answer_groups(row["candidates"])) for row in bank])
    correctness={}
    for name,record in records.items():
        prediction=json.loads((output/name/"validation/predictions.json").read_text())
        validate_choices(bank,prediction["ids"],prediction["selected"])
        correct=np.array([parse_answer(row["candidates"][j]["text"])==answer
                          for row,j,answer in zip(bank,prediction["selected"],gold)])
        assert int(correct.sum())==record["official"]["correct"]
        correctness[name]=correct
        record["conflict"]=accuracy(correct[counts>=2])
        record["by_group_count"]={str(g):accuracy(correct[counts==g]) for g in sorted(set(counts))}
    base=correctness["incumbent"]
    for name,correct in correctness.items():
        records[name]["corrected_vs_incumbent"]=int(np.sum(~base & correct))
        records[name]["regressed_vs_incumbent"]=int(np.sum(base & ~correct))
        records[name]["difference_95ci_pp"]=interval(correct.astype(int)-base.astype(int))
    order=tuple(records)
    selected=min((name for name in order if records[name]["eligible"]),key=lambda name:(
        -records[name]["official"]["correct"],-records[name]["train_oof"]["accuracy"],
        records[name]["train_oof"]["nll"],records[name]["parameter_count"],order.index(name)))
    chosen=records[selected]
    write(output/"validation_summary.json",{"methods":records})
    write(output/"selection.json",{"selected":selected,"checkpoint":chosen["checkpoint"],"method":chosen["method"],
                                  "eligibility":plan["eligibility"],"policy":plan["selection"],"validation":chosen["official"]})
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
    lines=["# 手工特征扩展后续实验","","本轮为独立后续实验，不改写 residual_v1 的预设规则与测试结果。16 个扩展指标在第一轮残差网络训练前已实现，未针对 test 错题设计。","",
           "本轮收紧资格：五折及最终优化均收敛，且 OOF 正确题数不低于原模型；随后按 validation 选型。资格和搜索范围在本轮训练前写入 plan.json。","",
           "| 模型 | OOF 冲突题 | Validation | 冲突题 | 纠正/改错 | 可选资格 |",
           "|---|---:|---:|---:|---:|---|"]
    for name,record in records.items():
        lines.append(f"| {name} | {record['train_oof']['accuracy']:.2f}% | {record['official']['correct']}/{record['official']['n']} ({record['official']['accuracy']:.2f}%) | {record['conflict']['accuracy']:.2f}% | {record['corrected_vs_incumbent']}/{record['regressed_vs_incumbent']} | {record['eligible']} |")
    lines+=["",f"冻结前选定：**{selected}**。入口：{chosen['method']}。Checkpoint：{chosen['checkpoint']}。",
            "历史 test 分数已知，本轮不构成完全未见测试；本轮未据新 test 调参或重选。"]
    if test:lines+=["",f"官方 test：**{test['official']['correct']}/{test['official']['n']}，{test['official']['accuracy']:.8f}%**。",
                    f"本机完整预测加评分 {test['total_wall_seconds']:.3f} 秒；资产 {test['inference_asset_bytes']} 字节；参考 CPU 未验证。"]
    lines+=["",f"本轮外部耗时 {time.perf_counter()-start:.1f} 秒。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
