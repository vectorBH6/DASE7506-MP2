"""Bounded, prespecified 24-signal expansion and train/validation ablation."""
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
from solution.handcrafted_ranker.experiment import CODE,CHILD_ENV,accuracy,interval,write,digest
from solution.residual_ranker.experiment import evaluate,freeze
from solution.residual_ranker.extended_features import (
    SCHEMAS,CANDIDATE_NAMES,INTERACTION_INDICES,answer_groups,feature_dim,feature_names,
)

INCUMBENT=CODE/"solution/runs/residual_features_v1/expanded_residual_r01"
OLD_METHOD="solution/residual_ranker/submission.py"
METHOD="solution/residual_ranker/extended_submission.py"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--test",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():parser.error("New output directory required")
    output.mkdir(parents=True)
    start=time.perf_counter()
    training=json.loads((INCUMBENT/"training.json").read_text())
    baseline=training["search"][training["selected_search_index"]]["oof"]
    plan={"created_utc":datetime.now(timezone.utc).isoformat(),"seed":7506,"folds":5,
        "candidate_features":list(CANDIDATE_NAMES),
        "schemas":{s:{"dim":feature_dim(s),"extra_feature_names":feature_names(s)} for s in SCHEMAS},
        "interaction_indices":list(INTERACTION_INDICES),"l2":[.01,.03,.1],"maxiter":400,
        "within_schema_selection":"Converged folds only; train OOF correct descending, NLL ascending, stronger L2.",
        "eligibility":"Selected five folds and final optimization converged; train OOF correct >= incumbent 1745/2114. Incumbent always eligible.",
        "selection":"Eligible models only: validation correct descending, train OOF correct descending, OOF NLL ascending, fewer parameters, fixed order.",
        "known_test_history":"Current frozen incumbent 388/467 (83.08351%); earlier public test scores known. No test errors inspected or used for features.",
        "test_policy":"Freeze the selected method before any new test; evaluate one new candidate only if selected over incumbent. No post-test tuning.",
        "model":"Linear group softmax; no additional nonlinear search. NumPy-only inference."}
    write(output/"plan.json",plan)
    command=[sys.executable,"-m","unittest","solution.group_ranker.test_group_ranker",
        "solution.handcrafted_ranker.test_handcrafted","solution.handcrafted_ranker.test_optimization",
        "solution.residual_ranker.test_residual","solution.residual_ranker.test_extended","-v"]
    tests=subprocess.run(command,cwd=CODE,env=CHILD_ENV,capture_output=True,text=True)
    write(output/"unit_tests.json",{"command":command,"returncode":tests.returncode,
                                  "stdout":tests.stdout,"stderr":tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stderr)
    print("31 tests passed; starting 4 schemas x 3 L2 train-only CV.",flush=True)
    records={"incumbent":{"train_oof":baseline,"eligible":True,"all_optimizers_converged":True,
        "parameter_count":2088+42*8,"method":OLD_METHOD,
        "checkpoint":str((INCUMBENT/"checkpoint").relative_to(CODE)),
        **evaluate(INCUMBENT/"checkpoint","validation",output/"incumbent/validation",OLD_METHOD)}}
    archived=json.loads((INCUMBENT/"validation/predictions.json").read_text())["selected"]
    assert json.loads((output/"incumbent/validation/predictions.json").read_text())["selected"]==archived
    command=[sys.executable,"-m","solution.residual_ranker.extended_train","--output",str(output/"models")]
    with (output/"train.log").open("w") as log:
        subprocess.run(command,cwd=CODE,env=CHILD_ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    old_oof=np.load(INCUMBENT/"oof.npz",allow_pickle=False)
    for schema in SCHEMAS:
        directory=output/"models"/schema
        report=json.loads((directory/"training.json").read_text())
        selected=report["search"][report["selected_search_index"]]
        oof=np.load(directory/"oof.npz",allow_pickle=False)
        np.testing.assert_array_equal(oof["folds"],old_oof["folds"])
        np.testing.assert_array_equal(oof["labels"],old_oof["labels"])
        converged=all(f["converged"] for f in selected["folds"]) and report["final_optimizer"]["converged"]
        records[schema]={"train_oof":selected["oof"],"all_optimizers_converged":converged,
            "eligible":bool(converged and selected["oof"]["correct"]>=baseline["correct"]),
            "l2":selected["l2"],"parameter_count":feature_dim(schema),"method":METHOD,
            "checkpoint":str((directory/"checkpoint").relative_to(CODE)),
            **evaluate(directory/"checkpoint","validation",output/schema/"validation",METHOD)}
        print(f"{schema}: OOF={selected['oof']['correct']}/2114, val={records[schema]['official']['correct']}/798, eligible={records[schema]['eligible']}",flush=True)
    bank=load_bank("validation")
    gold=[parse_answer(r["answer"]) for r in read_jsonl(DATA/"gsm8k_validation.jsonl")]
    counts=np.array([len(answer_groups(r["candidates"])) for r in bank])
    correctness={}
    for name,r in records.items():
        prediction=json.loads((output/name/"validation/predictions.json").read_text())
        validate_choices(bank,prediction["ids"],prediction["selected"])
        correct=np.array([parse_answer(row["candidates"][j]["text"])==answer
                          for row,j,answer in zip(bank,prediction["selected"],gold)])
        assert int(correct.sum())==r["official"]["correct"]
        correctness[name]=correct
        r["conflict"]=accuracy(correct[counts>=2])
    base=correctness["incumbent"]
    for name,correct in correctness.items():
        records[name]["corrected_vs_incumbent"]=int(np.sum(~base & correct))
        records[name]["regressed_vs_incumbent"]=int(np.sum(base & ~correct))
        records[name]["difference_95ci_pp"]=interval(correct.astype(int)-base.astype(int))
    order=tuple(records)
    selected=min((n for n in order if records[n]["eligible"]),key=lambda n:(
        -records[n]["official"]["correct"],-records[n]["train_oof"]["correct"],
        records[n]["train_oof"]["nll"],records[n]["parameter_count"],order.index(n)))
    chosen=records[selected]
    write(output/"validation_summary.json",{"methods":records})
    write(output/"selection.json",{"selected":selected,"method":chosen["method"],
        "checkpoint":chosen["checkpoint"],"eligibility":plan["eligibility"],"policy":plan["selection"]})
    print(f"Selected before test: {selected}",flush=True)
    frozen=freeze(output,CODE/chosen["checkpoint"],chosen["method"])
    test=None
    if args.test and selected!="incumbent":
        test=evaluate(CODE/chosen["checkpoint"],"test",output/"test",chosen["method"])
        test.update(freeze_id=frozen["freeze_id"],
            frozen_files_unchanged=all(digest(CODE/n)==v for n,v in frozen["sha256"].items()),
            reference_cpu_verified=False)
        assert test["frozen_files_unchanged"]
        write(output/"test/summary.json",test)
        print(f"Frozen test: {test['official']}",flush=True)
    else:
        write(output/"test_decision.json",{"evaluated":False,"reason":"Incumbent retained; no redundant test." if selected=="incumbent" else "Test not requested."})
    lines=["# 第二轮手工特征扩展","","新增24个逐候选信号，均值/最大值48列；完整版本另含24列组内标准化与8列票数交互。原2088列逐位保留。","",
        "范围、资格和选型规则在训练前写入plan.json；按题五折，标准化仅在训练fold拟合，L2只看训练OOF。","",
        "| 特征块 | 维度 | OOF | Validation | 纠正/改错 | 可选资格 |",
        "|---|---:|---:|---:|---:|---|"]
    for name,r in records.items():
        lines.append(f"| {name} | {r['parameter_count']} | {r['train_oof']['correct']}/2114 ({r['train_oof']['accuracy']:.2f}%) | {r['official']['correct']}/798 ({r['official']['accuracy']:.2f}%) | {r['corrected_vs_incumbent']}/{r['regressed_vs_incumbent']} | {r['eligible']} |")
    lines+=["",f"冻结前选定：**{selected}**。入口：{chosen['method']}。Checkpoint：{chosen['checkpoint']}。",
        "词法块关注题目目标与正文/结尾的呼应、目标单位、答案量级及新数字；算术块关注有效步骤、最终答案一致性、错误步骤到达答案、结果复用与悬空结果。数字复用仅作软证据，不是符号证明。",
        "历史公开test成绩已知，本轮未读取test错题设计特征，未依据本轮test调参。Validation反复使用会增加选择偏差，应完整报告试验历史。"]
    if test:
        lines+=["",f"官方冻结test：**{test['official']['correct']}/{test['official']['n']}，{test['official']['accuracy']:.8f}%**。",
            f"本机完整预测加评分 {test['total_wall_seconds']:.3f} 秒；资产 {test['inference_asset_bytes']} 字节；参考CPU未验证。"]
    else:lines+=["","本轮保留原模型，未重新评测test。当前历史冻结成绩仍为388/467，83.08351%。"]
    lines+=["",f"外部总用时：{time.perf_counter()-start:.1f}秒。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
