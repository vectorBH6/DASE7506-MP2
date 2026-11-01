"""Fixed-method supplementary test comparisons for README reporting."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[name]="1"
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,platform,sys,tarfile,time
import numpy as np
from common import load_bank,parse_answer
from solution.handcrafted_ranker.experiment import CODE,measured_command,write,digest,cpu_name

output=Path(__file__).resolve().parent
methods={
 "baseline":{"method":"solution/baselines/submission.py","checkpoint":None},
 "voting":{"method":"solution/baselines/submission_voling_only.py","checkpoint":None},
 "embedding":{"method":"solution/baselines/submission_emb_only.py","checkpoint":None},
 "expanded_linear":{"method":"solution/residual_ranker/submission.py",
                    "checkpoint":"solution/runs/residual_features_v1/expanded_linear/checkpoint"},
}
plan={"created_utc":datetime.now(timezone.utc).isoformat(),
 "purpose":"User requested baseline and linear test accuracies for the README. These are fixed, existing implementations; no training, parameter search or reselection.",
 "methods":methods,"test_history_known":True,
 "final_model_unchanged":"solution/runs/handcrafted_extended_v2/models/lexical/checkpoint",
 "final_freeze_id":"f77196da60e07a6279161d7c6fed6455d30657eeb757d0c2a2d2aba1239a771a",
 "policy":"Report every listed method. Do not use these scores to change the selected final model.",
 "environment":{"python":platform.python_version(),"numpy":np.__version__,"cpu":cpu_name(),
                "child_thread_limit":3,"coordinator_thread_limit":1}}
write(output/"plan.json",plan)
paths=[CODE/name for name in ("common.py","predict.py","score_predictions.py")]
paths+=list((CODE/"solution").glob("*.py"))
for package in ("baselines","group_ranker","handcrafted_ranker","residual_ranker"):
    paths+=list((CODE/"solution"/package).glob("*.py"))
for checkpoint in (methods["expanded_linear"]["checkpoint"],plan["final_model_unchanged"]):
    paths+=list((CODE/checkpoint).iterdir())
paths += [output/"plan.json",Path(__file__).resolve()]
hashes={str(p.relative_to(CODE)):digest(p) for p in sorted(set(paths))}
write(output/"freeze.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"sha256":hashes})
with tarfile.open(output/"source.tar","w") as archive:
    for name in hashes:archive.add(CODE/name,arcname=name)
records={}
for name,method in methods.items():
    directory=output/name
    directory.mkdir()
    command=[sys.executable,"predict.py","--method",method["method"],"--split","test",
             "--output",str(directory/"predictions.json")]
    if method["checkpoint"]:command+=["--checkpoint",method["checkpoint"]]
    commands=[command,[sys.executable,"score_predictions.py","--split","test",
         "--predictions",str(directory/"predictions.json"),"--output",str(directory/"score.json")]]
    tick=time.perf_counter()
    processes=[measured_command(command) for command in commands]
    result={**method,"official":json.loads((directory/"score.json").read_text()),
       "total_wall_seconds":time.perf_counter()-tick,"processes":processes,
       "total_peak_threads":max(p["total_peak_threads"] for p in processes),
       "inference_asset_bytes":sum(p.stat().st_size for p in (CODE/method["checkpoint"]).iterdir()) if method["checkpoint"] else 0,
       "reference_cpu_verified":False}
    write(directory/"summary.json",result)
    records[name]=result
    print(f"{name}: {result['official']['correct']}/{result['official']['n']} ({result['official']['accuracy']:.8f}%)",flush=True)
bank=load_bank("test")
answer_choices={}
for name in ("baseline","voting"):
    selected=json.loads((output/name/"predictions.json").read_text())["selected"]
    answer_choices[name]=[parse_answer(row["candidates"][j]["text"]) for row,j in zip(bank,selected)]
assert answer_choices["baseline"]==answer_choices["voting"]
assert all(digest(CODE/name)==value for name,value in hashes.items())
old=json.loads((CODE/"solution/runs/handcrafted_extended_v2/freeze.json").read_text())
assert all(digest(CODE/name)==value for name,value in old["sha256"].items())
write(output/"summary.json",{"methods":records,"baseline_voting_test_answers_identical":True,
   "all_frozen_files_unchanged":True,"selected_final_model_unchanged":True,
   "source_archive_sha256":digest(output/"source.tar")})
