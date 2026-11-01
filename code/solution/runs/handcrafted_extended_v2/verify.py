import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[name]="3"
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from common import load_inputs,parse_answer
from solution.residual_ranker.extended_submission import predict
from solution.residual_ranker.extended_features import answer_groups,build_features,feature_names,BASE_DIM
from solution.residual_ranker.extended_model import load

code=Path.cwd()
output=code/"solution/runs/handcrafted_extended_v2"
selection=json.loads((output/"selection.json").read_text())
checkpoint=code/selection["checkpoint"]
model,config=load(checkpoint)
report={"inference_imports_scipy":any(n=="scipy" or n.startswith("scipy.") for n in sys.modules)}
assert not report["inference_imports_scipy"]
perm=[3,1,0,2]
report["splits"]={}
for split in ("validation","test"):
    bank,arrays=load_inputs(split)
    snapshot=arrays["embeddings"].copy()
    expected=json.loads((output/("lexical/validation" if split=="validation" else "test")/"predictions.json").read_text())["selected"]
    actual=predict(bank,arrays,checkpoint)
    assert actual==expected
    changed=[{**row,"candidates":[row["candidates"][j] for j in perm]} for row in bank]
    e=np.asarray(arrays["embeddings"])[:,perm,:]
    other=predict(changed,{"embeddings":e},checkpoint)
    before=[parse_answer(row["candidates"][j]["text"]) for row,j in zip(bank,actual)]
    after=[parse_answer(row["candidates"][j]["text"]) for row,j in zip(changed,other)]
    assert before==after
    checked=0
    for i,row in enumerate(bank):
        groups=answer_groups(row["candidates"])
        if len(groups)>1:
            a=build_features(row["candidates"],arrays["embeddings"][i],groups,
                question=row["question"],schema=config["feature_schema"])
            b=build_features(changed[i]["candidates"],e[i],answer_groups(changed[i]["candidates"]),
                question=row["question"],schema=config["feature_schema"])
            np.testing.assert_array_equal(a,b)
            checked+=1
    limit=min(131,len(bank))
    assert predict(bank[:limit],{"embeddings":arrays["embeddings"][:limit]},checkpoint)==actual[:limit]
    for i in range(min(4,len(bank))):
        assert predict(bank[i:i+1],{"embeddings":arrays["embeddings"][i:i+1]},checkpoint)==actual[i:i+1]
    np.testing.assert_array_equal(snapshot,arrays["embeddings"])
    report["splits"][split]={"questions":len(bank),"replay_identical":True,
        "permuted_selected_answers_identical":True,"conflict_feature_matrices_bitwise_identical":checked,
        "batching_identical":True,"inputs_unmodified":True}
freeze=json.loads((output/"freeze.json").read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(digest(code/name)==value for name,value in freeze["sha256"].items())
assert digest(output/"source.tar")==freeze["source_archive_sha256"]
report["freeze"]={"files_verified":len(freeze["sha256"]),"source_archive_verified":True,"freeze_id":freeze["freeze_id"]}
report["historical_freezes"]={}
for name in ("residual_features_v1","residual_v1"):
    frozen=json.loads((code/"solution/runs"/name/"freeze.json").read_text())
    assert all(digest(code/file)==value for file,value in frozen["sha256"].items())
    report["historical_freezes"][name]=True
names=feature_names(config["feature_schema"])
order=np.argsort(-np.abs(model["weight"][BASE_DIM:]))[:8]
report["largest_new_standardized_coefficients"]=[
    {"name":names[int(j)],"weight":float(model["weight"][BASE_DIM+int(j)])} for j in order]
report["inference_still_without_scipy"]=not any(n=="scipy" or n.startswith("scipy.") for n in sys.modules)
assert report["inference_still_without_scipy"]
(output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
