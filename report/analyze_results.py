"""Collect recorded results and descriptive validation diagnostics; never read test references."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
from pathlib import Path
import sys,json,hashlib
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/"code"
sys.path.insert(0,str(CODE))
from common import DATA,load_inputs,load_bank,read_jsonl,parse_answer,validate_choices
from solution.group_ranker.features import answer_groups,best_group
from solution.residual_ranker.extended_features import build_features,feature_names
from solution.residual_ranker.extended_model import load,scores

OUT=ROOT/"report"
RUNS=CODE/"solution/runs"
def read(path):return json.loads(path.read_text())
def selected_training(folder):
    d=read(RUNS/folder/"training.json")
    i=d.get("selected_search_index")
    if i is None:i=next(i for i,r in enumerate(d["search"]) if r["l2"]==d["selected_l2"])
    return d,d["search"][i]
def score(relative):return read(RUNS/relative)
definitions=[
 ("vote","Majority vote","linear_v1/validation/baseline-predictions.json","readme_comparison_test_v1/baseline/score.json",None),
 ("embed","Embedding only","linear_v1/validation/embedding-predictions.json","readme_comparison_test_v1/embedding/score.json",None),
 ("linear","Original linear","linear_v1/validation/linear-predictions.json","linear_v1/test/score.json","linear_v1"),
 ("math","Linear + arithmetic","handcrafted_v1/math/validation/predictions.json","handcrafted_v1/test/score.json","handcrafted_v1/math"),
 ("expanded","Expanded linear","residual_features_v1/expanded_linear/validation/predictions.json","readme_comparison_test_v1/expanded_linear/score.json","residual_features_v1/expanded_linear"),
 ("residual","Expanded residual","residual_features_v1/expanded_residual_r01/validation/predictions.json","residual_features_v1/test/score.json","residual_features_v1/expanded_residual_r01"),
 ("final","Final lexical","handcrafted_extended_v2/lexical/validation/predictions.json","handcrafted_extended_v2/test/score.json","handcrafted_extended_v2/models/lexical"),
]
bank,arrays=load_inputs("validation")
gold=[parse_answer(row["answer"]) for row in read_jsonl(DATA/"gsm8k_validation.jsonl")]
groups=[answer_groups(row["candidates"]) for row in bank]
group_counts=np.array([len(g) for g in groups])
hard=group_counts>1
models={}
for key,name,predpath,testpath,training in definitions:
    pred=read(RUNS/predpath)
    validate_choices(bank,pred["ids"],pred["selected"])
    chosen=[parse_answer(row["candidates"][j]["text"]) for row,j in zip(bank,pred["selected"])]
    correct=np.array([a==b for a,b in zip(chosen,gold)])
    models[key]={"name":name,"selected":pred["selected"],"answers":chosen,
       "correct":correct.tolist(),"val_correct":int(correct.sum()),"val_n":len(bank),
       "val_accuracy":100*float(correct.mean()),"conflict_correct":int(correct[hard].sum()),
       "conflict_n":int(hard.sum()),"conflict_accuracy":100*float(correct[hard].mean()),
       "test":score(testpath),"oof":selected_training(training)[1]["oof"] if training else None}
final=np.array(models["final"]["correct"],bool)
rng=np.random.default_rng(7506)
paired={}
for key in ("vote","linear","math","residual"):
    before=np.array(models[key]["correct"],bool)
    delta=final.astype(int)-before.astype(int)
    boot=rng.choice(delta,size=(2000,len(delta))).mean(axis=1)*100
    # Reproduce the archived per-comparison bootstrap seed.
    r=np.random.default_rng(7506)
    boot=r.choice(delta,size=(2000,len(delta))).mean(axis=1)*100
    paired[key]={"corrected":int(np.sum(final & ~before)),"regressed":int(np.sum(~final & before)),
      "net":int(delta.sum()),"difference_pp":100*float(delta.mean()),
      "bootstrap_95ci_pp":np.quantile(boot,[.025,.975]).tolist()}
checkpoint=RUNS/"handcrafted_extended_v2/models/lexical/checkpoint"
model,config=load(checkpoint)
confidence=np.ones(len(bank),float)
group_probabilities=[]
for i,row in enumerate(bank):
    if len(groups[i])>1:
        x=build_features(row["candidates"],arrays["embeddings"][i],groups[i],
                         question=row["question"],schema="lexical")
        logits=scores(x,model)
        probability=np.exp(logits-logits.max());probability/=probability.sum()
        winner=int(best_group(logits))
        assert list(groups[i])[winner]==models["final"]["answers"][i]
        confidence[i]=probability[winner]
        group_probabilities.append(probability.tolist())
    else:group_probabilities.append([1.0])
bins=[]
for low,high in zip([0,.4,.6,.8,.9],[.4,.6,.8,.9,1.000001]):
    mask=hard & (confidence>=low) & (confidence<high)
    if mask.any():
        bins.append({"low":low,"high":min(high,1.0),"n":int(mask.sum()),
          "confidence":float(confidence[mask].mean()),"accuracy":float(final[mask].mean())})
ece=sum(r["n"]*abs(r["accuracy"]-r["confidence"]) for r in bins)/int(hard.sum())
strata=[]
for g in (1,2,3,4):
    mask=group_counts==g
    if not mask.any():continue
    strata.append({"groups":g,"n":int(mask.sum()),"final_correct":int(final[mask].sum()),
       "final_accuracy":100*float(final[mask].mean()),
       "previous_correct":sum(models["residual"]["correct"][j] for j in np.flatnonzero(mask)),
       "linear_correct":sum(models["linear"]["correct"][j] for j in np.flatnonzero(mask))})
case_lists={"corrected":[],"regressed":[],"remaining_wrong":[],"fraction_wrong":[]}
before=np.array(models["residual"]["correct"],bool)
for i,row in enumerate(bank):
    categories=[]
    if final[i] and not before[i]:categories.append("corrected")
    if before[i] and not final[i]:categories.append("regressed")
    if not final[i]:categories.append("remaining_wrong")
    if not final[i] and Fraction(gold[i]).denominator!=1:categories.append("fraction_wrong")
    if not categories:continue
    case={"row":i,"id":row["id"],"question":row["question"],"gold":gold[i],
          "final_answer":models["final"]["answers"][i],
          "previous_answer":models["residual"]["answers"][i],
          "baseline_answer":models["vote"]["answers"][i],
          "confidence":float(confidence[i]),
          "candidates":[c["text"] for c in row["candidates"]],
          "final_index":models["final"]["selected"][i],
          "previous_index":models["residual"]["selected"][i],
          "baseline_index":models["vote"]["selected"][i]}
    for name in categories:case_lists[name].append(case)
for name in case_lists:
    case_lists[name].sort(key=lambda c:(len(c["question"]),sum(len(t) for t in c["candidates"])))
case_lists={name:values[:8] for name,values in case_lists.items()}
full=read(RUNS/"handcrafted_extended_v2/validation_summary.json")["methods"]
ablations={}
grid=[]
for schema in ("lexical","trace","combined","contextual"):
    d,selected=selected_training("handcrafted_extended_v2/models/"+schema)
    ablations[schema]={"oof":selected["oof"],"l2":selected["l2"],
        "dim":d["config"]["feature_dim"],"validation":full[schema]["official"],
        "conflict":full[schema]["conflict"],"corrected":full[schema]["corrected_vs_incumbent"],
        "regressed":full[schema]["regressed_vs_incumbent"],
        "converged":all(f["converged"] for f in selected["folds"]) and d["final_optimizer"]["converged"]}
    grid.append({"schema":schema,"values":[{"l2":r["l2"],**r["oof"]} for r in d["search"]]})
coefficients=sorted([{"name":name,"weight":float(model["weight"][2088+j])}
             for j,name in enumerate(feature_names("lexical"))],key=lambda r:-abs(r["weight"]))
split_counts={}
for split in ("train","validation","test"):
    rows=load_bank(split)
    counts=[len(answer_groups(r["candidates"])) for r in rows]
    split_counts[split]={"n":len(rows),"single":counts.count(1),"conflict":sum(n>1 for n in counts),
                         "by_groups":{str(n):counts.count(n) for n in sorted(set(counts))}}
hashes={}
freeze=read(RUNS/"handcrafted_extended_v2/freeze.json")
assert all(hashlib.sha256((CODE/name).read_bytes()).hexdigest()==value for name,value in freeze["sha256"].items())
for name in ("model.npz","config.json","asset_manifest.json"):
    hashes[name]=hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
result={"models":models,"splits":split_counts,"paired_validation":paired,
  "strata":strata,"calibration":{"scope":"355 validation conflict questions; five descriptive bins",
     "bins":bins,"ece":ece,"high_confidence_errors_ge_09":int(np.sum(hard & ~final & (confidence>=.9)))},
  "cases":case_lists,"ablations":ablations,"l2_grid":grid,"largest_new_coefficients":coefficients[:10],
  "selected_config":config,"active_features":int(model["active"].sum()),"checkpoint_hashes":hashes,
  "freeze_id":freeze["freeze_id"],"source_archive_sha256":freeze["source_archive_sha256"],
  "report_analysis_scope":"Descriptive analysis after freeze, using validation references only. Test references were not opened. No model or hyperparameter was changed.",
  "reference_profiles":{split:{"n":len(values),"nonnegative_integer":sum(v>=0 and v.denominator==1 for v in values),"negative":sum(v<0 for v in values),"fraction":sum(v.denominator!=1 for v in values)} for split in ("train","validation") for values in [[Fraction(parse_answer(r["answer"])) for r in read_jsonl(DATA/f"gsm8k_{split}.jsonl")]]},
  "source_paths":{"final_run":"code/solution/runs/handcrafted_extended_v2",
       "supplementary_test":"code/solution/runs/readme_comparison_test_v1"}}
(OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"splits":split_counts,"paired":paired,"strata":strata,
                 "calibration":result["calibration"],"active":result["active_features"]},indent=2))
for name,cases in case_lists.items():
    print("\nCASES",name)
    for c in cases[:3]:print(json.dumps({k:c[k] for k in ("row","question","gold","final_answer","previous_answer","confidence")}))
