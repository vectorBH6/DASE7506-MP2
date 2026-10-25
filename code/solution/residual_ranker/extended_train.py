"""Train-only five-fold linear ablations of additional handcrafted features."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[name] = "4"
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from solution.group_ranker.features import standardize
from solution.group_ranker.train_linear import fit_standardizer, objective, make_folds, metrics, export_model, write_json
from solution.residual_ranker.extended_features import (
    SCHEMAS, answer_groups, build_features, feature_dim, feature_names, columns,
)
from solution.residual_ranker.extended_model import VERSION, scores


def prepare():
    check_data(DATA, ("train",))
    bank, arrays = load_inputs("train")
    gold = read_jsonl(DATA/"gsm8k_train.jsonl")
    if len(bank) != len(gold):
        raise ValueError("Training count mismatch")
    features, labels, patterns = [], [], []
    for row, reference, embedding in zip(bank, gold, arrays["embeddings"]):
        if (row["id"],row["question"]) != (reference["id"],reference["question"]):
            raise ValueError("Training alignment mismatch")
        groups = answer_groups(row["candidates"])
        target = parse_answer(reference["answer"])
        if target not in groups:
            raise ValueError("No correct training answer group")
        if len(groups) < 2:
            continue
        x = build_features(row["candidates"], embedding, groups, question=row["question"])
        features.append(np.pad(x, ((0,4-len(groups)),(0,0))))
        labels.append(list(groups).index(target))
        patterns.append(tuple(sorted(map(len, groups.values()), reverse=True)))
    mask = np.array([[g<len(p) for g in range(4)] for p in patterns])
    return np.stack(features), mask, np.array(labels), patterns


def fit(features, mask, labels, l2, maxiter=400):
    model = fit_standardizer(features, mask)
    d = features.shape[-1]
    x = standardize(features, **model).astype(np.float64).reshape(-1,d)
    result = minimize(objective, np.zeros(d), args=(x,mask,labels,l2), method="L-BFGS-B", jac=True,
                      options={"maxiter":maxiter,"ftol":1e-9,"gtol":1e-6})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("Nonfinite extended optimization")
    model["weight"] = result.x.astype(np.float32)
    return model, {"converged":bool(result.success),"iterations":int(result.nit),
                   "objective":float(result.fun),"message":str(result.message)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", nargs="+", choices=SCHEMAS, default=list(SCHEMAS))
    parser.add_argument("--l2", nargs="+", type=float, default=[.01,.03,.1])
    args = parser.parse_args()
    if args.output.exists() or any(not np.isfinite(v) or v<=0 for v in args.l2):
        parser.error("New output directory and positive finite L2 required")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    full, mask, labels, patterns = prepare()
    folds = make_folds(patterns, 5, 7506)
    for schema in args.schemas:
        features = full[:,:,columns(schema)]
        records, oofs = [], []
        tick = time.perf_counter()
        for l2 in args.l2:
            logits = np.full(mask.shape, -np.inf)
            fold_records = []
            for fold in range(5):
                train, valid = folds!=fold, folds==fold
                model, status = fit(features[train],mask[train],labels[train],l2)
                logits[valid] = np.where(mask[valid],scores(features[valid].reshape(-1,features.shape[-1]),model).reshape(mask[valid].shape),-np.inf)
                fold_records.append({"fold":fold,**metrics(logits[valid],labels[valid]),**status})
            record = {"l2":l2,"oof":metrics(logits,labels),"folds":fold_records}
            records.append(record);oofs.append(logits)
            print(f"{schema}, L2={l2}: {record['oof']}; converged={all(f['converged'] for f in fold_records)}",flush=True)
        # Ineligible optimizers cannot win the within-schema search.
        eligible = [i for i,r in enumerate(records) if all(f["converged"] for f in r["folds"])]
        if not eligible:
            raise RuntimeError("No converged configuration")
        best = min(eligible,key=lambda i:(-records[i]["oof"]["correct"],records[i]["oof"]["nll"],-records[i]["l2"]))
        selected = records[best]
        model, status = fit(features,mask,labels,selected["l2"])
        directory = args.output/schema
        directory.mkdir()
        config = {"protocol":PROTOCOL,"feature_version":VERSION,"feature_schema":schema,
                  "feature_dim":feature_dim(schema),"extra_feature_names":feature_names(schema),
                  "l2":selected["l2"],"folds":5,"seed":7506,"training_split":"train","training_questions":len(labels)}
        size = export_model(directory/"checkpoint",model,config)
        np.savez(directory/"oof.npz",scores=oofs[best],labels=labels,folds=folds)
        paths = list(Path(__file__).parent.glob("extended_*.py"))
        write_json(directory/"training.json",{"config":config,"search":records,
            "selected_search_index":best,"final_optimizer":status,"inference_asset_bytes":size,
            "training_seconds":time.perf_counter()-tick,
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    print(f"Training finished in {time.perf_counter()-start:.1f}s",flush=True)


if __name__ == "__main__":
    main()
