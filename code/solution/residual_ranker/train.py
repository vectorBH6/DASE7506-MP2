"""Train group CE with a linear skip plus tanh on scalar group evidence."""
import os
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[variable]="4"

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy
from scipy.optimize import minimize

from common import DATA,PROTOCOL,check_data,load_inputs,parse_answer,read_jsonl
from solution.group_ranker.train_linear import fit_linear,make_folds,metrics,write_json
from solution.group_ranker.features import FEATURE_DIM,standardize
from solution.residual_ranker.features import answer_groups,build_features
from solution.residual_ranker.model import VERSION,export,scores


def unpack(theta,d,q,h):
    offset=d
    w=theta[:d]
    u=theta[offset:offset+q*h].reshape(q,h);offset+=q*h
    b=theta[offset:offset+h];offset+=h
    v=theta[offset:offset+h]
    return w,u,b,v


def objective(theta,x,z,mask,labels,hidden,linear_l2,residual_l2):
    d,q=x.shape[1],z.shape[1]
    w,u,b,v=unpack(theta,d,q,hidden)
    activation=np.tanh(z @ u+b)
    logits=(x @ w+activation @ v).reshape(mask.shape)
    logits=np.where(mask,logits,-np.inf)
    shifted=logits-logits.max(axis=1,keepdims=True)
    exponent=np.exp(shifted)
    denominator=exponent.sum(axis=1,keepdims=True)
    n=len(labels)
    loss=np.mean(np.log(denominator[:,0])-shifted[np.arange(n),labels])
    residual=exponent/denominator
    residual[np.arange(n),labels]-=1
    residual=residual.ravel()/n
    delta=residual[:,None]*v[None,:]*(1-activation*activation)
    grad_w=x.T @ residual+linear_l2*w
    grad_u=z.T @ delta+residual_l2*u
    grad_b=delta.sum(axis=0)+residual_l2*b
    grad_v=activation.T @ residual+residual_l2*v
    loss+=linear_l2*(w @ w)/2+residual_l2*(theta[d:] @ theta[d:])/2
    return loss,np.concatenate((grad_w,grad_u.ravel(),grad_b,grad_v))


def prepare(schema):
    check_data(DATA,("train",))
    bank,arrays=load_inputs("train")
    gold=read_jsonl(DATA/"gsm8k_train.jsonl")
    features,labels,patterns=[],[],[]
    for row,reference,embedding in zip(bank,gold,arrays["embeddings"]):
        if (row["id"],row["question"])!=(reference["id"],reference["question"]):
            raise ValueError("Training alignment mismatch")
        groups=answer_groups(row["candidates"])
        target=parse_answer(reference["answer"])
        if target not in groups:
            raise ValueError("No correct training answer group")
        if len(groups)<2:
            continue
        x=build_features(row["candidates"],embedding,groups,question=row["question"],schema=schema)
        features.append(np.pad(x,((0,4-len(groups)),(0,0))))
        labels.append(list(groups).index(target))
        patterns.append(tuple(sorted(map(len,groups.values()),reverse=True)))
    mask=np.array([[g<len(p) for g in range(4)] for p in patterns])
    return np.stack(features),mask,np.array(labels),patterns,len(bank)


def fit(features,mask,labels,hidden,residual_l2,linear_l2,schema,seed,maxiter):
    model,warm_status=fit_linear(features,mask,labels,linear_l2,300)
    indices=np.arange(2048,2072 if schema=="math" else 2088,dtype=np.int32)
    q=len(indices)
    if hidden==0:
        model.update(hidden_weight=np.empty((q,0),np.float32),hidden_bias=np.empty(0,np.float32),
                     hidden_output=np.empty(0,np.float32),hidden_indices=indices)
        return model,warm_status
    x=standardize(features,model["mean"],model["scale"],model["active"]).astype(np.float64).reshape(-1,FEATURE_DIM)
    z=x[:,indices]
    rng=np.random.default_rng(seed)
    u=rng.normal(0,.35,size=(q,hidden))
    theta=np.concatenate((model["weight"].astype(np.float64),u.ravel(),np.zeros(2*hidden)))
    result=minimize(objective,theta,args=(x,z,mask,labels,hidden,linear_l2,residual_l2),
                    method="L-BFGS-B",jac=True,options={"maxiter":maxiter,"gtol":1e-5,"ftol":1e-8})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("Nonfinite residual optimization")
    w,u,b,v=unpack(result.x,FEATURE_DIM,q,hidden)
    model.update(weight=w.astype(np.float32),hidden_weight=u.astype(np.float32),
                 hidden_bias=b.astype(np.float32),hidden_output=v.astype(np.float32),hidden_indices=indices)
    return model,{"converged":bool(result.success),"iterations":int(result.nit),
                  "objective":float(result.fun),"message":str(result.message),"warm_optimizer":warm_status}


def group_scores(features,mask,model):
    logits=scores(features.reshape(-1,FEATURE_DIM),model).reshape(mask.shape)
    return np.where(mask,logits,-np.inf)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--schema",choices=("math","expanded"),default="math")
    parser.add_argument("--hidden",type=int,default=8)
    parser.add_argument("--residual-l2",type=float,nargs="+",default=[.001,.01])
    parser.add_argument("--linear-l2",type=float,nargs="+",default=[.01])
    parser.add_argument("--maxiter",type=int,default=300)
    args=parser.parse_args()
    if args.output.exists() or args.hidden<0 or args.maxiter<1 or any(not np.isfinite(v) or v<=0 for v in args.residual_l2+args.linear_l2):
        parser.error("New output directory, nonnegative hidden size and positive finite penalties required")
    start=time.perf_counter()
    features,mask,labels,patterns,total=prepare(args.schema)
    folds=make_folds(patterns,5,7506)
    records,oof=[],[]
    for linear_l2 in args.linear_l2:
        for residual_l2 in (args.residual_l2 if args.hidden else [args.residual_l2[0]]):
            logits=np.full(mask.shape,-np.inf)
            fold_records=[]
            for fold in range(5):
                train,valid=folds!=fold,folds==fold
                model,status=fit(features[train],mask[train],labels[train],args.hidden,residual_l2,
                                 linear_l2,args.schema,7506+fold,args.maxiter)
                logits[valid]=group_scores(features[valid],mask[valid],model)
                result={"fold":fold,**metrics(logits[valid],labels[valid]),**status}
                fold_records.append(result)
                print(f"h={args.hidden}, res_l2={residual_l2}, linear_l2={linear_l2}, fold={fold}: {result}",flush=True)
            records.append({"hidden":args.hidden,"residual_l2":residual_l2,"linear_l2":linear_l2,
                            "oof":metrics(logits,labels),"folds":fold_records})
            oof.append(logits)
    best=min(range(len(records)),key=lambda i:(-records[i]["oof"]["accuracy"],records[i]["oof"]["nll"],
                                               -records[i]["linear_l2"],-records[i]["residual_l2"]))
    selected=records[best]
    model,status=fit(features,mask,labels,args.hidden,selected["residual_l2"],selected["linear_l2"],
                     args.schema,7506,args.maxiter)
    args.output.mkdir(parents=True)
    config={"protocol":PROTOCOL,"feature_version":VERSION,"feature_dim":FEATURE_DIM,
            "feature_schema":args.schema,"hidden":args.hidden,"linear_l2":selected["linear_l2"],
            "residual_l2":selected["residual_l2"],"seed":7506,"folds":5,"maxiter":args.maxiter,
            "training_split":"train","training_questions":len(labels)}
    size=export(args.output/"checkpoint",model,config)
    np.savez(args.output/"oof.npz",scores=oof[best],labels=labels,folds=folds)
    paths=list(Path(__file__).parent.glob("*.py"))
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"config":config,
            "train_total":total,"train_conflict":len(labels),"search":records,"selected_search_index":best,
            "final_optimizer":status,"training_seconds":time.perf_counter()-start,"inference_asset_bytes":size,
            "environment":{"python":sys.version.split()[0],"numpy":np.__version__,"scipy":scipy.__version__},
            "source_sha256":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    write_json(args.output/"training.json",report)
    print(json.dumps({"selected":selected["oof"],"asset_bytes":size,"training_seconds":report["training_seconds"]}),flush=True)


if __name__=="__main__":
    main()
