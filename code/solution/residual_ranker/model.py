"""Pure NumPy residual forward, loading and export; no SciPy import."""
import hashlib
import json
from pathlib import Path
import numpy as np

from solution.group_ranker.features import FEATURE_DIM, standardize

VERSION="answer-group-residual-v1"
ARRAY_NAMES=("weight","mean","scale","active","hidden_weight","hidden_bias","hidden_output","hidden_indices")


def scores(features,model):
    x=standardize(features,model["mean"],model["scale"],model["active"]).astype(np.float64)
    value=x @ model["weight"]
    if model["hidden_output"].size:
        z=x[:,model["hidden_indices"]]
        value=value+np.tanh(z @ model["hidden_weight"]+model["hidden_bias"]) @ model["hidden_output"]
    return value


def export(directory,model,config):
    directory.mkdir(parents=True)
    np.savez(directory/"model.npz",**model)
    (directory/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    names=("model.npz","config.json")
    manifest={"sha256":{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}}
    (directory/"asset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    size=sum(path.stat().st_size for path in directory.iterdir())
    if size>16*2**20:
        raise ValueError("Inference asset limit exceeded")
    return size


def load(checkpoint):
    if checkpoint is None:
        raise ValueError("Checkpoint required")
    directory=Path(checkpoint)
    manifest=json.loads((directory/"asset_manifest.json").read_text())
    for name in ("model.npz","config.json"):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=manifest["sha256"][name]:
            raise ValueError("Checkpoint hash mismatch")
    config=json.loads((directory/"config.json").read_text())
    if config.get("feature_version")!=VERSION or config.get("feature_schema") not in ("math","expanded"):
        raise ValueError("Unsupported residual checkpoint schema")
    with np.load(directory/"model.npz",allow_pickle=False) as archive:
        model={name:archive[name].copy() for name in ARRAY_NAMES}
    d=FEATURE_DIM
    h=config["hidden"]
    q=24 if config["feature_schema"]=="math" else 40
    expected={"weight":(d,),"mean":(d,),"scale":(d,),"active":(d,),
              "hidden_weight":(q,h),"hidden_bias":(h,),"hidden_output":(h,),"hidden_indices":(q,)}
    if any(model[name].shape!=shape for name,shape in expected.items()):
        raise ValueError("Residual model array shape mismatch")
    if any(not np.isfinite(array).all() for array in model.values()):
        raise ValueError("Nonfinite residual checkpoint")
    if np.any(model["scale"]<=0) or model["active"].dtype!=np.bool_:
        raise ValueError("Invalid standardizer")
    if not np.issubdtype(model["hidden_indices"].dtype,np.integer) or not np.array_equal(model["hidden_indices"],np.arange(2048,2048+q)):
        raise ValueError("Invalid residual input columns")
    return model,config
