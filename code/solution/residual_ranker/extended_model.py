"""NumPy-only linear inference for the separately versioned extended features."""
import hashlib
import json
from pathlib import Path
import numpy as np
from common import PROTOCOL
from solution.group_ranker.features import standardize
from solution.residual_ranker.extended_features import SCHEMAS, feature_dim, feature_names

VERSION = "answer-group-extended-linear-v1"
ARRAY_NAMES = ("weight", "mean", "scale", "active")


def scores(features, model):
    x = standardize(features, model["mean"], model["scale"], model["active"]).astype(np.float64)
    return x @ model["weight"]


def load(checkpoint):
    if checkpoint is None:
        raise ValueError("Checkpoint required")
    directory = Path(checkpoint)
    manifest = json.loads((directory/"asset_manifest.json").read_text())
    for name in ("model.npz", "config.json"):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError("Checkpoint hash mismatch")
    config = json.loads((directory/"config.json").read_text())
    schema = config.get("feature_schema")
    if config.get("protocol") != PROTOCOL or config.get("feature_version") != VERSION or schema not in SCHEMAS:
        raise ValueError("Unsupported extended checkpoint")
    d = feature_dim(schema)
    if config.get("feature_dim") != d or config.get("extra_feature_names") != feature_names(schema):
        raise ValueError("Extended feature layout mismatch")
    with np.load(directory/"model.npz", allow_pickle=False) as archive:
        model = {name:archive[name].copy() for name in ARRAY_NAMES}
    if any(v.shape != (d,) or not np.isfinite(v).all() for v in model.values()):
        raise ValueError("Invalid extended model arrays")
    if model["active"].dtype != np.bool_ or np.any(model["scale"] <= 0):
        raise ValueError("Invalid extended standardizer")
    return model, config
