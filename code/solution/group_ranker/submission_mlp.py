"""M0/M1/M2 官方预测接口：按冻结配置提取特征，纯 NumPy 推理。"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from solution.group_ranker.features import (
    ABLATIONS, EMBED_DIM, FEATURE_DIM, FEATURE_VERSIONS,
    answer_groups, best_group, build_features, feature_mask, standardize,
)
from solution.group_ranker.mlp import HIDDEN_SIZES, PARAMETERS, forward


def load_model(checkpoint):
    if checkpoint is None:
        raise ValueError("请指定 MLP checkpoint")
    directory = Path(checkpoint)
    manifest = json.loads((directory / "asset_manifest.json").read_text(encoding="utf-8"))
    for name in ("config.json", "model.npz"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"推理资产哈希不符：{name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    feature_set = config.get("feature_set", "basic")  # 兼容已经冻结的 M0。
    ablation = config.get("ablation", "none")
    if (feature_set not in FEATURE_VERSIONS or config.get("model") != "mlp"
            or ablation not in ABLATIONS or config.get("loss", "ce") not in ("ce", "pairwise")
            or config.get("feature_version") != FEATURE_VERSIONS[feature_set]
            or config.get("enabled_scalar_indices") != np.flatnonzero(feature_mask(feature_set, ablation)[2048:]).tolist()
            or config.get("feature_dim") != FEATURE_DIM or config.get("hidden") not in HIDDEN_SIZES):
        raise ValueError("MLP checkpoint 结构或特征版本错误")
    h1, h2 = config["hidden"]
    shapes = {"w1": (FEATURE_DIM, h1), "b1": (h1,), "w2": (h1, h2), "b2": (h2,),
              "w3": (h2, 1), "b3": (1,), "mean": (FEATURE_DIM,),
              "scale": (FEATURE_DIM,), "active": (FEATURE_DIM,)}
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        model = {name: archive[name].copy() for name in (*PARAMETERS, "mean", "scale", "active")}
    if any(model[name].shape != shape or not np.isfinite(model[name]).all() for name, shape in shapes.items()):
        raise ValueError("MLP 数组形状或数值错误")
    if np.any(model["scale"] <= 0) or model["active"].dtype != np.bool_:
        raise ValueError("MLP 标准化参数错误")
    if np.any(model["active"] & ~feature_mask(feature_set, ablation)):
        raise ValueError("checkpoint 启用了应关闭的特征")
    model["feature_set"] = feature_set
    return model


def predict_profiled(bank, arrays, checkpoint, *, include_scores=False):
    """同一预测路径附带无标签计时/覆盖统计，供 validation 分析使用。"""
    profile = {"feature_seconds": 0.0, "forward_seconds": 0.0}
    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape != (len(bank), 4, EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("bank 与 embedding 的数量、形状或数值不符合协议")
    if not bank:
        return [], profile
    model = load_model(checkpoint)
    choices = [0] * len(bank)
    if include_scores:
        profile["group_scores"] = [[] for _ in bank]
    for start in range(0, len(bank), 128):
        pending, features = [], []
        for i in range(start, min(start + 128, len(bank))):
            candidates = bank[i]["candidates"]
            if len(candidates) != 4:
                raise ValueError("每题必须有四个候选")
            groups = answer_groups(candidates)
            if len(groups) == 1:
                choices[i] = int(min(next(iter(groups.values()))))
                if include_scores:
                    profile["group_scores"][i] = [0.0]
            elif groups:
                pending.append((i, groups))
                tick = time.perf_counter()
                features.append(build_features(candidates, embeddings[i], groups,
                                               question=bank[i]["question"], feature_set=model["feature_set"],
                                               diagnostics=profile))
                profile["feature_seconds"] += time.perf_counter() - tick
        if pending:
            tick = time.perf_counter()
            x = standardize(np.concatenate(features), model["mean"], model["scale"], model["active"])
            scores, _ = forward(x, model)
            profile["forward_seconds"] += time.perf_counter() - tick
            offset = 0
            for i, groups in pending:
                local = scores[offset:offset + len(groups)]
                winner = int(best_group(local))
                choices[i] = int(min(list(groups.values())[winner]))
                if include_scores:
                    profile["group_scores"][i] = local.tolist()
                offset += len(groups)
    return choices, profile


def predict(bank, arrays, checkpoint):
    return predict_profiled(bank, arrays, checkpoint)[0]
