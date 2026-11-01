"""官方 predict 接口：加载冻结线性模型，以 NumPy 完成离线选择。"""
import hashlib
import json
from pathlib import Path

import numpy as np

from solution.handcrafted_ranker.features import (
    EMBED_DIM, FEATURE_DIM, FEATURE_VERSION, answer_groups, best_group,
    build_features, standardize,
)


def load_model(checkpoint):
    if checkpoint is None:
        raise ValueError("请通过 --checkpoint 指定训练导出的 checkpoint 目录")
    directory = Path(checkpoint)
    manifest = json.loads((directory / "asset_manifest.json").read_text(encoding="utf-8"))
    for name in ("config.json", "model.npz"):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != manifest["sha256"][name]:
            raise ValueError(f"推理资产哈希不符：{name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    from solution.handcrafted_ranker.arithmetic import feature_version
    feature_set = config.get("feature_set", "basic")
    if config.get("feature_version") != feature_version(feature_set) or config.get("feature_dim") != FEATURE_DIM:
        raise ValueError("checkpoint 特征版本与当前代码不匹配")
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        model = {name: archive[name].copy() for name in ("weight", "mean", "scale", "active")}
    if any(value.shape != (FEATURE_DIM,) for value in model.values()):
        raise ValueError("checkpoint 数组形状错误")
    if any(not np.isfinite(value).all() for value in model.values()):
        raise ValueError("checkpoint 包含非有限数值")
    if np.any(model["scale"] <= 0) or model["active"].dtype != np.bool_:
        raise ValueError("checkpoint 标准化参数错误")
    model["feature_set"] = feature_set
    return model


def predict(bank, arrays, checkpoint):
    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape != (len(bank), 4, EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("bank 与 embedding 的数量、形状或数值不符合协议")
    if not bank:
        return []
    model = load_model(checkpoint)
    choices = [0] * len(bank)

    # 以问题分块；无需模型的题直接返回，其余答案组一起做矩阵乘法。
    for start in range(0, len(bank), 128):
        pending, features = [], []
        for i in range(start, min(start + 128, len(bank))):
            candidates = bank[i]["candidates"]
            if len(candidates) != 4:
                raise ValueError("每题必须恰有四个候选")
            groups = answer_groups(candidates)
            if len(groups) == 1:
                choices[i] = min(next(iter(groups.values())))
            elif len(groups) >= 2:
                pending.append((i, groups))
                features.append(build_features(candidates, embeddings[i], groups,
                                               question=bank[i].get("question"), feature_set=model["feature_set"]))
            # 全 None 的题无候选能得分，保留合法的第 0 个索引。
        if pending:
            x = standardize(np.concatenate(features), model["mean"], model["scale"], model["active"])
            scores = x @ model["weight"]
            offset = 0
            for i, groups in pending:
                winner = int(best_group(scores[offset:offset + len(groups)]))
                choices[i] = int(min(list(groups.values())[winner]))
                offset += len(groups)
    return choices
