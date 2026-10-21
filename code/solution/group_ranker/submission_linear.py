"""官方 predict 接口：加载冻结线性模型，以 NumPy 完成离线选择。"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from solution.group_ranker.features import (
    EMBED_DIM, FEATURE_DIM, FEATURE_VERSIONS, SCALAR_COUNTS, answer_groups, best_group,
    build_features, feature_mask, standardize,
)


def load_model(checkpoint, *, feature_set="basic"):
    if feature_set not in ("basic", "text"):
        raise ValueError("线性入口只支持基础或文本特征")
    if checkpoint is None:
        raise ValueError("请通过 --checkpoint 指定训练导出的 checkpoint 目录")
    directory = Path(checkpoint)
    manifest = json.loads((directory / "asset_manifest.json").read_text(encoding="utf-8"))
    for name in ("config.json", "model.npz"):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != manifest["sha256"][name]:
            raise ValueError(f"推理资产哈希不符：{name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("feature_version") != FEATURE_VERSIONS[feature_set] or config.get("feature_dim") != FEATURE_DIM:
        raise ValueError("checkpoint 特征版本与当前代码不匹配")
    if feature_set == "text" and (config.get("model") != "linear" or config.get("feature_set") != "text"
            or config.get("enabled_scalar_indices") != list(range(SCALAR_COUNTS["text"]))):
        raise ValueError("文本线性模型的配置不完整")
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        model = {name: archive[name].copy() for name in ("weight", "mean", "scale", "active")}
    if any(value.shape != (FEATURE_DIM,) for value in model.values()):
        raise ValueError("checkpoint 数组形状错误")
    if any(not np.isfinite(value).all() for value in model.values()):
        raise ValueError("checkpoint 包含非有限数值")
    if np.any(model["scale"] <= 0) or model["active"].dtype != np.bool_:
        raise ValueError("checkpoint 标准化参数错误")
    if feature_set == "text" and np.any(model["active"] & ~feature_mask("text")):
        raise ValueError("文本线性模型启用了预留的算术维度")
    return model


def predict_profiled(bank, arrays, checkpoint, *, feature_set="basic", include_scores=False):
    """共用预测路径；仅按显式指定的 schema 提取特征，附带无标签计时。"""
    profile = {"feature_seconds": 0.0, "forward_seconds": 0.0}
    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape != (len(bank), 4, EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("bank 与 embedding 的数量、形状或数值不符合协议")
    if not bank:
        if include_scores:
            profile["group_scores"] = []
        return [], profile
    model = load_model(checkpoint, feature_set=feature_set)
    choices = [0] * len(bank)
    if include_scores:
        profile["group_scores"] = [[] for _ in bank]

    # 以问题分块；无需模型的题直接返回，其余答案组一起做矩阵乘法。
    for start in range(0, len(bank), 128):
        pending, features = [], []
        for i in range(start, min(start + 128, len(bank))):
            candidates = bank[i]["candidates"]
            if len(candidates) != 4:
                raise ValueError("每题必须恰有四个候选")
            groups = answer_groups(candidates)
            if len(groups) == 1:
                choices[i] = int(min(next(iter(groups.values()))))
                if include_scores:
                    profile["group_scores"][i] = [0.0]
            elif len(groups) >= 2:
                pending.append((i, groups))
                tick = time.perf_counter()
                features.append(build_features(candidates, embeddings[i], groups,
                    question=bank[i].get("question", ""), feature_set=feature_set, diagnostics=profile))
                profile["feature_seconds"] += time.perf_counter() - tick
            # 全 None 的题无候选能得分，保留合法的第 0 个索引。
        if pending:
            tick = time.perf_counter()
            x = standardize(np.concatenate(features), model["mean"], model["scale"], model["active"])
            scores = x @ model["weight"]
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
