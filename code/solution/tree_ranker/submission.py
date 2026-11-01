"""树排序器的官方入口：加载冻结数组，以 NumPy 给答案组打分。"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import PROTOCOL
from solution.group_ranker.features import (
    EMBED_DIM, FEATURE_DIM, FEATURE_VERSIONS, answer_groups, best_group, build_features, standardize,
)
from solution.tree_ranker.model import MODEL, forward, validate_model


def load_model(checkpoint):
    if checkpoint is None:
        raise ValueError("请指定树模型 checkpoint")
    directory = Path(checkpoint)
    files = {p.name for p in directory.iterdir()}
    if files != {"model.npz", "config.json", "asset_manifest.json"}:
        raise ValueError("树资产不完整或含未登记文件")
    manifest = json.loads((directory / "asset_manifest.json").read_text(encoding="utf-8"))
    if set(manifest["sha256"]) != {"model.npz", "config.json"}:
        raise ValueError("树资产清单不完整")
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"树资产哈希不符：{name}")
    if sum(p.stat().st_size for p in directory.iterdir()) > 16 * 2**20:
        raise ValueError("树资产超过 16 MiB")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (config.get("protocol") != PROTOCOL or config.get("model") != MODEL
            or config.get("feature_version") != FEATURE_VERSIONS["text"]
            or config.get("feature_dim") != FEATURE_DIM or config.get("feature_set") != "text"
            or config.get("enabled_scalar_indices") != list(range(32))):
        raise ValueError("树 checkpoint 的模型或特征 schema 不匹配")
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        model = {name: archive[name].copy() for name in archive.files}
    validate_model(model, config)
    return model


def predict_profiled(bank, arrays, checkpoint, *, include_scores=False):
    profile = {"feature_seconds": 0.0, "forward_seconds": 0.0}
    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape != (len(bank), 4, EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("bank 与 embedding 的数量、形状或数值不符合协议")
    if include_scores:
        profile["group_scores"] = [[] for _ in bank]
    if not bank:
        return [], profile
    model, choices = load_model(checkpoint), [0] * len(bank)
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
            elif groups:
                pending.append((i, groups))
                tick = time.perf_counter()
                features.append(build_features(candidates, embeddings[i], groups,
                    question=bank[i].get("question", ""), feature_set="text", diagnostics=profile))
                profile["feature_seconds"] += time.perf_counter() - tick
        if pending:
            tick = time.perf_counter()
            x = standardize(np.concatenate(features), model["mean"], model["scale"], model["active"])
            scores = forward(x, model)
            profile["forward_seconds"] += time.perf_counter() - tick
            offset = 0
            for i, groups in pending:
                local = scores[offset:offset + len(groups)]
                choices[i] = int(min(list(groups.values())[int(best_group(local))]))
                if include_scores:
                    profile["group_scores"][i] = local.tolist()
                offset += len(groups)
    return choices, profile


def predict(bank, arrays, checkpoint):
    return predict_profiled(bank, arrays, checkpoint)[0]
