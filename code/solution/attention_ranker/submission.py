"""注意力/门控及其 L0 融合的统一预测入口；仅加载冻结资产。"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import PROTOCOL
from solution.group_ranker.features import (EMBED_DIM, FEATURE_DIM, FEATURE_VERSIONS, answer_groups,
    best_group, build_features, feature_mask, standardize)
from solution.group_ranker.submission_ensemble import average_scores
from solution.group_ranker.submission_linear import load_model as load_linear
from solution.attention_ranker.model import WIDTH, HEADS, FF_DIM, MODES, forward, shapes

MODEL, ENSEMBLE = "group-context-v1", "group-context-ensemble-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, files):
    manifest = json.loads((directory / "asset_manifest.json").read_text(encoding="utf-8"))
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(files) | {"asset_manifest.json"} or set(manifest["sha256"]) != set(files):
        raise ValueError("推理资产不完整或含未登记文件")
    if any(digest(directory / name) != manifest["sha256"][name] for name in files):
        raise ValueError("推理资产哈希不符")
    if sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) > 16 * 2**20:
        raise ValueError("推理资产超过 16 MiB")


def load_ranker(directory):
    verify(directory, ("config.json", "model.npz"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (config.get("model") != MODEL or config.get("protocol") != PROTOCOL
            or config.get("feature_version") != FEATURE_VERSIONS["text"]
            or config.get("feature_set") != "text" or config.get("feature_dim") != FEATURE_DIM
            or config.get("mixing") not in MODES or config.get("width") != WIDTH
            or config.get("heads") != HEADS or config.get("ff_dim") != FF_DIM
            or config.get("positional_encoding") is not False):
        raise ValueError("注意力模型结构或特征版本不符")
    expected = {**shapes(FEATURE_DIM), "mean": (FEATURE_DIM,), "scale": (FEATURE_DIM,), "active": (FEATURE_DIM,)}
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("模型数组清单不符")
        model = {name: archive[name].copy() for name in archive.files}
    if any(model[name].shape != shape or not np.isfinite(model[name]).all() for name, shape in expected.items()):
        raise ValueError("模型形状或有限数值检查失败")
    if (np.any(model["scale"] <= 0) or model["active"].dtype != np.bool_
            or np.any(model["active"] & ~feature_mask("text"))):
        raise ValueError("标准化参数或特征掩码不符")
    model["mixing"] = config["mixing"]
    return model


def load_model(checkpoint):
    if checkpoint is None:
        raise ValueError("请指定 checkpoint")
    directory = Path(checkpoint)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model") == MODEL:
        return load_ranker(directory), None
    files = ("config.json", *(f"{part}/{name}" for part in ("linear", "ranker")
                              for name in ("model.npz", "config.json", "asset_manifest.json")))
    verify(directory, files)
    if (config.get("model") != ENSEMBLE or config.get("protocol") != PROTOCOL
            or config.get("weights") != [0.5, 0.5] or config.get("temperature") != 1.0):
        raise ValueError("只支持固定 L0 等权概率融合")
    return load_ranker(directory / "ranker"), load_linear(directory / "linear")


def predict_profiled(bank, arrays, checkpoint, *, include_scores=False):
    profile = {"feature_seconds": 0.0, "forward_seconds": 0.0}
    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape != (len(bank), 4, EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("输入须为有限的 [N,4,1024] embedding")
    if include_scores:
        profile["group_scores"] = [[] for _ in bank]
    if not bank:
        return [], profile
    model, linear = load_model(checkpoint)
    choices = [0] * len(bank)
    for start in range(0, len(bank), 128):
        pending, features = [], []
        for i in range(start, min(start + 128, len(bank))):
            candidates = bank[i]["candidates"]
            if len(candidates) != 4:
                raise ValueError("每题须有四个候选")
            groups = answer_groups(candidates)
            if len(groups) == 1:
                choices[i] = int(min(next(iter(groups.values()))))
                if include_scores:
                    profile["group_scores"][i] = [0.0]
            elif groups:
                tick = time.perf_counter()
                features.append(np.pad(build_features(candidates, embeddings[i], groups,
                    question=bank[i]["question"], feature_set="text"), ((0, 4-len(groups)), (0, 0))))
                pending.append((i, groups))
                profile["feature_seconds"] += time.perf_counter() - tick
        if not pending:
            continue
        tick = time.perf_counter()
        features = np.stack(features)
        mask = np.array([[g < len(groups) for g in range(4)] for _, groups in pending])
        x = standardize(features, model["mean"], model["scale"], model["active"]) * mask[..., None]
        scores, _ = forward(x, mask, model, model["mixing"])
        if linear is not None:
            basic = features.copy()
            basic[..., 2 * EMBED_DIM + 16:] = 0
            left = standardize(basic, linear["mean"], linear["scale"], linear["active"]) @ linear["weight"]
            scores = average_scores(np.where(mask, left, -np.inf), scores)
        profile["forward_seconds"] += time.perf_counter() - tick
        for row, (i, groups) in zip(scores, pending):
            choices[i] = int(min(list(groups.values())[int(best_group(row))]))
            if include_scores:
                profile["group_scores"][i] = row[:len(groups)].tolist()
    return choices, profile


def predict(bank, arrays, checkpoint):
    return predict_profiled(bank, arrays, checkpoint)[0]


def create_bundle(linear, ranker, output):
    import shutil
    from solution.group_ranker.training import write_json
    if output.exists():
        raise ValueError("已有 checkpoint 不覆盖")
    load_linear(linear)
    load_ranker(ranker)
    output.mkdir(parents=True)
    for part, source in (("linear", linear), ("ranker", ranker)):
        (output / part).mkdir()
        for name in ("model.npz", "config.json", "asset_manifest.json"):
            shutil.copyfile(source / name, output / part / name)
    write_json(output / "config.json", {"protocol": PROTOCOL, "model": ENSEMBLE,
                                       "weights": [0.5, 0.5], "temperature": 1.0})
    files = [p for p in output.rglob("*") if p.is_file()]
    write_json(output / "asset_manifest.json", {"sha256": {str(p.relative_to(output)): digest(p) for p in files}})
    load_model(output)
    return sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
