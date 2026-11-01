"""线性与 MLP 共用的训练数据、按题划分、统计和资产导出。"""
import hashlib
import json
import time

import numpy as np

from common import check_data, load_inputs, parse_answer, read_jsonl
from solution.group_ranker.features import FEATURE_DIM, answer_groups, best_group, build_features, feature_mask


def prepare_training(data_dir, feature_set="basic", *, ablation="none", diagnostics=None):
    check_data(data_dir, ("train",))
    bank, arrays = load_inputs("train", data_dir)
    gold = read_jsonl(data_dir / "gsm8k_train.jsonl")
    if len(gold) != len(bank):
        raise ValueError("训练参考数量不匹配")
    features, labels, patterns = [], [], []
    enabled = feature_mask(feature_set, ablation)
    for row, reference, embedding in zip(bank, gold, arrays["embeddings"]):
        if (row["id"], row["question"]) != (reference["id"], reference["question"]):
            raise ValueError("训练输入与参考未对齐")
        groups = answer_groups(row["candidates"])
        target = parse_answer(reference["answer"])
        if target not in groups:
            raise ValueError("题库保证被违反：没有最终答案正确的有效候选")
        if len(groups) < 2:
            continue  # 单答案题无需学习，组间交叉熵梯度为零。
        tick = time.perf_counter()
        x = build_features(row["candidates"], embedding, groups, question=row["question"],
                           feature_set=feature_set, diagnostics=diagnostics)
        x *= enabled  # 在每折标准化之前消融；关闭的维度不参与学习。
        if diagnostics is not None:
            diagnostics["feature_seconds"] = diagnostics.get("feature_seconds", 0) + time.perf_counter() - tick
        features.append(np.pad(x, ((0, 4 - len(groups)), (0, 0))))
        labels.append(list(groups).index(target))
        patterns.append(tuple(sorted((len(v) for v in groups.values()), reverse=True)))
    mask = np.array([[g < len(p) for g in range(4)] for p in patterns])
    return np.stack(features), mask, np.array(labels), patterns, len(bank)


def make_folds(patterns, folds, seed):
    """同题的组永远在一起，按有效答案票型分层，不依赖候选顺序。"""
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(patterns), dtype=int)
    for pattern in sorted(set(patterns)):
        indices = np.array([i for i, p in enumerate(patterns) if p == pattern])
        rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % folds
    return assignment


def fit_standardizer(features, mask):
    # 每题总权重相同：每个有效组获得 1/G 权重，padding 不参与统计。
    values = features[mask].astype(np.float64)
    weights = np.broadcast_to(1 / mask.sum(axis=1, keepdims=True), mask.shape)[mask]
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    active = scale >= 1e-6
    return {"mean": mean.astype(np.float32),
            "scale": np.where(active, scale, 1).astype(np.float32), "active": active}


def metrics(scores, labels):
    correct = int(np.sum(best_group(scores) == labels))
    shifted = scores.astype(np.float64) - scores.max(axis=1, keepdims=True)
    nll = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels])
    return {"correct": correct, "n": len(labels), "accuracy": 100 * correct / len(labels),
            "nll": float(nll)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_model(directory, model, config):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "model.npz", **model)
    write_json(directory / "config.json", config)
    assets = [directory / name for name in ("model.npz", "config.json")]
    write_json(directory / "asset_manifest.json", {
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets},
        "array_bytes": sum(array.nbytes for array in model.values()),
    })
    total = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    if total > 16 * 2**20:
        raise ValueError("checkpoint 超过 16 MiB")
    return total
