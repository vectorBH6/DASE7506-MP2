"""仅用 train 拟合线性答案组排序器：按题交叉验证，导出纯 NumPy 权重。"""
import os

# 在导入数值库前限制线程；训练与后续复现实验都使用 CPU。
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "4"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.optimize import minimize

from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from solution.group_ranker.features import (
    FEATURE_DIM, FEATURE_VERSION, answer_groups, best_group, build_features, standardize,
)


def prepare_training(data_dir):
    check_data(data_dir, ("train",))
    bank, arrays = load_inputs("train", data_dir)
    gold = read_jsonl(data_dir / "gsm8k_train.jsonl")
    if len(gold) != len(bank):
        raise ValueError("训练参考数量不匹配")
    features, labels, patterns = [], [], []
    for row, reference, embedding in zip(bank, gold, arrays["embeddings"]):
        if (row["id"], row["question"]) != (reference["id"], reference["question"]):
            raise ValueError("训练输入与参考未对齐")
        groups = answer_groups(row["candidates"])
        target = parse_answer(reference["answer"])
        if target not in groups:
            raise ValueError("题库保证被违反：没有最终答案正确的有效候选")
        if len(groups) < 2:
            continue  # 单答案题无需学习，组间交叉熵梯度为零。
        x = build_features(row["candidates"], embedding, groups)
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
    return {
        "mean": mean.astype(np.float32),
        "scale": np.where(active, scale, 1).astype(np.float32),
        "active": active,
    }


def objective(weight, x, mask, labels, l2):
    """平均组交叉熵及解析梯度；L2 约定为 l2/2 * ||w||²。"""
    n = len(labels)
    scores = (x @ weight).reshape(mask.shape)
    scores = np.where(mask, scores, -np.inf)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denominator = exp_scores.sum(axis=1, keepdims=True)
    nll = np.mean(np.log(denominator[:, 0]) - shifted[np.arange(n), labels])
    residual = exp_scores / denominator
    residual[np.arange(n), labels] -= 1
    gradient = x.T @ residual.ravel() / n + l2 * weight
    return nll + l2 * (weight @ weight) / 2, gradient


def fit_linear(features, mask, labels, l2, maxiter):
    model = fit_standardizer(features, mask)
    x = standardize(features, **model).astype(np.float64).reshape(-1, FEATURE_DIM)
    result = minimize(
        objective, np.zeros(FEATURE_DIM), args=(x, mask, labels, l2),
        method="L-BFGS-B", jac=True,
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("优化结果非有限，停止导出")
    model["weight"] = result.x.astype(np.float32)
    status = {"converged": bool(result.success), "iterations": int(result.nit),
              "objective": float(result.fun), "message": str(result.message)}
    return model, status


def group_scores(features, mask, model):
    x = standardize(features, model["mean"], model["scale"], model["active"])
    return np.where(mask, x @ model["weight"], -np.inf)


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
    np.savez(directory / "model.npz", **model)  # 不压缩，资源大小可直接检查。
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True, help="新实验目录，例如 solution/runs/linear_v2")
    parser.add_argument("--l2", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()
    if args.folds < 2 or args.maxiter < 1 or any(v <= 0 for v in args.l2):
        parser.error("folds 至少为 2，maxiter 和 l2 必须为正")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录已含文件；请使用新的实验目录，保留旧记录")
    tick = time.perf_counter()
    features, mask, labels, patterns, total = prepare_training(args.data_dir)
    if args.folds > min(Counter(patterns).values()):
        parser.error("folds 超过最小票型样本数")
    fold_ids = make_folds(patterns, args.folds, args.seed)
    records, oof_scores = [], []
    for l2 in args.l2:
        scores = np.full(mask.shape, -np.inf, dtype=np.float32)
        fold_records = []
        for fold in range(args.folds):
            train, valid = fold_ids != fold, fold_ids == fold
            model, status = fit_linear(features[train], mask[train], labels[train], l2, args.maxiter)
            scores[valid] = group_scores(features[valid], mask[valid], model)
            fold_records.append({"fold": fold, **metrics(scores[valid], labels[valid]), **status})
            print(f"l2={l2:g}, fold={fold}: {fold_records[-1]}", flush=True)
        records.append({"l2": l2, "oof": metrics(scores, labels), "folds": fold_records})
        oof_scores.append(scores)
    # 只看 train OOF；准确率、NLL 依次比较，再优先更强的正则。
    selected = min(range(len(records)), key=lambda i: (
        -records[i]["oof"]["accuracy"], records[i]["oof"]["nll"], -records[i]["l2"],
    ))
    best_l2 = records[selected]["l2"]
    model, status = fit_linear(features, mask, labels, best_l2, args.maxiter)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": PROTOCOL, "feature_version": FEATURE_VERSION, "feature_dim": FEATURE_DIM,
        "model": "linear", "enabled_scalar_indices": list(range(16)), "l2": best_l2,
        "seed": args.seed, "folds": args.folds, "maxiter": args.maxiter,
        "training_split": "train", "training_questions": len(labels),
    }
    asset_bytes = export_model(args.output / "checkpoint", model, config)
    # OOF 与标签只存训练记录目录，不放入推理 checkpoint。
    np.savez(args.output / "oof.npz", scores=oof_scores[selected], labels=labels, folds=fold_ids)
    sources = [Path(__file__), Path(__file__).with_name("features.py"),
               Path(__file__).with_name("submission_linear.py")]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "train_total": total, "train_conflict": len(labels),
        "search": records, "selected_l2": best_l2, "final_optimizer": status,
        "inference_asset_bytes": asset_bytes, "training_seconds": time.perf_counter() - tick,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(), "threads": 4},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "data_manifest_sha256": hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest(),
    }
    write_json(args.output / "training.json", report)
    print(json.dumps({"selected_l2": best_l2, "oof": records[selected]["oof"],
                      "asset_bytes": asset_bytes, "training_seconds": report["training_seconds"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
