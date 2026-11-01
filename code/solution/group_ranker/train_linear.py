"""仅用 train 拟合线性答案组排序器：按题交叉验证，导出纯 NumPy 权重。"""
import os

# NumPy 与 SciPy 可能各载入一套 BLAS；统一单线程，避免线程池叠加。
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

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

from common import DATA, PROTOCOL
from solution.group_ranker.features import (
    FEATURE_DIM, FEATURE_VERSIONS, SCALAR_COUNTS, standardize,
)
from solution.group_ranker.training import (
    export_model, fit_standardizer, make_folds, metrics, prepare_training, write_json,
)


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


def main(*, feature_set="basic", method=None):
    if feature_set not in ("basic", "text"):
        raise ValueError("线性训练只支持基础或文本特征")
    method = Path(method) if method else Path(__file__).with_name("submission_linear.py")
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
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "state": "running",
              "feature_set": feature_set, "planned_l2": args.l2, "folds": args.folds, "seed": args.seed,
              "command": [sys.executable, *sys.argv], "search": [],
              "development_policy": "仅 train 拟合与 OOF 选参数；validation 开发比较；不读取 test 参考或用 test 选型"}
    write_json(args.output / "training.json", report)
    tick = time.perf_counter()
    diagnostics = {}
    features, mask, labels, patterns, total = prepare_training(args.data_dir, feature_set, diagnostics=diagnostics)
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
        report["search"] = records
        write_json(args.output / "training.json", report)
    # 只看 train OOF；准确率、NLL 依次比较，再优先更强的正则。
    selected = min(range(len(records)), key=lambda i: (
        -records[i]["oof"]["accuracy"], records[i]["oof"]["nll"], -records[i]["l2"],
    ))
    best_l2 = records[selected]["l2"]
    model, status = fit_linear(features, mask, labels, best_l2, args.maxiter)
    config = {
        "protocol": PROTOCOL, "feature_version": FEATURE_VERSIONS[feature_set], "feature_dim": FEATURE_DIM,
        "feature_set": feature_set, "model": "linear",
        "enabled_scalar_indices": list(range(SCALAR_COUNTS[feature_set])), "l2": best_l2,
        "seed": args.seed, "folds": args.folds, "maxiter": args.maxiter,
        "training_split": "train", "training_questions": len(labels),
    }
    asset_bytes = export_model(args.output / "checkpoint", model, config)
    # OOF 与标签只存训练记录目录，不放入推理 checkpoint。
    np.savez(args.output / "oof.npz", scores=oof_scores[selected], labels=labels, folds=fold_ids)
    sources = {Path(__file__).resolve(), Path(sys.argv[0]).resolve(), method.resolve(),
               *(Path(__file__).with_name(name) for name in
                 ("training.py", "features.py", "evidence.py", "submission_linear.py"))}
    code = Path(__file__).resolve().parents[2]
    report.update({
        "state": "complete", "config": config, "feature_diagnostics": diagnostics,
        "fit_count": len(args.l2) * args.folds + 1,
        "train_total": total, "train_conflict": len(labels),
        "search": records, "selected_l2": best_l2, "final_optimizer": status,
        "inference_asset_bytes": asset_bytes, "training_seconds": time.perf_counter() - tick,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(), "threads": 1,
                        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                     if line.startswith("model name")), platform.processor())},
        "source_sha256": {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        "data_manifest_sha256": hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest(),
    })
    write_json(args.output / "training.json", report)
    print(json.dumps({"selected_l2": best_l2, "oof": records[selected]["oof"],
                      "asset_bytes": asset_bytes, "training_seconds": report["training_seconds"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
