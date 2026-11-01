"""同 M1 特征的浅树排序：预设深度和预算、按题五折、折内早停、全量 train 重训。"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import catboost
from catboost import CatBoostRanker, Pool
import numpy as np

from common import DATA, PROTOCOL
from solution.group_ranker.features import FEATURE_DIM, FEATURE_VERSIONS, best_group, standardize
from solution.group_ranker.training import (
    export_model, fit_standardizer, make_folds, metrics, prepare_training, write_json,
)
from solution.tree_ranker.model import MAX_TREES, MODEL, extract_trees, forward
from solution.tree_ranker.submission import load_model

# 默认复现深度 3/5、300 棵；预算对照显式传 --depths 5 --max-trees 1000。
DEPTHS = (3, 5)
PATIENCE = 30
PARAMETERS = {"loss_function": "QuerySoftMax", "eval_metric": "QuerySoftMax",
    "learning_rate": 0.05, "l2_leaf_reg": 3.0, "border_count": 64,
    "grow_policy": "SymmetricTree", "boosting_type": "Plain", "bootstrap_type": "No",
    "random_strength": 0.0, "leaf_estimation_iterations": 1,
    "task_type": "CPU", "thread_count": 1, "allow_writing_files": False, "verbose": False}


def make_pool(features, mask, labels):
    # CatBoost 的 group_id 表示一道题；同题各答案组连续排列，题号不进入特征。
    group_id = np.repeat(np.arange(len(labels)), mask.sum(axis=1))
    targets = (np.arange(mask.shape[1])[None, :] == labels[:, None])[mask].astype(np.float32)
    return Pool(features[mask], label=targets, group_id=group_id, thread_count=1)


def fit(features, mask, labels, *, depth, iterations, seed, validation=None):
    scaler = fit_standardizer(features, mask)
    x = standardize(features, **scaler)
    valid_pool = None
    if validation is not None:
        valid_x, valid_mask, valid_labels = validation
        valid_pool = make_pool(standardize(valid_x, **scaler), valid_mask, valid_labels)
    model = CatBoostRanker(**PARAMETERS, depth=depth, iterations=iterations, random_seed=seed)
    model.fit(make_pool(x, mask, labels), eval_set=valid_pool,
              early_stopping_rounds=PATIENCE if valid_pool is not None else None,
              use_best_model=valid_pool is not None)
    return model, scaler


def convert_model(model):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "model.json"
        model.save_model(str(path), format="json")
        return extract_trees(json.loads(path.read_text(encoding="utf-8")))


def checked_scores(model, scaler, features, mask):
    x = standardize(features, **scaler)[mask]
    arrays = convert_model(model)
    native = np.asarray(model.predict(x, thread_count=1), dtype=np.float64)
    exported = forward(x, arrays)
    error = float(np.max(np.abs(native - exported)))
    if error > 1e-10:
        raise ValueError(f"CatBoost 与 NumPy 前向不一致：{error}")
    scores = np.full(mask.shape, -np.inf, dtype=np.float64)
    scores[mask] = exported
    native_scores = scores.copy()
    native_scores[mask] = native
    if not np.array_equal(best_group(scores), best_group(native_scores)):
        raise ValueError("CatBoost 与 NumPy 的答案组选择不一致")
    return scores, arrays, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", choices=DEPTHS, default=DEPTHS)
    parser.add_argument("--max-trees", type=int, default=300, help="预设训练上限，默认复现原预算")
    args = parser.parse_args()
    if not 1 <= args.max_trees <= MAX_TREES or len(set(args.depths)) != len(args.depths):
        parser.error(f"树数须在 1–{MAX_TREES} 内，深度不可重复")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录已有记录，请使用新实验名")
    args.output.mkdir(parents=True, exist_ok=True)
    seed, folds = 7506, 5
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "state": "running",
        "command": [sys.executable, *sys.argv], "depths": list(args.depths), "max_trees": args.max_trees,
        "patience": PATIENCE, "parameters": PARAMETERS, "seed": seed, "folds": folds,
        "feature_set": "text", "fit_count": 0, "search": [],
        "scope": "仅 supplied train 训练与 OOF 选型，validation 开发比较；既有 test 已接触，本轮不读取 test 参考或运行 test"}
    write_json(args.output / "training.json", report)
    tick, diagnostics = time.perf_counter(), {}
    features, mask, labels, patterns, total = prepare_training(args.data_dir, "text", diagnostics=diagnostics)
    if min(Counter(patterns).values()) < folds:
        raise ValueError("最小票型样本数不足五折")
    fold_ids = make_folds(patterns, folds, seed)
    oof = []
    for depth in args.depths:
        scores = np.full(mask.shape, -np.inf, dtype=np.float64)
        record = {"depth": depth, "folds": []}
        report["search"].append(record)
        for fold in range(folds):
            train, valid = fold_ids != fold, fold_ids == fold
            model, scaler = fit(features[train], mask[train], labels[train], depth=depth,
                iterations=args.max_trees, seed=seed, validation=(features[valid], mask[valid], labels[valid]))
            scores[valid], _, error = checked_scores(model, scaler, features[valid], mask[valid])
            status = {"fold": fold, "trees": model.tree_count_,
                      **metrics(scores[valid], labels[valid]), "native_numpy_max_abs_error": error}
            record["folds"].append(status)
            report["fit_count"] += 1
            write_json(args.output / "training.json", report)
            print(f"depth={depth}, {status}", flush=True)
        record["oof"] = metrics(scores, labels)
        oof.append(scores)
        write_json(args.output / "training.json", report)
    selected = min(range(len(args.depths)), key=lambda i: (
        -report["search"][i]["oof"]["accuracy"], report["search"][i]["oof"]["nll"], args.depths[i]))
    chosen = report["search"][selected]
    iterations = int(np.median([fold["trees"] for fold in chosen["folds"]]))
    model, scaler = fit(features, mask, labels, depth=chosen["depth"], iterations=iterations, seed=seed)
    _, arrays, error = checked_scores(model, scaler, features, mask)
    arrays.update(scaler)
    config = {"protocol": PROTOCOL, "model": MODEL, "loss": "QuerySoftMax",
        "feature_dim": FEATURE_DIM, "feature_set": "text", "feature_version": FEATURE_VERSIONS["text"],
        "enabled_scalar_indices": list(range(32)), "depth": chosen["depth"],
        "max_depth": int(arrays["split_features"].shape[1]), "tree_count": model.tree_count_,
        "preprocessing": "fold-train standardization and clip8, identical to M1",
        "seed": seed, "folds": folds, "training_split": "train", "training_questions": len(labels)}
    asset_bytes = export_model(args.output / "checkpoint", arrays, config)
    load_model(args.output / "checkpoint")
    # 原生模型仅作训练复核，不放入推理 checkpoint，也不由 predict 读取。
    model.save_model(str(args.output / "native.cbm"))
    np.savez(args.output / "oof.npz", scores=oof[selected], labels=labels, folds=fold_ids)
    code = Path(__file__).resolve().parents[2]
    sources = {*Path(__file__).parent.glob("*.py"), code / "solution/evaluate.py",
        code / "solution/requirements-train.txt",
        *(code / "solution/group_ranker" / name for name in ("features.py", "evidence.py", "training.py"))}
    report.update(state="complete", fit_count=report["fit_count"] + 1, config=config,
        selected_depth=chosen["depth"], final_trees=model.tree_count_,
        final_native_numpy_max_abs_error=error, train_total=total, train_conflict=len(labels),
        feature_diagnostics=diagnostics, inference_asset_bytes=asset_bytes,
        training_seconds=time.perf_counter() - tick,
        environment={"python": sys.version.split()[0], "numpy": np.__version__,
            "catboost": catboost.__version__, "platform": platform.platform(), "threads": 1,
            "packages": dict(sorted((package.metadata["Name"], package.version) for package in distributions())),
            "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), platform.processor())},
        source_sha256={str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        data_manifest_sha256=hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest())
    write_json(args.output / "training.json", report)
    print(json.dumps({"selected_depth": chosen["depth"], "final_trees": model.tree_count_,
        "oof": chosen["oof"], "asset_bytes": asset_bytes, "training_seconds": report["training_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
