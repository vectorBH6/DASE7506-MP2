"""仅用 train 拟合线性答案组排序器：按题交叉验证，导出纯 NumPy 权重。"""
import os

# 在导入数值库前限制线程；训练与后续复现实验都使用 CPU。
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "4"

import argparse
from itertools import product
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

from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from solution.handcrafted_ranker.features import (
    FEATURE_DIM, FEATURE_VERSION, answer_groups, best_group, build_features, standardize,
)


from solution.handcrafted_ranker.arithmetic import FEATURE_SETS, feature_version, scalar_indices


from solution.group_ranker.train_linear import (
    export_model, group_scores, make_folds, metrics, write_json,
)

from solution.handcrafted_ranker.optimization import fit_linear


def prepare_training(data_dir, feature_set="basic"):
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
        x = build_features(row["candidates"], embedding, groups,
                           question=row["question"], feature_set=feature_set)
        features.append(np.pad(x, ((0, 4 - len(groups)), (0, 0))))
        labels.append(list(groups).index(target))
        patterns.append(tuple(sorted((len(v) for v in groups.values()), reverse=True)))
    mask = np.array([[g < len(p) for g in range(4)] for p in patterns])
    return np.stack(features), mask, np.array(labels), patterns, len(bank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True, help="新实验目录，例如 solution/runs/handcrafted_full_v2")
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="basic")
    parser.add_argument("--l2", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--handcrafted-l2-multiplier", nargs="+", type=float, default=[1.0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()
    if args.folds < 2 or args.maxiter < 1 or any(not np.isfinite(v) or v <= 0 for v in args.l2 + args.handcrafted_l2_multiplier):
        parser.error("folds 至少为 2，maxiter 和 l2 必须为正")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录已含文件；请使用新的实验目录，保留旧记录")
    tick = time.perf_counter()
    features, mask, labels, patterns, total = prepare_training(args.data_dir, args.feature_set)
    if args.folds > min(Counter(patterns).values()):
        parser.error("folds 超过最小票型样本数")
    fold_ids = make_folds(patterns, args.folds, args.seed)
    records, oof_scores = [], []
    for l2, multiplier in product(args.l2, args.handcrafted_l2_multiplier):
        scores = np.full(mask.shape, -np.inf, dtype=np.float32)
        fold_records = []
        for fold in range(args.folds):
            train, valid = fold_ids != fold, fold_ids == fold
            model, status = fit_linear(features[train], mask[train], labels[train], l2, args.maxiter, multiplier)
            scores[valid] = group_scores(features[valid], mask[valid], model)
            fold_records.append({"fold": fold, **metrics(scores[valid], labels[valid]), **status})
            print(f"l2={l2:g}, multiplier={multiplier:g}, fold={fold}: {fold_records[-1]}", flush=True)
        records.append({"l2": l2, "handcrafted_l2_multiplier": multiplier, "oof": metrics(scores, labels), "folds": fold_records})
        oof_scores.append(scores)
    # 只看 train OOF；准确率、NLL 依次比较，再优先更强的正则。
    selected = min(range(len(records)), key=lambda i: (
        -records[i]["oof"]["accuracy"], records[i]["oof"]["nll"], -records[i]["l2"],
        abs(np.log(records[i]["handcrafted_l2_multiplier"])), -records[i]["handcrafted_l2_multiplier"],
    ))
    best_l2 = records[selected]["l2"]
    best_multiplier = records[selected]["handcrafted_l2_multiplier"]
    model, status = fit_linear(features, mask, labels, best_l2, args.maxiter, best_multiplier)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": PROTOCOL, "feature_version": feature_version(args.feature_set), "feature_dim": FEATURE_DIM,
        "feature_set": args.feature_set, "model": "linear",
        "enabled_scalar_indices": scalar_indices(args.feature_set), "l2": best_l2,
        "handcrafted_l2_multiplier": best_multiplier,
        "seed": args.seed, "folds": args.folds, "maxiter": args.maxiter,
        "training_split": "train", "training_questions": len(labels),
    }
    asset_bytes = export_model(args.output / "checkpoint", model, config)
    # OOF 与标签只存训练记录目录，不放入推理 checkpoint。
    np.savez(args.output / "oof.npz", scores=oof_scores[selected], labels=labels, folds=fold_ids)
    sources = [Path(__file__), Path(__file__).with_name("features.py"),
               Path(__file__).with_name("submission_linear.py"), Path(__file__).with_name("arithmetic.py"),
               Path(__file__).with_name("optimization.py"),
               Path(__file__).resolve().parent.parent / "group_ranker" / "features.py",
               Path(__file__).resolve().parent.parent / "group_ranker" / "train_linear.py"]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "train_total": total, "train_conflict": len(labels),
        "search": records, "selected_l2": best_l2, "selected_handcrafted_l2_multiplier": best_multiplier,
        "selected_search_index": selected, "final_optimizer": status,
        "inference_asset_bytes": asset_bytes, "training_seconds": time.perf_counter() - tick,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(), "threads": 4},
        "source_sha256": {str(p.resolve().relative_to(Path(__file__).resolve().parents[2])):
                          hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "data_manifest_sha256": hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest(),
    }
    write_json(args.output / "training.json", report)
    print(json.dumps({"selected_l2": best_l2, "oof": records[selected]["oof"],
                      "asset_bytes": asset_bytes, "training_seconds": report["training_seconds"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
