"""M0 有限选型；特征/损失消融复用已选配置；稳定性复核不导出新候选。"""
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from common import DATA, PROTOCOL
from solution.group_ranker.features import (
    ABLATIONS, FEATURE_DIM, FEATURE_VERSIONS, feature_mask, standardize,
)
from solution.group_ranker.mlp import HIDDEN_SIZES, PARAMETERS, forward, initialize, loss_gradient
from solution.group_ranker.training import (
    export_model, fit_standardizer, make_folds, metrics, prepare_training, write_json,
)

SEED, FOLDS = 7506, 5
EPOCHS, PATIENCE, BATCH_SIZE = 150, 15, 64
LEARNING_RATE, DROPOUT = 1e-3, 0.1
SEARCH = [{"hidden": [width, 16], "weight_decay": decay}
          for width in (64, 32) for decay in (0.01, 0.1)]


def scores_for(x, mask, model):
    scores, _ = forward(x.reshape(-1, FEATURE_DIM), model)
    return np.where(mask, scores.reshape(mask.shape), -np.inf)


def fit_mlp(x, mask, labels, config, seed, epochs, validation=None):
    rng = np.random.default_rng(seed)
    model = initialize(FEATURE_DIM, config["hidden"], rng)
    first = {key: np.zeros_like(model[key]) for key in PARAMETERS}
    second = {key: np.zeros_like(model[key]) for key in PARAMETERS}
    best_nll, best_epoch, best_model, step = np.inf, 0, None, 0
    history = []
    tick = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order, training_loss = rng.permutation(len(labels)), 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            loss, gradient = loss_gradient(x[batch].reshape(-1, FEATURE_DIM), mask[batch],
                                           labels[batch], model, rng, DROPOUT, config.get("loss", "ce"))
            norm = np.sqrt(sum(np.sum(value.astype(np.float64) ** 2) for value in gradient.values()))
            if not np.isfinite(loss) or not np.isfinite(norm):
                raise RuntimeError("MLP 训练出现非有限损失或梯度")
            training_loss += loss * len(batch) / len(labels)
            step += 1
            for key in PARAMETERS:
                g = gradient[key] / max(1.0, norm)  # 全局梯度范数裁剪到 1。
                first[key] *= 0.9
                first[key] += 0.1 * g
                second[key] *= 0.999
                second[key] += 0.001 * g * g
                # AdamW 的衰减与梯度分开；权重和偏置统一衰减。
                model[key] *= 1 - LEARNING_RATE * config["weight_decay"]
                model[key] -= LEARNING_RATE * (first[key] / (1 - 0.9 ** step)) / (
                    np.sqrt(second[key] / (1 - 0.999 ** step)) + 1e-8)
        record = {"epoch": epoch, "train_dropout_loss": training_loss}
        if validation is not None:
            vx, vm, vy = validation
            result = metrics(scores_for(vx, vm, model), vy)
            if not np.isfinite(result["nll"]):
                raise RuntimeError("MLP 折内评估出现非有限损失")
            record.update(validation_nll=result["nll"], validation_accuracy=result["accuracy"])
            if result["nll"] < best_nll - 1e-6:
                best_nll, best_epoch = result["nll"], epoch
                best_model = {key: value.copy() for key, value in model.items()}
        history.append(record)
        if validation is not None and epoch - best_epoch >= PATIENCE:
            break
    if validation is None:
        best_model, best_epoch = model, epochs
    if any(not np.isfinite(value).all() for value in best_model.values()):
        raise RuntimeError("拒绝导出非有限权重")
    return best_model, {"best_epoch": best_epoch, "epochs_run": len(history), "steps": step,
                        "seconds": time.perf_counter() - tick, "history": history}


def cross_validate(features, mask, labels, folds, config, seed):
    scores = np.full(mask.shape, -np.inf, dtype=np.float32)
    records = []
    for fold in range(FOLDS):
        train, valid = folds != fold, folds == fold
        scaler = fit_standardizer(features[train], mask[train])
        x = standardize(features[train], **scaler)
        vx = standardize(features[valid], **scaler)
        model, status = fit_mlp(x, mask[train], labels[train], config, seed + fold, EPOCHS,
                                (vx, mask[valid], labels[valid]))
        scores[valid] = scores_for(vx, mask[valid], model)
        result = metrics(scores[valid], labels[valid])
        records.append({"fold": fold, "training_seed": seed + fold, **result, **status})
        print(json.dumps({"seed": seed, **config, "fold": fold, **result,
                          "best_epoch": status["best_epoch"], "epochs_run": status["epochs_run"]}), flush=True)
    return {"seed": seed, **config, "oof": metrics(scores, labels), "folds": records}, scores


def finish_report(report, args, tick):
    root = Path(__file__).parent
    sources = [root / name for name in ("train_mlp.py", "training.py", "mlp.py", "features.py",
                                       "evidence.py", "submission_mlp.py")]
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    report.update(state="complete", training_seconds=time.perf_counter() - tick,
                  fit_count=sum(len(r["folds"]) for r in report["search"] + report["stability"])
                            + int("final_training" in report),
                  environment={"python": sys.version.split()[0], "numpy": np.__version__,
                               "backend": "NumPy AdamW, CPU", "cpu": cpu,
                               "platform": platform.platform(), "blas_threads": 4},
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  data_manifest_sha256=hashlib.sha256((args.data_dir / "manifest.json").read_bytes()).hexdigest())
    write_json(args.output / "training.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-set", choices=FEATURE_VERSIONS, default="basic")
    parser.add_argument("--reuse-config", type=Path, help="复用 M0/M1 配置，不重新搜索超参数")
    parser.add_argument("--hidden", type=int, nargs=2, metavar=("H1", "H2"),
                        help="容量对照：复用配置，只覆盖两个隐层宽度")
    parser.add_argument("--ablation", choices=ABLATIONS, default="none")
    parser.add_argument("--loss", choices=("ce", "pairwise"), default="ce")
    parser.add_argument("--stability-only", action="store_true", help="只复核 7507/7508 OOF，不重训或导出候选")
    args = parser.parse_args()
    if (args.feature_set != "basic") != (args.reuse_config is not None):
        parser.error("P2 须提供 --reuse-config；基础 M0 按原四配置流程运行")
    if ((args.ablation != "none" or args.loss != "ce" or args.stability_only) and not args.reuse_config
            or args.ablation != "none" and args.loss != "ce"):
        parser.error("消融/复核须复用配置；每次只改变一个特征块或损失")
    if args.hidden and (args.hidden not in HIDDEN_SIZES or not args.reuse_config
                        or args.ablation != "none" or args.loss != "ce" or args.stability_only):
        parser.error("容量对照须复用配置，使用登记宽度，且不同时修改特征块/损失或复核种子")
    search = SEARCH
    if args.reuse_config:
        base = json.loads(args.reuse_config.read_text(encoding="utf-8"))
        if (base.get("model") != "mlp"
                or base.get("feature_version") not in (FEATURE_VERSIONS["basic"], FEATURE_VERSIONS[args.feature_set])
                or base.get("ablation", "none") != "none" or base.get("loss", "ce") != "ce"
                or base.get("hidden") not in HIDDEN_SIZES
                or base.get("weight_decay") not in (0.01, 0.1)
                or base.get("seed") != SEED or base.get("fold_seed") != SEED
                or base.get("learning_rate") != LEARNING_RATE or base.get("dropout") != DROPOUT
                or base.get("batch_size_questions") != BATCH_SIZE):
            parser.error("复用配置必须与既定 M0 训练协议一致")
        if args.stability_only and (args.ablation != "none" or args.loss != "ce"
                                    or base.get("feature_set", "basic") != args.feature_set):
            parser.error("稳定性复核须使用相同特征的原始 CE 配置")
        search = [{"hidden": args.hidden or base["hidden"],
                   "weight_decay": base["weight_decay"], "loss": args.loss}]
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("请使用新的实验目录，已有记录不覆盖")
    args.output.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    diagnostics = {}
    features, mask, labels, patterns, total = prepare_training(
        args.data_dir, args.feature_set, ablation=args.ablation, diagnostics=diagnostics)
    folds = make_folds(patterns, FOLDS, SEED)  # 所有配置和复核种子使用同一划分。
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "state": "running",
              "train_total": total, "train_conflict": len(labels), "planned_search": search,
              "feature_set": args.feature_set, "feature_diagnostics": diagnostics,
              "ablation": args.ablation, "loss": args.loss, "stability_only": args.stability_only,
              "hidden_override": args.hidden,
              "command": [sys.executable, *sys.argv],
              "fold_seed": SEED, "search": [], "stability": []}
    if args.reuse_config:
        report["reused_config"] = {"path": str(args.reuse_config),
                                   "sha256": hashlib.sha256(args.reuse_config.read_bytes()).hexdigest()}
    report_path = args.output / "training.json"
    write_json(report_path, report)
    if args.stability_only:
        for seed in (7507, 7508):
            record, scores = cross_validate(features, mask, labels, folds, search[0], seed)
            report["stability"].append(record)
            np.savez(args.output / f"oof-{seed}.npz", scores=scores, labels=labels, folds=folds)
            write_json(report_path, report)
        finish_report(report, args, tick)
        return
    all_scores = []
    for config in search:
        record, scores = cross_validate(features, mask, labels, folds, config, SEED)
        report["search"].append(record)
        all_scores.append(scores)
        write_json(report_path, report)
    chosen = min(range(len(search)), key=lambda i: (
        -report["search"][i]["oof"]["accuracy"], report["search"][i]["oof"]["nll"],
        search[i]["hidden"][0], -search[i]["weight_decay"],
    ))
    config = search[chosen]
    report["selected_search_index"] = chosen
    # 复核只观察初始化/训练随机性的影响，不参与选配置或挑选幸运种子。
    for seed in (() if args.reuse_config else (7507, 7508)):
        record, _ = cross_validate(features, mask, labels, folds, config, seed)
        report["stability"].append(record)
        write_json(report_path, report)
    epochs = int(np.median([fold["best_epoch"] for fold in report["search"][chosen]["folds"]]))
    scaler = fit_standardizer(features, mask)
    model, final = fit_mlp(standardize(features, **scaler), mask, labels, config, SEED, epochs)
    model.update(scaler)
    exported = {
        "protocol": PROTOCOL, "feature_version": FEATURE_VERSIONS[args.feature_set],
        "feature_set": args.feature_set, "feature_dim": FEATURE_DIM,
        "ablation": args.ablation, "enabled_input_dimensions": int(feature_mask(args.feature_set, args.ablation).sum()),
        "model": "mlp", **config,
        "enabled_scalar_indices": np.flatnonzero(feature_mask(args.feature_set, args.ablation)[2048:]).tolist(),
        "seed": SEED, "fold_seed": SEED, "folds": FOLDS, "epochs": epochs,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "dropout": DROPOUT,
        "early_stopping_metric": "group_ce_nll",
        "batch_size_questions": BATCH_SIZE, "gradient_clip": 1.0,
        "training_split": "train", "training_questions": len(labels),
    }
    asset_bytes = export_model(args.output / "checkpoint", model, exported)
    np.savez(args.output / "oof.npz", scores=all_scores[chosen], labels=labels, folds=folds)
    report.update(config=exported, final_training=final, inference_asset_bytes=asset_bytes)
    finish_report(report, args, tick)
    print(json.dumps({"selected": exported, "oof": report["search"][chosen]["oof"],
                      "asset_bytes": asset_bytes, "seconds": report["training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
