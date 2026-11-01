"""固定两种结构：train 五折早停，再全量重训；不读取 validation/test。"""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "4"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from common import DATA, PROTOCOL
from solution.group_ranker.features import FEATURE_DIM, FEATURE_VERSIONS, standardize
from solution.group_ranker.training import (prepare_training, make_folds, fit_standardizer,
    metrics, export_model, write_json)
from solution.group_ranker.submission_ensemble import average_scores
from solution.attention_ranker.model import WIDTH, HEADS, FF_DIM, MODES, initialize, forward, loss_gradient
from solution.attention_ranker.submission import MODEL, create_bundle, digest

CODE = Path(__file__).resolve().parents[2]
SETTINGS = dict(folds=5, fold_seed=7506, seed=7506, max_epochs=150, patience=15,
                learning_rate=0.001, weight_decay=0.01, batch_questions=64, gradient_clip=1.0)
ARCHITECTURE = dict(input_dim=FEATURE_DIM, width=WIDTH, heads=HEADS, ff_dim=FF_DIM,
                    blocks=1, positional_encoding=False, feature_set="text", dropout=0.1)


def fit(x, mask, labels, mode, seed, epochs, validation=None):
    rng = np.random.default_rng(seed)
    model = initialize(FEATURE_DIM, rng)
    first, second = ({key: np.zeros_like(value) for key, value in model.items()} for _ in range(2))
    best_nll, best_epoch, best_model, step = np.inf, 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        order = rng.permutation(len(labels))
        for start in range(0, len(order), SETTINGS["batch_questions"]):
            batch = order[start:start + SETTINGS["batch_questions"]]
            value, gradient = loss_gradient(x[batch], mask[batch], labels[batch], model, mode, rng)
            norm = np.sqrt(sum(np.sum(g.astype(np.float64) ** 2) for g in gradient.values()))
            if not np.isfinite(value) or not np.isfinite(norm):
                raise RuntimeError("拒绝非有限损失或梯度")
            total_loss += value * len(batch) / len(labels)
            step += 1
            for key in model:
                g = gradient[key] / max(1.0, norm)
                first[key] = 0.9 * first[key] + 0.1 * g
                second[key] = 0.999 * second[key] + 0.001 * g * g
                model[key] *= 1 - SETTINGS["learning_rate"] * SETTINGS["weight_decay"]
                model[key] -= SETTINGS["learning_rate"] * (first[key] / (1 - 0.9 ** step)) / (
                    np.sqrt(second[key] / (1 - 0.999 ** step)) + 1e-8)
        row = {"epoch": epoch, "train_dropout_loss": total_loss}
        if validation is not None:
            vx, vm, vy = validation
            result = metrics(forward(vx, vm, model, mode)[0], vy)
            row.update(validation_nll=result["nll"], validation_accuracy=result["accuracy"])
            if result["nll"] < best_nll - 1e-6:
                best_nll, best_epoch = result["nll"], epoch
                best_model = {key: value.copy() for key, value in model.items()}
        history.append(row)
        if validation is not None and epoch - best_epoch >= SETTINGS["patience"]:
            break
    return (model if validation is None else best_model), {
        "best_epoch": epochs if validation is None else best_epoch, "epochs_run": len(history), "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-checkpoint", type=Path, default=CODE / "solution/runs/linear_v1/checkpoint")
    args = parser.parse_args()
    output = args.output.resolve()
    if any((output / name).exists() for name in (*MODES, *(mode + "_ensemble" for mode in MODES))):
        parser.error("已有实验不覆盖，请使用新的输出目录")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["architecture"] != ARCHITECTURE or plan["training"] != SETTINGS:
            parser.error("实验计划与固定结构/训练设置不一致")
    else:
        write_json(plan_path, {"created_utc": datetime.now(timezone.utc).isoformat(),
            "architecture": ARCHITECTURE, "training": SETTINGS, "training_split": "train",
            "variants": [*MODES, *(mode + "_ensemble" for mode in MODES)],
            "selection_rule": "完整 validation 准确率，同分按启用维数、资产、耗时、名称；test 不选型",
            "test_policy": "事后扩展；test 前固定四个版本，test 不调参或重选"})
    sources = [*Path(__file__).parent.glob("*.py"), CODE / "common.py", *[
        CODE / "solution/group_ranker" / name for name in ("features.py", "evidence.py", "training.py", "submission_linear.py", "submission_ensemble.py")]]
    tick = time.perf_counter()
    diagnostics = {}
    features, mask, labels, patterns, total = prepare_training(DATA, "text", diagnostics=diagnostics)
    folds = make_folds(patterns, SETTINGS["folds"], SETTINGS["fold_seed"])
    with np.load(args.linear_checkpoint.parent / "oof.npz", allow_pickle=False) as archive:
        assert np.array_equal(labels, archive["labels"]) and np.array_equal(folds, archive["folds"])
        linear_oof = archive["scores"].copy()
    training = {}
    for mode in MODES:
        directory = output / mode
        directory.mkdir()
        scores = np.full(mask.shape, -np.inf, dtype=np.float32)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "state": "running",
            "mixing": mode, "settings": SETTINGS, "architecture": ARCHITECTURE,
            "train_total": total, "train_conflict": len(labels), "folds": [], "feature_diagnostics": diagnostics,
            "command": [sys.executable, *sys.argv]}
        started = time.perf_counter()
        for fold in range(SETTINGS["folds"]):
            train, valid = folds != fold, folds == fold
            scaler = fit_standardizer(features[train], mask[train])
            x, vx = (standardize(features[part], **scaler) * mask[part, :, None] for part in (train, valid))
            model, status = fit(x, mask[train], labels[train], mode, SETTINGS["seed"] + fold,
                SETTINGS["max_epochs"], (vx, mask[valid], labels[valid]))
            scores[valid] = forward(vx, mask[valid], model, mode)[0]
            result = {"fold": fold, **metrics(scores[valid], labels[valid]), **status}
            report["folds"].append(result)
            write_json(directory / "training.json", report)
            print(json.dumps({"mode": mode, "fold": fold, "best_epoch": status["best_epoch"],
                              **metrics(scores[valid], labels[valid])}), flush=True)
        epochs = int(np.median([fold["best_epoch"] for fold in report["folds"]]))
        scaler = fit_standardizer(features, mask)
        model, final = fit(standardize(features, **scaler) * mask[..., None], mask, labels,
                           mode, SETTINGS["seed"], epochs)
        parameter_count = sum(value.size for value in model.values())
        model.update(scaler)
        config = {"protocol": PROTOCOL, "model": MODEL, "mixing": mode,
            "feature_version": FEATURE_VERSIONS["text"], "feature_dim": FEATURE_DIM,
            "feature_set": "text", "width": WIDTH, "heads": HEADS, "ff_dim": FF_DIM,
            "positional_encoding": False, "dropout": 0.1, "epochs": epochs,
            "training_split": "train", "training_questions": len(labels), **SETTINGS}
        assets = export_model(directory / "checkpoint", model, config)
        np.savez(directory / "oof.npz", scores=scores, labels=labels, folds=folds)
        report.update(state="complete", final_training=final, config=config, fit_count=6,
            oof=metrics(scores, labels), parameter_count=parameter_count, inference_asset_bytes=assets,
            training_seconds=time.perf_counter()-started,
            source_sha256={str(p.relative_to(CODE)): digest(p) for p in sources},
            data_manifest_sha256=digest(DATA / "manifest.json"),
            environment={"python": platform.python_version(), "numpy": np.__version__,
                "backend": "NumPy, CPU", "blas_threads": 4})
        write_json(directory / "training.json", report)
        bundle = output / (mode + "_ensemble")
        assets = create_bundle(args.linear_checkpoint, directory / "checkpoint", bundle / "checkpoint")
        combined = average_scores(linear_oof, scores)
        np.savez(bundle / "oof.npz", scores=combined, labels=labels, folds=folds)
        write_json(bundle / "preparation.json", {"fit_count": 0, "weights": [0.5, 0.5],
            "temperature": 1.0, "inference_asset_bytes": assets, "oof": metrics(combined, labels),
            "component_paths": {"linear": str(args.linear_checkpoint), "ranker": str(directory / "checkpoint")}})
        training[mode] = {key: report[key] for key in ("oof", "parameter_count", "training_seconds", "inference_asset_bytes")}
    write_json(output / "training_summary.json", {"fit_count": 12, "training_seconds": time.perf_counter()-tick, "methods": training})
    print(json.dumps(training, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
