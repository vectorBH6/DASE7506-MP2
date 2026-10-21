"""固定等权概率融合：复用线性与 M1 权重；模块命令负责打包和 train OOF 分析。"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import PROTOCOL
from solution.group_ranker.features import EMBED_DIM, answer_groups, best_group, build_features, standardize
from solution.group_ranker.mlp import forward
from solution.group_ranker.submission_linear import load_model as load_linear
from solution.group_ranker.submission_mlp import load_model as load_mlp

MODEL = "equal-probability-v1"
CALIBRATED_MODEL = "temperature-equal-probability-v1"
ASSETS = ("config.json", *(f"{kind}/{name}" for kind in ("linear", "mlp")
                          for name in ("config.json", "model.npz", "asset_manifest.json")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_temperatures(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("两个成员的温度必须是有限正数")
    return values


def average_scores(left, right, *, temperatures=(1.0, 1.0)):
    """每题、每模型单独 softmax 后各占一半；返回 log 概率供共用排序接口使用。"""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim not in (1, 2):
        raise ValueError("两个模型的答案组分数必须对齐")
    mask = np.isfinite(left)
    if (not np.array_equal(mask, np.isfinite(right)) or not mask.any(axis=-1).all()
            or np.isnan(left).any() or np.isnan(right).any()
            or np.isposinf(left).any() or np.isposinf(right).any()):
        raise ValueError("只允许有限分数及一致的 -inf padding")
    probabilities = []
    for scores, temperature in zip((left, right), validate_temperatures(temperatures)):
        p = np.exp((scores - scores.max(axis=-1, keepdims=True)) / temperature)
        probabilities.append(p / p.sum(axis=-1, keepdims=True))
    probability = (probabilities[0] + probabilities[1]) * 0.5
    return np.where(mask, np.log(np.maximum(probability, np.finfo(np.float64).tiny)), -np.inf)


def _load_bundle(checkpoint, *, calibrated=False):
    if checkpoint is None:
        raise ValueError("请指定包含两个成员的融合 checkpoint")
    directory = Path(checkpoint)
    manifest = json.loads((directory / "asset_manifest.json").read_text(encoding="utf-8"))
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if files != set(ASSETS) | {"asset_manifest.json"} or set(manifest["sha256"]) != set(ASSETS):
        raise ValueError("融合资产不完整或含未登记文件")
    if any(digest(directory / name) != value for name, value in manifest["sha256"].items()):
        raise ValueError("融合资产哈希不符")
    if sum((directory / name).stat().st_size for name in files) > 16 * 2**20:
        raise ValueError("融合资产超过 16 MiB")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    mlp_config = json.loads((directory / "mlp/config.json").read_text(encoding="utf-8"))
    if (config.get("protocol") != PROTOCOL
            or config.get("model") != (CALIBRATED_MODEL if calibrated else MODEL)
            or config.get("weights") != [0.5, 0.5]
            or config.get("components") != ["linear", "mlp"]
            or mlp_config.get("feature_set") != "text" or mlp_config.get("ablation", "none") != "none"
            or mlp_config.get("loss", "ce") != "ce"):
        raise ValueError("本实验只支持预定的线性 + M1 等权概率融合")
    if calibrated:
        if config.get("calibration_split") != "train_oof":
            raise ValueError("校准参数必须来自 train OOF")
        temperatures = validate_temperatures(config.get("temperatures"))
    else:
        if config.get("temperature") != 1.0:
            raise ValueError("原融合固定使用温度 1")
        temperatures = np.ones(2)
    return load_linear(directory / "linear"), load_mlp(directory / "mlp"), temperatures


def load_model(checkpoint):
    linear, mlp, _ = _load_bundle(checkpoint)
    return linear, mlp


def predict_profiled(bank, arrays, checkpoint, *, include_scores=False, calibrated=False):
    profile = {"feature_seconds": 0.0, "forward_seconds": 0.0}
    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape != (len(bank), 4, EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("bank 与 embedding 的数量、形状或数值不符合协议")
    if include_scores:
        profile["group_scores"] = [[] for _ in bank]
    if not bank:
        return [], profile
    linear, mlp, temperatures = _load_bundle(checkpoint, calibrated=calibrated)
    choices = [0] * len(bank)
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
                    question=bank[i]["question"], feature_set="text", diagnostics=profile))
                profile["feature_seconds"] += time.perf_counter() - tick
        if not pending:
            continue
        tick = time.perf_counter()
        text = np.concatenate(features)
        basic = text.copy()
        basic[:, 2 * EMBED_DIM + 16:] = 0  # 线性成员严格沿用基础特征；两套 scaler 分别使用。
        left = standardize(basic, linear["mean"], linear["scale"], linear["active"]) @ linear["weight"]
        right, _ = forward(standardize(text, mlp["mean"], mlp["scale"], mlp["active"]), mlp)
        offset = 0
        for i, groups in pending:
            end = offset + len(groups)
            scores = average_scores(left[offset:end], right[offset:end], temperatures=temperatures)
            choices[i] = int(min(list(groups.values())[int(best_group(scores))]))
            if include_scores:
                profile["group_scores"][i] = scores.tolist()
            offset = end
        profile["forward_seconds"] += time.perf_counter() - tick
    return choices, profile


def predict(bank, arrays, checkpoint):
    return predict_profiled(bank, arrays, checkpoint)[0]


def create_bundle(linear, mlp, directory, *, temperatures=None):
    """自包含打包，普通文件复制使权重计数与外部路径无关；不重新训练。"""
    import shutil
    from solution.group_ranker.training import write_json

    calibrated = temperatures is not None
    if calibrated:
        temperatures = validate_temperatures(temperatures).tolist()
    if directory.exists():
        raise ValueError("请使用新的 checkpoint 目录")
    load_linear(linear)
    load_mlp(mlp)
    directory.mkdir(parents=True)
    for kind, source in (("linear", linear), ("mlp", mlp)):
        (directory / kind).mkdir()
        for name in ("model.npz", "config.json", "asset_manifest.json"):
            shutil.copyfile(source / name, directory / kind / name)
    config = {"protocol": PROTOCOL, "model": CALIBRATED_MODEL if calibrated else MODEL,
        "weights": [0.5, 0.5], "components": ["linear", "mlp"],
        "aggregation": "per-question answer-group probabilities"}
    if calibrated:
        config.update(temperatures=temperatures, calibration_split="train_oof")
    else:
        config["temperature"] = 1.0
    write_json(directory / "config.json", config)
    write_json(directory / "asset_manifest.json", {"sha256": {name: digest(directory / name) for name in ASSETS}})
    _load_bundle(directory, calibrated=calibrated)
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def main():
    import argparse
    from datetime import datetime, timezone
    import sys
    from solution.group_ranker.training import metrics, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--stability-dir", type=Path, help="可选；须为相同结构的 7507/7508 OOF 复核")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mlp_config = json.loads((args.mlp_checkpoint / "config.json").read_text(encoding="utf-8"))
    if mlp_config.get("seed") != 7506:
        parser.error("固定方案只打包既有 7506 最终模型，其余种子只作 OOF 复核")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("已有实验不覆盖，请使用新目录")
    args.output.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "state": "running",
        "command": [sys.executable, *sys.argv], "weights": [0.5, 0.5], "temperature": 1.0,
        "fit_count": 0, "search_count": 0, "final_mlp_seed": 7506,
        "component_directories": {"linear": str(args.linear_checkpoint), "mlp": str(args.mlp_checkpoint)},
        "stability_seeds_are_diagnostics_only": True,
        "scope": "原路线预留的事后扩展；已看过单模型 test。配方在本轮评测前固定，不按 test 调整或选型。",
        "seeds": []}
    write_json(args.output / "preparation.json", report)
    with np.load(args.linear_checkpoint.parent / "oof.npz", allow_pickle=False) as archive:
        left, labels, folds = archive["scores"], archive["labels"], archive["folds"]
    linear_correct = best_group(left) == labels
    rescues = []
    paths = [(7506, args.mlp_checkpoint.parent / "oof.npz")]
    if args.stability_dir:
        stability = json.loads((args.stability_dir / "training.json").read_text(encoding="utf-8"))
        if (stability["planned_search"][0]["hidden"] != mlp_config["hidden"]
                or stability.get("feature_set") != mlp_config.get("feature_set")):
            parser.error("稳定性记录与当前 MLP 的结构或特征不一致")
        paths += [(seed, args.stability_dir / f"oof-{seed}.npz") for seed in (7507, 7508)]
    for seed, path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if not (np.array_equal(labels, archive["labels"]) and np.array_equal(folds, archive["folds"])):
                raise ValueError("OOF 标签或折划分不一致")
            right = archive["scores"]
        mlp_correct = best_group(right) == labels
        rescue = ~linear_correct & mlp_correct
        rescues.append(rescue)
        scores = average_scores(left, right)
        record = {"mlp_seed": seed, "mlp_oof_sha256": digest(path),
            "linear": metrics(left, labels), "mlp": metrics(right, labels), "ensemble": metrics(scores, labels),
            "mlp_rescues": int(rescue.sum()), "mlp_regresses": int(np.sum(linear_correct & ~mlp_correct)),
            "folds": [{"fold": int(f), "mlp_rescues": int(np.sum(rescue & (folds == f))),
                       "ensemble": metrics(scores[folds == f], labels[folds == f])} for f in sorted(set(folds))]}
        report["seeds"].append(record)
        if seed == 7506:
            np.savez(args.output / "oof.npz", scores=scores, labels=labels, folds=folds)
    report["stable_rescues_all_three_seeds"] = int(np.logical_and.reduce(rescues).sum()) if len(rescues) == 3 else None
    report["complementarity_in_every_seed_and_fold"] = all(
        fold["mlp_rescues"] > 0 for record in report["seeds"] for fold in record["folds"])
    report["linear_oof_sha256"] = digest(args.linear_checkpoint.parent / "oof.npz")
    report["inference_asset_bytes"] = create_bundle(args.linear_checkpoint, args.mlp_checkpoint, args.output / "checkpoint")
    report.update(state="complete", preparation_seconds=time.perf_counter() - tick,
                  source_sha256={Path(__file__).name: digest(Path(__file__))}, numpy=np.__version__)
    write_json(args.output / "preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
