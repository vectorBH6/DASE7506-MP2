"""用已有 train OOF 分数各拟合一个温度，冻结等权概率融合，不重训成员。"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from common import DATA
from solution.group_ranker.features import best_group
from solution.group_ranker.submission_ensemble import average_scores, create_bundle, digest
from solution.group_ranker.training import metrics, write_json

# 固定边界与精度，不根据 validation 或 test 修改；beta=1/T 时 NLL 为凸函数。
BETA_BOUNDS = (0.05, 20.0)
BISECTION_STEPS = 60


def fit_temperature(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if (scores.ndim != 2 or not len(scores) or labels.shape != (len(scores),)
            or not np.issubdtype(labels.dtype, np.integer) or np.isnan(scores).any()
            or np.isposinf(scores).any() or np.any(np.isfinite(scores).sum(axis=1) < 2)
            or np.any(labels < 0) or np.any(labels >= scores.shape[1])):
        raise ValueError("需要逐题对齐的有效组分数、-inf padding 和整数标签")
    indices = np.arange(len(scores))
    if not np.isfinite(scores[indices, labels]).all():
        raise ValueError("监督标签不能指向 padding")
    shifted = scores - scores.max(axis=1, keepdims=True)
    finite = np.where(np.isfinite(shifted), shifted, 0.0)
    target = shifted[indices, labels]

    def objective(beta):
        probability = np.exp(beta * shifted)
        normalizer = probability.sum(axis=1)
        probability /= normalizer[:, None]
        nll = np.mean(np.log(normalizer) - beta * target)
        gradient = np.mean(np.sum(probability * finite, axis=1) - target)
        return float(nll), float(gradient)

    low, high = BETA_BOUNDS
    boundary, steps = None, 0
    if objective(low)[1] >= 0:
        beta, boundary = low, "lower_beta"
    elif objective(high)[1] <= 0:
        beta, boundary = high, "upper_beta"
    else:
        # 导数单调，用二分求零点；无需增加优化库或搜索温度网格。
        for _ in range(BISECTION_STEPS):
            middle = (low + high) / 2
            if objective(middle)[1] > 0:
                high = middle
            else:
                low = middle
        beta, steps = (low + high) / 2, BISECTION_STEPS
    nll, gradient = objective(beta)
    return {"temperature": 1.0 / beta, "inverse_temperature": beta,
        "nll_before": objective(1.0)[0], "nll_after": nll,
        "final_gradient": gradient, "boundary": boundary, "bisection_steps": steps,
        "n": len(scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("已有实验不覆盖，请使用新目录")
    tick = time.perf_counter()
    members, hashes = {}, {}
    for name, checkpoint in (("linear", args.linear_checkpoint), ("mlp", args.mlp_checkpoint)):
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if config.get("training_split") != "train" or config.get("seed") != 7506:
            parser.error("只复用既有 train 训练、种子 7506 的 L0 和 M1")
        training = json.loads((checkpoint.parent / "training.json").read_text(encoding="utf-8"))
        if training.get("data_manifest_sha256") != digest(DATA / "manifest.json"):
            parser.error("成员训练记录与当前课程数据不一致")
        path = checkpoint.parent / "oof.npz"
        with np.load(path, allow_pickle=False) as archive:
            members[name] = {key: archive[key].copy() for key in ("scores", "labels", "folds")}
        hashes[name] = {"oof_sha256": digest(path), "training_sha256": digest(checkpoint.parent / "training.json"),
            "checkpoint_sha256": {p.name: digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}}
    left, right = members["linear"], members["mlp"]
    if (left["scores"].shape != right["scores"].shape
            or not np.array_equal(np.isfinite(left["scores"]), np.isfinite(right["scores"]))
            or not all(np.array_equal(left[key], right[key]) for key in ("labels", "folds"))):
        raise ValueError("成员 OOF 的题序、答案组、标签或折划分不一致")
    fitted = {name: fit_temperature(member["scores"], member["labels"]) for name, member in members.items()}
    temperatures = [fitted[name]["temperature"] for name in ("linear", "mlp")]
    scores = average_scores(left["scores"], right["scores"], temperatures=temperatures)
    uncalibrated = average_scores(left["scores"], right["scores"])
    args.output.mkdir(parents=True, exist_ok=True)
    asset_bytes = create_bundle(args.linear_checkpoint, args.mlp_checkpoint,
                               args.output / "checkpoint", temperatures=temperatures)
    # 这是在拟合温度所用 OOF 上的训练诊断，不伪装成独立 OOF 验证。
    np.savez(args.output / "oof.npz", scores=scores, labels=left["labels"], folds=left["folds"])
    code = Path(__file__).resolve().parents[2]
    paths = sorted({*Path(__file__).parent.glob("*.py"),
                    *(code / "solution/group_ranker").glob("*.py"), code / "solution/evaluate.py"})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "state": "complete",
        "command": [sys.executable, *sys.argv], "fit_count": 2, "base_model_fit_count": 0,
        "weights": [0.5, 0.5], "temperatures": temperatures,
        "calibration": fitted, "beta_bounds": list(BETA_BOUNDS), "bisection_steps": BISECTION_STEPS,
        "objective": "per-question answer-group NLL on supplied train OOF logits",
        "train_diagnostics": {"uncalibrated": metrics(uncalibrated, left["labels"]),
            "calibrated": metrics(scores, left["labels"]),
            "scope": "温度拟合数据上的描述性指标；成员 logits 是 OOF，但温度使用全部这些标签拟合，不能视为独立验证",
            "member_argmax_unchanged": all(np.array_equal(np.argmax(member["scores"], axis=1),
                np.argmax(member["scores"] / fitted[name]["temperature"], axis=1)) for name, member in members.items()),
            "fusion_choice_changes": int(np.sum(best_group(scores) != best_group(uncalibrated)))},
        "member_artifacts": hashes, "inference_asset_bytes": asset_bytes,
        "source_sha256": {str(p.relative_to(code)): digest(p) for p in paths},
        "data_manifest_sha256": digest(DATA / "manifest.json"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(), "threads": 1},
        "preparation_seconds": time.perf_counter() - tick,
        "scope": "既有 test 已接触；本轮仅 train OOF 拟合两个温度，validation 比较，不读取 test 参考、不运行 test，不按旧 test 调参、挑成员或权重",
    }
    write_json(args.output / "preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
