"""调用官方 validation 工具，比较基线、线性和可选 MLP，保存资源与分层记录。"""
import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT.parent

from common import DATA, check_data, load_bank, load_inputs, parse_answer, read_jsonl
from solution.group_ranker.features import answer_groups, best_group, feature_mask
from solution.group_ranker.training import metrics

# 方法文件与是否需要 checkpoint 只在此登记；新增模型无需改动评分流程。
METHODS = {
    "baseline": ("baselines/submission.py", None),
    "voting": ("baselines/submission_voling_only.py", None),
    "embedding": ("baselines/submission_emb_only.py", None),
    "b1": ("baselines/submission_tiebreak.py", None),
    "b2": ("baselines/submission_group_center.py", None),
    "linear": ("group_ranker/submission_linear.py", "linear"),
}


def thread_count(pid):
    try:
        return next(int(line.split()[1]) for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                    if line.startswith("Threads:"))
    except (FileNotFoundError, ProcessLookupError):
        return 0


def run_official(command, env):
    """Linux 下每毫秒采样父/子进程线程；监测工作在父进程主线程完成。"""
    monitored = Path("/proc/self/status").exists()
    peaks = {"coordinator_peak_threads": 0, "child_peak_threads": 0, "total_peak_threads": 0}
    samples = 0
    with subprocess.Popen(command, cwd=OFFICIAL, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        while process.poll() is None:
            if monitored:
                parent, child = thread_count(os.getpid()), thread_count(process.pid)
                for key, value in zip(peaks, (parent, child, parent + child)):
                    peaks[key] = max(peaks[key], value)
                samples += 1
            time.sleep(0.001)
        stdout, stderr = process.communicate()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    if monitored and peaks["total_peak_threads"] > 4:
        raise RuntimeError(f"评测总线程超过 4：{peaks}")
    return {"command": command, "stdout": stdout, "stderr": stderr, "thread_samples": samples,
            **(peaks if monitored else dict.fromkeys(peaks))}


def accuracy(correct):
    n = len(correct)
    return {"correct": int(np.sum(correct)), "n": n,
            "accuracy": float(100 * np.mean(correct)) if n else None}


def bootstrap_interval(difference):
    """同一批题成对抽样；返回准确率差值的百分点区间。"""
    rng = np.random.default_rng(7506)
    sampled = rng.choice(difference, size=(2000, len(difference)), replace=True)
    return (100 * np.quantile(sampled.mean(axis=1), [0.025, 0.975])).tolist()


def confidence_summary(scores, groups, answers, correct, hard):
    """按题归一化，记录 NLL/Brier/ECE；固定分桶只作描述，不在这里拟合参数。"""
    confidence, nll, brier, probability_margin, score_margin = [], [], [], [], []
    for row, group, answer in zip(scores, groups, answers):
        logits = np.asarray(row, dtype=np.float64)
        shifted = logits - logits.max()
        log_z = np.log(np.exp(shifted).sum())
        probabilities = np.exp(shifted - log_z)
        target = list(group).index(answer)
        ordered = np.sort(probabilities)
        confidence.append(float(ordered[-1]))
        nll.append(float(log_z - shifted[target]))
        brier.append(float(np.sum(probabilities ** 2) - 2 * probabilities[target] + 1))
        probability_margin.append(float(ordered[-1] - ordered[-2]) if len(row) > 1 else 1.0)
        score_margin.append(float(np.sort(logits)[-1] - np.sort(logits)[-2]) if len(row) > 1 else 0.0)
    confidence, nll, brier = np.array(confidence), np.array(nll), np.array(brier)
    bins = np.minimum((confidence * 10).astype(int), 9)

    def calibration(selected):
        if not selected.any():
            return {"n": 0, "nll": None, "brier": None, "ece_10_bins": None}
        ece = sum(np.sum(selected & (bins == bucket)) * abs(
            float(correct[selected & (bins == bucket)].mean())
            - float(confidence[selected & (bins == bucket)].mean()))
            for bucket in range(10) if np.any(selected & (bins == bucket))) / np.sum(selected)
        return {"n": int(np.sum(selected)), "nll": float(nll[selected].mean()),
                "brier": float(brier[selected].mean()), "ece_10_bins": float(ece)}

    counts = np.array([len(group) for group in groups])
    result = {"nll": float(nll.mean()), "conflict_nll": float(nll[hard].mean()),
              "conflict_mean_max_probability": float(confidence[hard].mean()),
              "conflict_mean_probability_margin": float(np.array(probability_margin)[hard].mean()),
              "conflict_mean_score_margin": float(np.array(score_margin)[hard].mean()),
              "calibration": {"overall": calibration(np.ones(len(scores), dtype=bool)),
                  "conflict": calibration(hard),
                  "by_group_count": {str(g): calibration(counts == g) for g in sorted(set(counts))},
                  "definitions": "NLL 为逐题交叉熵；Brier 为组概率与 one-hot 的平方差之和；ECE 为最高组概率的十个等宽桶加权绝对偏差，非校准保证"},
              "conflict_confidence_bins": []}
    for low, high in zip((0, .5, .7, .85, .95), (.5, .7, .85, .95, 1.0)):
        selected = hard & (confidence >= low) & ((confidence < high) if high < 1 else confidence <= high)
        result["conflict_confidence_bins"].append({"low_inclusive": low, "high": high,
            **accuracy(correct[selected]),
            "mean_confidence": float(confidence[selected].mean()) if selected.any() else None})
    return result


def oof_summary(path, patterns):
    """只分析 train 中已保存的折外分数，不重新训练或接触 test。"""
    with np.load(path, allow_pickle=False) as archive:
        scores, labels = archive["scores"], archive["labels"]
    if len(labels) != len(patterns):
        raise ValueError("OOF 与训练冲突题数量不一致")
    correct = best_group(scores) == labels
    counts = np.isfinite(scores).sum(axis=1)
    return {"overall": metrics(scores, labels),
        "by_group_count": {str(g): accuracy(correct[counts == g]) for g in sorted(set(counts))},
        "by_vote_pattern": {p: accuracy(correct[patterns == p]) for p in sorted(set(patterns))},
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="线性模型 checkpoint")
    parser.add_argument("--mlp-checkpoint", type=Path, nargs="+", help="同时比较 MLP 及各项消融 checkpoint")
    parser.add_argument("--ensemble-checkpoint", type=Path, help="额外比较固定等权概率融合")
    parser.add_argument("--linear-text-checkpoint", type=Path, help="额外比较线性 + 文本特征")
    parser.add_argument("--calibrated-checkpoint", type=Path, help="额外比较 train OOF 温度校准后的等权融合")
    parser.add_argument("--tree-checkpoint", type=Path, help="额外比较同 M1 特征的浅树排序器")
    parser.add_argument("--output", type=Path, help="默认写入该实验的 validation/ 目录")
    args = parser.parse_args()
    if args.output is None:
        args.output = (args.tree_checkpoint or args.calibrated_checkpoint or args.linear_text_checkpoint or args.ensemble_checkpoint or (
            args.mlp_checkpoint[-1] if args.mlp_checkpoint else args.checkpoint)).parent / "validation"
    checkpoints = {"linear": args.checkpoint.resolve()}
    methods, configs = dict(METHODS), {}
    for directory in args.mlp_checkpoint or []:
        if not (directory / "config.json").is_file():
            parser.error(f"缺少配置：{directory}")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        name = {"basic": "mlp", "text": "m1", "arithmetic": "m2"}[config.get("feature_set", "basic")]
        if config.get("ablation", "none") != "none":
            name += "_" + config["ablation"]
        if config.get("loss", "ce") != "ce":
            name += "_" + config["loss"]
        if name in checkpoints:
            parser.error(f"重复的 MLP 特征组：{name}")
        checkpoints[name] = directory.resolve()
        configs[name] = config
        methods[name] = ("group_ranker/submission_mlp.py", name)
    if args.linear_text_checkpoint:
        directory = args.linear_text_checkpoint
        if not (directory / "config.json").is_file():
            parser.error("缺少文本线性配置，请先训练")
        checkpoints["linear_text"] = directory.resolve()
        configs["linear_text"] = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        methods["linear_text"] = ("linear_text/submission.py", "linear_text")
    if args.tree_checkpoint:
        directory = args.tree_checkpoint
        if not (directory / "config.json").is_file():
            parser.error("缺少树模型配置，请先训练")
        checkpoints["tree"] = directory.resolve()
        configs["tree"] = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        methods["tree"] = ("tree_ranker/submission.py", "tree")
    if any(not (directory / "model.npz").is_file() for directory in checkpoints.values()):
        parser.error("请先训练并指定已有 checkpoint")
    for name, directory, method in (
            ("ensemble", args.ensemble_checkpoint, "group_ranker/submission_ensemble.py"),
            ("calibrated", args.calibrated_checkpoint, "calibrated_ensemble/submission.py")):
        if directory is None:
            continue
        if not (directory / "config.json").is_file():
            parser.error("缺少融合配置，请先打包")
        checkpoints[name] = directory.resolve()
        configs[name] = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        methods[name] = (method, name)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录已含记录，请使用新的 --output 目录")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    # 编排进程占 1 线程，官方子进程最多 3 线程；两者合计不超过 4。
    child_env = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "3")}
    records, predictions = {}, {}
    for name, (method, checkpoint_name) in methods.items():
        prediction_path, score_path = output / f"{name}-predictions.json", output / f"{name}-score.json"
        command = [
            sys.executable, str(OFFICIAL / "predict.py"), "--method", str(ROOT / method),
            "--split", "validation", "--output", str(prediction_path),
        ]
        if checkpoint_name:
            command += ["--checkpoint", str(checkpoints[checkpoint_name])]
        scoring = [
            sys.executable, str(OFFICIAL / "score_predictions.py"), "--split", "validation",
            "--predictions", str(prediction_path), "--output", str(score_path),
        ]
        # 外部计时覆盖两个新进程，包含启动、import、哈希检查与载入。
        tick = time.perf_counter()
        processes = [run_official(process_command, child_env) for process_command in (command, scoring)]
        elapsed = time.perf_counter() - tick
        records[name] = {
            "official": json.loads(score_path.read_text(encoding="utf-8")),
            "total_wall_seconds": elapsed, "commands": [command, scoring], "processes": processes,
            "method_sha256": hashlib.sha256((ROOT / method).read_bytes()).hexdigest(),
        }
        records[name]["inference_asset_bytes"] = (
            sum(p.stat().st_size for p in checkpoints[checkpoint_name].rglob("*") if p.is_file())
            if checkpoint_name else 0)
        predictions[name] = json.loads(prediction_path.read_text(encoding="utf-8"))["selected"]
        print(f"{name}: accuracy={records[name]['official']['accuracy']:.4f}%, wall={elapsed:.3f}s", flush=True)

    # 参考答案仅在分析阶段读取，不传给任何 predict 函数。
    check_data(DATA, ("validation",))
    bank, arrays = load_inputs("validation")
    # 单独重放无标签预测以记录特征/前向耗时，完整资源计时仍以上方新进程为准。
    group_scores = {}
    for name, config in configs.items():
        module = import_module("solution." + methods[name][0].removesuffix(".py").replace("/", "."))
        choices, profile = module.predict_profiled(bank, arrays, checkpoints[name], include_scores=True)
        if choices != predictions[name]:
            raise ValueError("分析重放与官方预测不一致")
        group_scores[name] = profile.pop("group_scores")
        records[name]["profile_replay"] = profile
        records[name]["config"] = config
        records[name]["enabled_input_dimensions"] = int(feature_mask(
            "text" if name in ("ensemble", "calibrated") else config.get("feature_set", "basic"),
            config.get("ablation", "none")).sum())
    records["linear"]["enabled_input_dimensions"] = int(feature_mask("basic").sum())
    training_patterns = []
    for row in load_bank("train"):
        group = answer_groups(row["candidates"])
        if len(group) >= 2:
            training_patterns.append("+".join(map(str, sorted(map(len, group.values()), reverse=True))))
    for name, checkpoint in checkpoints.items():
        path = checkpoint.parent / "oof.npz"
        if path.is_file():
            records[name]["train_oof"] = oof_summary(path, np.array(training_patterns))
            if configs.get(name, {}).get("calibration_split") == "train_oof":
                records[name]["train_oof"]["scope"] = "成员 logits 为 OOF，但温度使用这些标签拟合；此处是拟合数据诊断，不是独立验证"
    gold = read_jsonl(DATA / "gsm8k_validation.jsonl")
    answers = [parse_answer(row["answer"]) for row in gold]
    groups = [answer_groups(row["candidates"]) for row in bank]
    group_counts = np.array([len(g) for g in groups])
    hard = group_counts >= 2
    patterns = np.array(["+".join(map(str, sorted((len(v) for v in g.values()), reverse=True))) or "none"
                         for g in groups])
    correctness = {}
    for name, choices in predictions.items():
        selected = [parse_answer(row["candidates"][j]["text"]) for row, j in zip(bank, choices)]
        correct = np.array([a is not None and a == reference for a, reference in zip(selected, answers)])
        if int(correct.sum()) != records[name]["official"]["correct"]:
            raise ValueError("分析指标与官方评分不一致")
        correctness[name] = correct
        records[name]["conflict"] = accuracy(correct[hard])
        records[name]["by_group_count"] = {
            str(g): accuracy(correct[group_counts == g]) for g in sorted(set(group_counts))}
        records[name]["by_vote_pattern"] = {
            p: accuracy(correct[patterns == p]) for p in sorted(set(patterns))}
        if name in group_scores:
            records[name]["confidence"] = confidence_summary(group_scores[name], groups, answers, correct, hard)
    if "calibrated" in configs:
        records["calibrated"]["component_calibration"] = {}
        for name, temperature in zip(("linear", "m1"), configs["calibrated"]["temperatures"]):
            if name in group_scores:
                records["calibrated"]["component_calibration"][name] = {
                    "temperature": temperature,
                    "before": records[name]["confidence"]["calibration"],
                    "after": confidence_summary([np.asarray(row) / temperature for row in group_scores[name]],
                        groups, answers, correctness[name], hard)["calibration"]}
    equivalent = bool(np.array_equal(correctness["baseline"], correctness["voting"]))
    if not equivalent:
        raise ValueError("主基线与纯投票的逐题得分不一致，需要检查实现")
    baseline = correctness["baseline"]
    for name, correct in correctness.items():
        records[name]["corrected_vs_baseline"] = int(np.sum(~baseline & correct))
        records[name]["regressed_vs_baseline"] = int(np.sum(baseline & ~correct))
        records[name]["examples_vs_baseline"] = {}
        for label, selected in (("corrected", ~baseline & correct), ("regressed", baseline & ~correct)):
            records[name]["examples_vs_baseline"][label] = [
                {"id": bank[i]["id"], "question": bank[i]["question"], "reference_answer": answers[i],
                 "baseline_answer": parse_answer(bank[i]["candidates"][predictions["baseline"][i]]["text"]),
                 "selected_answer": parse_answer(bank[i]["candidates"][predictions[name][i]]["text"]),
                 "vote_pattern": str(patterns[i])} for i in np.flatnonzero(selected)[:3]]
        if name != "baseline":
            difference = correct.astype(int) - baseline.astype(int)
            records[name]["difference_95ci_pp"] = bootstrap_interval(difference)
            records[name]["conflict_difference_95ci_pp"] = bootstrap_interval(difference[hard])
        for reference_name in ("linear", "mlp", "m1", "ensemble", "linear_text"):
            if name == reference_name or reference_name not in correctness:
                continue
            reference = correctness[reference_name]
            difference = correct.astype(int) - reference.astype(int)
            records[name][f"vs_{reference_name}"] = {
                "corrected": int(np.sum(~reference & correct)),
                "regressed": int(np.sum(reference & ~correct)),
                "difference_95ci_pp": bootstrap_interval(difference),
                "conflict_difference_95ci_pp": bootstrap_interval(difference[hard]),
            }
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpuinfo.exists() else platform.processor()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": "validation",
        "n": len(bank), "conflict_n": int(hard.sum()),
        "baseline_voting_score_equivalent": equivalent,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "cpu": cpu, "platform": platform.platform(),
                        "thread_budget_total": 4, "coordinator_threads": 1, "child_threads": 3},
        "methods": records,
        "checkpoint_sha256": {name: {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in directory.rglob("*") if p.is_file()}
                              for name, directory in checkpoints.items()},
        "source_sha256": {str(p.relative_to(OFFICIAL)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for directory in sorted({ROOT, *((ROOT / method).parent for method, _ in methods.values())})
                          for p in sorted(directory.glob("*.py"))},
        "data_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
    }
    # 预先固定的有限选型规则；本机资源检查不能代替参考机资格确认。
    screened = [name for name, r in records.items()
                if r["inference_asset_bytes"] <= 16 * 2**20 and r["total_wall_seconds"] <= 30]
    selected = min(screened, key=lambda name: (-records[name]["official"]["accuracy"],
        records[name].get("enabled_input_dimensions", 0), records[name]["inference_asset_bytes"],
        records[name]["total_wall_seconds"], name))
    report["selection"] = {"method": selected,
        "checkpoint": str(checkpoints[selected]) if selected in checkpoints else None,
        "rule": "完整 validation 准确率；同分优先更少启用维度、更少资产、更短完整耗时、固定名称",
        "scope": "本机 validation 筛选；参考机资格待确认，未运行 test"}
    if args.ensemble_checkpoint:
        report["selection"]["scope"] = "事后扩展的 validation 比较；已有单模型 test 已见，融合配方预先固定，不能称新盲测"
    if args.linear_text_checkpoint:
        report["selection"]["scope"] = "仅 train OOF/validation 的开发比较；本轮未读取 test 参考或重跑 test，既有 test 已接触"
    if args.calibrated_checkpoint:
        report["selection"]["scope"] = "温度仅在 train OOF 拟合；validation 比较；既有 test 已接触，本轮不读取 test 参考或运行 test"
    if args.tree_checkpoint:
        report["selection"]["scope"] = "树仅在 train 五折 OOF 选深度和轮数，全量 train 重训；validation 比较；既有 test 已接触，本轮不读取 test 参考或运行 test"
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"记录已保存：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
