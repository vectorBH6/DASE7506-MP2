"""运行四种方法的官方 validation 评测，保存耗时、分层指标与纠错记录。"""
import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT.parent

from common import DATA, check_data, load_bank, parse_answer, read_jsonl
from solution.group_ranker.features import answer_groups

# 方法文件与是否需要 checkpoint 只在此登记；新增模型无需改动评分流程。
METHODS = {
    "baseline": ("baselines/submission.py", False),
    "voting": ("baselines/submission_voling_only.py", False),
    "embedding": ("baselines/submission_emb_only.py", False),
    "linear": ("group_ranker/submission_linear.py", True),
}


def accuracy(correct):
    n = len(correct)
    return {"correct": int(np.sum(correct)), "n": n,
            "accuracy": float(100 * np.mean(correct)) if n else None}


def bootstrap_interval(difference):
    """同一批题成对抽样；返回准确率差值的百分点区间。"""
    rng = np.random.default_rng(7506)
    sampled = rng.choice(difference, size=(2000, len(difference)), replace=True)
    return (100 * np.quantile(sampled.mean(axis=1), [0.025, 0.975])).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="待评测实验的 checkpoint 目录")
    parser.add_argument("--output", type=Path, help="默认写入该实验的 validation/ 目录")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.checkpoint.parent / "validation"
    if not (args.checkpoint / "model.npz").is_file():
        parser.error("请先训练并指定 checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录已含记录，请使用新的 --output 目录")
    args.output.mkdir(parents=True, exist_ok=True)
    output, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    # 编排进程占 1 线程，官方子进程最多 3 线程；两者合计不超过 4。
    child_env = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "3")}
    records, predictions = {}, {}
    for name, (method, needs_checkpoint) in METHODS.items():
        prediction_path, score_path = output / f"{name}-predictions.json", output / f"{name}-score.json"
        command = [
            sys.executable, str(OFFICIAL / "predict.py"), "--method", str(ROOT / method),
            "--split", "validation", "--output", str(prediction_path),
        ]
        if needs_checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        scoring = [
            sys.executable, str(OFFICIAL / "score_predictions.py"), "--split", "validation",
            "--predictions", str(prediction_path), "--output", str(score_path),
        ]
        # 外部计时覆盖两个新进程，包含启动、import、哈希检查与载入。
        tick = time.perf_counter()
        for process_command in (command, scoring):
            subprocess.run(process_command, cwd=OFFICIAL, env=child_env,
                           check=True, capture_output=True, text=True)
        elapsed = time.perf_counter() - tick
        records[name] = {
            "official": json.loads(score_path.read_text(encoding="utf-8")),
            "total_wall_seconds": elapsed, "commands": [command, scoring],
            "method_sha256": hashlib.sha256((ROOT / method).read_bytes()).hexdigest(),
        }
        predictions[name] = json.loads(prediction_path.read_text(encoding="utf-8"))["selected"]
        print(f"{name}: accuracy={records[name]['official']['accuracy']:.4f}%, wall={elapsed:.3f}s", flush=True)

    # 参考答案仅在分析阶段读取，不传给任何 predict 函数。
    check_data(DATA, ("validation",))
    bank = load_bank("validation")  # 分层分析只需要文本，不重复加载 embedding。
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
    equivalent = bool(np.array_equal(correctness["baseline"], correctness["voting"]))
    if not equivalent:
        raise ValueError("主基线与纯投票的逐题得分不一致，需要检查实现")
    baseline = correctness["baseline"]
    for name, correct in correctness.items():
        records[name]["corrected_vs_baseline"] = int(np.sum(~baseline & correct))
        records[name]["regressed_vs_baseline"] = int(np.sum(baseline & ~correct))
        if name != "baseline":
            difference = correct.astype(int) - baseline.astype(int)
            records[name]["difference_95ci_pp"] = bootstrap_interval(difference)
            records[name]["conflict_difference_95ci_pp"] = bootstrap_interval(difference[hard])
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpuinfo.exists() else platform.processor()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": "validation",
        "n": len(bank), "conflict_n": int(hard.sum()),
        "baseline_voting_score_equivalent": equivalent,
        "inference_asset_bytes": sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file()),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "cpu": cpu, "platform": platform.platform(),
                        "thread_budget_total": 4, "coordinator_threads": 1, "child_threads": 3},
        "methods": records,
        "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in checkpoint.iterdir() if p.is_file()},
        "data_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"记录已保存：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
