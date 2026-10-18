# MP2 作业实现

资料：[算法路线](../../class%20pdf/MP2最终算法路线.md) · [课程知识点](../../class%20pdf/MP2算法知识点总结与方案灵感.md) · [作业要求](../../guide/GUIDE.md) · [官方运行说明](../README_cn.md)。

## 结构

```text
code/
├── common.py、predict.py、score_predictions.py、data/ …  # 课程原始文件
└── solution/                  # 作业实现，按功能组织
    ├── baselines/             # 原基线、纯投票、纯 embedding
    ├── group_ranker/          # 共享特征、训练、预测接口与检查
    ├── evaluate.py            # 调用官方工具，统一验证与记录
    ├── requirements-train.txt
    └── runs/<实验名>/         # checkpoint/、训练记录、validation/、冻结后的 test/
```

源码在 `solution/` 中复用和扩展；每轮产物集中到对应实验目录，已有记录不覆盖。`runs/` 与虚拟环境默认由 Git 忽略；`linear_v1` 的权重及训练、验证、冻结测试记录作为复现快照随仓库提交。历史 JSON 保留生成时的命令和路径。

官方 `common.py` 负责数据与答案解析，`predict.py`、`score_predictions.py` 负责独立预测与评分；本目录只实现算法、训练和实验编排。

本文件只维护运行入口和简短结果；算法推导见路线文档，详细 CV、分层指标、耗时与哈希以实验目录内 JSON 为准。

更改特征顺序或维度时，应升级 `FEATURE_VERSION` 并重新训练，以保持 checkpoint 与特征一致。

## 运行

以下命令从 **`code/`** 执行，Python 3.12。通过 `python -m solution.…` 运行，自然导入同目录的官方 `common.py`，无需手工修改 `sys.path`。

首次安装；已有环境跳过：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 -m venv solution/.venv-train
solution/.venv-train/bin/python -m pip install -r solution/requirements-train.txt
```

训练与四方法验证；`linear_v1` 已有归档，新实验使用新名称：

```bash
run_dir=solution/runs/linear_v2
solution/.venv-train/bin/python -m solution.group_ranker.train_linear --output "$run_dir"
.venv/bin/python -m solution.evaluate --checkpoint "$run_dir/checkpoint"
```

训练默认按题五折 CV、种子 7506，仅比较预设的三个 L2 系数。评测默认写到所选实验的 `validation/`，也可用 `--output` 指定新目录。训练进程限制为 4 线程；评测编排进程占 1 线程，顺序启动的官方子进程最多 3 线程，合计最多 4 线程。

检查：

```bash
solution/.venv-train/bin/python -m unittest solution.group_ranker.test_group_ranker -v
.venv/bin/python -m unittest solution.group_ranker.test_group_ranker -v
```

纯 NumPy 环境会跳过训练梯度检查。推理入口是 `solution/group_ranker/submission_linear.py`，实现 `predict(bank, arrays, checkpoint)`，为每题返回一个 0–3 的 Python 整数。作业允许自定义文件名，通过 `--method` 传入；官方 `submission.py` 保留原始占位接口。

直接使用官方工具时，预留 1 线程给外层编排，并顺序运行预测、评分：

```bash
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3
```

## 实验摘要

`linear_v1`，2026-10-18：遍历全部 4,755 道 train 题；其中 2,641 道单答案组题无需组间决策，其余 2,114 道冲突题进行五折 OOF，选择 L2=0.01 后，再用全部 2,114 题从头训练最终模型。随后在完整 validation 上比较方法。

| 方法 | 完整 validation | 冲突题 |
|---|---:|---:|
| 原基线 / 纯投票 | 688/798，86.22% | 245/355，69.01% |
| 纯 embedding | 624/798，78.20% | 200/355，56.34% |
| 线性答案组排序 | 730/798，91.48% | 287/355，80.85% |

线性模型在 validation 净增 42 题。修正评测总线程配置后，四种方法的逐题选择均与原记录一致。冻结前检查：训练环境 7 项通过；纯 NumPy 推理环境 6 项通过、1 项训练梯度检查因无 SciPy 跳过。

### 冻结 test

2026-10-18，冻结上述模型后，仅运行这一方案的完整 test 预测和官方独立评分；未据 test 结果调参或重选模型。测试前保存代码、权重、配置的 SHA-256 和源码快照，测试后核对均未变化。

| 项目 | 结果 |
|---|---:|
| 官方 test accuracy | **375/467，80.29978586723769%** |
| 完整预测 + 独立评分外部总耗时 | **0.243 秒** |
| 实测总线程峰值 | **4**（编排 1 + 子进程 3） |
| 未压缩推理资产 | **28,771 字节**，约 28.10 KiB |
| 本机 CPU / 推理环境 | Core Ultra 9 275HX / Python 3.12.3、NumPy 2.5.3 |

耗时包含两个新进程的启动、导入、数据与资产哈希校验、加载、预处理、预测和评分；本机文件缓存状态未控制。**该成绩不是 Xeon Platinum 8457C 参考机测时，正式资格仍以参考机为准。**

记录：[官方评分](runs/linear_v1/test/score.json) · [耗时、线程与完整命令](runs/linear_v1/test/summary.json) · [冻结清单与检查结果](runs/linear_v1/test/freeze.json) · [冻结源码](runs/linear_v1/test/source.tar)。匹配权重位于 [checkpoint/](runs/linear_v1/checkpoint/)，冻结 ID 前缀为 `22c93832f59d`。

免训练复现（仍从 `code/` 执行，使用前述 3 线程环境设置；输出到新的目录）：

```bash
out=solution/runs/linear_v1/test_reproduction
if mkdir "$out"; then
  time {
    .venv/bin/python predict.py \
      --method solution/group_ranker/submission_linear.py \
      --checkpoint solution/runs/linear_v1/checkpoint --split test \
      --output "$out/predictions.json" &&
    .venv/bin/python score_predictions.py --split test \
      --predictions "$out/predictions.json" --output "$out/score.json"
  }
fi
```

原始记录集中在 `runs/linear_v1/`，历史基线记录在本地 `runs/baselines/`。仓库包含 `linear_v1` 的匹配 checkpoint 和冻结记录；新实验产物继续默认忽略。正式提交还需确认代码版本与权重匹配，并完成参考机测时、至多 10 页报告及实际 AI 辅助与复用披露；保留数据许可。
