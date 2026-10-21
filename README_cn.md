# DASE7506 MP2：线性与 MLP 答案组排序

**姓名：** Li Shanghang · **学号：** 3036705979

[English](README.md) | 简体中文

最终方案为 **L0＋M1**：基础线性答案组排序器与带文本特征的小型 MLP，各自输出组概率后等权融合。最终 checkpoint 为 `ensemble_v1`：**validation 740/798（92.73%）**，**test 377/467（80.72805139186295%）**。

**选型与实验历史：** 当前恢复原记录中按 validation 保留的方案。此前因 test 较高而临时指定注意力方案的决定已撤回，注意力保留为探索性对照。原融合开发时已经看过单模型 test，因此历史过程仍按事后扩展披露，不称新的独立盲测；恢复原选型不会抹去此前接触 test 的事实。权重、分数及历史实验记录均保留。

[英文报告 PDF，8 页](report/report.pdf) · [可编辑报告](report/report.md) · [开发与实验记录](code/solution/README.md) · [作业指南](guide/GUIDE_cn.md)

## 1. 最终算法与关键参数

| 项目 | 固定配置／已有测量 |
|---|---|
| 预测入口 | `code/solution/group_ranker/submission_ensemble.py`，实现 `predict(bank, arrays, checkpoint)` |
| 完整 checkpoint | `code/solution/runs/ensemble_v1/checkpoint/` |
| 输入 | 题目、四个候选文本、给定 `[N, 4, 1024]` embedding |
| 答案分组 | 用官方 `common.parse_answer` 合并最终数值等价的候选 |
| 特征结构 | 共 2088 个位置：1024 维组 embedding 均值、1024 维上下文交互、40 个标量位置 |
| L0 特征 | 2048 维向量＋16 个基础标量，其余位置关闭 |
| M1 特征 | 2048 维向量＋32 个基础／文本标量，8 个算术位置关闭 |
| 标准化 | 两个分支各自使用 train 统计量；OOF 每折单独拟合；裁剪至 `[-8, 8]` |
| L0 | 组交叉熵＋L2；train OOF 从 `0.0001/0.001/0.01` 选得 L2=`0.01`；L-BFGS-B，最多 300 次迭代 |
| M1 结构 | `2088 -> 64 -> 16 -> 1`，隐藏层 ReLU，各答案组共用打分网络 |
| M1 优化 | AdamW；学习率 `0.001`、weight decay `0.01`、dropout `0.1`、每批 64 题、梯度范数上限 1 |
| M1 轮数 | 最多 150 轮，折内 NLL 早停耐心值 15；五折最佳 `[5,5,5,3,6]`，最终重训 **5 轮** |
| 折划分与种子 | 按题、按票型分层五折，划分种子 `7506`；M1 每折种子 `7506 + fold`，最终种子 `7506`；`7507/7508` 仅作诊断，不增加集成成员 |
| 融合 | 题内组间各自 softmax，再计算 `0.5 * P_L0 + 0.5 * P_M1`；温度均固定为 1 |
| 存储的打分参数 | L0 为 2088，M1 为 134753，共 **136841**，含存储但关闭的坐标；标准化数组另计入资产 |
| 推理分块 | 每块最多 128 题 |
| 未压缩推理资产 | **590639 字节，约 0.563 MiB**，含两个成员、统计量、配置及清单 |
| 历史完整 test 耗时 | **0.414246 秒**，新预测进程＋独立评分进程，含启动、加载、特征及两个分支 |
| 评测总线程峰值 | **4**：协调进程 1＋子进程 3；预测、评分子进程顺序运行 |
| 历史测量环境 | Core Ultra 9 275HX；Linux；Python `3.12.3`；NumPy `2.5.3`；缓存状态未控制 |
| 冻结 ID | `79062305cd4b895b0c26d7f2eed97c321a68b1fe2b83d07e40905bb29b72db1b` |

训练扫描全部 4755 道 train 题。其中 2641 题只有一个有效答案组，无需组间选择且组交叉熵损失为零；两个分支均用全部 **2114 道冲突题**拟合。validation／test 参考答案不参与权重与标准化统计量拟合。

| 评测范围 | 正确数 | 准确率 |
|---|---:|---:|
| train 五折 OOF，仅冲突题 | 1717/2114 | 81.22% |
| 完整 validation | 740/798 | 92.73% |
| validation 冲突题 | 297/355 | 83.66% |
| **完整 test** | **377/467** | **80.72805139186295%** |

validation 含 443 道单组题，test 全部为冲突题，两个总体准确率反映的问题构成不同。

依据：[官方 test 分数](code/solution/runs/ensemble_v1/test/score.json)、[validation 选型](code/solution/runs/ensemble_v1/validation/summary.json)、[test／资源分析](code/solution/runs/ensemble_v1/test/summary.json)、[冻结清单](code/solution/runs/ensemble_v1/test/freeze.json)、[冻结源码](code/solution/runs/ensemble_v1/test/source.tar)、[后续保留原选型的记录](code/solution/runs/attention_v1/validation_summary.json)。

## 2. 算法原理简述

1. **相同答案先组队。** 用官方解析器合并数值等价的答案；仅一个有效组时直接选择，没有有效组时返回下标 0。
2. **为每组提取特征。** 使用给定 embedding 均值、票数、语义几何、上下文交互及轻量文本／题目关系特征。
3. **两个分支分别打分。** L0 对基础特征加权求和；M1 通过两个小隐藏层学习基础和文本特征的非线性组合，每组输出一个分数。
4. **概率等权融合。** 每个分支分别在题内组间 softmax，概率各占一半；选最高组并返回组内最小候选下标，接近平票按固定规则处理。

最终模型不包含注意力层，手工特征已表达候选组之间的关系。官方指标检查**最终数值是否正确**，不代表验证了每一步推理。注意力、扩容 MLP、树和校准保留为对照，不按它们较高的已观测 test 成绩替换原 validation 选型。

## 3. 作业要求对照与未完成项

| 要求 | 实现依据／状态 |
|---|---|
| 仅用给定 train 输入和参考答案拟合 | 权重／标准化仅用 train，采用 train 五折 OOF。 |
| validation 开发与选型 | 最终方案恢复到原记录的 validation 选择；历史接触 test 的情况仍披露，不称独立盲测。 |
| 数据／解析器／评分器不变，无外部数据或预训练权重 | 使用给定数据／embedding 和官方工具，排序模型本地训练。 |
| 离线选择，无生成／API／答案查表或固定题目 ID | NumPy 推理按传入输入选择原候选。 |
| 未压缩推理资产不超过 16 MiB | 完整八个文件合计 **0.563 MiB**，满足上限。 |
| 所有评测进程合计最多 4 线程 | 历史本机总峰值 **4**。 |
| Xeon Platinum 8457C 完整评测不超过 30 秒 | 本机完整评测 **0.414 秒**；**参考 CPU 资格尚未验证**。 |
| CPU 或最多一张 RTX 4090 训练 | 全部 CPU；L0 搜索／重训 **46.23 秒**，固定 M1 五折／重训 **69.60 秒**，融合打包 **0.011 秒**；不含此前结构搜索。 |
| 报告不超过 10 页，公开匹配源码、权重及记录 | 八页报告、匹配源码、权重与记录一并纳入版本控制；正式提交使用可公开访问的版本。 |

`code/solution/runs/` 纳入版本控制，包含 checkpoint、OOF、官方分数、实验记录与冻结源码快照；虚拟环境、缓存及报告临时文件继续忽略。当前源码在历史冻结后已有演进，因此 5.2 节明确使用原冻结源码。哈希证明文件完整性，不能证明开发过程未受此前 test 反馈影响。

## 4. 代码框架

```text
DASE7506-MP2/
├── README.md / README_cn.md          # 最终方案与复现命令
├── report/                          # 英文正文、PDF、图表和生成脚本
├── guide/ / class pdf/               # 作业要求与学习资料
└── code/
    ├── common.py                    # 官方解析器与数据接口
    ├── predict.py / score_predictions.py
    ├── data/                        # 给定输入与表示
    └── solution/
        ├── group_ranker/
        │   ├── features.py / evidence.py     # 共享答案组与文本特征
        │   ├── training.py / mlp.py          # 训练工具与 NumPy MLP
        │   ├── train_linear.py / train_mlp.py
        │   ├── submission_linear.py / submission_mlp.py
        │   └── submission_ensemble.py        # 最终预测入口与打包
        ├── baselines/ / linear_text/
        ├── calibrated_ensemble/ / tree_ranker/
        ├── attention_ranker/        # 探索性注意力与门控对照
        ├── requirements-train.txt
        └── runs/ensemble_v1/
            ├── checkpoint/          # 完整最终推理资产
            ├── validation/          # 开发比较与选型
            └── test/                # 官方分数、预测和冻结源码
```

推理仅需给定输入、匹配源码和完整 checkpoint，不读取 OOF 预测、参考答案或其他实验的权重。

## 5. 复现命令指南

### 5.1 统一环境与权重

从项目根目录进入 `code/`，复用 **`code/solution/.venv`**。首次克隆时，仅在缺少它的情况下创建这一个环境：

```bash
cd code
unset PYTHONPATH
if [ ! -x solution/.venv/bin/python ]; then
  python3.12 -m venv solution/.venv
fi
solution/.venv/bin/python -m pip install -r solution/requirements-train.txt
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3
```

统一环境固定 NumPy `1.26.4`、SciPy `1.11.4`、CatBoost `1.2.8`，使用 Python 3.12。最终推理仅需 NumPy 和标准库；SciPy 用于 L0 训练，CatBoost 用于历史树对照。历史 **0.414 秒**测量使用 NumPy **2.5.3**，与当前环境重放属于不同测量。

完整八个文件应放在 `solution/runs/ensemble_v1/checkpoint/`：

```text
checkpoint/
├── config.json
├── asset_manifest.json
├── linear/   # model.npz、config.json、asset_manifest.json
└── mlp/      # model.npz、config.json、asset_manifest.json
```

同时提供 `solution/runs/ensemble_v1/test/source.tar`，作为匹配的历史源码。加载器检查资产哈希、形状和大小。

### 5.2 不重新训练，复现官方分数

以下把冻结源码解压到新的实验目录，复用原始数据和 checkpoint，避免依赖后续重构后的源码；解压不会创建虚拟环境。`time` 包含新预测进程和独立评分进程，这段 shell 本身不采样线程峰值。

```bash
out="$PWD/solution/runs/ensemble_reproduction"
checkpoint="$PWD/solution/runs/ensemble_v1/checkpoint"
data="$PWD/data"
if mkdir "$out"; then
  tar -xf solution/runs/ensemble_v1/test/source.tar -C "$out"
  for split in validation test; do
    time {
      solution/.venv/bin/python "$out/code/predict.py" \
        --method "$out/code/solution/group_ranker/submission_ensemble.py" \
        --checkpoint "$checkpoint" --data-dir "$data" \
        --split "$split" --output "$out/$split-predictions.json" &&
      solution/.venv/bin/python "$out/code/score_predictions.py" \
        --data-dir "$data" --split "$split" \
        --predictions "$out/$split-predictions.json" --output "$out/$split-score.json"
    }
  done
fi
```

对应正确数为 **validation 740/798、test 377/467**。记录的官方 test `accuracy` 字段为百分数 **`80.72805139186295`**。当前环境的耗时可能不同，参考机资格仍未验证。

### 5.3 按固定配方重训，可选

当前源码可在统一环境中重训已选配方。L0 重复原 train OOF 正则选择；M1 从最终 checkpoint 复用已选的 64/16 结构与优化参数。更早的 M0 结构搜索和诊断种子见 [solution/README](code/solution/README.md)，此处不重复结构搜索，也不增加种子集成成员。

```bash
base=solution/runs/final_retrain
if mkdir "$base"; then
  solution/.venv/bin/python -m solution.group_ranker.train_linear \
    --l2 0.0001 0.001 0.01 --folds 5 --seed 7506 --maxiter 300 \
    --output "$base/linear" &&
  solution/.venv/bin/python -m solution.group_ranker.train_mlp \
    --feature-set text --reuse-config solution/runs/ensemble_v1/checkpoint/mlp/config.json \
    --output "$base/m1" &&
  solution/.venv/bin/python -m solution.group_ranker.submission_ensemble \
    --linear-checkpoint "$base/linear/checkpoint" \
    --mlp-checkpoint "$base/m1/checkpoint" --output "$base/ensemble"
fi
```

新权重在 `$base/ensemble/checkpoint/`。重训仅使用 train，不自动评测 test；它不是复现分数的必要步骤，不同源码／库版本下不保证历史字节完全相同。复现本文成绩使用 5.2 节的原权重与源码快照。

### 5.4 重新生成报告

从项目根目录执行，沿用同一个环境：

```bash
code/solution/.venv/bin/python -m pip install -r report/requirements.txt
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  code/solution/.venv/bin/python report/build_report.py
```

生成脚本读取已有记录，重建四张图、`report/evidence.json` 和 PDF，不运行训练、预测、评分或模型单元检查。

## 6. Codex 辅助声明

OpenAI Codex 辅助算法讨论、实现与重构、工程检查、实验执行／记录，以及文档与报告编写。数值来自已有本地记录和官方评分 JSON；离线推理不调用 Codex。作者负责理解实现并披露实验历史。保留给定的[数据许可证](code/data/LICENSE.txt)和[来源声明](code/data/NOTICE.txt)。
