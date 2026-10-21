# MP2 作业实现

资料：[算法路线](../../class%20pdf/MP2最终算法路线.md) · [课程知识点](../../class%20pdf/MP2算法知识点总结与方案灵感.md) · [作业要求](../../guide/GUIDE.md) · [官方运行说明](../README_cn.md)。

当前最终方案（恢复原 validation 选择，2026-10-21）：**L0＋M1 等权概率融合**，validation **740/798，92.73%**，冻结 test **377/467，80.72805139186295%**。入口为 `group_ranker/submission_ensemble.py`，完整权重在 `runs/ensemble_v1/checkpoint/`。最终参数、资源与复现指南见[根目录中文 README](../../README_cn.md)，研究分析见[英文报告](../../report/report.pdf)。源码、报告与实验产物一并纳入版本控制，参考机测时待完成。

**选型与历史：** 恢复原记录中按 validation 保留的 `ensemble_v1`；此前依据更高 test 临时指定注意力方案的决定已撤回。原融合开发时单模型 test 已见，仍按事后扩展披露，不称新的独立盲测；不改写历史 JSON、冻结源码或原权重。根目录复现命令使用原源码快照，区别于后续重构后的当前源码。

注意力实验作为探索性对照保留：单层 2 头、64 维编码器及等参数门控；L0＋注意力为 validation **737/798**、test **381/467**。源码在 `attention_ranker/`，产物在 `runs/attention_v1/`，较高的观测 test 不作为最终选型依据。

树对照：树模型延长训练预算已完成，复用 `tree_ranker/` 和统一的 `solution/.venv`。深度 5、1000 棵树在 validation 为 **717/798，89.85%**，旧 300 棵为 **718/798，89.97%**；随后冻结 test 为 **358/467，76.66%**，旧树为 **352/467，75.37%**。validation 准确率未改善，原选型继续保持融合，test 仅作报告。产物位于 `runs/tree_v2/`。

温度对照：两个温度仅在 train OOF 拟合；validation 为 **740/798，92.73%**，随后固定参数的官方 test 为 **378/467，80.94%**，比原融合多对 1 题。NLL 略降、ECE 升高；test 只作报告，不用于调参或改变原 validation 选型。产物位于 `runs/calibrated_ensemble_v1/`。

线性对照 L1：仅用 train 拟合、validation 开发，validation 为 **731/798，91.60%**；随后固定权重的官方 test 为 **378/467，80.94%**。该 test 只作评分，不用于调参或更改既有选型。方法代码位于 `linear_text/`，产物统一存放在 `runs/linear_text_v1/`。

## 结构

```text
code/
├── common.py、predict.py、score_predictions.py、data/ …  # 课程原始文件
└── solution/                  # 作业实现，按功能组织
    ├── baselines/             # 原基线、两项原消融、B1 平票与 B2 组中心性
    ├── group_ranker/          # 共享特征/训练工具，线性、MLP、等权融合与检查
    ├── linear_text/           # L1 薄训练/预测入口与检查，复用 group_ranker
    ├── calibrated_ensemble/   # train OOF 温度拟合、薄预测入口与检查
    ├── tree_ranker/           # 同 M1 特征的浅树排序，训练 CatBoost、推理 NumPy
    ├── attention_ranker/      # 等参数门控/单层注意力、L0 融合及独立实验编排
    ├── evaluate.py            # 调用官方工具，统一验证与记录
    ├── requirements-train.txt
    └── runs/<实验名>/         # checkpoint/、训练记录、validation/、冻结后的 test/
```

源码在 `solution/` 中复用和扩展；每轮产物集中到对应实验目录，已有记录不覆盖。`runs/` 中的权重、OOF、训练／评测记录与冻结源码随仓库提交；虚拟环境、缓存与报告临时文件继续忽略。历史 JSON 保留生成时的命令和路径。

官方 `common.py` 负责数据与答案解析，`predict.py`、`score_predictions.py` 负责独立预测与评分；本目录只实现算法、训练和实验编排。

本文件只维护运行入口和简短结果；算法推导见路线文档，详细 CV、分层指标、耗时与哈希以实验目录内 JSON 为准。

特征版本在 `group_ranker/features.py` 的 `FEATURE_VERSIONS` 中登记；配置记录启用的标量范围，旧 checkpoint 继续使用原基础特征。

## 运行

以下命令从 **`code/`** 执行，Python 3.12。通过 `python -m solution.…` 运行，自然导入同目录的官方 `common.py`，无需手工修改 `sys.path`。

统一使用现有 **`code/solution/.venv`**；以下相对路径从 `code/` 执行，按需补齐依赖：

```bash
unset PYTHONPATH
solution/.venv/bin/python -m pip install -r solution/requirements-train.txt
```

训练、评测和检查均使用这一环境。先清除当前项目终端继承的 `PYTHONPATH`，避免本机 ROS 路径混入虚拟环境；不修改系统配置。历史实验 JSON 保留当时的解释器路径与依赖版本；精确复现历史冻结结果时以对应记录和源码快照为准。

线性训练与六方法验证（包含 B1/B2）；`linear_v1` 已有归档，新实验使用新名称：

```bash
run_dir=solution/runs/linear_v2
solution/.venv/bin/python -m solution.group_ranker.train_linear --output "$run_dir"
solution/.venv/bin/python -m solution.evaluate --checkpoint "$run_dir/checkpoint"
```

线性训练按题五折 CV、种子 7506，仅比较预设的三个 L2 系数。MLP 训练及七方法比较：

```bash
run_dir=solution/runs/mlp_v2
solution/.venv/bin/python -m solution.group_ranker.train_mlp --output "$run_dir"
solution/.venv/bin/python -m solution.evaluate \
  --checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint "$run_dir/checkpoint"
```

`mlp_v1` 已有本轮记录，命令使用新名称以免覆盖。训练环境沿用 `requirements-train.txt`；本次实际用相同版本的系统 `python3` 训练。MLP 的前向、反传和 AdamW 均用 NumPy 实现，数据准备、按题划分、标准化和导出共用 `group_ranker/training.py`。

P2 分别启用文本、文本加算术特征，复用 M0 的已选配置；每版只做五折 OOF 和一次全量重训，轮数仍取折内最佳轮数中位数。运行时使用新的实验目录：

```bash
base=solution/runs/mlp_v1/checkpoint
text_run=solution/runs/mlp_text_v2
arithmetic_run=solution/runs/mlp_arithmetic_v2
solution/.venv/bin/python -m solution.group_ranker.train_mlp \
  --feature-set text --reuse-config "$base/config.json" --output "$text_run"
solution/.venv/bin/python -m solution.group_ranker.train_mlp \
  --feature-set arithmetic --reuse-config "$base/config.json" --output "$arithmetic_run"
solution/.venv/bin/python -m solution.evaluate \
  --checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint "$base" "$text_run/checkpoint" "$arithmetic_run/checkpoint"
```

本轮 P2 目录为 `mlp_text_v1`、`mlp_arithmetic_v1`。同一 `submission_mlp.py` 按配置提取特征，`evidence.py` 实现文本与有界算术检查，不复制阶段源码。`mlp_v1/source.tar` 保存增加 P2 特征前的 M0 源码。

P3 复用 M1 配置，四项特征消融各自重训，另比较 pairwise loss；不同时改变两项设计。以下使用新目录 `p3_repeat`，本轮记录在 `runs/p3/`：

```bash
base=solution/runs/mlp_text_v1/checkpoint/config.json
out=solution/runs/p3_repeat
for block in no_votes no_embedding no_context no_style; do
  solution/.venv/bin/python -m solution.group_ranker.train_mlp \
    --feature-set text --reuse-config "$base" --ablation "$block" --output "$out/$block" || break
done
solution/.venv/bin/python -m solution.group_ranker.train_mlp \
  --feature-set text --reuse-config "$base" --loss pairwise --output "$out/pairwise"
solution/.venv/bin/python -m solution.group_ranker.train_mlp \
  --feature-set text --reuse-config "$base" --stability-only --output "$out/stability"
solution/.venv/bin/python -m solution.evaluate \
  --checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint solution/runs/mlp_v1/checkpoint \
    solution/runs/mlp_text_v1/checkpoint solution/runs/mlp_arithmetic_v1/checkpoint \
    "$out/no_votes/checkpoint" "$out/no_embedding/checkpoint" \
    "$out/no_context/checkpoint" "$out/no_style/checkpoint" "$out/pairwise/checkpoint" \
  --output "$out/validation"
```

`--stability-only` 只输出固定划分下 7507/7508 的 OOF 记录，不导出其他种子的候选模型。所有版本使用相同的折内 CE NLL 早停规则；pairwise 只改变训练损失。选型预先固定为完整 validation 准确率最高，同分依次比较更少启用维度、更少推理资产、更短完整耗时。`mlp_arithmetic_v1/source.tar` 保存 P3 修改前的 P2 源码。

固定融合复用上述线性、M1 与稳定性 OOF 记录，打包两个成员并分析等权平均，不进行训练、温度拟合或权重搜索。新实验使用新目录：

```bash
run_dir=solution/runs/ensemble_v2
solution/.venv/bin/python -m solution.group_ranker.submission_ensemble \
  --linear-checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint solution/runs/mlp_text_v1/checkpoint \
  --stability-dir solution/runs/p3/stability --output "$run_dir"
solution/.venv/bin/python -m solution.evaluate \
  --checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint solution/runs/mlp_text_v1/checkpoint \
  --ensemble-checkpoint "$run_dir/checkpoint"
```

本轮目录为 `ensemble_v1`。打包阶段读取 train OOF 作分析，预测入口只加载自包含 checkpoint 内的两个模型，不读取 OOF、参考答案或原始 checkpoint 路径。

L1 线性 + 文本特征：启用已有 16–31 维文本统计与题目关系，不启用算术块；沿用三个 L2、五折和种子。新目录只保留入口与针对性检查，优化器、特征、标准化和推理逻辑共用原工具：

```bash
run_dir=solution/runs/linear_text_v2
solution/.venv/bin/python -m solution.linear_text.train \
  --l2 0.0001 0.001 0.01 --folds 5 --seed 7506 --maxiter 300 --output "$run_dir"
solution/.venv/bin/python -m solution.evaluate \
  --checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint solution/runs/mlp_text_v1/checkpoint \
  --ensemble-checkpoint solution/runs/ensemble_v1/checkpoint \
  --linear-text-checkpoint "$run_dir/checkpoint"
```

温度校准：复用既有 L0 与 M1，各自在 train OOF 上最小化逐题 NLL，固定等权融合，不重训成员或搜索权重。只需原 NumPy 推理环境，不新增依赖；新实验使用新目录：

```bash
run_dir=solution/runs/calibrated_ensemble_v2
solution/.venv/bin/python -m solution.calibrated_ensemble.calibrate \
  --linear-checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint solution/runs/mlp_text_v1/checkpoint --output "$run_dir"
solution/.venv/bin/python -m solution.evaluate \
  --checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint solution/runs/mlp_text_v1/checkpoint \
  --ensemble-checkpoint solution/runs/ensemble_v1/checkpoint \
  --linear-text-checkpoint solution/runs/linear_text_v1/checkpoint \
  --calibrated-checkpoint "$run_dir/checkpoint"
```

小型树排序：同 M1 原始特征及折内标准化/clip8，只换成 `CatBoostRanker(QuerySoftMax)`。原配置比较深度 3/5、最多 300 棵；延长预算对照固定深度 5，仅将上限提高到 1000 棵。折内 NLL 早停仍为 30 轮，全量重训的树数取折内最佳树数中位数。本轮记录在 `runs/tree_v2/`，复现使用新目录：

```bash
run_dir=solution/runs/tree_v2_repeat
solution/.venv/bin/python -m solution.tree_ranker.train \
  --depths 5 --max-trees 1000 --output "$run_dir"
solution/.venv/bin/python -m solution.evaluate \
  --checkpoint solution/runs/linear_v1/checkpoint \
  --mlp-checkpoint solution/runs/mlp_text_v1/checkpoint \
  --ensemble-checkpoint solution/runs/ensemble_v1/checkpoint \
  --linear-text-checkpoint solution/runs/linear_text_v1/checkpoint \
  --calibrated-checkpoint solution/runs/calibrated_ensemble_v1/checkpoint \
  --tree-checkpoint "$run_dir/checkpoint"
```

省略 `--depths` 和 `--max-trees` 仍使用原深度 3/5、300 棵预算。树训练会将原生模型导出为数值分裂、叶子、缩放与偏置数组，并核对 CatBoost/NumPy 前向。正式入口 `solution/tree_ranker/submission.py` 只依赖 NumPy、标准库及共用特征；`native.cbm` 留在训练记录目录作复核，不由推理加载。checkpoint 只含 `model.npz`、配置和资产清单。算法与导出依据见 [QuerySoftMax](https://catboost.ai/docs/en/concepts/loss-functions-ranking?lang=en) 和 [官方 JSON 导出说明](https://catboost.ai/docs/en/features/export-model-to-json)。

评测默认依次优先写到指定的树、校准融合、文本线性、原融合、最后一个 MLP 或线性实验的 `validation/`，也可用 `--output` 指定新目录。当前线性与树训练将数值库设为单线程，避免线程池叠加；MLP 训练设为 4 线程。评测编排进程占 1 线程，顺序启动的官方子进程最多 3 线程，合计最多 4 线程。Linux 下记录实测线程峰值。官方开发评测固定使用 validation；`profile_replay` 是额外无标签重放的特征/标准化加前向计时，完整资源计时以新进程的 `total_wall_seconds` 为准。

检查：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
solution/.venv/bin/python -m unittest solution.group_ranker.test_group_ranker solution.group_ranker.test_ensemble solution.linear_text.test_linear_text solution.calibrated_ensemble.test_calibration solution.tree_ranker.test_tree_ranker -v
```

检查覆盖线性、MLP、特征消融与算术边界；历史纯 NumPy 检查只跳过需要 SciPy 的线性梯度检查，MLP 的 CE/pairwise 梯度差分仍执行。训练本身采用 NumPy，因此前向一致性采用独立的逐组标量计算核对批量前向。候选置换比较映射后的解析答案；“同组代表替换”只要求官方最终数值评分不变，不要求改写候选正文后特征不变。

最终推理入口为 `solution/group_ranker/submission_ensemble.py`；线性、MLP 和探索性注意力入口各自保留，均实现 `predict(bank, arrays, checkpoint)`，为每题返回一个 0–3 的 Python 整数。作业允许自定义文件名，通过 `--method` 传入；官方 `submission.py` 保留原始占位接口。

L1 的官方方法文件为 `solution/linear_text/submission.py`。其 checkpoint 显式要求文本 schema，基础线性入口会拒绝该权重，避免特征与模型错配。

校准融合的官方方法文件为 `solution/calibrated_ensemble/submission.py`，与原融合共用特征、两个成员及前向；独立 schema 防止把校准 checkpoint 当作温度 1 使用。推理仅加载自包含 checkpoint，不读取 OOF 或参考答案。

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
| B1：最多票组平票时按中心性选择 | 677/798，84.84% | 234/355，65.92% |
| B2：答案组中心性 | 610/798，76.44% | 167/355，47.04% |
| 线性答案组排序 | 730/798，91.48% | 287/355，80.85% |
| L1：线性 + 文本特征 | 731/798，91.60% | 288/355，81.13% |
| M0：相同基础特征的小 MLP | 733/798，91.85% | 290/355，81.69% |
| M1：M0 增加文本/题目关系 | 738/798，92.48% | 295/355，83.10% |
| M2：M1 增加局部算术 | 733/798，91.85% | 290/355，81.69% |
| 固定线性 + M1 等权概率融合 | **740/798，92.73%** | **297/355，83.66%** |
| 同成员、train OOF 温度校准后等权融合 | **740/798，92.73%** | **297/355，83.66%** |
| 同 M1 特征的小型树排序 | 718/798，89.97% | 275/355，77.46% |

线性模型在 validation 净增 42 题。修正评测总线程配置后，四种方法的逐题选择均与原记录一致。冻结前检查：训练环境 7 项通过；纯 NumPy 推理环境 6 项通过、1 项训练梯度检查因无 SciPy 跳过。

### B1/B2 与 M0：2026-10-18

- **规则**：候选中心性为与其他有效候选的平均余弦，排除自身；组分数为组内候选中心性的均值。B1 仅比较并列最多票组，B2 比较全部有效答案组。两者均复用官方答案解析，不训练。
- **相同特征**：M0 沿用 2,088 维基础特征与标量 0–15，16–39 仍为零。四配置为隐层 `(64,16)/(32,16)` × weight decay `0.01/0.1`。AdamW 学习率 `0.001`、dropout `0.1`、batch 为 64 道题、梯度范数上限 1；最多 150 轮，折内 NLL 连续 15 轮无改善则早停。
- **选型**：仅按 train 五折 OOF 选择 `64→16`、weight decay `0.01`；最佳轮数 `[5,4,5,4,4]` 的中位数为 4，随后在全部 2,114 道训练冲突题上重训。折划分种子固定 7506，最终模型种子固定 7506；7507/7508 仅做 OOF 稳定性复核，不挑种子或集成。
- **结果解释**：线性 OOF 为 81.13%，MLP OOF 为 79.09%，复核为 78.43%/78.71%。MLP 在 validation 相比线性纠正 13 题、改错 10 题，净增 3 题；完整 validation 差值的 paired bootstrap 95% 区间为 **[-0.88, 1.63] 个百分点**，当前证据不足以证明稳定优势。B1/B2 均未改善原投票基线。特征与关键消融结果见下文 P2/P3。
- **资源**：CPU 完成 20 次选型折训练、10 次复核折训练及 1 次最终重训，共约 **307.73 秒**。MLP 未压缩推理资产 **560,785 字节（约 0.535 MiB）**；本机完整 validation 预测加独立评分约 **0.265 秒**。七种方法的预测、评分阶段实测总线程峰值均为 **4**。这些是本机 validation 资源记录，参考机资格另行确认。

完整记录：[训练与搜索](runs/mlp_v1/training.json) · [七方法比较、分层、耗时与线程](runs/mlp_v1/validation/summary.json) · [MLP 官方评分](runs/mlp_v1/validation/mlp-score.json) · [checkpoint](runs/mlp_v1/checkpoint/)。本轮未评测 test。

### P2：文本与算术，2026-10-19

M1 启用标量 16–31：长度、重复、问题与正文的词/数字关系，以及答案与题目数值的关系。M2 在 M1 上启用 32–39：已检查等式的数量、通过率、矛盾率和覆盖标志。算术使用受限 AST 与 Fraction，仅检查明确边界中的纯数字等式；不生成答案，不执行候选代码，也不因局部计算错误直接否决答案组。精确定义及限制见算法路线 §4。

两版均复用 M0 的 `64→16`、weight decay `0.01` 及其余训练参数，不另搜结构、正则或种子；固定原五折划分，各折独立标准化与早停。M1 最佳轮数 `[5,5,5,3,6]`，最终重训 5 轮；M2 为 `[4,3,5,4,3]`，最终重训 4 轮。

| 方法 | train OOF（冲突题） | 完整 validation | 本机完整预测 + 评分 | 推理资产 |
|---|---:|---:|---:|---:|
| 线性 L0 | 1715/2114，81.13% | 730/798，91.48% | 0.263 秒 | 28,771 字节 |
| 基础 M0 | 1672/2114，79.09% | 733/798，91.85% | 0.265 秒 | 560,785 字节 |
| 文本 M1 | 1692/2114，80.04% | **738/798，92.48%** | 0.382 秒 | 560,937 字节 |
| 算术 M2 | 1674/2114，79.19% | 733/798，91.85% | 0.440 秒 | 561,013 字节 |

- **文本有初步收益**：M1 相比 M0 在 validation 纠正 9 题、改错 4 题，净增 5 题；相比线性净增 8 题。但相对 M0 的完整 validation 差值 95% 区间为 `[-0.25, 1.50]` 个百分点，仍跨 0；目前不能声称稳定优势。
- **算术未带来增益**：M2 相比 M1 纠正 6 题、改错 11 题，净减 5 题，OOF 也下降。保持记录，当前候选关闭算术块。这里比较的是相同训练协议下的重训版本，轮数由各自 OOF 早停确定，并非固定权重后直接增减特征。
- **覆盖有限**：train 成功检查覆盖 `1037/2114` 道冲突题、`1950/8082` 个有效候选；validation 为 `155/355` 题（43.66%）、`296/1344` 个有效候选（22.02%）。validation 共检查 420 条等式，其中 402 条通过、18 条矛盾；另有 6,405 个未能检查的数学/美元符号片段，未知不计为错误。局部算式通过也不能证明题意理解正确。
- **成本**：M1/M2 各 5 次折训练加 1 次全量重训，分别约 69.60/66.01 秒。validation 单线程分析重放中，特征提取约 0.158/0.216 秒，标准化加前向约 0.004/0.004 秒；M2 其中算术检查约 0.045 秒。九种方法新进程预测及评分阶段实测总线程峰值均为 4。本机耗时不能代替参考机认证。

P2 按预设 validation 标准选 M1 进入 P3，线性和 M0 保留为对照；后续结果见下节。P2 未运行 test，也未根据旧 test 结果修改特征或参数。

记录：[M1 训练](runs/mlp_text_v1/training.json) · [M2 训练](runs/mlp_arithmetic_v1/training.json) · [九方法 validation、分层、比较与资源](runs/mlp_arithmetic_v1/validation/summary.json) · [M1 checkpoint](runs/mlp_text_v1/checkpoint/) · [M2 checkpoint](runs/mlp_arithmetic_v1/checkpoint/)。

### P3：有限选型与检查，2026-10-19

本轮预先固定五项消融，复用 M1 的网络与优化参数，每项五折 OOF 后全量重训；另对原 M1 做两个种子的 OOF 复核。所有模型保持 2,088 维输入 schema，以掩码关闭特征，消融版本仍存储相同形状的权重。标量索引与向量区间均为左闭右开。

| 方案 | 移除或改变的内容 | train OOF | 完整 validation | 冲突题 |
|---|---|---:|---:|---:|
| 原 M1 | 文本 + CE，对照 | 80.04% | **738/798，92.48%** | **295/355，83.10%** |
| no_votes | 标量 `[0,8)` | 79.85% | 736/798，92.23% | 293/355，82.54% |
| no_embedding | 两个向量块及几何标量 `[8,16)` | 73.56% | 709/798，88.85% | 266/355，74.93% |
| no_context | 仅交互向量 `[1024,2048)` | 79.38% | 732/798，91.73% | 289/355，81.41% |
| no_style | 长度/数字密度/重复标量 `[16,24)` | 79.19% | 735/798，92.11% | 292/355，82.25% |
| pairwise | 同特征，CE 换成按题归一化的成对损失 | 78.29% | 736/798，92.23% | 293/355，82.54% |

- **表示信息贡献最明显**：去 embedding 版本相比 M1 纠正 12 题、改错 41 题，净减 29 题；完整 validation 差值 95% 区间为 `[-5.39,-1.88]` 个百分点。该消融同时移除了依赖 embedding 的几何标量，不能单独归因到某一个向量块。
- **其余差异较小**：去票数、交互、风格、换损失分别净减 2、6、3、2 题，差值区间均跨 0。去显式票数后仍有组 pooling、范数等间接重复信息；去交互向量仍保留其他几何标量。因此这些消融不代表彻底消除了所有票数/题内关系。风格块的作用也不能证明模型掌握了完整推理验证。
- **稳定性**：固定划分 7506，M1 在训练种子 7506/7507/7508 下 OOF 为 `80.04% / 79.04% / 79.75%`；同种子的 M0 为 `79.09% / 78.43% / 78.71%`。最终种子仍为 7506，复核不用于挑种子或集成。线性 OOF 81.13% 仍更高；选型遵循预设的完整 validation 标准。
- **有限选型结果**：14 种方法比较后，继续选原 M1，checkpoint 未改动。它比线性在 validation 净增 8 题，但差值区间 `[-0.25,2.38]` 个百分点仍跨 0。所有 bootstrap 区间均未校正多次比较，不能将本次选型写成确定的泛化优势。
- **置信度**：M1 冲突题 NLL 为 0.4781；最高组概率在 `[0.85,0.95)` 的 67 题中只答对 54 题（80.60%）。softmax 是未校准的相对分数，不直接等于正确概率；本轮未拟合温度或阈值。
- **资源与检查**：共 40 次训练，累计约 **499.65 秒**；训练环境 17 项全通过，纯 NumPy 环境 16 项通过、1 项线性 SciPy 检查跳过。14 种方法实测总线程峰值均为 4，历史线性/M0/M1/M2 逐题选择不变，OOF 标签与 fold 一致，官方代码哈希不变。M1 本机完整 validation 预测加独立评分约 **0.385 秒**，推理资产 **560,937 字节**；参考机资格待确认。

P3 阶段未运行 test；后续 M1 冻结 test 结果见下节。参考机测时、报告、复用披露和公开交付待完成；旧线性冻结记录保留。

记录：[14 方法与选型](runs/p3/validation/summary.json) · [稳定性复核](runs/p3/stability/training.json) · [检查结果和命令](runs/p3/checks.json)。各消融的权重、训练过程和 OOF 分数位于 `runs/p3/<方案>/`，validation JSON 包含各票型、组数、置信度分桶与纠错/改错案例。

### L1：线性 + 文本特征，2026-10-19

仅用 train 拟合，按原票型分层五折、种子 7506 比较 L2=`0.0001/0.001/0.01`。每折标准化只拟合该折训练部分；按 OOF 准确率、NLL、较强正则依次选择，最终选 0.01，再用全部 2,114 道训练冲突题重训。

- **结果**：OOF `1716/2114，81.17%`，NLL 0.4662；基础线性为 `1715/2114，81.13%`，NLL 0.4767。validation `731/798，91.60%`，冲突题 `288/355，81.13%`；相对基础线性纠正 6 题、改错 5 题，净增 1 题，差值 95% 区间约 `[-0.63,1.00]` 个百分点。当前证据不足以认为文本块稳定改善线性选择准确率。
- **成本**：15 次折训练加 1 次全量重训，全部收敛，累计约 40.56 秒；未压缩推理资产 28,923 字节。本机完整 validation 新进程预测加独立评分约 0.382 秒，9 方法实测总线程峰值均为 4；参考机资格待确认。
- **实现与检查**：`linear_text/train.py`、`submission.py` 为薄入口，`test_linear_text.py` 检查新增特征确实生效、schema/资产校验、全部候选排列、跨分块、空输入/无效答案/零向量。训练环境 24 项全通过，纯 NumPy 环境 23 项通过、1 项 SciPy 检查跳过；基础线性、M1、融合及原基线的 validation 逐题选择不变。
- **数据范围**：训练与开发阶段只读取 train 监督和 validation 评分参考，未按已有 test 错题增加规则。随后应用户要求，对已固定的 L1 单独做官方 test 评分；validation 仍由既有融合最高，不根据 test 改变选型。训练、validation 与检查 JSON 保留测试前的历史记录。
- **冻结 test**：**378/467，80.94218415417559%**。相比原线性纠正 9 题、改错 6 题，净增 3 题（+0.64 个百分点），paired bootstrap 95% 区间 `[-1.07,2.36]`；相比融合纠正 13 题、改错 12 题，净增 1 题（+0.21 个百分点），区间 `[-1.93,2.36]`。这些小幅观测差异不能证明稳定优势。两个、三个、四个答案组分别为 `267/303、92/126、19/38`，四组互异题仍较困难。
- **测试资源与身份**：预测前保存源码快照与权重哈希，评分前锁定预测文件；冻结 ID 前缀 `5e70afdc2bd9`，评分后冻结文件哈希一致。本机完整 test 新进程预测加独立评分约 **0.397 秒**，实测总线程峰值 **4**，推理资产 **28,923 字节**。参考机资格待确认；既有 test 已接触，本次评分不称新盲测，不据此修改模型。

记录：[训练、OOF、选型与代码哈希](runs/linear_text_v1/training.json) · [validation、分层与资源](runs/linear_text_v1/validation/summary.json) · [检查与数据边界核对](runs/linear_text_v1/checks.json) · [checkpoint](runs/linear_text_v1/checkpoint/)。

测试记录：[官方 test 成绩](runs/linear_text_v1/test/score.json) · [比较、分层、命令与资源](runs/linear_text_v1/test/summary.json) · [冻结清单](runs/linear_text_v1/test/freeze.json) · [评分前预测哈希](runs/linear_text_v1/test/prediction_freeze.json) · [冻结源码](runs/linear_text_v1/test/source.tar)。

### 小型树排序，2026-10-19

原实验 `tree_v1` 复用 M1 的完整文本特征、按题划分及标准化/clip8，输入仍为 2,088 维 schema（启用向量块与标量 0–31）。最终全量 train 的 mean/scale/active 与 M1 完全一致，只替换打分器。每个答案组一行，正确组标签 1、其余 0；CatBoost 的 query 是一道题，临时序号仅用于组织 query，不进入特征。

固定 `QuerySoftMax`、学习率 0.05、叶子 L2=3、数值分桶 64、CPU 单线程；不使用随机采样或额外随机分裂扰动。仅比较深度 3/5，最多 300 棵，按原票型分层五折、种子 7506。早停仅使用各折留出部分的 NLL；按 OOF 准确率、NLL、较浅深度依次选择，再取最佳树数中位数做全量 train 重训。

| 深度 | train OOF | NLL | 五折最佳树数 |
|---|---:|---:|---|
| 3 | 1551/2114，73.37% | 0.6165 | 300/300/300/300/300 |
| **5，选中** | **1567/2114，74.12%** | **0.6074** | 300/300/300/300/300 |

- **validation**：完整 `718/798，89.97%`，冲突题 `275/355，77.46%`，冲突题 NLL 0.5714。相对原投票基线净增 30 题，但相对线性纠正 21 题、改错 33 题，净减 12；相对 M1 纠正 14 题、改错 34 题，净减 20，差值 95% 区间约 `[-4.26,-0.75]` 个百分点，未校正多次比较。此次配置不替换现有融合。
- **结果解释**：这组浅树与预算下的 OOF、validation 均低于现有监督模型。可能原因是高维向量的信息依赖多个维度联合，而浅树通过逐维阈值划分学习；这些原因尚未通过单独消融验证。10 折的最佳树数均触及 300 上限，结论也受本轮固定预算限制，不追加树数、深度或种子搜索。
- **实现与检查**：新目录只包含训练、数值树前向、预测入口及针对性检查；共用原特征、标准化、OOF 与导出工具。纯 NumPy 阶段 34 项中 32 项通过、2 项依赖检查跳过；依赖安装后，同一个 `solution/.venv` 中 **34 项全部通过**。包含标量参考、阈值相等时走左侧、JSON 特征映射、原生前向、候选排列、跨分块、边界和资产/schema 校验。禁用 CatBoost/SciPy 导入的预测仍与官方结果一致；实际 validation 的原生与 NumPy 分数最大误差约 `2.44e-15`，训练复核最大误差约 `3.11e-15`。
- **成本与范围**：共 **11 次训练，约 123.28 秒**；推理资产 **117,722 字节（约 114.96 KiB）**，本机完整 validation 新进程预测加独立评分约 **0.406 秒**。11 方法实测总线程峰值均为 **4**，10 个历史对照在统一环境中的逐题选择、旧权重和官方文件哈希不变。训练与开发阶段仅用 train 拟合、validation 比较；随后应用户要求对固定模型做官方 test 评分。既有 test 已接触，本次不称新盲测，不据 test 修改树数、深度、特征、种子、融合或选型；训练、validation 和检查 JSON 保留测试前记录。参考机资格待确认。
- **冻结 test**：**352/467，75.37473233404711%**，答错 115 题。相对原线性纠正 27 题、改错 50 题，净减 23 题（-4.93 个百分点），paired bootstrap 95% 区间约 `[-8.57,-1.50]`；相对原融合纠正 19 题、改错 44 题，净减 25 题（-5.35 个百分点），区间约 `[-8.57,-2.14]`。区间未校正多次比较，结论限定此次固定配置。两个、三个、四个答案组分别为 `249/303、85/126、18/38`；未校准 softmax 的 NLL 0.5797、Brier 0.3422、ECE 7.71%。
- **测试冻结与资源**：预测前保存源码快照、配置和权重哈希，评分前锁定预测文件；冻结 ID 前缀 `f92188e14813`，评分后冻结文件哈希一致。使用统一 `solution/.venv`，清除继承的 `PYTHONPATH`；本机完整 test 新进程预测加独立评分约 **0.432 秒**，实测总线程峰值 **4**，推理资产 **117,722 字节**。参考机资格待确认；描述性概率分析为评分后的单独重放，不混入完整计时。

记录：[预设参数、逐折训练、依赖版本与哈希](runs/tree_v1/training.json) · [11 方法 validation、分层、比较与资源](runs/tree_v1/validation/summary.json) · [检查](runs/tree_v1/checks.json) · [NumPy checkpoint](runs/tree_v1/checkpoint/)。

测试记录：[官方 test 成绩](runs/tree_v1/test/score.json) · [分层、比较、概率指标、命令与资源](runs/tree_v1/test/summary.json) · [冻结清单](runs/tree_v1/test/freeze.json) · [评分前预测哈希](runs/tree_v1/test/prediction_freeze.json) · [冻结源码](runs/tree_v1/test/source.tar)。

**延长预算对照 `tree_v2`**：根据旧 train OOF 最佳树数触及 300 上限，预设仅将深度 5 的训练上限提高到 1000；特征、学习率 0.05、L2=3、五折、种子 7506 和早停规则均保持不变。

- **训练与 OOF**：五折最佳树数为 `999/999/1000/1000/1000`，中位数 1000 用于全量 train 重训。共 **6 次拟合、297.90 秒**；OOF 从 `1567/2114（74.12%）` 提升到 **`1589/2114（75.17%）`**，NLL 从 `0.6074` 降到 `0.5682`。树数仍接近本轮上限，不能由此认定已经收敛。
- **validation**：从 `718/798（89.97%）` 变为 **`717/798（89.85%）`**；冲突题 `274/355（77.18%）`。相对旧树纠正 5 题、改错 6 题，净减 1；准确率差值 95% 区间约 `[-0.88,0.75]` 个百分点。冲突题 NLL 从 `0.5714` 降到 `0.5270`，概率损失改善未转化为选择准确率提升；继续保留 validation 为 `740/798` 的原等权融合。
- **资源与检查**：推理资产 **341,723 字节（约 333.71 KiB）**，本机完整 validation 新进程预测加独立评分约 **0.416 秒**，实测总线程峰值 **4**；参考机资格待确认。**35 项检查全部通过**，原生与 NumPy 的 validation 分数最大误差约 `6.66e-15`；新模型前 300 棵的原生分数与旧模型完全一致。10 个历史对照的 validation 逐题选择不变，290 个已登记官方文件及历史产物的哈希不变；禁用 CatBoost/SciPy 后推理结果仍一致。开发阶段仅用 train 拟合和 OOF、validation 比较；训练、validation 与检查 JSON 保留测试前记录。
- **冻结 test**：随后应用户要求，仅评分固定的 1000 棵树模型，官方结果为 **358/467，76.65952890792292%**，答错 109 题。相对旧树纠正 13 题、改错 7 题，净增 **6 题（+1.28 个百分点）**；配对 bootstrap 95% 区间约 `[-0.43,3.21]`，包含 0，不能认定稳定提升。仍比基础线性少对 17 题、比原融合少对 19 题。两个、三个、四个答案组分别为 `255/303、83/126、20/38`；NLL 为 `0.5469`、Brier 为 `0.3249`、ECE 为 `4.48%`，概率指标仅作评分后描述。
- **测试身份与资源**：预测前冻结源码、配置与权重，评分前锁定预测哈希；冻结 ID 前缀 `bafc3f8c0dbe`，评分后哈希一致，历史产物未覆盖。本机完整 test 新进程预测加独立评分约 **0.457 秒**，总线程峰值 **4**，资产仍为 **341,723 字节**；参考 CPU 资格待确认。既有 test 已接触，本次不称新盲测，不据 test 修改树数、深度、特征、种子或选型，原 validation 选型保持融合。

记录：[预设配置与训练](runs/tree_v2/training.json) · [官方 validation 评分](runs/tree_v2/validation/tree-score.json) · [比较、选型与资源](runs/tree_v2/validation/summary.json) · [工程与历史兼容检查](runs/tree_v2/checks.json) · [checkpoint](runs/tree_v2/checkpoint/)。

测试记录：[官方 test 成绩](runs/tree_v2/test/score.json) · [逐题比较、分层、命令与资源](runs/tree_v2/test/summary.json) · [冻结清单](runs/tree_v2/test/freeze.json) · [评分前预测哈希](runs/tree_v2/test/prediction_freeze.json) · [冻结源码](runs/tree_v2/test/source.tar)。

### 温度校准概率融合，2026-10-19

固定使用原 L0 与 M1（种子 7506），公式为 `0.5 × softmax(s_linear / T_linear) + 0.5 × softmax(s_M1 / T_M1)`。每个成员各拟合一个正温度；用 `beta=1/T` 下 NLL 的凸性，在预设 `[0.05,20]` 边界二分导数 60 步。没有改变成员、特征、种子、融合权重或门控，也没有增加外部数据和依赖。

- **拟合结果**：`T_linear=1.2278854512`、`T_M1=1.3378831943`，均未触及边界。正温度不改变单成员数学 argmax；两个成员的概率尺度变化后，融合选择可以变化。拟合数据上，线性 NLL `0.4767→0.4693`、M1 NLL `0.5041→0.4892`。
- **统计边界**：成员 logits 来自 train OOF，但两个温度使用全部这些 OOF 标签拟合；融合在同一数据上的 `1715/2114、NLL 0.4624` 只是拟合数据诊断，不能称独立 OOF 验证。独立开发比较使用完整 validation。

以下概率指标只统计 355 道冲突题，避免 443 道单答案组题稀释结果；NLL、Brier 和 ECE 越低越好：

| 方法 | 完整 validation | 冲突题 NLL | Brier | ECE（10 等宽桶） |
|---|---:|---:|---:|---:|
| 原等权融合 | 740/798，92.73% | 0.4560 | 0.2613 | 3.38% |
| 温度校准后融合 | 740/798，92.73% | 0.4480 | 0.2604 | 4.29% |

- **结论**：相对原融合，纠正 1 题、改错 1 题，净增 0；准确率差值的 paired bootstrap 95% 区间 `[-0.38,0.38]` 个百分点。NLL/Brier 略好，ECE 反而变差，因此不能声称准确率或各项校准指标都改善。三个答案组的 NLL 降低，四组的 NLL 略升；分组明细保存在 validation JSON。按预设同分优先更少资产的规则，继续保留原融合，不再追加温度或权重搜索。
- **成本与检查**：仅拟合 2 个标量，不重训成员；拟合、打包与记录约 0.024 秒。未压缩推理资产 **590,736 字节**（比原融合多 97 字节），本机完整 validation 新进程预测加独立评分约 **0.392 秒**，10 方法实测总线程峰值均为 **4**。训练环境 29 项通过，纯 NumPy 环境 28 项通过、1 项 SciPy 梯度检查跳过；检查包含已知温度最优解、NLL/Brier/ECE 手算、独立 scaler、候选排列、跨分块、协议边界及资产校验。九个历史对照的 validation 逐题选择、原成员和官方文件哈希保持不变。
- **实验范围**：开发阶段只从既有 train OOF 拟合参数、用 validation 比较，不依据旧 test 错题构造规则。随后应用户要求对这份固定参数做官方 test 评分；已有 test 已接触，不称新的盲测，不据此调整温度、成员、权重或选型。准备、validation 和检查 JSON 保留测试前记录，旧冻结源码和记录不变。
- **冻结 test**：**378/467，80.94218415417559%**。相对原融合只改变 1 题：纠正 1 题、改错 0 题，净增 1 题（+0.21 个百分点）。差值的 paired bootstrap 95% 区间为 `[0.00,0.64]` 个百分点，包含 0，且只有 1 道差异题，不能据此认定稳定改善。相对原线性纠正 16 题、改错 13 题，净增 3；与 L1 同分，但分别纠正和改错 12 题，逐题选择并不相同。
- **test 概率指标与资源**：全部 467 题均为冲突题；原融合与校准融合的 NLL 为 `0.4568→0.4525`、Brier 为 `0.2668→0.2660`、ECE 为 `2.24%→3.20%`，仍是部分指标略好、ECE 更差。预测前冻结源码、温度与全部权重，评分前锁定预测哈希；冻结 ID 前缀 `26433ec90ced`，评分后哈希一致。本机完整 test 新进程预测加独立评分约 **0.415 秒**，实测总线程峰值 **4**，全部未压缩推理资产 **590,736 字节**；参考机资格待确认。概率分析为评分后的单独重放，不混入完整评测计时。

记录：[温度、拟合目标、来源哈希与成本](runs/calibrated_ensemble_v1/preparation.json) · [10 方法 validation、校准分组与选型](runs/calibrated_ensemble_v1/validation/summary.json) · [检查](runs/calibrated_ensemble_v1/checks.json) · [自包含 checkpoint](runs/calibrated_ensemble_v1/checkpoint/)。

测试记录：[官方 test 成绩](runs/calibrated_ensemble_v1/test/score.json) · [比较、概率指标、命令与资源](runs/calibrated_ensemble_v1/test/summary.json) · [冻结清单](runs/calibrated_ensemble_v1/test/freeze.json) · [评分前预测哈希](runs/calibrated_ensemble_v1/test/prediction_freeze.json) · [冻结源码](runs/calibrated_ensemble_v1/test/source.tar)。

### 固定等权融合扩展，2026-10-19

原路线 §9 预留的模型平均扩展：两个模型先在本题的相同答案组上各自做 softmax，再取 `0.5 × P_linear + 0.5 × P_M1`。特征提取共用一次；线性成员关闭新增文本维度，两个成员分别使用自己的冻结标准化参数。平均概率后取 log 接入共用排序和平票规则。最终只打包原 7506 线性与 M1 权重。

| 方法 | train OOF（2114 冲突题） | validation（798 题） | 冻结 test（467 题） |
|---|---:|---:|---:|
| 线性 | 1715，81.13% | 730，91.48% | 375，80.30% |
| M1 | 1692，80.04% | 738，92.48% | 372，79.66% |
| 固定等权融合 | **1717，81.22%** | **740，92.73%** | **377，80.73%** |

- **互补与稳定性**：三个 M1 种子分别救回 92/96/101 道线性 OOF 错题，每个 fold 都有救回题，53 题在三个种子下均能救回。相同等权公式在三种子 OOF 上为 `81.22% / 80.79% / 81.17%`，准确率优势并不稳定；NLL 为 `0.4637 / 0.4623 / 0.4583`，均低于线性的 0.4767。7507/7508 只作诊断，未挑种子或增加集成成员。
- **validation**：相对 M1 纠正 7 题、改错 5 题，净增 2 题；冲突题为 `297/355，83.66%`。该差值的完整 validation 95% 区间为 `[-0.63,1.13]` 个百分点，仍跨 0。
- **test**：相对线性纠正 16 题、改错 14 题，净增 2 题（+0.43 个百分点），区间 `[-1.93,2.78]`；相对 M1 纠正 12 题、改错 7 题，净增 5 题（+1.07 个百分点），区间 `[-0.64,3.00]`。只能报告小幅观测改善，尚不能声称稳定泛化优势。
- **资源与冻结**：无新增训练或搜索；自包含资产 **590,639 字节（约 0.563 MiB）**，计入两个成员及所有配置/清单。测试前冻结源码与资产，ID 前缀 `79062305cd4b`，测试后哈希保持一致。本机完整 test 新进程预测加独立评分约 **0.414 秒**，总线程峰值 **4**。参考 CPU 资格待确认。
- **检查与实验范围**：训练环境 21 项全通过；纯 NumPy 环境 20 项通过、1 项 SciPy 检查跳过。检查包含概率平均公式、独立 scaler 等价性、候选置换、分块、边界及完整资产计数。融合配方在本轮评测前固定，但此前已查看单模型 test；本轮属于事后扩展的冻结评分，不作为新的盲测，不据结果调整权重、温度或重选种子。

记录：[OOF 互补、准备成本与配方](runs/ensemble_v1/preparation.json) · [validation 比较](runs/ensemble_v1/validation/summary.json) · [官方 test 成绩](runs/ensemble_v1/test/score.json) · [test 分层与资源](runs/ensemble_v1/test/summary.json) · [检查](runs/ensemble_v1/checks.json) · [冻结清单](runs/ensemble_v1/test/freeze.json) · [冻结源码](runs/ensemble_v1/test/source.tar)。

### 答案组注意力与等参数门控，2026-10-21

将每个答案组当作一个 token，复用原 2088 维文本版特征，投影到 64 维，使用单层、2 头、`64→128→64` 前馈层、残差、LayerNorm 和 dropout 0.1，不加位置编码。门控对照使用相同形状的 Q/K/V/O 参数，只在组内计算门控，不交换各组 value；两者打分参数均为 **167233**，初始化、五折划分和最终种子同为 7506。两版均保留原手工上下文，因此该对照只检验额外学习到的组间交互，不能称完全独立的逐候选判断。

训练保持 AdamW 学习率 0.001、weight decay 0.01、batch 64、梯度上限 1、最多 150 轮、折内 NLL patience 15；只用 train。门控最佳轮数 `[2,1,2,1,2]`，注意力均为 2，最终都重训 2 轮。两者分别与原 L0 按 0.5/0.5 概率融合，温度固定 1；全部用现有 `solution/.venv` 的 NumPy/CPU 实现。

| 方法 | train OOF / 2114 | validation / 798 | test / 467 | 资产 MiB | 本机完整 test |
|---|---:|---:|---:|---:|---:|
| 等参数门控 | 1640，77.58% | 727，91.10% | 374，80.09% | 0.662 | 0.472 秒 |
| 单层注意力 | 1659，78.48% | 729，91.35% | 372，79.66% | 0.662 | 0.493 秒 |
| L0+门控 | 1702，80.51% | 738，92.48% | 379，81.16% | 0.690 | 0.486 秒 |
| L0+注意力 | 1704，80.61% | 737，92.36% | **381，81.58%** | 0.690 | 0.478 秒 |
| 原 L0+M1 | 1717，81.22% | **740，92.73%** | 377，80.73% | 0.563 | 历史 0.414 秒 |

- **有互补信号，注意力收益尚不稳定**：注意力单体相对门控 OOF 多对 19 题、validation 多对 2 题，test 却少对 2 题。注意力融合相对门控融合，validation 少对 1 题，test 纠正 10 题、改错 8 题，净增 2，差值 95% 区间 `[-1.28,2.14]` 个百分点。相对原融合纠正 10 题、改错 6 题，净增 4（+0.86 个百分点），区间 `[-0.86,2.57]`；均包含 0，且未校正多次比较，不能声称稳定泛化优势。
- **相同单体分数不代表相同错误**：注意力与原 M1 的 test 都是 372，但分别纠正/改错 13 题。新融合的高分可能来自与 L0 的错误互补，不能全部归因于注意力。两、三、四个答案组分别为 `268/303、91/126、22/38`；相比原融合的净增主要来自三组题，四组题仍无改善。
- **资源与检查**：共 12 次拟合，全流程训练及打包约 **35.27 秒**。单体资产 694073/694077 字节，融合 723682/723686 字节，均未压缩。Python 3.12.3 / NumPy 1.26.4、Core Ultra 9 275HX；四种评测的总线程峰值均为 **4**。27 项单元检查通过，包含新模型梯度差分（误差小于 1e-6）、组排列等变、padding、参数匹配、融合公式和资产校验；实际训练权重另通过 24 种候选排列、129 题跨块及临时损坏资产拒绝检查。参考 CPU 资格待确认。
- **选型与冻结**：在任一新 test 前，根据 validation 保留原 `ensemble_v1`，见 `validation_summary.json` 的 selection。四份源码/权重在预测前冻结，预测哈希在独立评分前锁定；419 个历史文件及全部冻结文件保持不变。此前 test 已见，本轮是事后扩展；当时最高观测 test 为 381，评分完成时未替换 validation 选型或追加搜索。作者后来曾依据 test 临时指定注意力，现已撤回并恢复原 validation 选择；当前方案见页首，历史记录保持原样。

复现（从 `code/` 执行，沿用现有环境；使用新的输出目录）：

```bash
unset PYTHONPATH
run=solution/runs/attention_repeat
if mkdir "$run"; then
  echo '{}' > "$run/historical_sha256.json"
  solution/.venv/bin/python -m solution.attention_ranker.train --output "$run" &&
  solution/.venv/bin/python -m solution.attention_ranker.evaluate --run "$run" --phase validation &&
  solution/.venv/bin/python -m solution.attention_ranker.evaluate --run "$run" --phase test
fi
```

`train` 顺序完成两版五折和重训，并打包两个融合；`evaluate` 复用官方预测、独立评分与统一线程监测。复现新目录写入空历史快照，本轮原目录保留完整的 `historical_sha256.json` 以审核旧文件；新目录仍会保存模型、源码和预测冻结记录。免训练复现任选一个原 checkpoint，输出到新目录：

```bash
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3
out=solution/runs/attention_test_reproduction
if mkdir "$out"; then
  time {
    solution/.venv/bin/python predict.py --method solution/attention_ranker/submission.py \
      --checkpoint solution/runs/attention_v1/attention_ensemble/checkpoint --split test \
      --output "$out/predictions.json" &&
    solution/.venv/bin/python score_predictions.py --split test \
      --predictions "$out/predictions.json" --output "$out/score.json"
  }
fi
```

记录：[总体结果](runs/attention_v1/summary.json) · [训练](runs/attention_v1/training_summary.json) · [validation 与测试前选型](runs/attention_v1/validation_summary.json) · [27 项检查](runs/attention_v1/checks.json) · [实际权重检查](runs/attention_v1/post_score_checks.json) · [注意力融合官方 test](runs/attention_v1/attention_ensemble/test/score.json)。四个版本各有 checkpoint、OOF、冻结清单和 test 记录，共用 [source.tar](runs/attention_v1/source.tar)。集合建模参考 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)，不加载其预训练权重。

### M1 容量对照，2026-10-21

保持文本特征、原五折划分、种子 7506、AdamW 学习率 0.001、weight decay 0.01、dropout 0.1、batch 64 和早停协议不变，分别将 M1 扩为 `128→32`、`256→64`；再与原 L0 做固定 0.5/0.5 概率融合。复用现有 `group_ranker`，训练入口增加 `--hidden H1 H2`，训练/载入共用有限宽度登记；融合打包可不提供额外种子复核目录。产物集中于 `runs/mlp_capacity_v1/`，没有新建环境、算法目录或复制推理实现。

| 方法 | train OOF / 2114 | validation / 798 | test / 467 | 推理资产 MiB | 本轮完整 test |
|---|---:|---:|---:|---:|---:|
| 原 M1 `64→16` | 1692，80.04% | 738，92.48% | 372，79.66% | 0.535 | 历史 0.405 秒 |
| M1 `128→32` | 1670，79.00% | 735，92.11% | 364，77.94% | 1.057 | 0.434 秒 |
| M1 `256→64` | 1677，79.33% | 733，91.85% | 370，79.23% | 2.124 | 0.436 秒 |
| 原 L0+M1 | 1717，81.22% | 740，92.73% | 377，80.73% | 0.563 | 历史 0.414 秒 |
| L0+M1 `128→32` | 1702，80.51% | 740，92.73% | 374，80.09% | 1.085 | 0.443 秒 |
| L0+M1 `256→64` | 1708，80.79% | 740，92.73% | 375，80.30% | 2.152 | 0.448 秒 |

- **本轮未观察到扩容收益**：两档单模型均低于原 M1；两档融合在 validation 持平，test 分别比原融合少对 3/2 题。相对原融合的 test 差值 95% 区间分别为 `[-2.14,0.86]`、`[-2.14,1.28]` 个百分点，均含 0；不能推广为所有大网络都更差。
- **更早达到折内最优**：128 档五折最佳轮数均为 3，256 档为 `[3,2,2,2,2]`，最终分别重训 3/2 轮。128 档第 0 折从第 3 到第 18 轮，训练 dropout loss `0.348→0.055`，折外 NLL 却 `0.537→0.907`；256 档相应为 `0.291→0.036`、`0.551→0.991`。这支持继续训练会过拟合的判断，早停已回退到最佳轮数；不能据此证明容量是准确率下降的唯一原因。
- **成本与检查**：新增各五折加一次重训，共 12 次拟合，当前统一环境训练累计约 23.18 秒。Python 3.12.3 / NumPy 1.26.4、Core Ultra 9 275HX；21 项原有单元检查通过，四个新 checkpoint 的全部 24 种候选排列、129 题跨分块、OOF 标签/划分及 validation 重放一致，旧 M1/融合 798 个 validation 选择保持一致。四个新版本完整评测均实测总线程峰值 4，资产均小于 16 MiB，参考 CPU 资格待确认。历史控制模型未在当前环境重训，时间和训练差异不视作严格受控的单变量因果结论。
- **选型与冻结**：在任何新 test 评分之前，按 validation 同分优先更少资产锁定原 `ensemble_v1`，保存 [选型记录](runs/mlp_capacity_v1/validation_selection.json)，再冻结全部四个新版本。预测先落盘并锁定哈希，随后独立官方评分。329 个历史实验文件及本轮冻结文件哈希保持不变。此前 test 已见，本轮是事后扩展，不据 test 继续调宽度、轮数、种子、权重或重选方案；不追加新的种子搜索。

复现训练、打包与 validation（从 `code/` 执行，复用现有 `solution/.venv`；使用新的目录名）：

```bash
unset PYTHONPATH
out=solution/runs/mlp_capacity_repeat
for width in 128 256; do
  second=$((width / 4))
  solution/.venv/bin/python -m solution.group_ranker.train_mlp \
    --feature-set text --reuse-config solution/runs/mlp_text_v1/checkpoint/config.json \
    --hidden "$width" "$second" --output "$out/m1_$width" || break
  solution/.venv/bin/python -m solution.group_ranker.submission_ensemble \
    --linear-checkpoint solution/runs/linear_v1/checkpoint \
    --mlp-checkpoint "$out/m1_$width/checkpoint" --output "$out/ensemble_$width" || break
  solution/.venv/bin/python -m solution.evaluate \
    --checkpoint solution/runs/linear_v1/checkpoint \
    --mlp-checkpoint "$out/m1_$width/checkpoint" \
    --ensemble-checkpoint "$out/ensemble_$width/checkpoint" --output "$out/validation_$width" || break
done
```

免训练 test 复现使用官方 `predict.py` 与 `score_predictions.py`，方法文件分别为 `group_ranker/submission_mlp.py`、`group_ranker/submission_ensemble.py`，指定上述各版本的原 checkpoint，输出到新目录。完整命令保存在各 `test/summary.json`。

记录：[总体结果](runs/mlp_capacity_v1/summary.json) · [固定实验计划](runs/mlp_capacity_v1/plan.json) · [128 validation](runs/mlp_capacity_v1/validation_128/summary.json) · [256 validation](runs/mlp_capacity_v1/validation_256/summary.json) · [检查](runs/mlp_capacity_v1/checks.json)。各版本目录含训练/准备记录、OOF、checkpoint 和 `test/`；四份冻结记录共用上层 [source.tar](runs/mlp_capacity_v1/source.tar)，避免重复源码快照。

### M0/M2 补充冻结 test，2026-10-21

应要求补测历史固定的基础 MLP M0 与算术 MLP M2。两者均沿用原 `64→16`、最终种子 7506、train OOF 确定的 4 轮重训权重；未重新训练。测试前确认当前推理代码与各自历史 validation 的 798 个选择完全一致，并在任一 test 评分前冻结两份源码、配置和资产；评分前另存预测哈希。

| 方法 | 原 validation / 798 | 完整 test / 467 |
|---|---:|---:|
| 基础线性 L0 | 730，91.48% | 375，80.30% |
| 基础 MLP M0 | 733，91.85% | **369，79.01%** |
| 文本 MLP M1 | 738，92.48% | 372，79.66% |
| 算术 MLP M2 | 733，91.85% | **373，79.87%** |
| 当时选定的 L0+M1 等权融合 | 740，92.73% | 377，80.73% |

- **基础网络没有显示稳定优势**：M0 相对 L0 纠正 20 题、改错 26 题，净减 6 题；差值的 paired bootstrap 95% 区间为 `[-4.07,1.50]` 个百分点。M1 比 M0 多对 3 题，与 validation 多对 5 题方向一致，但差异仍不足以证明稳定提升。网络容量增加本身没有带来确定收益。
- **算术有局部信号，总体收益有限**：M2 相对 M1 纠正 13 题、改错 12 题，只净增 1 题（+0.21 个百分点），区间 `[-1.93,2.36]` 包含 0；validation 则净减 5 题。Test 中能检查算式的 223 题，M2/M1 为 `180/223、171/223`，净增 9；其余 244 题为 `193/244、201/244`，净减 8。此分层是评分后描述，且两个网络分别重训、轮数不同，不能将差异全部归因于算术特征，也不据此设计 test 门控规则。
- **覆盖与难题**：仅 `223/467` 题、`418/1758` 个有效候选有成功检查；601 条可检查等式中 577 条通过、24 条矛盾，另有 8025 个未知片段，未知不当作错误。M0/M2 在两、三、四个答案组中分别为 `263/303、84/126、22/38` 与 `266/303、83/126、24/38`；四组仅 38 题，不能据其小样本结果重选模型。
- **资源与身份**：统一使用 `solution/.venv`，Python 3.12.3 / NumPy 1.26.4，Core Ultra 9 275HX。M0/M2 完整新进程预测加独立评分约 **0.272/0.510 秒**，全部未压缩推理资产 **560785/561013 字节**，预测和评分的实测总线程峰值均为 **4**。冻结 ID 前缀分别为 `2adf7544991d`、`3696b4f94fc3`；评分后冻结文件与此前 317 个实验文件哈希均未变化。满足本机时间、线程和资产限额，参考 CPU 资格仍待确认，缓存状态未控制。

区间按题配对重采样 2000 次，种子 7506，未校正多次比较。已有 test 已接触，此次是固定模型的事后补充评分，不称新盲测；不据 test 调参、换种子、添加规则或融合，原 validation 选定的最终等权融合保持不变。此前各阶段的“未运行 test”保留为当时的历史状态。

记录：M0 [官方成绩](runs/mlp_v1/test/score.json) · [比较、分层、命令与资源](runs/mlp_v1/test/summary.json) · [冻结清单](runs/mlp_v1/test/freeze.json)；M2 [官方成绩](runs/mlp_arithmetic_v1/test/score.json) · [比较、覆盖、命令与资源](runs/mlp_arithmetic_v1/test/summary.json) · [冻结清单](runs/mlp_arithmetic_v1/test/freeze.json)。两目录各自保存 `source.tar`、预测文件与评分前预测哈希；免训练复现共用 `group_ranker/submission_mlp.py`，分别指定 `runs/mlp_v1/checkpoint`、`runs/mlp_arithmetic_v1/checkpoint`，使用新的输出目录。

### M1 冻结 test，2026-10-19

先保存源码快照、代码/配置/权重 SHA-256 与 P3 选型依据，再仅运行已选 M1 的完整 test 官方预测及独立评分。冻结 ID 前缀 `1147e6f4ce3b`；测试后冻结文件哈希保持不变。当前工作区含未提交源码，精确冻结版本由 `source.tar` 与 SHA-256 清单标识。

| 项目 | 结果 |
|---|---:|
| 官方完整 test accuracy | **372/467，79.65738758029978%** |
| 错题数 | 95 |
| 本机新进程预测 + 独立评分 | **0.405 秒** |
| 实测总线程峰值 | **4**（编排 1 + 子进程 3） |
| 未压缩推理资产 | **560,937 字节**，约 0.535 MiB |

- **与 validation 的关系**：test 467 题全是冲突题；validation 有 443 道单答案组题。对应的冲突题准确率为 validation `295/355，83.10%`，test `372/467，79.66%`，下降约 3.44 个百分点。
- **与历史线性结果比较**：旧线性为 `375/467，80.30%`。M1 纠正 23 题、改错 26 题，净减 3 题（-0.64 个百分点）；配对 bootstrap 95% 区间为 `[-3.64,2.36]` 个百分点，不能据此认定两者存在稳定优劣。validation 上 M1 的收益未在本次 test 重现。
- **分歧越多仍越难**：M1 在 2/3/4 个有效答案组的问题上分别为 `266/303（87.79%）`、`85/126（67.46%）`、`21/38（55.26%）`。相比旧线性分别净增 3、净减 8、净增 2 题；本次净差主要来自三答案组题。

以上为冻结后的描述性评分分析，不用于调参、换种子、集成或重选模型。耗时包含两个新进程的启动、导入、哈希校验、载入、预处理、输出与评分；硬件为 Core Ultra 9 275HX，Python 3.12.3 / NumPy 2.5.3，缓存状态未控制，**参考 CPU 资格尚未确认**。

记录：[官方评分](runs/mlp_text_v1/test/score.json) · [完整命令、资源与分层比较](runs/mlp_text_v1/test/summary.json) · [冻结清单](runs/mlp_text_v1/test/freeze.json) · [冻结源码](runs/mlp_text_v1/test/source.tar)。分析复用旧线性的冻结预测，本轮未重跑其他 test 方法。

### linear_v1 冻结 test

2026-10-18，冻结 `linear_v1` 后，仅运行该线性方案的完整 test 预测和官方独立评分；未据 test 结果调参或重选模型。测试前保存代码、权重、配置的 SHA-256 和源码快照，测试后核对均未变化。

| 项目 | 结果 |
|---|---:|
| 官方 test accuracy | **375/467，80.29978586723769%** |
| 完整预测 + 独立评分外部总耗时 | **0.243 秒** |
| 实测总线程峰值 | **4**（编排 1 + 子进程 3） |
| 未压缩推理资产 | **28,771 字节**，约 28.10 KiB |
| 本机 CPU / 推理环境 | Core Ultra 9 275HX / Python 3.12.3、NumPy 2.5.3 |

耗时包含两个新进程的启动、导入、数据与资产哈希校验、加载、预处理、预测和评分；本机文件缓存状态未控制。**该成绩不是 Xeon Platinum 8457C 参考机测时，正式资格仍以参考机为准。**

记录：[官方评分](runs/linear_v1/test/score.json) · [耗时、线程与完整命令](runs/linear_v1/test/summary.json) · [冻结清单与检查结果](runs/linear_v1/test/freeze.json) · [冻结源码](runs/linear_v1/test/source.tar)。匹配权重位于 [checkpoint/](runs/linear_v1/checkpoint/)，冻结 ID 前缀为 `22c93832f59d`。

## 冻结结果复现

最终 L0＋M1 的免训练复现、可选重训及报告生成命令统一维护在[根目录中文 README §5](../../README_cn.md#5-复现命令指南)，避免多处维护。使用 `group_ranker/submission_ensemble.py` 与 `runs/ensemble_v1/checkpoint/`，并按根目录命令加载对应的冻结源码快照；注意力仅作为探索性对照。

复现历史单模型时使用各自冻结源码：线性的方法为 `submission_linear.py`、checkpoint 为 `solution/runs/linear_v1/checkpoint`；M1 为 `submission_mlp.py`、`solution/runs/mlp_text_v1/checkpoint`。使用新的输出目录；历史 JSON 保留原始完整命令。源码持续演进，历史精确版本由各自 `source.tar` 和 SHA-256 清单标识。

仓库一并维护 L0、M1、最终融合和其他实验的 checkpoint、训练／评测记录与冻结源码，`runs/` 不再整目录忽略。正式报告及 AI 辅助说明见 [report/](../../report/report.md)。正式提交应使用可公开访问的匹配仓库版本与权重链接，并完成参考机资格确认、保留数据许可。
