# 轻量手工特征与答案组排序

## 目录与入口

```text
handcrafted_ranker/
├── arithmetic.py          # 算术检查与候选/答案组手工指标
├── features.py            # 复用基础特征，追加手工特征
├── train_linear.py        # 本方案训练入口，复用基础线性优化函数
├── submission_linear.py   # 本方案官方预测入口
├── experiment.py          # 四配置消融、选型及冻结测试
├── test_handcrafted.py
└── README.md
```

`group_ranker/` 保持基础方案，不依赖本目录；本方案单向复用其特征及线性训练函数。实验产物仍集中在 `solution/runs/handcrafted_v1/`。

## 特征定义

保留 basic-v1 的 2,088 维布局，仅使用原预留位置 2064:2072。输入不读取参考答案，答案解析与官方 common.parse_answer 一致。特征先按候选计算，再在同一解析答案组内聚合；组内成员按原有稳定顺序聚合。

| 相对索引 | 答案组特征 |
|---:|---|
| 0 | mean(log1p(候选词数)) |
| 1 | mean(log1p(可检查等式步数)) |
| 2 | mean(错误等式步数 / 可检查步数)，无可检查步骤取 0 |
| 3 | mean(最后一个可检查等式成立且右端匹配最终答案) |
| 4 | mean(至少一个成立等式的右端匹配最终答案) |
| 5 | mean(题面不同数值在解答正文出现的比例) |
| 6 | max(特征 3 对应的逐候选指标) |
| 7 | max(特征 4 对应的逐候选指标) |

正文排除最后一个 #### 及其后内容；无 #### 时截断于官方支持的最后一个平坦 boxed 标记之前。数字覆盖按精确数值集合比较，重复题面数字不加权，无题面数字取 0。该特征只代表数字出现，不代表正确使用条件。无法解析最终答案的候选逐候选特征全零，也不进入答案组。

局部算术使用 Fraction 和完整递归下降解析，不使用 eval。支持加减乘除、括号、负号、3(18) 隐式乘法、LaTeX 分式、乘除符号和百分号；每个相邻等号形成一步，至少一侧含算术运算。绝对差严格小于 1/1000 视为匹配。分数最终答案同样参与 trace。除零、未知变量、截断和资源边界超限的片段跳过，不标为错误。

单位文字保留分隔标记，避免把单位转换两端直接当作相同量纲数值。仍可能遗漏表达式；舍入、百分比结果的常见书写惯例及未支持的格式也可能产生误报。这是有限语法下的局部证据，不是题意或推理正确性的认证。算术提取限制单行长度、表达式长度、token 数、嵌套深度及有理数位数。

## 四个预设配置

- basic：无新增特征，保留 answer-group-basic-v1，兼容归档 checkpoint。
- math：新增索引 1、2、3、4、6、7，共 6 维。
- math_question：在 math 上加入索引 5，共 7 维。
- full：全部 8 维。

非 basic 配置使用 answer-group-handcrafted-v2；具体掩码写入 checkpoint 的 feature_set 和 enabled_scalar_indices。推理从 checkpoint 选择配置，训练和推理共用同一特征函数。当前实现对非 basic 配置统一提取 8 维后屏蔽禁用列，消融耗时不代表各特征块的独立成本。

每个配置均按题目、票型分层进行同样的五折训练，种子 7506；标准化只在该 fold 的训练部分拟合。L2 仅比较 0.0001、0.001、0.01，按训练 OOF 准确率、NLL、较强正则依次选择，再在所有训练冲突题上重训。

配置选型规则在训练前保存到 plan.json：validation 正确题数优先，其次 train OOF 准确率、OOF NLL、较少新增特征。选型后仅对一个冻结模型运行 test；不根据 test 重选或调参。本项目此前已有 linear_v1 的公开 test 成绩，本轮是后续实验，不能声称 test 完全未见。

## 命令与产物

从 code/ 执行。此次使用 Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4，版本对应 solution/requirements-train.txt。推理入口只使用 NumPy 和标准库，不导入 SciPy。

    python3 -m unittest solution.group_ranker.test_group_ranker solution.handcrafted_ranker.test_handcrafted -v

新建实验目录运行四个配置并冻结测试；已有目录会被拒绝，不覆盖历史记录：

    python3 -m solution.handcrafted_ranker.experiment --output solution/runs/handcrafted_v2 --test

省略 --test 则只做训练与 validation。也可单独训练：

    python3 -m solution.handcrafted_ranker.train_linear --feature-set full --output solution/runs/handcrafted_full_v2

已有选定模型无需重训，使用新的预测入口复现（从 code/ 执行，输出到新目录）：

~~~bash
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3
python3 predict.py --method solution/handcrafted_ranker/submission_linear.py \
  --checkpoint solution/runs/handcrafted_v1/math/checkpoint \
  --split validation --output solution/runs/handcrafted_reproduction/validation-predictions.json
python3 score_predictions.py --split validation \
  --predictions solution/runs/handcrafted_reproduction/validation-predictions.json \
  --output solution/runs/handcrafted_reproduction/validation-score.json
~~~

本轮结果位于 [handcrafted_v1/REPORT.md](../runs/handcrafted_v1/REPORT.md)。该目录保留 plan.json、四个训练日志、checkpoint、OOF、官方 validation 预测和评分、分层/成对比较、算术覆盖率样例、单元测试记录、selection.json；使用 --test 时另存 freeze.json、source.tar 和官方 test 结果。

推理计时包围两个顺序启动的新进程，包括启动、导入、哈希、加载、预处理、输出和独立评分；编排进程 1 线程，子进程 3 线程，每 1ms 采样线程数。本机文件缓存状态未控制，正式资格仍需指定参考 CPU 验证。新 runs 默认受 Git 忽略，交付时应包含匹配的选定 checkpoint 和记录。

AI 辅助披露：本轮手工特征设计讨论、提取器/实验脚本实现、测试与结果分析使用了 Codex 协助；复用了项目已有答案分组、线性训练和官方解析/评分代码。特征计算无 LLM、网络或外部数据调用。

## 本轮实测结果（2026-10-25）

| 配置 | 新增维数 | Train OOF 冲突题 | Validation 全量 | Validation 冲突题 |
|---|---:|---:|---:|---:|
| basic | 0 | 81.13% | 730/798，91.48% | 287/355，80.85% |
| math | 6 | 82.12% | 733/798，91.85% | 290/355，81.69% |
| math_question | 7 | 82.12% | 732/798，91.73% | 289/355，81.41% |
| full | 8 | 82.07% | 732/798，91.73% | 289/355，81.41% |

四个配置均选择 L2=0.01，全部优化收敛；折号及标签逐项一致。重新训练的 basic 权重、标准化数组、OOF 分数与原 linear_v1 逐位一致。

按预设 validation 规则选择 math，验证集纠正 6 题、改错 3 题，净增 3 题（+0.376 个百分点）。成对 bootstrap 95% 区间为 [-0.376, 1.128] 个百分点，包含 0；这是小幅、尚不稳定的验证收益，不足以声称每个算术特征都有效。数字覆盖和长度未带来额外 validation 收益，本轮不采用；也未在 test 之后继续删特征或调参。

冻结后的唯一 test 运行得到 **384/467，82.22698072805139%**。相对原归档 375/467 多对 9 题，分数高 1.927 个百分点。预测与独立评分的本机外部总耗时 **0.520 秒**，采样总线程峰值 **2**（设置上限为 4），未压缩推理资产 **28,850 字节**。本机为 Core Ultra 9 275HX，非参考 CPU 测试。

17 项单元测试通过。选定模型的完整 validation 预测重放一致，全部 798 题候选重排后所选答案不变；旧 checkpoint 重放一致，实际推理未导入 SciPy。原实验结束时冻结的 22 个文件哈希及源码归档哈希复核通过；目录迁移后对历史源码归档核验，并另存迁移记录。

算术覆盖率：全部 train 的 19,020 个候选中有 16,316 个提取到可检查步骤；validation 为 2,758/3,192。提取的 train/validation 步数分别为 41,793/6,764，其中 739/126 步被标记为数值不一致；这些标记仍可能包含格式或百分比书写造成的误报。原始样例见实验目录的 arithmetic_diagnostics.json。

选定 checkpoint 为 runs/handcrafted_v1/math/checkpoint；迁移后的运行入口是 handcrafted_ranker/submission_linear.py，特征配置由 checkpoint 自动确定。详细记录与官方结果见 [实验报告](../runs/handcrafted_v1/REPORT.md)。

## 目录迁移

手工特征相关源码和本文从 group_ranker/ 及 HANDCRAFTED.md 移到本目录。已有 checkpoint、OOF、官方预测/评分、freeze.json 与 source.tar 均保持原样，历史 JSON 中的命令保留当时路径。历史冻结哈希对应 source.tar 中的源码；当前目录结构及行为一致性复核另记于实验目录的 migration.json。迁移不重训、不调参、不重选模型。

## 提交前有限优化

新增训练参数 --handcrafted-l2-multiplier，可传多个正数。旧列使用 L2=lambda，手工及交互列 2064:2074 使用 lambda*multiplier；惩罚在训练 fold 的标准化之后施加。倍率为 1 时直接调用原优化函数，保留数值兼容性。推理仍只加载单个线性权重，不增加模型成员。

新增 math_interactions 配置，保留原 6 维算术指标，追加两维：
- 2072：答案组票数 / 4 × mean_invalid_equation_fraction；
- 2073：答案组票数 / 4 × mean_last_equation_matches。

分母包含全部四个候选；未解析候选也占分母。新配置采用 answer-group-handcrafted-v3，维度仍为 2,088；原 v1/v2 checkpoint 保持兼容。

本轮预设步骤：
1. math 配置搜索 L2=[0.01,0.03,0.1]，手工倍率=1；
2. 固定第一步 OOF 选出的 L2，搜索手工倍率=[0.3,1,3]；
3. math_interactions 使用同一个 L2，搜索相同三个手工倍率；
4. 每阶段只将 OOF 选出的配置送至 validation，与原 6 维模型一起按预设规则选一个；冻结后仅运行该方案 test。

从 code/ 执行：

    python3 -m unittest solution.group_ranker.test_group_ranker solution.handcrafted_ranker.test_handcrafted solution.handcrafted_ranker.test_optimization -v
    python3 -m solution.handcrafted_ranker.optimize --output solution/runs/handcrafted_opt_v1 --test

输出目录不得已存在。省略 --test 则只训练与验证。规则、全部搜索记录、官方评分、冻结源码和资源测量保存到实验目录；结果见 [有限优化报告](../runs/handcrafted_opt_v1/REPORT.md)。本轮不改变原来的算术解析规则，不根据 test 继续扩展搜索。
