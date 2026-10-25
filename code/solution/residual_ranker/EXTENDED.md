# 新增手工特征扩展

本模块独立保存在 residual_ranker/extended_*.py，不修改原 group_ranker、handcrafted_ranker 或已有 residual 模型的特征与权重。新模型使用线性组排序，推理仅依赖 NumPy 与标准库；没有参考答案、网络访问、LLM 或随机特征。

候选级提取器 candidate_extensions(question, candidates) 支持任意候选数，返回 float64 [k,24]。答案无法解析时整行置零。官方组排序接口继续遵循固定四候选协议。答案解析、可检查等式、精确 Fraction 与千分位处理复用已有实现。

| 列 | 信号 | 含义 |
|---:|---|---|
| 0 | focus_body_recall | 题面最后一句的实词在解答正文中的覆盖率 |
| 1 | focus_tail_recall | 目标实词在正文末三条非空行中的覆盖率 |
| 2 | focus_lastline_recall | 目标实词在正文最后一条非空行中的覆盖率 |
| 3 | question_lastline_jaccard | 题面与正文末行实词的 Jaccard |
| 4 | focus_unit_lastline_recall | 目标单位词在正文末行中的覆盖率 |
| 5 | log_nonempty_lines | 正文非空行数的 log1p |
| 6 | log_raw_equals | 原始等号数的 log1p |
| 7 | log_answer_mentions | 正文数字 token 等于最终答案的次数的 log1p |
| 8 | answer_nonnegative_integer | 最终答案是否为非负整数 |
| 9 | signed_log_answer | 带符号的 log1p(答案绝对值)，幅值上限 1e100 |
| 10 | answer_question_log_ratio | log1p(答案绝对值)减去log1p(题面最大数字)，相同幅值上限 |
| 11 | novel_body_number_fraction | 正文不同数字中没有出现在题面的比例 |
| 12 | log_valid_steps | 有效算术步骤数的 log1p |
| 13 | all_checked_steps_valid | 至少一个可检查步骤且全部算术成立 |
| 14 | last_checked_rhs_matches | 最后一个可检查步骤的右端是否匹配答案 |
| 15 | last_valid_rhs_matches | 最后一个算术成立步骤的右端是否匹配答案 |
| 16 | all_steps_match_fraction | 全部可检查步骤中右端匹配答案的比例 |
| 17 | invalid_step_reaches | 是否存在算术不成立、但右端匹配答案的步骤 |
| 18 | adjacent_valid_link_fraction | 相邻有效步骤间，前一步结果出现在后一步左端数字 token 的比例 |
| 19 | valid_result_reused_fraction | 有效步骤结果在后续有效步骤左端数字 token 中复用的比例 |
| 20 | dangling_valid_result_fraction | 有效结果既不匹配答案，也未被后续左端数字 token 复用的比例 |
| 21 | valid_lhs_match_fraction | 有效步骤中左端求值匹配最终答案的比例 |
| 22 | last_valid_match_position | 最后一次有效结果匹配在有效步骤序列中的相对位置 |
| 23 | last_valid_result_disagreement | (答案−最后有效结果)/max(1,两者绝对值)，无有效步骤为0 |

词法只使用固定正则、停用词、小写及简单去复数；题面目标取最后一句。这些都是软信号，不是语义判断。数字 token 复用使用已有 unsigned numbers_of，因此不等价于带符号表达式、分数表达式或符号变量的数据流分析；等式检查也不验证解题思路。

每个答案组按候选文本和嵌入字节稳定排序后聚合。新矩阵保留原 expanded 的前2088列；追加部分如下：

| schema | 追加方式 | 新增列 | 总维度 |
|---|---|---:|---:|
| lexical | 词法12项的组均值、最大值 | 24 | 2112 |
| trace | 算术12项的组均值、最大值 | 24 | 2112 |
| combined | 全24项的组均值、最大值 | 48 | 2136 |
| contextual | combined + 24项组均值的组内Z分数 + 8项票数交互 | 80 | 2168 |

组内Z分数对同题不同答案组等权计算，std+1e-6；票数交互为组候选数/k乘以候选特征0、2、11、12、15、17、18、20的组均值。这与训练fold拟合的全局标准化分开：全局标准化沿用原排序器并裁剪至[-8,8]，所有训练、验证和推理入口调用同一实现。

从 code/ 执行：

    PYTHONDONTWRITEBYTECODE=1 python3 -m solution.residual_ranker.extended_experiment --output solution/runs/handcrafted_extended_reproduction --test

只训练四种特征块、三档L2并保存OOF：

    PYTHONDONTWRITEBYTECODE=1 python3 -m solution.residual_ranker.extended_train --output solution/runs/extended_training_reproduction

测试：

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest solution.residual_ranker.test_extended -v

本轮选中lexical，L2=0.03，实际新增24列，总维度2112；OOF1761/2114，validation751/798，冻结test393/467（84.15418%）。四种方案和预设计划见[本轮报告](../runs/handcrafted_extended_v2/REPORT.md)，当前入口与复现命令见[RESULTS.md](RESULTS.md)。新模型入口为 extended_submission.py，必须使用匹配 schema 的 checkpoint；已有 submission.py 与83.08351%模型继续可用。runs/ 默认被Git忽略，提交时需另外包含选定权重及记录。
