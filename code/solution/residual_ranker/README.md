# 小型非线性残差答案组排序

本目录独立于 group_ranker/ 和 handcrafted_ranker/，复用它们的基础特征与线性训练函数，不修改已有方案。

模型：s = x @ w + tanh(z @ U + b) @ v。x 为原 2,088 维 math 特征；z 只取组统计与算术的 24 维。隐藏层为 8 或 16，新增参数分别为 208、416。线性头 L2=0.01；残差头 L2 预设 0.001、0.01。训练用当前 fold 内拟合的线性模型初始化 w，固定随机种子初始化 U，v=0，因此初始预测等于该 fold 的线性模型。

按问题与票型做相同五折，fold 内拟合标准化及 warm start。训练用 float64 优化，导出 float32 权重；训练指标和推理共用 model.scores 前向。推理只需 NumPy 与标准库，训练另需 SciPy。

若四个初始配置均没有提高 validation 正确题数，则运行预先定义的手工扩展：在原预留槽位 2072:2088 填入下列八个候选指标的组内均值与最大值，共 16 维：
1. log1p(候选词数)；
2. log1p(正文不同数值个数)；
3. 题面内容词在正文出现的比例；
4. 题面末句与正文最后三行的内容词 Jaccard；
5. 可检查等式数 / 原文等号数；
6. log1p(错误等式数)；
7. 成立等式中右端匹配最终答案的比例；
8. 题面不同数值在正文出现的比例。

固定英文停用词和简单复数词干规则，无外部语料或权重。正文范围沿用已有 body_without_final；未知答案候选全部置零。上述特征是软证据，包含长度和表达风格影响，不能解释为推理正确性保证。

扩展后先用线性模型在 L2=[0.003,0.01,0.03] 中按 OOF 选一个；再用该 L2、8 隐藏单元和残差惩罚 [0.001,0.01] 按 OOF 选一个。每个模型只使用 train 监督；validation 用于条件触发与最终选型，test 只评分最终一个冻结方案。

从 code/ 执行：

    python3 -m unittest solution.residual_ranker.test_residual -v
    python3 -m solution.residual_ranker.experiment --output solution/runs/residual_v1 --test

旧目录不覆盖。完整记录见 [实验报告](../runs/residual_v1/REPORT.md)，包含初始计划、每配置训练/OOF、官方预测与评分、条件回退决定、选型、源码与权重快照及资源记录。历史公开 test 成绩已知，本轮属于后续实验，不能声称 test 完全未见。

新预测入口为 solution/residual_ranker/submission.py；若最终仍选原方案，则继续使用 selection.json 指定的原 handcrafted_ranker 入口。运行 predict.py 时遵循记录中的入口与匹配 checkpoint。

AI/复用披露：本轮模型设计、代码、梯度测试及实验分析由 Codex 协助；复用已有 NumPy 特征、SciPy 线性优化和官方解析/评分工具。特征与推理无 LLM、联网或外部预训练模型。
