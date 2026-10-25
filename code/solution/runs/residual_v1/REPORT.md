# 非线性残差头与手工扩展实验

试验网格、回退条件和选型规则在训练前写入 plan.json。训练按题五折；只在训练 fold 拟合标准化与线性 warm start。

| 配置 | OOF 冲突题 | Validation | 冲突题 | 纠正/改错（对原6维模型） |
|---|---:|---:|---:|---:|
| incumbent | 82.12% | 733/798 (91.85%) | 81.69% | 0/0 |
| residual_h8_r0.001 | 80.04% | 731/798 (91.60%) | 81.13% | 13/15 |
| residual_h8_r0.01 | 82.12% | 733/798 (91.85%) | 81.69% | 0/0 |
| residual_h16_r0.001 | 79.28% | 737/798 (92.36%) | 82.82% | 17/13 |
| residual_h16_r0.01 | 82.17% | 733/798 (91.85%) | 81.69% | 0/0 |

是否触发手工扩展：False。按预设规则选定：**residual_h16_r0.001**。
预测入口：solution/residual_ranker/submission.py。Checkpoint：solution/runs/residual_v1/residual_h16_r0.001/checkpoint。
隐藏层/正则搜索记录、优化收敛状态、成对区间见各 training.json 及 validation_summary.json。
公开 test 历史成绩已知；本轮未据 test 调参、扩展特征或重选模型。

冻结 test：**376/467，80.51391863%**。
本机完整预测加评分 0.530 秒，线程峰值 2，推理资产 31442 字节；参考 CPU 尚未验证。

实验外部总用时：157.0 秒。
