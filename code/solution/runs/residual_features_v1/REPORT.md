# 手工特征扩展后续实验

本轮为独立后续实验，不改写 residual_v1 的预设规则与测试结果。16 个扩展指标在第一轮残差网络训练前已实现，未针对 test 错题设计。

本轮收紧资格：五折及最终优化均收敛，且 OOF 正确题数不低于原模型；随后按 validation 选型。资格和搜索范围在本轮训练前写入 plan.json。

| 模型 | OOF 冲突题 | Validation | 冲突题 | 纠正/改错 | 可选资格 |
|---|---:|---:|---:|---:|---|
| incumbent | 82.12% | 733/798 (91.85%) | 81.69% | 0/0 | True |
| expanded_linear | 82.36% | 734/798 (91.98%) | 81.97% | 6/5 | True |
| expanded_residual_r003 | 81.36% | 740/798 (92.73%) | 83.66% | 16/9 | False |
| expanded_residual_r01 | 82.54% | 734/798 (91.98%) | 81.97% | 6/5 | True |

冻结前选定：**expanded_residual_r01**。入口：solution/residual_ranker/submission.py。Checkpoint：solution/runs/residual_features_v1/expanded_residual_r01/checkpoint。
历史 test 分数已知，本轮不构成完全未见测试；本轮未据新 test 调参或重选。

官方 test：**388/467，83.08351178%**。
本机完整预测加评分 0.800 秒；资产 31188 字节；参考 CPU 未验证。

本轮外部耗时 118.1 秒。

## 复核与解释

27 项测试通过；选定模型的 validation/test 重放与全部候选重排检查通过，推理未导入 SciPy；39 个冻结文件和源码归档哈希复核通过。

选定配置 validation 纠正 6 题、改错 5 题，净增 1；成对 bootstrap 增益区间 [-0.752,1.003] 个百分点包含 0，验证证据仍较弱。选定残差和扩展线性模型的 validation 预测一致：True。隐藏权重/输出范数均约 0.003，残差很小，不能单独声称非线性带来增益。

相对原 6 维模型，最终官方 test 从 384/467 提升为 388/467，增加 4 题，约 0.86 个百分点；此分数只用于报告本轮冻结结果，未据此重选模型。

当前复现入口和完整命令见 solution/residual_ranker/RESULTS.md。
