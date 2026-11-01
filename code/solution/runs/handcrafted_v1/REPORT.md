# 手工特征消融实验

四个配置和选型规则在训练前写入 plan.json；每个配置使用相同 train 五折，L2 仅由 train OOF 选择。

| 配置 | 新增维数 | OOF 冲突题 | validation | validation 冲突题 | 纠正/改错（对基本模型） |
|---|---:|---:|---:|---:|---:|
| basic | 0 | 81.13% | 730/798 (91.48%) | 80.85% | 0/0 |
| math | 6 | 82.12% | 733/798 (91.85%) | 81.69% | 6/3 |
| math_question | 7 | 82.12% | 732/798 (91.73%) | 81.41% | 6/4 |
| full | 8 | 82.07% | 732/798 (91.73%) | 81.41% | 6/4 |

按预设 validation 规则选择：**math**。

完整逐题结果、分层指标和成对 bootstrap 区间见 validation_summary.json。算术覆盖率及输入样例见 arithmetic_diagnostics.json。
本轮属于已知公开 test 历史成绩后的后续研究，不能声称测试集完全未见；本轮没有根据 test 调参或重选模型。

冻结 test：384/467，82.22698073%；本机完整预测加评分 0.520 秒；总线程峰值 2；推理资产 28850 字节。
该时间是本机测量，非 Xeon Platinum 8457C 资格认证。源码和 checkpoint 的冻结哈希见 freeze.json，快照见 source.tar。
