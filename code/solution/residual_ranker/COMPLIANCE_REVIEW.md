# 当前84.15418%方案的规则审阅

审阅日期：2026-10-25。范围为当前冻结lexical模型、完整推理调用链、训练和选型流程、原始发布文件及提交材料；不只检查编辑器打开的handcrafted_ranker/features.py。

**结论：未在当前训练或推理实现中发现参考答案泄漏、答案查表、外部预训练模型、联网推理或资源资产超限等明确硬性违规。手工特征和线性答案组排序属于课程开放设计范围。但实验历史的test反馈需要完整披露，正式提交及参考机资格尚未全部完成，不能据此保证“全流程100%合规”。**

规则以[英文Guide](../../../guide/GUIDE.md)为准，尤其第3节约束、第4节交付要求；[官方运行说明](../../README.md)明确允许自定义method文件和自行锁定依赖。

**需要优先说明的发现**

1. **实验流程存在test反馈风险。** 规则要求test参考只用于最终评分和复现，不参与训练、调参或选型。工作区至少保留六轮test汇总：linear_v1 375/467、handcrafted_v1 384/467、handcrafted_opt_v1 384/467、residual_v1 376/467、residual_features_v1 388/467、handcrafted_extended_v2 393/467。多次评分本身不足以证明违规，规则也允许复现及历史提交；关键在于是否把test反馈用于后续决策。当前轮的12个训练配置预先固定，L2按训练OOF选择、模型按validation选择，先写selection/freeze再打test，这部分代码检查通过。但整个项目在看到过test成绩后继续扩展，不能把每轮冻结解释成重新得到一个未见test，或证明历史反馈毫无影响。报告应保留完整时间线与失败结果，停止用test反馈驱动新的选择；如需对历史流程资格作最终认定，应由授课方判断。
2. **当前复现环境与官方默认安装文件不同，已补齐独立依赖声明。** 实测Python3.12.3、NumPy1.26.4；官方code/requirements.txt指定NumPy2.5.3。Guide允许添加并锁定自有依赖，因此使用1.26.4不是已确认的规则冲突，但不能把1.26.4的验证结果当作2.5.3的验证结果。新建requirements-inference.txt并在RESULTS.md补充独立环境命令；官方文件与冻结源码、权重未改。未安装或验证NumPy2.5.3。
3. **正式交付尚不完整。** 当前新目录未加入Git记录，所选checkpoint又被runs规则忽略。仅有本地文件不能保证公开仓库可复现。最终还需公开可访问的代码与匹配checkpoint链接、可核对不超过10页的报告、官方JSON、种子/资源/哈希/AI披露及明确代码版本。工作区未发现已整理好的最终报告或本轮公开提交记录。source.tar是源码和权重快照，包含45个文件，但不包含完整数据、最新RESULTS/EXTENDED说明、本审阅或新依赖声明，不能独自等同完整作业交付。
4. **参考CPU资格待验证。** 本机Core Ultra9 275HX，完整新预测进程加独立评分1.166秒、实测总线程峰值2，远低于本地限额；资格依据仍是Xeon Platinum8457C，不应写成已通过参考机。复现应保留文档中的三线程环境设置，不能省略后以默认BLAS线程数作合规保证。
5. **必须使用正确自定义入口。** 官方允许其他Python文件实现predict。当前入口为solution/residual_ranker/extended_submission.py，checkpoint为solution/runs/handcrafted_extended_v2/models/lexical/checkpoint。官方code/submission.py仍是空接口，直接使用它会报NotImplementedError。当前RESULTS.md提供匹配命令。该问题属于复现入口要求，不是模型规则违规。

**逐项核查**

| 项目 | 检查结果与证据 |
|---|---|
| 官方数据、解析器、评分器 | PACKAGE_MANIFEST中20个发布文件全部哈希匹配；check_data.py通过全部三个split的哈希、互斥、输入形状和引用对齐检查。没有更改官方最终答案解析语义。 |
| 训练数据范围 | extended_train.prepare只打开train输入和gsm8k_train.jsonl，目标从训练参考提取；ID只用于训练对齐。最终只在train上重拟合，没有用validation/test拟合权重或标准化。 |
| OOF与选型 | 按题、票型分层五折，种子7506；每fold仅在训练fold拟合标准化。L2按train OOF选，四种特征块按validation选；selected模型全部优化收敛。当前轮test在selection与freeze之后运行。 |
| 手工特征 | 最终答案值来自候选文本；词法、单位词、格式和算术均为确定性输入规则。非负整数等数值先验、答案共识和题内标准化没有被作业禁止；它们不构成“数学推理能力”的独立证明。 |
| 单答案组捷径 | 仅一个可解析答案组时选择该组，是利用作业“每题至少一个正确最终答案”的已公开保证，不读取标签。作业按最终数值评分，不要求逐步推理全部正确。 |
| 预测来源与网络 | 在禁止重开code/data、读取其它runs记录、联网和启动子进程的运行时守卫下，预测通过；模型只读取model.npz、config.json、asset_manifest.json。没有读取保存的predictions.json、训练参考或测试参考；未导入SciPy。 |
| 新题库适配 | 无ID及重命名ID结果相同；题目倒序后预测相应倒序；N=0、1、7、129均通过；一道人造新题可接受。四候选、1024维、chunk128是协议或批处理参数，不是写死测试题数。 |
| 候选重排和batch | 上一轮核验覆盖798道validation和467道test；选择答案重排不变，822道冲突题特征矩阵逐位一致，批次大小变化不改预测。 |
| 资产 | 模型实际读取的三个资产总计29,766字节，数组27,456字节；npz四个成员均为ZIP_STORED，不靠压缩规避16MiB限制。其它旧模型和OOF记录未在推理时读取。 |
| 离线CPU与训练资源 | 推理只有NumPy和标准库；没有新LLM生成或API调用，没有额外预训练权重或训练数据。训练使用CPU。原四方案三档L2训练日志记录总计约91秒，完整实验约97.8秒。 |
| 复现 | 审阅时只重训固定lexical/L2=0.03，没有新增搜索；mean、scale、active、weight四数组，OOF scores/labels/folds及配置均与原文件逐位一致。仍需在公开交付环境按匹配命令复现。 |
| 冻结与成绩 | 45个冻结文件及source.tar哈希匹配；当前官方JSON为393/467、84.1541755888651%，完整467题评分。没有按题过滤后冒充全test分数。 |
| AI与复用、许可证 | 允许AI，但要求理解与披露。已新增AI_AND_REUSE.md覆盖最新特征、脚本、分析与本次审阅；最终报告应继续明确披露。数据LICENSE.txt和NOTICE.txt均保持原始哈希。提交者是否已充分理解代码不能由程序核验。 |

**本次审阅新增或修改的材料**

- requirements-inference.txt：锁定已验证的NumPy1.26.4推理环境。
- AI_AND_REUSE.md：记录实质性AI辅助、内部及课程工具复用、数据来源和test历史说明。
- RESULTS.md：补充独立安装命令和审阅入口。
- ../runs/handcrafted_extended_v2/review/runtime_audit.json：禁止数据重开、实验缓存、联网和子进程后的运行时检查。
- ../runs/handcrafted_extended_v2/review/official_data_check.json：官方数据检查输出。
- ../runs/handcrafted_extended_v2/review/reproducibility.json：固定配置重训及发布文件、冻结文件哈希检查。

本次未修改冻结模型源码、参数、解析器、评分器或数据，没有进行新的test选型。此前“每轮test后未再调参”的表述只适用于对应轮次，不能替代整个项目的实验历史披露。
