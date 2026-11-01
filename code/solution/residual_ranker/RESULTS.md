# 当前选定模型与复现

当前模型为 lexical：在原 expanded 的2088列后追加24列手工组特征，总维度2112，线性组排序，L2=0.03。没有隐藏层。所有五折及最终优化均收敛。该模型在新test前已按预设的OOF/validation规则冻结。

| 项目 | 上一版 expanded_residual_r01 | 当前 lexical |
|---|---:|---:|
| Train 五折 OOF 冲突题 | 1745/2114，82.54% | 1761/2114，83.30% |
| Validation | 734/798，91.98% | 751/798，94.11% |
| Validation 冲突题 | 291/355，81.97% | 308/355，86.76% |
| 官方 test | 388/467，83.08351% | **393/467，84.15418%** |

Test比上一版多对5题，提升1.07个百分点。Validation纠正21题、改错4题，净增17题；成对bootstrap增益区间为[0.877,3.383]个百分点。此区间未校正多次选型，历史公开test也已知，不能将这轮描述为完全未见测试。

本轮新增24个逐候选信号，分别对词法12项和算术追踪12项做消融；组均值/最大值形成48列，完整版本再加24列组内标准化和8列票数交互。四种特征块均只在train OOF选择L2，然后在validation选模型。选定lexical只保留词法12项的均值与最大值，新增24列；完整80列扩展未胜出。

Lexical包括题面目标与正文/结尾的词重合、单位呼应、非空行/等号/答案提及数、答案数值形态和正文新数字比例。没有逐个特征消融，不能归因于某一项；新系数绝对值较大的项包含“答案是否为非负整数”，收益也包含数据数值先验。详细定义见[EXTENDED.md](EXTENDED.md)。

本机完整预测加独立评分1.166秒，线程峰值2，未压缩推理资产29,766字节（29.07 KiB）。环境为Python3.12.3、NumPy1.26.4；训练另需SciPy1.11.4，推理未导入SciPy。参考CPU尚未验证。

从 code/ 执行。使用已验证的独立推理环境；课程原始requirements.txt指定NumPy2.5.3，本模型本机验证使用1.26.4，未验证2.5.3。官方文件保持原样，自有依赖单独锁定：

    python3 -m venv solution/.venv-inference
    solution/.venv-inference/bin/python -m pip install -r solution/residual_ranker/requirements-inference.txt
    source solution/.venv-inference/bin/activate

免训练复现：

    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3
    python3 predict.py --method solution/residual_ranker/extended_submission.py --checkpoint solution/runs/handcrafted_extended_v2/models/lexical/checkpoint --split test --output solution/runs/extended_final_reproduction/predictions.json
    python3 score_predictions.py --split test --predictions solution/runs/extended_final_reproduction/predictions.json --output solution/runs/extended_final_reproduction/score.json

训练另用包含SciPy的独立环境：

    python3 -m venv solution/.venv-train
    solution/.venv-train/bin/python -m pip install -r solution/requirements-train.txt

只重训选定配置：

    PYTHONDONTWRITEBYTECODE=1 solution/.venv-train/bin/python -m solution.residual_ranker.extended_train --schemas lexical --l2 0.03 --output solution/runs/extended_lexical_training_reproduction

复现四组有限消融：

    PYTHONDONTWRITEBYTECODE=1 solution/.venv-train/bin/python -m solution.residual_ranker.extended_experiment --output solution/runs/handcrafted_extended_reproduction --test

31项测试通过；真实validation/test预测重放、全部候选重排、批次变化、822道冲突题特征矩阵逐位一致检查通过。45个冻结文件哈希、源码包哈希及此前两轮冻结哈希通过；模型源码和权重未在本轮test后更改。核验记录见[verification.json](../runs/handcrafted_extended_v2/verification.json)。

原group_ranker及已有特征实现保持原样；扩展保存在extended_*.py。上一版83.08351%权重及首次残差80.51392%的记录继续保留，旧入口submission.py仍可复现旧模型；本轮入口必须与新的lexical checkpoint配对。

[本轮报告](../runs/handcrafted_extended_v2/REPORT.md) · [官方评分](../runs/handcrafted_extended_v2/test/score.json) · [冻结源码与选定权重](../runs/handcrafted_extended_v2/source.tar) · [上一轮手工报告](../runs/residual_features_v1/REPORT.md)

完整复现需保留新checkpoint内的model.npz、config.json和asset_manifest.json。runs/默认被Git忽略，最终提交需要另外包含匹配权重、官方JSON和实验记录。

[规则审阅](COMPLIANCE_REVIEW.md) · [AI与复用披露](AI_AND_REUSE.md)。当前资产与代码仍需加入最终公开提交；新source.tar只是冻结快照，不包含所有交付材料。
