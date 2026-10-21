"""校准融合的官方预测入口；成员、特征和前向共用原融合实现。"""
from solution.group_ranker.submission_ensemble import predict_profiled as _predict


def predict_profiled(bank, arrays, checkpoint, *, include_scores=False):
    return _predict(bank, arrays, checkpoint, include_scores=include_scores, calibrated=True)


def predict(bank, arrays, checkpoint):
    return predict_profiled(bank, arrays, checkpoint)[0]
