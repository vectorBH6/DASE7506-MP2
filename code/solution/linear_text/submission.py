"""L1 官方预测入口：固定启用文本 schema，复用共用线性推理。"""
from solution.group_ranker.submission_linear import predict_profiled as _predict


def predict_profiled(bank, arrays, checkpoint, *, include_scores=False):
    return _predict(bank, arrays, checkpoint, feature_set="text", include_scores=include_scores)


def predict(bank, arrays, checkpoint):
    return predict_profiled(bank, arrays, checkpoint)[0]
