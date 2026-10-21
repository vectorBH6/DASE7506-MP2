"""B1：先选最多票组，仅平票时按答案组余弦中心性决定。"""
from solution.baselines.submission_group_center import select_groups


def predict(bank, arrays, checkpoint):
    return select_groups(bank, arrays, prefer_votes=True)
