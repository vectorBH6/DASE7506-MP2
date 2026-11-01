"""B2：按有效候选的余弦中心性给答案组排序，无需训练。"""
import numpy as np

from solution.group_ranker.features import EMBED_DIM, answer_groups, best_group


def select_groups(bank, arrays, *, prefer_votes):
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float64)
    if embeddings.shape != (len(bank), 4, EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("需要与 bank 对齐的有限 [N, 4, 1024] embedding")
    choices = []
    for row, embedding in zip(bank, embeddings):
        candidates = row["candidates"]
        if len(candidates) != 4:
            raise ValueError("每题必须有四个候选")
        groups = answer_groups(candidates)
        if not groups:
            choices.append(0)
            continue
        if len(groups) == 1:
            choices.append(int(min(next(iter(groups.values())))))
            continue
        # 排除无法解析的候选，并固定聚合顺序，保证候选重排后答案一致。
        valid = sorted((j for indices in groups.values() for j in indices),
                       key=lambda j: (candidates[j]["text"], embedding[j].tobytes()))
        unit = embedding[valid]
        unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
        cosine = unit @ unit.T
        np.fill_diagonal(cosine, 0)  # 排除自身；零向量也不能直接减去 1。
        centrality = np.zeros(4)
        centrality[valid] = cosine.sum(axis=1) / (len(valid) - 1)
        scores = np.array([centrality[[j for j in valid if j in indices]].mean()
                           for indices in groups.values()])
        if prefer_votes:
            counts = np.array([len(indices) for indices in groups.values()])
            scores[counts < counts.max()] = -np.inf  # B1 只比较并列最多票的组。
        winner = int(best_group(scores))
        choices.append(int(min(list(groups.values())[winner])))
    return choices


def predict(bank, arrays, checkpoint):
    return select_groups(bank, arrays, prefer_votes=False)
