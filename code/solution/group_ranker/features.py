"""答案组共享特征：只依赖候选与固定 embedding，不读取文件或标签。"""
import numpy as np

from common import parse_answer

FEATURE_VERSION = "answer-group-basic-v1"
EMBED_DIM = 1024
FEATURE_DIM = 2088  # 1024 组均值 + 1024 题内交互 + 40 标量


def answer_groups(candidates):
    """忽略 None，以官方规范数值分组；字典序使组顺序不依赖候选排列。"""
    groups = {}
    for j, candidate in enumerate(candidates):
        answer = parse_answer(candidate["text"])
        if answer is not None:
            groups.setdefault(answer, []).append(j)
    return {answer: groups[answer] for answer in sorted(groups)}


def build_features(candidates, embeddings, groups):
    e = np.asarray(embeddings, dtype=np.float64)
    if len(candidates) != 4 or e.shape != (4, EMBED_DIM) or not np.isfinite(e).all():
        raise ValueError("每题需要 4 个候选及有限的 [4, 1024] embedding")
    if not groups:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    norms = np.linalg.norm(e, axis=1)
    unit = e / np.maximum(norms[:, None], 1e-12)
    cosine = unit @ unit.T
    # 固定求和顺序，避免候选重排改变浮点聚合结果。
    order = sorted(range(4), key=lambda j: (candidates[j]["text"], e[j].tobytes()))
    members = [[j for j in order if j in indices] for indices in groups.values()]
    counts = np.array([len(indices) for indices in members])
    valid = [j for j in order if any(j in indices for indices in members)]
    means = np.stack([unit[indices].mean(axis=0) for indices in members])
    context = means.mean(axis=0)  # 每个答案组等权，票数另用显式特征表达。
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p)) / np.log(4)
    features = np.zeros((len(groups), FEATURE_DIM), dtype=np.float32)

    for g, indices in enumerate(members):
        others = [j for j in valid if j not in indices]
        inside = cosine[np.ix_(indices, indices)]
        inside = inside[~np.eye(len(indices), dtype=bool)]
        cross = cosine[np.ix_(indices, others)].ravel()
        mean_norm = np.linalg.norm(means[g])
        context_cos = means[g] @ context / max(mean_norm * np.linalg.norm(context), 1e-12)
        scalar = [
            counts[g] / 4, p[g], np.log(counts[g]), len(groups) / 4, len(valid) / 4,
            (counts[g] - max(np.delete(counts, g), default=0)) / 4,
            (np.sum(counts == counts[g]) - 1) / 3, entropy,
            inside.mean() if inside.size else 0, inside.min() if inside.size else 0,
            cross.mean() if cross.size else 0, cross.max() if cross.size else 0,
            cross.min() if cross.size else 0, mean_norm, context_cos, norms[indices].mean(),
        ]
        features[g, :EMBED_DIM] = means[g]
        features[g, EMBED_DIM:2 * EMBED_DIM] = means[g] * context
        features[g, 2 * EMBED_DIM:2 * EMBED_DIM + len(scalar)] = scalar
    # 后 24 维预留给文本/算术实验，当前固定为零，不引入未验证的复杂度。
    return features


def standardize(features, mean, scale, active):
    """训练和推理共享变换；常量维关闭，极端值裁剪。"""
    return np.clip((features - mean) / scale, -8, 8) * active


def best_group(scores):
    """组已按答案字符串排序；接近平分时取首组，保持确定性。"""
    return np.argmax(scores >= np.max(scores, axis=-1, keepdims=True) - 1e-7, axis=-1)
