"""共享三层 MLP：NumPy 前向、组 CE/成对排序反传，不读数据或 checkpoint。"""
import numpy as np

PARAMETERS = ("w1", "b1", "w2", "b2", "w3", "b3")
HIDDEN_SIZES = ([32, 16], [64, 16], [128, 32], [256, 64])  # 训练与载入共用有限容量范围。


def initialize(input_dim, hidden, rng):
    model = {}
    sizes = (input_dim, *hidden, 1)
    for layer, (left, right) in enumerate(zip(sizes, sizes[1:]), 1):
        gain = 2 if layer < 3 else 1
        model[f"w{layer}"] = (rng.standard_normal((left, right)) * np.sqrt(gain / left)).astype(np.float32)
        model[f"b{layer}"] = np.zeros(right, dtype=np.float32)
    return model


def forward(x, model, *, dropout=0.0, rng=None):
    """同时返回反传所需中间值；推理保持 dropout=0。"""
    h1 = np.maximum(x @ model["w1"] + model["b1"], 0)
    keep = None
    if dropout:
        keep = (rng.random(h1.shape) >= dropout).astype(np.float32) / (1 - dropout)
    used = h1 if keep is None else h1 * keep
    h2 = np.maximum(used @ model["w2"] + model["b2"], 0)
    scores = (h2 @ model["w3"] + model["b3"]).ravel()
    return scores, (h1, used, h2, keep)


def loss_gradient(x, mask, labels, model, rng, dropout=0.1, loss="ce"):
    scores, (h1, used, h2, keep) = forward(x, model, dropout=dropout, rng=rng)
    scores = np.where(mask, scores.reshape(mask.shape), -np.inf)
    rows = np.arange(len(labels))
    if loss == "ce":
        shifted = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        denominator = probabilities.sum(axis=1, keepdims=True)
        value = np.mean(np.log(denominator[:, 0]) - shifted[rows, labels])
        probabilities /= denominator
        probabilities[rows, labels] -= 1
        delta = probabilities / len(labels)
    elif loss == "pairwise":
        negatives = mask & (np.arange(mask.shape[1])[None, :] != labels[:, None])
        difference = scores - scores[rows, labels, None]
        divisor = np.maximum(1, negatives.sum(axis=1, keepdims=True))
        value = np.mean((np.logaddexp(0, difference) * negatives).sum(axis=1) / divisor[:, 0])
        delta = (np.exp(-np.logaddexp(0, -difference)) * negatives / divisor / len(labels)).astype(scores.dtype)
        delta[rows, labels] -= delta.sum(axis=1)  # 每题先对负组平均，再对题平均。
    else:
        raise ValueError(f"未知损失：{loss}")
    delta = delta.reshape(-1, 1)  # padding 梯度为零。
    gradients = {"w3": h2.T @ delta, "b3": delta.sum(axis=0)}
    delta = (delta @ model["w3"].T) * (h2 > 0)
    gradients.update(w2=used.T @ delta, b2=delta.sum(axis=0))
    delta = delta @ model["w2"].T
    if keep is not None:
        delta *= keep
    delta *= h1 > 0
    gradients.update(w1=x.T @ delta, b1=delta.sum(axis=0))
    return float(value), gradients
