"""单层集合编码器：NumPy 前向/反传；门控对照与注意力参数完全相同。"""
import numpy as np

WIDTH, HEADS, FF_DIM = 64, 2, 128
MODES = ("gated", "attention")


def shapes(input_dim, width=WIDTH, ff_dim=FF_DIM):
    result = {"inp_w": (input_dim, width), "inp_b": (width,)}
    for name in ("q", "k", "v", "out"):
        result.update({name + "_w": (width, width), name + "_b": (width,)})
    for name in ("ln1", "ln2"):
        result.update({name + "_g": (width,), name + "_b": (width,)})
    result.update(ff1_w=(width, ff_dim), ff1_b=(ff_dim,), ff2_w=(ff_dim, width),
                  ff2_b=(width,), score_w=(width, 1), score_b=(1,))
    return result


def initialize(input_dim, rng, width=WIDTH, ff_dim=FF_DIM):
    model = {}
    for name, shape in shapes(input_dim, width, ff_dim).items():
        if name.endswith("_w"):
            gain = 2 if name in ("inp_w", "ff1_w") else 1
            model[name] = (rng.normal(size=shape) * np.sqrt(gain / shape[0])).astype(np.float32)
        else:
            model[name] = np.full(shape, float(name.endswith("_g")), dtype=np.float32)
    return model


def norm(x, gain, bias):
    centered = x - x.mean(axis=-1, keepdims=True)
    inverse = 1 / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    unit = centered * inverse
    return unit * gain + bias, (unit, inverse)


def norm_backward(dy, cache, gain):
    unit, inverse = cache
    derivative = dy * gain
    dx = inverse * (derivative - derivative.mean(axis=-1, keepdims=True)
                    - unit * (derivative * unit).mean(axis=-1, keepdims=True))
    return dx, (dy * unit).sum(axis=(0, 1)), dy.sum(axis=(0, 1))


def forward(x, mask, model, mode, *, heads=HEADS, dropout=0.0, rng=None):
    if (mode not in MODES or x.ndim != 3 or x.shape[:2] != mask.shape
            or not mask.any(axis=1).all()):
        raise ValueError("无效模式、特征形状或空答案组")
    batch, groups, _ = x.shape
    width = model["inp_b"].size
    if width % heads:
        raise ValueError("隐藏维数须被注意力头数整除")
    valid = mask[..., None]
    h = np.maximum(x @ model["inp_w"] + model["inp_b"], 0) * valid
    q, k, v = [(h @ model[name + "_w"] + model[name + "_b"])
               .reshape(batch, groups, heads, width // heads).transpose(0, 2, 1, 3)
               for name in ("q", "k", "v")]
    scale = np.sqrt(width // heads)
    if mode == "attention":
        logits = np.where(mask[:, None, None, :], (q @ k.swapaxes(-1, -2)) / scale, -np.inf)
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        mixed = weights @ v  # 在有效答案组之间交换 value；padding 不参与 softmax。
    else:
        gate = (q * k).sum(axis=-1, keepdims=True) / scale
        weights = np.exp(-np.logaddexp(0, -gate))
        mixed = weights * v  # 相同 Q/K/V/O 参数，只做本组门控，不交换各组 value。
    context = mixed.transpose(0, 2, 1, 3).reshape(batch, groups, width)

    def drop(value):
        keep = (rng.random(value.shape) >= dropout).astype(value.dtype) / (1 - dropout) if dropout else 1.0
        return value * keep, keep

    branch, keep1 = drop(context @ model["out_w"] + model["out_b"])
    y, cache1 = norm(h + branch, model["ln1_g"], model["ln1_b"])
    y *= valid
    u = np.maximum(y @ model["ff1_w"] + model["ff1_b"], 0)
    branch, keep2 = drop(u @ model["ff2_w"] + model["ff2_b"])
    z, cache2 = norm(y + branch, model["ln2_g"], model["ln2_b"])
    z *= valid
    scores = np.where(mask, (z @ model["score_w"])[..., 0] + model["score_b"][0], -np.inf)
    cache = (x, valid, h, q, k, v, weights, context, keep1, cache1, y, u, keep2, cache2, z)
    return scores, cache


def loss_gradient(x, mask, labels, model, mode, rng, *, dropout=0.1, heads=HEADS):
    scores, cache = forward(x, mask, model, mode, dropout=dropout, rng=rng, heads=heads)
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    denominator = probabilities.sum(axis=1, keepdims=True)
    rows = np.arange(len(labels))
    loss = np.mean(np.log(denominator[:, 0]) - shifted[rows, labels])
    probabilities /= denominator
    probabilities[rows, labels] -= 1
    delta = probabilities / len(labels)
    x, valid, h, q, k, v, weights, context, keep1, cache1, y, u, keep2, cache2, z = cache
    gradients = {}

    def linear_backward(name, values, derivative):
        gradients[name + "_w"] = values.reshape(-1, values.shape[-1]).T @ derivative.reshape(-1, derivative.shape[-1])
        gradients[name + "_b"] = derivative.sum(axis=(0, 1))
        return derivative @ model[name + "_w"].T

    dz = linear_backward("score", z, delta[..., None])
    residual, gradients["ln2_g"], gradients["ln2_b"] = norm_backward(dz * valid, cache2, model["ln2_g"])
    du = linear_backward("ff2", u, residual * keep2) * (u > 0)
    dy = residual + linear_backward("ff1", y, du)
    residual, gradients["ln1_g"], gradients["ln1_b"] = norm_backward(dy * valid, cache1, model["ln1_g"])
    dc = linear_backward("out", context, residual * keep1)
    batch, groups, width = dc.shape
    dc = dc.reshape(batch, groups, heads, width // heads).transpose(0, 2, 1, 3)
    if mode == "attention":
        dw = dc @ v.swapaxes(-1, -2)
        dv = weights.swapaxes(-1, -2) @ dc
        dl = weights * (dw - (dw * weights).sum(axis=-1, keepdims=True)) / np.sqrt(width // heads)
        dq, dk = dl @ k, dl.swapaxes(-1, -2) @ q
    else:
        dg = (dc * v).sum(axis=-1, keepdims=True) * weights * (1 - weights) / np.sqrt(width // heads)
        dq, dk, dv = dg * k, dg * q, dc * weights
    dh = residual.copy()
    for name, derivative in zip(("q", "k", "v"), (dq, dk, dv)):
        derivative = derivative.transpose(0, 2, 1, 3).reshape(batch, groups, width)
        dh += linear_backward(name, h, derivative)
    linear_backward("inp", x, dh * (h > 0) * valid)
    return float(loss), gradients
