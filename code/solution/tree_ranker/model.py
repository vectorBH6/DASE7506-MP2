"""将纯数值对称树导出为数组；预测只依赖 NumPy。"""
import numpy as np

from solution.group_ranker.features import FEATURE_DIM, feature_mask

MODEL = "oblivious-tree-v1"
MAX_TREES = 1000  # 允许延长训练预算；旧的 300 棵 checkpoint 仍使用同一数组格式。


def extract_trees(document):
    info = document["features_info"]
    if info.get("categorical_features") or info.get("text_features") or info.get("embedding_features"):
        raise ValueError("本实验只导出既有数值特征的对称树")
    mapping = {f["feature_index"]: f["flat_feature_index"] for f in info["float_features"]}
    trees = document["oblivious_trees"]
    depth = np.array([len(tree.get("splits") or []) for tree in trees], dtype=np.int32)
    if not len(depth) or depth.max() > 5:
        raise ValueError("只支持深度不超过 5 的非空树集合")
    width = max(1, int(depth.max()))
    indices = np.zeros((len(trees), width), dtype=np.int32)
    borders = np.zeros_like(indices, dtype=np.float64)
    leaves = np.zeros((len(trees), 1 << width), dtype=np.float64)
    for i, tree in enumerate(trees):
        values = np.asarray(tree["leaf_values"], dtype=np.float64)
        if values.shape != (1 << int(depth[i]),):
            raise ValueError("只支持单输出树，叶子数量须与深度一致")
        leaves[i, :len(values)] = values
        for level, split in enumerate(tree.get("splits") or []):
            if split["split_type"] != "FloatFeature":
                raise ValueError("不支持分类或 CTR 分裂")
            indices[i, level] = mapping[split["float_feature_index"]]
            borders[i, level] = split["border"]
    scale, bias = document["scale_and_bias"]
    if len(bias) != 1:
        raise ValueError("只支持单输出模型")
    return {"split_features": indices, "borders": borders, "leaves": leaves,
            "depth": depth, "scale_bias": np.array([scale, bias[0]], dtype=np.float64)}


def validate_model(model, config):
    count, width = config["tree_count"], config["max_depth"]
    if (type(count) is not int or not 1 <= count <= MAX_TREES
            or type(width) is not int or not 1 <= width <= config["depth"] <= 5):
        raise ValueError("树数量或深度超出预定范围")
    shapes = {"split_features": (count, width), "borders": (count, width),
              "leaves": (count, 1 << width), "depth": (count,), "scale_bias": (2,),
              "mean": (FEATURE_DIM,), "scale": (FEATURE_DIM,), "active": (FEATURE_DIM,)}
    if set(model) != set(shapes) or any(model[k].shape != s for k, s in shapes.items()):
        raise ValueError("树 checkpoint 数组结构不符")
    if any(not np.isfinite(value).all() for value in model.values()):
        raise ValueError("树 checkpoint 包含非有限数值")
    indices, depth = model["split_features"], model["depth"]
    if (not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(depth.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= FEATURE_DIM)
            or np.any(depth < 0) or np.any(depth > width)):
        raise ValueError("分裂特征或树深度非法")
    if (model["active"].dtype != np.bool_ or np.any(model["scale"] <= 0)
            or np.any(model["active"] & ~feature_mask("text"))):
        raise ValueError("标准化参数或特征掩码非法")
    used = np.arange(width)[None, :] < depth[:, None]
    if np.any(~model["active"][indices[used]]):
        raise ValueError("树使用了关闭的特征")


def forward(features, model):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
        raise ValueError("树前向需要有限的二维特征矩阵")
    count, width = model["split_features"].shape
    leaf = np.zeros((len(x), count), dtype=np.int32)
    # 每层的判断是一位二进制数；严格 > 边界，与 CatBoost 数值分裂一致。
    for level in range(width):
        take_right = ((x[:, model["split_features"][:, level]] > model["borders"][:, level])
                      & (level < model["depth"]))
        leaf |= take_right.astype(np.int32) << level
    values = model["leaves"][np.arange(count)[None, :], leaf].sum(axis=1)
    return values * model["scale_bias"][0] + model["scale_bias"][1]
