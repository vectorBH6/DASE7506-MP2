"""Handcrafted extension of the unchanged basic answer-group feature builder."""
import numpy as np

from solution.group_ranker.features import (
    EMBED_DIM, FEATURE_DIM, FEATURE_VERSION, answer_groups, best_group,
    build_features as build_basic_features, standardize,
)
from solution.handcrafted_ranker.arithmetic import FEATURE_SETS, aggregate_features


def build_features(candidates, embeddings, groups, *, question=None, feature_set="basic"):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set != "basic" and not isinstance(question, str):
        raise ValueError("Handcrafted features require the question string")
    features = build_basic_features(candidates, embeddings, groups)
    if feature_set != "basic" and groups:
        e = np.asarray(embeddings, dtype=np.float64)
        order = sorted(range(len(candidates)), key=lambda j: (candidates[j]["text"], e[j].tobytes()))
        members = [[j for j in order if j in indices] for indices in groups.values()]
        features[:, 2064:2072] = aggregate_features(question, candidates, members)
        if feature_set == "math_interactions":
            # Share uses all four candidates, including unparseable ones.
            features[:, 2072] = features[:, 2048] * features[:, 2066]
            features[:, 2073] = features[:, 2048] * features[:, 2067]
        disabled = np.array([i for i in range(10) if i not in FEATURE_SETS[feature_set]], dtype=int)
        features[:, 2064 + disabled] = 0
    return features
