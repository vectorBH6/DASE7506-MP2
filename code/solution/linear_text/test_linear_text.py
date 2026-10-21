"""文本线性入口检查；使用合成输入，不读取任何课程参考答案。"""
import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import parse_answer
from solution.group_ranker.features import FEATURE_DIM, FEATURE_VERSIONS, feature_mask
from solution.group_ranker.submission_linear import load_model
from solution.group_ranker.training import export_model
from solution.linear_text.submission import predict, predict_profiled


class LinearTextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name)
        weight = np.zeros(FEATURE_DIM, dtype=np.float32)
        weight[2064] = 1  # 只读取新增的字符长度统计，防止入口误用基础 schema。
        model = {"weight": weight, "mean": np.zeros(FEATURE_DIM, dtype=np.float32),
                 "scale": np.ones(FEATURE_DIM, dtype=np.float32), "active": feature_mask("text")}
        config = {"model": "linear", "feature_set": "text", "feature_version": FEATURE_VERSIONS["text"],
                  "feature_dim": FEATURE_DIM, "enabled_scalar_indices": list(range(32))}
        export_model(self.checkpoint, model, config)
        self.row = {"id": "unseen", "question": "How many of 2 apples remain?",
                    "candidates": [{"text": text} for text in
                        ("x\n#### 1", "a much longer explanation about apples\n#### 2", "unknown", "y\n#### 3")]}
        self.embedding = np.random.default_rng(7506).normal(size=(4, 1024)).astype(np.float32)

    def test_new_feature_is_used_and_checkpoint_schema_is_explicit(self):
        choices, profile = predict_profiled([self.row], {"embeddings": self.embedding[None]},
                                            self.checkpoint, include_scores=True)
        self.assertEqual(choices, [1])
        self.assertEqual(len(profile["group_scores"][0]), 3)
        load_model(self.checkpoint, feature_set="text")
        with self.assertRaises(ValueError):
            load_model(self.checkpoint)  # 基础入口不能静默加载文本权重。

    def test_all_permutations_batches_and_boundaries(self):
        for order in itertools.permutations(range(4)):
            row = {**self.row, "candidates": [self.row["candidates"][j] for j in order]}
            choice = predict([row], {"embeddings": self.embedding[list(order)][None]}, self.checkpoint)[0]
            self.assertEqual(parse_answer(row["candidates"][choice]["text"]), "2")
        none = {**self.row, "candidates": [{"text": "unknown"}] * 4}
        one = {**self.row, "candidates": [{"text": "unknown"}] * 3 + [{"text": "#### 2"}]}
        bank = [none, one] + [self.row] * 129
        result = predict(bank, {"embeddings": np.tile(self.embedding, (len(bank), 1, 1))}, self.checkpoint)
        self.assertEqual(result, [0, 3] + [1] * 129)
        self.assertTrue(all(type(j) is int for j in result))
        self.assertEqual(predict([], {"embeddings": np.empty((0, 4, 1024))}, None), [])
        self.assertEqual(predict([self.row], {"embeddings": np.zeros((1, 4, 1024))}, self.checkpoint), [1])
        for invalid in (np.zeros((2, 4, 1024)), np.full((1, 4, 1024), np.nan)):
            with self.assertRaises(ValueError):
                predict([self.row], {"embeddings": invalid}, self.checkpoint)

    def test_asset_integrity_and_scalar_mask(self):
        config_path = self.checkpoint / "config.json"
        config = json.loads(config_path.read_text())
        config["enabled_scalar_indices"] = list(range(16))
        config_path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            load_model(self.checkpoint, feature_set="text")
        manifest_path = self.checkpoint / "asset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_model(self.checkpoint, feature_set="text")


if __name__ == "__main__":
    unittest.main()
