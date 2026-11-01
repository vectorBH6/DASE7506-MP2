"""浅树数组、原生前向等价性与预测协议检查；只使用合成数据。"""
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import PROTOCOL, parse_answer
from solution.group_ranker.features import FEATURE_DIM, FEATURE_VERSIONS, feature_mask
from solution.group_ranker import test_group_ranker as fixtures
from solution.group_ranker.training import export_model
from solution.tree_ranker.model import MAX_TREES, MODEL, extract_trees, forward, validate_model
from solution.tree_ranker.submission import load_model, predict, predict_profiled


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MLPTests()
        self.fixture.setUp()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "checkpoint"
        self.config = {"protocol": PROTOCOL, "model": MODEL,
            "feature_dim": FEATURE_DIM, "feature_set": "text", "feature_version": FEATURE_VERSIONS["text"],
            "enabled_scalar_indices": list(range(32)), "depth": 3, "max_depth": 2, "tree_count": 2}
        self.model = {"split_features": np.array([[2048, 2064], [2053, 0]], dtype=np.int32),
            "borders": np.array([[.3, 2.], [0., 0.]], dtype=np.float64),
            "leaves": np.array([[0., .3, .4, .7], [-.1, .2, 0., 0.]], dtype=np.float64),
            "depth": np.array([2, 1], dtype=np.int32), "scale_bias": np.array([.8, -.1]),
            "mean": np.zeros(FEATURE_DIM, dtype=np.float32), "scale": np.ones(FEATURE_DIM, dtype=np.float32),
            "active": feature_mask("text")}
        export_model(self.directory, self.model, self.config)

    def test_scalar_reference_and_strict_border(self):
        x = np.zeros((7, FEATURE_DIM), dtype=np.float32)
        x[:, 2048] = [0, .3, .5, .5, 1., .1, .9]
        x[:, 2064] = [0, 2., 2., 3., 1., 3., 4.]
        x[:, 2053] = [-1., 0., 1., -1., 0., .5, .8]
        expected = []
        for row in x:
            value = 0.
            for i, depth in enumerate(self.model["depth"]):
                index = sum(int(row[self.model["split_features"][i, j]] > self.model["borders"][i, j]) << j
                            for j in range(int(depth)))
                value += self.model["leaves"][i, index]
            expected.append(value * .8 - .1)
        np.testing.assert_allclose(forward(x, self.model), expected, atol=1e-14)
        # 用恰好可表示的 0.5，核对等于阈值时走左侧。
        boundary = copy.deepcopy(self.model)
        boundary["borders"][0, 0] = .5
        row = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        row[0, 2048] = .5
        self.assertAlmostEqual(float(forward(row, boundary)[0]), -.18)

    def test_json_mapping_constant_tree_and_schema_rejection(self):
        document = {"features_info": {"float_features": [{"feature_index": 0, "flat_feature_index": 2048}]},
            "oblivious_trees": [{"splits": [{"split_type": "FloatFeature", "float_feature_index": 0, "border": .5}],
                                   "leaf_values": [-1., 2.]}, {"splits": None, "leaf_values": [.3]}],
            "scale_and_bias": [2., [.1]]}
        arrays = extract_trees(document)
        x = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        x[1, 2048] = 1.
        np.testing.assert_allclose(forward(x, arrays), [-1.3, 4.7])
        document["oblivious_trees"][0]["splits"][0]["split_type"] = "OneHotFeature"
        with self.assertRaises(ValueError):
            extract_trees(document)
        for key, value in (("depth", np.array([3, 1])), ("borders", np.full((2, 2), np.nan)),
                           ("split_features", np.full((2, 2), FEATURE_DIM))):
            broken = {**self.model, key: value}
            with self.assertRaises(ValueError):
                validate_model(broken, self.config)

    def test_permutations_batches_shortcuts_and_shape(self):
        f = self.fixture
        selected = predict([f.row], {"embeddings": f.embedding[None]}, self.directory)[0]
        expected = parse_answer(f.candidates[selected]["text"])
        for order in itertools.permutations(range(4)):
            row = {**f.row, "candidates": [f.candidates[j] for j in order]}
            chosen = predict([row], {"embeddings": f.embedding[list(order)][None]}, self.directory)[0]
            self.assertEqual(parse_answer(row["candidates"][chosen]["text"]), expected)
        none = {**f.row, "candidates": [{"text": "unknown"}] * 4}
        one = {**f.row, "candidates": [{"text": "unknown"}] * 3 + [{"text": "#### 2"}]}
        bank = [none, one] + [f.row] * 129
        result, profile = predict_profiled(bank, {"embeddings": np.tile(f.embedding, (len(bank), 1, 1))},
                                           self.directory, include_scores=True)
        self.assertEqual(result, [0, 3] + [selected] * 129)
        self.assertTrue(all(type(j) is int for j in result))
        self.assertEqual(len(profile["group_scores"]), len(bank))
        self.assertEqual(predict([], {"embeddings": np.empty((0, 4, 1024))}, None), [])
        self.assertEqual(len(predict([f.row], {"embeddings": np.zeros((1, 4, 1024))}, self.directory)), 1)
        for values in (np.zeros((2, 4, 1024)), np.full((1, 4, 1024), np.nan)):
            with self.assertRaises(ValueError):
                predict([f.row], {"embeddings": values}, self.directory)
        with self.assertRaises(ValueError):
            predict([{**f.row, "candidates": f.candidates[:3]}], {"embeddings": f.embedding[None]}, self.directory)

    def test_hash_and_feature_schema(self):
        path = self.directory / "config.json"
        config = {**self.config, "feature_set": "basic"}
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_model(self.directory)
        manifest_path = self.directory / "asset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"]["config.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_model(self.directory)

    def test_extended_budget_and_old_checkpoint(self):
        # 原格式不变：旧 checkpoint 可读，1000 棵可导出，超过预算则拒绝。
        original = load_model(self.directory)
        model = {name: value.copy() for name, value in original.items()}
        for name in ("split_features", "borders", "leaves"):
            model[name] = np.tile(model[name], (MAX_TREES // 2, 1))
        model["depth"] = np.tile(model["depth"], MAX_TREES // 2)
        config = {**self.config, "tree_count": MAX_TREES}
        export_model(self.directory, model, config)
        restored = load_model(self.directory)
        x = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        x[1, 2048] = 1.
        bias = model["scale_bias"][1]
        expected = (forward(x, original) - bias) * (MAX_TREES // 2) + bias
        np.testing.assert_allclose(forward(x, restored), expected, atol=1e-11)
        with self.assertRaises(ValueError):
            validate_model(model, {**config, "tree_count": MAX_TREES + 1})

    @unittest.skipUnless(importlib.util.find_spec("catboost"), "原生树复核需要统一环境中的 CatBoost")
    def test_native_conversion_and_question_groups(self):
        from solution.tree_ranker.train import checked_scores, fit, make_pool
        rng = np.random.default_rng(7506)
        features = np.zeros((12, 4, FEATURE_DIM), dtype=np.float32)
        features[:, :, :8] = rng.normal(size=(12, 4, 8))
        mask = np.array([[True, True, False, False], [True, True, True, False], [True] * 4] * 4)
        labels = np.argmax(np.where(mask, features[:, :, 0], -np.inf), axis=1)
        pool = make_pool(features, mask, labels)
        ids = np.asarray(pool.get_group_id_hash())
        self.assertEqual(len(np.unique(ids)), len(labels))
        self.assertEqual(int(np.sum(pool.get_label())), len(labels))
        model, scaler = fit(features, mask, labels, depth=3, iterations=5, seed=7506,
                            validation=(features[:3], mask[:3], labels[:3]))
        scores, _, error = checked_scores(model, scaler, features, mask)
        self.assertLess(error, 1e-10)
        np.testing.assert_array_equal(np.isfinite(scores), mask)


if __name__ == "__main__":
    unittest.main()
