"""等权融合检查：概率公式、共用特征的等价性、输入协议及自包含资产。"""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import parse_answer
from solution.group_ranker import test_group_ranker as fixtures
from solution.group_ranker.features import answer_groups, best_group, build_features, standardize
from solution.group_ranker.mlp import forward
from solution.group_ranker.submission_ensemble import (
    average_scores, create_bundle, digest, load_model, predict, predict_profiled,
)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MLPTests()
        self.fixture.setUp()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        linear, mlp = root / "linear", root / "mlp"
        linear.mkdir()
        fixtures.GroupRankerTests().bundle(linear)
        self.fixture.bundle(mlp)
        self.checkpoint = root / "ensemble"
        self.asset_bytes = create_bundle(linear, mlp, self.checkpoint)

    def test_probability_average_not_logit_average(self):
        left, right = np.array([100., 99., 0.]), np.array([0., 0., 10.])
        expected = []
        for values in (left, right):
            p = np.exp(values - values.max())
            expected.append(p / p.sum())
        actual = average_scores(left, right)
        np.testing.assert_allclose(np.exp(actual), (expected[0] + expected[1]) / 2)
        self.assertEqual(int(best_group(actual)), 2)
        self.assertEqual(int(best_group((left + right) / 2)), 0)
        padded = np.array([[1000., 0., -np.inf], [-1000., -1001., -np.inf]])
        result = average_scores(padded, padded)
        self.assertTrue(np.isneginf(result[:, -1]).all())
        np.testing.assert_allclose(np.exp(result).sum(axis=1), 1)
        for invalid in (np.array([np.nan, 0., 0.]), np.array([np.inf, 0., 0.]),
                        np.array([-np.inf, -np.inf, -np.inf]), np.zeros(4)):
            with self.assertRaises(ValueError):
                average_scores(invalid, right)

    def test_matches_separate_member_features_and_scalers(self):
        f = self.fixture
        linear, mlp = load_model(self.checkpoint)
        groups = answer_groups(f.candidates)
        basic = build_features(f.candidates, f.embedding, groups)
        text = build_features(f.candidates, f.embedding, groups, question=f.row["question"], feature_set="text")
        left = standardize(basic, linear["mean"], linear["scale"], linear["active"]) @ linear["weight"]
        right = forward(standardize(text, mlp["mean"], mlp["scale"], mlp["active"]), mlp)[0]
        expected = average_scores(left, right)
        choices, profile = predict_profiled([f.row], {"embeddings": f.embedding[None]}, self.checkpoint, include_scores=True)
        np.testing.assert_allclose(profile["group_scores"][0], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(choices[0], min(list(groups.values())[int(best_group(expected))]))

    def test_permutations_batches_and_boundary_protocol(self):
        f = self.fixture
        chosen = predict([f.row], {"embeddings": f.embedding[None]}, self.checkpoint)[0]
        expected = parse_answer(f.candidates[chosen]["text"])
        for order in itertools.permutations(range(4)):
            row = {**f.row, "candidates": [f.candidates[j] for j in order]}
            result = predict([row], {"embeddings": f.embedding[list(order)][None]}, self.checkpoint)[0]
            self.assertEqual(parse_answer(row["candidates"][result]["text"]), expected)
        none = {**f.row, "candidates": [{"text": "unknown"}] * 4}
        one = {**f.row, "candidates": [{"text": "unknown"}] * 3 + [{"text": "#### 2"}]}
        bank = [none, one] + [f.row] * 129
        result = predict(bank, {"embeddings": np.tile(f.embedding, (len(bank), 1, 1))}, self.checkpoint)
        self.assertEqual(result, [0, 3] + [chosen] * 129)
        self.assertTrue(all(type(j) is int for j in result))
        self.assertEqual(predict([], {"embeddings": np.empty((0, 4, 1024))}, None), [])
        self.assertEqual(len(predict([f.row], {"embeddings": np.zeros((1, 4, 1024))}, self.checkpoint)), 1)
        for e in (np.zeros((2, 4, 1024)), np.full((1, 4, 1024), np.nan)):
            with self.assertRaises(ValueError):
                predict([f.row], {"embeddings": e}, self.checkpoint)
        with self.assertRaises(ValueError):
            predict([{**f.row, "candidates": f.candidates[:3]}], {"embeddings": f.embedding[None]}, self.checkpoint)

    def test_bundle_hash_fixed_recipe_and_complete_asset_count(self):
        self.assertEqual(self.asset_bytes, sum(p.stat().st_size for p in self.checkpoint.rglob("*") if p.is_file()))
        self.assertLess(self.asset_bytes, 16 * 2**20)
        path = self.checkpoint / "config.json"
        config = json.loads(path.read_text())
        config["weights"] = [0.4, 0.6]
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            load_model(self.checkpoint)
        manifest_path = self.checkpoint / "asset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"]["config.json"] = digest(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_model(self.checkpoint)


if __name__ == "__main__":
    unittest.main()
