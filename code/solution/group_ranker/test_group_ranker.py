"""答案组排序的正确性检查，使用标准库 unittest。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import parse_answer
from solution.group_ranker.features import (
    EMBED_DIM, FEATURE_DIM, FEATURE_VERSION, answer_groups, build_features,
)
from solution.group_ranker.submission_linear import load_model, predict


class GroupRankerTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"text": f"reason {j}\n#### {a}"} for j, a in
                           enumerate(("0.5", "1/2", "2", "3"))]
        self.embedding = np.random.default_rng(7506).normal(size=(4, EMBED_DIM)).astype(np.float32)
        self.row = {"question": "unused by basic features", "candidates": self.candidates}

    def bundle(self, directory):
        arrays = {
            "weight": np.random.default_rng(7506).normal(size=FEATURE_DIM).astype(np.float32),
            "mean": np.zeros(FEATURE_DIM, dtype=np.float32),
            "scale": np.ones(FEATURE_DIM, dtype=np.float32),
            "active": np.ones(FEATURE_DIM, dtype=bool),
        }
        np.savez(directory / "model.npz", **arrays)
        (directory / "config.json").write_text(json.dumps({
            "feature_version": FEATURE_VERSION, "feature_dim": FEATURE_DIM,
        }), encoding="utf-8")
        hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                  for name in ("model.npz", "config.json")}
        (directory / "asset_manifest.json").write_text(json.dumps({"sha256": hashes}), encoding="utf-8")
        return arrays

    def test_equivalent_answers_and_reserved_features(self):
        groups = answer_groups(self.candidates)
        self.assertEqual(groups["1/2"], [0, 1])
        x = build_features(self.candidates, self.embedding, groups)
        self.assertEqual(x.shape, (3, FEATURE_DIM))
        np.testing.assert_array_equal(x[:, 2064:], 0)

    def test_permutation_invariance_of_group_features(self):
        permutation = [3, 1, 2, 0]
        reordered = [self.candidates[j] for j in permutation]
        original = build_features(self.candidates, self.embedding, answer_groups(self.candidates))
        actual = build_features(reordered, self.embedding[permutation], answer_groups(reordered))
        np.testing.assert_array_equal(original, actual)

    def test_zero_embeddings_and_missing_answers(self):
        candidates = [{"text": "no final answer"}] + self.candidates[1:]
        x = build_features(candidates, np.zeros((4, EMBED_DIM)), answer_groups(candidates))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:, :2048], 0)

    def test_prediction_is_independent_of_batch_and_permutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = self.bundle(directory)
            groups = answer_groups(self.candidates)
            x = build_features(self.candidates, self.embedding, groups)
            expected = list(groups)[int(np.argmax(np.clip(x, -8, 8) @ model["weight"]))]
            single = predict([self.row], {"embeddings": self.embedding[None]}, directory)[0]
            self.assertEqual(parse_answer(self.candidates[single]["text"]), expected)
            repeated = predict([self.row] * 130, {"embeddings": np.tile(self.embedding, (130, 1, 1))}, directory)
            self.assertEqual(repeated, [single] * 130)
            permutation = [3, 1, 2, 0]
            row = {"candidates": [self.candidates[j] for j in permutation]}
            chosen = predict([row], {"embeddings": self.embedding[permutation][None]}, directory)[0]
            self.assertEqual(parse_answer(row["candidates"][chosen]["text"]), expected)

    def test_shortcuts_and_shape_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.bundle(directory)
            none = {"candidates": [{"text": "unknown"}] * 4}
            one = {"candidates": [{"text": "unknown"}] * 3 + [{"text": "#### 2"}]}
            choices = predict([none, one], {"embeddings": np.zeros((2, 4, EMBED_DIM))}, directory)
            self.assertEqual(choices, [0, 3])
            self.assertTrue(all(type(j) is int for j in choices))
            self.assertEqual(predict([], {"embeddings": np.empty((0, 4, EMBED_DIM))}, None), [])
            with self.assertRaises(ValueError):
                predict([one], {"embeddings": np.zeros((2, 4, EMBED_DIM))}, directory)

    def test_modified_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.bundle(directory)
            with (directory / "config.json").open("a", encoding="utf-8") as file:
                file.write(" ")
            with self.assertRaises(ValueError):
                load_model(directory)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "训练梯度检查需要 SciPy")
    def test_objective_gradient_and_fold_preprocessing(self):
        from solution.group_ranker.train_linear import fit_standardizer, objective

        rng = np.random.default_rng(7506)
        x = rng.normal(size=(12, 7))
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
        labels, weight = np.array([1, 2, 0]), rng.normal(size=7)
        _, gradient = objective(weight, x, mask, labels, 0.01)
        numerical = np.empty(7)
        for j in range(7):
            delta = np.eye(7)[j] * 1e-5
            numerical[j] = (objective(weight + delta, x, mask, labels, 0.01)[0]
                            - objective(weight - delta, x, mask, labels, 0.01)[0]) / 2e-5
        np.testing.assert_allclose(gradient, numerical, atol=1e-7)
        values = np.zeros((2, 4, 1), dtype=np.float32)
        values[0, :2, 0], values[1, :4, 0] = 2, 10
        model = fit_standardizer(values, np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool))
        self.assertAlmostEqual(float(model["mean"][0]), 6.0)  # 每题等权，而非每组等权。


if __name__ == "__main__":
    unittest.main()
