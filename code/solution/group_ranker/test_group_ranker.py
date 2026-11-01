"""答案组排序的正确性检查，使用标准库 unittest。"""
import hashlib
import importlib.util
import itertools
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
from solution.group_ranker.evidence import arithmetic_counts, body_of, expression_value, numbers_in
from solution.group_ranker.features import ABLATIONS, FEATURE_VERSIONS, best_group, feature_mask, standardize
from solution.group_ranker.mlp import PARAMETERS, forward, initialize, loss_gradient
from solution.group_ranker.submission_mlp import load_model as load_mlp, predict as predict_mlp
from solution.group_ranker.training import export_model, fit_standardizer


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


class MLPTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"text": text + "\n#### " + answer} for text, answer in (
            ("Use 2 apples: <<1/2=0.5>>", "0.5"), ("Use 2 apples: $2/4=0.5$", "1/2"),
            ("Other plan: 1+1=2", "2"), ("A mistake: <<1+1=3>>", "3"))]
        self.row = {"id": "new-id", "question": "How many of 2 apples remain?", "candidates": self.candidates}
        self.embedding = np.random.default_rng(7506).normal(size=(4, EMBED_DIM)).astype(np.float32)

    def bundle(self, directory, mode="text", ablation="none"):
        enabled = feature_mask(mode, ablation)
        model = initialize(FEATURE_DIM, (32, 16), np.random.default_rng(7506))
        model.update(mean=np.zeros(FEATURE_DIM, dtype=np.float32),
                     scale=np.ones(FEATURE_DIM, dtype=np.float32), active=enabled)
        config = {"model": "mlp", "feature_version": FEATURE_VERSIONS[mode], "feature_set": mode,
                  "feature_dim": FEATURE_DIM, "hidden": [32, 16], "ablation": ablation,
                  "enabled_scalar_indices": np.flatnonzero(enabled[2048:]).tolist()}
        export_model(directory, model, config)
        return model

    def test_all_permutations_features_and_prediction(self):
        groups = answer_groups(self.candidates)
        with tempfile.TemporaryDirectory() as temporary:
            for mode in FEATURE_VERSIONS:
                directory = Path(temporary) / mode
                self.bundle(directory, mode)
                expected = build_features(self.candidates, self.embedding, groups,
                                          question=self.row["question"], feature_set=mode)
                chosen = predict_mlp([self.row], {"embeddings": self.embedding[None]}, directory)[0]
                answer = parse_answer(self.candidates[chosen]["text"])
                for order in itertools.permutations(range(4)):
                    reordered = [self.candidates[j] for j in order]
                    row = {**self.row, "candidates": reordered}
                    e = self.embedding[list(order)]
                    actual = build_features(reordered, e, answer_groups(reordered),
                                            question=row["question"], feature_set=mode)
                    np.testing.assert_array_equal(actual, expected)
                    result = predict_mlp([row], {"embeddings": e[None]}, directory)[0]
                    self.assertEqual(parse_answer(reordered[result]["text"]), answer)

    def test_batch_shortcuts_zero_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.bundle(directory)
            one = {**self.row, "candidates": [{"text": "unknown"}] * 3 + [{"text": "#### 2"}]}
            none = {**self.row, "candidates": [{"text": "unknown"}] * 4}
            bank = [none, one] + [self.row] * 129
            arrays = {"embeddings": np.tile(self.embedding, (len(bank), 1, 1))}
            result = predict_mlp(bank, arrays, directory)
            single = predict_mlp([self.row], {"embeddings": self.embedding[None]}, directory)[0]
            self.assertEqual(result, [0, 3] + [single] * 129)
            self.assertTrue(all(type(j) is int and 0 <= j < 4 for j in result))
            self.assertEqual(predict_mlp([], {"embeddings": np.empty((0, 4, 1024))}, None), [])
            zero = predict_mlp([self.row], {"embeddings": np.zeros((1, 4, 1024))}, directory)
            self.assertEqual(len(zero), 1)
            for invalid in (np.zeros((2, 4, 1024)), np.full((1, 4, 1024), np.nan)):
                with self.assertRaises(ValueError):
                    predict_mlp([self.row], {"embeddings": invalid}, directory)
            with self.assertRaises(ValueError):
                predict_mlp([{**self.row, "candidates": self.candidates[:3]}],
                            {"embeddings": self.embedding[None]}, directory)
            # 换同组返回代表不改变官方数值答案；不要求替换正文后特征不变。
            self.assertEqual(parse_answer(self.candidates[0]["text"]), parse_answer(self.candidates[1]["text"]))

    def test_ablation_scaler_and_inference_masks(self):
        rng = np.random.default_rng(7506)
        x = rng.normal(size=(3, 4, FEATURE_DIM)).astype(np.float32)
        mask = np.ones((3, 4), dtype=bool)
        with tempfile.TemporaryDirectory() as temporary:
            for ablation in ABLATIONS:
                enabled = feature_mask("text", ablation)
                scaler = fit_standardizer(x * enabled, mask)
                transformed = standardize(x, **scaler)
                np.testing.assert_array_equal(transformed[..., ~enabled], 0)
                directory = Path(temporary) / ablation
                self.bundle(directory, ablation=ablation)
                model = load_mlp(directory)
                np.testing.assert_array_equal(model["active"], enabled)
                chosen = predict_mlp([self.row], {"embeddings": self.embedding[None]}, directory)
                self.assertEqual(len(chosen), 1)
                if ablation == "no_embedding":
                    changed = predict_mlp([self.row], {"embeddings": np.zeros((1, 4, 1024))}, directory)
                    self.assertEqual(chosen, changed)
        self.assertEqual(int(feature_mask("text", "no_embedding").sum()), 24)

    def test_forward_matches_independent_scalar_calculation(self):
        rng = np.random.default_rng(7506)
        x = rng.normal(size=(7, 5))
        model = initialize(5, (4, 3), rng)
        actual, _ = forward(x, model)
        expected = []
        for row in x:
            value = row.tolist()
            for layer in range(1, 4):
                w, b = model[f"w{layer}"], model[f"b{layer}"]
                value = [sum(value[i] * float(w[i, j]) for i in range(len(value))) + float(b[j])
                         for j in range(len(b))]
                if layer < 3:
                    value = [max(0, v) for v in value]
            expected.append(value[0])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_deterministic_ties(self):
        np.testing.assert_array_equal(best_group(np.array([[0, .9e-7], [0, 1.1e-7]])), [0, 1])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.bundle(directory)
            with np.load(directory / "model.npz", allow_pickle=False) as archive:
                model = {key: archive[key].copy() for key in archive.files}
            for key in PARAMETERS:
                model[key].fill(0)
            config = json.loads((directory / "config.json").read_text())
            export_model(directory, model, config)
            chosen = predict_mlp([self.row], {"embeddings": self.embedding[None]}, directory)[0]
            self.assertEqual(parse_answer(self.candidates[chosen]["text"]), "1/2")

    def test_ce_pairwise_values_gradients_and_padding(self):
        rng = np.random.default_rng(7506)
        x = rng.normal(size=(8, 5))
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool)
        labels = np.array([1, 2])
        model = {key: v.astype(np.float64) for key, v in initialize(5, (4, 3), rng).items()}
        for loss in ("ce", "pairwise"):
            value, gradient = loss_gradient(x, mask, labels, model, rng, dropout=0, loss=loss)
            scores = forward(x, model)[0].reshape(mask.shape)
            expected = []
            for row, valid, target in zip(scores, mask, labels):
                if loss == "ce":
                    expected.append(np.logaddexp.reduce(row[valid]) - row[target])
                else:
                    expected.append(np.mean([np.logaddexp(0, row[j] - row[target])
                                             for j in np.flatnonzero(valid) if j != target]))
            self.assertAlmostEqual(value, np.mean(expected), places=12)
            for key in PARAMETERS:
                for index in range(min(3, model[key].size)):
                    original, epsilon = model[key].flat[index], 1e-5
                    model[key].flat[index] = original + epsilon
                    plus = loss_gradient(x, mask, labels, model, rng, dropout=0, loss=loss)[0]
                    model[key].flat[index] = original - epsilon
                    minus = loss_gradient(x, mask, labels, model, rng, dropout=0, loss=loss)[0]
                    model[key].flat[index] = original
                    self.assertAlmostEqual(gradient[key].flat[index], (plus - minus) / (2 * epsilon), places=6)
            changed = x.copy().reshape(2, 4, 5)
            changed[~mask] = 100
            _, actual = loss_gradient(changed.reshape(8, 5), mask, labels, model, rng, dropout=0, loss=loss)
            for key in PARAMETERS:
                np.testing.assert_allclose(actual[key], gradient[key], atol=1e-12)

    def test_checkpoint_hash_and_feature_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.bundle(directory)
            config_path = directory / "config.json"
            config = json.loads(config_path.read_text())
            config["feature_version"] = "incompatible-version"
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_mlp(directory)
            manifest_path = directory / "asset_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sha256"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_mlp(directory)
            for invalid in ("weight_shape", "nonfinite_weight", "zero_scale", "disabled_dimension"):
                model = self.bundle(directory)
                if invalid == "weight_shape":
                    model["w2"] = model["w2"][:-1]
                elif invalid == "nonfinite_weight":
                    model["w1"][0, 0] = np.nan
                elif invalid == "zero_scale":
                    model["scale"][0] = 0
                else:
                    model["active"][-1] = True
                np.savez(directory / "model.npz", **model)
                manifest = json.loads(manifest_path.read_text())
                manifest["sha256"]["model.npz"] = hashlib.sha256((directory / "model.npz").read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_mlp(directory)


class EvidenceTests(unittest.TestCase):
    def test_body_and_numeric_literals(self):
        self.assertEqual(body_of("body #### 2 #### 3"), "body #### 2 ")
        self.assertEqual(body_of(r"a \boxed{2} b \boxed{3} end"), r"a \boxed{2} b  end")
        self.assertEqual(numbers_in("1,000 and 0.5 and 1/2 and 1/0"), {1000, 0.5})

    def test_checked_failed_unknown_and_overlap(self):
        for text, expected in (("<<0.1+0.2=0.3>>", (1, 0, 0)),
                               ("$2×3=7$", (0, 1, 0)), ("$$6÷2=3$$", (1, 0, 0)),
                               ("1 + 2 = 3\n", (1, 0, 0)), ("$1/3=0.333333$", (1, 0, 0)),
                               ("<<x+3=5>>", (0, 0, 1)), ("$1/0=2$", (0, 0, 1)),
                               ("$2**3=8$", (0, 0, 1)), ("$1=1=1$", (0, 0, 1)),
                               ("<<$1+1=2$>>", (0, 0, 1)), ("x + 3 = 5", (0, 0, 0))):
            with self.subTest(text=text):
                self.assertEqual(arithmetic_counts(text), expected)
        self.assertEqual(arithmetic_counts("\n".join(["<<1=1>>"] * 70)), (64, 0, 0))

    def test_arithmetic_limits_and_unsafe_syntax(self):
        for expression in ("__import__('os')", "2**3", "4//2", "1/0", "1000000000001",
                           "0." + "1" * 24, "1e2", "+" * 13 + "1", "1+" * 64 + "1",
                           "1/(1000000000000*1000000000000)/10"):
            with self.subTest(expression=expression), self.assertRaises((ValueError, SyntaxError, ZeroDivisionError)):
                expression_value(expression)


if __name__ == "__main__":
    unittest.main()
