"""温度拟合、校准融合与官方接口检查；只使用合成数据。"""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import parse_answer
from solution.evaluate import confidence_summary
from solution.calibrated_ensemble.calibrate import fit_temperature
from solution.calibrated_ensemble.submission import predict, predict_profiled
from solution.group_ranker import test_group_ranker as fixtures
from solution.group_ranker.features import answer_groups, best_group, build_features, standardize
from solution.group_ranker.mlp import forward
from solution.group_ranker.submission_ensemble import average_scores, create_bundle, digest, load_model


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MLPTests()
        self.fixture.setUp()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.linear, self.mlp = root / "linear", root / "mlp"
        self.linear.mkdir()
        fixtures.GroupRankerTests().bundle(self.linear)
        self.fixture.bundle(self.mlp)
        # 两个成员故意使用不同 scaler，核对共用特征路径仍与独立计算相同。
        for index, directory in enumerate((self.linear, self.mlp)):
            with np.load(directory / "model.npz", allow_pickle=False) as archive:
                model = {key: archive[key].copy() for key in archive.files}
            model["mean"][:] = index * 0.3 - 0.1
            model["scale"][:] = index * 0.2 + 0.9
            np.savez(directory / "model.npz", **model)
            self.rehash(directory)
        self.temperatures = [2.0, 0.7]
        self.checkpoint = root / "calibrated"
        self.asset_bytes = create_bundle(self.linear, self.mlp, self.checkpoint,
                                         temperatures=self.temperatures)

    @staticmethod
    def rehash(directory):
        path = directory / "asset_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["sha256"] = {name: digest(directory / name) for name in manifest["sha256"]}
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_temperature_fit_known_optimum_and_invalid_inputs(self):
        scores = np.tile([np.log(9.0), 0.0, -np.inf], (4, 1))
        fitted = fit_temperature(scores, np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(fitted["temperature"], 2.0, places=7)
        self.assertLess(fitted["nll_after"], fitted["nll_before"])
        self.assertAlmostEqual(fitted["final_gradient"], 0.0, places=10)
        self.assertEqual(fit_temperature(scores, np.zeros(4, dtype=int))["boundary"], "upper_beta")
        self.assertEqual(fit_temperature(scores, np.ones(4, dtype=int))["boundary"], "lower_beta")
        for bad_scores, labels in ((scores, [0, 0, 0, 2]), (scores, [0., 0., 0., 1.]),
                                   (scores, [-1, 0, 0, 1]), (scores[:0], []),
                                   (np.full((4, 3), np.nan), [0] * 4),
                                   (np.full((4, 3), -np.inf), [0] * 4)):
            with self.assertRaises(ValueError):
                fit_temperature(bad_scores, np.asarray(labels))

    def test_probability_formula_padding_and_temperature_validation(self):
        left, right = np.array([4., 3., -np.inf]), np.array([0., 2., -np.inf])
        expected = []
        for values, temperature in zip((left, right), self.temperatures):
            p = np.exp((values - values.max()) / temperature)
            expected.append(p / p.sum())
        np.testing.assert_allclose(np.exp(average_scores(left, right, temperatures=self.temperatures)),
                                   (expected[0] + expected[1]) / 2)
        np.testing.assert_array_equal(average_scores(left, right),
                                      average_scores(left, right, temperatures=[1., 1.]))
        for temperatures in ([0., 1.], [-1., 1.], [np.nan, 1.], [np.inf, 1.], [1.]):
            with self.assertRaises(ValueError):
                average_scores(left, right, temperatures=temperatures)
        measured = confidence_summary([[np.log(3.), 0.], [0., np.log(3.)]],
            [{"0": [0], "1": [1]}] * 2, ["0", "0"], np.array([True, False]), np.ones(2, dtype=bool))
        calibration = measured["calibration"]["conflict"]
        self.assertAlmostEqual(calibration["brier"], 0.625)
        self.assertAlmostEqual(calibration["ece_10_bins"], 0.25)
        self.assertAlmostEqual(calibration["nll"], -(np.log(.75) + np.log(.25)) / 2)

    def test_shared_runtime_matches_separate_features_and_scalers(self):
        f = self.fixture
        # 原入口只能加载原 schema，不能静默把校准权重当成温度 1。
        with self.assertRaises(ValueError):
            load_model(self.checkpoint)
        original = self.checkpoint.parent / "original"
        create_bundle(self.linear, self.mlp, original)
        linear, mlp = load_model(original)
        groups = answer_groups(f.candidates)
        basic = build_features(f.candidates, f.embedding, groups)
        text = build_features(f.candidates, f.embedding, groups,
                              question=f.row["question"], feature_set="text")
        left = standardize(basic, linear["mean"], linear["scale"], linear["active"]) @ linear["weight"]
        right = forward(standardize(text, mlp["mean"], mlp["scale"], mlp["active"]), mlp)[0]
        expected = average_scores(left, right, temperatures=self.temperatures)
        choices, profile = predict_profiled([f.row], {"embeddings": f.embedding[None]},
                                            self.checkpoint, include_scores=True)
        np.testing.assert_allclose(profile["group_scores"][0], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(choices, [min(list(groups.values())[int(best_group(expected))])])

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
        choices = predict(bank, {"embeddings": np.tile(f.embedding, (len(bank), 1, 1))}, self.checkpoint)
        self.assertEqual(choices, [0, 3] + [chosen] * 129)
        self.assertTrue(all(type(j) is int for j in choices))
        self.assertEqual(predict([], {"embeddings": np.empty((0, 4, 1024))}, None), [])
        self.assertEqual(len(predict([f.row], {"embeddings": np.zeros((1, 4, 1024))}, self.checkpoint)), 1)
        for values in (np.zeros((2, 4, 1024)), np.full((1, 4, 1024), np.nan)):
            with self.assertRaises(ValueError):
                predict([f.row], {"embeddings": values}, self.checkpoint)
        with self.assertRaises(ValueError):
            predict([{**f.row, "candidates": f.candidates[:3]}], {"embeddings": f.embedding[None]}, self.checkpoint)

    def test_assets_hash_schema_and_size(self):
        self.assertEqual(self.asset_bytes, sum(p.stat().st_size for p in self.checkpoint.rglob("*") if p.is_file()))
        self.assertLess(self.asset_bytes, 16 * 2**20)
        path = self.checkpoint / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["temperatures"] = [0., 1.]
        path.write_text(json.dumps(config), encoding="utf-8")
        f = self.fixture
        with self.assertRaises(ValueError):
            predict([f.row], {"embeddings": f.embedding[None]}, self.checkpoint)
        self.rehash(self.checkpoint)
        with self.assertRaises(ValueError):
            predict([f.row], {"embeddings": f.embedding[None]}, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
