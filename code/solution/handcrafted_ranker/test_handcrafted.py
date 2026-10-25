"""Arithmetic correctness, schema compatibility and actual inference tests."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import parse_answer
from solution.handcrafted_ranker.features import FEATURE_DIM, answer_groups, build_features
from solution.handcrafted_ranker.arithmetic import (
    FEATURE_SETS, aggregate_features, candidate_features, checked_equations,
    evaluate_expression, feature_version, scalar_indices,
)
from solution.handcrafted_ranker.submission_linear import predict


class HandcraftedTests(unittest.TestCase):
    def test_expression_grammar(self):
        for text, expected in [("3(18)", 54), ("-(2+3)/2", Fraction(-5, 2)),
                               ("1/4+1/4", Fraction(1, 2)), ("2*(3+4)-5", 9)]:
            self.assertEqual(evaluate_expression(text)[0], expected)
        for text in ("2 3", "2**3", "2^3", "x+2", "2+", "(2", "__import__('os')"):
            with self.assertRaises(ValueError, msg=text):
                evaluate_expression(text)
        with self.assertRaises(ZeroDivisionError):
            evaluate_expression("1/0")

    def test_equations_numeric_rhs_and_chains(self):
        steps = checked_equations("3*20=60\n400-80=320\n2+3=5=10/2\n2+2=5\n5=5")
        self.assertEqual(len(steps), 5)
        self.assertEqual([s.valid for s in steps], [True, True, True, True, False])

    def test_latex_units_percent_and_nested_fraction(self):
        text = r"3 \times 20 = \boxed{60}" + "\n"
        text += r"25% * 200 = 50 \text{dollars}" + "\n"
        text += r"\frac{1}{\frac{2}{3}} = 1.5" + "\n"
        text += r"1,200 / 3 = 400"
        steps = checked_equations(text)
        self.assertEqual(len(steps), 4)
        self.assertTrue(all(s.valid for s in steps))

    def test_variables_truncation_and_resource_limits_are_skipped(self):
        for text in ("2x+3=7", "x + 2*3 = 7", "3+x + 2*3 = 9", "2^3+2=10",
                     "a_2+3=5", "3+2=5x", "3+2=5^2", "1/0=2", "2 3+1=24",
                     "9"*100+"+1=2", "("*100+"2+3"+")"*100+"=5"):
            self.assertEqual(checked_equations(text), [], text)
        # Prose before an independent expression remains acceptable.
        self.assertTrue(checked_equations("The total is 3+2 = 5")[0].valid)

    def test_unicode_markdown_and_unit_conversion(self):
        steps = checked_equations("150 + 620 × 0.50 = 150 + 310 = **460**")
        self.assertEqual(len(steps), 2)
        self.assertTrue(all(step.valid for step in steps))
        steps = checked_equations(r"6 \text{ hours} = 6 \times 60 = 360 \text{ minutes}")
        self.assertEqual(len(steps), 1)
        self.assertTrue(steps[0].valid)
        for text in ("40 houses × 80% = 32 houses", "2 ^ 3+2=10"):
            self.assertEqual(checked_equations(text), [])
        steps = checked_equations(r"60\% \text{ of } 2000 = 0.6 \times 2000 = 1200")
        self.assertEqual(len(steps), 1)
        self.assertTrue(steps[0].valid)

    def test_fraction_trace_and_invalid_equation_not_evidence(self):
        rows = [{"text": "1/4+1/4=0.5\n#### 0.5"},
                {"text": "2+2=5\n#### 5"},
                {"text": "2+3=5\n5*2=10\n#### 5"},
                {"text": "2+3=5\nno answer marker"}]
        x = candidate_features("", rows)
        self.assertEqual(x.dtype, np.float64)
        np.testing.assert_array_equal(x[0, 3:5], [1, 1])
        np.testing.assert_array_equal(x[1, 2:5], [1, 0, 0])
        np.testing.assert_array_equal(x[2, 3:5], [0, 1])
        np.testing.assert_array_equal(x[3], 0)

    def test_question_coverage_and_arbitrary_k(self):
        rows = [{"text": "15+1=16\n#### 20"}, {"text": "5+5=10\n#### 10"}]
        x = candidate_features("There are 5 and 20, also 5.", rows)
        np.testing.assert_array_equal(x[:, 5], [0, 0.5])
        self.assertEqual(candidate_features("", []).shape, (0, 6))
        self.assertEqual(candidate_features("", rows[:1]).shape, (1, 6))
        self.assertEqual(candidate_features("", rows*3).shape, (6, 6))

    def test_group_pooling_mean_and_any(self):
        rows = [{"text": "2+3=5\n#### 5"}, {"text": "#### 5"}, {"text": "2+2=4\n#### 4"}]
        x = aggregate_features("", rows, [[0, 1], [2]])
        np.testing.assert_array_equal(x[0, 3:5], [0.5, 0.5])
        np.testing.assert_array_equal(x[0, 6:], [1, 1])

    def fixture(self):
        rows = [{"text": t} for t in ("2+3=5\n#### 5", "#### 5", "2+2=4\n#### 4", "#### 3")]
        embedding = np.random.default_rng(7506).normal(size=(4, 1024)).astype(np.float32)
        return rows, embedding

    def test_feature_set_mask_and_candidate_permutation(self):
        rows, embedding = self.fixture()
        permutation = [2, 0, 3, 1]
        reordered = [rows[i] for i in permutation]
        for feature_set, enabled in FEATURE_SETS.items():
            x = build_features(rows, embedding, answer_groups(rows), question="2 and 3", feature_set=feature_set)
            y = build_features(reordered, embedding[permutation], answer_groups(reordered),
                               question="2 and 3", feature_set=feature_set)
            np.testing.assert_array_equal(x, y)
            self.assertEqual(x.shape, (3, FEATURE_DIM))
            disabled = [i for i in range(24) if i not in enabled]
            np.testing.assert_array_equal(x[:, 2064 + np.array(disabled)], 0)
            self.assertEqual(len(scalar_indices(feature_set)), 16 + len(enabled))
        with self.assertRaises(ValueError):
            build_features(rows, embedding, answer_groups(rows), feature_set="full")

    def test_new_checkpoint_forward_and_batching(self):
        rows, embedding = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            weight = np.zeros(FEATURE_DIM, dtype=np.float32)
            weight[2071] = 1
            np.savez(directory/"model.npz", weight=weight, mean=np.zeros(FEATURE_DIM, np.float32),
                     scale=np.ones(FEATURE_DIM, np.float32), active=np.ones(FEATURE_DIM, bool))
            (directory/"config.json").write_text(json.dumps({
                "feature_version": feature_version("full"), "feature_set": "full", "feature_dim": FEATURE_DIM}))
            (directory/"asset_manifest.json").write_text(json.dumps({"sha256": {
                name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                for name in ("model.npz", "config.json")}}))
            row = {"question": "2 and 3", "candidates": rows}
            choices = predict([row]*130, {"embeddings": np.tile(embedding, (130, 1, 1))}, directory)
            self.assertEqual(len(set(choices)), 1)
            self.assertEqual(parse_answer(rows[choices[0]]["text"]), "4")
            permutation = [2, 0, 3, 1]
            reordered = {"question": row["question"], "candidates": [rows[i] for i in permutation]}
            j = predict([reordered], {"embeddings": embedding[permutation][None]}, directory)[0]
            self.assertEqual(parse_answer(reordered["candidates"][j]["text"]), "4")


if __name__ == "__main__":
    unittest.main()
