"""注意力梯度、参数匹配、padding、候选排列和冻结资产检查。"""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import PROTOCOL, parse_answer
from solution.attention_ranker.model import initialize, forward, loss_gradient, MODES
from solution.attention_ranker.submission import MODEL, predict, predict_profiled, load_model, create_bundle, digest
from solution.group_ranker.features import FEATURE_DIM, FEATURE_VERSIONS, feature_mask
from solution.group_ranker.training import export_model
from solution.group_ranker import test_group_ranker as fixtures


class AttentionTests(unittest.TestCase):
    def bundle(self, root, mode):
        model = initialize(FEATURE_DIM, np.random.default_rng(7506))
        model.update(mean=np.zeros(FEATURE_DIM, dtype=np.float32), scale=np.ones(FEATURE_DIM, dtype=np.float32),
                     active=feature_mask("text"))
        config = dict(protocol=PROTOCOL, model=MODEL, mixing=mode, width=64, heads=2, ff_dim=128,
                      positional_encoding=False, feature_set="text", feature_dim=FEATURE_DIM,
                      feature_version=FEATURE_VERSIONS["text"])
        export_model(root, model, config)

    def test_gradients_and_masked_rows(self):
        x = np.random.default_rng(7506).normal(size=(2, 4, 5))
        mask, labels = np.array([[1,1,0,0], [1,1,1,0]], bool), np.array([1,2])
        for mode in MODES:
            rng = np.random.default_rng(7506)
            model = {k:v.astype(np.float64) for k,v in initialize(5, rng, width=4, ff_dim=7).items()}
            _, gradient = loss_gradient(x, mask, labels, model, mode, rng, dropout=0)
            for name in model:
                for index in range(min(3, model[name].size)):
                    original, epsilon = model[name].flat[index], 1e-5
                    model[name].flat[index] = original + epsilon
                    plus = loss_gradient(x, mask, labels, model, mode, rng, dropout=0)[0]
                    model[name].flat[index] = original - epsilon
                    minus = loss_gradient(x, mask, labels, model, mode, rng, dropout=0)[0]
                    model[name].flat[index] = original
                    self.assertAlmostEqual(gradient[name].flat[index], (plus-minus)/(2*epsilon), places=6)
            changed = x.copy()
            changed[~mask] = 10000
            _, actual = loss_gradient(changed, mask, labels, model, mode, rng, dropout=0)
            for name in model:
                np.testing.assert_allclose(gradient[name], actual[name], atol=1e-12)

    def test_equal_parameters_and_group_exchange(self):
        rng = np.random.default_rng(7506)
        x, mask = rng.normal(size=(1,4,5)), np.ones((1,4),bool)
        model = initialize(5, rng, width=4, ff_dim=7)
        changed = x.copy()
        changed[:,1] += np.arange(5) * 3
        for mode in MODES:
            before = forward(x, mask, model, mode)[0][0,0]
            after = forward(changed, mask, model, mode)[0][0,0]
            if mode == "gated":
                self.assertEqual(before, after)
            else:
                self.assertGreater(abs(before-after), 1e-6)
        self.assertEqual(sum(p.size for p in initialize(FEATURE_DIM, rng).values()), 167233)

    def test_group_permutation_equivariance(self):
        rng = np.random.default_rng(7506)
        x, mask = rng.normal(size=(2,4,5)), np.array([[1,1,0,0],[1,1,1,0]], bool)
        model, order = initialize(5, rng, width=4, ff_dim=7), [2,0,3,1]
        for mode in MODES:
            expected = forward(x, mask, model, mode)[0][:, order]
            actual = forward(x[:,order], mask[:,order], model, mode)[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_candidate_permutations_chunks_and_boundary_inputs(self):
        f = fixtures.MLPTests()
        f.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            for mode in MODES:
                root = Path(temporary)/mode
                self.bundle(root, mode)
                chosen = predict([f.row], {"embeddings":f.embedding[None]}, root)[0]
                answer = parse_answer(f.candidates[chosen]["text"])
                for order in itertools.permutations(range(4)):
                    row = {**f.row, "candidates":[f.candidates[j] for j in order]}
                    actual = predict([row], {"embeddings":f.embedding[list(order)][None]}, root)[0]
                    self.assertEqual(parse_answer(row["candidates"][actual]["text"]), answer)
                none = {**f.row, "candidates":[{"text":"unknown"}]*4}
                one = {**f.row, "candidates":[{"text":"unknown"}]*3+[{"text":"#### 2"}]}
                bank = [none,one]+[f.row]*129
                choices = predict(bank,{"embeddings":np.tile(f.embedding,(len(bank),1,1))},root)
                self.assertEqual(choices,[0,3]+[chosen]*129)
                self.assertTrue(all(type(j) is int for j in choices))
                self.assertEqual(predict([],{"embeddings":np.empty((0,4,1024))},None),[])
                self.assertEqual(len(predict([f.row],{"embeddings":np.zeros((1,4,1024))},root)),1)
                for invalid in (np.zeros((2,4,1024)),np.full((1,4,1024),np.nan)):
                    with self.assertRaises(ValueError):
                        predict([f.row],{"embeddings":invalid},root)

    def test_bundle_probability_formula_and_asset_count(self):
        from solution.group_ranker.submission_linear import predict_profiled as linear_predict
        from solution.group_ranker.submission_ensemble import average_scores
        f = fixtures.MLPTests()
        f.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            linear = root/"linear"
            linear.mkdir()
            fixtures.GroupRankerTests().bundle(linear)
            for mode in MODES:
                checkpoint, bundle = root/mode, root/(mode+"_ensemble")
                self.bundle(checkpoint,mode)
                size = create_bundle(linear,checkpoint,bundle)
                self.assertEqual(size,sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()))
                args=([f.row],{"embeddings":f.embedding[None]})
                left=linear_predict(*args,linear,include_scores=True)[1]["group_scores"][0]
                right=predict_profiled(*args,checkpoint,include_scores=True)[1]["group_scores"][0]
                actual=predict_profiled(*args,bundle,include_scores=True)[1]["group_scores"][0]
                np.testing.assert_allclose(actual,average_scores(left,right),rtol=1e-5,atol=1e-6)
                self.assertLess(size,16*2**20)

    def test_hash_schema_and_numeric_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.bundle(root,"attention")
            path=root/"config.json"
            config=json.loads(path.read_text())
            config["positional_encoding"]=True
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_model(root)
            manifest=json.loads((root/"asset_manifest.json").read_text())
            manifest["sha256"]["config.json"]=digest(path)
            (root/"asset_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_model(root)


if __name__ == "__main__":
    unittest.main()
