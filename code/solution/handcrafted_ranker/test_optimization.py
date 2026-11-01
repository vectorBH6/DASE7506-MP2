"""Block-gradient, interaction semantics, compatibility and forward integration."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import parse_answer
from solution.group_ranker.train_linear import fit_linear as uniform_fit, objective
from solution.handcrafted_ranker.arithmetic import feature_version
from solution.handcrafted_ranker.features import FEATURE_DIM, answer_groups, build_features
from solution.handcrafted_ranker.optimization import block_objective, fit_linear, penalty_vector
from solution.handcrafted_ranker.submission_linear import load_model, predict


class OptimizationTests(unittest.TestCase):
    def test_block_objective_gradient(self):
        rng = np.random.default_rng(7506)
        x, weight = rng.normal(size=(12, 7)), rng.normal(size=7)
        mask = np.array([[1,1,0,0], [1,1,1,0], [1,1,1,1]], dtype=bool)
        labels = np.array([1, 2, 0])
        penalty = np.array([.01,.01,.03,.03,.003,.003,.1])
        _, gradient = block_objective(weight, x, mask, labels, penalty)
        numerical = np.empty_like(weight)
        for j in range(len(weight)):
            shift = np.eye(len(weight))[j]*1e-5
            numerical[j] = (block_objective(weight+shift,x,mask,labels,penalty)[0] -
                            block_objective(weight-shift,x,mask,labels,penalty)[0])/2e-5
        np.testing.assert_allclose(gradient,numerical,atol=1e-7)
        loss, grad = block_objective(weight,x,mask,labels,np.full(7,.01))
        expected_loss, expected_grad = objective(weight,x,mask,labels,.01)
        self.assertAlmostEqual(loss, expected_loss, places=12)
        np.testing.assert_allclose(grad,expected_grad,atol=1e-12)

    def test_penalty_scope_and_validation(self):
        penalty = penalty_vector(.03,.3)
        np.testing.assert_array_equal(penalty[:2064],.03)
        np.testing.assert_allclose(penalty[2064:2074],.009)
        np.testing.assert_array_equal(penalty[2074:],.03)
        for l2, multiplier in ((0,1),(.01,0),(.01,float("nan")),(-1,1)):
            with self.assertRaises(ValueError):
                penalty_vector(l2,multiplier)

    def test_uniform_fit_is_bitwise_compatible(self):
        rng = np.random.default_rng(7506)
        x = rng.normal(size=(5,4,FEATURE_DIM)).astype(np.float32)
        mask = np.ones((5,4),bool)
        labels = np.array([0,1,2,3,0])
        old,_ = uniform_fit(x,mask,labels,.01,30)
        new,_ = fit_linear(x,mask,labels,.01,30,1)
        for name in old:
            np.testing.assert_array_equal(new[name],old[name])

    def fixture(self):
        candidates = [{"text":t} for t in (
            "2+2=5\n#### 5", "#### 5", "2+2=4\n#### 4", "#### 3")]
        embeddings = np.random.default_rng(7506).normal(size=(4,1024)).astype(np.float32)
        return candidates, embeddings

    def test_interactions_and_old_features(self):
        rows, embedding = self.fixture()
        groups = answer_groups(rows)
        old = build_features(rows,embedding,groups,question="2",feature_set="math")
        new = build_features(rows,embedding,groups,question="2",feature_set="math_interactions")
        np.testing.assert_array_equal(old[:,:2072],new[:,:2072])
        np.testing.assert_array_equal(new[:,2072],new[:,2048]*new[:,2066])
        np.testing.assert_array_equal(new[:,2073],new[:,2048]*new[:,2067])
        np.testing.assert_array_equal(new[:,2074:],0)
        self.assertEqual(feature_version("math_interactions"),"answer-group-handcrafted-v3")

    def bundle(self, directory, version=None):
        weight = np.zeros(FEATURE_DIM,np.float32);weight[2072]=1
        np.savez(directory/"model.npz",weight=weight,mean=np.zeros(FEATURE_DIM,np.float32),
                 scale=np.ones(FEATURE_DIM,np.float32),active=np.ones(FEATURE_DIM,bool))
        (directory/"config.json").write_text(json.dumps({
            "feature_version":version or feature_version("math_interactions"),
            "feature_set":"math_interactions","feature_dim":FEATURE_DIM}))
        (directory/"asset_manifest.json").write_text(json.dumps({"sha256":{
            name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ("model.npz","config.json")}}))

    def test_interaction_checkpoint_forward(self):
        rows, embedding = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);self.bundle(directory)
            row={"question":"2","candidates":rows}
            j=predict([row],{"embeddings":embedding[None]},directory)[0]
            self.assertEqual(parse_answer(rows[j]["text"]),"5")

    def test_interaction_version_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);self.bundle(directory,"answer-group-handcrafted-v2")
            with self.assertRaises(ValueError):
                load_model(directory)


if __name__=="__main__":
    unittest.main()
