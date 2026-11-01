import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common import parse_answer

from solution.residual_ranker.features import FEATURE_DIM,answer_groups,build_features,candidate_extras
from solution.residual_ranker.model import VERSION,export,load,scores
from solution.residual_ranker.submission import predict
from solution.residual_ranker.train import objective


class ResidualTests(unittest.TestCase):
    def test_objective_gradient(self):
        rng=np.random.default_rng(7506)
        x,z=rng.normal(size=(12,5)),rng.normal(size=(12,3))
        theta=rng.normal(size=5+3*2+2+2)*.2
        mask=np.array([[1,1,0,0],[1,1,1,0],[1,1,1,1]],bool)
        labels=np.array([1,2,0])
        _,gradient=objective(theta,x,z,mask,labels,2,.01,.003)
        numerical=np.empty_like(theta)
        for j in range(len(theta)):
            shift=np.eye(len(theta))[j]*1e-5
            numerical[j]=(objective(theta+shift,x,z,mask,labels,2,.01,.003)[0]-
                          objective(theta-shift,x,z,mask,labels,2,.01,.003)[0])/2e-5
        np.testing.assert_allclose(gradient,numerical,atol=1e-7)

    def fixture(self):
        rows=[{"text":text} for text in ("2+3=5\n#### 5","#### 5","2+2=4\n#### 4","#### 3")]
        embedding=np.random.default_rng(7506).normal(size=(4,1024)).astype(np.float32)
        return rows,embedding

    def test_expanded_features_and_permutation(self):
        rows,e=self.fixture();groups=answer_groups(rows)
        basic=build_features(rows,e,groups,question="How many apples remain?",schema="math")
        expanded=build_features(rows,e,groups,question="How many apples remain?",schema="expanded")
        np.testing.assert_array_equal(basic[:,:2072],expanded[:,:2072])
        self.assertTrue(np.isfinite(expanded).all())
        perm=[3,1,0,2];changed=[rows[i] for i in perm]
        actual=build_features(changed,e[perm],answer_groups(changed),question="How many apples remain?",schema="expanded")
        np.testing.assert_array_equal(expanded,actual)
        self.assertEqual(candidate_extras("",[]).shape,(0,8))
        np.testing.assert_array_equal(candidate_extras("",[{"text":"unknown"}]),0)

    def model(self,schema,hidden):
        q=24 if schema=="math" else 40
        rng=np.random.default_rng(7506)
        return {"weight":rng.normal(size=FEATURE_DIM).astype(np.float32),
                "mean":np.zeros(FEATURE_DIM,np.float32),"scale":np.ones(FEATURE_DIM,np.float32),
                "active":np.ones(FEATURE_DIM,bool),"hidden_weight":rng.normal(size=(q,hidden)).astype(np.float32),
                "hidden_bias":np.zeros(hidden,np.float32),"hidden_output":rng.normal(size=hidden).astype(np.float32),
                "hidden_indices":np.arange(2048,2048+q,dtype=np.int32)}

    def test_export_forward_batching_and_permutation(self):
        rows,e=self.fixture()
        for schema,hidden in (("math",8),("expanded",0),("expanded",8)):
            with tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)/"checkpoint";model=self.model(schema,hidden)
                export(directory,model,{"feature_version":VERSION,"feature_schema":schema,"hidden":hidden})
                loaded,_=load(directory)
                x=build_features(rows,e,answer_groups(rows),question="2 and 3",schema=schema)
                np.testing.assert_array_equal(scores(x,model),scores(x,loaded))
                row={"question":"2 and 3","candidates":rows}
                one=predict([row],{"embeddings":e[None]},directory)[0]
                many=predict([row]*130,{"embeddings":np.tile(e,(130,1,1))},directory)
                self.assertEqual(many,[one]*130)
                perm=[3,1,0,2]
                changed={"question":row["question"],"candidates":[rows[j] for j in perm]}
                other=predict([changed],{"embeddings":e[perm][None]},directory)[0]
                self.assertEqual(parse_answer(changed["candidates"][other]["text"]),parse_answer(rows[one]["text"]))

    def test_shortcuts_and_asset_validation(self):
        self.assertEqual(predict([],{"embeddings":np.empty((0,4,1024))},None),[])
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/"checkpoint"
            export(directory,self.model("math",8),{"feature_version":VERSION,"feature_schema":"math","hidden":8})
            rows=[{"question":"","candidates":[{"text":"unknown"}]*4},
                  {"question":"","candidates":[{"text":"unknown"}]*3+[{"text":"#### 2"}]}]
            self.assertEqual(predict(rows,{"embeddings":np.zeros((2,4,1024))},directory),[0,3])
            (directory/"config.json").write_text("{}")
            with self.assertRaises(ValueError):load(directory)


if __name__=="__main__":
    unittest.main()
