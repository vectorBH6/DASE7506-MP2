import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common import PROTOCOL, parse_answer
from solution.group_ranker.features import standardize
from solution.group_ranker.train_linear import export_model
from solution.residual_ranker.features import build_features as previous_features
from solution.residual_ranker.extended_features import (
    BASE_DIM, SCHEMAS, answer_groups, build_features, candidate_extensions,
    feature_dim, feature_names, columns,
)
from solution.residual_ranker.extended_model import VERSION, load, scores
from solution.residual_ranker.extended_submission import predict


class ExtendedTests(unittest.TestCase):
    def fixture(self):
        rows = [{"text":text} for text in (
            "2+3=5\n5*2=10\n10 apples remain.\n#### 10",
            "2+2=5\n#### 5", "There are 3 apples.\n#### 3", "unparseable")]
        e = np.random.default_rng(7506).normal(size=(4,1024)).astype(np.float32)
        return rows,e

    def test_trace_and_target_evidence(self):
        rows,_ = self.fixture()
        x = candidate_extensions("There were 2 apples. How many apples remain?",rows)
        self.assertEqual(x.shape,(4,24))
        self.assertEqual(x.dtype,np.float64)
        self.assertEqual(x[0,4],1)
        self.assertEqual(x[0,13],1)
        self.assertEqual(x[0,15],1)
        self.assertEqual(x[0,18],1)
        self.assertEqual(x[0,19],.5)
        self.assertEqual(x[0,20],0)
        self.assertEqual(x[1,17],1)
        self.assertEqual(x[1,13],0)
        np.testing.assert_array_equal(x[3],0)

    def test_empty_fraction_negative_and_large_values(self):
        self.assertEqual(candidate_extensions("",[]).shape,(0,24))
        rows=[{"text":"#### -2"},{"text":"#### 1/2"},{"text":"#### "+'9'*400}]
        x=candidate_extensions("",rows)
        self.assertTrue(np.isfinite(x).all())
        self.assertLess(x[0,9],0)
        self.assertEqual(x[0,8],0)
        self.assertEqual(x[1,8],0)

    def test_layout_base_preservation_and_permutation(self):
        rows,e = self.fixture()
        groups=answer_groups(rows)
        question="2 apples and 3 apples. How many apples remain?"
        full=build_features(rows,e,groups,question=question)
        old=previous_features(rows,e,groups,question=question,schema="expanded")
        np.testing.assert_array_equal(full[:,:BASE_DIM],old)
        self.assertEqual(full.shape,(3,2168))
        self.assertTrue(np.isfinite(full).all())
        for schema in SCHEMAS:
            actual=build_features(rows,e,groups,question=question,schema=schema)
            self.assertEqual(actual.shape[1],feature_dim(schema))
            self.assertEqual(len(feature_names(schema)),feature_dim(schema)-BASE_DIM)
            np.testing.assert_array_equal(actual,full[:,columns(schema)])
        perm=[3,1,0,2];changed=[rows[j] for j in perm]
        other=build_features(changed,e[perm],answer_groups(changed),question=question)
        np.testing.assert_array_equal(full,other)
        z=full[:,BASE_DIM+48:BASE_DIM+72]
        np.testing.assert_allclose(z.mean(axis=0),0,atol=1e-7)

    def test_checkpoint_and_real_interface(self):
        rows,e=self.fixture()
        rng=np.random.default_rng(7506)
        with tempfile.TemporaryDirectory() as tmp:
            for schema in SCHEMAS:
                d=feature_dim(schema)
                model={"weight":rng.normal(size=d).astype(np.float32),
                       "mean":np.zeros(d,np.float32),"scale":np.ones(d,np.float32),
                       "active":np.ones(d,bool)}
                config={"protocol":PROTOCOL,"feature_version":VERSION,
                        "feature_schema":schema,"feature_dim":d,"extra_feature_names":feature_names(schema)}
                directory=Path(tmp)/schema
                export_model(directory,model,config)
                loaded,_=load(directory)
                x=build_features(rows,e,answer_groups(rows),question="2 apples",schema=schema)
                np.testing.assert_array_equal(scores(x,model),scores(x,loaded))
                row={"question":"2 apples","candidates":rows}
                one=predict([row],{"embeddings":e[None]},directory)
                many=predict([row]*130,{"embeddings":np.tile(e,(130,1,1))},directory)
                self.assertEqual(many,one*130)
                perm=[3,1,0,2]
                changed={"question":"2 apples","candidates":[rows[j] for j in perm]}
                other=predict([changed],{"embeddings":e[perm][None]},directory)
                self.assertEqual(parse_answer(rows[one[0]]["text"]),
                                 parse_answer(changed["candidates"][other[0]]["text"]))
                invalid=[{"question":"","candidates":[{"text":"unknown"}]*4},
                         {"question":"","candidates":[{"text":"unknown"}]*3+[{"text":"#### 2"}]}]
                self.assertEqual(predict(invalid,{"embeddings":np.zeros((2,4,1024))},directory),[0,3])
                (directory/"config.json").write_text("{}")
                with self.assertRaises(ValueError):load(directory)
        self.assertEqual(predict([],{"embeddings":np.empty((0,4,1024))},None),[])


if __name__=="__main__":
    unittest.main()
