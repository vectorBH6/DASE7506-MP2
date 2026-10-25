"""Official predict interface for the independently packaged residual scorer."""
import numpy as np
from solution.residual_ranker.features import EMBED_DIM,answer_groups,best_group,build_features
from solution.residual_ranker.model import load,scores


def predict(bank,arrays,checkpoint):
    embeddings=np.asarray(arrays["embeddings"])
    if embeddings.shape!=(len(bank),4,EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid bank/embedding shapes or values")
    if not bank:
        return []
    model,config=load(checkpoint)
    choices=[0]*len(bank)
    for start in range(0,len(bank),128):
        pending,features=[],[]
        for i in range(start,min(start+128,len(bank))):
            candidates=bank[i]["candidates"]
            if len(candidates)!=4:
                raise ValueError("Exactly four candidates required")
            groups=answer_groups(candidates)
            if len(groups)==1:
                choices[i]=int(min(next(iter(groups.values()))))
            elif len(groups)>1:
                features.append(build_features(candidates,embeddings[i],groups,
                                question=bank[i]["question"],schema=config["feature_schema"]))
                pending.append((i,groups))
        if pending:
            logits=scores(np.concatenate(features),model)
            offset=0
            for i,groups in pending:
                winner=int(best_group(logits[offset:offset+len(groups)]))
                choices[i]=int(min(list(groups.values())[winner]))
                offset+=len(groups)
    return choices
