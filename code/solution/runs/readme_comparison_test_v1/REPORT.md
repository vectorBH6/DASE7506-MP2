# Fixed baseline and linear test comparisons

These supplementary evaluations were requested for the README after the final lexical model had already been frozen. Existing source and checkpoints were hashed before evaluation; all listed methods are reported. No training, parameter search, or model reselection was performed.

| Method | Correct / total | Accuracy |
|---|---:|---:|
| Original baseline | 323/467 | 69.16488223% |
| Voting only | 323/467 | 69.16488223% |
| Embedding only | 271/467 | 58.02997859% |
| Expanded linear | 388/467 | 83.08351178% |

Official predict.py and score_predictions.py were run in separate fresh processes for each method. plan.json, freeze.json, source.tar, per-method prediction/score/summary files and summary.json preserve the evaluation evidence. The original baseline and voting-only select identical parsed answers on all 467 test questions. The selected lexical checkpoint and its original 45 frozen files remain unchanged.
