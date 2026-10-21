# DASE7506 MP2: Linear and MLP Answer Group Ranking

**Name:** Li Shanghang · **Student ID:** 3036705979

English | [简体中文](README_cn.md)

The final method is **L0 + M1**, an equal probability ensemble of a basic linear answer group ranker and a small MLP with text features. The final checkpoint is `ensemble_v1`: **740/798 validation (92.73%)** and **377/467 test (80.72805139186295%)**.

**Selection and chronology.** This restores the original choice retained by the saved validation rule. The temporary designation of L0 + attention based on its higher observed test score has been withdrawn; attention remains an exploratory comparison. Earlier individual-model test scores were already visible when the original ensemble was developed, so the development history is disclosed as post hoc, not a new independent blind test. Restoring the recorded choice does not erase that history. Weights, scores and historical experiment records are preserved.

[English report: PDF, 8 pages](report/report.pdf) · [Editable report](report/report.md) · [Development records, Chinese](code/solution/README.md) · [Assignment guide](guide/GUIDE.md)

## 1. Final Algorithm and Parameters

| Item | Fixed configuration / recorded measurement |
|---|---|
| Prediction entry | `code/solution/group_ranker/submission_ensemble.py`: `predict(bank, arrays, checkpoint)` |
| Self-contained checkpoint | `code/solution/runs/ensemble_v1/checkpoint/` |
| Inputs | Question, four candidate texts, supplied `[N, 4, 1024]` embeddings |
| Answer groups | Equivalent numerical answers grouped using official `common.parse_answer` |
| Feature schema | 2088 slots: 1024 group embedding means, 1024 context interactions, 40 scalar slots |
| L0 features | 2048 vector coordinates + 16 basic scalars; remaining slots disabled |
| M1 features | 2048 vector coordinates + 32 basic/text scalars; 8 arithmetic slots disabled |
| Normalization | Separate train-fitted scaler per member; fold-specific fitting in OOF; clipping to `[-8, 8]` |
| L0 | Group cross entropy + L2; L2=`0.01`, selected from `0.0001/0.001/0.01` by train OOF; L-BFGS-B, at most 300 iterations |
| M1 architecture | `2088 -> 64 -> 16 -> 1`, with ReLU hidden layers and a shared scoring head across groups |
| M1 optimizer | AdamW; learning rate `0.001`, weight decay `0.01`, dropout `0.1`, batch 64 questions, gradient norm limit 1 |
| M1 epochs | At most 150, early-stopping patience 15 on held-out train-fold NLL; fold optima `[5,5,5,3,6]`, final refit **5 epochs** |
| Folds / seeds | Five question-level folds stratified by vote pattern, split seed `7506`; M1 fold seeds `7506 + fold`, final seed `7506`; diagnostic seeds `7507/7508` add no ensemble members |
| Fusion | Per-member group softmax, then `0.5 * P_L0 + 0.5 * P_M1`; both temperatures fixed at 1 |
| Stored scoring parameters | L0: 2088; M1: 134753; total **136841**, including inactive stored coordinates; scaler arrays counted separately as assets |
| Inference chunk size | At most 128 questions |
| Uncompressed inference assets | **590639 bytes = 0.563 MiB**, including both members, statistics, configurations and manifests |
| Historical complete test evaluation | **0.414246 seconds**, fresh prediction + independent scoring processes, including startup, loading, features and both members |
| Total observed peak threads | **4**: coordinator 1 + child 3; prediction/scoring child processes run sequentially |
| Historical measurement environment | Core Ultra 9 275HX; Linux; Python `3.12.3`; NumPy `2.5.3`; cache state uncontrolled |
| Freeze ID | `79062305cd4b895b0c26d7f2eed97c321a68b1fe2b83d07e40905bb29b72db1b` |

Training scans all 4755 train questions. Of these, 2641 have one valid answer group, requiring no choice between numerical answers and giving zero group-classification loss. Both members fit all **2114 conflicting train questions**. Validation/test references are excluded from fitting weights and normalization statistics.

| Evaluation scope | Correct | Accuracy |
|---|---:|---:|
| Train OOF, conflicting questions | 1717/2114 | 81.22% |
| Complete validation | 740/798 | 92.73% |
| Conflicting validation questions | 297/355 | 83.66% |
| **Complete test** | **377/467** | **80.72805139186295%** |

Validation contains 443 single-group questions, whereas every test question has conflicting answers. Their overall accuracies reflect different question compositions.

Evidence: [official test score](code/solution/runs/ensemble_v1/test/score.json), [validation selection](code/solution/runs/ensemble_v1/validation/summary.json), [test/resource analysis](code/solution/runs/ensemble_v1/test/summary.json), [freeze manifest](code/solution/runs/ensemble_v1/test/freeze.json), [frozen source](code/solution/runs/ensemble_v1/test/source.tar), and [later validation retention](code/solution/runs/attention_v1/validation_summary.json).

## 2. Algorithm in Brief

1. **Group equivalent answers.** Merge candidates with the same final numerical answer using the official parser. A single valid group is selected directly; no valid groups fall back to index 0.
2. **Describe each group.** Use supplied embedding means, votes, semantic geometry, context interactions and lightweight text/question evidence.
3. **Score with two branches.** L0 computes a weighted sum of basic features. M1 learns nonlinear combinations of basic and text features through two small hidden layers. Both output one score per group.
4. **Average probabilities.** Each branch applies softmax over the groups of its question. Average probabilities equally, select the winning group, then return its smallest candidate index. Near ties use a fixed rule.

The final method uses no attention block. Its handcrafted features already encode relationships among candidate groups. The official metric checks **final numerical correctness**, not every reasoning step. Attention, wider MLPs, trees and calibration remain comparison experiments; their higher observed test scores do not replace the saved validation choice.

## 3. Assignment Requirements and Remaining Limits

| Requirement | Evidence / status |
|---|---|
| Fit only supplied train inputs and references | Train-only weights/scalers; five-fold train OOF. |
| Use validation for development and selection | Final method restored to the original recorded validation preference; historical test exposure remains disclosed, so no independent blind-test claim. |
| Unchanged data/parser/scorer; no external data or pretrained weights | Uses supplied data/embeddings and official tools; ranking models trained locally. |
| Offline selection; no generation/API, answer lookup or fixed bank IDs | NumPy inference chooses supplied candidates from the given inputs. |
| At most 16 MiB of uncompressed inference assets | **0.563 MiB** for all eight files, within the limit. |
| At most four total evaluation threads | Historical local peak **4** across coordinator and child. |
| At most 30 s on Xeon Platinum 8457C | Local full evaluation **0.414 s**; **reference CPU eligibility unverified**. |
| CPU or at most one RTX 4090 for training | CPU-only. L0 search/refit **46.23 s**; fixed M1 train OOF/refit **69.60 s**; ensemble packaging **0.011 s**, excluding prior architecture search. |
| Report at most 10 pages; matching public sources, weights and records | Eight-page report, matching source, checkpoints and records are versioned together; use a publicly accessible submitted commit. |

`code/solution/runs/` is tracked, including checkpoints, OOF predictions, official scores, experiment records and frozen source archives. Virtual environments, caches and temporary report files remain ignored. Current source has evolved since the historical freeze; Section 5.2 explicitly uses the original source snapshot. Hashes establish artifact integrity, not independence from prior test feedback.

## 4. Code Structure

```text
DASE7506-MP2/
├── README.md / README_cn.md          # Final method and reproduction commands
├── report/                          # English manuscript, PDF, figures and builder
├── guide/ / class pdf/               # Assignment guide and study materials
└── code/
    ├── common.py                    # Official parser and data interface
    ├── predict.py / score_predictions.py
    ├── data/                        # Supplied inputs and representations
    └── solution/
        ├── group_ranker/
        │   ├── features.py / evidence.py     # Shared group and text features
        │   ├── training.py / mlp.py          # Training tools and NumPy MLP
        │   ├── train_linear.py / train_mlp.py
        │   ├── submission_linear.py / submission_mlp.py
        │   └── submission_ensemble.py        # FINAL prediction entry and packaging
        ├── baselines/ / linear_text/
        ├── calibrated_ensemble/ / tree_ranker/
        ├── attention_ranker/        # Exploratory attention and gated controls
        ├── requirements-train.txt
        └── runs/ensemble_v1/
            ├── checkpoint/          # Complete final inference assets
            ├── validation/          # Development comparison and selection
            └── test/                # Official score, predictions and frozen source
```

Inference needs only the supplied inputs, matching source and complete checkpoint; it does not read OOF predictions, reference answers or other runs' weights.

## 5. Reproduction Guide

### 5.1 Shared Environment and Checkpoint

From the project root, enter `code/`. Reuse **`code/solution/.venv`**. On a fresh clone, create that one environment only if it is missing:

```bash
cd code
unset PYTHONPATH
if [ ! -x solution/.venv/bin/python ]; then
  python3.12 -m venv solution/.venv
fi
solution/.venv/bin/python -m pip install -r solution/requirements-train.txt
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3
```

The shared environment pins NumPy `1.26.4`, SciPy `1.11.4` and CatBoost `1.2.8`; use Python 3.12. Final inference needs only NumPy and the standard library. SciPy supports L0 training; CatBoost is for historical tree comparisons. The historical **0.414-second** measurement used NumPy **2.5.3** and is distinct from a replay in the current environment.

Place the eight files at `solution/runs/ensemble_v1/checkpoint/`:

```text
checkpoint/
├── config.json
├── asset_manifest.json
├── linear/   # model.npz, config.json, asset_manifest.json
└── mlp/      # model.npz, config.json, asset_manifest.json
```

Also provide `solution/runs/ensemble_v1/test/source.tar` for the matching historical source. The loader verifies asset hashes, shapes and total size.

### 5.2 Reproduce Official Scores Without Retraining

This extracts the frozen source into a new run directory and reuses the original inputs/checkpoint. It avoids relying on later source refactors. The extraction creates no virtual environment. `time` covers fresh prediction and independent scoring; the shell command itself does not sample thread counts.

```bash
out="$PWD/solution/runs/ensemble_reproduction"
checkpoint="$PWD/solution/runs/ensemble_v1/checkpoint"
data="$PWD/data"
if mkdir "$out"; then
  tar -xf solution/runs/ensemble_v1/test/source.tar -C "$out"
  for split in validation test; do
    time {
      solution/.venv/bin/python "$out/code/predict.py" \
        --method "$out/code/solution/group_ranker/submission_ensemble.py" \
        --checkpoint "$checkpoint" --data-dir "$data" \
        --split "$split" --output "$out/$split-predictions.json" &&
      solution/.venv/bin/python "$out/code/score_predictions.py" \
        --data-dir "$data" --split "$split" \
        --predictions "$out/$split-predictions.json" --output "$out/$split-score.json"
    }
  done
fi
```

Expected counts: **740/798 validation; 377/467 test**. The recorded official test `accuracy` field is **`80.72805139186295`**, in percent. Current-environment timing may differ; reference CPU eligibility remains unverified.

### 5.3 Reproduce the Fixed Training Recipe (Optional)

The current source can refit the selected recipe in the shared environment. L0 repeats its original train OOF regularization selection; M1 reuses the selected 64/16 architecture and optimizer settings from the final checkpoint. The earlier M0 architecture search and diagnostic seeds are documented in [solution/README](code/solution/README.md); this command does not repeat that search or add seed ensembles.

```bash
base=solution/runs/final_retrain
if mkdir "$base"; then
  solution/.venv/bin/python -m solution.group_ranker.train_linear \
    --l2 0.0001 0.001 0.01 --folds 5 --seed 7506 --maxiter 300 \
    --output "$base/linear" &&
  solution/.venv/bin/python -m solution.group_ranker.train_mlp \
    --feature-set text --reuse-config solution/runs/ensemble_v1/checkpoint/mlp/config.json \
    --output "$base/m1" &&
  solution/.venv/bin/python -m solution.group_ranker.submission_ensemble \
    --linear-checkpoint "$base/linear/checkpoint" \
    --mlp-checkpoint "$base/m1/checkpoint" --output "$base/ensemble"
fi
```

New weights are under `$base/ensemble/checkpoint/`. Refitting uses train only and does not automatically evaluate test. It is not necessary for score reproduction and is not guaranteed to reproduce historical bytes across source/library versions; use the original weights and frozen source in Section 5.2 for the reported result.

### 5.4 Rebuild the Report

From the project root, using the same environment:

```bash
code/solution/.venv/bin/python -m pip install -r report/requirements.txt
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  code/solution/.venv/bin/python report/build_report.py
```

The builder reads saved records to regenerate four figures, `report/evidence.json` and the PDF. It performs no training, prediction, scoring or model-test execution.

## 6. Codex Assistance Disclosure

OpenAI Codex assisted algorithm discussion, implementation/refactoring, engineering checks, experiment execution/records, and documentation/report preparation. Numbers come from saved local records and official scorer JSON. Codex is not called during offline inference. The author is responsible for understanding the implementation and disclosing the experimental chronology. Retain the supplied [data license](code/data/LICENSE.txt) and [attribution notice](code/data/NOTICE.txt).
