# MP2: Reliable Math Solution Selection

**Name:** li shanghang  
**Student ID:** 3036705979

[Final report (PDF, 8 pages)](report/MP2_Final_Report.pdf) · [Report source](report/REPORT.md)

A lightweight answer-group ranker for DASE7506 MP2. The final frozen model scores **393/467 (84.15418%)** on the complete public test bank.

## Final method

The system groups candidates by their final numeric answer using the unchanged official parser, then ranks groups with a linear scorer and returns one candidate index.

The **2,112 inputs** combine embedding statistics, answer consensus, arithmetic/text evidence, and **24 new columns**: means and maxima of 12 lexical/numeric signals covering question overlap, ending units, formatting, answer shape/magnitude, and number usage. These provide plausibility evidence, not a proof of correct reasoning.

Training uses group softmax cross-entropy, **L2 = 0.03**, five question-level folds, and seed **7506**. Standardization is fitted within each training fold. Training OOF selects L2; validation selects the variant. The smaller lexical extension matched the full extension on validation with better OOF accuracy.

## Results

OOF covers the 2,114 training questions with conflicting parsed answers. Validation and test results cover their complete banks.

| Model | Train OOF | Validation / 798 | Test correct / total (accuracy) |
|---|---:|---:|---:|
| Original baseline / majority vote | N/A | 688 | 323/467 (69.16488%) |
| Embedding-only baseline | N/A | 624 | 271/467 (58.02998%) |
| Original linear group ranker | 81.13% | 730 | 375/467 (80.29979%) |
| Linear + arithmetic features | 82.12% | 733 | 384/467 (82.22698%) |
| Linear + expanded features | 82.36% | 734 | 388/467 (83.08351%) |
| Previous expanded residual model | 82.54% | 734 | 388/467 (83.08351%) |
| **Final lexical extension (+24 columns)** | **83.30%** | **751** | **393/467 (84.15418%)** |

Training-free baselines have no OOF result. The original baseline and voting-only method have identical answer scores on both splits. Baseline and expanded-linear test scores were added after the final model was frozen, for reporting only; the selected model was unchanged. See the [validation records](code/solution/runs/linear_v1/validation/summary.json), [supplementary test records](code/solution/runs/readme_comparison_test_v1/summary.json), and [original linear test score](code/solution/runs/linear_v1/test/score.json).

The final model gains **18 test questions / 3.85 percentage points** over the original linear ranker, and **5 questions / 1.07 percentage points** over the previous expanded residual model. Local prediction plus independent scoring took **1.166 s**, with **2 total threads observed** and **29,766 bytes** of uncompressed inference assets on an **Intel Core Ultra 9 275HX**. The four-variant training search took approximately 91 s.

[Experiment report](code/solution/runs/handcrafted_extended_v2/REPORT.md) · [Official score](code/solution/runs/handcrafted_extended_v2/test/score.json) · [Frozen source and checkpoint](code/solution/runs/handcrafted_extended_v2/source.tar)

## Assignment alignment

| Requirement | Implementation / evidence |
|---|---|
| Supplied training data only | Weights and standardization use train; OOF/validation drive the current selection. |
| Fixed benchmark and offline selection | Official data/parser/scorer hashes match; inference uses NumPy and the supplied inputs, without reference access, ID lookups, cached choices, network calls, or new LLM generation. |
| New-bank support | Variable bank sizes, renamed/absent IDs, reordered candidates, and a synthetic new question were checked. |
| ≤16 MiB assets; ≤30 s CPU, ≤4 threads | Assets and local measurements are below the limits. Eligibility on the required Xeon Platinum 8457C remains unverified. |
| Reproduction and disclosure | 31 tests passed; fixed-configuration retraining reproduced model and OOF arrays bitwise in the recorded environment. AI/reuse assistance is disclosed below. |

No explicit implementation-level violation was found. **Earlier test scores were known during later development**; this is not an unseen-test experiment. The current model was frozen before its test run. See the [compliance review](code/solution/residual_ranker/COMPLIANCE_REVIEW.md) and [assignment guide](guide/GUIDE.md).

## Reproduce

Use **Python 3.12**. Inference is pinned to **NumPy 1.26.4**; training additionally requires **SciPy 1.11.4**. Run from the repository root:

```bash
cd code
python3 -m venv solution/.venv-inference
solution/.venv-inference/bin/python -m pip install -r solution/residual_ranker/requirements-inference.txt
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 BLIS_NUM_THREADS=3

solution/.venv-inference/bin/python predict.py \
  --method solution/residual_ranker/extended_submission.py \
  --checkpoint solution/runs/handcrafted_extended_v2/models/lexical/checkpoint \
  --split test --output solution/runs/readme_reproduction/predictions.json
solution/.venv-inference/bin/python score_predictions.py --split test \
  --predictions solution/runs/readme_reproduction/predictions.json \
  --output solution/runs/readme_reproduction/score.json
```

Expected official accuracy: **84.1541755888651**. The checkpoint must contain `model.npz`, `config.json`, and `asset_manifest.json`; no retraining is needed.

Optional retraining and tests, still from `code/` (use a new training output directory):

```bash
python3 -m venv solution/.venv-train
solution/.venv-train/bin/python -m pip install -r solution/requirements-train.txt
PYTHONDONTWRITEBYTECODE=1 solution/.venv-train/bin/python \
  -m solution.residual_ranker.extended_train --schemas lexical --l2 0.03 \
  --output solution/runs/readme_retrain
PYTHONDONTWRITEBYTECODE=1 solution/.venv-train/bin/python \
  -m unittest discover -s solution -p 'test_*.py'
```

Codex assisted with design, implementation, tests, experiments, analysis, and documentation; see [AI and reuse disclosure](code/solution/residual_ranker/AI_AND_REUSE.md). Retain the supplied [license](code/data/LICENSE.txt) and [attribution](code/data/NOTICE.txt) when redistributing data.

The final submission run is `code/solution/runs/handcrafted_extended_v2/`. Its matching checkpoint, evaluation records, and the report are tracked, together with archived comparison experiments. New runs and virtual environments remain Git-ignored.
