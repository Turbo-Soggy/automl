# Plan: EMBER2024 malware detection, manual tuning vs AutoML (IA-3 Part B)

## Context
This plan covers Part B (30 marks) of the course project. The task is binary benign/malicious classification of Windows PE files from static features, plus a research question: how much do random-split scores overstate real-world performance, and do AutoML tools protect against leakage and temporal drift or fall for them?

`C:\projects\automl` is empty and is not a git repo yet. Machine facts:
- Python 3.14 is the default, with 3.12 and 3.11 also installed.
- Java 21 is installed.
- 63 GB RAM, 643 GB free disk.
- WSL Ubuntu is installed but not needed.
- `nvidia-smi` couldn't run without admin rights.

**Decisions taken:**
- **Dataset:** EMBER2024, Win64 PE slice. It is Apache-2.0, has 64 weekly snapshots (Sep 2023 to Dec 2024), and is about 15 GB. The data includes `first_submission_date` and `detection_ratio` fields. `detection_ratio` gives a realistic leaky feature: a real proxy for the label, not a synthetic one.
- **AutoML:** FLAML, AutoGluon and H2O, each with the same 20-minute budget.
- **Defaults I picked** (stated in the README, change on request):
  - Python 3.12 venv, because AutoGluon and H2O don't support 3.14.
  - `run_all.py` instead of a Makefile, since this is Windows.
  - `git init` with `data/` and `models/*.bin` gitignored.
  - All tools on CPU (16 cores) so timing is comparable. GPU is noted as an optional extra.

## Repo layout
Same as the brief's §7, with these changes:
- `run_all.py` instead of `Makefile`.
- `src/logging_utils.py` added.
- `tests/` added.
- Notebooks become thin narrative wrappers around `src/` functions.

## Step 0: Environment and data verification (then report before modelling)
1. **Environment.** `py -3.12 -m venv .venv`, then a pinned `requirements.txt`:
   - Modelling: numpy, pandas, pyarrow, scikit-learn, lightgbm, optuna, flaml[automl], autogluon.tabular, h2o, shap.
   - Serving and plots: matplotlib, fastapi, uvicorn, pydantic, streamlit.
   - Other: orjson, joblib, pytest.
   - thrember installed from git at a pinned commit.
   - After installing, freeze the exact versions into the file.
2. **`config.py`.** Holds:
   - `SEED=42` and the `set_seed()` helper (numpy, random, `PYTHONHASHSEED`, and a library seed passed into every model and split).
   - Paths.
   - `AUTOML_BUDGET_S=1200`, `SUBSAMPLE_N=200_000`, `TUNE_TRIALS=36`, `CV_FOLDS=3`.
   - A `QUICK` flag for smoke runs.
3. **`src/data.py`, download.** Use `thrember.download_dataset(data/raw, file_type="Win64")` for the train and test splits. Confirm the real function signature first.
4. **`src/data.py`, `build_cache()`.** Stream the jsonl with orjson:
   - Vectorize each row using thrember's feature extractor (`process_raw_features`), so features and metadata stay row-aligned.
   - Write `X.npy` (float32 memmap) and `meta.parquet`. The parquet holds sha256, label, first_submission_date, week index, detection_ratio, family, file size, packer and signed tags where present.
   - Drop rows with `label == -1`.
5. **`scripts/verify_data.py`.** Prints and saves `reports/data_check.json`:
   - Rows per split and week, class balance per week.
   - Feature dimension, and feature-group names and offsets taken from the thrember extractor.
   - Min and max dates, plus the format of `detection_ratio` and how it relates to the label.
   - **Flags:** anything odd, such as a near-perfect `detection_ratio` vs label relationship (expected, that is the point) or overlapping sha256s across splits.

## Splits (`src/data.py`)
- **Random split:**
  - Draw a stratified sample of `SUBSAMPLE_N=200k` (by label × week) from the 52 training weeks, seed 42. Save the indices to `data/splits/`.
  - Split it 80/20 stratified into `train` (160k) and `test_random` (40k).
- **Temporal test:** a stratified 40k sample from the official future test weeks (53 to 64) → `test_future`. The same trained models are scored on both test sets. The gap between them is the drift and overstatement number.
- **CV:** the same `StratifiedKFold(3, seed)` folds are saved and reused by every tuning method. A week-blocked forward CV (train on earlier weeks, validate on later ones) is added in Step 4 to compare CV estimates.
- Every method uses the same `train` and never sees either test set until final scoring. A `pytest` test checks for no index overlap and that every `test_future` date is after every `train` date.

## Step 2: Baseline and tuning (`src/tune.py`), then STOP for review
- **Model family:** LightGBM, the standard EMBER baseline. It is fast and SHAP-friendly, and the "why" note goes in the README.
- **Baselines:**
  - LightGBM with default parameters.
  - Logistic regression on standardized features, as a sanity floor.
- **Search space:** num_leaves, learning_rate, min_child_samples, feature_fraction, bagging_fraction, lambda_l2, with n_estimators capped and early stopping on each fold.
- **Methods:**
  - Grid Search: 36 grid points over 4 key parameters.
  - Random Search: 36 samples.
  - Optuna TPE: 36 trials, seeded sampler.
- **"Equal budget":** the same 36 configurations evaluated × the same 3 folds × the same data. Wall-clock time is reported but isn't capped. Tuning runs on a 60k stratified subset of `train` to keep it tractable, then the best config is refit on all 160k.
- **Metrics:**
  - ROC-AUC, F1, accuracy, precision and recall.
  - FPR on benign files, plus TPR at 1% FPR (the security-relevant number).
  - Fit and search time.
- **Logging and outputs:** every trial and every final evaluation is logged through `src/logging_utils.log_experiment()` to `reports/experiments.jsonl`. `tables.py` builds `reports/tables/tuning.csv` and `reports/figures/tuning_compare.png` from the log, and also plots convergence (best score against trial number).

## Step 3: AutoML (`src/automl_runs.py`), then STOP for review
- **Same inputs for every tool:** same `train`, `test_random`, `test_future`, 1200 s each, seed 42, metric roc_auc.
  - **FLAML:** `AutoML.fit(time_budget=1200, metric="roc_auc", seed=42)`. Save the search log, best estimator and config.
  - **AutoGluon:** `TabularPredictor(eval_metric="roc_auc").fit(time_limit=1200, presets="medium_quality")`. Save the leaderboard, `feature_importance` on a 5k validation subsample, and preprocessing info (`feature_metadata`, dropped and unused features).
  - **H2O:** `H2OAutoML(max_runtime_secs=1200, seed=42, sort_metric="AUC")` with `h2o.init(max_mem_size="24G")`. Save the leaderboard and the leader's variable importance.
- **Logging:** wall-clock time and the scores on both test sets are logged for every tool.
- **Outputs:**
  - Leaderboards go to `reports/leaderboards/`.
  - `reports/tables/automl_vs_manual.csv` and a chart cover F1, AUC, TPR@1%FPR, time and an interpretability note.
  - A short markdown write-up says where each tool was better, worse or just different.

## Step 4: Critical evaluation (`src/eval.py`, `src/explain.py`), then STOP for review
1. **Random vs temporal:** a table of each model's score on `test_random` against `test_future`, showing the gap. Also compare stratified CV against week-blocked forward CV to see which CV estimate predicts `test_future` better.
2. **Leaky feature:**
   - Add `detection_ratio` (parsed to a float) as a column.
   - Rerun the tuned LightGBM, FLAML, AutoGluon and H2O. To save time, AutoML reruns use a 10-minute budget, stated as a deviation.
   - Show the score jump, the feature's importance rank and SHAP for it, then show the drop after removal.
   - Second, subtler variant: raw `first_submission_date` as a feature, to show time-proxy leakage under the random split and that it fails to help on `test_future`.
   - Report whether any tool warned about or dropped either feature.
3. **Bias and imbalance:**
   - Report class balance.
   - Show per-slice metrics by week, file-size bucket (quartiles), packer tag, and signed vs unsigned where the metadata exists.
   - Report FPR on benign files per slice.
   - Run a `class_weight="balanced"` or `scale_pos_weight` variant and report whether it helped.
4. **SHAP:**
   - Use TreeExplainer on the tuned LightGBM over a 5k sample.
   - Produce a beeswarm plot, a bar chart of importance grouped by EMBER feature family (using the extractor's group offsets), and 3 local waterfall plots: a true positive, a false positive and a false negative.
   - **Cross-tool agreement:** compute top-k family overlap between manual SHAP, FLAML's winner (SHAP if tree-based) and AutoGluon or H2O importance.
5. **Stretch goals done here:**
   - Calibration: reliability curve and Brier score.
   - Cost-sensitive threshold: PR curve and the threshold minimising `C_FN·FN + C_FP·FP`, with a stated default `C_FN/C_FP = 10`.
   - Drift monitor: retrain on a sliding 12-week window and plot F1 and AUC per future week.
6. **Limitations:** written to `reports/limitations.md`.

## Step 5: Deployment
- **Model choice:** pick the model by `test_future` performance and interpretability, most likely the tuned LightGBM. Export it with joblib together with a `schema.json` (feature dimension, group names, thrember version, chosen threshold).
- **`app/api.py` (FastAPI):**
  - `GET /health`.
  - `POST /predict`, which takes `{features: list[float]}`. Pydantic checks the length equals D and every value is finite, and the endpoint returns `{p_malicious, label, threshold}`.
  - Optional `POST /predict_file`, which runs thrember extraction on an uploaded PE. It is demoed only on benign system binaries.
- **`app/streamlit_app.py`:** pick a sample from `test_future` or upload a file, call the API, and show the probability plus top SHAP contributions.
- **`MODEL_CARD.md`:** data, metrics on both test sets, intended use, known limits, out-of-scope uses.
- **Cloud deploy** (Hugging Face Spaces) is a bonus left for last.

## Step 6: Report and slides (after review)
- PDF report built from the generated tables and figures (pdf skill), including an architecture diagram.
- A 5-minute deck (slides artifact or pptx).
- Both pull numbers from `reports/` only, never hand-typed.

## Reproducibility
`python run_all.py [--quick] [--steps data,tune,automl,eval,deploy]` regenerates everything from raw data with seed 42. `--quick` uses a 10k subsample, 4 trials and 60 s budgets for smoke testing.

## Verification
- **Data:** `python scripts/verify_data.py` shows the counts, dates and feature dimension and ends without errors.
- **Tests:** `pytest` covers split disjointness and temporal ordering, seed reproducibility (two quick runs give identical metrics), and API validation (wrong length or NaN gives 422, a valid vector gives 200).
- **Smoke run:** `python run_all.py --quick` runs end to end in about 15 minutes before the full run.
- **API:** `uvicorn app.api:app` plus a test request to `/predict` with a real `test_future` row returns the same probability as offline scoring.
- **UI:** the Streamlit page loads in the browser pane and shows a prediction.
- **Review gates (adversarial agents, see Orchestration):** after Steps 2, 3 and 4, flag anything suspicious, such as AUC > 0.999 on the random split, a train/test mismatch, or imbalance artefacts. I also give you a short summary after each workflow ends.

## Definition of done (the run ends only when every item is verified)
Each item must be backed by a file in the repo. An adversarial agent, not the agent that did the work, checks it.

| # | Done means | Evidence |
|---|---|---|
| D1 Reproducible | A fresh venv built from `requirements.txt` installs on Python 3.12 / Windows. `pytest` passes, with no skips when the data is present. `python run_all.py --quick` runs end to end from deleted quick outputs. Seed 42 everywhere. | test output, `reports/repro_check.md` |
| D2 Data | `reports/data_check.json` is clean: 0 sha256 overlap across splits, `test_future` is strictly later than `train`, the feature-alignment check passes, label -1 is dropped, and the thrember commit and feature dimension are recorded. | `data_check.json`, `splits_info.json` |
| D3 Step 2 | Rows for LGBM default, LogReg, Grid, Random and Optuna, each with CV AUC, both test sets, F1/acc/TPR@1%FPR and time. Comparison and convergence figures. The equal-budget definition is in the README. | `tables/tuning_summary.csv`, figures |
| D4 Step 3 | FLAML, AutoGluon and H2O ran on identical data with a 1200 s budget. Their leaderboards are saved. `automl_vs_manual.csv` and a chart exist. A written comparison covers the models tried, the winners, each tool's preprocessing, the time taken and interpretability. | `leaderboards/`, `tables/`, `reports/automl_comparison.md` |
| D5 Step 4 | Random vs future gap table. Stratified vs forward-CV comparison. Leaky `detection_ratio` and `first_submission_date` runs, with and without the feature, for all 4 model types, including importance rank. Bias slices with benign FPR. Class-weight variant. SHAP: beeswarm, family bar chart, 3 waterfalls. Cross-tool agreement. Calibration and Brier score. Cost-sensitive threshold. Drift plot. `limitations.md`. | `tables/`, `figures/`, `reports/critical_eval.md` |
| D6 Step 5 | `models/final/` holds the model, `schema.json` and the threshold. FastAPI `/health` and `/predict` validate input (tested). API and offline scores match. The Streamlit app runs and shows a prediction, with screenshots saved. `MODEL_CARD.md` and a `DEMO.md` demo script exist. | `tests/test_api.py`, `reports/screenshots/` |
| D7 Step 6 | `reports/report.pdf` has every section the brief lists, every number traces to a `reports/` file, and it renders correctly (pages checked as images). A 5-minute `.pptx` deck with speaker notes. `reports/qa_prep.md`. Notebooks executed with outputs. Report and deck are built by `scripts/build_report.py` and `build_slides.py`. | the files plus rendered page images |
| D8 Gate | `reports/SUBMISSION_CHECKLIST.md` maps every rubric item to evidence. Two consecutive adversarial rounds with zero blocker or major issues across four checks: reproducibility, rubric coverage, numbers and honesty, and code readability. After that I do my own spot-check. | checklist, the workflow's review log |

**Left for you (I can't do these):**
- Part A certificate.
- Team size, deadline and topic-approval TODOs.
- Recording the demo video.
- The cloud deploy, which needs your Hugging Face or Streamlit account.
- `git commit` and pushing to GitHub. I'll ask before doing either.

## Orchestration (replaces the human review stops)
You asked for adversarial agents instead of pausing for your review after Steps 2, 3 and 4. So every stage runs as **do → verify → fix → re-verify** until it passes:
- **Verifiers:** 2–3 independent agents, each with a distinct check. They are read-only and must give evidence (a command, a line or a number).
- **Fixers:** fix root causes and regenerate outputs from logs, never edit numbers by hand. They may dispute an issue with evidence, and the next round's verifiers see the dispute.
- **Real results are not bugs.** A large drift gap or AutoML falling for the leak is a finding to report honestly, not a defect to fix.
- **Cap:** at most 5 rounds per stage. If a stage hits the cap, its unresolved issues are logged and carried into the final gate, never dropped silently.
- **Workflow 1** (already running) covers Steps 0–2. **Workflow 2** starts when it finishes:
  - It assesses state and recovers anything incomplete.
  - Then it runs the verify/fix loops for Steps 0–2, then Steps 3 → 4 → 5 → 6.
  - Last comes the D8 gate.
- I update the workflow script as results come in, and I check the results myself between workflows.

## Open TODOs (yours, not blocking)
Team size, deadline and topic approval stay as `TODO` in the README.
