# Submission checklist

This file maps every item of the brief (`docs/PROJECT_BRIEF.md`) and every "Definition of
done" item of the plan (`docs/PLAN.md`, D1-D8) to the files that show it is done. It holds
no results: every number lives in a generated file under `reports/` (or `data/splits/`,
`models/`), and the paths below point to them. Paths are relative to the repository root.

Status key:
* **Done**: the evidence file exists and was produced by the project's scripts.
* **Done, with a stated deviation**: done, but differently from the brief. The deviation and
  its reason are written in the linked file.
* **User**: only the student can do it (accounts, admin, video). See the last section.

## Brief, Section 3: Dataset

| Brief item | Status | Evidence |
|---|---|---|
| Check the exact version, size, license and download method, and document what was used | Done | `README.md` (Dataset, Download), `scripts/download_data.py` (`download_zip` verifies each zip against its Hugging Face sha256), `reports/data_check.json` (`download`: per-file line-count checks; `cache_info`: thrember commit, feature dimension) |
| Drop unlabeled rows (label = -1) | Done | `reports/data_check.json` (`labels_only_0_1`, `hard_checks`), `src/data.py` |
| Use the released `.jsonl` files; keep the date metadata for the temporal split | Done | `src/data.py` (cache builder, `first_submission_date` and week in the metadata), `data/splits/splits_info.json` |
| Stratified subsample of 150k-250k rows, the same one for every method, size and seed recorded | Done | `data/splits/splits_info.json` (seed 42, sizes), `config.py` (`SUBSAMPLE_N`, `SEED`), `tests/test_splits.py` |

## Brief, Section 4: Environment and constraints

| Brief item | Status | Evidence |
|---|---|---|
| Virtual environment with a pinned `requirements.txt` | Done | `requirements.txt`; fresh-venv install in `reports/repro_check.md` |
| FLAML and AutoGluon (primary), H2O optional | Done (all three) | `reports/leaderboards/`, `reports/automl_comparison.md` |
| One global `SEED = 42` for numpy, random, the model libraries and every split | Done | `config.py` (`SEED`, `set_seed`), `src/tune.py` lines 107-111 (LightGBM `seed`) and 361-363 (Optuna samplers), `data/splits/splits_info.json`; `reports/experiments.jsonl` records the seed only on the AutoML and leakage `final_eval` records and the `explain_oof` and `threshold` records (19 of 159), not on the Step 2 tuning records; seeds the AutoML tools ignore are listed in `reports/automl_comparison.md` |
| Same fixed AutoML budget (20 min) for every tool; wall-clock time logged for every method | Done | `config.py` (`AUTOML_BUDGET_S`), `reports/tables/automl_vs_manual.csv`, `reports/tables/tuning_summary.csv` (time columns), `reports/experiments.jsonl` |
| Hardware | Done, with a stated deviation | Steps 2-3 and 5-6 ran on the laptop; Step 4 on a 4-core / 15.7 GiB Linux cloud VM (the challenge-set scoring alone on the laptop). All CPU, no GPU. `README.md` (Environment), `reports/limitations.md` (Compute, hardware and budgets), `reports/report.pdf` section 10.1 |

## Brief, Step 1: Problem and dataset

| Brief item | Status | Evidence |
|---|---|---|
| Problem statement | Done | `README.md` (Research question), `reports/report.pdf` section 1 |
| Dataset source, class balance | Done | `README.md` (Dataset), `reports/tables/class_balance_by_week.csv`, `reports/tables/class_balance.csv`, `reports/report.pdf` section 2 |
| Feature groups (histogram, byte-entropy, strings, general, header, section, imports, exports, data directories, ...) | Done | `src/data.py` (`feature_groups()`), `notebooks/01_eda.ipynb`, `reports/report.pdf` section 2.3 |
| Subsample documented | Done | `data/splits/splits_info.json`, `README.md` (Design decisions) |
| Topic approval by the instructor | **User** | `README.md` (Team and admin) |

## Brief, Step 2: Baseline and hyperparameter tuning (5 marks)

| Brief item | Status | Evidence |
|---|---|---|
| Stratified random split with a held-out test set never used in tuning, plus a time-based split | Done | `data/splits/` (train, test_random, test_future rows), `reports/data_check.json` (`splits`: no sha256 overlap, test_future strictly later), `tests/test_splits.py` |
| Manual baseline (LightGBM; logistic regression as a floor) | Done | `reports/tables/tuning_summary.csv` (rows `lgbm_default`, `logreg`) |
| Grid, Random and Bayesian (Optuna) search on the same model | Done | `reports/tables/tuning_summary.csv`, `reports/tables/tuning_trials.csv`, `src/tune.py` |
| F1, ROC-AUC, accuracy and time for each method; comparison table and chart | Done | `reports/tables/tuning_summary.csv`, `reports/tables/tuning_pairwise.csv`, `reports/figures/tuning_compare.png`, `reports/figures/tuning_convergence.png`, `reports/tuning_writeup.md` |
| Same CV folds and the same budget framing; say what "equal budget" means | Done | `data/splits/cv_fold_ids.npy`, `README.md` ("Equal budget for Grid, Random and Optuna"), `reports/tuning_writeup.md` |

## Brief, Step 3: AutoML and tool comparison (10 marks)

| Brief item | Status | Evidence |
|---|---|---|
| At least two frameworks on the same data with the same time budget | Done (FLAML, AutoGluon, H2O at 1200 s) | `reports/run_automl.log`, `reports/tables/automl_vs_manual.csv` (`budget_seconds`, `wall_seconds`) |
| Study each leaderboard: models tried, the winner, preprocessing and feature engineering; leaderboards saved | Done | `reports/leaderboards/` (per tool: leaderboard, fit/search/event log, preprocessing JSON, importances), `reports/automl_comparison.md` |
| AutoML vs manual on F1/accuracy, training time and interpretability, with an honest comment | Done | `reports/tables/automl_vs_manual.csv`, `reports/tables/automl_pairwise.csv`, `reports/figures/automl_vs_manual.png`, `reports/automl_comparison.md`, `reports/report.pdf` sections 5 and 8 |

## Brief, Step 4: Critical evaluation (5 marks)

| Brief item | Status | Evidence |
|---|---|---|
| Cross-validation plus a held-out test set; random vs chronological split; gap quantified | Done | `reports/tables/random_vs_future.csv` (paired-bootstrap CIs), `reports/tables/cv_comparison.csv` (stratified vs week-blocked forward CV), `reports/tables/gap_*.csv`, `reports/temporal_bias.md`, `reports/critical_eval.md` sections 1 and 3 |
| A deliberate leaky feature: does each AutoML tool exploit it (score jump, importance, SHAP)? Remove it and show the drop | Done, with a stated deviation (600 s per leakage run instead of 1200 s, on the cloud VM; clean reruns at the same 600 s) | `reports/tables/leakage.csv` (`detection_ratio` and `first_submission_date`, with and without, for tuned LightGBM, FLAML, AutoGluon and H2O, with importance ranks), `reports/figures/leakage_jump.png`, `reports/leaderboards/leakage/` (incl. `log_scan.json`: did a tool warn or drop it), `reports/leakage.md`, `reports/tables/time_proxy_ranks.csv` (time-proxy ablation) |
| Bias and imbalance: class balance, per-slice performance, FPR on benign files; class weights, and whether they helped | Done | `reports/tables/class_balance.csv`, `reports/tables/bias_slices.csv` (week, file size, packer, signed, entropy, family; benign FPR per slice), `reports/figures/bias_slices.png`, `reports/tables/class_weight.csv`, `reports/tables/prevalence.csv`, `reports/tables/challenge.csv`, `reports/temporal_bias.md` |
| Explainability: SHAP TreeExplainer on the best tree model, summary plot plus local explanations, grouped by feature family | Done | `reports/figures/shap_beeswarm.png`, `reports/figures/shap_family_bar.png`, `reports/figures/shap_waterfall_{tp,fp,fn}.png`, `reports/tables/shap_family_importance.csv`, `reports/tables/shap_cases.csv`, `reports/explainability.md` |
| Limitations (static features, packed files, drift, dataset age, subsampling, adversarial evasion) | Done | `reports/limitations.md` (each of these has its own bullet), `reports/report.pdf` section 9 |

## Brief, Step 5: Deployment (5 marks)

| Brief item | Status | Evidence |
|---|---|---|
| Export the best model with its preprocessing, so inference is self-contained | Done | `models/final/model.txt` (LightGBM native format), `models/final/schema.json` (feature order, preprocessing, threshold, sha256, parity check), `scripts/export_model.py` |
| FastAPI `/predict` with input validation | Done | `app/api.py` (`/health`, `/predict`, `/predict_batch`, `/predict_file`), `tests/test_api.py` (422 on wrong length, non-finite or non-numeric input; 400 on a bad upload; size caps; uploads never written to disk or executed; API scores equal offline scores) |
| Streamlit front end for the demo | Done | `app/streamlit_app.py`, `reports/screenshots/` (3 screenshots), `tests/test_api.py::test_streamlit_app_runs_in_process` |
| Local deployment | Done | `DEMO.md` (commands to start both servers) |
| Cloud deployment (bonus) | **User** | Needs the student's Hugging Face or Streamlit account; `app/requirements.txt` is ready for it |
| Working prediction demo | Done locally; the recording is **User** | `DEMO.md` (5-minute script), `reports/screenshots/` |
| Model card: data used, metrics, intended use, known limits, out-of-scope uses | Done | `MODEL_CARD.md` (generated by `app/model_card.py`) |

## Brief, Step 6: Documentation and presentation (5 marks)

| Brief item | Status | Evidence |
|---|---|---|
| PDF report: problem and dataset, workflow/architecture diagram, experiment tables, screenshots, tool comparison, leakage and bias findings, reflections on AutoML | Done | `reports/report.pdf` (sections 1-3, 4-5, 6, 7 with screenshots, 8 reflections), `reports/figures/architecture.png`; built by `scripts/build_report.py` |
| Slides for a 5-minute demo plus short Q&A | Done | `reports/slides.pptx` (with speaker notes; built by `scripts/build_slides.py`), `reports/qa_prep.md` |

## Brief, Section 6: Beyond the rubric (stretch goals)

| Stretch goal | Status | Evidence |
|---|---|---|
| 1. Drift monitor: sliding-window retrain, F1 against time | Done | `reports/tables/drift.csv` (expanding, sliding 12-week and size-matched windows), `reports/figures/drift.png`, `reports/temporal_bias.md` |
| 2. Cost-sensitive threshold with an explicit cost ratio, precision/recall trade-off | Done | `models/threshold.json` (C_FN/C_FP = 10, chosen on train out-of-fold scores), `reports/tables/threshold.csv`, `reports/figures/pr_cost.png`, `reports/explainability.md` |
| 3. Calibration: reliability curve and Brier score | Done | `reports/tables/calibration.csv`, `reports/figures/calibration.png`, `reports/explainability.md` |
| 4. Reproducibility package (`run_all.py`) | Done | `run_all.py` (steps data, tune, automl, eval, deploy), `reports/repro_check.md` |
| 5. Cross-tool agreement of the features SHAP flags | Done | `reports/tables/cross_tool_agreement.csv`, `reports/tables/cross_tool_top20.csv`, `reports/tables/cross_tool_family_shares.csv`, `reports/figures/cross_tool_agreement.png` |
| 6. Optional cloud deployment | **User** | See Step 5 |

## Brief, Section 8: Deliverables

| Deliverable | Status | Evidence |
|---|---|---|
| Course completion certificate (Part A) | **User** (out of scope for this repo) | |
| Project code on GitHub or as notebooks, runnable and reproducing the results | Done locally; pushing is **User** | `run_all.py`, `src/`, `scripts/`, `tests/`, `notebooks/01_eda.ipynb` to `04_critical_eval.ipynb` (executed, with outputs), `reports/repro_check.md` |
| Project report (PDF) | Done | `reports/report.pdf` |
| Deployed app link or demo recording | **User** (the app runs locally; the link or the video needs the student) | `DEMO.md`, `reports/screenshots/` |
| Presentation slides | Done | `reports/slides.pptx`, `reports/qa_prep.md` |
| Model card | Done | `MODEL_CARD.md` |

## Brief, Section 9: Grading guidance and ground rules

| Rule | Status | Evidence |
|---|---|---|
| Honest reporting; negative or surprising results reported | Done | `reports/critical_eval.md` (the small tree gap, AutoML taking the leak, the challenge set), `reports/limitations.md`, `reports/automl_comparison.md` (FLAML and H2O over budget), `README.md` (Data quirks) |
| Fixed random seed everywhere | Done | `config.py` (`SEED`, `set_seed`), `src/tune.py` lines 107-111 and 361-363, `data/splits/splits_info.json`; the seed appears in `reports/experiments.jsonl` only on the AutoML/leakage `final_eval`, `explain_oof` and `threshold` records |
| Never tune on the test set; the test sets stay untouched until the final evaluation | Done | `src/tune.py` (tuning sees only CV folds of train; the only test scoring is after the refit), `src/calibration.py` (threshold and calibrators from train out-of-fold scores), `README.md` (Two test sets), `tests/test_splits.py` |
| Comment non-obvious decisions; a "why" note per major choice | Done | `README.md` (Design decisions and why), "why" comments in `src/` |
| Log every experiment (method, parameters, score, time); tables generated from logs | Done | `reports/experiments.jsonl`; `src/tables.py`, `src/*_report.py`, `src/*_writeup.py` build every table and write-up from it and from `reports/predictions/` |

## Plan, Definition of done (D1-D8)

| Item | Status | Evidence |
|---|---|---|
| D1 Reproducible: fresh venv installs on Python 3.12 / Windows; `pytest` passes with no skips when the data is present; `run_all.py --quick` runs end to end from deleted quick outputs; seed 42 | Done | `reports/repro_check.md` |
| D2 Data: 0 sha256 overlap, test_future strictly later, feature alignment, label -1 dropped, thrember commit and feature dimension recorded | Done | `reports/data_check.json` (`ok: true`, `hard_checks`), `data/splits/splits_info.json`, `data/splits/split_fingerprints.json` |
| D3 Step 2: LGBM default, LogReg, Grid, Random, Optuna with CV AUC, both test sets, F1/acc/TPR@1%FPR and time; comparison and convergence figures; equal budget in the README | Done | `reports/tables/tuning_summary.csv`, `reports/figures/tuning_compare.png`, `reports/figures/tuning_convergence.png`, `README.md` |
| D4 Step 3: FLAML, AutoGluon, H2O on identical data at 1200 s; leaderboards; `automl_vs_manual.csv` and chart; written comparison | Done | `reports/leaderboards/`, `reports/tables/automl_vs_manual.csv`, `reports/figures/automl_vs_manual.png`, `reports/automl_comparison.md` |
| D5 Step 4: gap table, CV comparison, both leaks with and without for all 4 model types with importance rank, bias slices with benign FPR, class weights, SHAP (beeswarm, family bar, 3 waterfalls), cross-tool agreement, calibration and Brier, cost threshold, drift plot, `limitations.md` | Done, with the stated 600 s / cloud-VM deviation | See the Step 4 and stretch-goal rows above; `reports/critical_eval.md`, `reports/limitations.md` |
| D6 Step 5: `models/final/` (model, `schema.json`, threshold); `/health` and `/predict` validate input (tested); API and offline scores match; Streamlit runs and shows a prediction, with screenshots; `MODEL_CARD.md`, `DEMO.md` | Done | `models/final/`, `tests/test_api.py`, `reports/screenshots/`, `MODEL_CARD.md`, `DEMO.md` |
| D7 Step 6: `report.pdf` with every section the brief lists, numbers traced to `reports/`, pages checked as images; 5-minute `.pptx` with speaker notes; `qa_prep.md`; notebooks executed; both built by scripts | Done | `reports/report.pdf` (`scripts/build_report.py`; page images checked as recorded in `reports/repro_check.md` section 4), `reports/slides.pptx` (`scripts/build_slides.py`), `reports/qa_prep.md`, `notebooks/*.ipynb` |
| D8 Gate: this checklist; two consecutive adversarial rounds with no blocker or major issue (reproducibility, rubric coverage, numbers and honesty, readability); then a spot-check | Done, with a relaxed bar the student chose for the deadline: one clean round instead of two, and at most 2-3 rounds per stage with leftovers documented here. Final gate rounds 1 and 2 each found 2 major issues, which were fixed; round 3 found no blocker or major issue (3 minor, below). Spot-check afterwards: pytest 160 passed, 0 skipped; API `/predict` equals offline `model.predict` on 3 demo rows (difference 0.0) and a 5-value vector returns 422; all 30 report pages rendered and viewed | This file; the workflow's review log |

## Issues carried from earlier review gates

| Issue | Resolution | Evidence |
|---|---|---|
| Step 4 (major, reported by three reviewers): `temporal_bias.md` section 7 said AutoGluon "could not be scored" on the challenge set "(its predictor is not on disk)", which contradicted section 6 of the same file | Resolved. The sentence was a hard-coded literal in the template; `src/temporal_writeup.py` now builds the clause from `facts['challenge_scoring']['not_scored']`, so a rebuild cannot bring it back | `grep -rn "not on disk" reports/*.md` finds nothing; the Limits bullet of `reports/temporal_bias.md` section 7 now ends "...measures misses, not false alarms." |
| Step 4 (deferred to the README owner): the README environment table gave the Step 4 H2O heap as 9G and titled the column "Step 4 onwards"; the status block said Steps 5-6 were not started | Resolved in `README.md` (Environment, Status) | `README.md` |
| Found by the final reproducibility check: `run_all.py --quick` overwrote the real `MODEL_CARD.md` with the quick model's card (`scripts/export_model.py` read its own `--quick` flag, which `run_all.py` does not pass, instead of `EMBER_QUICK`) | Fixed: the card's location now follows `config.QUICK`; `MODEL_CARD.md` was restored and the quick deploy step re-run | `reports/repro_check.md` section 3 |
| Found by the final reproducibility check: a venv on a deep Windows path fails to install (`WinError 206`, path too long) | Documented: no pin is wrong; `README.md` (Environment) says to keep the venv on a short path | `reports/repro_check.md` section 1 |
| Final gate round 3 (minor): report section 10 said every experiments.jsonl record logs the machine; 128 of 159 do not | Resolved: the sentence now says only Step 4 records log the machine (`scripts/build_report.py`); PDF rebuilt | `pdftotext -f 27 -l 27 reports/report.pdf -` shows "Step 4 records also log the machine" |
| Final gate round 3 (minor): `leakage.md` said H2O's date deltas were "more likely" search noise, stronger than `critical_eval.md` section 4 | Resolved: the wording in `src/leakage_writeup.py` now says the deltas may be the search taking another path and that run-to-run variation was not measured; `run_leakage.py --tables-only` regenerated `leakage.md` (tables unchanged) | `reports/leakage.md` H2O paragraph |
| Final gate round 3 (minor): RAM is written as "15 GB" in `docs/handoff/HANDOFF.md` and "15.7 GB" in `explainability.md`/`leakage.md`, but 15.7 GiB elsewhere; eval records log `ram_gb` in decimal GB (16.9) and leakage/explain records in GiB (15.7) under the same key | Not fixed (cosmetic, deadline). `scripts/build_report.py` explains the two units in the report | `reports/experiments.jsonl` `machine.ram_gb` values |

## Left for the student (cannot be done from this repository)

1. **Part A certificate** (online certification, handled outside this repo).
2. **Team size, deadline and topic approval**: the three `TODO` lines in `README.md`
   (Team and admin), and the instructor's approval of the topic. The report's cover page
   also says "TODO: team members": fill in the names at the `"Team members"` line of
   `title_page()` in `scripts/build_report.py` (and, if wanted, on slide 1 in
   `scripts/build_slides.py`), then rerun both build scripts.
3. **Recording the demo video**: follow `DEMO.md` (about 5 minutes).
4. **Cloud deployment** (bonus): needs the student's own Hugging Face or Streamlit
   Community Cloud account. `app/requirements.txt` lists what the app needs.
5. **Pushing to GitHub** (`https://github.com/Turbo-Soggy/automl`) and opening the pull
   request: the work is on the branch `step4-6-cloud`. Before pushing, check that everything is
   committed (`git status` clean), including this checklist, `reports/repro_check.md` and
   `reports/repro_logs/`, which `repro_check.md` links to.
