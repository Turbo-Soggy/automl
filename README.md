# Malware detection on EMBER2024: manual tuning vs AutoML

Course project (IA-3, Part B). We train binary classifiers that label Windows
64-bit PE files as **benign (0)** or **malicious (1)** from static features.
Then we compare hand-tuned LightGBM (grid, random and Optuna search) with three
AutoML tools (FLAML, AutoGluon, H2O).

> **Status (2026-10-29):** Steps 0-3 (environment, data and splits, baselines and
> tuning, AutoML) are done and passed adversarial review. Step 4 (critical
> evaluation) is being redone from scratch, and Steps 5-6 are not started. See
> [`docs/handoff/HANDOFF.md`](docs/handoff/HANDOFF.md) for the current state and
> how to continue on a fresh machine. No results are typed into this README. All
> numbers live in `reports/` and are generated from the experiment log and the
> saved test-set predictions.

## Research question

> How much do random-split scores overstate real-world (future) performance on
> malware detection, and do AutoML tools protect against leakage and temporal
> drift, or do they fall for them?

To answer it, every model is scored on two test sets:

* `test_random` is a random hold-out from the same weeks as the training data.
  This is the classic, optimistic setup.
* `test_future` is drawn from the 12 official test weeks that come after every
  training week. This is closer to deployment, where the malware is new.

The gap between the two scores measures how much the random split overstates
performance. Step 4 also adds a known leaky feature (`detection_ratio`) and a
time-proxy feature (`first_submission_date`) on purpose, to see which tools
catch them.

## Dataset

| | |
|---|---|
| Name | EMBER2024 (Joyce et al., KDD 2025), Win64 PE slice |
| Source | Hugging Face dataset `joyce8/EMBER2024` |
| Feature extractor | `thrember` 0.1.0 at commit `0ef753e81d98bf209f71b03cd331dfc190b5b54d` (2025-08-22) of `github.com/FutureComputing4AI/EMBER2024` |
| License | Apache-2.0 (see `third_party/EMBER2024/LICENSE` after setup) |
| Files used | `Win64_train.zip` (5.85 GB), `Win64_test.zip` (1.18 GB), `challenge.zip` (32 MB, all file types) |
| On disk | 6.6 GB of zips under `data/raw/_zips/`, plus 32 GB of extracted jsonl unless `--keep-zipped` is used (see Download) |
| Weeks | 52 training weeks (2023-09-24 to 2024-09-21) and 12 test weeks (2024-09-22 to 2024-12-14) |
| Unique samples | 10,000 per week (5,000 benign, 5,000 malicious), which is 520,000 train and 119,993 test after the global sha256 dedup (120,000 before; see quirk 2) |
| Feature vector | 2,568 float32 columns in 12 families (general, histogram, byteentropy, strings, header, section, imports, exports, datadirectories, richheader, authenticode, pefilewarnings) |

Citation:

```
@inproceedings{joyce2025ember,
  title     = {EMBER2024 - A Benchmark Dataset for Holistic Evaluation of Malware Classifiers},
  author    = {Robert J. Joyce and Gideon Miller and Phil Roth and Richard Zak and
               Elliott Zaresky-Williams and Hyrum Anderson and Edward Raff and James Holt},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2025}
}
```

### Download

The commands in this README use the Windows interpreter path. On Linux or macOS use
`.venv/bin/python` in place of `.venv/Scripts/python.exe` (and `python3.12 -m venv .venv`
in place of `py -3.12 -m venv .venv`).

```
.venv/Scripts/python.exe scripts/download_data.py            # Win64 train + test + challenge
.venv/Scripts/python.exe scripts/download_data.py --no-challenge
.venv/Scripts/python.exe scripts/download_data.py --keep-zipped  # do not extract the Win64 zips (32 GB less disk)
```

`run_all.py --steps data` calls the same script (and forwards `--keep-zipped`). It
skips the download when `data/raw/DOWNLOAD_DONE` already exists (pass
`--redownload` to force it).

With `--keep-zipped`, `src.data` streams each weekly jsonl straight out of its zip
(`src.data.RawFile`). The bytes and byte offsets are identical to the extracted
files, so the cache and the splits are the same either way. Extracted files take
precedence when they exist; a partial extraction is an error.

We do **not** use `thrember.download_dataset()`. It calls `os.chdir()` into the
data folder, which would make untrusted downloaded data the working directory
of our Python process, and it deletes each zip without checking its hash. Our
script downloads the same files from the same Hugging Face repo, but it:

* checks every zip's sha256 against the hash published on Hugging Face,
* extracts only flat `*.jsonl` members, rejecting absolute paths and `..`,
* never runs anything from `data/raw/`, and
* can resume after an interruption.

### Data quirks found during setup (flagged, not hidden)

1. **Every weekly file contains each sample twice.** Each file has 20,000 lines,
   and lines 10,001-20,000 repeat lines 1-10,000 (same sha256, same order, same
   features and label; only the capa outputs `caps`/`ttps`/`mbc` can differ).
   Without deduplication, a random split would put identical feature vectors in
   both train and `test_random`, which is direct leakage. The cache builder
   therefore keeps only the first occurrence of each sha256. After
   deduplication there are 639,993 unique sha256 values.
2. **7 sha256 values appear in two different test weeks.** No sha256 appears in
   both train and test. Deduplication is global, and `tests/test_splits.py`
   checks that no sha256 is shared between splits.
3. **`detection_ratio` is an almost perfect label proxy.** It is the share of
   antivirus engines that flag the file, which is essentially how the label was
   made. It is excluded from `X` in every main run and only used in the Step 4
   leak experiment.
4. **The standard feature vector already contains time proxies.** These are the
   PE compile timestamp (column 696, `header.coff.timestamp`) and the
   authenticode signing time. Expect them to rank high in importance and to
   behave differently on random vs future splits.
5. **Classes are exactly 50/50 every week by construction.** That is not
   realistic prevalence, so precision on real traffic would be lower. This is
   discussed in the bias section.
6. **The challenge set is not Win64-only** (814 of 6,315 rows are Win64), and
   every row is malicious, so only recall/TPR is meaningful on it.
7. **Dependency fix:** `signify` 0.9.x removed a class that `thrember` imports,
   so `import thrember` fails. `requirements.txt` pins `signify==0.8.1`.

## Environment

Two machines were used. Wall-clock times are only compared between runs on the
same machine.

| | Steps 0–3 (data, tuning, AutoML) | Step 4 onwards (and the cloud data rebuild) |
|---|---|---|
| CPU | Intel Core Ultra 9 285H, 16 cores | Intel Xeon @ 2.1 GHz, 4 vCPU (cloud VM) |
| RAM / disk | 63 GB, about 640 GB free | 15 GB (no swap), about 13 GB free |
| OS | Windows 11 | Linux |
| `N_JOBS` / H2O memory | 16 / 24G | 4 / 9G |

No GPU was used on either (see "all-CPU" below). `config.N_JOBS` is
`min(16, cpu_count)` and the H2O memory is capped at about 60% of RAM, both
computed automatically; every record in `reports/experiments.jsonl` logs its own
`n_jobs`. The data rebuilt on the cloud VM matches the laptop exactly (split
sha256 order and feature hashes in `data/splits/split_fingerprints.json`).

Software: Python 3.12 (AutoGluon and H2O do not support 3.14) and Java 21
(needed by H2O).

```
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins exact versions frozen from the working venv, including
`thrember` at the commit above and a CPU-only PyTorch wheel (pulled in by
AutoGluon). `nbformat`, `nbconvert` and `nbclient` are pinned too, so the
notebooks can be executed headless (`jupyter nbconvert --execute`).

## Reproducing

From the project root (on Linux or macOS, `.venv/bin/python` instead of
`.venv/Scripts/python.exe`):

```
.venv/Scripts/python.exe run_all.py --quick                   # smoke run on tiny samples
.venv/Scripts/python.exe run_all.py                           # full run
.venv/Scripts/python.exe run_all.py --steps data,tune         # only some steps
.venv/Scripts/python.exe -m pytest tests                      # checks on the persisted splits
.venv/Scripts/python.exe scripts/run_tuning.py --refit-only   # reuse logged searches: refit + score only
.venv/Scripts/python.exe scripts/run_tuning.py --tables-only  # rebuild Step 2 tables, figures, write-up
.venv/Scripts/python.exe scripts/run_automl.py                # Step 3: FLAML, AutoGluon, H2O, then tables (~1.5 h)
.venv/Scripts/python.exe scripts/run_automl.py --tools h2o    # re-run one AutoML tool, then rebuild tables
.venv/Scripts/python.exe scripts/run_automl.py --tables-only  # rebuild Step 3 tables, figure, write-up
.venv/Scripts/python.exe scripts/run_leakage.py              # Step 4A: leakage experiments (~3 h), then outputs
.venv/Scripts/python.exe scripts/run_leakage.py --skip-done  # resume: only runs not yet logged
.venv/Scripts/python.exe scripts/run_leakage.py --tables-only  # rebuild leakage table, figure, write-up
.venv/Scripts/python.exe scripts/run_automl.py --refresh-notes  # re-derive notes from saved models (no refit)
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/02_baseline_tuning.ipynb
```

* `--steps` takes a comma-separated subset of `data,tune,automl,eval,deploy`.
  Steps always run in that order. `eval` and `deploy` are not
  implemented yet and say so.
* `--quick` sets `EMBER_QUICK=1`. That shrinks every size in `config.py` (10k
  pool, 2k future test, 5k tuning subset, 4 trials, 60 s AutoML budget) and
  writes to `*_quick` split and log files, so a smoke run never overwrites real
  results.
* **`--quick` still needs the full download and the full feature cache.** Only
  the splits, models and logs are small: the quick splits are drawn from the same
  `data/cache/` that the full raw data builds (6.6 GB of zips, plus 32 GB of
  extracted jsonl unless `--keep-zipped`, then the full cache build). So on a fresh machine,
  run `run_all.py --steps data` once (or `run_all.py --quick`, which runs the data
  step first) before expecting a fast smoke run.
* `scripts/run_tuning.py --refit-only` does not search again. It rebuilds each
  tuner's latest search from `reports/experiments.jsonl`, refits the winner and
  scores it: minutes instead of the hours the three searches took. The
  baselines have no search and are always re-run (they are cheap).
* Every child process of one `run_all.py` call gets the same `EMBER_RUN_ID`,
  so its records can be grouped in `reports/experiments.jsonl`.

| Step | Scripts run by `run_all.py` | Main outputs |
|---|---|---|
| `data` | `scripts/download_data.py`, `scripts/build_cache.py`, `scripts/verify_data.py` | `data/raw/`, `data/cache/`, `data/splits/` (incl. `splits_info.json`), `reports/data_check.json`, `reports/tables/class_balance_by_week.csv` |
| `tune` | `scripts/run_tuning.py` | `reports/experiments.jsonl`, `reports/predictions/*.npz` (test-set scores), `reports/tables/tuning_summary.csv` (with 95% CIs), `tuning_pairwise.csv`, `tuning_trials.csv`, `tuning_refit_versions.csv`, `reports/figures/tuning_compare.png`, `reports/figures/tuning_convergence.png`, `reports/tuning_writeup.md` |
| `automl` | `scripts/run_automl.py` (one child process per tool, with a sleep guard) | `reports/leaderboards/` (per tool: leaderboard, fit/search/event log, preprocessing notes JSON with saved-model checks, importances), `reports/predictions/automl_*.npz`, `reports/tables/automl_vs_manual.csv`, `reports/tables/automl_pairwise.csv`, `reports/figures/automl_vs_manual.png`, `reports/automl_comparison.md`, `models/flaml/`, `models/autogluon/`, `models/h2o/` |
| `eval`, `deploy` | not implemented yet in `run_all.py`; draft Step 4A (leakage) code runs on its own with `scripts/run_leakage.py` (unreviewed, outputs not yet produced) | `reports/tables/leakage.csv`, `reports/figures/leakage_jump.png`, `reports/leakage.md`, `reports/leakage_logs/`, `reports/leaderboards/leakage/`, `reports/predictions/leakage/`, `models/leakage/` |

## Repository layout

```
config.py              seed, paths and experiment sizes (single source of truth)
run_all.py             runs the pipeline steps in order
requirements.txt       pinned dependencies
src/
  data.py              cache loading, splits, CV folds, feature groups
  eval.py              compute_metrics(): AUC, F1, accuracy, precision, recall, FPR, TPR@1%FPR;
                       paired_bootstrap() / percentile_ci() for 95% confidence intervals
  logging_utils.py     log_experiment() / read_experiments() over reports/experiments.jsonl
  tune.py              baselines, Grid / Random / Optuna search, refit and the only test scoring
  tables.py            Step 2 and 3 tables and figures, built from the log and saved predictions only
  tuning_writeup.py    renders reports/tuning_writeup.md from the Step 2 tables
  automl_runs.py       Step 3: FLAML, AutoGluon, H2O runners, sleep guard, the only Step 3 test scoring
  automl_checks.py     facts read back from the saved AutoML models (seeds used, metalearner weights, ...)
  automl_writeup.py    renders reports/automl_comparison.md from the Step 3 tables and leaderboards
  leakage.py           Step 4A: leak / ablation variants of X, time-proxy columns from thrember's source,
                       tuned-LightGBM and AutoML reruns, importance ranks and TreeSHAP of the leak column
  leakage_report.py    Step 4A table, figure and log scan (did a tool warn about or drop the leak?)
  leakage_writeup.py   renders reports/leakage.md from the leakage table and log records
scripts/
  download_data.py     verified download of the raw jsonl
  probe_raw.py         looks at the first rows of one remote file (setup only)
  build_cache.py       raw jsonl -> deduplicated feature matrix + metadata
  verify_data.py       data checks -> reports/data_check.json
  run_tuning.py        Step 2 entry point
  run_automl.py        Step 3 entry point (--tools, --tables-only, --refresh-notes, --quick)
  run_leakage.py       Step 4A entry point (--runs, --skip-done, --tables-only, --quick)
tests/
  conftest.py          puts the project root on sys.path; skip helpers
  test_splits.py       disjointness, temporal order, sizes, balance, CV folds, feature groups
  test_tune.py         tuners, seeding, refit rule, bootstrap, tables (synthetic data, no cache needed)
  test_automl.py       Step 3 tables, pairwise rows, FLAML log parser, sleep guard (synthetic data)
  test_leakage.py      Step 4A variants, time-proxy positions in thrember's source, ranks, SHAP, log scan
notebooks/
  01_eda.ipynb                 narrative EDA (imports from src, no own logic)
  02_baseline_tuning.ipynb     narrative view of the Step 2 tables and figures
reports/               generated tables, figures, leaderboards, experiment log, predictions/, write-ups
data/                  raw, cache, splits (gitignored, regenerated)
models/                saved models (binaries gitignored)
app/                   API and UI (Step 5)
third_party/EMBER2024  thrember source at the pinned commit (gitignored)
```

## Design decisions and why

**EMBER2024 rather than EMBER2018.** EMBER2024 has a real
`first_submission_date` per sample, so the "future" test set is genuinely
later in time rather than simulated. It has a real `detection_ratio` field,
which gives an honest leaky feature for the leakage experiment instead of a
synthetic one. Its feature extractor (`thrember`) is built on `pefile` and
maintained, whereas the 2018 extractor depends on an old LIEF version.

**Win64 slice, 200k subsample, seed 42.** Win64 is one coherent file type, so
its feature distribution is consistent. The 200k pool drawn from the 52
training weeks is stratified by label and week, so every week keeps its share.
It is large enough for stable AUC and TPR@1%FPR estimates, while each AutoML
tool can still fit several models within 20 minutes on 16 cores (the Steps 2–3 laptop). `SEED=42` is
passed to every split, sampler and model, because libraries keep their own
random generators and ignore the global ones.

**Two test sets: random vs future weeks.**

* `train` (160k) and `test_random` (40k) are a stratified 80/20 split of the
  200k pool from the training weeks.
* `test_future` (40k) is a stratified sample of the 12 later test weeks.
* The same trained models are scored on both test sets. The difference is the
  temporal drift, which is the core of the research question.
* Neither test set is used for any tuning or model selection. Tuning only sees
  cross-validation folds inside `train`.

**Same folds for everyone.** The 3-fold `StratifiedKFold(shuffle=True,
random_state=42)` indices are saved to `data/splits/`. Every tuning method
loads the same file, so differences between methods cannot come from
different folds.

**LightGBM as the single manual model family.** It is the standard EMBER
baseline, it is fast on 2,568 dense features, it handles unscaled and
categorical-like columns without preprocessing, and SHAP's TreeExplainer gives
exact explanations for it. Using one family keeps the comparison about
*tuning methods*, not about model families. Logistic regression on
standardized features is included only as a sanity floor.

**Equal budget for Grid, Random and Optuna.** Each method evaluates exactly
36 configurations. Each configuration is scored with the same 3 CV folds, on
the same 60k stratified tuning subset of `train`. Every fold uses the same cap
on boosting rounds with early stopping. Wall-clock time is measured and
reported, but not capped: a time cap would make the result depend on machine
load, while a trial count is exactly reproducible. Tuning on 60k rather than
160k keeps 3 methods × 36 configurations × 3 folds tractable. The grid is
3 × 3 × 2 × 2 = 36 points over `num_leaves`, `learning_rate`,
`min_child_samples` and `feature_fraction`. Random Search and Optuna sample the
same 6-parameter space.

**Random and Optuna share their first 10 trials.** Optuna's TPE sampler starts
with 10 seeded random trials (`TPE_STARTUP_TRIALS`, Optuna's default), drawn
with the same seed as Random Search. So the first 10 configurations of the two
methods are identical, and they differ only in the remaining 26 of 36 trials.
Every comparison of their efficiency (best trial number, time to reach a
score) has to be read with that in mind; the convergence figure shades the
shared trials.

**Refit rule (refit version 2).** The best configuration of each tuner is refit
once on **all 160k rows of `train`**, with a fixed number of trees:
`n_estimators = round(mean of the winning trial's 3 fold best_iterations)`, no
hold-out and no early stopping. It is not scaled up for the 4× larger training
set: the tuning logs cannot give the right factor (all their folds have the
same size, and we did not run an extra CV on larger folds), a linear ×4 would ask for
more trees than any fold could explore, and near the optimum the AUC curve is
flat. We *assume*, without having tested it, that more data would favour more
trees, so the rule may give slightly too few; it is the same for every tuner.
The first refit (version 1) used early stopping on a 10% hold-out instead; that
stop was noisy (compare the tree counts in
`reports/tables/tuning_refit_versions.csv` with the CV best_iterations) and fit
on only 90% of `train` (column `train_rows_fit`), although its log records say
all of `train`. Its records stay in the log, marked as version 1,
and its model files are kept in `models/refit_v1/`. The log is append-only, so
a version 1 record's `model_path` still names `models/lgbm_<tuner>.txt`, which now
holds the version 2 model; the `model_file` column of `tuning_refit_versions.csv`
gives the real location of each version's model. Version 2 supersedes it:
`reports/tables/tuning_refit_versions.csv` compares the two, every other table
uses version 2, and the table build refuses to mix versions across methods.

**Confidence intervals.** Test-set AUC and TPR@1%FPR get 95% paired bootstrap
intervals (1000 resamples; seed 42 for `test_random` and 43 for `test_future`,
`tables.BOOTSTRAP_SEEDS`, so the two sets are resampled independently, which
the random-vs-future gap CI needs) computed from the saved predictions in
`reports/predictions/`. "Paired" means every model is scored on the same
resampled rows, so `tuning_pairwise.csv` can say whether one tuner really beats
another, or whether the difference is test-set luck. The intervals cover
test-set sampling noise only, not training randomness (one seed, one fold
split). The tuners' CV AUC is slightly optimistic (early stopping picks its round
on the fold that is scored, and the value is the best of 36 trials), so compare
CV AUC only among the tuners; the test scores compare tuners with baselines.

**All tools on CPU (16 cores on the Steps 2–3 laptop, 4 on the cloud VM).** Not
every tool can use the GPU in the same way, so a mix of GPU and CPU runs would
make timings meaningless. Every library gets `n_jobs=config.N_JOBS` (16 for
Steps 2–3, 4 for Step 4 onwards), and every pair of runs that is compared ran on
the same machine.

**20 minutes per AutoML tool.** `AUTOML_BUDGET_S=1200` gives each tool
(FLAML, AutoGluon `medium_quality`, H2O) time to try several model families on
160k × 2,568 features. It also keeps the total runtime manageable, which
matters because Step 4 reruns the tools for the leakage experiment. Every
tool gets the same budget and data, and seed 42 wherever the tool accepts
one. Some random choices inside the tools ignore it: FLAML's holdout split
and its estimators keep FLAML's own fixed seeds, and AutoGluon's base models
use their default seed 0 (only the train/validation split takes seed 42). The
seeds actually used are read back from the saved models and listed in
`reports/automl_comparison.md`.

**Leakage experiments at 600 s per AutoML run (Step 4A).** Re-running FLAML, AutoGluon
and H2O for every leak variant at the Step 3 budget of 1200 s would take about 4 hours, so
each leakage run gets `LEAKAGE_BUDGET_S=600` (a stated deviation). The clean variant is re-run
at the same 600 s rather than reusing the 1200 s Step 3 runs, so each with/without-leak
comparison changes one thing only: the leak column. The tuned LightGBM uses the Step 2
winner's configuration and fixed tree count for every variant (no re-tuning). The leak columns
are appended to `X` only inside `src/leakage.py`; every other step keeps them out.

**Our own cache builder instead of `thrember.create_vectorized_features`.**
The thrember function does not deduplicate (see quirk 1) and writes no
metadata. It also fails when no challenge file sits in the same folder, and it
reopens the output file once per row. Our builder streams each jsonl file,
drops repeated sha256 values, vectorizes with the same
`PEFeatureExtractor.process_raw_features`, and writes features and metadata
row-aligned.

## Team and admin

* Team size: TODO
* Deadline: TODO
* Topic approval: TODO
* Part A certificate: TODO (outside this repo)

## Status

| Step | What | Status |
|---|---|---|
| 0 | Environment, download, data verification | done: download sha256-verified; checks in [`reports/data_check.json`](reports/data_check.json) |
| 1 | Cache, splits, CV folds | done: [`data/splits/splits_info.json`](data/splits/splits_info.json) (committed, with `split_sha256.parquet` for checking a rebuild), [`reports/tables/class_balance_by_week.csv`](reports/tables/class_balance_by_week.csv); `pytest` passes with no skips |
| 2 | Baselines (LightGBM default, logistic regression) and Grid / Random / Optuna tuning | done, passed adversarial review (refit version 2 for all five methods; run log [`reports/run_tuning_refit_v2.log`](reports/run_tuning_refit_v2.log); its final table step failed because the running process mixed the old `tables.py` with the edited `tuning_writeup.py`, so the outputs were rebuilt with `--tables-only`, see [`reports/run_tuning_tables_v2.log`](reports/run_tuning_tables_v2.log)): [`reports/tuning_writeup.md`](reports/tuning_writeup.md), [`reports/tables/tuning_summary.csv`](reports/tables/tuning_summary.csv), [`reports/tables/tuning_pairwise.csv`](reports/tables/tuning_pairwise.csv), [`reports/tables/tuning_trials.csv`](reports/tables/tuning_trials.csv), [`reports/tables/tuning_refit_versions.csv`](reports/tables/tuning_refit_versions.csv), [`reports/figures/tuning_compare.png`](reports/figures/tuning_compare.png), [`reports/figures/tuning_convergence.png`](reports/figures/tuning_convergence.png), raw log [`reports/experiments.jsonl`](reports/experiments.jsonl) |
| 3 | AutoML: FLAML, AutoGluon, H2O (1200 s each) | done, passed adversarial review (run log [`reports/run_automl.log`](reports/run_automl.log); a first attempt where the laptop slept is in [`reports/run_automl_attempt1_slept.log`](reports/run_automl_attempt1_slept.log); notes re-derived from the saved models with `--refresh-notes`, no refit): [`reports/automl_comparison.md`](reports/automl_comparison.md), [`reports/tables/automl_vs_manual.csv`](reports/tables/automl_vs_manual.csv), [`reports/tables/automl_pairwise.csv`](reports/tables/automl_pairwise.csv), [`reports/figures/automl_vs_manual.png`](reports/figures/automl_vs_manual.png), [`reports/leaderboards/`](reports/leaderboards/) |
| 4 | Critical evaluation: random vs future, leakage, bias slices, SHAP, calibration, drift | to redo from scratch; draft leakage code (`src/leakage*.py`, `scripts/run_leakage.py`) exists but is unreviewed and its outputs were discarded |
| 5 | Deployment: FastAPI, Streamlit, model card | pending |
| 6 | Report and slides | pending |
