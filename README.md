# Malware detection on EMBER2024: manual tuning vs AutoML

Course project (IA-3, Part B). We train binary classifiers that label Windows
64-bit PE files as **benign (0)** or **malicious (1)** from static features.
Then we compare hand-tuned LightGBM (grid, random and Optuna search) with three
AutoML tools (FLAML, AutoGluon, H2O).

## Start here (new to the project?)

**What this is.** We built malware detectors for Windows programs that judge a file
from its structure, without running it. We built them two ways: by tuning one model
by hand, and by letting three AutoML tools pick and tune models on their own. Then we
checked how fair the usual testing is, by also testing on malware from *later* weeks
than the training data, and by planting a "cheat" column to see who notices it.

**See it in about 10 minutes** (no dataset download needed). You need Python 3.12 and
`git`. The slim `app/requirements.txt` installs only what the presentation needs; the
40 GB dataset and the AutoML tools are only needed to re-run the experiments. On Windows,
run the commands in PowerShell or Git Bash (not cmd.exe), and clone into a short path
such as `C:\automl`: Streamlit ships deeply nested files that can exceed Windows'
260-character path limit.

```bash
git clone https://github.com/Turbo-Soggy/automl.git
cd automl
py -3.12 -m venv .venv                                    # Linux / macOS: python3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r app/requirements.txt   # Linux / macOS: .venv/bin/python -m pip ...
.venv/Scripts/python.exe present.py                       # Linux / macOS: .venv/bin/python present.py
```

`present.py` starts the prediction API and the presentation page and opens
http://127.0.0.1:8501 in your browser (Ctrl+C stops both). The page is the whole
talk, top to bottom: the question, the data, the contenders, how we compared them,
the results, the full scorecard of evaluation metrics, the winner and why, a live demo,
and what we learned. [`reports/talk_script.md`](reports/talk_script.md) is the script
for presenting it (what to say, where to scroll and what to click), and `DEMO.md`
covers setup and backups.

**Then read, in this order:**

1. The presentation page (above). Every number on it is read from the saved results
   (`reports/tables/`, `models/final/`, `data/splits/splits_info.json`, `config.py`).
2. [`reports/report.pdf`](reports/report.pdf): the full write-up of every step.
3. [`reports/critical_eval.md`](reports/critical_eval.md): the answer to the research question.
4. [`reports/qa_prep.md`](reports/qa_prep.md): likely questions, with answers.
5. The code, by step (details in [Repository layout](#repository-layout)):

| Step | What | Code | Results |
|---|---|---|---|
| 1 | Data, splits | `src/data.py`, `scripts/build_cache.py` | `reports/data_check.json`, `data/splits/` |
| 2 | Hand-tuned LightGBM | `src/tune.py`, `scripts/run_tuning.py` | `reports/tuning_writeup.md` |
| 3 | AutoML | `src/automl_runs.py`, `scripts/run_automl.py` | `reports/automl_comparison.md` |
| 4 | Critical evaluation | `src/leakage.py`, `src/temporal.py`, `src/explain.py`, ... | `reports/critical_eval.md` |
| 5 | Deployment | `app/api.py`, `app/streamlit_app.py`, `app/story.py` | `MODEL_CARD.md`, `DEMO.md` |
| 6 | Report, slides | `scripts/build_report.py`, `scripts/build_slides.py` | `reports/report.pdf`, `reports/slides.pptx` |

**Words you will meet**

| Term | Meaning |
|---|---|
| test_random / test_future | The two exams: files from the same weeks as training, and files from 12 later weeks |
| TPR at 1% FPR | Share of malware caught when only 1 in 100 safe files may be flagged (our headline score) |
| AUC | How well the scores rank malware above safe files: 0.5 is a coin flip, 1.0 is perfect |
| Threshold | The score above which a file is flagged; ours is set low because a missed malware costs more |
| SHAP | A way to split one prediction among the inputs, to see what pushed the decision |
| AutoML | Tools that choose the model type and its settings by themselves |
| Leakage | A column that gives away the answer during training but would not exist in real use |
| Drift | Malware changing over time, so older training data fits newer files less well |
| 95% CI | The range the true difference probably lies in; if it includes 0, two models are tied |

**Re-running the experiments** needs the full dataset (about 40 GB on disk), the full
`requirements.txt` and several hours: see [Reproducing](#reproducing).

> **Status (2026-10-29):** all six steps are done (see [Status](#status) and
> [Deliverables](#deliverables)). Steps 2-3 and 5-6 ran on the student's laptop,
> Step 4 on a 4-core cloud VM. The reproducibility check (fresh venv, `pytest`,
> `run_all.py --quick` from deleted quick outputs) is in
> [`reports/repro_check.md`](reports/repro_check.md), and every rubric item is mapped
> to its evidence in [`reports/SUBMISSION_CHECKLIST.md`](reports/SUBMISSION_CHECKLIST.md).
> No results are typed into this README. All numbers live in `reports/` and are
> generated from the experiment log and the saved test-set predictions.

## Deliverables

| Deliverable | File |
|---|---|
| Report (PDF) | [`reports/report.pdf`](reports/report.pdf), built by `scripts/build_report.py` |
| Skill-based evaluation report (the college's Word template; certificates page left for the student) | [`reports/skill_report.docx`](reports/skill_report.docx) and [`reports/skill_report.pdf`](reports/skill_report.pdf), built by `scripts/build_skill_report.py` (the PDF is printed by Microsoft Word, Windows only) |
| Slides (5-minute talk, with speaker notes) and Q&A preparation | [`reports/slides.pptx`](reports/slides.pptx) (`scripts/build_slides.py`), [`reports/qa_prep.md`](reports/qa_prep.md) |
| Model card | [`MODEL_CARD.md`](MODEL_CARD.md) |
| Presentation script (what to say and where to click on the page), demo notes and screenshots | [`reports/talk_script.md`](reports/talk_script.md) (`scripts/build_talk_script.py`), [`DEMO.md`](DEMO.md), [`reports/screenshots/`](reports/screenshots/) |
| Deployed model and app | [`models/final/`](models/final/) (model, `schema.json`, threshold), [`app/api.py`](app/api.py) (FastAPI), [`app/streamlit_app.py`](app/streamlit_app.py) (the presentation page; start it with `present.py`) |
| Answer to the research question | [`reports/critical_eval.md`](reports/critical_eval.md), [`reports/limitations.md`](reports/limitations.md) |
| Step write-ups | [`reports/tuning_writeup.md`](reports/tuning_writeup.md) (Step 2), [`reports/automl_comparison.md`](reports/automl_comparison.md) (Step 3), [`reports/leakage.md`](reports/leakage.md), [`reports/temporal_bias.md`](reports/temporal_bias.md), [`reports/explainability.md`](reports/explainability.md) (Step 4) |
| Executed notebooks | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb), [`02_baseline_tuning.ipynb`](notebooks/02_baseline_tuning.ipynb), [`03_automl.ipynb`](notebooks/03_automl.ipynb), [`04_critical_eval.ipynb`](notebooks/04_critical_eval.ipynb) |
| Experiment log, tables, figures | [`reports/experiments.jsonl`](reports/experiments.jsonl), [`reports/tables/`](reports/tables/), [`reports/figures/`](reports/figures/), [`reports/leaderboards/`](reports/leaderboards/) |
| Reproducibility check and rubric checklist | [`reports/repro_check.md`](reports/repro_check.md), [`reports/SUBMISSION_CHECKLIST.md`](reports/SUBMISSION_CHECKLIST.md) |
| Handoff notes (state, how to continue on another machine) | [`docs/handoff/HANDOFF.md`](docs/handoff/HANDOFF.md) |

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

| | Steps 0–3 and 5–6 (data, tuning, AutoML, deployment, report) | Step 4 (cloud VM; also the cloud data rebuild) |
|---|---|---|
| CPU | Intel Core Ultra 9 285H, 16 cores | Intel Xeon @ 2.1 GHz, 4 vCPU |
| RAM / disk | 63 GB, about 640 GB free | 15.7 GiB (no swap), about 13 GB free |
| OS | Windows 11 | Linux |
| `N_JOBS` / H2O memory | 16 / 24G | 4 / 7G for the leakage runs (`H2O_STEP4_MAX_MEM` in `src/leakage.py`, XGBoost excluded) |

The Step 4 challenge-set scoring alone was re-run on the laptop: it fits nothing, it only
scores the saved models, and the laptop holds every saved Step 2-3 model.

No GPU was used on either (see "all-CPU" below). `config.N_JOBS` is
`min(16, cpu_count)` and the H2O memory is capped at about 60% of RAM, both
computed automatically. The Step 3 AutoML, Step 4 leakage, eval-fit and explain records
in `reports/experiments.jsonl` log their own `n_jobs`; the Step 2 tuning and threshold
records do not (Step 2 ran on the laptop with `num_threads = config.N_JOBS = 16`, set in
`src/tune.py` `BASE_PARAMS`). The data rebuilt on the cloud VM matches the laptop exactly (split
sha256 order and feature hashes in `data/splits/split_fingerprints.json`).

Software: Python 3.12 (AutoGluon and H2O do not support 3.14) and Java 21
(needed by H2O).

```
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Windows path length.** Keep the venv on a short path (for example the repository
at `C:\projects\automl`, as here). Windows limits paths to 260 characters unless
long paths are enabled, and some installed files (inside the torch and streamlit
packages) are about 140 characters deep under `site-packages`. A venv created
under a deep folder fails with `OSError: [WinError 206] The filename or extension
is too long` (see `reports/repro_check.md`).

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
.venv/Scripts/python.exe scripts/run_explain.py               # Step 4C: OOF scores, SHAP, agreement, calibration, threshold
.venv/Scripts/python.exe scripts/run_eval_bias.py             # Step 4B: random vs future, CV, bias, drift, challenge set
.venv/Scripts/python.exe scripts/run_critical_eval.py         # Step 4 synthesis: critical_eval.md, limitations.md
.venv/Scripts/python.exe run_all.py --steps eval              # all of Step 4 in that order (leakage with --skip-done)
.venv/Scripts/python.exe scripts/run_automl.py --refresh-notes  # re-derive notes from saved models (no refit)
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/02_baseline_tuning.ipynb
```

* `--steps` takes a comma-separated subset of `data,tune,automl,eval,deploy`.
  Steps always run in that order. `deploy` exports the model to `models/final/`
  (with `schema.json`, the threshold and `MODEL_CARD.md`) and then runs
  `tests/test_api.py` against it.
* The report and the deck are rebuilt from the saved tables in a few seconds:
  `.venv/Scripts/python.exe scripts/build_report.py` and
  `.venv/Scripts/python.exe scripts/build_slides.py`. Starting the API and the
  Streamlit page is described in [`DEMO.md`](DEMO.md).
* **Linux or a cloud VM.** Use `.venv/bin/python`. With `--keep-zipped` the Win64
  zips stay packed and `src.data` streams the weekly jsonl out of them (see
  Download); `config.N_JOBS` and the H2O memory scale down to the machine
  automatically. The Linux install was verified on the 4-core cloud VM (see
  `reports/repro_check.md`); `requirements.txt` pins `oscrypto` to a git commit
  because the PyPI release cannot parse OpenSSL 3.0.x version strings there.
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
| `eval` | `scripts/run_explain.py`, `scripts/run_eval_bias.py`, `scripts/run_leakage.py --skip-done`, `scripts/run_critical_eval.py` (in that order: explain writes the train out-of-fold scores and `models/threshold.json` that the others reuse) | `reports/critical_eval.md`, `reports/limitations.md`, `reports/leakage.md`, `reports/temporal_bias.md`, `reports/explainability.md`; tables in `reports/tables/` (`random_vs_future`, `precision_both_sets`, `gap_vs_tuned`, `automl_self_estimate`, `leakage`, `cv_comparison`, `bias_slices`, `prevalence`, `challenge`, `drift`, `shap_*`, `calibration`, `threshold`, ...), figures in `reports/figures/`, `reports/leakage_logs/`, `reports/leaderboards/leakage/`, `reports/predictions/{eval,leakage}/`, `models/threshold.json`, `models/eval/`, `models/leakage/` |
| `deploy` | `scripts/export_model.py`, then `pytest tests/test_api.py` | `models/final/` (`model.txt`, `schema.json`, `demo_samples.npz`), `MODEL_CARD.md` |

## Repository layout

```
config.py              seed, paths and experiment sizes (single source of truth)
run_all.py             runs the pipeline steps in order
present.py             starts the API and the presentation page and opens the browser
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
  temporal.py          Step 4B: random-vs-future gaps, stratified and forward CV, the extra train-only fits
  bias.py              Step 4B: class balance, per-slice TPR/FPR, class weights, precision at a base rate
  drift.py             Step 4B: weekly scores of expanding, sliding and size-matched training windows
  challenge.py         Step 4B: vectorises the Win64 challenge-set rows into separate cache files
  eval_bias_report.py  Step 4B tables and figures; temporal_writeup.py renders reports/temporal_bias.md
  explain.py           Step 4C: TreeSHAP, feature-family grouping, cross-tool agreement
  calibration.py       Step 4C: out-of-fold scores, reliability, Brier/ECE, cost-sensitive threshold
  critical_eval.py     Step 4 synthesis tables (precision on both sets, gap vs tuned, AutoML self-estimates)
  critical_eval_writeup.py  renders reports/critical_eval.md and reports/limitations.md from the tables
app/
  api.py               Step 5 FastAPI service: /health, /predict (validated vector), /predict_file (in memory)
  streamlit_app.py     Step 5 presentation page: the whole talk in ten sections, with the live demo (calls the API)
  story.py             the page's presentation sections, every number read from reports/tables/
  model_card.py        renders MODEL_CARD.md from models/final/schema.json and the tables
  take_screenshots.py  saves the demo screenshots to reports/screenshots/
scripts/
  download_data.py     verified download of the raw jsonl
  probe_raw.py         looks at the first rows of one remote file (setup only)
  build_cache.py       raw jsonl -> deduplicated feature matrix + metadata
  verify_data.py       data checks -> reports/data_check.json
  run_tuning.py        Step 2 entry point
  run_automl.py        Step 3 entry point (--tools, --tables-only, --refresh-notes, --quick)
  run_leakage.py       Step 4A entry point (--runs, --skip-done, --tables-only, --quick)
  run_eval_bias.py     Step 4B entry point (--stages fits,challenge,tables,writeup, --quick)
  run_explain.py       Step 4C entry point (--stages oof,shap,agreement,calibration,threshold,report, --quick)
  run_critical_eval.py Step 4 synthesis entry point (--stages tables,writeup, --quick)
  export_model.py      Step 5: copies the tuned LightGBM to models/final/ with schema.json and the model card
  build_report.py      Step 6: reports/report.pdf (and figures/architecture.png) from the saved outputs
  build_slides.py      Step 6: reports/slides.pptx with speaker notes
  build_skill_report.py  Step 6: reports/skill_report.docx/.pdf in the college's report template
  report_diagrams.py   the UML class and sequence diagrams for that report
  build_talk_script.py reports/talk_script.md: the presenter's script for the page, numbers from the tables
tests/
  conftest.py          puts the project root on sys.path; skip helpers
  test_splits.py       disjointness, temporal order, sizes, balance, CV folds, feature groups
  test_tune.py         tuners, seeding, refit rule, bootstrap, tables (synthetic data, no cache needed)
  test_automl.py       Step 3 tables, pairwise rows, FLAML log parser, sleep guard (synthetic data)
  test_leakage.py      Step 4A variants, time-proxy positions in thrember's source, ranks, SHAP, log scan
  test_eval_bias.py    Step 4B folds, metrics, thresholds, base-rate maths, challenge filtering (synthetic)
  test_explain.py      Step 4C SHAP checks, family grouping, calibration and threshold maths (synthetic)
  test_critical_eval.py  Step 4 synthesis tables and the sentences chosen from them
  test_raw_files.py    reading the weekly jsonl from extracted files or straight from the zips
  test_api.py          Step 5 API validation, upload safety, API vs offline scores, Streamlit page
notebooks/
  01_eda.ipynb                 narrative EDA (imports from src, no own logic)
  02_baseline_tuning.ipynb     narrative view of the Step 2 tables and figures
  03_automl.ipynb              narrative view of the Step 3 leaderboards and comparison
  04_critical_eval.ipynb       narrative view of the Step 4 tables and figures
reports/               generated tables, figures, leaderboards, experiment log, predictions/, write-ups
                       (method id `optuna` in the Step 2 tables and `lgbm_tuned` in the Step 3-4
                       tables are the same model: the Optuna-tuned LightGBM that is deployed)
data/                  raw/ and cache/ gitignored (regenerated); splits/ committed
models/                committed boosters and H2O leaders (AutoGluon and quick H2O folders gitignored)
models/final/          the deployed model, schema.json and threshold (Step 5)
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
make timings meaningless. Every library gets `n_jobs=config.N_JOBS` (16 on
the laptop for Steps 2–3 and 5–6; 4 on the cloud VM for the Step 4 leakage runs,
and 2 for the Step 4 bias and explain fits, which ran as two parallel jobs), and
every pair of runs that is compared ran on the same machine.

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
| 2 | Baselines (LightGBM default, logistic regression) and Grid / Random / Optuna tuning | done, passed adversarial review (refit version 2 for all five methods; run log [`reports/run_tuning_refit_v2.log`](reports/run_tuning_refit_v2.log), after an aborted first attempt that only got through the baselines, [`reports/run_tuning_refit_v2_attempt1.log`](reports/run_tuning_refit_v2_attempt1.log); its final table step failed because the running process mixed the old `tables.py` with the edited `tuning_writeup.py`, so the outputs were rebuilt with `--tables-only`, see [`reports/run_tuning_tables_v2.log`](reports/run_tuning_tables_v2.log)): [`reports/tuning_writeup.md`](reports/tuning_writeup.md), [`reports/tables/tuning_summary.csv`](reports/tables/tuning_summary.csv), [`reports/tables/tuning_pairwise.csv`](reports/tables/tuning_pairwise.csv), [`reports/tables/tuning_trials.csv`](reports/tables/tuning_trials.csv), [`reports/tables/tuning_refit_versions.csv`](reports/tables/tuning_refit_versions.csv), [`reports/figures/tuning_compare.png`](reports/figures/tuning_compare.png), [`reports/figures/tuning_convergence.png`](reports/figures/tuning_convergence.png), raw log [`reports/experiments.jsonl`](reports/experiments.jsonl) |
| 3 | AutoML: FLAML, AutoGluon, H2O (1200 s each) | done, passed adversarial review (run log [`reports/run_automl.log`](reports/run_automl.log); a first attempt where the laptop slept is in [`reports/run_automl_attempt1_slept.log`](reports/run_automl_attempt1_slept.log); notes re-derived from the saved models with `--refresh-notes`, no refit): [`reports/automl_comparison.md`](reports/automl_comparison.md), [`reports/tables/automl_vs_manual.csv`](reports/tables/automl_vs_manual.csv), [`reports/tables/automl_pairwise.csv`](reports/tables/automl_pairwise.csv), [`reports/figures/automl_vs_manual.png`](reports/figures/automl_vs_manual.png), [`reports/leaderboards/`](reports/leaderboards/) |
| 4 | Critical evaluation: random vs future, leakage, bias slices, SHAP, calibration, drift | done on the 4-core cloud VM (600 s leakage budget, a stated deviation), reviewed: [`reports/critical_eval.md`](reports/critical_eval.md) (the answer to the research question), [`reports/limitations.md`](reports/limitations.md), [`reports/leakage.md`](reports/leakage.md), [`reports/temporal_bias.md`](reports/temporal_bias.md), [`reports/explainability.md`](reports/explainability.md), tables in [`reports/tables/`](reports/tables/), figures in [`reports/figures/`](reports/figures/), [`models/threshold.json`](models/threshold.json) |
| 5 | Deployment: FastAPI, Streamlit, model card | done on the laptop, reviewed: [`models/final/`](models/final/), [`app/api.py`](app/api.py), [`app/streamlit_app.py`](app/streamlit_app.py), [`tests/test_api.py`](tests/test_api.py), [`reports/screenshots/`](reports/screenshots/), [`MODEL_CARD.md`](MODEL_CARD.md), [`DEMO.md`](DEMO.md). Cloud deployment is left to the student (needs their own account) |
| 6 | Report, slides, notebooks | done on the laptop: [`reports/report.pdf`](reports/report.pdf), [`reports/slides.pptx`](reports/slides.pptx), [`reports/qa_prep.md`](reports/qa_prep.md), [`notebooks/`](notebooks/) (all four executed, with outputs) |
| Final | Reproducibility check and submission checklist | [`reports/repro_check.md`](reports/repro_check.md), [`reports/SUBMISSION_CHECKLIST.md`](reports/SUBMISSION_CHECKLIST.md); final adversarial gate passed on its third round (no blocker or major issue; the bar was relaxed to one clean round for the deadline, see the checklist D8 row) |

Left for the student: the Part A certificate, the team / deadline / topic-approval
`TODO`s above, recording the demo video, the cloud deployment, and pushing the
branch to GitHub.
