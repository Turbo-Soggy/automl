# Reproducibility check (D1)

Run on 2026-10-29, 22:45-23:36 IST, on the student's laptop: Windows 11, Intel Core Ultra 9
285H (16 cores), 63 GB RAM, Python 3.12.10, branch `step4-6-cloud`. The full data (extracted
Win64 jsonl and the feature cache) was present. The logs of the runs below are in
[`repro_logs/`](repro_logs/): `fresh_venv_install.log` (section 1), `repro_pytest_full.log` and
`repro_pytest_fresh_venv.log` (section 2) and `repro_quick_run.log` (section 3). The commands are
given so anyone can repeat them.

D1 asks for four things. Summary:

| Check | Result |
|---|---|
| A fresh venv built from `requirements.txt` installs on Python 3.12 / Windows | **Pass** (on a short path; see the path-length finding) |
| `pytest` passes with no skips when the data is present | **Pass**: 160 passed, 0 skipped (project venv and fresh venv) |
| `run_all.py --quick` runs end to end from deleted quick outputs | **Pass** after one fix: the quick deploy step overwrote the real `MODEL_CARD.md` (fixed, see below) |
| Seed 42 everywhere | **Pass**: regenerated quick splits are byte-identical, and the quick Step 2 scores repeat exactly |

## 1. Fresh virtual environment

```
py -3.12 -m venv <folder>
<folder>/Scripts/python.exe -m pip install --upgrade pip
<folder>/Scripts/python.exe -m pip install -r requirements.txt
<folder>/Scripts/python.exe -m pip check
```

* **Attempt 1, venv in the session's scratch folder (a 107-character path): failed.**
  pip resolved every pin and built both git dependencies (`thrember`, `oscrypto`), then
  stopped while unpacking torch with `OSError: [WinError 206] The filename or extension is
  too long`. This laptop has `LongPathsEnabled = 0`, so paths are limited to 260
  characters, and some installed files sit about 140 characters below `site-packages`
  (inside torch's license folder and streamlit's templates). This is a property of the
  folder, not of `requirements.txt`, so no pin was changed. `README.md` (Environment) now
  tells users to keep the venv on a short path, as the project's own `C:\projects\automl\.venv`
  is.
* **Attempt 2, venv at `C:\Users\benja\AppData\Local\Temp\claude\fv`: passed.**
  * Creating the venv and upgrading pip took 30 s; `pip install -r requirements.txt`
    took 576 s (606 s in total).
  * Every wheel came from the local pip cache (0 downloads), so a first install on a new
    machine will take longer (torch alone is a 124 MB wheel).
  * `pip check`: "No broken requirements found."
  * Importing `lightgbm, flaml, autogluon.tabular, h2o, shap, fastapi, streamlit, thrember,
    reportlab, pptx, playwright, torch` works (LightGBM 4.7.0, FLAML 2.7.0, H2O 3.46.0.12,
    torch 2.14.1+cpu).
  * The full test suite run with this fresh venv on the project: **160 passed, 0 skipped,
    1 warning, 56.6 s.**
* **Only the direct dependencies are pinned.** Comparing `pip freeze` of the fresh venv with
  the project venv, 11 transitive packages resolved to newer patch releases (MarkupSafe,
  boto3, botocore, filelock, fsspec, hf-xet, networkx, platformdirs, pytz, setuptools,
  tzdata; `oscrypto` only differs in how pip names the git install). Every pinned package
  matched. The tests pass with the newer versions, so this is recorded and not changed.
* **Linux.** The install on Linux was verified earlier on the 4-core cloud VM where Step 4
  ran (`docs/handoff/HANDOFF.md`, section 0b): it needed the `oscrypto` git pin, because the
  PyPI release cannot parse OpenSSL 3.0.x version strings there, and `pytest` passed on the
  VM (118 tests at that time; the suite has grown since). It was not re-run for this check.

## 2. `pytest` with the data present

```
.venv/Scripts/python.exe -m pytest -q -rs
```

* Project venv: **160 passed, 0 skipped, 1 warning, 97.5 s** (it ran while the quick run's
  tuning step was using the CPU).
* Fresh venv (section 1): **160 passed, 0 skipped, 1 warning, 56.6 s.**
* The one warning is SHAP's notice that its TreeExplainer output format for LightGBM binary
  models changed (`tests/test_explain.py`); the test checks the values, so it is harmless.
* `-rs` lists no skip. The skip guards in `tests/conftest.py` and `tests/test_api.py` only
  fire without the cache, without `models/final/`, or off Windows (the `notepad.exe` upload
  tests).

## 3. `run_all.py --quick` from deleted quick outputs

All 218 `*_quick` files and folders (splits, logs, tables, figures, predictions, models,
leaderboards, write-ups) were moved out of the repository first, so nothing from an earlier
quick run could be reused. Then:

```
.venv/Scripts/python.exe run_all.py --quick
```

**Result: exit code 0, "done in 2620 s" (43.7 min), all five steps.**

| Step | Script | Time (s) |
|---|---|---|
| data | `build_cache.py` (cache complete, only the quick splits are made) | 5 |
| data | `verify_data.py` | 24 |
| tune | `run_tuning.py` (baselines, Grid, Random, Optuna, 4 trials each) | 806 |
| automl | `run_automl.py` (FLAML, AutoGluon, H2O at 60 s each, then tables) | 379 |
| eval | `run_explain.py` | 99 |
| eval | `run_eval_bias.py` | 338 |
| eval | `run_leakage.py --skip-done` (13 runs, 30 s AutoML budget each, then tables) | 930 |
| eval | `run_critical_eval.py` | 19 |
| deploy | `export_model.py`, then `pytest tests/test_api.py` (41 passed) | 7 + 12 |

The machine was not reserved for this run: the fresh-venv install (section 1) and one
`pytest` run overlapped with its tune step, and two idle API/Streamlit demo servers were
running. Quick-run times are therefore only a rough guide; quick runs are smoke tests and
their scores are not used anywhere.

### What the quick run found, and the fix

* **The quick deploy step overwrote the real `MODEL_CARD.md`.** `scripts/export_model.py`
  chose the card's location from its own `--quick` flag. `run_all.py --quick` does not pass
  that flag; it only sets `EMBER_QUICK=1`. So the model and schema went to `models/final_quick/`
  as intended, but the card was written over the root `MODEL_CARD.md` with quick-model numbers.
  That breaks the rule that quick runs never overwrite real results.
  * **Fix:** `export()` now reads the mode from `config.QUICK` (which follows `EMBER_QUICK`),
    so the quick card goes to `models/final_quick/MODEL_CARD.md` however quick mode is set.
  * The real `MODEL_CARD.md` was restored from git (the version committed in `7d821d6`).
  * Re-ran `run_all.py --quick --steps deploy`: exit 0 in 21 s, `test_api.py` 41 passed,
    the card was written to `models/final_quick/MODEL_CARD.md`, and the md5 of the root
    `MODEL_CARD.md` was unchanged.
  * The full `pytest` run with the fresh venv (section 1) ran after this fix.
* **`verify_data.py` writes `reports/tables/class_balance_by_week.csv` without a `_quick`
  suffix in quick mode too.** The table counts labels per week over the whole feature
  cache, not the quick splits, so it is the same in both modes: the rewritten file is
  byte-identical and `git status` shows no change. Left as is.

### Outputs that differ from the previous quick run

* **Byte-identical:** all 7 quick split files in `data/splits/` (row indices, CV folds,
  `splits_info_quick.json`).
* **Identical scores:** every Step 2 quick row (`tuning_summary_quick.csv`: CV AUC and
  test_future AUC of the two baselines and the three tuners match to every printed digit).
  The files themselves differ only in run times and run ids.
* **Close but not identical:** the AutoML quick scores (for example the FLAML test_future
  AUC moved in the third decimal). Time-budgeted searches depend on how fast the machine
  is at that moment, so this is expected; `reports/limitations.md` says the same for the
  full runs.
* **Deleted with the old quick outputs and not regenerated, by design** (so they are no
  longer in the repository):
  * `reports/predictions/eval/cv_stratified_fold{0,1,2}_quick.npz`. `run_eval_bias.py`
    reuses the stratified out-of-fold scores that `run_explain.py` already wrote
    (`src/temporal.tuned_stratified_folds`) and writes its own only when they are missing.
  * `reports/run_automl_quick.log` and `data/cache/verify_data_quick.log`. They were shell
    redirects from an earlier session, not script outputs.
* **New:** `models/leakage/*/autogluon_quick/` holds only AutoGluon's log file. The
  predictors are deleted after scoring, as in the full Step 4 runs. The H2O quick model
  folders have new timestamped names.

### Code changed after this run

The end-to-end quick run above predates some small code edits made in the two final review
rounds, so the pipeline exactly as committed has not been run end to end. The edits:

* `src/automl_runs.py`: `run_h2o` now deletes the previous H2O model folder before a fit
  (4 lines; H2O names leader folders by timestamp, so old leaders piled up). This path has
  not run end to end since the change.
* `run_all.py`, `src/__init__.py`, `src/tune.py`: dead code removed (no remaining references
  to the removed names).
* Write-up text, docstrings and unused names only: `src/tuning_writeup.py`, `src/critical_eval_writeup.py`,
  `src/eval_bias_report.py`, `src/leakage_writeup.py`, `src/temporal_writeup.py`,
  `app/model_card.py`, `scripts/build_report.py`, `scripts/export_model.py`; a comment in `config.py`.

After these edits `pytest` passed (160 passed, 0 skipped, 1 warning, on 2026-10-30), the
write-ups were rebuilt from the saved logs with `run_leakage.py --tables-only`,
`run_eval_bias.py --stages writeup` and `run_critical_eval.py --stages writeup` (real and
`--quick`), and `reports/report.pdf` and `reports/slides.pptx` were rebuilt. The full quick run
was not repeated because of the submission deadline.

## 4. Report and slides rebuilt last

After the checks above, the report and the deck were rebuilt from the final tables:

```
.venv/Scripts/python.exe scripts/build_report.py   # reports/report.pdf: 30 pages, 18 figures, 29 tables
.venv/Scripts/python.exe scripts/build_slides.py   # reports/slides.pptx: 10 slides, ~5.3 min of speaker notes
```

Both builds exited with code 0. The report reported no missing figure or screenshot.

On 2026-10-29 all 30 report pages were rendered to images with `pdftoppm` (MiKTeX) and inspected
by eye, as D7 asks: every figure, table and screenshot is in place, with no overflow and no blank
page.
