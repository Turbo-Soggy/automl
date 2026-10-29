# Handoff: where the work stopped and how to continue

Written 2026-10-29, when the work moved from the student's Windows laptop to a Claude cloud
session. Read `CLAUDE.md` first for the working rules.

## 0. Current state: final preparation on the laptop (2026-10-29, evening IST)
All six steps are done on branch `step4-6-cloud`. Steps 5-6 and the final preparation ran on
the laptop; Step 4 numbers and settings are still the cloud-VM ones (section 0b) and were not
re-run. **Next: the final gate (D8): two consecutive clean adversarial rounds, then a
spot-check, then commit.** Then the items only the student can do (below).

### Stage status
| Stage | Status | Evidence |
|---|---|---|
| Steps 0-3 | Done, reviewed | section 1 table, `docs/handoff/review_log.md` |
| Step 4 (cloud VM) | Done, reviewed. The gate's carried major issue (`temporal_bias.md` said AutoGluon "could not be scored ... not on disk") is fixed at the source (`src/temporal_writeup._limitations(facts)`) | `reports/critical_eval.md`, `reports/limitations.md`, `reports/{leakage,temporal_bias,explainability}.md` |
| Step 5 deployment (laptop) | Done, reviewed (commits `9ab3a32`, `7d821d6`) | `models/final/`, `app/`, `tests/test_api.py`, `reports/screenshots/`, `MODEL_CARD.md`, `DEMO.md` |
| Step 6 report, slides, notebooks (laptop) | Done (commit `7d821d6` plus the uncommitted Step 6 fix-round edits); report and deck rebuilt at the end of final prep | `reports/report.pdf`, `reports/slides.pptx`, `reports/qa_prep.md`, `notebooks/*.ipynb` (all executed) |
| Final prep (D1 evidence, checklist, README) | Done | `reports/repro_check.md`, `reports/SUBMISSION_CHECKLIST.md`, `README.md` (status, deliverables, reproduce, Windows path note) |
| Final gate (D8) | **Next** | |

### What final prep found and did
- Fresh venv (Python 3.12.10, Windows): `pip install -r requirements.txt` resolves and
  installs; no pin had to change. Under the session scratchpad the first attempt failed
  with `WinError 206` (path too long: this laptop has `LongPathsEnabled = 0`, and torch and
  streamlit install files about 140 characters deep). It succeeded on a short path. The
  README now says to keep the venv on a short path. Times and details: `reports/repro_check.md`.
- All 218 `*_quick` outputs were moved out of the repo and `run_all.py --quick` was run end to
  end (all five steps, exit 0, about 44 min). It found one bug: the quick deploy step
  overwrote the real `MODEL_CARD.md`, because `scripts/export_model.py` read its own
  `--quick` flag instead of `EMBER_QUICK`. Fixed (it now follows `config.QUICK`), the card
  restored from git and the quick deploy step re-run. The regenerated quick split files are
  byte-identical to the ones before.
- Full `pytest` with the data present: 160 passed, 0 skipped, in both the project venv and
  the fresh venv.
- `reports/report.pdf` and `reports/slides.pptx` were rebuilt last, from the final tables.
- The Linux install was verified earlier on the 4-core cloud VM (section 0b: `oscrypto` git
  pin; 118 tests passed there at that time).

### Left for the student
Part A certificate; the team size, deadline and topic-approval `TODO`s in `README.md`;
recording the demo video (`DEMO.md`); the cloud deployment (own Hugging Face or Streamlit
account); pushing `step4-6-cloud` to GitHub and opening the PR.

## 0b. Previous handoff: cloud VM → laptop (2026-10-29, 14:45 UTC; kept for history)
The work moved to a Claude cloud session (4-core Xeon, 15 GB RAM, Linux) on branch
`step4-6-cloud`, ran Step 0 checks and all Step 4 experiments, and was then stopped at the
user's request so it can continue on the laptop. **Pick up from the Step 4 review gate.**

### Continue on the laptop
```bash
git fetch origin
git checkout step4-6-cloud
.venv/Scripts/python.exe -m pip install -r requirements.txt   # adds the oscrypto git pin (see below)
.venv/Scripts/python.exe -m pytest -q
```
- **The laptop's data is still valid.** The cloud rebuilt the cache from the same zips and it
  matched the laptop exactly (split sha256 order, first-1000-row feature hashes and
  `cache_info.json` counts). Nothing needs re-downloading. `pytest` at this handoff: 118 passed
  on the cloud VM.
- **One new local file must be regenerated:** the challenge-set cache
  (`data/cache/X_challenge_win64.npy`, `meta_challenge_win64.parquet`,
  `challenge_win64_info.json`; gitignored). Build it from `data/raw/challenge` with
  `.venv/Scripts/python.exe scripts/run_eval_bias.py --stages challenge` (uses `src/challenge.py`;
  it also re-scores the saved models on those rows, which should reproduce `tables/challenge.csv`).
- **What changed for Linux, and why it is safe on Windows:**
  - `src/data.RawFile` / `raw_files()` read weekly jsonl straight out of the zips **only when
    no extracted files exist**. The laptop has `data/raw/win64/*.jsonl`, so it uses them as before.
  - `config.N_JOBS = min(16, cpu_count)`: 16 again on the laptop.
  - `src/automl_runs._h2o_max_mem()`: 24G on the laptop (it caps at 60% of RAM).
  - `requirements.txt` pins `oscrypto` to an upstream git commit (PyPI 1.3.0 cannot parse
    OpenSSL 3.0.13 on Linux). Pure Python, so it installs on Windows too; reinstall once.
- **Step 4 settings are tied to the cloud VM, keep them as they are.** Every leakage run
  (clean and leaky) ran on the 4-core VM with the same settings: 600 s budget, FLAML "lean"
  preprocessing (constant columns dropped by our code, `skip_transform=True`, because FLAML's
  own transformer ran out of memory), H2O with XGBoost excluded and a 7G heap
  (`H2O_STEP4_MAX_MEM` in `src/leakage.py`). If you re-run any leakage run on the laptop,
  re-run its whole with/without set there too, or the comparison stops being equal-hardware.
  The Step 4 bias and explain agents ran side by side with 2 threads each, so their timings
  are from a shared machine (the write-ups say so).

### Stage status
| Stage | Status | Evidence |
|---|---|---|
| Steps 0–3 | Done, reviewed (see the table in section 1) | |
| Step 0 cloud data path (zip reader, machine scaling) | **Done, passed review** (2 rounds), commit `ed12fcf` | `tests/test_raw_files.py`, `reports/build_cache_cloud.log` |
| Step 4A leakage | **Done, NOT reviewed** | `reports/leakage.md`, `tables/leakage.csv`, `figures/leakage_jump.png`, `models/leakage/`, `reports/leakage_logs/`, `reports/leaderboards/leakage/` |
| Step 4B temporal, bias, drift, challenge set | **Done, NOT reviewed** | `reports/temporal_bias.md`, `tables/{random_vs_future,cv_comparison,gap_*,class_balance,bias_slices,class_weight,prevalence,drift,challenge}.csv`, `figures/{bias_slices,drift}.png`, `models/eval/` |
| Step 4C SHAP, cross-tool agreement, calibration, cost threshold | **Done, NOT reviewed** | `reports/explainability.md`, `tables/{shap_*,cross_tool_*,calibration,threshold}.csv`, `figures/{shap_*,cross_tool_agreement,calibration,pr_cost}.png`, `models/threshold.json` |
| Step 4 synthesis | **Done, NOT reviewed** | `reports/critical_eval.md`, `reports/limitations.md`, `tables/{precision_both_sets,gap_vs_tuned,automl_self_estimate}.csv`; `run_all.py --steps eval` wired |
| Step 4 review gate | **Next.** Round 1 reviewers had just started when the run was stopped; nothing from them was kept | |
| Step 5 deployment | Not started (`app/` is empty) | |
| Step 6 report, slides, notebooks | Not started | |
| Final gate (D8) | Not started | |

### Known points the Step 4 review must judge (raised by the agents themselves)
- FLAML and H2O ran 3–7 min past the 600 s budget (`leakage.md`, run-time table); AutoGluon
  stayed within ~10 s. The sleep guard does not flag overruns, only suspend gaps.
- On the VM, H2O at 600 s scored clearly below its Step 3 run (test_future AUC ~0.992 vs
  ~0.997), and AutoGluon trained in effect one LightGBM model (memory). Both are stated.
- The AutoGluon leakage predictors (~2 GB each) were deleted after scoring to save disk, so
  those rows can only be re-scored by re-running them.
- Two failed attempts (FLAML clean OOM, H2O clean killed with a 9G heap) left logs but no
  records; all final records come from the fixed code.
- Threshold and calibrators are fitted on train out-of-fold scores from 2/3-size fold models,
  while the deployed model is trained on all of train (stated, not corrected).
- 1%-FPR thresholds exist only for tuned LightGBM, default LightGBM and LogReg (only they
  have train OOF scores); the other models are reported at 0.5 on the challenge set.
- The forward CV confounds recency with training-set size (stated; a size-matched control
  exists only for drift).
- `README.md` was updated by the synthesis agent (status row 4: "outputs produced, review gate
  pending"); check it is current.

### How to resume the orchestration
`docs/handoff/workflow_cloud_reference.js` is the workflow script that ran on the cloud. To
continue: drop its Step 0 gate, the Step 4 "do" agents and the synthesis agent (all done),
start at `const g4 = await gate('Step 4 critical evaluation', ...)`, and change the MACHINE
block of `CONTEXT` back to the laptop (Windows paths, 16 cores, 63 GB, plenty of disk, no
Playwright/LibreOffice notes; a benign PE for upload tests is `C:\Windows\System32\notepad.exe`).
Its prompts for the Step 4 gate, Steps 5–6 and the final gate are the detailed spec.

The `wip-backup` branch on GitHub holds unreviewed snapshots taken while the workflow ran.
It is superseded by `step4-6-cloud` and can be deleted.

## 1. State at the first handoff (laptop → cloud; kept for history)

| Stage | Status | Evidence |
|---|---|---|
| Step 0: environment, data, splits | Done, reviewed | `reports/data_check.json`, `data/splits/splits_info.json`, `tests/test_splits.py` |
| Step 2: baselines and tuning (Grid / Random / Optuna, 36 configs each) | Done. Passed adversarial review (2 rounds; round 2 clean apart from minor issues, which were then fixed) | `reports/tables/tuning_summary.csv`, `tuning_pairwise.csv`, `reports/figures/tuning_*.png`, `reports/tuning_writeup.md` |
| Step 3: AutoML (FLAML, AutoGluon, H2O, 1200 s each, run one after another) | Done. Passed adversarial review (2 rounds; minor issues fixed afterwards) | `reports/leaderboards/*`, `reports/tables/automl_vs_manual.csv`, `automl_pairwise.csv`, `reports/figures/automl_vs_manual.png`, `reports/automl_comparison.md` |
| **Step 4: critical evaluation** | **Not done. Redo it fully** (the user's decision). | see section 3 |
| Step 5: deployment (FastAPI, Streamlit, model card) | Not started (`app/` is empty) | |
| Step 6: report PDF, slides, executed notebooks | Not started (2 draft notebooks exist, not executed) | |
| Final gate (D8) | Not started | |

All the review verdicts and fixer reports from Steps 2–3 are in `docs/handoff/review_log.md`.
They show what was already checked, and how.

`pytest` passed at handoff: 67 tests in about 34 s, on Windows with the full data present.

### Step 4 at handoff: partially run, archived, to be redone
The Step 4 leakage agent was interrupted midway. Before handoff:
- **Code it wrote is kept but unreviewed:** `src/leakage.py`, `src/leakage_report.py`,
  `src/leakage_writeup.py`, `scripts/run_leakage.py`, `tests/test_leakage.py`, and
  `LEAKAGE_BUDGET_S` in `config.py`. It also edited `README.md` and possibly
  `src/automl_runs.py`. Treat it as a draft. Review it, then reuse or rewrite it. Nothing
  in it has passed a gate.
- **All of its outputs were removed from the repo** and moved to a local archive on the
  laptop (`_archive/step4_partial_20261029/`, gitignored). That covers models, predictions,
  leaderboards, logs, `leakage_quick.md/csv/png`, and its 4 full plus 13 quick records in
  `reports/experiments*.jsonl`.
- Step 4 starts clean: no `step == "leakage"` records remain in the experiment logs.

## 2. Key results so far (read from the generated tables, do not retype them)
- **`reports/tables/automl_vs_manual.csv`:** tuned LightGBM, AutoGluon, FLAML and H2O all
  reach test_future ROC-AUC of about 0.997–0.998, and their CIs overlap. LogReg falls to
  about 0.93.
- **Random-vs-future gap:** small for the tree models (AUC gap about 0.001; TPR@1%FPR drops
  about 1 point), large for LogReg (about 0.06). This is an honest, somewhat surprising
  finding. Step 4 must explain it (the time proxies already inside X, the 50/50 prevalence,
  the weekly stratification), not hide it.
- **Tuner differences** (Grid / Random / Optuna) are within the bootstrap noise.
- **H2O ran past its 1200 s budget** (wall time about 1487 s), because the stacked ensemble
  is built after the search. This is logged in `over_budget_seconds` and explained in the
  write-up.
- **Step 2 searches took about 4.7 h and Step 3 about 1.5 h on the laptop.** Do not re-run
  them unless strictly necessary. `run_tuning.py --refit-only` and
  `run_automl.py --tables-only` rebuild outputs from the logs.

## 3. What is left (spec and Definition of done)
The full spec is the D1–D8 table in `docs/PLAN.md`. The prompts in
`docs/handoff/workflow_windows_reference.js` give the per-stage detail. In short:

**Step 4 (D5), to redo from scratch:**
1. **Leakage:**
   - Add `detection_ratio` (the float in meta, `detection_ratio_float`) as a leak column,
     and separately `first_submission_date`.
   - Run tuned LightGBM plus FLAML, AutoGluon and H2O with and without each leak, at equal
     budget. 600 s is the stated deviation from 1200 s; the clean reruns also use 600 s.
   - Report the score jump, the leak's importance rank and SHAP, and whether each tool
     warned about or dropped the leak.
   - Also ablate the time-proxy columns already in X (COFF timestamp, column 696, and the
     authenticode signing times).
2. **Random vs future gap table:** for all models, with paired-bootstrap CIs (already
   implemented in `src/eval.py`).
3. **CV comparison:** stratified CV vs a week-blocked forward CV. Which one predicts
   test_future better?
4. **Bias:**
   - Class balance, per-slice metrics (week, file-size quartile, packer, signed or
     unsigned) and benign FPR per slice.
   - A class-weight variant, and whether it helped.
   - A prevalence discussion: the classes are 50/50 by construction.
   - The challenge set: 814 Win64 rows, all malicious, so recall only.
5. **SHAP** on tuned LightGBM:
   - A beeswarm plot, a bar chart grouped by feature family (offsets from
     `src/data.feature_groups()`), and 3 waterfalls (one TP, one FP, one FN).
   - Cross-tool agreement with the AutoML importances in `reports/leaderboards/`.
6. **Stretch goals:**
   - Calibration: a reliability curve and the Brier score.
   - A cost-sensitive threshold at C_FN/C_FP = 10, written to `models/threshold.json`.
   - A drift monitor: a sliding 12-week retrain, with F1 and AUC per future week.
7. **Write-ups:** `reports/critical_eval.md` and `reports/limitations.md`, then the review
   gate (leakage, stats and honesty lenses).

**Step 5 (D6):**
- `models/final/` holding the model, `schema.json` and the threshold.
- `app/api.py`, a FastAPI service:
  - `/health`.
  - `/predict`: a vector of length 2568 with finite values. Anything else gets a 422.
  - `/predict_file`: in memory only, 50 MB cap, 400 on a parse error. Never write uploads
    to disk or execute them.
- `app/streamlit_app.py`.
- `tests/test_api.py`, including a check that API scores match offline scoring.
- Screenshots in `reports/screenshots/`, plus `MODEL_CARD.md` and `DEMO.md`.

**Step 6 (D7):**
- `scripts/build_report.py` produces `reports/report.pdf`. Render the pages to images and
  check them.
- `scripts/build_slides.py` produces a `.pptx` of 8–10 slides with speaker notes, plus
  `reports/qa_prep.md`.
- All notebooks executed, with their outputs.

**Final (D1, D8):**
- `reports/repro_check.md`: a fresh venv, `pytest`, and `run_all.py --quick` run from
  deleted quick outputs.
- `reports/SUBMISSION_CHECKLIST.md`, which maps every rubric item to evidence.
- `run_all.py` gets its `eval` and `deploy` steps wired in.
- 2 consecutive clean adversarial rounds, then your own spot-check.

## 4. Setting up a fresh (Linux / cloud) machine

### 4.1 Python environment
The pins were frozen on Windows with Python 3.12.10. On Linux:
```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```
- `run_all.py` already looks for both `.venv/Scripts/python.exe` and `.venv/bin/python`.
  Some docs and prompts use the Windows path; translate it.
- If a pin has no Linux wheel, change only that pin and record why in `requirements.txt`.
- **H2O needs Java** (Java 21 was used), for example
  `apt-get install -y openjdk-21-jre-headless` or `openjdk-17-jre-headless`.
- `H2O_MAX_MEM` in `src/automl_runs.py` is computed at import: 24G (as on the 63 GB laptop
  of Steps 2–3), capped at about 60% of RAM. It is 9G on the 15 GB cloud VM. No edit needed.

### 4.2 Data (not in git: 6.6 GB of zips, optionally 32 GB extracted, plus 6.2 GB cache)
- **Check the disk and RAM first** with `df -h .` and `free -g`. Disk needed:
  - 6.6 GB of zips (always kept in `data/raw/_zips/`)
  - 6.2 GB of cache
  - plus 32 GB of extracted jsonl **only without `--keep-zipped`** (about 50 GB in total;
    about 13 GB with `--keep-zipped`)
- **`--keep-zipped`** (on `run_all.py` and `scripts/download_data.py`) leaves the Win64 zips
  packed. `src.data.raw_files()` / `raw_file(name)` then return `RawFile` objects whose
  `.open()` streams the zip member. The bytes and the byte offsets are identical to the
  extracted files, so the cache and the splits are the same. Extracted files take
  precedence when present; a partial extraction raises an error instead of silently hiding
  weeks. Code that re-reads raw lines must use `data.raw_file(name).open()`, never a path
  under `data/raw/win64/`. The cloud VM was built this way (only `challenge/` is extracted).
- Loading the 160k-row train split takes about 1.6 GB of RAM.
- If the machine is too small, stop and tell the user. Do not shrink the experiment.

```bash
.venv/bin/python run_all.py --steps data                  # download (sha256-checked) + extract + build cache + verify
.venv/bin/python run_all.py --steps data --keep-zipped    # same, without extracting 32 GB of jsonl
```
This downloads from Hugging Face `joyce8/EMBER2024` and needs network access to
huggingface.co. It took about 5 min to build the cache on the laptop, plus the download time.

### 4.3 Check the rebuilt data matches the laptop exactly
- `data/splits/*.npy` (the row indices) are committed.
- `data/splits/split_sha256.parquet` holds the sha256 and label for every row of train,
  test_random and test_future.
- `data/splits/split_fingerprints.json` holds hashes of each split's sha256 order and of
  the first 1000 feature rows.

After the data step:
```bash
git status data/splits          # must show no modified files (split files are deterministic)
.venv/bin/python - <<'EOF'
import hashlib, json, sys
import numpy as np
sys.path.insert(0, ".")
from src import data
fp = json.load(open("data/splits/split_fingerprints.json"))
for name in ["train", "test_random", "test_future"]:
    rows = data.split_rows(name)
    meta = data.load_cache_meta(data.SPLIT_SOURCE[name]).iloc[rows]
    assert hashlib.sha256("\n".join(meta["sha256"]).encode()).hexdigest() == fp[name], name
    X = np.asarray(data.open_cache_X(data.SPLIT_SOURCE[name])[rows[:1000]], dtype=np.float32)
    ok = hashlib.sha256(X.tobytes()).hexdigest() == fp[name + "_X_first1000_sha256"]
    print(name, "sha256 order OK;", "features identical" if ok else "FEATURES DIFFER (see note)")
EOF
.venv/bin/python -m pytest -q
```
**If the features differ but the sha256 order matches,** the thrember extraction is not
bit-identical across OSes. Check the size of the difference first:
- Tiny float noise: say so in `reports/repro_check.md` and carry on.
- A real difference: score the committed Step 2 models on the rebuilt test sets and compare
  with `tuning_summary.csv` before deciding anything. Tell the user.

### 4.4 Models
Committed:
- the LightGBM models (`models/lgbm_*.txt` with their params)
- `models/logreg.joblib`
- FLAML (`models/flaml/`)
- H2O (`models/h2o/`)

**Not committed:** the AutoGluon Step 3 predictor (`models/autogluon/`, 1.8 GB). Its
leaderboard, importances and saved test predictions are committed in
`reports/leaderboards/` and `reports/predictions/automl_autogluon.npz`. Rebuild it only if a
later step really needs the predictor object, and expect different timings on different
hardware:
```bash
.venv/bin/python scripts/run_automl.py --tools autogluon
```

### 4.5 Hardware differences
- Steps 2–3 ran on a 16-core Intel Core Ultra 9 285H, 63 GB RAM, Windows 11, CPU only, with
  `N_JOBS = 16` and H2O at 24G.
- Step 4 onwards runs on a cloud VM: 4 vCPU Intel Xeon @ 2.1 GHz, 15 GB RAM, no swap,
  Linux, about 13 GB free disk. There `config.N_JOBS = min(16, cpu_count) = 4` and
  `H2O_MAX_MEM = 9G`, both computed automatically. The Step 3 AutoML, Step 4 leakage,
  eval-fit and explain records in `experiments.jsonl` log their own `n_jobs`; the Step 2
  tuning and threshold records do not (Step 2 used `num_threads = N_JOBS = 16`, see
  `src/tune.py` `BASE_PARAMS`).
- On a cloud machine, record its CPU count, RAM and OS in the Step 4 write-up.
- Run every with/without-leak pair on the same machine with the same settings, so those
  comparisons stay fair.
- Do not compare cloud wall-clock times with the laptop times from Steps 2–3 without
  saying so.
- `N_JOBS` follows the core count automatically (capped at 16). State the deviation in
  every write-up that reports times or budgets.

### 4.6 Line endings in the history
`.gitattributes` is `* -text` (files stored byte-exact). Most files were committed with
CRLF on Windows. Commit `070f658` (cloud data path) rewrote 8 files with LF, so its plain
diff shows whole-file rewrites (e.g. 1111 changed lines in `src/data.py`) that hide a
~100-line real change, and the repo now has mixed endings (harmless at runtime). Review
that commit with `git show -w --ignore-cr-at-eol 070f658`. When editing a CRLF file, keep
its CRLF endings (`git ls-files --eol <file>` shows which a file has).

## 5. Lessons from the first run (avoid repeating them)
- **Raw duplicates.** Each weekly jsonl lists every sample twice. The global sha256 dedup
  is essential; without it the random split leaks.
- **signify pin.** `signify` 0.9.x breaks `import thrember`, so it is pinned to 0.8.1.
- **Shell timeouts.** A long refit was killed by a shell call's 2-minute timeout. Run long
  jobs as detached processes that write a log.
- **Sleep and timed budgets.**
  - The laptop slept during a timed AutoML run, which invalidates the budget.
  - `src/automl_runs.py` now has a heartbeat sleep guard. It logs `heartbeat_max_gap_s`
    and `timing_valid`, and `run_automl.py` reruns invalid attempts. Keep it for every
    timed run, including the Step 4 leakage reruns.
- **Network errors and false passes.**
  - Transient API or DNS failures killed agents.
  - In the first workflow version, a failed reviewer counted as "no issues", which caused
    a false pass.
  - The orchestration must treat a missing result as a failure.
- **Early stopping on the refit.** Early stopping on a held-out slice cut the grid model to
  187 trees, against about 719 in CV. The refit now uses a fixed round count taken from the
  mean CV best iteration (`REFIT_VERSION = 2` in `src/tune.py`).
