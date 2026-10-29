export const meta = {
  name: 'ember-steps4-6-cloud',
  description: 'EMBER2024 project on the cloud VM: review the zip-reader change, then Steps 4-6 each with adversarial verify/fix loops, then the final submission gate',
  phases: [
    { title: 'Step 0 cloud check', detail: 'review the zip-reader/cloud changes' },
    { title: 'Step 4 Critical eval', detail: 'leakage, temporal, bias, SHAP, calibration, cost, drift, challenge set' },
    { title: 'Step 5 Deploy', detail: 'export, FastAPI, Streamlit, model card, screenshots' },
    { title: 'Step 6 Docs', detail: 'PDF report, slides, Q&A prep, executed notebooks' },
    { title: 'Final gate', detail: 'checklist + repro, then 2 consecutive clean adversarial rounds' },
  ],
}

const SCRATCH = '/tmp/claude-0/-home-user-automl/4d6a0ab6-5dd7-59e6-bd5d-588de4ea1cb4/scratchpad'

// An agent that returns null (API/network failure) is retried. If it keeps failing, the
// workflow aborts so it can be resumed later. A missing result must never count as a pass.
async function run(prompt, opts, tries = 4) {
  for (let t = 1; t <= tries; t++) {
    const r = await agent(t === 1 ? prompt : `${prompt}\n\n(Retry ${t}: a previous attempt died on an API error. Inspect the current state of the files first and continue from where it left off rather than redoing finished work.)`, { ...opts, label: t === 1 ? opts.label : `${opts.label}:retry${t}` })
    if (r) return r
    log(`${opts.label}: attempt ${t} returned nothing`)
  }
  throw new Error(`ABORT: ${opts.label} failed ${tries} times; resume this workflow once the API is reachable`)
}
function allOk(results, what) {
  if (results.some(r => !r)) throw new Error(`ABORT: an agent in ${what} failed after retries; resume later`)
  return results
}

const CONTEXT = `
PROJECT: /home/user/automl (Linux cloud VM, bash; venv python: .venv/bin/python, Python 3.12). Graded course project: malware detection on EMBER2024 Win64 PE, manual LightGBM tuning vs AutoML (FLAML, AutoGluon, H2O). CLAUDE.md (already in your context) has the binding ground rules.

READ FIRST: docs/handoff/HANDOFF.md (state + spec of what is left), docs/PROJECT_BRIEF.md (rubric), docs/PLAN.md (Definition of done D1-D8). README.md, config.py, src/, reports/ show the current state.

MACHINE (differs from the laptop that ran Steps 2-3; say so wherever timings or budgets appear):
- 4 vCPU Intel Xeon 2.1 GHz, 15 GB RAM, no swap, Linux. Steps 2-3 ran on a 16-core Intel Core Ultra 9 285H with 63 GB RAM (Windows). config.N_JOBS is now min(16, cpu_count) = 4; H2O_MAX_MEM in src/automl_runs.py is capped at ~60% of RAM (9G). Never compare cloud wall-clock times with laptop times without saying so. All with/without-leak pairs run on THIS machine with the same settings.
- DISK IS TIGHT: about 13 GB free. Check "df -h ." before anything that writes models. AutoGluon predictors are ~2 GB each: after a run has been scored and its leaderboard/importances/predictions saved, delete its predictor folder (or fit with save_space / keep only the best model), and record that in the write-up. Never fill the disk.
- Data is rebuilt and verified identical to the laptop (split sha256 order and first-1000-row feature hashes match split_fingerprints.json; pytest 67 passed). The Win64 weekly jsonl are NOT extracted: they stay inside data/raw/_zips/Win64_{train,test}.zip and src.data.raw_files()/raw_file(name) return RawFile objects whose .open() streams the member (byte offsets identical to the extracted files). Any code that re-reads raw lines (e.g. src/leakage.py around line 149) must use data.raw_file(name).open(), not a path under data/raw/win64. The challenge set IS extracted under data/raw/challenge/*.jsonl.
- Java is installed for H2O. LibreOffice (soffice) and pdftoppm are installed for rendering. Chromium + Node Playwright are preinstalled: require('playwright') works from node and chromium.launch() needs no options; do NOT run "playwright install". If you use Python playwright, pin it and launch with executable_path='/opt/pw-browsers/chromium' (check the exact binary path under /opt/pw-browsers).
- Benign Windows PE files for upload tests (Linux has no notepad.exe): .venv/lib/python3.12/site-packages/pip/_vendor/distlib/t64.exe and w64.exe (pip's launcher stubs). Use them read-only.

ESTABLISHED FACTS (from Steps 0-3, independently verified; read the tables, do not retype numbers):
- Raw weekly files repeat each sample twice; global sha256 dedup leaves 520k official-train and 119,993 official-test rows. D=2568 thrember features in 12 groups (offsets from src.data.feature_groups()). Splits: train 160k and test_random 40k, stratified by label x week from weeks 0-51; test_future 40k from weeks 52-63, strictly later. Tuning subset 60k with 3 persisted folds. Classes 50/50 by construction.
- detection_ratio ('k/n' string; detection_ratio_float in meta) alone separates the labels (AUC 1.0). It is in meta ONLY, never in X: the planned leaky feature. first_submission_date is in meta (the time leak).
- Time proxies already inside X: column 696 (header COFF timestamp) and the authenticode signing-time columns (around 2472-2479; confirm exact indices from thrember source).
- Challenge set: data/raw/challenge, 6,315 malicious rows that AV initially missed, 814 Win64 (filter on file_type). Unused so far.
- Step 2: reports/tables/tuning_summary.csv; refit uses fixed rounds (REFIT_VERSION=2 in src/tune.py). Tuned LightGBM test_future AUC ~0.998; LogReg ~0.93 with a ~0.06 random-vs-future gap; tree models' gap ~0.001.
- Step 3: reports/tables/automl_vs_manual.csv, reports/leaderboards/*, reports/automl_comparison.md. Models committed: models/lgbm_*.txt (+params), models/logreg.joblib, models/flaml/, models/h2o/. The Step 3 AutoGluon predictor is NOT on disk (its predictions are in reports/predictions/automl_autogluon.npz). The sleep/heartbeat guard in src/automl_runs.py (timing_valid) must be kept for every timed run.
- Draft Step 4 code from an interrupted earlier agent exists and is UNREVIEWED: src/leakage.py, src/leakage_report.py, src/leakage_writeup.py, scripts/run_leakage.py, tests/test_leakage.py, LEAKAGE_BUDGET_S=600 in config.py. Review it critically; reuse or rewrite. None of its outputs exist (no step=="leakage" records in the logs).

RULES FOR THIS RUN:
- Temporary scripts and throwaway outputs go in ${SCRATCH}, not the repo.
- Do NOT git commit or push (a dedicated step commits after each gate passes).
- Long jobs (>2 min) run detached (nohup ... > log 2>&1 &) and you poll the log with short checks (each shell call under ~9 minutes). A shell timeout must never kill a run.
- Timed AutoML runs need the machine to themselves: before starting one, check with ps that no other heavy python job (training, SHAP, AutoML, nbconvert) is running.
- Never select, tune, early-stop or pick thresholds on test_random/test_future.
`

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          file: { type: 'string' },
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'command run / code line / number that proves it' },
          suggested_fix: { type: 'string' },
        },
        required: ['severity', 'file', 'claim', 'evidence'],
      },
    },
    checks_run: { type: 'string', description: 'the concrete commands/recomputations you ran, with their outcomes' },
    verdict: { type: 'string' },
  },
  required: ['issues', 'checks_run', 'verdict'],
}

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    fixed: { type: 'string', description: 'each issue fixed: what changed (files) and what was re-run' },
    disputed: { type: 'string', description: 'issues not changed because they are wrong or are genuine findings, with concrete evidence' },
    deferred: { type: 'string', description: 'minor issues documented instead of fixed, and where' },
    tests: { type: 'string', description: 'pytest result after the fixes' },
  },
  required: ['fixed', 'disputed', 'deferred', 'tests'],
}

const DO_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    files_written: { type: 'string' },
    results: { type: 'string', description: 'key numbers, as markdown tables copied from the generated CSVs' },
    observations: { type: 'string', description: 'honest observations, surprises, anything suspicious' },
    problems: { type: 'string' },
  },
  required: ['ok', 'files_written', 'results', 'observations', 'problems'],
}

const REVIEW_BASE = `${CONTEXT}
YOU ARE AN INDEPENDENT, ADVERSARIAL REVIEWER. You did not do this work. Assume it is wrong until your own checks show otherwise.
- Do NOT edit project files. Read files, run read-only checks, run pytest, and write throwaway scripts only in ${SCRATCH}.
- Report only issues backed by concrete evidence: a command and its output, a file:line, or a recomputed number. List what you ran in checks_run; a review with no concrete checks is worthless.
- Severity:
  - blocker: invalid results, leakage, crashes, a missing required deliverable.
  - major: a number or claim that is wrong or unsupported, a rubric item only partly met, a broken reproduction path.
  - minor: clarity, polish, small inconsistencies.
- A genuine but unflattering experimental result is NOT an issue. Its absence from the write-up would be.`

const ledger = []

async function commitStage(stage, phaseName) {
  await run(`${CONTEXT}
YOUR TASK: stage "${stage}" just passed its adversarial review. Commit the verified work LOCALLY on the current branch (step4-6-cloud). Do NOT push.
- Run git status. Stage the project files for this stage. Never stage data/raw, data/cache, .venv, AutoGluon predictor folders, or any file over ~50 MB (check sizes with du; respect .gitignore).
- If an artefact over 50 MB is needed for the deliverable, leave it out and list it in your reply.
- Commit with author "Isaac Benjamin David <isaac.ibddavid@gmail.com>" (git -c user.name=... -c user.email=... commit), a short descriptive message, and these two trailer lines at the end of the message:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TZer4U2jgmUFpCzuXNABao
Reply with the commit hash and the file list.`, { label: `commit:${stage}`, phase: phaseName, effort: 'low' })
}

// Generic verify -> fix loop. Returns { passed, rounds, unresolved }.
async function gate(stage, phaseName, scope, lenses, opts = {}) {
  const maxRounds = opts.maxRounds || 5
  const needClean = opts.consecutiveClean || 1
  let clean = 0
  let notes = ''
  let lastSerious = []
  for (let round = 1; round <= maxRounds; round++) {
    const raw = await parallel(lenses.map(l => () => run(`${REVIEW_BASE}
STAGE UNDER REVIEW: ${stage}
SCOPE: ${scope}
YOUR LENS: ${l.prompt}
${notes ? `NOTES FROM THE PREVIOUS FIX ROUND. Judge these on evidence. Do not re-raise an issue that was refuted with evidence unless you have new evidence; do re-raise it if the "fix" did not actually fix it:\n${notes}` : ''}`,
      { label: `verify:${stage}:${l.key}:r${round}`, phase: phaseName, schema: REVIEW_SCHEMA })))
    allOk(raw, `${stage} review round ${round}`)
    const issues = []
    raw.forEach((r, i) => r.issues.forEach(x => issues.push({ ...x, lens: lenses[i].key })))
    const serious = issues.filter(x => x.severity !== 'minor')
    const minors = issues.filter(x => x.severity === 'minor')
    lastSerious = serious
    ledger.push({ stage, round, serious: serious.length, minor: minors.length, verdicts: raw.map(r => r.verdict.slice(0, 400)) })
    log(`${stage} round ${round}: ${serious.length} blocker/major, ${minors.length} minor`)
    if (serious.length === 0) {
      clean++
      if (minors.length) {
        const f = await run(`${CONTEXT}
YOU ARE THE FIXER for stage "${stage}". The reviewers found no blocker or major issues, only these minor ones. Fix each one if it's cheap (<15 min). Otherwise document it in the relevant markdown (README, the stage write-up, or reports/limitations.md). Never hand-edit numbers. Regenerate outputs with the scripts. Run pytest at the end.
ISSUES (JSON): ${JSON.stringify(minors)}`, { label: `fix-minor:${stage}:r${round}`, phase: phaseName, schema: FIX_SCHEMA })
        notes = `fixed: ${f.fixed}\ndisputed: ${f.disputed}\ndeferred: ${f.deferred}\ntests: ${f.tests}`
      }
      if (clean >= needClean) { await commitStage(stage, phaseName); return { passed: true, rounds: round, unresolved: [] } }
      continue
    }
    clean = 0
    if (round === maxRounds) break
    const f = await run(`${CONTEXT}
YOU ARE THE FIXER for stage "${stage}" (round ${round}). Reviewers raised the issues below.
- Blocker/major: fix the ROOT CAUSE in code, re-run only the affected computations, and regenerate tables and figures from the logs.
- Minor: fix if cheap, otherwise document.
- If an issue is wrong, or describes a genuine experimental finding, do not change anything for it: DISPUTE it with concrete evidence.
- Never hand-edit numbers. Never delete experiment-log records. Append new ones and make the table builders pick the latest valid record, with a version or supersedes field documented in code.
- Keep the code readable and commented.
- Run pytest at the end.
ISSUES (JSON): ${JSON.stringify(issues)}`, { label: `fix:${stage}:r${round}`, phase: phaseName, schema: FIX_SCHEMA })
    notes = `fixed: ${f.fixed}\ndisputed: ${f.disputed}\ndeferred: ${f.deferred}\ntests: ${f.tests}`
  }
  log(`${stage}: hit the ${maxRounds}-round cap with ${lastSerious.length} unresolved serious issues; carrying them to the final gate`)
  await commitStage(`${stage} (unresolved issues carried to final gate)`, phaseName)
  return { passed: false, rounds: maxRounds, unresolved: lastSerious }
}

const carried = []

// ---------------- Step 0: cloud changes ----------------
phase('Step 0 cloud check')
const g0 = await gate('Step 0 cloud data path', 'Step 0 cloud check', 'Commit 070f658 (git show 070f658): src/data.py RawFile / flat_jsonl_name / raw_files / raw_file, scripts/download_data.py --keep-zipped, scripts/verify_data.py, config.py N_JOBS, src/automl_runs.py _h2o_max_mem, requirements.txt oscrypto pin. Plus the rebuilt data/cache and reports/data_check.json.', [
  { key: 'data', prompt: 'Correctness of the zip-reader path and the rebuilt data: independently recompute the split fingerprints in data/splits/split_fingerprints.json from the cache; re-read 10 random cached rows from the zips by their source_file/source_offset and re-vectorise them with thrember, compare to the cached X; check that the extracted-file path still works (extract ONE small weekly test file to ' + SCRATCH + ', point raw_files at that dir, compare RawFile bytes); check the zip-slip protection still holds; check verify_data.py ran clean (reports/data_check.json hard checks). Also judge readability of the new code for a student team and whether README/HANDOFF need a note about --keep-zipped and the 4-core/15 GB machine. Run pytest.' },
])
if (!g0.passed) carried.push({ stage: 'Step 0', issues: g0.unresolved })

// ---------------- Step 4 Critical evaluation ----------------
phase('Step 4 Critical eval')
const s4leak = await run(`${CONTEXT}
YOUR TASK: Step 4 part A, leakage experiments. CPU-heavy and timed, so you run ALONE (nothing else runs while you work). You own src/leakage.py, src/leakage_report.py, src/leakage_writeup.py, scripts/run_leakage.py and tests/test_leakage.py. Review the existing draft first; reuse what is right, fix or rewrite what is not.
- Leak 1, detection_ratio_float: append it as an extra column to X for train, test_random and test_future. Run:
  - the tuned LightGBM (best Step 2 config, fixed n_estimators as in src/tune.py REFIT_VERSION 2),
  - FLAML, AutoGluon and H2O, each with LEAKAGE_BUDGET_S=600 s (a stated deviation from 1200 s). Also run the matching 600 s CLEAN versions on this same machine, so every with/without comparison is at an equal budget and on equal hardware.
- Leak 2, raw first_submission_date as unix seconds: same procedure for all four model types. Show whether it helps on the random split and whether it transfers to test_future, where the dates are out of the training range.
- Ablation of the time proxies already in X: remove column 696 (COFF timestamp) and the authenticode signing-time columns (identify exact columns from thrember's source and document them). Retrain the tuned LightGBM and compare random vs future.
- For each run, log metrics on both test sets with paired-bootstrap CIs (src/eval.py). Log the leaky feature's importance rank (the tool's own importance where available, SHAP for tree models), and whether each tool warned about, flagged or dropped it (read the tool logs to check for warnings and record what you grepped).
- Keep the heartbeat sleep guard (timing_valid) for every timed run; rerun invalid attempts.
- Mind the disk: delete each AutoGluon predictor folder after scoring and saving its artefacts.
- Outputs: reports/tables/leakage.csv, reports/figures/leakage_jump.png, reports/leakage.md. The write-up says honestly which tools "fell for" the leak (likely all: a feature with AUC 1.0 is perfectly predictive and no tool can know it is post-hoc), why that matters, and what protects against it: a domain review of feature provenance, and temporal validation for the date leak. State the machine (4 cores) and the 600 s budget deviation.
- Use --quick first (writes *_quick files), then the full run detached in the background; poll the log until done. Run pytest at the end.`, { label: 'step4-leakage', phase: 'Step 4 Critical eval', schema: DO_SCHEMA })

const [s4bias, s4explain] = allOk(await parallel([
  () => run(`${CONTEXT}
YOUR TASK: Step 4 part B, temporal, bias and drift. Another agent is computing SHAP at the same time on this 4-core machine, so don't publish timing claims from this work, and use n_jobs=2 for your training jobs. You own src/temporal.py, src/bias.py, src/drift.py, scripts/run_eval.py (shared with the explain agent: put your entry points in your own modules and add a section to run_eval.py; or create scripts/run_eval_bias.py if that's simpler) and their tests. Use the best tuned LightGBM, plus the AutoML winners where cheap, by loading the saved models or saved predictions (reports/predictions/*.npz).
1. Random vs temporal table:
   - Every model from Steps 2-3 on test_random vs test_future: AUC, F1, TPR@1%FPR, FPR, with paired-bootstrap CIs and the gap → reports/tables/random_vs_future.csv.
   - Compare stratified 3-fold CV against week-blocked forward CV (expanding window over weeks 0-51 of the train split, e.g. 4 folds) for the tuned config. Which estimate is closer to test_future? → reports/tables/cv_comparison.csv.
   - Explain WHY the tree models' gap is small (time proxies inside X, 50/50 prevalence, weekly stratification) using evidence, not speculation.
2. Bias and imbalance:
   - Report the class balance.
   - Per-slice metrics on test_future and test_random, including FPR on benign: week; file-size quartile; packed vs not packed (with packer names where n is big enough); signed vs unsigned; family known vs unknown (malicious only, recall). Show n per slice and CIs; mark slices that are too small.
   - Train a class-weighted / scale_pos_weight variant and report whether it helped (probably not at 50/50; explain why).
   - Evaluate at realistic prevalence: precision and alarms per 10k files at 1%, 5% and 10% malware prevalence from TPR/FPR (the base-rate effect).
   → reports/tables/bias_slices.csv, reports/figures/bias_slices.png, reports/tables/prevalence.csv, reports/tables/class_weight.csv.
3. Drift monitor: retrain the tuned config on a sliding 12-week window (weeks 40-51) and an expanding window, then plot AUC, F1 and TPR@1%FPR per future week 52-63 → reports/figures/drift.png, reports/tables/drift.csv.
4. Challenge set: vectorise the 814 Win64 challenge rows (malicious samples AV initially missed) from data/raw/challenge with the same thrember extractor and the same conventions (dedupe by sha256, label filter). Store them as a new cache file in data/cache without touching the existing cache. Report each model's detection rate at the deployment threshold (models/threshold.json if it exists yet, otherwise 0.5 and say so) and at the 1%-FPR threshold chosen on TRAIN out-of-fold or the threshold from tuning, never from the test sets → reports/tables/challenge.csv. Be honest if recall collapses.
5. Write reports/temporal_bias.md with the findings. Use --quick first where it applies. Run pytest at the end.`, { label: 'step4-temporal-bias', phase: 'Step 4 Critical eval', schema: DO_SCHEMA }),
  () => run(`${CONTEXT}
YOUR TASK: Step 4 part C, explainability, calibration and cost. Another agent is running drift and bias jobs at the same time on this 4-core machine; use n_jobs=2. You own src/explain.py, src/calibration.py, scripts/run_explain.py and their tests.
1. SHAP:
   - TreeExplainer on the best tuned LightGBM over a 5k stratified sample of test_future. Explaining is not tuning, so using test data is fine.
   - A beeswarm (top 20 features with readable names: map indices to thrember feature names where possible), a bar chart of mean |SHAP| grouped by the 12 EMBER feature families (offsets from src.data.feature_groups()), and 3 local waterfall plots: a confidently correct malicious (TP), a false positive and a false negative, each explained in words.
   - Comment on whether the time-proxy columns (696, signing time) rank high.
   → reports/figures/shap_*.png, reports/tables/shap_family_importance.csv.
2. Cross-tool agreement: family-level importance from the manual LightGBM SHAP, FLAML's winner (SHAP if tree model, else its importance), AutoGluon's feature_importance and H2O's varimp (reports/leaderboards/), all aggregated to the 12 families. Report top-k overlap (k=5 and 20 features) and Spearman rank correlation at family level → reports/tables/cross_tool_agreement.csv and a figure.
3. Calibration: reliability curve and Brier score on test_future for the tuned LightGBM and the AutoML winners (saved predictions are fine). Optionally isotonic/Platt calibration fitted on a TRAIN hold-out, never the test set → reports/figures/calibration.png, reports/tables/calibration.csv.
4. Cost-sensitive threshold:
   - Choose the threshold on TRAIN out-of-fold CV predictions (never the test sets), minimising C_FN*FN + C_FP*FP with default C_FN/C_FP=10. Also show sensitivity over ratios 1, 5, 10, 50 and the precision-recall curve.
   - Report the chosen threshold's performance on test_future.
   - Save to models/threshold.json (threshold, ratio, how it was chosen, source of predictions); deployment uses it.
   → reports/figures/pr_cost.png, reports/tables/threshold.csv.
5. Write reports/explainability.md. Use --quick first. Run pytest at the end.`, { label: 'step4-explain', phase: 'Step 4 Critical eval', schema: DO_SCHEMA }),
]), 'Step 4 parallel work')

const s4write = await run(`${CONTEXT}
YOUR TASK: synthesise Step 4. Read reports/leakage.md, temporal_bias.md, explainability.md and all their tables and figures. If the challenge-set table used a 0.5 placeholder because models/threshold.json did not exist yet, regenerate it now with the real threshold. Write:
- reports/critical_eval.md: answer the research question directly: how much random-split scores overstate real-world performance (use test_future, prevalence-adjusted precision and the challenge set), and whether AutoML tools protect against leakage and drift or fall for them. Cite numbers by pointing to the CSVs (numbers in markdown must be generated from the CSVs by a script, e.g. a writeup module, not typed by hand).
- reports/limitations.md: static features only, packed/obfuscated files, drift, dataset age and recency, 50/50 balance vs real prevalence, subsampling, label noise from AV-derived labels, adversarial evasion, the dedupe issue, CPU-only and the hardware change (Steps 2-3 on a 16-core laptop, Step 4+ on a 4-core cloud VM), the 600 s leakage budget.
Wire scripts/run_leakage.py, run_eval*.py and run_explain.py into run_all.py as --steps eval, with --quick working (remove "eval" from NOT_IMPLEMENTED). Run pytest.`, { label: 'step4-synthesis', phase: 'Step 4 Critical eval', schema: DO_SCHEMA })

const g4 = await gate('Step 4 critical evaluation', 'Step 4 Critical eval', 'src/leakage*.py, src/temporal.py, src/bias.py, src/drift.py, src/explain.py, src/calibration.py, scripts/run_leakage.py, scripts/run_eval*.py, scripts/run_explain.py, run_all.py eval step, reports/tables/{leakage,random_vs_future,cv_comparison,bias_slices,prevalence,class_weight,drift,challenge,shap_family_importance,cross_tool_agreement,calibration,threshold}.csv, reports/figures/*, reports/{leakage,temporal_bias,explainability,critical_eval,limitations}.md, models/threshold.json, step=="leakage" and Step 4 records in reports/experiments.jsonl.', [
  { key: 'leakage', prompt: 'Leakage experiment validity: the leaky columns really are added and removed; with/without budgets and hardware match; timing_valid holds; the importance ranks are real (check the artefacts); the claims about which tools fell for the leak or warned are evidenced by the logs; the threshold and calibration were NOT chosen on test data; the challenge-set vectorisation matches the main pipeline (re-vectorise a few rows yourself).' },
  { key: 'stats', prompt: 'Statistical correctness: recompute at least 4 numbers independently (one bias-slice benign FPR, one prevalence-adjusted precision, one drift-week AUC, one Brier score) and compare; check slice n and CIs are shown and tiny slices flagged; check the forward CV is truly forward in time; check SHAP family aggregation uses the correct offsets; check the bootstrap CIs are paired and seeded.' },
  { key: 'honesty', prompt: 'Honesty and rubric coverage: critical_eval.md answers the research question with evidence, does not overclaim, reports negative and surprising results, and covers every Step 4 bullet in docs/PROJECT_BRIEF.md and D5 in docs/PLAN.md plus the stretch goals (drift, cost threshold, calibration, cross-tool agreement). limitations.md is complete and specific. Numbers in markdown trace to CSVs. Code is readable for a student team.' },
])
if (!g4.passed) carried.push({ stage: 'Step 4', issues: g4.unresolved })

// ---------------- Step 5 Deployment ----------------
phase('Step 5 Deploy')
const s5 = await run(`${CONTEXT}
YOUR TASK: Step 5, deployment. You own app/, models/final/, MODEL_CARD.md, DEMO.md, tests/test_api.py and scripts/export_model.py.
1. Choose the deployment model by test_future TPR@1%FPR (with CIs), interpretability and inference simplicity (likely the tuned LightGBM). Justify the choice in MODEL_CARD.md.
   scripts/export_model.py writes models/final/model.txt and schema.json with: feature_dim 2568, feature names and group offsets, the thrember commit, the signify pin, the threshold from models/threshold.json, the training data description, the metrics source CSV, and a sha256 of the model file. Inference is self-contained; the model needs no scaler; state that.
2. app/api.py (FastAPI):
   - GET /health returns the model version and hash.
   - POST /predict takes {features: list[float]}. Pydantic validates length == 2568 and that all values are finite (422 otherwise); batch variant optional. Returns {p_malicious, label, threshold, model_version}.
   - POST /predict_file accepts an uploaded PE (50 MB cap), runs thrember PEFeatureExtractor on the bytes IN MEMORY and predicts. Parse errors give 400. Never write the upload to disk or execute it.
3. app/streamlit_app.py:
   - Choose a sample from test_future by index or filter (benign or malicious, correct or misclassified), or upload a file.
   - Call the API (configurable URL) and show p_malicious, the threshold, the label, the true label for test samples, and the top-10 SHAP contributions for that sample grouped by family.
   - A banner with the model card essentials and limits.
4. tests/test_api.py with FastAPI TestClient: health; a valid vector gives a probability equal to offline booster.predict to 1e-6; wrong length 422; NaN/inf 422 (note JSON cannot carry NaN normally, test what the API actually receives); non-numeric 422; /predict_file on the benign PE .venv/lib/python3.12/site-packages/pip/_vendor/distlib/t64.exe gives 200; random bytes give 400; oversize gives 413 or 400.
5. Start uvicorn and streamlit for real in the background. Exercise them with curl. Take screenshots of the Streamlit app with Playwright (preinstalled Chromium, see MACHINE notes) showing (a) a correct malicious prediction, (b) a benign prediction and (c) a file upload of t64.exe. Save to reports/screenshots/. LOOK at each PNG (Read it) to confirm it shows what it should. Stop the servers when done.
6. MODEL_CARD.md: model details, intended use (triage aid for static PE analysis, research), out-of-scope uses (sole verdict for blocking, non-Win64 files, adversarial settings), training data, metrics on both test sets plus the challenge set and prevalence-adjusted precision (generated from or citing the CSVs), threshold and cost ratio, calibration, known limits, ethical considerations, how to reproduce.
7. DEMO.md: a step-by-step 5-minute demo script with exact commands (Linux and Windows variants), plus how the user can deploy to Hugging Face Spaces or Streamlit Cloud with their own account.
8. Add --steps deploy to run_all.py (export + API tests); remove it from NOT_IMPLEMENTED. Run pytest.`, { label: 'step5-deploy', phase: 'Step 5 Deploy', schema: DO_SCHEMA })
const g5 = await gate('Step 5 deployment', 'Step 5 Deploy', 'app/api.py, app/streamlit_app.py, models/final/, scripts/export_model.py, tests/test_api.py, MODEL_CARD.md, DEMO.md, reports/screenshots/, run_all.py deploy step.', [
  { key: 'api', prompt: 'Adversarial API testing: start uvicorn yourself (a different port, background, stop it afterwards). Send malformed inputs (wrong length, NaN, strings, huge payloads, an empty upload, a non-PE upload), check status codes, and check /predict parity with offline scoring on 20 real test_future rows. Confirm uploads are never written to disk or executed (read the code; check no temp files appear). Run tests/test_api.py. Start the Streamlit app headless and confirm it loads without errors.' },
  { key: 'card', prompt: 'Model card and demo truthfulness: every number in MODEL_CARD.md matches the generated CSVs; intended/out-of-scope uses and limits match critical_eval.md and limitations.md; DEMO.md commands work as written (run the Linux ones); the screenshots exist and show what DEMO.md claims (open the PNGs).' },
])
if (!g5.passed) carried.push({ stage: 'Step 5', issues: g5.unresolved })

// ---------------- Step 6 Docs ----------------
phase('Step 6 Docs')
const [s6report, s6slides, s6nb] = allOk(await parallel([
  () => run(`${CONTEXT}
YOUR TASK: the PDF report. You own scripts/build_report.py, reports/report.pdf, reports/figures/architecture.png and reports/report_src/ if you need it.
- Build the PDF programmatically with reportlab (pin it) or another pure-Python library. Numbers, tables and figures must be pulled from reports/ CSVs and PNGs at build time, never typed by hand.
- Draw an architecture/workflow diagram (data → dedupe/vectorise → splits → tuning / AutoML → critical eval → export → FastAPI/Streamlit) with matplotlib, saved as architecture.png.
- Sections: title page (student Isaac Benjamin David, URK23AI1035, course from the brief, team TODO); abstract; problem statement and research question; dataset (version, license, size, dedupe finding, splits, class balance, feature families); methodology and workflow diagram; Step 2 tuning results; Step 3 AutoML comparison with leaderboards summarised; Step 4 critical evaluation (random vs temporal, leakage, bias/prevalence, SHAP, calibration, cost threshold, drift, challenge set); deployment with screenshots; reflections (what AutoML did well and where it failed); limitations; reproducibility (including the hardware change); references (EMBER2024 paper arXiv:2506.05074, EMBER 2018, the tools); appendix.
- About 15-25 pages. Readable fonts, captioned figures, tables that fit the page.
- After building, render every page to PNG (pymupdf, pinned, or pdftoppm) into ${SCRATCH}/report_pages and LOOK at each page image (Read the PNGs). Fix overflow, cut-off tables, missing figures and blank pages; repeat until clean.`, { label: 'step6-report', phase: 'Step 6 Docs', schema: DO_SCHEMA }),
  () => run(`${CONTEXT}
YOUR TASK: slides and Q&A prep. You own scripts/build_slides.py, reports/slides.pptx and reports/qa_prep.md.
- A deck for a 5-minute demo with python-pptx (pin it): 8-10 slides, generated from reports/ CSVs and PNGs. Slides: title; problem and why it matters; data and the duplicate finding; method and pipeline diagram (reports/figures/architecture.png if present, else draw your own); tuning results; AutoML vs manual; the research-question answer (random vs future, prevalence, challenge set); leakage (did AutoML fall for it); SHAP; deployment demo with a screenshot; limitations and takeaways.
- Speaker notes on every slide, about 30 s each, totalling about 5 min.
- Consistent design, body text at least 18 pt, no overflowing text.
- Render each slide to an image (soffice --headless --convert-to pdf, then pdftoppm) in ${SCRATCH}/slides and LOOK at each one. Fix any overflow.
- reports/qa_prep.md: 15-20 likely instructor questions with crisp answers and pointers to evidence (why EMBER2024, why LightGBM, what equal budget means, why the drift gap is small, why the AutoML tools fell for the leak, production deployment, adversarial evasion, the hardware change), including questions any team member must be able to answer about the code.`, { label: 'step6-slides', phase: 'Step 6 Docs', schema: DO_SCHEMA }),
  () => run(`${CONTEXT}
YOUR TASK: notebooks. You own notebooks/*.ipynb.
- Update 01_eda and 02_baseline_tuning, and create 03_automl and 04_critical_eval. Thin narrative wrappers that import from src and read reports/ outputs, with no duplicated logic and markdown cells explaining each step for a student audience.
- Execute all four with nbconvert/nbclient using the .venv kernel (register it with ipykernel if needed), so the saved notebooks contain outputs. Cells must not retrain expensive models; load saved artefacts. Keep RAM in mind (15 GB, other agents are working).
- Make sure every notebook executes without errors from a fresh kernel.`, { label: 'step6-notebooks', phase: 'Step 6 Docs', schema: DO_SCHEMA }),
]), 'Step 6 parallel work')
const g6 = await gate('Step 6 documentation', 'Step 6 Docs', 'reports/report.pdf, scripts/build_report.py, reports/slides.pptx, scripts/build_slides.py, reports/qa_prep.md, notebooks/*.ipynb, README.md.', [
  { key: 'numbers', prompt: 'Number traceability: extract the text of report.pdf and slides.pptx, pick at least 15 numbers across both, trace each one to a reports/ CSV, and flag any mismatch, stale value or untraceable number.' },
  { key: 'rubric', prompt: 'Rubric coverage: check report.pdf against brief Step 6 (problem statement, dataset description, workflow diagram, experiment tables, screenshots, tool comparison, leakage and bias findings, reflections) and the deck against "5-minute demo plus Q&A". Check the notebooks execute (nbconvert --execute into ' + SCRATCH + ', check for errors).' },
  { key: 'render', prompt: 'Visual quality: render every PDF page and every slide to PNG in ' + SCRATCH + ' and LOOK at them. Flag cut-off tables, overflowing text, unreadable fonts, missing or broken images, blank pages, and figures without captions.' },
])
if (!g6.passed) carried.push({ stage: 'Step 6', issues: g6.unresolved })

// ---------------- Final gate ----------------
phase('Final gate')
const pre = await run(`${CONTEXT}
YOUR TASK: prepare for the final submission gate.
1. reports/repro_check.md:
   - Create a FRESH venv in ${SCRATCH}/fresh_venv with python3.12, pip install -r requirements.txt (watch the disk: delete the fresh venv and run "pip cache purge" afterwards), and record whether it succeeds, how long it takes and any issues. Fix requirements.txt if it fails. State that this was verified on Linux; the Windows install was verified on the laptop in Steps 0-3 and the oscrypto git pin is the only requirement added since.
   - Delete all *_quick outputs (logs, splits, tables, figures, models) and run ".venv/bin/python run_all.py --quick" end to end in the background, recording the time and the result.
   - Run the full pytest suite and record the result. There should be no skips with the data present; investigate any skip.
2. reports/SUBMISSION_CHECKLIST.md: map EVERY rubric bullet in docs/PROJECT_BRIEF.md (Steps 1-6, the deliverables checklist, the ground rules, each stretch goal) and every D1-D8 item in docs/PLAN.md to concrete evidence files, with a status. List explicitly what only the user can do: Part A certificate, team/deadline/topic TODOs, recording the demo video, the cloud deploy, pushing to GitHub.
3. Update README.md (status table all done, links to every deliverable, how to reproduce, the cloud/zip notes) and docs/handoff/HANDOFF.md (current state).
Carried unresolved issues from earlier gates (must be resolved or explicitly documented): ${JSON.stringify(carried)}`, { label: 'final-prep', phase: 'Final gate', schema: DO_SCHEMA })
const gF = await gate('FINAL submission readiness', 'Final gate', 'The whole repository and every deliverable. Use docs/PROJECT_BRIEF.md and the Definition of done D1-D8 in docs/PLAN.md as acceptance criteria.', [
  { key: 'repro', prompt: 'Reproducibility (actually run things): pytest (all pass, no unexplained skips), run_all.py --quick end to end (in the background, poll), and spot-reproduce one Step 2 and one Step 4 number from saved artefacts. Check requirements.txt pins everything imported in src/ app/ scripts/ (grep the imports). Check the README instructions work as written.' },
  { key: 'completeness', prompt: 'Completeness critic: go through reports/SUBMISSION_CHECKLIST.md and verify each claimed evidence file exists and contains what is claimed. Find any brief rubric item, deliverable or D1-D8 criterion that is missing, partial or only claimed. Check the user-only items are clearly listed.' },
  { key: 'honesty', prompt: 'Numbers and honesty: cross-check headline numbers across README, report.pdf, slides.pptx, MODEL_CARD.md, critical_eval.md and the CSVs. They must be consistent and current. Flag overclaims, hidden negative results, any test-set use for selection, and inconsistencies between documents.' },
  { key: 'readability', prompt: 'Code readability and explainability for a student team: every module has a docstring; non-obvious choices carry "why" comments; functions are small and typed; no dead code, leftover debug code or synthetic-test artefacts in reports/ or models/; consistent naming. Flag files a team member would struggle to explain.' },
], { maxRounds: 6, consecutiveClean: 2 })

return { g0, s4: { leak: s4leak.observations, bias: s4bias.observations, explain: s4explain.observations, synth: s4write.observations }, g4, s5: s5.observations, g5, s6: { report: s6report.problems, slides: s6slides.problems, nb: s6nb.problems }, g6, pre: pre.results, gF, carried, ledger }
