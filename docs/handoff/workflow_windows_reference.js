export const meta = {
  name: 'ember-automl-to-submission',
  description: 'EMBER2024 project: fix Step 2, then Steps 3-6 each with adversarial verify/fix loops, then a final submission gate',
  phases: [
    { title: 'Step 2 fix', detail: 'fix refit/early-stopping + minor issues, re-verify' },
    { title: 'Step 3 AutoML', detail: 'FLAML, AutoGluon, H2O at 1200 s each, comparison' },
    { title: 'Step 4 Critical eval', detail: 'leakage, temporal, bias, SHAP, calibration, cost, drift, challenge set' },
    { title: 'Step 5 Deploy', detail: 'export, FastAPI, Streamlit, model card, screenshots' },
    { title: 'Step 6 Docs', detail: 'PDF report, slides, Q&A prep, executed notebooks' },
    { title: 'Final gate', detail: 'checklist + repro, then 2 consecutive clean adversarial rounds' },
  ],
}

const SCRATCH = 'C:\\Users\\benja\\AppData\\Local\\Temp\\claude\\C--projects-automl\\f98d520c-42c4-49c1-8817-509927a4b249\\scratchpad'

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
PROJECT: C:\\projects\\automl (Windows 11; Git Bash and PowerShell available; venv python: .venv/Scripts/python.exe, Python 3.12.10). Course project (graded): malware detection on EMBER2024 Win64 PE, comparing manual tuning with AutoML.

READ FIRST:
- The brief/rubric is docs/PROJECT_BRIEF.md.
- The approved plan, including the "Definition of done" table D1-D8 and the "Orchestration" section, is at C:\\Users\\benja\\.claude\\plans\\pasted-content-id-fe42-project-brief-warm-sutton.md.
- README.md, config.py, src/*.py and reports/ show the current state.

ESTABLISHED FACTS (from Steps 0-2, independently verified):
- Data:
  - The raw weekly jsonl files repeat each sample twice. The pipeline dedupes by sha256 globally, leaving 520k official-train rows and 119,993 official-test rows.
  - D=2568 thrember features in 12 groups; offsets come from src/data.feature_groups().
  - Splits: train 160k and test_random 40k, both stratified by label x week from weeks 0-51; test_future is 40k from weeks 52-63, strictly later in time. Tuning subset is 60k with 3 persisted folds.
  - Classes are 50/50 by construction.
- detection_ratio ('k/n' string, plus detection_ratio_float in meta) separates the labels perfectly on its own (AUC 1.0). It is in meta ONLY, never in X. It is the planned leaky feature.
- Time proxies already inside X: column 696 (header COFF timestamp) and the authenticode signing-time columns (around 2472-2479).
- The challenge set is in data/raw/challenge: 6,315 malicious rows that AV initially missed, 814 of them Win64 (filter on file_type). It has not been used yet.
- Step 2 results are in reports/tables/tuning_summary.csv and reports/experiments.jsonl, run_id 20261027T201359Z-a9e8a1.
  - Tuned LightGBM gets test_future AUC around 0.998; default LightGBM 0.9967; LogReg 0.930 (gap 0.063).
  - The random-vs-future AUC gap for LightGBM is only about 0.001; TPR@1%FPR drops about 1 point. This is an honest, somewhat surprising finding: report it, don't hide it.
- Dependency notes: signify is pinned to 0.8.1 (0.9.x breaks thrember). torch is the CPU build. H2O 3.46 works with Java 21. AutoGluon 1.6.3, FLAML 2.7.0.
- The full Step 2 tuning took about 4.7 h. Re-running the searches is expensive, so avoid it unless strictly needed.

GROUND RULES:
- SEED=42 everywhere.
- Never select or tune anything on test_random or test_future. They are used only for final scoring.
- Code must be readable, typed, small functions, with short "why" comments. A student team must be able to explain every line.
- All tables and figures are generated from logs or saved artifacts by scripts, never typed by hand.
- Report negative or surprising results honestly. A credible, slightly lower score beats an inflated one.
- Do not git commit or push.
- Never run Python with data/raw as the working directory or on sys.path.
- Long jobs (>2 min): run them in the background with a log file, and wait by checking the log periodically or using Monitor. Never block on one foreground command for more than ~9 minutes.
- CPU-heavy timed runs (AutoML, tuning) must have the machine to themselves.
- Temporary scripts go in ${SCRATCH}, not the repo.
- If you add a dependency, pin it in requirements.txt.
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
    verdict: { type: 'string' },
  },
  required: ['issues', 'verdict'],
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
- Report only issues backed by concrete evidence: a command and its output, a file:line, or a recomputed number.
- Severity:
  - blocker: invalid results, leakage, crashes, a missing required deliverable.
  - major: a number or claim that is wrong or unsupported, a rubric item only partly met, a broken reproduction path.
  - minor: clarity, polish, small inconsistencies.
- A genuine but unflattering experimental result is NOT an issue. Its absence from the write-up would be.`

const ledger = []

// Generic do -> verify -> fix loop. Returns { passed, rounds, unresolved }.
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
    raw.forEach((r, i) => { if (r) r.issues.forEach(x => issues.push({ ...x, lens: lenses[i].key })) })
    const serious = issues.filter(x => x.severity !== 'minor')
    const minors = issues.filter(x => x.severity === 'minor')
    lastSerious = serious
    ledger.push({ stage, round, serious: serious.length, minor: minors.length, verdicts: raw.map(r => r ? r.verdict.slice(0, 400) : 'reviewer failed') })
    log(`${stage} round ${round}: ${serious.length} blocker/major, ${minors.length} minor`)
    if (serious.length === 0) {
      clean++
      if (minors.length && clean === 1) {
        // Fix cheap minor issues once, then keep going if more clean rounds are needed.
        const f = await run(`${CONTEXT}
YOU ARE THE FIXER for stage "${stage}". The reviewers found no blocker or major issues, only these minor ones. Fix each one if it's cheap (<15 min). Otherwise document it in the relevant markdown (README, the stage write-up, or limitations). Never hand-edit numbers. Regenerate outputs with the scripts. Run pytest at the end.
ISSUES (JSON): ${JSON.stringify(minors)}`, { label: `fix-minor:${stage}`, phase: phaseName, schema: FIX_SCHEMA })
        notes = f ? `fixed: ${f.fixed}\ndisputed: ${f.disputed}\ndeferred: ${f.deferred}` : ''
      }
      if (clean >= needClean) return { passed: true, rounds: round, unresolved: [] }
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
    notes = f ? `fixed: ${f.fixed}\ndisputed: ${f.disputed}\ndeferred: ${f.deferred}\ntests: ${f.tests}` : 'fixer failed to return'
  }
  log(`${stage}: hit the ${maxRounds}-round cap with ${lastSerious.length} unresolved serious issues; carrying them to the final gate`)
  return { passed: false, rounds: maxRounds, unresolved: lastSerious }
}

const carried = []

// ---------------- Step 2 fix ----------------
phase('Step 2 fix')
const STEP2_ISSUES = `Known issues from the Step 2 review (verified):
1. MAJOR: the refit of the tuned models (src/tune.py fit_lgbm_with_holdout) uses early stopping with patience 50 on a 10% hold-out. It stopped the grid winner at 187 trees, against a CV mean best_iteration of about 719, so the grid/random/optuna test scores reflect a noisy refit, not the tuner.
   Fix: refit each tuned config on ALL 160k train rows with a FIXED n_estimators derived from the CV best_iterations. Use the mean fold best_iteration, scaled for the larger training size if you can justify it; document the rule. No hold-out. Then train_rows=160000 is true for every model, which fixes minor issue 2 as well.
   Re-run ONLY the refits plus final evaluation (not the searches), append the new final_eval records with a refit_version field, and make tables.py use the latest refit_version.
2. minor: train_rows was logged as 160000 although the fit used 144000 rows (resolved by 1).
3. minor: no uncertainty in tuning_summary.csv. Add a reusable bootstrap helper in src/eval.py (paired, 1000 resamples, seed 42) and add 95% CIs for test AUC and TPR@1%FPR on both test sets, so claims about which tuner wins are statistically honest.
4. minor: trials don't flag hitting MAX_ROUNDS. Add a hit_round_cap column derived from the logged best_iterations (no re-run needed).
5. minor: the comment at src/tune.py ~110 says every grid value lies inside SEARCH_SPACE, but lambda_l2=0.0 does not. Fix the comment.
6. minor: Random and Optuna share their first 10 trials (TPE n_startup_trials=10 with the same seed). Make sure the README and the tuning write-up state this wherever efficiency is compared.
7. minor: the src/data.py docstring says duplicate copies are identical. In fact caps/mbc/ttps differ, while the features and label are identical. Fix the wording.
8. Docs agent notes: nbformat/nbconvert/nbclient aren't installed (needed later to execute notebooks, so install and pin them now). --quick still needs the full cache (document this in the README).
Also write reports/tuning_writeup.md, generated or filled from the tables: methods, the equal-budget definition, results with CIs, timing, the convergence plot, which tuner "won" and whether that is significant, and the honest random-vs-future observation.`
const s2 = await run(`${CONTEXT}
YOUR TASK: fix the Step 2 issues below, re-run what's needed, and update outputs. The machine is yours (no other heavy jobs).
${STEP2_ISSUES}`, { label: 'step2-fix', phase: 'Step 2 fix', schema: DO_SCHEMA })
const g2 = await gate('Step 2 tuning', 'Step 2 fix', 'Steps 0-2: src/data.py, src/tune.py, src/eval.py, src/tables.py, scripts/, tests/, reports/tables/tuning_summary.csv, reports/tuning_writeup.md, reports/experiments.jsonl, models/lgbm_*.', [
  { key: 'method', prompt: 'Tuning methodology and the refit: equal budget really equal; no test-set use in search; the refit now uses all 160k rows with a justified fixed n_estimators; the bootstrap CIs are computed correctly (re-implement one independently and compare); claims in tuning_writeup.md are supported by the CIs.' },
  { key: 'numbers', prompt: 'Numbers and reproducibility: recompute test metrics from the saved models for at least 2 methods and compare with the CSV; check that the table builder picks the latest refit_version records; run pytest; check that the README describes the current procedure.' },
])
if (!g2.passed) carried.push({ stage: 'Step 2', issues: g2.unresolved })

// ---------------- Step 3 AutoML ----------------
phase('Step 3 AutoML')
const s3 = await run(`${CONTEXT}
YOUR TASK: Step 3 of the plan (AutoML) end to end. You own src/automl_runs.py, scripts/run_automl.py, reports/automl_comparison.md and tests/test_automl.py. You may also add automl support to src/tables.py and run_all.py (--steps automl).
- Same inputs for every tool:
  - train (160k x 2568, float32) from src.data.load_split.
  - Feature names "{group}__{i}" built from feature_groups(), so importances can be grouped by family later.
  - Seed 42.
  - Budget AUTOML_BUDGET_S=1200 s per tool.
  - Optimisation metric roc_auc.
  - n_jobs/nthreads 16.
  - Tools run SEQUENTIALLY, with the machine otherwise idle.
- Wall-clock time is measured per tool, including data conversion and preprocessing, and logged separately from the search budget.
- FLAML:
  - AutoML().fit(X, y, task='classification', metric='roc_auc', time_budget=1200, seed=42, n_jobs=16, log_file_name=reports/leaderboards/flaml_search_log.json).
  - Build a leaderboard from the search log: per estimator, the best validation loss, the config and the number of trials.
  - Save the best estimator with joblib, its config and FLAML's eval method.
- AutoGluon:
  - TabularPredictor(label, eval_metric='roc_auc', path=models/autogluon).fit(train_df, time_limit=1200, presets='medium_quality').
  - Save leaderboard(silent=True) using its internal validation only.
  - Save the feature_metadata / features dropped by preprocessing, and which models were trained or skipped.
  - Save feature_importance on a 5k sample of TRAIN or its validation data, never the test sets, with a capped time_limit.
- H2O:
  - h2o.init(max_mem_size='24G', nthreads=16).
  - Pass data as a parquet or CSV file in ${SCRATCH} or models/ rather than a slow pandas upload. The label is a factor.
  - H2OAutoML(max_runtime_secs=1200, seed=42, sort_metric='AUC').
  - Save the leaderboard with extra columns (training_time_ms, algo), the leader's varimp, and the leader as a MOJO or binary model in models/h2o.
  - Note that H2O XGBoost isn't available on Windows.
  - Shut H2O down cleanly.
- After each fit, score test_random and test_future with compute_metrics plus the bootstrap CIs from src/eval.py, and log a final_eval record (tool, best model name, budget, wall_s, metrics) to experiments.jsonl.
- Outputs:
  - reports/leaderboards/{flaml,autogluon,h2o}_leaderboard.csv and per-tool preprocessing notes.
  - reports/tables/automl_vs_manual.csv, generated from the log: tools + manual tuned LightGBM + default LightGBM + LogReg, with AUC/F1/acc/TPR@1%FPR/FPR on both test sets, CIs, wall time and an interpretability column.
  - reports/figures/automl_vs_manual.png.
- reports/automl_comparison.md:
  - What each tool tried, what won, and what preprocessing or feature engineering it applied, taken from the saved artifacts, not assumed.
  - Where the tools were better, worse or just different from manual tuning, and whether differences exceed the CIs.
  - The time comparison: 20 min AutoML vs about 4.7 h of manual search. Be explicit that this isn't apples-to-apples.
  - Interpretability.
- Smoke-test with --quick first, which uses 60 s budgets and the _quick suffix. Add a fast pytest that checks the tables builder on a synthetic log. Then run the full thing in the background and wait. If a tool crashes or runs out of memory, fix it and re-run only that tool, and document what happened.

STATE FROM EARLIER ATTEMPTS (check it on disk, don't trust it blindly):
- Earlier attempts already wrote src/automl_runs.py, src/automl_writeup.py and scripts/run_automl.py. The --quick run finished at 16:17–16:23 on 10-28 (the *_quick leaderboards, tables and automl_comparison_quick.md exist).
- The full run (reports/run_automl.log, run_id 20261028T105459Z) started FLAML at 16:24:59. It stalled when the laptop went to sleep and was then killed. No full-mode results exist. Discard that partial FLAML output (flaml_fit_log.txt, flaml_search_log.json, models/flaml) and rerun FLAML from scratch.
- Keep-awake is now held by the parent session. Still guard against sleep: for each tool, compare the wall-clock duration with the time the process actually ran (for example a heartbeat thread logging every 30 s, or a check for gaps over 60 s in its log timestamps). If a run's wall time includes a gap of more than 60 s, mark it invalid and rerun it, because a timed budget that included sleep is not an equal budget.
- Start the long run so it survives a single shell call ending: a detached process, e.g. PowerShell Start-Process with output redirected to a log file. Then poll the log with short checks (each under 9 minutes) until all three tools and the tables are done.`, { label: 'step3-automl', phase: 'Step 3 AutoML', schema: DO_SCHEMA })
const g3 = await gate('Step 3 AutoML', 'Step 3 AutoML', 'src/automl_runs.py, scripts/run_automl.py, reports/leaderboards/*, reports/tables/automl_vs_manual.csv, reports/figures/automl_vs_manual.png, reports/automl_comparison.md, models/autogluon, models/h2o, the FLAML model, experiments.jsonl automl records.', [
  { key: 'fairness', prompt: 'Fairness and validity: all tools got identical data, budget and seed; no test data was used for model selection, early stopping, leaderboards or feature importance; wall time is logged correctly; the runs did not overlap in time (check timestamps); the saved models reproduce the logged test metrics (reload at least FLAML and one other and recompute).' },
  { key: 'claims', prompt: 'Claims and rubric coverage: every statement in automl_comparison.md (models tried, winners, preprocessing, interpretability, better/worse) is backed by the saved leaderboards and artifacts and by numbers in the CSV; the CIs are used correctly; nothing is overstated; and the brief Step 3 bullets are fully covered.' },
])
if (!g3.passed) carried.push({ stage: 'Step 3', issues: g3.unresolved })

// ---------------- Step 4 Critical evaluation ----------------
phase('Step 4 Critical eval')
const s4leak = await run(`${CONTEXT}
YOUR TASK: Step 4 part A, leakage experiments. CPU-heavy, so you run alone. You own src/leakage.py, scripts/run_leakage.py and their tests.
- Leak 1, detection_ratio_float: append it as an extra column to X for train, test_random and test_future. Re-run:
  - the tuned LightGBM (best Step 2 config, fixed n_estimators),
  - FLAML, AutoGluon and H2O, each with a 600 s budget. This is a stated deviation from 1200 s, made to save time. Also run the matching 600 s clean versions WITHOUT the leak, so the with/without comparison is at an equal budget, unless you can justify reusing the 1200 s Step 3 runs.
- Leak 2, raw first_submission_date as unix seconds: same procedure, at least for LightGBM and FLAML, plus the others if time allows. Show that it helps or doesn't help on the random split and that it fails to transfer to test_future, where the dates are out of range.
- Ablation of the time proxies already in X: remove column 696 (the COFF timestamp) and the authenticode signing-time columns. Identify the exact columns from thrember's source and document them. Retrain the tuned LightGBM and compare random vs future.
- For each run, log metrics on both test sets with CIs. Also log the leaky feature's importance rank, using the tool's own importance where available and SHAP for tree models, and whether the tool warned about, flagged or dropped it. Read the logs to check for warnings.
- Outputs: reports/tables/leakage.csv, reports/figures/leakage_jump.png and reports/leakage.md.
  The write-up says honestly which tools "fell for" the leak (all of them likely will: a feature with AUC 1.0 is perfectly predictive and no tool can know it is post-hoc), why that matters, and what protects against it: a domain review of feature provenance, and temporal validation for the date leak.
- Use --quick first, then the full run in the background.`, { label: 'step4-leakage', phase: 'Step 4 Critical eval', schema: DO_SCHEMA })

const [s4bias, s4explain] = allOk(await parallel([
  () => run(`${CONTEXT}
YOUR TASK: Step 4 part B, temporal, bias and drift. Another agent is computing SHAP at the same time, so don't publish timing claims from this work. You own src/temporal.py, src/bias.py, src/drift.py, scripts/run_eval.py (shared with the explain agent: coordinate by putting your entry points in your own modules and adding a section to run_eval.py; or create scripts/run_eval_bias.py if that's simpler) and their tests. Use the best tuned LightGBM, plus the AutoML winners where cheap, by loading the saved models.
1. Random vs temporal table:
   - Every model from Steps 2-3 on test_random vs test_future: AUC, F1, TPR@1%FPR, FPR, with CIs and the gap → reports/tables/random_vs_future.csv.
   - Compare stratified 3-fold CV against week-blocked forward CV (expanding window over weeks 0-51 of the train split, e.g. 4 folds) for the tuned config. Which estimate is closer to test_future? → reports/tables/cv_comparison.csv.
2. Bias and imbalance:
   - Report the class balance.
   - Per-slice metrics on test_future and test_random, including FPR on benign:
     - week
     - file-size quartile
     - packed vs not packed (with packer names where n is big enough)
     - signed vs unsigned
     - family known vs unknown (malicious only, recall)
     Show n per slice and CIs; mark slices that are too small.
   - Train a class-weighted / scale_pos_weight variant AND, more realistically, evaluate at realistic prevalence. Classes are 50/50 by construction, so compute precision and alarms per 10k files at 1%, 5% and 10% malware prevalence from TPR/FPR (the base-rate effect). Say whether weighting helped; it probably won't at 50/50, so explain why.
   → reports/tables/bias_slices.csv, reports/figures/bias_slices.png, reports/tables/prevalence.csv.
3. Drift monitor: retrain the tuned config on a sliding 12-week window, and also an expanding window, then plot AUC, F1 and TPR@1%FPR per future week 52-63, plus within-train later weeks if useful → reports/figures/drift.png, reports/tables/drift.csv.
4. Challenge set: vectorize the 814 Win64 challenge rows (malicious samples AV initially missed) with the same thrember extractor and data-loading conventions. Store them in data/cache as a new cache file without touching the existing cache. Report each model's detection rate at the deployment threshold and at the 1%-FPR threshold → reports/tables/challenge.csv. This is a strong real-world check: be honest if recall collapses.
5. Write reports/temporal_bias.md with the findings. Use --quick first where it applies.`, { label: 'step4-temporal-bias', phase: 'Step 4 Critical eval', schema: DO_SCHEMA }),
  () => run(`${CONTEXT}
YOUR TASK: Step 4 part C, explainability, calibration and cost. Another agent is running drift and bias jobs at the same time. You own src/explain.py, src/calibration.py, scripts/run_explain.py and their tests.
1. SHAP:
   - TreeExplainer on the best tuned LightGBM over a 5k stratified sample of test_future. Explaining is not tuning, so using test data is fine.
   - Produce a beeswarm (top 20 features with readable names: map indices to thrember feature names where possible, e.g. group plus position, or the actual names from the extractor), a bar chart of mean |SHAP| grouped by the 12 EMBER feature families, and 3 local waterfall plots: a confidently correct malicious, a false positive and a false negative, with an explanation of each.
   - Comment on whether the time-proxy columns (696, signing time) rank high.
   → reports/figures/shap_*.png, reports/tables/shap_family_importance.csv.
2. Cross-tool agreement:
   - Family-level importance from the manual LightGBM SHAP, FLAML's winner (SHAP if it's a tree model, else its importance), AutoGluon's feature_importance and H2O's varimp, all aggregated to the 12 families.
   - Report the top-k overlap (k=5 and 20 features) and Spearman rank correlation at family level → reports/tables/cross_tool_agreement.csv and a figure.
3. Calibration: reliability curve and Brier score on test_future for the tuned LightGBM and the AutoML winners. Optionally add isotonic or Platt calibration fitted on a TRAIN hold-out, never the test set → reports/figures/calibration.png, reports/tables/calibration.csv.
4. Cost-sensitive threshold:
   - Choose the threshold on a validation portion of TRAIN (out-of-fold CV predictions are best), never the test set, minimising C_FN*FN + C_FP*FP with a stated default C_FN/C_FP=10.
   - Also show a sensitivity curve over ratios of 1, 5, 10, 50 and the precision-recall curve.
   - Report the chosen threshold's performance on test_future.
   - Save it to models/threshold.json; deployment will use it.
   → reports/figures/pr_cost.png, reports/tables/threshold.csv.
5. Write reports/explainability.md. Use --quick first.`, { label: 'step4-explain', phase: 'Step 4 Critical eval', schema: DO_SCHEMA }),
]), "Step 4 parallel work")
const s4write = await run(`${CONTEXT}
YOUR TASK: synthesise Step 4. Read reports/leakage.md, temporal_bias.md, explainability.md and all their tables and figures. Write:
- reports/critical_eval.md: answer the research question directly, i.e. how much random-split scores overstate real-world performance (use test_future, prevalence-adjusted precision and the challenge set) and whether AutoML tools protect against leakage and drift or fall for them. Cite numbers by pointing to the CSVs.
- reports/limitations.md: static features only, packed/obfuscated files, drift, dataset age and recency, 50/50 balance vs real prevalence, subsampling, label noise from AV-derived labels, adversarial evasion, the dedupe issue, and the Windows/CPU-only constraints.
Wire scripts/run_leakage.py, run_eval*.py and run_explain.py into run_all.py as --steps eval, with --quick working. Run pytest.`, { label: 'step4-synthesis', phase: 'Step 4 Critical eval', schema: DO_SCHEMA })
const g4 = await gate('Step 4 critical evaluation', 'Step 4 Critical eval', 'src/leakage.py, src/temporal.py, src/bias.py, src/drift.py, src/explain.py, src/calibration.py, the scripts/run_* for eval, reports/tables/{leakage,random_vs_future,cv_comparison,bias_slices,prevalence,drift,challenge,shap_family_importance,cross_tool_agreement,calibration,threshold}.csv, reports/figures/*, reports/{leakage,temporal_bias,explainability,critical_eval,limitations}.md, models/threshold.json.', [
  { key: 'leakage', prompt: 'Leakage experiment validity: the leaky columns really are added and removed; with/without budgets match; the importance ranks are real (check the artifacts); the claims about which tools fell for it are evidenced; the threshold and calibration were NOT chosen on test data; the challenge-set vectorisation matches the main pipeline (re-vectorise a few rows).' },
  { key: 'stats', prompt: 'Statistical correctness: recompute at least 3 numbers independently (one bias-slice FPR, one prevalence-adjusted precision, one drift-week AUC, plus a Brier score) and compare; check that slice n and CIs are shown and tiny slices are flagged; check that forward CV is truly forward in time; check that SHAP family aggregation uses the correct offsets.' },
  { key: 'honesty', prompt: 'Honesty and rubric coverage: critical_eval.md answers the research question with evidence, doesn\'t overclaim, reports negative and surprising results, and covers every Step 4 brief bullet plus the stretch goals (drift, cost threshold, calibration, cross-tool agreement). limitations.md is complete and specific.' },
])
if (!g4.passed) carried.push({ stage: 'Step 4', issues: g4.unresolved })

// ---------------- Step 5 Deployment ----------------
phase('Step 5 Deploy')
const s5 = await run(`${CONTEXT}
YOUR TASK: Step 5, deployment. You own app/, models/final/, MODEL_CARD.md, DEMO.md, tests/test_api.py and scripts/export_model.py.
1. Choose the deployment model by test_future TPR@1%FPR (with CIs), interpretability and inference simplicity. This is likely the tuned LightGBM. Justify the choice in MODEL_CARD.md.
   scripts/export_model.py writes models/final/model.txt (or .joblib) and schema.json with: feature_dim 2568, feature names and group offsets, the thrember commit, the signify pin, the threshold from models/threshold.json, the training data description, the metrics source CSV, and a sha256 of the model file. Inference must be self-contained. The model needs no scaler; state that.
2. app/api.py (FastAPI):
   - GET /health returns the model version and hash.
   - POST /predict takes {features: list[float]}. Pydantic validates length == 2568 and that all values are finite (422 otherwise), with a batch variant optional. It returns {p_malicious, label, threshold, model_version}.
   - POST /predict_file accepts an uploaded PE (size-capped, e.g. 50 MB), runs thrember PEFeatureExtractor on the bytes in memory and predicts. Handle parse errors with a 400. Never write the upload to disk or execute it.
3. app/streamlit_app.py:
   - Choose a sample from test_future by index or filter (benign or malicious, correct or misclassified), or upload a file.
   - Call the API (configurable URL) and show p_malicious, the threshold, the label, the true label for test samples, and the top-10 SHAP contributions for that sample grouped by family.
   - Show a banner with the model card essentials and limits.
4. tests/test_api.py with FastAPI TestClient:
   - health
   - a valid vector gives a probability equal to offline booster.predict to 1e-6
   - wrong length gives 422
   - NaN/inf gives 422
   - a non-numeric value gives 422
   - /predict_file on a benign system binary (C:\\Windows\\System32\\notepad.exe, read-only) gives 200, and random bytes give 400
5. Start uvicorn and streamlit for real in the background. Exercise them with curl. Take screenshots of the Streamlit app showing (a) a correct malicious prediction, (b) a benign prediction and (c) a file upload of notepad.exe. Use the built-in browser tools if available via ToolSearch (mcp__Claude_Browser__*), otherwise Playwright (pip install playwright, then playwright install chromium; pin it). Save them to reports/screenshots/. Stop the servers when done.
6. MODEL_CARD.md: model details, intended use (triage aid for static PE analysis, research), out-of-scope uses (sole verdict for blocking, non-Win64 files, adversarial settings), training data, metrics on both test sets plus the challenge set and prevalence-adjusted precision (cite the CSVs), threshold and cost ratio, calibration, known limits, ethical considerations, how to reproduce.
7. DEMO.md: a step-by-step 5-minute demo script with exact commands, plus how to deploy to Hugging Face Spaces or Streamlit Cloud (the user has to do that with their account).
8. Add --steps deploy to run_all.py (export + tests).`, { label: 'step5-deploy', phase: 'Step 5 Deploy', schema: DO_SCHEMA })
const g5 = await gate('Step 5 deployment', 'Step 5 Deploy', 'app/api.py, app/streamlit_app.py, models/final/, scripts/export_model.py, tests/test_api.py, MODEL_CARD.md, DEMO.md, reports/screenshots/.', [
  { key: 'api', prompt: 'Adversarial API testing: actually start uvicorn yourself (a different port, background, stop it afterwards). Send malformed inputs (wrong length, NaN, strings, huge payloads, an empty upload, a non-PE upload), check the status codes, and check that /predict parity with offline scoring holds on 20 real test_future rows. Confirm uploads are never written to disk or executed. Run tests/test_api.py. Also start the Streamlit app headless and confirm it loads without errors.' },
  { key: 'card', prompt: 'Model card and demo truthfulness: every number in MODEL_CARD.md matches the generated CSVs; the intended and out-of-scope uses and limits match the findings in critical_eval.md and limitations.md; DEMO.md commands work as written (run them); the screenshots exist and show what DEMO.md claims (open the PNGs).' },
])
if (!g5.passed) carried.push({ stage: 'Step 5', issues: g5.unresolved })

// ---------------- Step 6 Docs ----------------
phase('Step 6 Docs')
const [s6report, s6slides, s6nb] = allOk(await parallel([
  () => run(`${CONTEXT}
YOUR TASK: the PDF report. You own scripts/build_report.py, reports/report.pdf, reports/figures/architecture.png and reports/report_src/ if you need it.
- Build the PDF programmatically: reportlab (pin it) or another Windows-friendly library with no system dependencies. The numbers, tables and figures must be pulled from reports/ CSVs and PNGs at build time, never typed by hand.
- Draw an architecture/workflow diagram (data → dedupe/vectorise → splits → tuning / AutoML → critical eval → export → FastAPI/Streamlit) with matplotlib, saved as architecture.png.
- Sections:
  - title page (student name and ID from the brief, course, team TODO)
  - abstract
  - problem statement and research question
  - dataset description (version, license, size, dedupe finding, splits, class balance, feature families)
  - methodology and workflow diagram
  - Step 2 tuning results
  - Step 3 AutoML comparison, with leaderboards summarised
  - Step 4 critical evaluation (random vs temporal, leakage, bias/prevalence, SHAP, calibration, cost threshold, drift, challenge set)
  - deployment with screenshots
  - reflections: what AutoML did well and where it failed
  - limitations
  - reproducibility
  - references (EMBER2024 paper arXiv:2506.05074, EMBER 2018, the tools)
  - appendix
- About 15-25 pages. Readable fonts, captioned figures, tables that fit the page.
- After building, render every page to PNG with pymupdf (pin it) into ${SCRATCH}/report_pages, and LOOK at each page image (Read the PNGs). Fix overflow, cut-off tables, missing figures and blank pages, and repeat until it is clean.`, { label: 'step6-report', phase: 'Step 6 Docs', schema: DO_SCHEMA }),
  () => run(`${CONTEXT}
YOUR TASK: slides and Q&A prep. You own scripts/build_slides.py, reports/slides.pptx and reports/qa_prep.md.
- Build a deck for a 5-minute demo with python-pptx (pin it): 8-10 slides, generated from the reports/ CSVs and PNGs.
  Slides: title; problem and why it matters; data and the duplicate finding; method and pipeline diagram (reports/figures/architecture.png if present, else draw your own); tuning results; AutoML vs manual; the research-question answer (random vs future, prevalence, challenge set); leakage (did AutoML fall for it); SHAP; deployment demo with a screenshot; limitations and takeaways.
- Speaker notes on every slide, about 30 s each, totalling about 5 min.
- Consistent design, readable font sizes (at least 18 pt body), no overflowing text.
- Render each slide to an image and check it. LibreOffice may not be installed; if so, check with python-pptx shape bounds plus a matplotlib mock render, or by converting through PowerPoint COM if available. Fix any overflow.
- reports/qa_prep.md: 15-20 likely instructor questions with crisp answers and pointers to evidence, e.g. why EMBER2024, why LightGBM, what equal budget means, why the drift gap is small, why the AutoML tools fell for the leak, how to deploy in production, adversarial evasion. Include questions any team member must be able to answer about the code.`, { label: 'step6-slides', phase: 'Step 6 Docs', schema: DO_SCHEMA }),
  () => run(`${CONTEXT}
YOUR TASK: notebooks. You own notebooks/*.ipynb.
- Update 01_eda and 02_baseline_tuning, and create 03_automl and 04_critical_eval. They are thin narrative wrappers that import from src and read reports/ outputs, with no duplicated logic and markdown cells explaining each step for a student audience.
- Execute all four with nbconvert/nbclient using the .venv kernel (register it with ipykernel if needed), so the saved notebooks contain outputs. Cells must not retrain expensive models; load the saved artifacts.
- Make sure every notebook executes without errors from a fresh kernel.`, { label: 'step6-notebooks', phase: 'Step 6 Docs', schema: DO_SCHEMA }),
]), "Step 6 parallel work")
const g6 = await gate('Step 6 documentation', 'Step 6 Docs', 'reports/report.pdf, scripts/build_report.py, reports/slides.pptx, scripts/build_slides.py, reports/qa_prep.md, notebooks/*.ipynb, README.md.', [
  { key: 'numbers', prompt: 'Number traceability: extract the text of report.pdf (pymupdf) and slides.pptx (python-pptx), pick at least 15 numbers across both, trace each one to a reports/ CSV, and flag any mismatch, stale value or untraceable number.' },
  { key: 'rubric', prompt: 'Rubric coverage: check report.pdf against brief Step 6 (problem statement, dataset description, workflow diagram, experiment tables, screenshots, tool comparison, leakage and bias findings, reflections) and the deck against "5-minute demo plus Q&A". Check that the notebooks execute (run nbconvert --execute into ' + SCRATCH + ' and check for errors).' },
  { key: 'render', prompt: 'Visual quality: render every PDF page and every slide to PNG in ' + SCRATCH + ' and LOOK at them. Flag cut-off tables, overflowing text, unreadable fonts, missing or broken images, blank pages, and figures without captions.' },
])
if (!g6.passed) carried.push({ stage: 'Step 6', issues: g6.unresolved })

// ---------------- Final gate ----------------
phase('Final gate')
const pre = await run(`${CONTEXT}
YOUR TASK: prepare for the final submission gate.
1. reports/repro_check.md:
   - Create a FRESH venv in ${SCRATCH}/fresh_venv with py -3.12, pip install -r requirements.txt, and record whether it succeeds, how long it takes and any issues. Fix requirements.txt if it fails.
   - Delete all *_quick outputs (logs, splits, tables, figures, models) and run "python run_all.py --quick" end to end with the project venv in the background, recording the time and the result.
   - Run the full pytest suite and record the result. There should be no skips now that the data exists; investigate any skip.
2. reports/SUBMISSION_CHECKLIST.md: map EVERY rubric bullet in docs/PROJECT_BRIEF.md (Steps 1-6, the deliverables checklist, the ground rules, and each stretch goal) and every plan D1-D8 item to concrete evidence files, with a status. List explicitly what only the user can do: the Part A certificate, team/deadline/topic TODOs, recording the demo video, the cloud deploy, and git commit/push to GitHub.
3. Update README.md: status table all done, links to every deliverable, and how to reproduce.
Carried unresolved issues from earlier gates (must be resolved or explicitly documented): ${JSON.stringify(carried)}`, { label: 'final-prep', phase: 'Final gate', schema: DO_SCHEMA })
const gF = await gate('FINAL submission readiness', 'Final gate', 'The whole repository and every deliverable. Use docs/PROJECT_BRIEF.md and the plan Definition of done D1-D8 as the acceptance criteria.', [
  { key: 'repro', prompt: 'Reproducibility (actually run things): pytest (all pass, no unexplained skips), run_all.py --quick end to end, and spot-reproduce one Step 2 and one Step 4 number from saved artifacts. Check that requirements.txt pins everything imported in src/ app/ scripts/ (grep the imports). Check that the README instructions work as written.' },
  { key: 'completeness', prompt: 'Completeness critic: go through SUBMISSION_CHECKLIST.md and verify each claimed evidence file exists and actually contains what is claimed. Find any brief rubric item, deliverable or D1-D8 criterion that is missing, partial or only claimed. Check that the user-only items are clearly listed.' },
  { key: 'honesty', prompt: 'Numbers and honesty: cross-check headline numbers across README, report.pdf, slides.pptx, MODEL_CARD.md, critical_eval.md and the CSVs. They must be consistent and current. Flag overclaims, hidden negative results, any test-set use for selection, and inconsistencies between documents.' },
  { key: 'readability', prompt: 'Code readability and explainability for a student team: every module has a docstring; non-obvious choices carry "why" comments; functions are small and typed; there is no dead code, leftover debug code or synthetic-test artefacts in reports/ or models/; naming is consistent. Flag files a team member would struggle to explain.' },
], { maxRounds: 6, consecutiveClean: 2 })

return { step2: s2 && s2.results, g2, s3: s3 && s3.results, g3, s4: { leak: s4leak && s4leak.observations, bias: s4bias && s4bias.observations, explain: s4explain && s4explain.observations, synth: s4write && s4write.observations }, g4, s5: s5 && s5.observations, g5, s6: { report: s6report && s6report.problems, slides: s6slides && s6slides.problems, nb: s6nb && s6nb.problems }, g6, pre: pre && pre.results, gF, carried, ledger }
