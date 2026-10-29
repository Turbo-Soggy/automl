# Q&A preparation: likely instructor questions

Short answers for the viva after the 5-minute demo, with a pointer to the evidence for each.
The numbers are copied from the named tables (they are the same ones the deck reads at build
time with `scripts/build_slides.py`). If a table is regenerated, check the numbers against it.
Every team member should be able to answer questions 17 to 20 about the code.

## Data and design choices

**1. Why EMBER2024 and not EMBER2018 or raw binaries?**
EMBER2024 has a real `first_submission_date` per file, so our "future" test set is truly later
in time, not simulated. It also has a real `detection_ratio` field, which gave us an honest leak
for the leakage experiment. Its feature extractor (`thrember`) is maintained, and we pinned it
to one commit. Raw binaries would mean handling live malware, which is unsafe and not needed for
static features.
*Evidence:* `README.md`, "Design decisions and why"; `reports/data_check.json` (`thrember_commit`).

**2. What was the duplicate finding, and why does it matter?**
Every weekly jsonl file lists each sample twice. The raw files hold 1,040,000 + 240,000 lines but
only 520,000 + 119,993 unique files (7 more duplicates across test files). We deduplicate
globally by sha256 before any split. Without that, a random split would put copies of the same
file in train and test and make test_random look even better.
*Evidence:* `reports/data_check.json` (`duplicates_in_file`, `rows`); `README.md`, "Data quirks";
`reports/limitations.md`, "The duplicate rows in the raw data".

**3. Why the Win64 slice and a 200k subsample (160k train, 40k + 40k test)?**
One file type gives one consistent feature distribution. 200k files from weeks 0-51,
stratified by label x week, is large enough for stable AUC and TPR@1%FPR estimates and still
lets each AutoML tool try several models in 20 minutes. test_future is a 40k stratified sample of
weeks 52-63. Classes are 50/50 by construction.
*Evidence:* `README.md`, "Win64 slice, 200k subsample"; `reports/tables/class_balance.csv`.

**4. Why LightGBM as the manual model?**
It is the standard EMBER baseline, it is fast on 2,568 dense features, it needs no scaling or
other preprocessing, and SHAP's TreeExplainer gives exact explanations for it. Using one family
keeps Step 2 about tuning methods, not model families. Logistic regression is only a sanity
floor.
*Evidence:* `README.md`, "LightGBM as the single manual model family".

## Method

**5. What does "equal budget" mean?**
Two different things. In Step 2 every tuner (grid, random, Optuna) gets exactly 36
configurations, each scored on the same 3 saved CV folds of the same 60,000-row subset, with the
same 2000-round cap and early stopping. That is equal trials, not equal time, because a trial
count is reproducible and a time cap depends on machine load. In Step 3 every AutoML tool gets
the same train split, metric (ROC-AUC), 16 threads and a 1200 s search budget. The budget caps
the search, not the total run: FLAML's final refit and H2O's import and scoring come on top, so
wall times differ (25.7 / 20.2 / 24.8 min).
*Evidence:* `reports/tuning_writeup.md`, "Equal budget"; `reports/automl_comparison.md`,
"Setup"; `reports/tables/automl_vs_manual.csv` (`wall_seconds`, `over_budget_seconds`).

**6. How do you know the test sets never influenced a choice?**
Tuners only see CV folds inside train, and the tuned model is picked by CV AUC (Optuna, 0.9978).
AutoML tools validate inside train, and neither test set is loaded before a fit ends. Both
operating thresholds (1% FPR and the cost-based one) are set on train out-of-fold scores. Test
sets are scored once at the end.
*Evidence:* `reports/critical_eval.md`, "Test-set discipline"; `models/threshold.json`
(`how_chosen`); `src/tune.py`, `data/splits/`.

**7. Random and Optuna share their first 10 trials. Doesn't that bias the comparison?**
It is deliberate: Optuna's TPE starts with 10 seeded random trials drawn with the same seed as
Random Search, so they differ only in the remaining 26 trials. We state it, and the convergence
figure shades the shared trials. The three tuners end within 0.00015 test_future AUC of each
other anyway.
*Evidence:* `README.md`, "Random and Optuna share their first 10 trials";
`reports/figures/tuning_convergence.png`; `reports/tables/tuning_summary.csv`.

## Results

**8. Is AutoML better than manual tuning?**
Not on accuracy, but much cheaper. On test_future AUC, AutoGluon is statistically tied with the
tuned LightGBM (-0.00006, CI includes 0). FLAML (-0.00053) and H2O (-0.00110) are slightly
worse, and all three beat untuned LightGBM. AutoGluon needed 20.2 wall minutes against 76.0 for
the selected manual search (283.5 for all three searches). Every winner is in effect one tree
model, which is why the scores are so close.
*Evidence:* `reports/tables/automl_pairwise.csv`, `automl_vs_manual.csv`;
`reports/automl_comparison.md`, "Verdict".

**9. Why did H2O come last?**
Its default (no cross-validation) split train into training, validation and blending frames, so
its models trained on 112,141 rows (70% of train) with no refit on all rows. Its "stacked
ensemble" put all weight on one GBM. Training on fewer rows is a plausible part of the gap; we
did not test it separately.
*Evidence:* `reports/automl_comparison.md`, "H2O AutoML" and "Verdict".

**10. The random-vs-future AUC gap is only about 0.001. Is that a bug?**
No. We checked three things. (a) The inputs do drift: a classifier tells train files from future
files with AUC 0.9383 (control 0.4973). But trees cannot extrapolate, so a value beyond the last
split lands in the same leaf as the newest training values. (b) Removing the time proxies (COFF
timestamp, signing-time columns) does not shrink the gap (0.00101 -> 0.00107), so they are not a
hidden time leak. (c) There is no label shift (50/50 in every split), and the horizon is short:
12 weeks, with a weekly AUC slope of -0.00011. Logistic regression, which cannot cap
out-of-range values, loses 0.0632 AUC, so the test set is not too easy for every model.
*Evidence:* `reports/critical_eval.md`, section 3; `reports/tables/gap_adversarial.csv`,
`leakage.csv` (`ablate_time_proxies`), `drift.csv`, `random_vs_future.csv`.

**11. So does a random split overstate real-world performance or not?**
Yes, but not where AUC looks. At a threshold fixed on train, the FPR rises from 0.95% on
test_random to 1.23% on test_future. Precision is 99.0% on the 50/50 test mix but 51.2% at 1%
malware prevalence, and 44.7% on future files (121 false alarms per 10,000 files). On the 814
challenge-set files (malware the AV engines first missed) the model catches only 53.7%. The
50/50 mix and the hard tail matter more than the random-vs-future step.
*Evidence:* `reports/critical_eval.md`, sections 1-2; `reports/tables/precision_both_sets.csv`,
`challenge.csv`.

**12. Why did every AutoML tool fall for the leak?**
`detection_ratio` is the share of AV engines that flagged the file, which is what the label is
made from. It separates the classes almost perfectly, and an AutoML search is built to find the
column that raises the validation score. Each tool validates on a random split of the same weeks,
where the leak is also present, so its own validation AUC was 1.0000 and nothing looked wrong.
No tool dropped it and no log line named it. Only a person asking "is this value known when the
file first arrives?" catches it. We report this as a finding, not a bug to fix.
*Evidence:* `reports/leakage.md`; `reports/critical_eval.md`, section 4;
`reports/tables/leakage.csv` (`leak_rank_*`, `leak_dropped_by_tool`), `automl_self_estimate.csv`;
`reports/leaderboards/leakage/log_scan.json`.

**13. Why did `first_submission_date` (the time leak) do almost nothing?**
Every future date lies after the last training date, so for a tree the column is a constant on
test_future. Our classes are 50/50 in every week, so the date carries almost no label
information. It moved test_future AUC by at most 0.0009, which is within the run-to-run noise
of a time-budgeted AutoML search. In data where the malware share changes over time, the same
column would let a model learn the calendar. This is not the result the plan expected, and we
report it as it came out.
*Evidence:* `reports/critical_eval.md`, section 4; `reports/tables/leakage.csv`.

**14. What does SHAP show, and can we trust the AutoML models' explanations?**
No single family dominates: strings, imports and histogram have the largest SHAP shares. The
strongest single features are `header: coff.char.DLL`, `authenticode: latest_signing_time` and
`datadirectories: EXPORT.size`. Each AutoML winner is in effect one tree model, so it can be
explained the same way. H2O is the awkward one: it must be explained inside a running H2O
cluster, because no MOJO could be exported.
*Evidence:* `reports/explainability.md`; `reports/tables/shap_family_importance.csv`,
`shap_feature_importance.csv`, `cross_tool_agreement.csv`; `reports/automl_comparison.md`,
"Interpretability".

## Hardware, deployment and robustness

**15. Steps 2-3 ran on the laptop and Step 4 on a cloud VM. Does that break the comparison?**
No comparison crosses machines. Steps 2-3 ran on the 16-core laptop (63 GB). Step 4 ran on a
4-core / 15 GB Linux cloud VM, where every with/without-leak pair ran with the same settings
(600 s per AutoML run, a lean FLAML preprocessing, a smaller H2O heap without XGBoost). What
changes is the absolute AutoML score: fewer cores search fewer models, so H2O's clean
test_future AUC was 0.9922 on the VM against 0.9970 in Step 3. That is why Step 4 reports
deltas within the VM, never Step 3 vs Step 4. The challenge-set scoring was re-run on the laptop,
but scoring fits nothing, so the hardware does not change it.
*Evidence:* `reports/limitations.md`, "Compute, hardware and budgets";
`reports/tables/automl_self_estimate.csv`; `docs/handoff/HANDOFF.md`, section 0.

**16. Your deployment threshold is 0.0069 with a 7.4% FPR. Why so low?**
We chose it on train out-of-fold scores to minimise cost when a missed malware costs 10 times a
false alarm. That favours recall: 99.5% of test_future malware is caught, and 79.9% of the
challenge set. The price is false alarms. That is acceptable for a triage aid that sends files to
an analyst, not for automatic blocking. With a different cost ratio or a real prevalence, the
threshold must be derived again (`models/threshold.json` says so).
*Evidence:* `models/threshold.json`; `reports/tables/threshold.csv`, `challenge.csv`;
`reports/figures/pr_cost.png`.

**17. How would you deploy this in production?**
The model ships as `models/final/model.txt` with `schema.json` (feature version, thrember commit,
threshold). It is served by a FastAPI service (`/health`, `/predict`, `/predict_batch`,
`/predict_file`) with a Streamlit front end. For real use: (a) re-derive the threshold for the
site's real prevalence and cost; (b) monitor the score distribution and FPR per week, since
inputs drift (adversarial AUC 0.94); (c) retrain on recent weeks on a schedule (our drift study
found more data beat recency over 12 weeks, so retrain on all data plus new weeks); (d) use it
as one signal next to dynamic analysis and AV, never as the only verdict; (e) keep the API
behind authentication and a size limit (it refuses uploads over 50 MB). Cloud hosting is left to
the user's own account.
*Evidence:* `app/api.py`, `app/streamlit_app.py`, `MODEL_CARD.md`; `reports/tables/drift.csv`,
`gap_adversarial.csv`.

**18. What about adversarial evasion?**
We did not test it, and we say so. The model uses static features only, which an attacker can
change without changing behaviour: append bytes, add imports or sections, or sign the file. The
model leans on signature columns: signed malware on test_future is caught 83.7% of the time at
the 1%-FPR threshold, against 98.6% for unsigned malware. Packed benign software cannot be
checked either, because packer tags exist only for malicious files. Defences would be
adversarial training, features that are harder to fake, and dynamic analysis.
*Evidence:* `reports/limitations.md`, "Adversarial evasion was not tested" and "Packed and
obfuscated files"; `reports/tables/bias_slices.csv` (slice `signed`).

## Questions about the code (every team member)

**19. How do I reproduce everything, and how do I rebuild a table without retraining?**
`.venv/Scripts/python.exe run_all.py` runs data, tune, automl, eval and deploy in order;
`--steps tune,automl` picks steps and `--quick` writes small `*_quick` files that never
overwrite real results. Each step script rebuilds its tables, figures and write-up from the
saved log without training: `scripts/run_tuning.py --tables-only`, `run_automl.py
--tables-only`, `run_leakage.py --tables-only`, `run_eval_bias.py --stages tables,writeup`. The
deck is `scripts/build_slides.py`. All results are appended to `reports/experiments.jsonl`, and
tables are built from it, never typed.
*Evidence:* `README.md`, "Reproducing"; `run_all.py` docstring; `tests/` (`pytest -q`).

**20. Where are the splits made, and how does every step use the same rows and folds?**
`src/data.py`: `build_cache` streams the weekly jsonl, drops repeated sha256 and vectorizes with
thrember; `make_splits` draws the label x week stratified samples with seed 42 and saves the row
indices and the 3-fold CV fold ids to `data/splits/`. Every step then calls `load_split(name)`
and `get_cv_folds()`, so no step can draw its own split. `config.SEED = 42` is passed to every
sampler and model, because libraries ignore the global seed.
*Evidence:* `src/data.py` (`make_splits`, `load_split`, `get_cv_folds`), `config.py`,
`tests/test_splits.py`.

**21. How are the confidence intervals computed?**
95% paired bootstrap over the test rows (1000 resamples, seed 42 for test_random and 43 for
test_future, so the two sets are resampled independently). "Paired" means every model is scored
on the same resampled rows, so a difference between two models gets its own CI. Rates on small
slices use Wilson intervals. The CIs cover test-set sampling only, not training randomness
(one seed).
*Evidence:* `src/tables.py` (`BOOTSTRAP_SEEDS`, `bootstrap_samples`, `pairwise_table`);
`README.md`, "Confidence intervals".

**22. What happens when someone uploads a file to the API? Is it safe?**
The file is read into memory (refused above 50 MB), parsed with thrember's static extractor into
the 2,568-feature vector, scored by the LightGBM booster, and the top SHAP contributions are
returned. The bytes are only parsed in memory, never saved or run. A file that `pefile` cannot
parse gets an error instead of a score, and a PE that is not AMD64 is scored with a warning that
it is out of distribution.
*Evidence:* `app/api.py` (`read_upload`, `extract_features`, `score_file`, `MAX_UPLOAD_BYTES`);
`tests/test_api.py`.
