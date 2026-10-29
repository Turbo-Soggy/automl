# Project Brief: Malware Detection on EMBER with Manual Tuning vs AutoML

This is the working spec, copied verbatim from the student's brief. Treat every rubric item as a floor, not a ceiling: do the required work, then go further where noted under "Beyond the rubric".

## 1. Context
- Course task: AutoML Skill-Based Project, Evaluation IA-3 (40 marks). Part A (online certification, 10 marks) is out of scope; it is handled separately. This brief covers Part B (30 marks) only.
- Student: Isaac Benjamin David (URK23AI1035), final-year B.Tech AI & Data Science, Karunya Institute of Technology and Sciences.
- Team: 3 to 4 members expected. Team size: `TODO`. Deadline: `TODO`. Every member must be able to explain any part of the code, so keep the code readable and well commented.
- Topic approval: The instructor must approve the topic, and it must not repeat a teammate's topic. `TODO: confirm approval`.
- Why this topic: The dataset should be unique and ready-made, with no time spent on data collection, and the generic suggestions (loan default, student performance, etc.) are deliberately avoided. Malware detection suits the security background (internship on an Offensive Security team, work on static analysis and multi-agent security pipelines) and gives a natural story about leakage, drift and bias.

## 2. Problem statement
Classify Windows PE files as malicious or benign from pre-extracted static features, then answer a research question:
How much do random-split scores overstate real-world performance for malware detection, and do AutoML tools protect against, or fall for, leakage and temporal drift?

## 3. Dataset
- EMBER (Endgame Malware BEnchmark for Research), a public benchmark of static PE features (roughly 1M samples, about 2,400 numeric features per sample, with benign/malicious labels, and a time-ordered train/test structure).
- Check the exact version, size, license and download method before starting (EMBER 2018 v2 vs a newer release). Document what was actually used.
- Drop unlabeled rows (label = -1) if present.
- Use the `ember` Python package or the released `.jsonl`/vectorised files. Pay attention to the date/appearance metadata, which is needed for the temporal split.
- Subsample for tuning and AutoML: a stratified 150k to 250k rows (the task requires only "a few thousand rows" minimum). Keep the same subsample for every method so comparisons are fair. Record the subsample size and seed.

## 4. Environment and constraints
- Hardware: ASUS ROG G16, Intel Core Ultra 9 285H, NVIDIA RTX 5070 Ti Mobile (GPU), Windows 11. Compute is not the bottleneck, but RAM during feature loading may be. Load in chunks or use float32 and avoid needless copies.
- Python: a virtual environment with a pinned `requirements.txt`.
- AutoML frameworks: FLAML and AutoGluon are the primary choices, since both run natively on Windows. H2O AutoML is an option if Java is available. Auto-sklearn is Linux-only (use WSL2 only if wanted).
- Seed: a single global `SEED = 42`, applied to numpy, random, the model libraries and every split. Results must be reproducible.
- Time budget: every AutoML tool gets the same fixed budget (default 20 minutes, within the rubric's 15 to 30). Log wall-clock time for every method.

## 5. Work plan (maps to the rubric)

### Step 1: Problem and dataset (approval)
Document the problem, dataset source, class balance, feature groups (byte histogram, byte-entropy histogram, strings, general file info, header, section, imports, exports, data directories) and the subsample.

### Step 2: Baseline and hyperparameter tuning (5 marks)
- Clean data and split properly. Use a stratified random split (held-out test set never touched during tuning) and also build the time-based split (see Step 4).
- Manual baseline with scikit-learn or LightGBM/HistGradientBoosting. Use one model family for all three tuning methods.
- Tune the same model with Grid Search, Random Search and Bayesian optimisation (Optuna).
- Record score (F1 and ROC-AUC, plus accuracy) and time taken for each method. Produce a comparison table and chart.
- Use the same CV folds and the same search budget framing across methods, and say what "equal budget" means.

### Step 3: AutoML and tool comparison (10 marks)
- Run at least two frameworks (FLAML and AutoGluon) on the same data with the same time budget.
- Study each leaderboard: which models were tried, which won, and what preprocessing and feature engineering the tool applied. Save the leaderboards as artifacts.
- Compare AutoML vs the manual baseline on F1/accuracy, training time and interpretability. Include an honest comment on where the tools were better, worse, or merely different.

### Step 4: Critical evaluation (5 marks)
- Leakage and overfitting: cross-validation plus a held-out test set. Compare random split vs chronological split and quantify the gap.
- Deliberate leaky feature: add one realistic leaky feature (for example a proxy of the label such as a detection-count-like column, or a time/first-seen proxy). Show whether each AutoML tool exploits it (score jump, feature importance, SHAP). Remove it and show the drop.
- Bias and imbalance: report class balance, per-slice performance (by year, file size bucket, or other available groupings) and false-positive rate on benign files. Apply resampling or class weights if useful and say whether it helped.
- Explainability: SHAP (TreeExplainer on the best tree model; summary plot plus a few local explanations). Group importances by EMBER feature family so the explanation is readable.
- Limitations: write down what was found (static features only, packed/obfuscated files, drift, dataset age, subsampling, adversarial evasion).

### Step 5: Deployment (5 marks)
- Export the best model (`joblib` or the framework's native format) with its preprocessing so inference is self-contained.
- Serve with FastAPI (a `/predict` endpoint accepting a feature vector, with input validation) and a Streamlit front end for the demo. Local deployment is acceptable, and a cloud deployment is a bonus.
- Working prediction demo, plus a model card covering data used, metrics, intended use, known limits and out-of-scope uses.

### Step 6: Documentation and presentation (5 marks)
- PDF report: problem statement and dataset description, workflow/architecture diagram, experiment tables, screenshots, tool comparison, leakage and bias findings, reflections on what AutoML did well and where it failed.
- Slides for a 5-minute demo plus short Q&A.

## 6. Beyond the rubric (stretch goals, do after the floors are covered)
1. Drift monitor: retrain on a sliding window and plot F1 against time to show decay.
2. Cost-sensitive thresholding: a missed malware sample is costlier than a false alarm. Plot the precision/recall trade-off and choose a threshold with an explicit cost ratio.
3. Calibration check: reliability curve and Brier score for the chosen model.
4. Reproducibility package: a `make reproduce` (or `run_all.py`) command that regenerates the tables and figures from raw data with the fixed seed.
5. Cross-tool agreement: compare which features SHAP flags across the manual model and the AutoML winners.
6. Optional: cloud deployment (Streamlit Community Cloud, Hugging Face Spaces or similar) in addition to local.

## 8. Deliverables checklist (what to submit)
- [ ] ~~Course completion certificate~~ (Part A, handled separately)
- [ ] Project code on GitHub or as notebooks, runnable and reproducing the results
- [ ] Project report (PDF)
- [ ] Deployed app link or demo recording
- [ ] Presentation slides
- [ ] Model card

## 9. Grading guidance and ground rules
- Marks depend on correct validation, justified choices and honest reporting, not on the highest accuracy. Prefer a credible, slightly lower score over an inflated one. Report negative or surprising results.
- Fixed random seed everywhere.
- Never tune on the test set. Keep the test set untouched until the final evaluation.
- Copied code or results without understanding are penalised, so comment non-obvious decisions and keep a short "why" note for each major choice (model family, search space, split strategy).
- Log every experiment (method, parameters, score, time) to a CSV or JSON so tables are generated from logs rather than typed by hand.
