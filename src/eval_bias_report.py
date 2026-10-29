"""Step 4 part B outputs: every table and figure, built from saved scores and saved models only.

build_challenge()  vectorise the Win64 challenge rows and score every saved model on them.
build_tables()     random_vs_future, cv_comparison, gap evidence, class balance, bias slices,
                   class weights, prevalence, drift, challenge -> reports/tables + reports/figures,
                   plus eval_bias_facts.json (thresholds and other facts the write-up quotes).
Nothing here fits a detection model on test data or picks anything on a test set.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from config import MODELS, PREDICTIONS, QUICK, RUN_SUFFIX, SEED, TABLES
from src import bias, challenge, data, drift, tables, temporal
from src.eval import paired_bootstrap, percentile_ci
from src.logging_utils import read_records
from src.temporal import DEFAULT_THRESHOLD, EVAL_N_JOBS, write_csv

ADVERSARIAL_N: int = 2_000 if QUICK else 20_000   # rows per side of each adversarial-validation run
CHALLENGE_SCORES: Path = temporal.EVAL_PREDICTIONS / f"challenge_scores{RUN_SUFFIX}.npz"
FACTS: Path = TABLES / f"eval_bias_facts{RUN_SUFFIX}.json"
RELOAD_CHECK_ROWS: int = 300
LGBM_FILES: dict[str, str] = {"lgbm_default": "lgbm_default", "grid": "lgbm_grid", "random": "lgbm_random",
                              "optuna": "lgbm_optuna"}
OOF_PREFIX: dict[str, str] = {"lgbm_default": "oof_lgbm_default", "logreg": "oof_logreg"}  # tuned: see oof_thresholds


# =============================================================================
# Loading saved models (for the challenge set and the gap evidence)
# =============================================================================
def load_booster(label: str) -> lgb.Booster:
    return lgb.Booster(model_file=str(MODELS / f"{LGBM_FILES[label]}{RUN_SUFFIX}.txt"))


def load_logreg() -> Any:
    """The Step 2 LogReg pipeline. joblib unpickles, which is safe here only because the file is
    this project's own committed artefact (never a downloaded or uploaded file)."""
    import joblib

    return joblib.load(MODELS / f"logreg{RUN_SUFFIX}.joblib")


def model_predictors() -> tuple[dict[str, Any], dict[str, str], Any]:
    """{label: X -> P(malicious)} for every Step 2-3 model whose saved model is on disk.

    Returns (predictors, notes for the models that cannot be loaded, a cleanup function).
    """
    from src.automl_runs import feature_names, to_frame, write_parquet

    names = feature_names()
    predictors: dict[str, Any] = {}
    notes: dict[str, str] = {}
    for label in LGBM_FILES:
        booster = load_booster(label)
        predictors[label] = lambda X, b=booster: b.predict(X, num_threads=EVAL_N_JOBS)
    logreg = load_logreg()
    predictors["logreg"] = lambda X: logreg.predict_proba(X)[:, 1]

    import joblib  # project's own saved models only (see load_logreg)

    flaml_path = MODELS / "flaml" / f"automl{RUN_SUFFIX}.joblib"
    if flaml_path.exists():
        automl = joblib.load(flaml_path)
        predictors["flaml"] = lambda X: automl.predict_proba(to_frame(X, names))[:, 1]
    else:
        notes["flaml"] = f"{flaml_path} not on disk"
    # The 1.8 GB Step 3 AutoGluon predictor is gitignored: the Step 4 cloud VM did not have it,
    # the laptop that ran Step 3 does. Load it when present (our own Step 3 output, so its
    # pickles are trusted); reload_check() then proves it is the model that was scored.
    ag_dir = MODELS / f"autogluon{RUN_SUFFIX}"
    if (ag_dir / "predictor.pkl").exists():
        from autogluon.tabular import TabularPredictor

        ag = TabularPredictor.load(str(ag_dir), verbosity=0)
        predictors["autogluon"] = lambda X: ag.predict_proba(to_frame(X, names), as_multiclass=False).to_numpy()
    else:
        notes["autogluon"] = (f"{ag_dir.relative_to(config.ROOT).as_posix()} not on this machine: the 1.8 GB Step 3 "
                              "predictor is not committed (its test scores are saved, so it is in every other table)")

    cleanup = lambda: None  # noqa: E731
    automl_recs = tables.automl_records()
    h2o_rec = automl_recs[automl_recs["tool"] == "h2o"]
    h2o_path = MODELS / f"h2o{RUN_SUFFIX}" / str(h2o_rec["best_model"].iloc[0]) if len(h2o_rec) else None
    if h2o_path is not None and h2o_path.exists():
        import h2o

        h2o.init(max_mem_size="3G", nthreads=EVAL_N_JOBS)  # small heap: scoring only, machine is shared
        h2o.no_progress()
        model = h2o.load_model(str(h2o_path))
        tmp = Path(tempfile.mkdtemp(prefix="h2o_score_"))

        def h2o_predict(X: np.ndarray) -> np.ndarray:
            path = write_parquet(X, None, names, tmp / f"score_{len(X)}.parquet")
            frame = h2o.import_file(str(path))
            p = model.predict(frame).as_data_frame(use_multi_thread=True)["p1"].to_numpy(dtype=float)
            h2o.remove(frame)
            return p

        predictors["h2o"] = h2o_predict

        def cleanup() -> None:
            h2o.cluster().shutdown()
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        notes["h2o"] = f"{h2o_path} not on disk"
    return predictors, notes, cleanup


def reload_check(predictors: dict[str, Any]) -> dict[str, float]:
    """Max |score - saved score| on the first test_future rows: proves each loaded file IS the scored model."""
    rows = data.split_rows("test_future")[:RELOAD_CHECK_ROWS]
    X = np.asarray(data.open_cache_X("test")[rows], dtype=np.float32)
    out = {}
    for label, predict in predictors.items():
        with np.load(PREDICTIONS / f"{temporal.STEP23_MODELS[label]}{RUN_SUFFIX}.npz") as saved:
            out[label] = float(np.max(np.abs(predict(X) - saved["test_future_p"][:RELOAD_CHECK_ROWS])))
    return out


# =============================================================================
# Challenge set
# =============================================================================
def build_challenge() -> None:
    """Vectorise the Win64 challenge rows (new cache files) and save every loadable model's scores."""
    if not challenge.cache_paths()["info"].exists():
        info = challenge.build_challenge_cache()
        print(f"[challenge] vectorised {info['rows']} {challenge.FILE_TYPE} rows "
              f"of {info['lines']} challenge lines", flush=True)
    X, meta = challenge.load_challenge()
    predictors, notes, cleanup = model_predictors()
    try:
        check = reload_check(predictors)
        scores = {label: predict(X) for label, predict in predictors.items()}
    finally:
        cleanup()
    temporal.EVAL_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CHALLENGE_SCORES, sha256=meta["sha256"].to_numpy(str),
                        **{f"p_{label}": p for label, p in scores.items()})
    CHALLENGE_SCORES.with_suffix(".json").write_text(json.dumps(
        {"reload_check_max_abs_diff": check, "reload_check_rows": RELOAD_CHECK_ROWS, "not_scored": notes,
         "machine": temporal.machine_info()},  # where this scoring ran (the fits ran elsewhere)
        indent=2), encoding="utf-8", newline="\n")
    print(f"[challenge] scored {len(X)} rows with {', '.join(scores)}; reload check {check}", flush=True)


def challenge_table(meta: pd.DataFrame, scores: dict[str, np.ndarray], thresholds: dict[str, dict[str, float]],
                    future_scores: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Detection rate per (model, threshold, subset) with Wilson CIs, next to the same threshold's
    TPR on ordinary test_future malware (the files AV did detect)."""
    weeks = meta["week"].to_numpy()
    # Files that are also in our train split were seen in training; they are left out everywhere
    # (the count is in challenge_win64_info.json, overlap.in_train_split).
    unseen = ~meta["in_train_split"].to_numpy(bool)
    subsets = {"all Win64 challenge": unseen,
               "weeks 0-51 (training period)": unseen & (weeks <= 51),
               "weeks 52-63 (test_future period)": unseen & (weeks >= 52)}
    rows = []
    for model, p in scores.items():
        for thr_name, thr in thresholds.get(model, {}).items():
            y_f, p_f = future_scores[model]
            future_tpr = float(np.mean(p_f[y_f == 1] >= thr))
            for subset, mask in subsets.items():
                detected, n = int(np.sum(p[mask] >= thr)), int(mask.sum())
                lo, hi = bias.wilson_ci(detected, n)
                rows.append({"model": model, "threshold_name": thr_name, "threshold": thr, "subset": subset,
                             "n": n, "detected": detected, "detection_rate": detected / n if n else np.nan,
                             "ci_low": lo, "ci_high": hi, "too_small": n < bias.MIN_SLICE_N,
                             "test_future_tpr_same_threshold": future_tpr,
                             "median_score": float(np.median(p[mask])) if n else np.nan})
    return pd.DataFrame(rows)


# =============================================================================
# Thresholds (train out-of-fold scores only)
# =============================================================================
def oof_thresholds(y_train: np.ndarray, cfg: temporal.TunedConfig) -> tuple[dict[str, float], dict[str, Any]]:
    """1%-FPR threshold per model with train OOF scores: tuned LightGBM, default LightGBM, LogReg."""
    folds, source = temporal.tuned_stratified_folds(y_train, cfg)
    y = np.concatenate([f["val_y"] for f in folds]).astype(int)
    p = np.concatenate([f["val_p"] for f in folds])
    out = {cfg.tuner: temporal.threshold_at_fpr(y, p)}
    sources = {cfg.tuner: source}
    for label, prefix in OOF_PREFIX.items():
        if temporal.saved_folds(prefix):
            yy, pp = temporal.oof_scores(prefix, len(y_train))
            out[label] = temporal.threshold_at_fpr(yy, pp)
            sources[label] = f"reports/predictions/eval/{prefix}_fold*.npz"
    return out, sources


# =============================================================================
# The tables
# =============================================================================
def cv_comparison(y_train: np.ndarray, cfg: temporal.TunedConfig, tuned_scores: dict[str, tuple[np.ndarray, np.ndarray]],
                  strat_source: str) -> pd.DataFrame:
    """Stratified vs forward CV of the tuned config, against both test sets (and Step 2's tuning CV AUC)."""
    strat, _ = temporal.tuned_stratified_folds(y_train, cfg)
    forward = temporal.saved_folds("cv_forward")
    notes = {
        "stratified_3fold_cv": f"persisted StratifiedKFold(3, shuffle, seed 42) over all of train; source: {strat_source}",
        "forward_cv_4fold": "; ".join(load_note(f"cv_forward_fold{k}") for k in range(len(forward))),
        "test_random": "refit on all of train (saved Step 2 model), same weeks as train",
        "test_future": "refit on all of train (saved Step 2 model), weeks 52-63: the target",
    }
    point = {test: temporal.operating_metrics(y, p) for test, (y, p) in tuned_scores.items()}
    table = temporal.cv_comparison_table({"stratified_3fold_cv": strat, "forward_cv_4fold": forward}, point, notes)
    rec = tables.latest_final_evals(tables.tuning_records())
    rec = rec[rec["method"] == cfg.tuner].iloc[0]
    step2 = {"estimate": "step2_tuning_cv (60k subset, early stopping)", "n_folds": 3, "roc_auc": rec["cv_auc_mean"],
             "roc_auc_std": rec["cv_auc_std"], "train_rows_mean": 40_000 if not QUICK else np.nan,
             "note": "best trial's CV AUC from the Step 2 search log; optimistic (early stopping saw the folds)"}
    step2["roc_auc_abs_error_vs_future"] = abs(step2["roc_auc"] - point["test_future"]["roc_auc"])
    return pd.concat([table, pd.DataFrame([step2])], ignore_index=True)


def load_note(name: str) -> str:
    return str(temporal.eval_records().loc[name, "note"])


def logreg_ablation(scores: temporal.Scores) -> pd.DataFrame:
    """LogReg with and without the 3 time-proxy columns: does dropping them close its gap?"""
    fit = temporal.load_fit("logreg_no_time_proxies")
    ablated = {test: (scores[test][0], {"logreg": scores[test][1]["logreg"],
                                       "logreg_no_time_proxies": fit[f"{test}_p"]}) for test in tables.TEST_SETS}
    for test in tables.TEST_SETS:
        if not np.array_equal(fit[f"{test}_y"].astype(int), scores[test][0]):
            raise RuntimeError(f"logreg_no_time_proxies was scored on different {test} rows")
    return pd.DataFrame(temporal.gap_rows(ablated, ["logreg", "logreg_no_time_proxies"]))


def adversarial_runs(X_train: np.ndarray, booster: lgb.Booster) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Past-vs-future and past-vs-past (control) adversarial validation, on rows outside every test set.

    'past' = a stratified sample of the train split. The other side comes from cache rows that are
    in NO split: same-period rows of weeks 0-51 (control: should be ~0.5), and rows of weeks 52-63
    that are not in test_future. So no test-set row is used, not even its features.
    Each top feature also gets its gain rank in the tuned detector (`booster`): does the detector
    rely on the columns that give the date away?
    """
    from src.automl_runs import feature_names

    names = feature_names()
    rng = np.random.default_rng(SEED)
    used_train = np.r_[data.split_rows("train"), data.split_rows("test_random")]
    unused_past = np.setdiff1d(np.arange(len(data.load_cache_meta("train"))), used_train)
    unused_future = np.setdiff1d(np.arange(len(data.load_cache_meta("test"))), data.split_rows("test_future"))
    n = min(ADVERSARIAL_N, len(X_train), len(unused_past), len(unused_future))
    past = X_train[np.sort(rng.choice(len(X_train), n, replace=False))]
    others = {"same weeks 0-51 (control)": ("train", unused_past), "future weeks 52-63": ("test", unused_future)}
    summary, tops = [], []
    for label, (official, pool) in others.items():
        rows = np.sort(rng.choice(pool, n, replace=False))
        other = np.asarray(data.open_cache_X(official)[rows], dtype=np.float32)
        result, top = temporal.adversarial_validation(past, other, names)
        summary.append({"comparison": f"train split vs {label}", **result})
        detector_rank = pd.Series(booster.feature_importance(importance_type="gain")).rank(ascending=False, method="min")
        tops.append(top.assign(comparison=f"train split vs {label}",
                               tuned_lgbm_gain_rank=detector_rank[top["column"]].astype(int).to_numpy()))
    return pd.DataFrame(summary), pd.concat(tops, ignore_index=True)


def class_weight_table(baseline: str, scores: dict[str, dict[str, np.ndarray]], y: dict[str, np.ndarray],
                       oof_thr: float) -> pd.DataFrame:
    """Each class-weight variant vs the saved tuned model: metrics, mean scores, paired diffs with CIs."""
    rows = []
    metric_fn = lambda yy, pp: temporal.operating_metrics(yy, pp, DEFAULT_THRESHOLD)  # noqa: E731
    for test in tables.TEST_SETS:
        samples = paired_bootstrap(y[test], scores[test], metric_fn=metric_fn, seed=tables.BOOTSTRAP_SEEDS[test])
        for model, p in scores[test].items():
            point = temporal.operating_metrics(y[test], p, DEFAULT_THRESHOLD)
            at_oof = temporal.operating_metrics(y[test], p, oof_thr)
            row = {"model": model, "test_set": test, **point,
                   "tpr_at_tuned_oof_1pct_threshold": at_oof["tpr"], "fpr_at_tuned_oof_1pct_threshold": at_oof["fpr"],
                   "mean_score_benign": float(p[y[test] == 0].mean()),
                   "mean_score_malicious": float(p[y[test] == 1].mean())}
            for metric in ("roc_auc", "tpr_at_1pct_fpr", "f1", "fpr"):
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = percentile_ci(samples[model][metric])
                diff = samples[model][metric] - samples[baseline][metric]
                row[f"{metric}_minus_baseline"] = point[metric] - temporal.operating_metrics(
                    y[test], scores[test][baseline], DEFAULT_THRESHOLD)[metric]
                row[f"{metric}_minus_baseline_ci_low"], row[f"{metric}_minus_baseline_ci_high"] = percentile_ci(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def split_meta(name: str) -> pd.DataFrame:
    return data.load_cache_meta(data.SPLIT_SOURCE[name]).iloc[data.split_rows(name)].reset_index(drop=True)


def fits_machine() -> dict[str, Any]:
    """The machine the Step 4 part B fits ran on, read from their own experiment-log records.

    Not the machine rendering the tables: a rebuild on another computer must not change the
    hardware the write-up reports. Falls back to this machine only when no fit is logged.
    """
    machines = [r["machine"] for r in read_records()
                if r.get("step") == "eval" and r.get("kind") == "fit" and r.get("quick") == QUICK and r.get("machine")]
    if not machines:
        return temporal.machine_info()
    m = dict(machines[-1])
    if "ram_gib" not in m and m.get("ram_gb") is not None:  # the fit records log decimal GB
        m["ram_gib"] = round(m.pop("ram_gb") * 1e9 / 2**30, 1)
    return m


def build_tables() -> dict[str, Path]:
    """Every Step 4 part B table and figure. Returns {name: path}."""
    out: dict[str, Path] = {}
    facts: dict[str, Any] = {"quick": QUICK, "seed": SEED, "n_bootstrap": config.N_BOOTSTRAP,
                             "eval_n_jobs": EVAL_N_JOBS, "machine": fits_machine()}
    cfg = temporal.tuned_config()
    tuned = cfg.tuner
    scores = temporal.load_step23_scores()
    tuned_scores = {test: (scores[test][0], scores[test][1][tuned]) for test in tables.TEST_SETS}
    X_train, y_train, meta_train = data.load_split("train")
    saved_params = json.loads((MODELS / f"lgbm_{tuned}{RUN_SUFFIX}_params.json").read_text(encoding="utf-8"))
    facts.update(tuned_tuner=tuned, tuned_rounds=cfg.n_rounds, saved_model_threads=saved_params["params"]["num_threads"])

    # 1. Random vs future, CV comparison
    out["random_vs_future"] = write_csv(temporal.random_vs_future_table(scores, tuned), "random_vs_future")
    print("[tables] random_vs_future", flush=True)
    thresholds_oof, thr_sources = oof_thresholds(y_train, cfg)
    deploy_thr, deploy_source = temporal.deployment_threshold()
    facts.update(oof_1pct_fpr_thresholds=thresholds_oof, oof_threshold_sources=thr_sources,
                 deployment_threshold=deploy_thr, deployment_threshold_source=deploy_source)
    out["cv_comparison"] = write_csv(cv_comparison(y_train, cfg, tuned_scores, thr_sources[tuned]), "cv_comparison")
    print("[tables] cv_comparison", flush=True)

    # 2. Why the gap is small
    X_random, _, _ = data.load_split("test_random")
    X_future, y_future, meta_future = data.load_split("test_future")
    booster, logreg = load_booster(tuned), load_logreg()
    out["gap_time_proxies"] = write_csv(temporal.time_proxy_table(X_train, X_future, y_future, booster, logreg),
                                        "gap_time_proxies")
    out["gap_logreg_shift"] = write_csv(temporal.logreg_shift_table(
        logreg, X_random, scores["test_random"][0], X_future, y_future), "gap_logreg_shift")
    out["gap_logreg_ablation"] = write_csv(logreg_ablation(scores), "gap_logreg_ablation")
    adv, adv_top = adversarial_runs(X_train, booster)
    out["gap_adversarial"] = write_csv(adv, "gap_adversarial")
    out["gap_adversarial_top"] = write_csv(adv_top, "gap_adversarial_top_features")
    entropy = {"test_random": X_random[:, bias.ENTROPY_COLUMN].copy(), "test_future": X_future[:, bias.ENTROPY_COLUMN].copy()}
    del X_random, X_future
    print("[tables] gap evidence", flush=True)

    # 3. Bias, imbalance, prevalence
    balance = {"train": y_train, "test_random": scores["test_random"][0], "test_future": scores["test_future"][0]}
    out["class_balance"] = write_csv(bias.class_balance_table(balance), "class_balance")
    metas = {"test_random": split_meta("test_random"), "test_future": meta_future}
    per_test = {test: (scores[test][0], scores[test][1][tuned], metas[test], entropy[test]) for test in tables.TEST_SETS}
    slice_thresholds = {"0.5": DEFAULT_THRESHOLD, "train_oof_1pct_fpr": thresholds_oof[tuned]}
    if deploy_source.startswith("models"):
        slice_thresholds["deployment"] = deploy_thr
    edges = bias.size_quartile_edges(meta_train["file_size"].to_numpy())
    facts["file_size_quartile_edges_train"] = edges.tolist()
    slices = bias.slice_table(per_test, edges, slice_thresholds)
    out["bias_slices"] = write_csv(slices, "bias_slices")
    out["bias_slices_png"] = Path(bias.plot_slices(slices, "train_oof_1pct_fpr"))

    variants = {"tuned (saved Step 2 model)": None, "class_weight_none": "class_weight_none",
                "class_weight_balanced": "class_weight_balanced", "class_weight_prior_10pct": "class_weight_prior_10pct"}
    cw_scores = {test: {} for test in tables.TEST_SETS}
    for label, fit_name in variants.items():
        for test in tables.TEST_SETS:
            cw_scores[test][label] = scores[test][1][tuned] if fit_name is None else temporal.load_fit(fit_name)[f"{test}_p"]
    out["class_weight"] = write_csv(class_weight_table("class_weight_none", cw_scores,
                                                       {t: scores[t][0] for t in tables.TEST_SETS},
                                                       thresholds_oof[tuned]), "class_weight")
    facts["class_weight_notes"] = {name: load_note(name) for name in variants.values() if name}

    future = {m: (scores["test_future"][0], p) for m, p in scores["test_future"][1].items()}
    prev_thresholds = {m: {"0.5": DEFAULT_THRESHOLD} for m in future}
    for model, thr in thresholds_oof.items():
        prev_thresholds[model]["train_oof_1pct_fpr"] = thr
    if deploy_source.startswith("models"):
        prev_thresholds[tuned]["deployment"] = deploy_thr
    out["prevalence"] = write_csv(pd.concat([bias.prevalence_table({m: future[m]}, prev_thresholds[m])
                                             for m in future], ignore_index=True), "prevalence")
    print("[tables] bias, class weight, prevalence", flush=True)

    # 4. Drift
    preds = {"expanding": scores["test_future"][1][tuned],
             "sliding_12w": temporal.load_fit("drift_sliding_12w")["test_future_p"],
             "size_matched": temporal.load_fit("drift_size_matched")["test_future_p"]}
    weekly = drift.weekly_table(scores["test_future"][0], meta_future["week"].to_numpy(), preds)
    overall = [drift.overall_rows(scores["test_future"][0], preds, "test_future"),
               drift.overall_rows(scores["test_random"][0], {
                   "expanding": scores["test_random"][1][tuned],
                   "sliding_12w": temporal.load_fit("drift_sliding_12w")["test_random_p"],
                   "size_matched": temporal.load_fit("drift_size_matched")["test_random_p"]}, "test_random")]
    drift_table = pd.concat([weekly, *overall], ignore_index=True)
    out["drift"] = write_csv(drift_table, "drift")
    out["drift_png"] = Path(drift.plot_drift(drift_table))
    facts["drift_slopes_per_week"] = {m: drift.trend_slope(weekly, m) for m in drift.DRIFT_METRICS}
    facts["drift_train_rows"] = {name: int(temporal.eval_records().loc[f"drift_{name}", "train_rows"])
                                 for name in ("sliding_12w", "size_matched")} | {"expanding": len(y_train)}
    print("[tables] drift", flush=True)

    # 5. Challenge set
    if CHALLENGE_SCORES.exists():
        _, meta_c = challenge.load_challenge()
        with np.load(CHALLENGE_SCORES) as saved:
            if not np.array_equal(saved["sha256"], meta_c["sha256"].to_numpy(str)):
                raise RuntimeError("challenge scores do not match the challenge cache rows; re-run --stages challenge")
            c_scores = {k.removeprefix("p_"): saved[k] for k in saved.files if k.startswith("p_")}
        c_thresholds = {m: {"0.5": DEFAULT_THRESHOLD} for m in c_scores}
        for model, thr in thresholds_oof.items():
            if model in c_thresholds:
                c_thresholds[model]["train_oof_1pct_fpr"] = thr
        if deploy_source.startswith("models") and tuned in c_thresholds:
            c_thresholds[tuned]["deployment"] = deploy_thr
        out["challenge"] = write_csv(challenge_table(meta_c, c_scores, c_thresholds, future), "challenge")
        facts["challenge_info"] = json.loads(challenge.cache_paths()["info"].read_text(encoding="utf-8"))
        facts["challenge_scoring"] = json.loads(CHALLENGE_SCORES.with_suffix(".json").read_text(encoding="utf-8"))
        print("[tables] challenge", flush=True)

    FACTS.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8", newline="\n")
    out["facts"] = FACTS
    return out
