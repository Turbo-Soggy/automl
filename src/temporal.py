"""Step 4 (part B): random vs temporal evaluation, and the extra fits Step 4 needs.

What this module does
  1. random_vs_future_table(): every Step 2-3 model scored on test_random and test_future,
     read from the SAVED test scores (reports/predictions/*.npz), with bootstrap CIs and the gap.
  2. Refits of the tuned LightGBM configuration on parts of `train` (FitSpec / run_fit()):
     stratified 3-fold CV, week-blocked forward CV, class-weighted variants, drift windows,
     and a LogReg without the time-proxy columns. Each fit saves its scores to
     reports/predictions/eval/<name>.npz and logs one record (step "eval") to the experiment log,
     so every table is rebuilt from artefacts and a crashed run resumes where it stopped.
  3. cv_comparison_table(): which CV scheme predicts the test_future score better?
  4. Gap evidence: why the tree models lose so little on future weeks (time_proxy_table(),
     logreg_shift_table(), adversarial validation).

Test-set discipline: no function here selects, tunes, early-stops or picks a threshold on
test_random or test_future. Fits use `train` rows only (fixed boosting rounds, no early
stopping); the test sets are only scored. Thresholds come from out-of-fold scores on `train`.
"""
from __future__ import annotations

import json
import platform
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
from config import EXPERIMENT_LOG, MODELS, PREDICTIONS, QUICK, RUN_SUFFIX, SEED, TABLES
from src import tables, tune
from src.eval import paired_bootstrap, percentile_ci
from src.logging_utils import RUN_ID, log_experiment, read_experiments

Scores = dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]  # test set -> (y, {model: p_mal})

# Step 4 shares a 4-core machine with the SHAP job, so its fits use 2 threads. Times measured
# under that load are logged but never reported as benchmarks.
EVAL_N_JOBS: int = 2
EVAL_PREDICTIONS: Path = PREDICTIONS / "eval"
EVAL_MODELS: Path = MODELS / "eval"
FORWARD_FOLDS: int = 4           # week-blocked expanding-window folds over weeks 0-51
TARGET_FPR: float = 0.01         # the operating point of the security metric (TPR at 1% FPR)
DEFAULT_THRESHOLD: float = 0.5   # every model's score cut-off unless a deployment threshold exists
THRESHOLD_FILE: Path = MODELS / f"threshold{RUN_SUFFIX}.json"  # written by the explain job (Step 4 part C)

# Step 2-3 models in table order: label -> saved-prediction stem in reports/predictions/.
STEP23_MODELS: dict[str, str] = {
    "lgbm_default": "lgbm_default", "logreg": "logreg",
    "grid": "grid", "random": "random", "optuna": "optuna",
    "flaml": "automl_flaml", "autogluon": "automl_autogluon", "h2o": "automl_h2o",
}
KIND: dict[str, str] = {"lgbm_default": "baseline", "logreg": "baseline", "grid": "tuned", "random": "tuned",
                        "optuna": "tuned", "flaml": "automl", "autogluon": "automl", "h2o": "automl"}


# =============================================================================
# Saved Step 2-3 scores
# =============================================================================
def logged_eval_ids(log: Path = EXPERIMENT_LOG) -> dict[str, str]:
    """{model label: eval_id of the final_eval record the Step 2-3 tables report}."""
    ids: dict[str, str] = {}
    tuning = tables.tuning_records(log)
    if not tuning.empty:
        for _, rec in tables.latest_final_evals(tuning).iterrows():
            ids[rec["method"]] = rec["eval_id"]
    automl = tables.automl_records(log)
    for _, rec in automl.iterrows():
        ids[rec["tool"]] = rec["eval_id"]
    return ids


def load_step23_scores(predictions: Path = PREDICTIONS, log: Path = EXPERIMENT_LOG,
                       models: dict[str, str] | None = None) -> Scores:
    """{test set: (y, {model: p_mal})} for the Step 2-3 models, from their saved .npz files.

    Refuses a file whose eval_id is not the logged one, or whose labels differ from the others'
    (every model must have been scored on the same rows, in the same order).
    """
    ids = logged_eval_ids(log)
    out: Scores = {}
    for label, stem in (models or STEP23_MODELS).items():
        path = predictions / f"{stem}{RUN_SUFFIX}.npz"
        if label not in ids or not path.exists():
            continue  # e.g. a quick run that skipped a tool; the table simply lacks that row
        with np.load(path) as saved:
            if str(saved["eval_id"]) != ids[label]:
                raise RuntimeError(f"{path} belongs to eval {saved['eval_id']}, not the logged {ids[label]}")
            for test in tables.TEST_SETS:
                y, p = saved[f"{test}_y"].astype(int), saved[f"{test}_p"].astype(float)
                if test in out and not np.array_equal(out[test][0], y):
                    raise RuntimeError(f"{label} was scored on different {test} rows than the other models")
                out.setdefault(test, (y, {}))[1][label] = p
    return out


def tuned_label(log: Path = EXPERIMENT_LOG) -> str:
    """The tuner Step 2 selected by CV AUC (never by a test score): its model is 'the tuned LightGBM'."""
    return tables.selected_tuner(tables.latest_final_evals(tables.tuning_records(log)))


# =============================================================================
# Metrics at an operating point, and the random-vs-future table
# =============================================================================
def operating_metrics(y: np.ndarray, p: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
                      max_fpr: float = TARGET_FPR) -> dict[str, float]:
    """AUC and TPR@1%FPR (threshold-free) plus F1, TPR and FPR at `threshold`, from one ROC sort.

    The bootstrap calls this thousands of times, so it counts confusion cells directly instead of
    calling sklearn's f1_score / confusion_matrix.
    """
    fpr_curve, tpr_curve, _ = roc_curve(y, p, drop_intermediate=False)
    flagged = p >= threshold
    tp = float(np.sum(flagged & (y == 1)))
    fp = float(np.sum(flagged & (y == 0)))
    n_pos = float(np.sum(y == 1))
    n_neg = float(len(y) - n_pos)
    return {
        "roc_auc": float(np.trapezoid(tpr_curve, fpr_curve)),
        "tpr_at_1pct_fpr": float(tpr_curve[fpr_curve <= max_fpr].max()),
        "f1": 2 * tp / (tp + fp + n_pos) if (tp + fp + n_pos) > 0 else 0.0,  # 2TP / (2TP + FP + FN)
        "tpr": tp / n_pos if n_pos else float("nan"),
        "fpr": fp / n_neg if n_neg else float("nan"),
    }


GAP_METRICS: list[str] = ["roc_auc", "f1", "tpr_at_1pct_fpr", "fpr"]


def gap_rows(scores: Scores, labels: list[str], threshold: float = DEFAULT_THRESHOLD,
             n_resamples: int = config.N_BOOTSTRAP) -> list[dict[str, Any]]:
    """One row per model: both test sets' metrics with 95% CIs, and the random-minus-future gap.

    Resamples are paired across models within a test set and independent between the two sets
    (one seed per set, as in Step 2), so the gap CI is (random sample) minus (future sample).
    """
    metric_fn = lambda y, p: operating_metrics(y, p, threshold)  # noqa: E731
    samples = {test: paired_bootstrap(scores[test][0], {m: scores[test][1][m] for m in labels},
                                      metric_fn=metric_fn, n_resamples=n_resamples,
                                      seed=tables.BOOTSTRAP_SEEDS[test])
               for test in tables.TEST_SETS}
    rows = []
    for label in labels:
        row: dict[str, Any] = {"model": label, "threshold": threshold}
        for test in tables.TEST_SETS:
            y, preds = scores[test]
            point = operating_metrics(y, preds[label], threshold)
            row[f"{test}_n"] = len(y)
            for metric in GAP_METRICS:
                row[f"{test}_{metric}"] = point[metric]
                row[f"{test}_{metric}_ci_low"], row[f"{test}_{metric}_ci_high"] = \
                    percentile_ci(samples[test][label][metric])
        for metric in GAP_METRICS:
            # Positive gap = the random split looks better than future weeks (for FPR: lower is
            # better, so a negative FPR gap means more false alarms on future weeks).
            gap = samples["test_random"][label][metric] - samples["test_future"][label][metric]
            row[f"{metric}_gap"] = row[f"test_random_{metric}"] - row[f"test_future_{metric}"]
            row[f"{metric}_gap_ci_low"], row[f"{metric}_gap_ci_high"] = percentile_ci(gap)
        rows.append(row)
    return rows


def random_vs_future_table(scores: Scores, tuned: str, n_resamples: int = config.N_BOOTSTRAP) -> pd.DataFrame:
    """All Step 2-3 models; the CV-selected tuner is marked `is_tuned_lgbm`."""
    labels = [m for m in STEP23_MODELS if m in scores["test_future"][1]]
    table = pd.DataFrame(gap_rows(scores, labels, n_resamples=n_resamples))
    table.insert(1, "kind", table["model"].map(KIND))
    table.insert(2, "is_tuned_lgbm", table["model"] == tuned)
    return table


# =============================================================================
# The tuned configuration and the Step 4 fits
# =============================================================================
@dataclass(frozen=True)
class TunedConfig:
    """The CV-selected LightGBM configuration and its fixed number of boosting rounds."""

    tuner: str
    params: dict[str, Any]
    n_rounds: int


def tuned_config(log: Path = EXPERIMENT_LOG) -> TunedConfig:
    """Read the selected tuner's saved parameters (models/lgbm_<tuner>_params.json)."""
    tuner = tuned_label(log)
    saved = json.loads((MODELS / f"lgbm_{tuner}{RUN_SUFFIX}_params.json").read_text(encoding="utf-8"))
    params = {**saved["params"], "num_threads": EVAL_N_JOBS}  # same model, fewer threads (shared machine)
    return TunedConfig(tuner, params, int(saved["num_trees"]))


def time_proxy_columns() -> dict[str, int]:
    """Columns of X that encode WHEN a file was built or signed (from thrember's feature layout).

    header offset 0 = COFF TimeDateStamp; authenticode offsets 6 and 7 = latest_signing_time and
    signing_time_diff (thrember features.py, AuthenticodeSignature.process_raw_features).
    """
    from src import data

    start = {name: lo for name, lo, _hi in data.feature_groups()}
    return {"coff_timestamp": start["header"] + 0,
            "latest_signing_time": start["authenticode"] + 6,
            "signing_time_diff": start["authenticode"] + 7}


@dataclass(frozen=True)
class FitSpec:
    """One Step 4 fit: which model, on which train positions, scored where.

    model         "lgbm_tuned" (the CV-selected config), "lgbm_default" or "logreg".
    fit_pos       positions in the train split to fit on.
    val_pos       train positions to score (CV folds); None = score test_random and test_future.
    lgbm_extra    LightGBM parameters added to the tuned ones (e.g. scale_pos_weight).
    drop_columns  columns of X removed before fitting and scoring.
    save_model    keep the fitted model under models/eval/ (full-train variants only).
    """

    name: str
    model: str
    fit_pos: np.ndarray
    val_pos: np.ndarray | None = None
    lgbm_extra: dict[str, Any] = field(default_factory=dict)
    drop_columns: tuple[int, ...] = ()
    save_model: bool = False
    note: str = ""


def stratified_cv_specs(model: str, prefix: str) -> list[FitSpec]:
    """The persisted stratified 3-fold folds over ALL of train (data.get_cv_folds())."""
    from src import data

    return [FitSpec(f"{prefix}_fold{k}", model, fit, val, note=f"stratified fold {k}")
            for k, (fit, val) in enumerate(data.get_cv_folds(subset=False))]


def forward_folds(weeks: np.ndarray, n_folds: int = FORWARD_FOLDS) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Expanding-window forward folds over the distinct weeks: fit on the past, validate on the next block.

    The weeks are cut into n_folds + 1 consecutive blocks of (about) equal length; fold k fits on
    blocks 0..k and validates on block k + 1. Every validation week is later than every fit week.
    """
    distinct = np.unique(weeks)
    block = len(distinct) // (n_folds + 1)
    first_val = len(distinct) - n_folds * block  # any remainder goes to the first fit block
    out = []
    for k in range(n_folds):
        lo, hi = first_val + k * block, first_val + (k + 1) * block
        fit_weeks, val_weeks = distinct[:lo], distinct[lo:hi]
        desc = f"fit weeks {fit_weeks.min()}-{fit_weeks.max()}, validate weeks {val_weeks.min()}-{val_weeks.max()}"
        out.append((np.flatnonzero(np.isin(weeks, fit_weeks)), np.flatnonzero(np.isin(weeks, val_weeks)), desc))
    return out


def forward_cv_specs(weeks: np.ndarray) -> list[FitSpec]:
    return [FitSpec(f"cv_forward_fold{k}", "lgbm_tuned", fit, val, note=desc)
            for k, (fit, val, desc) in enumerate(forward_folds(weeks))]


# --- Running one fit ----------------------------------------------------------------------
def prediction_path(name: str) -> Path:
    return EVAL_PREDICTIONS / f"{name}{RUN_SUFFIX}.npz"


def eval_records(log: Path = EXPERIMENT_LOG) -> pd.DataFrame:
    """The latest Step 4 fit record per name, of the current quick/full mode."""
    df = read_experiments(log)
    if df.empty or "step" not in df:
        return pd.DataFrame()
    df = df[(df["step"] == "eval") & (df["kind"] == "fit") & (df["quick"] == QUICK)]
    if df.empty:
        return pd.DataFrame()
    return df.sort_values("timestamp_utc", kind="stable").groupby("name").tail(1).set_index("name")


def fit_done(name: str, log: Path = EXPERIMENT_LOG) -> bool:
    """True when the saved scores exist AND belong to the latest logged record (resume support)."""
    path, records = prediction_path(name), eval_records(log)
    if not path.exists() or name not in records.index:
        return False
    with np.load(path) as saved:
        return str(saved["eval_id"]) == records.loc[name, "eval_id"]


def _fit_predictor(spec: FitSpec, X: np.ndarray, y: np.ndarray, cfg: TunedConfig) -> tuple[Callable, Any, int]:
    """Fit spec.model on X, y; return (X -> P(malicious), the fitted model, boosting rounds or 0)."""
    if spec.model == "logreg":
        model, _converged = tune._fit_logreg(X, y)
        return (lambda X_new: model.predict_proba(X_new)[:, 1]), model, 0
    if spec.model == "lgbm_default":
        params, rounds = {**tune.lgb_params({}), "num_threads": EVAL_N_JOBS}, 100  # LightGBM's default rounds
    else:
        params, rounds = {**cfg.params, **spec.lgbm_extra}, cfg.n_rounds
    dtrain = lgb.Dataset(X, y, params={k: params[k] for k in ("force_col_wise", "verbosity", "feature_pre_filter")})
    booster = lgb.train(params, dtrain, num_boost_round=rounds)
    return booster.predict, booster, rounds


def run_fit(spec: FitSpec, X_train: np.ndarray, y_train: np.ndarray, cfg: TunedConfig,
            test_sets: dict[str, tuple[np.ndarray, np.ndarray]] | None, log: Path = EXPERIMENT_LOG) -> dict[str, Any]:
    """Fit one spec on its train positions, save its scores and log a record. Skips finished fits."""
    if fit_done(spec.name, log):
        print(f"  {spec.name}: already done, skipped", flush=True)
        return {}
    keep = np.setdiff1d(np.arange(X_train.shape[1]), spec.drop_columns)
    columns = keep if spec.drop_columns else slice(None)
    start = time.perf_counter()
    predict, model, rounds = _fit_predictor(spec, X_train[spec.fit_pos][:, columns], y_train[spec.fit_pos], cfg)
    fit_seconds = time.perf_counter() - start

    arrays: dict[str, np.ndarray] = {}
    if spec.val_pos is not None:
        arrays["val_pos"] = spec.val_pos.astype(np.int64)
        arrays["val_y"] = y_train[spec.val_pos].astype(np.int8)
        arrays["val_p"] = predict(X_train[spec.val_pos][:, columns]).astype(np.float64)
    else:
        for test, (X_test, y_test) in (test_sets or {}).items():
            arrays[f"{test}_y"] = y_test.astype(np.int8)
            arrays[f"{test}_p"] = predict(X_test[:, columns]).astype(np.float64)
    eval_id = f"{RUN_ID}-{spec.name}-{uuid.uuid4().hex[:6]}"
    EVAL_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path(spec.name), eval_id=eval_id, fit_pos=spec.fit_pos.astype(np.int64), **arrays)
    model_path = _save_model(spec, model) if spec.save_model else None
    record = {
        "step": "eval", "kind": "fit", "name": spec.name, "model": spec.model, "eval_id": eval_id,
        "note": spec.note, "train_rows": len(spec.fit_pos), "n_rounds": rounds, "tuner": cfg.tuner,
        "lgbm_extra": spec.lgbm_extra, "drop_columns": list(spec.drop_columns),
        "fit_seconds": fit_seconds, "n_jobs": EVAL_N_JOBS, "shared_machine": True,
        "machine": machine_info(), "model_path": model_path,
        "predictions_path": prediction_path(spec.name),
    }
    log_experiment(record, path=log)
    print(f"  {spec.name}: fit on {len(spec.fit_pos)} rows in {fit_seconds:.0f}s", flush=True)
    return record


def _save_model(spec: FitSpec, model: Any) -> Path:
    EVAL_MODELS.mkdir(parents=True, exist_ok=True)
    if isinstance(model, lgb.Booster):
        path = EVAL_MODELS / f"{spec.name}{RUN_SUFFIX}.txt"
        model.save_model(str(path))
    else:
        path = EVAL_MODELS / f"{spec.name}{RUN_SUFFIX}.joblib"
        joblib.dump(model, path)
    return path


@lru_cache(maxsize=1)
def machine_info() -> dict[str, Any]:
    """CPU count, RAM and OS of this machine, logged with every fit (timings differ by machine)."""
    import os

    ram_gib = None
    try:
        ram_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):  # not available on Windows
        pass
    return {"cpu_count": os.cpu_count(), "ram_gib": ram_gib, "os": platform.platform(),
            "processor": platform.processor() or platform.machine()}


def load_fit(name: str) -> dict[str, np.ndarray]:
    """Saved arrays of a finished Step 4 fit (raises if it has not run yet)."""
    path = prediction_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/run_eval_bias.py --stage fits")
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


# =============================================================================
# Thresholds chosen on train only
# =============================================================================
def oof_scores(prefix: str, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """(y, out-of-fold score) over all of train, stitched from the stratified CV fits <prefix>_fold*."""
    folds = saved_folds(prefix)
    y = np.full(n_train, -1, dtype=np.int8)
    p = np.full(n_train, np.nan)
    for fold in folds:
        y[fold["val_pos"]], p[fold["val_pos"]] = fold["val_y"], fold["val_p"]
    if not folds or np.isnan(p).any():
        raise RuntimeError(f"Out-of-fold scores for {prefix} are incomplete ({len(folds)} folds found)")
    return y, p


def saved_folds(prefix: str) -> list[dict[str, np.ndarray]]:
    """The saved CV fits <prefix>_fold0, _fold1, ... in fold order."""
    folds, k = [], 0
    while prediction_path(f"{prefix}_fold{k}").exists():
        folds.append(load_fit(f"{prefix}_fold{k}"))
        k += 1
    return folds


# The Step 4 explain job (scripts/run_explain.py --stages oof) computes exactly the same
# stratified out-of-fold scores of the tuned config (same folds, parameters and rounds). When its
# file is there and checks out, it is reused instead of fitting the same 3 models twice.
SHARED_OOF: Path = PREDICTIONS / f"oof_lgbm_tuned{RUN_SUFFIX}.npz"


def shared_oof_folds(y_train: np.ndarray, cfg: TunedConfig) -> list[dict[str, np.ndarray]] | None:
    """The explain job's OOF scores as per-fold dicts (like saved_folds()), or None if unusable."""
    from src import data

    if not SHARED_OOF.exists():
        return None
    with np.load(SHARED_OOF, allow_pickle=False) as z:
        saved = {key: z[key] for key in z.files}
    persisted = data.get_cv_folds(subset=False)
    same = (str(saved["method"]) == cfg.tuner and int(saved["n_rounds"]) == cfg.n_rounds
            and np.array_equal(saved["y"], y_train) and not np.isnan(saved["p"]).any()
            and all(np.array_equal(np.flatnonzero(saved["fold_ids"] == k), val) for k, (_f, val) in enumerate(persisted)))
    if not same:
        return None
    return [{"fit_pos": fit, "val_pos": val, "val_y": saved["y"][val], "val_p": saved["p"][val]}
            for fit, val in persisted]


def tuned_stratified_folds(y_train: np.ndarray, cfg: TunedConfig) -> tuple[list[dict[str, np.ndarray]], str]:
    """Stratified 3-fold OOF folds of the tuned config, and where they came from."""
    own = saved_folds("cv_stratified")
    if own:
        return own, "this step's fits (reports/predictions/eval/cv_stratified_fold*.npz)"
    shared = shared_oof_folds(y_train, cfg)
    if shared is None:
        raise RuntimeError("No stratified OOF scores of the tuned config: run --stages fits")
    return shared, f"shared with the explain job ({SHARED_OOF.relative_to(config.ROOT)})"


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, max_fpr: float = TARGET_FPR) -> float:
    """Lowest score cut-off whose false-positive rate on (y, p) is at most max_fpr.

    Called with TRAIN out-of-fold scores only, so the threshold never sees a test label.
    """
    fpr, _tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    return float(thresholds[fpr <= max_fpr].min())


def deployment_threshold() -> tuple[float, str]:
    """(threshold, where it came from): models/threshold.json if the explain job wrote one, else 0.5."""
    if not THRESHOLD_FILE.exists():
        return DEFAULT_THRESHOLD, f"0.5 default ({THRESHOLD_FILE.name} does not exist yet)"
    saved = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
    source = THRESHOLD_FILE.relative_to(config.ROOT).as_posix()
    if saved.get("cost_ratio_fn_to_fp") is not None:
        source += f": cost-minimising on train OOF scores at C_FN/C_FP = {saved['cost_ratio_fn_to_fp']:g}"
    return float(saved["threshold"]), source


# =============================================================================
# CV comparison: stratified vs week-blocked forward CV
# =============================================================================
def fold_metrics(folds: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    """Per-fold validation metrics (F1 and FPR at the 0.5 threshold) of saved CV folds."""
    rows = []
    for k, fold in enumerate(folds):
        m = operating_metrics(fold["val_y"].astype(int), fold["val_p"])
        rows.append({"fold": k, "train_rows": len(fold["fit_pos"]), "val_rows": len(fold["val_pos"]), **m})
    return pd.DataFrame(rows)


def cv_comparison_table(scheme_folds: dict[str, list[dict[str, np.ndarray]]], test_point: dict[str, dict[str, float]],
                        notes: dict[str, str]) -> pd.DataFrame:
    """One row per estimate: the CV schemes (mean and std over folds) and the two test sets.

    abs_error_vs_future = |estimate - test_future| per metric: the smaller, the better that
    scheme predicts what the deployed model scores on later weeks.
    """
    metrics = ["roc_auc", "tpr_at_1pct_fpr", "f1", "fpr"]
    rows = []
    for scheme, fold_list in scheme_folds.items():
        folds = fold_metrics(fold_list)
        row: dict[str, Any] = {"estimate": scheme, "n_folds": len(folds),
                               "train_rows_mean": float(folds["train_rows"].mean()), "note": notes.get(scheme, "")}
        for metric in metrics:
            row[metric] = float(folds[metric].mean())
            row[f"{metric}_std"] = float(folds[metric].std(ddof=1)) if len(folds) > 1 else float("nan")
            row[f"{metric}_folds"] = " ".join(f"{v:.4f}" for v in folds[metric])
        rows.append(row)
    for test, point in test_point.items():
        rows.append({"estimate": test, "n_folds": 0, "note": notes.get(test, ""), **{m: point[m] for m in metrics}})
    table = pd.DataFrame(rows)
    future = test_point["test_future"]
    for metric in metrics:
        table[f"{metric}_abs_error_vs_future"] = (table[metric] - future[metric]).abs()
    return table


# =============================================================================
# Why the gap is small: evidence from the data and the saved models
# =============================================================================
def time_proxy_table(X_train: np.ndarray, X_future: np.ndarray, y_future: np.ndarray,
                     booster: lgb.Booster, logreg: Any) -> pd.DataFrame:
    """Per time-proxy column: how often future files fall outside the training range, how much
    the tuned LightGBM relies on it (gain rank), and how LogReg's logit moves on it.

    A tree splits on thresholds learned from train, so a future value above the training maximum
    lands in the same leaf as the largest training values (no extrapolation). A linear model
    multiplies the value by a fixed weight, so out-of-range values push its score without limit.
    """
    gain = booster.feature_importance(importance_type="gain")
    rank = pd.Series(gain).rank(ascending=False, method="min")
    scaler, clf = logreg[0], logreg[-1]
    rows = []
    for name, col in time_proxy_columns().items():
        train_col, fut_col = X_train[:, col], X_future[:, col]
        hi = float(np.max(train_col))
        z_shift = lambda c: (c - scaler.mean_[col]) / scaler.scale_[col] * clf.coef_[0, col]  # noqa: E731
        row = {"column": col, "name": name, "train_max": hi,
               "lgbm_gain_rank": int(rank[col]), "lgbm_gain_share": float(gain[col] / gain.sum()),
               "logreg_coef_standardized": float(clf.coef_[0, col])}
        for label, cls in (("benign", 0), ("malicious", 1)):
            mask = y_future == cls
            row[f"future_{label}_share_above_train_max"] = float(np.mean(fut_col[mask] > hi))
            row[f"future_{label}_share_nonzero"] = float(np.mean(fut_col[mask] != 0))
        row["logreg_logit_shift_future_minus_train_mean"] = float(z_shift(fut_col).mean() - z_shift(train_col).mean())
        rows.append(row)
    return pd.DataFrame(rows)


EXTREME_Z: float = 10.0  # a standardized value beyond +-10 lies far outside the training distribution


def logreg_shift_table(logreg: Any, X_random: np.ndarray, y_random: np.ndarray,
                       X_future: np.ndarray, y_future: np.ndarray) -> pd.DataFrame:
    """LogReg's logit contribution per feature family on each test set, per class (mean and median).

    LogReg's logit is a sum over columns of coef * standardized value, so the change from
    test_random to test_future splits exactly into per-family parts. "Separation lost" = benign
    shift minus malicious shift: positive means the family pulls the two classes together.
    The mean can be dominated by a few files with extreme values (a linear model extrapolates
    them without limit); the median shows the typical file. extreme_share = share of files with
    a standardized value beyond +-EXTREME_Z in that family; top_column = the family's column with
    the largest mean separation lost.
    """
    from src import data
    from src.automl_runs import feature_names

    scaler, clf = logreg[0], logreg[-1]
    # float32 halves memory (40k x 2568 per test set); sums and means are taken in float64.
    mean, scale, w = (a.astype(np.float32) for a in (scaler.mean_, scaler.scale_, clf.coef_[0]))
    z = {test: (np.asarray(X, dtype=np.float32) - mean) / scale
         for test, X in (("test_random", X_random), ("test_future", X_future))}
    ys = {"test_random": y_random, "test_future": y_future}
    names = feature_names()
    groups = data.feature_groups() + [("time_proxies", -1, -1)]
    proxies = list(time_proxy_columns().values())
    rows = []
    for name, lo, hi in groups:
        cols = proxies if name == "time_proxies" else list(range(lo, hi))
        row: dict[str, Any] = {"family": name, "n_columns": len(cols)}
        col_lost = np.zeros(len(cols))
        for cls, label in ((0, "benign"), (1, "malicious")):
            col_means = {}
            for test in ("test_random", "test_future"):
                parts = z[test][ys[test] == cls][:, cols] * w[cols]
                logit = parts.sum(axis=1, dtype=np.float64)
                row[f"{test}_{label}_mean_logit"] = float(logit.mean())
                row[f"{test}_{label}_median_logit"] = float(np.median(logit))
                col_means[test] = parts.mean(axis=0, dtype=np.float64)
            row[f"{label}_shift"] = row[f"test_future_{label}_mean_logit"] - row[f"test_random_{label}_mean_logit"]
            row[f"{label}_median_shift"] = (row[f"test_future_{label}_median_logit"]
                                            - row[f"test_random_{label}_median_logit"])
            shift = col_means["test_future"] - col_means["test_random"]
            col_lost += shift if cls == 0 else -shift
        row["separation_lost"] = row["benign_shift"] - row["malicious_shift"]
        row["separation_lost_median"] = row["benign_median_shift"] - row["malicious_median_shift"]
        top = int(np.argmax(col_lost))
        row["top_column"], row["top_column_name"] = cols[top], names[cols[top]]
        row["top_column_separation_lost"] = float(col_lost[top])
        for test in ("test_random", "test_future"):
            row[f"{test}_extreme_share"] = float(np.mean(np.abs(z[test][:, cols]).max(axis=1) > EXTREME_Z))
            row[f"{test}_max_abs_z"] = float(np.abs(z[test][:, cols]).max())
        rows.append(row)
    table = pd.DataFrame(rows)
    # Families sorted by the typical file's (median) separation lost; the time-proxy subset (its
    # columns are also counted inside header/authenticode) goes last as a separate line.
    families = table[table["family"] != "time_proxies"].sort_values("separation_lost_median", ascending=False)
    return pd.concat([families, table[table["family"] == "time_proxies"]], ignore_index=True)


def adversarial_validation(X_past: np.ndarray, X_future: np.ndarray, names: list[str],
                           n_jobs: int = EVAL_N_JOBS) -> tuple[dict[str, Any], pd.DataFrame]:
    """Can a model tell past files from future files by their features alone? (3-fold CV AUC)

    An AUC near 0.5 means the feature distribution did not move; near 1.0 means it moved a lot.
    The importances show WHICH features carry the date. Labels (benign/malicious) are not used.
    """
    X = np.vstack([X_past, X_future])
    y = np.r_[np.zeros(len(X_past), np.int8), np.ones(len(X_future), np.int8)]
    params = {**tune.lgb_params({}), "num_threads": n_jobs}
    oof = np.zeros(len(y))
    gain = np.zeros(X.shape[1])
    for fit, val in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y):
        booster = lgb.train(params, lgb.Dataset(X[fit], y[fit], params={"verbosity": -1}), num_boost_round=100)
        oof[val] = booster.predict(X[val])
        gain += booster.feature_importance(importance_type="gain")
    top = pd.DataFrame({"feature": names, "column": np.arange(len(names)), "gain_share": gain / gain.sum()})
    top = top.sort_values("gain_share", ascending=False).head(15).reset_index(drop=True)
    return {"auc": float(roc_auc_score(y, oof)), "n_past": len(X_past), "n_future": len(X_future)}, top


def write_csv(table: pd.DataFrame, stem: str, out_dir: Path = TABLES) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}{RUN_SUFFIX}.csv"
    table.to_csv(path, index=False)
    return path


def stratified_sample(pos: np.ndarray, strata: np.ndarray, n: int) -> np.ndarray:
    """Seeded stratified draw of n of `pos` (all of them if n >= len(pos))."""
    if n >= len(pos):
        return np.sort(pos)
    picked, _ = train_test_split(pos, train_size=n, stratify=strata, random_state=SEED)
    return np.sort(picked)

