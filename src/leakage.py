"""Step 4, part A: leakage experiments. Do the models, and the AutoML tools, fall for a leak?

Variants of the feature matrix (every one is compared with "clean" for the SAME model and budget)
---------------------------------------------------------------------------------------------
  clean                  the standard 2568 EMBER features (what Steps 2 and 3 used).
  leak_detection_ratio   + 1 column: the share of antivirus engines that flagged the file. It is
                         how the label was made, so it separates the classes perfectly (AUC 1.0),
                         and it only exists AFTER the file has been scanned: post-hoc information
                         a real detector never has at prediction time.
  leak_first_submission  + 1 column: VirusTotal's first-submission date as unix seconds. It is a
                         time stamp, not a property of the file. It is (almost) uninformative
                         on its own (50/50 classes every week), but trees can combine it with
                         other columns. On test_future every date lies after every training date,
                         so a tree sees all future rows as "the latest training date": the
                         feature cannot transfer.
  ablate_time_proxies    - 3 columns that ARE in the standard vector and carry time
                         (TIME_PROXIES below, located in thrember's source). LightGBM only.

Models
------
  lgbm_tuned   the Step 2 winner's configuration, refit on train with its fixed n_estimators
               (tuned_lgbm_config(): read from the log, selected by CV AUC, never by a test set).
  flaml, autogluon, h2o   the Step 3 runners (src/automl_runs.py) with LEAKAGE_BUDGET_S each.
               The clean variant is re-run at the same budget, so the leak is the only change.

What is logged per run (one record, step "leakage", kind "final_eval")
---------------------------------------------------------------------
  * metrics on test_random and test_future with 95% bootstrap CIs (same seeds as Steps 2-3),
  * the "watched" columns' importance rank: the leak column and the time proxies, by the tool's
    own importance and by TreeSHAP where the model is a tree model (importance_* below),
  * timings and the sleep-guard report. Whether a tool warned about or dropped the leak column
    is read from its saved logs and notes by src/leakage_report.py.

Test-set discipline: score_tests() is the only function that loads a test set, after the fit.
Importances and SHAP use rows of `train` (or the tool's own validation rows), never a test set.
"""
from __future__ import annotations

import inspect
import os
import platform
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    EXPERIMENT_LOG, LEADERBOARDS, LEAKAGE_BUDGET_S, MODELS, N_JOBS, PREDICTIONS, QUICK, ROOT, RUN_SUFFIX, SEED,
)
from src import automl_runs, tune
from src.automl_runs import AutomlPaths, FitResult, Heartbeat, feature_names, timed, to_frame
from src.data import feature_groups, load_split
from src.logging_utils import RUN_ID, log_experiment, read_records

# --- Variants and models ------------------------------------------------------------------
CLEAN, LEAK_DR, LEAK_DATE, ABLATE = "clean", "leak_detection_ratio", "leak_first_submission", "ablate_time_proxies"
VARIANTS: tuple[str, ...] = (CLEAN, LEAK_DR, LEAK_DATE, ABLATE)
LEAK_COLUMN: dict[str, str] = {LEAK_DR: "leak__detection_ratio", LEAK_DATE: "leak__first_submission_date"}
LGBM: str = "lgbm_tuned"
TOOLS: tuple[str, ...] = automl_runs.TOOLS
MODEL_NAMES: tuple[str, ...] = (LGBM, *TOOLS)
TEST_SETS: tuple[str, ...] = ("test_random", "test_future")

# Run order = priority order. If the long run is cut short, the most important comparisons
# already exist: LightGBM for every variant, then each tool clean + detection_ratio, with FLAML
# also doing the date leak early (the brief asks at least LightGBM and FLAML for it).
RUN_PLAN: tuple[tuple[str, str], ...] = (
    *((LGBM, v) for v in VARIANTS),
    ("flaml", CLEAN), ("flaml", LEAK_DR), ("flaml", LEAK_DATE),
    ("autogluon", CLEAN), ("autogluon", LEAK_DR), ("h2o", CLEAN), ("h2o", LEAK_DR),
    ("autogluon", LEAK_DATE), ("h2o", LEAK_DATE),
)

SHAP_ROWS: int = 2000  # rows of train explained with TreeSHAP (enough for a stable mean |SHAP| ranking)
LEAKAGE_LEADERBOARDS: Path = LEADERBOARDS / "leakage"
LEAKAGE_MODELS: Path = MODELS / "leakage"
LEAKAGE_PREDICTIONS: Path = PREDICTIONS / "leakage"


# --- Time proxies already inside X (documented from thrember's source) --------------------

@dataclass(frozen=True)
class TimeProxy:
    """One column of the standard vector that carries a time stamp.

    offset is the position inside its family's block: the order of the values in that thrember
    class's process_raw_features(). raw_key is the expression there, so a test can re-check it.
    """

    family: str
    offset: int
    raw_key: str
    meaning: str

    @property
    def name(self) -> str:
        return f"{self.family}__{self.offset}"


TIME_PROXIES: tuple[TimeProxy, ...] = (
    TimeProxy("header", 0, 'raw_obj["coff"]["timestamp"]',
              "COFF header TimeDateStamp (the linker's build time), unix seconds"),
    TimeProxy("authenticode", 6, 'raw_obj["latest_signing_time"]',
              "latest countersignature signing time, unix seconds (0 when unsigned)"),
    TimeProxy("authenticode", 7, 'raw_obj["signing_time_diff"]',
              "signing time minus the COFF TimeDateStamp, seconds (0 when unsigned)"),
)
THREMBER_CLASS: dict[str, str] = {"header": "HeaderFileInfo", "authenticode": "AuthenticodeSignature"}


def time_proxy_columns() -> dict[str, int]:
    """{"header__0": 696, ...}: absolute column of each time proxy, from the extractor's offsets."""
    start = {family: begin for family, begin, _ in feature_groups()}
    return {p.name: start[p.family] + p.offset for p in TIME_PROXIES}


def vector_order(class_name: str) -> tuple[list[str], int, Path]:
    """raw_obj[...] expressions in a thrember class's process_raw_features(), in source order.

    Every value of the two classes we need is one raw_obj[...] entry of a single np.hstack,
    so the n-th match is the n-th column of the family block (checked for the scalars at the
    start of the list, which is where all three time proxies are). Also returns the first line
    number of the method and the source file, so the write-up can cite them.
    """
    import thrember.features as tf

    method = getattr(tf, class_name).process_raw_features
    lines, first_line = inspect.getsourcelines(method)
    body = "".join(lines)
    stack = body[body.index("np.hstack"):]  # the values, after the "if not raw_obj" guard
    keys = re.findall(r'raw_obj\["\w+"\](?:\["\w+"\])?', stack)
    return keys, first_line, Path(inspect.getsourcefile(method) or "")


def time_proxy_provenance() -> list[dict[str, Any]]:
    """Per time proxy: its column, meaning, and where thrember's source puts it (verified, not assumed)."""
    columns = time_proxy_columns()
    rows = []
    for proxy in TIME_PROXIES:
        keys, line, source = vector_order(THREMBER_CLASS[proxy.family])
        rows.append({"column": columns[proxy.name], "name": proxy.name, "family": proxy.family,
                     "offset_in_family": proxy.offset, "thrember_class": THREMBER_CLASS[proxy.family],
                     "raw_key": proxy.raw_key, "meaning": proxy.meaning,
                     "source_matches_offset": keys.index(proxy.raw_key) == proxy.offset if proxy.raw_key in keys else False,
                     "source_file": source.name, "process_raw_features_line": line})
    return rows


# --- Building the variant matrices --------------------------------------------------------

def unix_seconds(dates: pd.Series) -> np.ndarray:
    """datetime64 column -> int64 seconds since 1970 (the raw form the date had in the jsonl)."""
    return dates.to_numpy().astype("datetime64[s]").astype(np.int64)


def leak_values(meta: pd.DataFrame, variant: str) -> np.ndarray:
    """The extra column of a leak variant, as float32 like the rest of X.

    float32 keeps 24 bits of precision: unix seconds around 1.7e9 are stored to the nearest
    128 s, far finer than the weekly structure that matters here.
    """
    if variant == LEAK_DR:
        values = meta["detection_ratio_float"].to_numpy()
    elif variant == LEAK_DATE:
        values = unix_seconds(meta["first_submission_date"])
    else:
        raise ValueError(f"{variant!r} has no leak column")
    return np.asarray(values, dtype=np.float32)


def apply_variant(X: np.ndarray, names: list[str], meta: pd.DataFrame, variant: str) -> tuple[np.ndarray, list[str]]:
    """(X, names) of a variant: unchanged, one leak column appended, or the time proxies removed."""
    if variant == CLEAN:
        return X, names
    if variant in LEAK_COLUMN:
        out = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float32)
        out[:, :-1] = X
        out[:, -1] = leak_values(meta, variant)
        return out, [*names, LEAK_COLUMN[variant]]
    if variant == ABLATE:
        drop = set(time_proxy_columns().values())
        keep = [i for i in range(X.shape[1]) if i not in drop]
        return X[:, keep], [names[i] for i in keep]
    raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")


def load_variant(split: str, variant: str) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """(X, y, names, meta) of a persisted split, transformed into `variant`."""
    X, y, meta = load_split(split)
    X, names = apply_variant(X, feature_names(), meta, variant)
    return X, y, names, meta


def watched_columns(names: list[str]) -> list[str]:
    """Columns whose importance we report: the leak column (if any) and the time proxies present."""
    watched = [n for n in names if n in LEAK_COLUMN.values()]
    return watched + [p.name for p in TIME_PROXIES if p.name in names]


# --- Importance ranks and TreeSHAP --------------------------------------------------------

def rank_of(importance: pd.Series, column: str) -> dict[str, Any]:
    """Rank (1 = most important; ties share the best rank, unused columns are last) and share of total.

    A column the model never received (dropped by the tool) gets rank None and in_model False.
    """
    importance = importance.astype(float)
    if column not in importance.index:
        return {"in_model": False, "rank": None, "value": None, "share": None, "n_features": len(importance)}
    ranks = importance.rank(ascending=False, method="min")
    total = float(np.nansum(importance.to_numpy()))
    value = float(importance[column])
    n_ranked = int(importance.notna().sum())
    # An unused column (importance 0) is tied LAST, not "first among the zeros": a model whose
    # importance sits on one column would otherwise show every other column as rank 2.
    rank = None if np.isnan(value) else (n_ranked if value == 0 else int(ranks[column]))
    return {"in_model": True, "rank": rank, "value": value,
            "share": value / total if total and not np.isnan(value) else None,
            "n_features": n_ranked}


def importance_summary(importance: pd.Series, kind: str, watched: list[str], top: int = 5) -> dict[str, Any]:
    """What the log keeps about one importance vector: watched columns' ranks plus the top features."""
    ordered = importance.astype(float).sort_values(ascending=False, na_position="last")
    return {"kind": kind, "ranks": {col: rank_of(importance, col) for col in watched},
            "top": [{"feature": f, "value": float(v)} for f, v in ordered.head(top).items()]}


def mean_abs_shap(contrib: np.ndarray, names: list[str]) -> pd.Series:
    """Mean |SHAP value| per feature: the usual global SHAP importance."""
    return pd.Series(np.abs(contrib).mean(axis=0), index=names)


def sample_rows(n_rows: int, k: int = SHAP_ROWS, seed: int = SEED) -> np.ndarray:
    """k seeded row positions (sorted), or all rows if there are fewer."""
    if n_rows <= k:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=k, replace=False))


def tree_shap(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """TreeSHAP contributions [n, f] for P(malicious), in the model's native way where it has one.

    LightGBM and XGBoost compute exact TreeSHAP themselves (pred_contrib / pred_contribs); that
    avoids version issues between the shap package and those libraries. Other tree models
    (CatBoost, random forests) go through shap.TreeExplainer. The bias column is dropped.
    """
    import lightgbm as lgb

    booster = model.booster_ if hasattr(model, "booster_") else model
    if isinstance(booster, lgb.Booster):
        return np.asarray(booster.predict(X, pred_contrib=True))[:, :-1]
    if hasattr(model, "get_booster"):  # XGBoost sklearn wrapper
        import xgboost as xgb

        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    values = values[1] if isinstance(values, list) else np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values  # (n, f, classes) -> class "malicious"


# --- Tuned LightGBM configuration (from the Step 2 log, selected by CV only) ----------------

def tuned_lgbm_config(log: Path = EXPERIMENT_LOG) -> dict[str, Any]:
    """Params and n_estimators of the Step 2 tuner with the best CV AUC (current refit version).

    Same rule as tables.selected_tuner(): the latest final_eval per tuner, best cv_auc_mean wins.
    """
    finals = [r for r in read_records(log) if r.get("step") == "tuning" and r.get("kind") == "final_eval"
              and r.get("quick") == QUICK and r.get("refit_version") == tune.REFIT_VERSION
              and r.get("method") in tune.TUNING_METHODS]
    latest = {r["method"]: r for r in finals}  # the log is in time order, so the last one per method wins
    if not latest:
        raise RuntimeError("No Step 2 tuner final_eval in the log; run scripts/run_tuning.py first")
    best = max(latest.values(), key=lambda r: r["cv_auc_mean"])
    return {"tuner": best["method"], "params": best["params"], "n_estimators": int(best["refit_n_estimators"]),
            "step2_eval_id": best["eval_id"], "cv_auc_mean": best["cv_auc_mean"],
            "step2_test_random_auc": best["test_random"]["roc_auc"],
            "step2_test_future_auc": best["test_future"]["roc_auc"]}


def leak_split_thresholds(booster: Any, names: list[str]) -> dict[str, Any] | None:
    """Split thresholds the LightGBM model uses on the date leak column (None if no date leak).

    Every split is "value <= threshold". If every threshold lies below the earliest test_future
    date, all future rows fall on the same side of every split on this column, so for the model
    the column is constant on test_future: it carries no information there.
    """
    if LEAK_COLUMN[LEAK_DATE] not in names:
        return None
    trees = booster.trees_to_dataframe()
    column = f"Column_{names.index(LEAK_COLUMN[LEAK_DATE])}"  # LightGBM's default name for an unnamed column
    thresholds = trees.loc[trees["split_feature"] == column, "threshold"].astype(float)
    return {"n_splits": int(len(thresholds)),
            "min_threshold": float(thresholds.min()) if len(thresholds) else None,
            "max_threshold": float(thresholds.max()) if len(thresholds) else None}


# --- Machine and disk ----------------------------------------------------------------------

MIN_FREE_DISK_GB: float = 5.0  # an AutoGluon predictor on 160k x 2569 takes about 2 GB; keep a margin


def machine_info() -> dict[str, Any]:
    """CPU, RAM and OS of this machine, logged with every run (Step 4 ran on a different machine than Steps 2-3)."""
    cpu = platform.processor() or ""
    try:  # on Linux the readable CPU name is in /proc/cpuinfo
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 if hasattr(os, "sysconf") else _windows_ram_gb()
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "ram_gb": None if ram is None else round(ram, 1),
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()}


def _windows_ram_gb() -> float | None:
    """Total RAM on Windows, which has no os.sysconf (GlobalMemoryStatusEx from kernel32)."""
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30
    except AttributeError:  # not Windows either: leave RAM unknown
        pass
    return None


def free_disk_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 2**30


def require_disk(min_gb: float = MIN_FREE_DISK_GB) -> None:
    """Refuse to start a run that could fill the disk (the cloud VM had about 13 GB free)."""
    free = free_disk_gb()
    if free < min_gb:
        raise RuntimeError(f"only {free:.1f} GB free on disk, need {min_gb} GB; delete old model folders first")


# --- Shared run plumbing -----------------------------------------------------------------

def run_paths(variant: str) -> AutomlPaths:
    """Every leakage artefact lives under a leakage/<variant>/ folder, apart from Step 3's files."""
    return AutomlPaths(log=EXPERIMENT_LOG, models=LEAKAGE_MODELS / variant,
                       leaderboards=LEAKAGE_LEADERBOARDS / variant, predictions=LEAKAGE_PREDICTIONS)


def method_id(model: str, variant: str) -> str:
    """"flaml__leak_detection_ratio": the name of a run's predictions file and log record."""
    return f"{model}__{variant}"


def score_tests(predict: automl_runs.Predictor, variant: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The ONLY place the test sets are loaded: after the fit, transformed into the same variant."""
    scored = {}
    for name in TEST_SETS:
        X_test, y_test, _, _ = load_variant(name, variant)
        scored[name] = (y_test, predict(X_test))
    return scored


def evaluate_and_log(model: str, variant: str, predict: automl_runs.Predictor, record: dict[str, Any],
                     paths: AutomlPaths) -> dict[str, Any]:
    """Score both test sets, save the scores, log one final_eval record and return it."""
    from src.tables import BOOTSTRAP_SEEDS  # same CI seeds as Steps 2-3 (imported late: pulls in matplotlib)

    method = method_id(model, variant)
    timings: dict[str, float] = {}
    with timed(timings, "score_s"):
        scored = score_tests(predict, variant)
        metrics = {name: automl_runs.metrics_with_ci(y, p, seed=BOOTSTRAP_SEEDS[name])
                   for name, (y, p) in scored.items()}
    eval_id = f"{RUN_ID}-{method}-{uuid.uuid4().hex[:6]}"
    # Same .npz format as Steps 2-3, so the report bootstraps leak-minus-clean differences from it.
    predictions = automl_runs._save_predictions(method, eval_id, scored, paths)
    full = {"step": "leakage", "kind": "final_eval", "method": method, "model": model, "variant": variant,
            "leak_column": LEAK_COLUMN.get(variant), "eval_id": eval_id, "seed": SEED, "n_jobs": N_JOBS,
            "machine": machine_info(), **record, **timings, **metrics, "predictions_path": predictions}
    log_experiment(full, path=paths.log)
    return full


# --- Tuned LightGBM ----------------------------------------------------------------------

def run_lgbm(variant: str) -> dict[str, Any]:
    """Refit the tuned LightGBM on train in `variant`, explain it, score it, log it."""
    config = tuned_lgbm_config()
    paths = run_paths(variant)
    X, y, names, _ = load_variant("train", variant)
    timings: dict[str, float] = {}
    with Heartbeat(f"{LGBM}:{variant}") as beat, timed(timings, "fit_s"):
        booster = tune.fit_lgbm_fixed_rounds(config["params"], X, y, config["n_estimators"])
    timings["wall_s"] = timings["fit_s"]

    with timed(timings, "artifacts_s"):
        importance = lgbm_importance(booster, X, names, paths)
        paths.models.mkdir(parents=True, exist_ok=True)
        model_path = paths.models / f"lgbm_tuned{RUN_SUFFIX}.txt"
        booster.save_model(str(model_path))
    record = {"budget_s": None, "best_model": f"LightGBM ({config['tuner']} config, {config['n_estimators']} trees)",
              "train_rows": len(y), "n_features": len(names), "lgbm_config": config, **timings,
              **beat.report(), "importance": importance, "model_path": model_path,
              "date_leak_thresholds": leak_split_thresholds(booster, names)}
    return evaluate_and_log(LGBM, variant, booster.predict, record, paths)


def lgbm_importance(booster: Any, X: np.ndarray, names: list[str], paths: AutomlPaths) -> dict[str, Any]:
    """Native gain and split importance plus mean |TreeSHAP| on SHAP_ROWS train rows; saved as CSV too."""
    watched = watched_columns(names)
    gain = pd.Series(booster.feature_importance("gain"), index=names)
    split = pd.Series(booster.feature_importance("split"), index=names)
    shap_imp = mean_abs_shap(tree_shap(booster, X[sample_rows(len(X))]), names)
    table = pd.DataFrame({"feature": names, "gain": gain.to_numpy(), "split": split.to_numpy(),
                          "mean_abs_shap": shap_imp.to_numpy()}).sort_values("mean_abs_shap", ascending=False)
    paths.leaderboards.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.leaderboards / f"lgbm_tuned_importance{RUN_SUFFIX}.csv", index=False)
    return {"native": importance_summary(gain, "LightGBM gain", watched),
            "native_split": importance_summary(split, "LightGBM split count", watched),
            "shap": importance_summary(shap_imp, f"mean |TreeSHAP| on {min(SHAP_ROWS, len(X))} train rows", watched)}


# --- AutoML tools ------------------------------------------------------------------------

def _csv(paths: AutomlPaths, stem: str) -> Path:
    """A Step 3 runner's output file for this variant (same naming as automl_runs._out)."""
    return paths.leaderboards / f"{stem}{RUN_SUFFIX}.csv"


def json_notes(paths: AutomlPaths, stem: str) -> dict[str, Any]:
    """A runner's JSON notes for this variant ({} if missing)."""
    import json

    path = paths.leaderboards / f"{stem}{RUN_SUFFIX}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _series(path: Path, key: str, value: str) -> pd.Series:
    frame = pd.read_csv(path)
    return pd.Series(frame[value].to_numpy(float), index=frame[key].astype(str))


def flaml_importance(fit: FitResult, X: np.ndarray, names: list[str], paths: AutomlPaths) -> dict[str, Any]:
    """FLAML's feature_importances_ of its winner, plus TreeSHAP if the winner is a tree model."""
    watched = watched_columns(names)
    out: dict[str, Any] = {}
    native_path = _csv(paths, "flaml_feature_importance")
    if native_path.exists():
        out["native"] = importance_summary(_series(native_path, "feature", "importance"),
                                           f"FLAML feature_importances_ of {fit.best_model}", watched)
    if fit.best_model not in automl_runs.TREE_LEARNERS:
        out["shap"] = {"skipped": f"winner {fit.best_model} is not a tree model"}
        return out
    try:
        # Safe to unpickle: run_flaml() wrote this file seconds ago in this same run (never downloaded).
        automl = joblib.load(paths.models / "flaml" / f"automl{RUN_SUFFIX}.joblib")
        # The columns FLAML's model received: we dropped the constant ones (lean preprocessing).
        kept = list(automl.feature_names_in_)
        X_t = to_frame(X[sample_rows(len(X))], names)[kept]
        shap_imp = mean_abs_shap(tree_shap(automl.model.estimator, X_t), list(X_t.columns))
        out["shap"] = importance_summary(shap_imp, f"mean |TreeSHAP| of {fit.best_model} on {len(X_t)} train rows",
                                         watched)
    except Exception as exc:  # noqa: BLE001 - SHAP is extra evidence; the native rank is already logged
        out["shap"] = {"error": repr(exc)}
    return out


def autogluon_importance(fit: FitResult, X: np.ndarray, names: list[str], paths: AutomlPaths) -> dict[str, Any]:
    """AutoGluon's permutation importance (its own validation rows), per feature and per family.

    TreeSHAP too when the winner is in effect one LightGBM model (all ensemble weight on it).
    """
    watched = watched_columns(names)
    out: dict[str, Any] = {}
    feature_path, family_path = _csv(paths, "autogluon_feature_importance"), _csv(paths, "autogluon_family_importance")
    if feature_path.exists():
        out["native"] = importance_summary(_series(feature_path, "feature", "importance"),
                                           "AutoGluon permutation importance (internal validation rows)", watched)
    if family_path.exists():
        families = _series(family_path, "family", "importance")
        out["family"] = importance_summary(families, "AutoGluon permutation importance per family block",
                                           [f for f in ("leak", "header", "authenticode") if f in families.index])
    out["shap"] = _autogluon_shap(fit, X, names, paths, watched)
    return out


def _autogluon_shap(fit: FitResult, X: np.ndarray, names: list[str], paths: AutomlPaths,
                    watched: list[str]) -> dict[str, Any]:
    weights = fit.info.get("ensemble_weights") or {}
    members = [m for m, w in weights.items() if w > 0] or [fit.best_model]
    if len(members) != 1 or not members[0].startswith("LightGBM"):
        return {"skipped": f"winner is not a single LightGBM model (members: {members})"}
    try:
        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor.load(str(paths.models / f"autogluon{RUN_SUFFIX}"), verbosity=0)
        member = predictor._trainer.load_model(members[0])  # no public accessor for a base model
        frame = predictor.transform_features(to_frame(X[sample_rows(len(X))], names), model=members[0])
        X_m = member.preprocess(frame)  # exactly what AutoGluon hands the LightGBM booster
        shap_imp = mean_abs_shap(tree_shap(member.model, X_m), list(X_m.columns))
        return importance_summary(shap_imp, f"mean |TreeSHAP| of {members[0]} on {len(X_m)} train rows", watched)
    except Exception as exc:  # noqa: BLE001 - extra evidence only
        return {"error": repr(exc)}


def h2o_importance(fit: FitResult, X: np.ndarray, names: list[str], paths: AutomlPaths) -> dict[str, Any]:
    """H2O variable importance of the best model that has one, plus its TreeSHAP (predict_contributions)."""
    watched = watched_columns(names)
    out: dict[str, Any] = {}
    varimp_path = _csv(paths, "h2o_varimp")
    model_id = fit.info.get("varimp_model")
    if varimp_path.exists():
        # H2O lists only the columns a model used; every other input column it kept has importance 0.
        # Constant columns are dropped by H2O itself, so they stay out (in_model False).
        constant = set(json_notes(paths, "h2o_preprocessing").get("constant_in_train") or [])
        varimp = _series(varimp_path, "feature", "relative_importance")
        varimp = varimp.reindex([n for n in names if n not in constant], fill_value=0.0)
        out["native"] = importance_summary(varimp, f"H2O varimp of {model_id} (unused columns = 0)", watched)
    if not model_id:
        out["shap"] = {"skipped": "no H2O model with variable importance"}
        return out
    try:
        import h2o

        model = h2o.get_model(model_id)
        path = automl_runs.write_parquet(X[sample_rows(len(X))], None, names,
                                         paths.models / f"h2o{RUN_SUFFIX}" / "data" / "shap_rows.parquet")
        frame = h2o.import_file(str(path))
        contrib = model.predict_contributions(frame).as_data_frame(use_multi_thread=True)
        contrib = contrib.drop(columns=["BiasTerm"], errors="ignore")
        shap_imp = mean_abs_shap(contrib.to_numpy(float), list(contrib.columns))
        out["shap"] = importance_summary(shap_imp, f"mean |TreeSHAP| of {model_id} on {len(contrib)} train rows",
                                         watched)
    except Exception as exc:  # noqa: BLE001 - GLM/DRF leaders may not support contributions; extra evidence only
        out["shap"] = {"error": repr(exc)}
    return out


TOOL_IMPORTANCE = {"flaml": flaml_importance, "autogluon": autogluon_importance, "h2o": h2o_importance}
H2O_EXCLUDE: tuple[str, ...] = ("XGBoost",)
# H2O heap for the Step 4 runs. The cloud VM's memory limit is about 14 GB for all our processes:
# with the default 9G heap the JVM reached ~9.9 GB and, with the Python worker holding train, the
# kernel killed it at the end of a 600 s run. 7G leaves room for the worker.
H2O_STEP4_MAX_MEM: str = "7G"
TOOL_OPTIONS = {  # extra runner arguments of the Step 4 runs (see run_tool)
    "flaml": lambda names: {"lean_preprocessing": True},
    "autogluon": lambda names: {"fi_first": tuple(watched_columns(names))},
    "h2o": lambda names: {"exclude_algos": H2O_EXCLUDE, "max_mem": H2O_STEP4_MAX_MEM},
}


def run_tool(tool: str, variant: str, budget_s: int = LEAKAGE_BUDGET_S) -> dict[str, Any]:
    """Run one AutoML tool on train in `variant` (Step 3 runner, LEAKAGE_BUDGET_S), explain, score, log."""
    paths = run_paths(variant)
    require_disk()
    X, y, names, _ = load_variant("train", variant)
    # AutoGluon's per-feature permutation importance stops at a time cap: compute the watched columns first.
    # FLAML: lean preprocessing (same data, far less memory; see automl_runs.run_flaml). H2O: no
    # XGBoost (memory outside the H2O heap; also absent in Step 3). The same for every variant.
    extra = TOOL_OPTIONS[tool](names)
    with Heartbeat(f"{tool}:{variant}") as beat:
        fit = automl_runs.RUNNERS[tool](X, y, budget_s, paths, names=names, **extra)
    ag_dir = paths.models / f"autogluon{RUN_SUFFIX}"
    try:
        start = time.perf_counter()
        importance = TOOL_IMPORTANCE[tool](fit, X, names, paths)
        importance_s = time.perf_counter() - start
        del X  # free train before the test sets are loaded
        record = {"budget_s": budget_s, "best_model": fit.best_model, "tool_version": fit.info.get("version"),
                  "train_rows": len(y), "n_features": len(names), **fit.timings, "importance_s": importance_s,
                  **beat.report(), "importance": importance, "interpretability": fit.interpretability,
                  "info": fit.info, "disk_free_gb_after_fit": round(free_disk_gb(), 1)}
        if tool == "autogluon":
            # The predictor folder (~2 GB) is deleted right after scoring (finally below): its
            # leaderboard, importances, notes and test scores are saved first. The disk is too small to keep it.
            record["model_folder_deleted_after_scoring"] = str(ag_dir.relative_to(ROOT).as_posix())
        return evaluate_and_log(tool, variant, fit.predict, record, paths)
    finally:
        fit.cleanup()  # always shut H2O down
        if tool == "autogluon":
            shutil.rmtree(ag_dir, ignore_errors=True)
