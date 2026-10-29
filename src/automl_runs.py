"""Step 3: AutoML with FLAML, AutoGluon and H2O, on exactly the inputs manual tuning used.

Same inputs for every tool
--------------------------
  * data      : the persisted `train` split (data.load_split), float32 [n, 2568], labels 0/1.
                No tool ever sees test_random or test_future before final scoring.
  * features  : named "{family}__{i}" (feature_names()), so importances group by EMBER family.
  * budget    : AUTOML_BUDGET_S seconds of search per tool (1200 s; 60 s in quick mode).
  * metric    : ROC-AUC, the metric Step 2 tuned for.
  * threads   : N_JOBS (16), the same cores the manual tuners used.
  * seed      : SEED (42), wherever a tool takes one. Parts of FLAML and AutoGluon use the
                library's own fixed seed instead; automl_checks reads the seeds actually used
                back from the fitted models and the write-up lists them.
Tools run one after another (scripts/run_automl.py starts one process per tool), so each
has the machine to itself.

Timing
------
Each tool's `timings` dict has
  init_s       H2O cluster start-up (0 for the in-process tools),
  convert_s    numpy -> the tool's input format (DataFrame / parquet + H2O import),
  fit_s        the tool's fit call, including anything it does after its search budget
               (FLAML's retrain of the winner, AutoGluon's ensembling and refits),
  wall_s       init_s + convert_s + fit_s: start to "model ready", the number compared with
               manual tuning,
  artifacts_s  saving leaderboards, importances and the model (not part of wall_s),
  score_s      predicting and scoring both test sets (not part of wall_s).
budget_s is logged next to them, so overrunning the budget is visible, not hidden.

Sleep guard
-----------
A laptop that goes to sleep freezes the process, but the clocks keep running, so a timed budget that
included sleep is not the same budget as the others. Heartbeat (below) notes the time every 30 s on a
background thread; a gap of more than 60 s between two beats means the process was suspended. The
gap is logged with the result (timing_valid), and scripts/run_automl.py re-runs an invalid tool.

Test-set discipline
-------------------
evaluate_and_log() is the only function here that loads the test sets, and it runs after the
fit has finished. Every leaderboard and importance comes from the tool's own internal
validation on `train` (FLAML holdout or CV, AutoGluon holdout, H2O's own validation frame or
CV, whichever it chose: on 160k rows with a 1200 s budget it chose a validation frame, not CV).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    AUTOML_BUDGET_S, EXPERIMENT_LOG, LEADERBOARDS, MODELS, N_JOBS, PREDICTIONS, RUN_SUFFIX, SEED,
)
from src.automl_checks import ag_checks, flaml_checks, h2o_checks, h2o_interpretability, importance_type
from src.data import feature_groups, load_split
from src.eval import compute_metrics, paired_bootstrap, percentile_ci
from src.logging_utils import RUN_ID, log_experiment

Predictor = Callable[[np.ndarray], np.ndarray]  # X float32 [n, D] -> P(malicious) [n]

TOOLS: tuple[str, ...] = ("flaml", "autogluon", "h2o")
LABEL: str = "label"
TEST_SETS: tuple[str, ...] = ("test_random", "test_future")
CI_METRICS: tuple[str, ...] = ("roc_auc", "tpr_at_1pct_fpr")
TREE_LEARNERS: frozenset[str] = frozenset({"lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree", "catboost"})

# AutoGluon permutation importance. It is computed on AutoGluon's own internal validation rows
# (never a test set). At 160k rows AutoGluon holds out only 2,500 rows, so "5k" means "all of them".
AG_FI_ROWS: int = 5000
AG_FI_FAMILY_SHUFFLES: int = 5      # 12 families: cheap, so average 5 shuffles for a stable ranking
AG_FI_FEATURE_CHUNK: int = 256      # per-feature importance is computed in chunks ...
AG_FI_TIME_LIMIT_S: float = 60.0 if AUTOML_BUDGET_S < 600 else 600.0  # ... until this cap is hit


def _h2o_max_mem() -> str:
    """24G as on the 63 GB laptop of Steps 2-3, but at most ~60% of RAM so a smaller machine does not swap."""
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 if hasattr(os, "sysconf") else 63
    return f"{min(24, int(0.6 * ram_gb))}G"


H2O_MAX_MEM: str = _h2o_max_mem()

HEARTBEAT_S: float = 30.0   # how often the sleep guard notes the time
MAX_GAP_S: float = 60.0     # a longer gap between two beats = the process was suspended (sleep)
HEARTBEAT_PRINT_EVERY: int = 10  # also print every 10th beat (5 min), so the run log shows progress


@dataclass(frozen=True)
class AutomlPaths:
    """Where logs, models, leaderboards and saved test scores go. Tests point these at a temp dir."""

    log: Path = EXPERIMENT_LOG
    models: Path = MODELS
    leaderboards: Path = LEADERBOARDS
    predictions: Path = PREDICTIONS


@dataclass
class FitResult:
    """What every tool runner returns: a ready predictor plus the facts to log about the run."""

    tool: str
    best_model: str
    predict: Predictor
    timings: dict[str, float]
    info: dict[str, Any] = field(default_factory=dict)       # tool facts for the log record
    interpretability: str = ""
    cleanup: Callable[[], None] = lambda: None              # e.g. shut H2O down after scoring


# --- Shared helpers -----------------------------------------------------------------------

def feature_names() -> list[str]:
    """Column names "{family}__{i}" (i counts from 0 inside each family), in matrix order."""
    names: list[str] = []
    for family, start, end in feature_groups():
        names += [f"{family}__{i}" for i in range(end - start)]
    return names


def family_of(name: str) -> str:
    """"header__0" -> "header": the inverse of feature_names(), used to group importances."""
    return name.split("__", 1)[0]


def to_frame(X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Wrap X as a DataFrame with the shared feature names (one float32 block, no per-column copies)."""
    return pd.DataFrame(np.asarray(X, dtype=np.float32), columns=names, copy=False)


@contextmanager
def timed(timings: dict[str, float], key: str) -> Iterator[None]:
    """Store the wall-clock seconds of the `with` block in timings[key]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - start


def finish_timings(timings: dict[str, float]) -> dict[str, float]:
    """Add wall_s = init + convert + fit (what is compared with manual tuning)."""
    timings.setdefault("init_s", 0.0)
    timings["wall_s"] = timings["init_s"] + timings["convert_s"] + timings["fit_s"]
    return timings


def max_gap(beats: list[float]) -> float:
    """Longest time between two consecutive beats (0 with fewer than two beats)."""
    return float(max(np.diff(beats), default=0.0)) if len(beats) > 1 else 0.0


class Heartbeat:
    """Sleep guard: a daemon thread notes time.time() every `interval` s while the block runs.

    A sleeping laptop freezes the thread, so the next beat comes late; time.time() (the calendar
    clock) still jumps over the sleep. report() says whether any gap exceeded `max_gap_s`.
    """

    def __init__(self, label: str, interval: float = HEARTBEAT_S, max_gap_s: float = MAX_GAP_S) -> None:
        self.label, self.interval, self.max_gap_s = label, interval, max_gap_s
        self.beats: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{label}", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):  # wait() returns True only when stopped
            self.beats.append(time.time())
            if len(self.beats) % HEARTBEAT_PRINT_EVERY == 0:
                print(f"[{self.label}] alive {time.strftime('%H:%M:%S')}, "
                      f"{self.beats[-1] - self.beats[0] + self.interval:.0f} s in, "
                      f"max gap {max_gap(self.beats):.0f} s", flush=True)

    def __enter__(self) -> Heartbeat:
        self.beats.append(time.time())
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.beats.append(time.time())  # the last stretch counts too

    def report(self) -> dict[str, Any]:
        """What gets logged with the result: the longest gap and whether the timing is usable."""
        gap = max_gap(self.beats)
        return {"heartbeat_interval_s": self.interval, "heartbeat_max_gap_s": gap,
                "heartbeat_seconds": self.beats[-1] - self.beats[0] if self.beats else 0.0,
                "timing_valid": gap <= self.max_gap_s}


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def _out(paths: AutomlPaths, stem: str, ext: str) -> Path:
    """reports/leaderboards/<stem>[_quick]<ext>."""
    paths.leaderboards.mkdir(parents=True, exist_ok=True)
    return paths.leaderboards / f"{stem}{RUN_SUFFIX}{ext}"


def missing_values(X: np.ndarray) -> int:
    """NaN count in train: shows whether a tool's imputation step had anything to do."""
    return int(np.isnan(X).sum())


def duplicate_of(X: np.ndarray, names: list[str], columns: list[str]) -> dict[str, str | None]:
    """For each column in `columns`: the first OTHER column with identical values in train (or None).

    Used to check, not assume, why a tool dropped a non-constant column.
    """
    index = {n: i for i, n in enumerate(names)}
    out: dict[str, str | None] = {}
    for col in columns:
        values = X[:, index[col]]
        same = np.flatnonzero((X == values[:, None]).all(axis=0))
        out[col] = next((names[j] for j in same if j != index[col]), None)
    return out


def constant_columns(X: np.ndarray, names: list[str]) -> list[str]:
    """Columns with a single value in train: every tool may drop these, so we record which ones exist."""
    return [n for n, is_const in zip(names, np.ptp(X, axis=0) == 0) if is_const]


@contextmanager
def capture_log(logger_name: str, path: Path) -> Iterator[None]:
    """Copy a library's log messages to `path` while the block runs (evidence for the write-up)."""
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        handler.close()


# --- FLAML --------------------------------------------------------------------------------

FLAML_LOG_PATTERNS: tuple[str, ...] = (
    r"Evaluation method", r"Data split method", r"Minimizing error metric", r"List of ML learners",
    r"[Rr]etrain", r"Fit started", r"Time taken to find the best model",
    r"Best ML lea?rner",  # FLAML's own message is spelled "Best ML leaner"
)


def flaml_log_highlights(path: Path) -> list[str]:
    """Lines of FLAML's own log about the evaluation method, learners and the final retrain."""
    if not path.exists():
        return []
    pattern = re.compile("|".join(FLAML_LOG_PATTERNS))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [line for line in lines if pattern.search(line) and "trial" not in line.lower()][:40]


def read_flaml_log(log_file: Path) -> pd.DataFrame:
    """One row per trial from FLAML's search log (the final checkpoint line has no learner and is skipped)."""
    rows = []
    with log_file.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line) if line.strip() else {}
            if "learner" in record:
                rows.append(record)
    return pd.DataFrame(rows)


def flaml_leaderboard(trials: pd.DataFrame) -> pd.DataFrame:
    """Per estimator: number of trials, best validation loss (1 - AUC) and the config that reached it."""
    rows = []
    for learner, part in trials.groupby("learner", sort=False):
        best = part.loc[part["validation_loss"].idxmin()]
        rows.append({
            "estimator": learner,
            "n_trials": len(part),
            "best_val_loss": float(best["validation_loss"]),
            "best_val_auc": 1.0 - float(best["validation_loss"]),
            "best_record_id": int(best["record_id"]),
            "best_found_at_s": float(best["wall_clock_time"]),
            "best_sample_size": int(best["sample_size"]),
            "max_sample_size": int(part["sample_size"].max()),
            "total_trial_time_s": float(part["trial_time"].sum()),
            "best_config": json.dumps(best["config"], sort_keys=True),
        })
    board = pd.DataFrame(rows).sort_values("best_val_loss", kind="stable").reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def run_flaml(X: np.ndarray, y: np.ndarray, budget_s: int, paths: AutomlPaths,
              names: list[str] | None = None) -> FitResult:
    """FLAML AutoML: its own estimator list, holdout/CV choice and search, within budget_s.

    names: column names of X; None = the standard 2568 EMBER features (the Step 4 leakage runs
    pass their own list, with an extra or a removed column).
    """
    from flaml import AutoML

    names, timings = names or feature_names(), {}
    with timed(timings, "convert_s"):
        X_df = to_frame(X, names)
    log_file = _out(paths, "flaml_search_log", ".json")
    automl = AutoML()
    with timed(timings, "fit_s"), capture_log("flaml.automl.logger", _out(paths, "flaml_fit_log", ".txt")):
        # log_type="all" logs every trial, not only improvements, so trials can be counted.
        automl.fit(X_df, y, task="classification", metric="roc_auc", time_budget=budget_s, seed=SEED,
                   n_jobs=N_JOBS, log_file_name=str(log_file), log_type="all", verbose=3)
    finish_timings(timings)

    with timed(timings, "artifacts_s"):
        board = flaml_leaderboard(read_flaml_log(log_file))
        board.to_csv(_out(paths, "flaml_leaderboard", ".csv"), index=False)
        info = _flaml_artifacts(automl, X, names, board, paths)
    learner = automl.best_estimator
    kind = "tree model: native importances and TreeSHAP" if learner in TREE_LEARNERS else "non-tree model"
    return FitResult(
        tool="flaml", best_model=learner, timings=timings, info=info,
        predict=lambda X_new: automl.predict_proba(to_frame(X_new, names))[:, 1],
        interpretability=f"single {learner} model ({kind})",
    )


def _flaml_artifacts(automl: Any, X: np.ndarray, names: list[str], board: pd.DataFrame,
                     paths: AutomlPaths) -> dict[str, Any]:
    """Save the winner (joblib), its config and FLAML's evaluation / preprocessing choices."""
    model_dir = paths.models / "flaml"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(automl.model, model_dir / f"best_estimator{RUN_SUFFIX}.joblib")   # the winning estimator
    joblib.dump(automl, model_dir / f"automl{RUN_SUFFIX}.joblib")  # with FLAML's input transformer, for predict
    # FLAML's DataTransformer (DataFrame input): drops columns with < 2 distinct values and
    # median-imputes the numeric ones. The columns it kept are the transformer's numeric list.
    transformer = automl._transformer  # no public accessor; read-only use
    kept = list(transformer._num_columns) + list(transformer._cat_columns)
    state = automl._state
    notes = {
        "tool": "flaml", "version": _version("flaml"),
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_val_auc": 1.0 - float(automl.best_loss),
        # What feature_importances_ measures for the winner.
        "importance_type": importance_type(automl.best_estimator,
                                           getattr(getattr(automl.model, "estimator", None), "importance_type", None)),
        "eval_method": state.eval_method,
        "split_ratio": automl._settings.get("split_ratio"),
        "n_splits": automl._settings.get("n_splits"),
        # FLAML deletes its data split after fit; with eval_method "holdout" it holds out
        # split_ratio of the rows (stratified), so this is the expected holdout size.
        "holdout_rows_expected": int(round(len(X) * automl._settings.get("split_ratio", 0.1)))
        if state.eval_method == "holdout" else None,
        "fit_log_highlights": flaml_log_highlights(_out(paths, "flaml_fit_log", ".txt")),
        "retrain_full": automl._settings.get("retrain_full"),
        "time_to_find_best_model_s": automl.time_to_find_best_model,
        "estimator_list": list(automl.estimator_list),
        "estimators_tried": board["estimator"].tolist(),
        "n_trials_total": int(board["n_trials"].sum()),
        "n_features_in": len(names),
        "n_features_used": len(kept),
        "dropped_features": sorted(set(names) - set(kept)),
        "new_features_created": sorted(set(kept) - set(names)),  # empty = no feature engineering
        "constant_in_train": constant_columns(X, names),
        "missing_values_in_train": missing_values(X),
        "imputation": "SimpleImputer(strategy='median') on numeric columns (FLAML DataTransformer)",
        "feature_engineering": "none (FLAML only converts types, drops constant columns and imputes)",
        "checks": flaml_checks(automl, X),  # seeds actually used, feature scale (see automl_checks)
    }
    _write_json(notes, _out(paths, "flaml_preprocessing", ".json"))
    importances = automl.feature_importances_
    if importances is not None:
        used = list(automl.feature_names_in_) if automl.feature_names_in_ is not None else kept
        fi = pd.DataFrame({"feature": used, "importance": np.asarray(importances, dtype=float)})
        fi["family"] = fi["feature"].map(family_of)
        fi.sort_values("importance", ascending=False).to_csv(
            _out(paths, "flaml_feature_importance", ".csv"), index=False)
    return {k: notes[k] for k in ("best_config", "best_val_auc", "eval_method", "holdout_rows_expected",
                                  "n_trials_total", "estimators_tried", "n_features_used", "version")}


# --- AutoGluon ----------------------------------------------------------------------------

AG_LOG_PATTERNS: tuple[str, ...] = (
    r"Useless Original Features", r"Unused Original Features", r"[Dd]uplicate", r"Types of features",
    r"Fitting model:", r"Skipping", r"Warning: Exception", r"[Nn]ot enough", r"[Tt]ime limit",
    r"holdout_frac|Automatically generating train/validation split", r"Stage \d Generators",
    r"Fitting \w+FeatureGenerator", r"Dropping", r"Fitting \d+ L1 models",
)


def ag_log_highlights(log_text: str) -> list[str]:
    """Lines of AutoGluon's fit log about preprocessing, models fitted and models skipped."""
    pattern = re.compile("|".join(AG_LOG_PATTERNS))
    return [line.rstrip() for line in log_text.splitlines() if pattern.search(line)]


def run_autogluon(X: np.ndarray, y: np.ndarray, budget_s: int, paths: AutomlPaths,
                  names: list[str] | None = None) -> FitResult:
    """AutoGluon TabularPredictor, medium_quality preset (no stacking or bagging), within budget_s.

    names: column names of X, as in run_flaml().
    """
    from autogluon.tabular import TabularPredictor

    names, timings = names or feature_names(), {}
    with timed(timings, "convert_s"):
        train_df = to_frame(X, names)
        train_df[LABEL] = y.astype(np.int64)
    model_dir = paths.models / f"autogluon{RUN_SUFFIX}"
    if model_dir.exists():
        shutil.rmtree(model_dir)  # a fresh fit: never mix models from an older run
    # random_state seeds AutoGluon's train/validation split only. The base models keep their own
    # default random_seed (0): AutoGluon would take a per-model random_seed in `hyperparameters`,
    # which this run did not set. automl_checks.ag_checks reads the seeds back from the models.
    predictor = TabularPredictor(label=LABEL, eval_metric="roc_auc", path=str(model_dir),
                                 log_to_file=True, learner_kwargs={"random_state": SEED})
    with timed(timings, "fit_s"):
        predictor.fit(train_df, time_limit=budget_s, presets="medium_quality", num_cpus=N_JOBS)
    finish_timings(timings)
    del train_df

    with timed(timings, "artifacts_s"):
        info = _autogluon_artifacts(predictor, X, y, names, model_dir, paths)
    best = predictor.model_best
    return FitResult(
        tool="autogluon", best_model=best, timings=timings, info=info,
        predict=lambda X_new: predictor.predict_proba(to_frame(X_new, names), as_multiclass=False).to_numpy(),
        interpretability=_ag_interpretability(best, info.get("ensemble_weights", {})),
    )


def _ag_interpretability(best: str, weights: dict[str, float]) -> str:
    members = [m for m, w in weights.items() if w > 0]
    if len(members) == 1:  # the "ensemble" puts all weight on one model, so that model is the predictor
        return (f"weighted ensemble with all weight on {members[0]}, i.e. one model; it can be explained "
                "like any single model of that type")
    if members:
        return (f"weighted ensemble of {len(members)} models ({', '.join(members)}); "
                "only model-agnostic (permutation) importance for the whole predictor")
    return f"single {best} model"


def _ensemble_weights(predictor: Any, best: str) -> dict[str, float]:
    """Base-model weights of a WeightedEnsemble winner ({} if the winner is a single model)."""
    if not best.startswith("WeightedEnsemble"):
        return {}
    model = predictor._trainer.load_model(best)  # no public accessor for the weights
    return {str(k): float(v) for k, v in model._get_model_weights().items()}


def _autogluon_artifacts(predictor: Any, X: np.ndarray, y: np.ndarray, names: list[str], model_dir: Path,
                         paths: AutomlPaths) -> dict[str, Any]:
    """Leaderboard (internal validation only), preprocessing notes, fit log and importances."""
    board = predictor.leaderboard(silent=True, extra_info=True)  # no data passed: validation scores only
    # fit_time is cumulative (an ensemble's includes its base models); fit_time_marginal is its own.
    keep = [c for c in ("model", "score_val", "eval_metric", "pred_time_val", "fit_time", "fit_time_marginal",
                        "stack_level", "can_infer", "fit_order", "num_features", "num_models") if c in board]
    board[keep].to_csv(_out(paths, "autogluon_leaderboard", ".csv"), index=False)

    log_file = model_dir / "logs" / "predictor_log.txt"
    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    if log_text:
        shutil.copyfile(log_file, _out(paths, "autogluon_fit_log", ".txt"))
    used = predictor.features()  # original features the models actually receive
    transformed = predictor.features(feature_stage="transformed")
    best = predictor.model_best
    weights = _ensemble_weights(predictor, best)
    info = predictor.info()
    notes = {
        "tool": "autogluon", "version": _version("autogluon.tabular"),
        "presets": "medium_quality",
        "best_model": best,
        "best_val_auc": float(board.loc[board["model"] == best, "score_val"].iloc[0]),
        "ensemble_weights": weights,
        "models_trained": board["model"].tolist(),
        "validation": "holdout split of train made by AutoGluon (no bagging or stacking in medium_quality)",
        "holdout_rows": _ag_holdout_rows(predictor),
        "n_features_in": len(names),
        "n_features_used": len(used),
        "dropped_features": sorted(set(names) - set(used)),
        # Dropped although not constant: is each one a copy of another column?
        "dropped_nonconstant_duplicate_of": duplicate_of(
            X, names, sorted(set(names) - set(used) - set(constant_columns(X, names)))),
        # Columns after AutoGluon's feature generators: any name not in the input is engineered.
        "n_features_transformed": len(transformed),
        "new_features_created": sorted(set(transformed) - set(names)),
        "constant_in_train": constant_columns(X, names),
        "missing_values_in_train": missing_values(X),
        "feature_types_after_preprocessing": predictor.feature_metadata.get_type_map_raw_counts()
        if hasattr(predictor.feature_metadata, "get_type_map_raw_counts") else None,
        "feature_generators": _ag_generators(predictor),
        "num_cpus": info.get("num_cpus"),
        "fit_log_highlights": ag_log_highlights(log_text),
    }
    # Seeds actually used, and how rare the dropped non-constant columns are (see automl_checks).
    notes["checks"] = ag_checks(predictor, X, y, names, sorted(notes["dropped_nonconstant_duplicate_of"]))
    notes["feature_importance"] = _ag_feature_importance(predictor, names, paths)
    _write_json(notes, _out(paths, "autogluon_preprocessing", ".json"))
    return {k: notes[k] for k in ("best_val_auc", "ensemble_weights", "models_trained", "holdout_rows",
                                  "n_features_used", "version", "feature_importance")}


def _ag_holdout_rows(predictor: Any) -> int | None:
    """Rows of AutoGluon's internal validation split (read back from its cached data)."""
    try:
        X_val, _ = predictor.load_data_internal(data="val")
        return int(len(X_val))
    except Exception:  # noqa: BLE001 - the count is a nice-to-have note, never worth failing a run
        return None


def _ag_generators(predictor: Any) -> list[str]:
    """Class names of the feature generators in AutoGluon's preprocessing pipeline, stage by stage."""
    pipeline = predictor._learner.feature_generator  # no public accessor; read-only use
    stages = getattr(pipeline, "generators", [])
    return [" + ".join(type(g).__name__ for g in stage) for stage in stages]


def _ag_feature_importance(predictor: Any, names: list[str], paths: AutomlPaths) -> dict[str, Any]:
    """Permutation importance on AutoGluon's internal validation rows (data=None), never a test set.

    data=None only works with feature_stage="transformed": AutoGluon keeps its validation rows
    after preprocessing, not the raw ones. Our features are all numeric, so preprocessing keeps
    their names; only the dropped columns are missing from the transformed stage.
    1. Per family: the 12 EMBER families shuffled as blocks (complete, AG_FI_FAMILY_SHUFFLES sets).
    2. Per feature: in chunks until AG_FI_TIME_LIMIT_S is used up; uncomputed features stay NaN and
       the coverage is recorded. (AutoGluon's own time_limit cannot stop inside the first shuffle
       set, which for 2,568 features takes far longer than the cap.)
    """
    stage = "transformed"
    used = set(predictor.features(feature_stage=stage))
    # Families in column order, taken from the names, so an extra column family (the Step 4 leak)
    # gets its own block; for the standard names this is exactly feature_groups() order.
    family_order = dict.fromkeys(family_of(n) for n in names)
    families = [(fam, [n for n in names if family_of(n) == fam and n in used]) for fam in family_order]
    families = [(fam, cols) for fam, cols in families if cols]
    start = time.perf_counter()
    fam_fi = predictor.feature_importance(data=None, feature_stage=stage, features=families,
                                          subsample_size=AG_FI_ROWS, num_shuffle_sets=AG_FI_FAMILY_SHUFFLES,
                                          silent=True)
    fam_fi.rename_axis("family").reset_index().to_csv(
        _out(paths, "autogluon_family_importance", ".csv"), index=False)
    family_s = time.perf_counter() - start

    features = [n for n in names if n in used]
    parts, start = [], time.perf_counter()
    for i in range(0, len(features), AG_FI_FEATURE_CHUNK):
        if time.perf_counter() - start > AG_FI_TIME_LIMIT_S:
            break
        parts.append(predictor.feature_importance(data=None, feature_stage=stage,
                                                  features=features[i:i + AG_FI_FEATURE_CHUNK],
                                                  subsample_size=AG_FI_ROWS, num_shuffle_sets=1, silent=True))
    fi = pd.concat(parts).reindex(features) if parts else pd.DataFrame(index=features)
    fi = fi.rename_axis("feature").reset_index()
    fi["family"] = fi["feature"].map(family_of)
    fi.sort_values("importance", ascending=False, na_position="last").to_csv(
        _out(paths, "autogluon_feature_importance", ".csv"), index=False)
    covered = int(fi["importance"].notna().sum()) if "importance" in fi else 0
    return {"data": "AutoGluon internal validation split (data=None, transformed features)",
            "rows_cap": AG_FI_ROWS,
            "family_shuffle_sets": AG_FI_FAMILY_SHUFFLES, "family_seconds": family_s,
            "feature_time_cap_s": AG_FI_TIME_LIMIT_S, "features_covered": covered,
            "features_total": len(features), "feature_seconds": time.perf_counter() - start}


# --- H2O ----------------------------------------------------------------------------------

def write_parquet(X: np.ndarray, y: np.ndarray | None, names: list[str], path: Path) -> Path:
    """Write X (and the label, if given) as parquet: H2O parses a file far faster than a pandas upload."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = to_frame(X, names)
    if y is not None:
        frame[LABEL] = y.astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def run_h2o(X: np.ndarray, y: np.ndarray, budget_s: int, paths: AutomlPaths,
            names: list[str] | None = None) -> FitResult:
    """H2OAutoML with its default algorithms (XGBoost is unavailable on Windows), within budget_s.

    names: column names of X, as in run_flaml().
    """
    import h2o
    from h2o.automl import H2OAutoML

    names, timings = names or feature_names(), {}
    model_dir = paths.models / f"h2o{RUN_SUFFIX}"
    data_dir = model_dir / "data"
    with timed(timings, "init_s"):
        h2o.init(max_mem_size=H2O_MAX_MEM, nthreads=N_JOBS)
        h2o.no_progress()
    with timed(timings, "convert_s"):
        train_path = write_parquet(X, y, names, data_dir / "train.parquet")
        train = h2o.import_file(str(train_path), destination_frame=f"train{RUN_SUFFIX}")
        train[LABEL] = train[LABEL].asfactor()  # a factor label makes H2O treat this as classification
    aml = H2OAutoML(max_runtime_secs=budget_s, seed=SEED, sort_metric="AUC", project_name=f"ember{RUN_SUFFIX}")
    with timed(timings, "fit_s"):
        aml.train(x=names, y=LABEL, training_frame=train)
    finish_timings(timings)

    with timed(timings, "artifacts_s"):
        info = _h2o_artifacts(h2o, aml, X, names, model_dir, paths)
    leader = aml.leader

    def predict(X_new: np.ndarray) -> np.ndarray:
        path = write_parquet(X_new, None, names, data_dir / f"score_{uuid.uuid4().hex[:8]}.parquet")
        frame = h2o.import_file(str(path))
        p_mal = leader.predict(frame).as_data_frame(use_multi_thread=True)["p1"].to_numpy(dtype=float)  # p1 = P(label "1")
        h2o.remove(frame)
        path.unlink()
        return p_mal

    def shutdown() -> None:
        h2o.cluster().shutdown()
        shutil.rmtree(data_dir, ignore_errors=True)  # the parquet copies are only needed while H2O runs

    return FitResult(tool="h2o", best_model=leader.model_id, timings=timings, info=info, predict=predict,
                     interpretability=_h2o_interpretability(info), cleanup=shutdown)


def _h2o_interpretability(info: dict[str, Any]) -> str:
    return h2o_interpretability(info["leader_algo"], info["base_models"], info["effective_base_models"])


def _h2o_artifacts(h2o: Any, aml: Any, X: np.ndarray, names: list[str], model_dir: Path,
                   paths: AutomlPaths) -> dict[str, Any]:
    """Leaderboard, event log, leader varimp, metalearner weights, and the leader saved to disk.

    The leaderboard metrics are those of H2O's leaderboard frame: CV metrics when H2O ran CV,
    otherwise (as here) the validation frame it split off train.
    """
    board = h2o.automl.get_leaderboard(aml, extra_columns="ALL").as_data_frame(use_multi_thread=True)
    board.to_csv(_out(paths, "h2o_leaderboard", ".csv"), index=False)
    events = aml.event_log.as_data_frame(use_multi_thread=True)
    events.to_csv(_out(paths, "h2o_event_log", ".csv"), index=False)
    leader = aml.leader
    leader_algo = str(board.loc[0, "algo"]).lower()
    base_models = _h2o_base_models(leader) if leader_algo == "stackedensemble" else []
    varimp_model = _h2o_varimp(h2o, board, names, paths)
    single = h2o.get_model(varimp_model) if varimp_model else None
    notes = {
        "tool": "h2o", "version": _version("h2o"),
        "leader": leader.model_id, "leader_algo": leader_algo,
        # Called leader_cv_auc in records logged before the rename (experiments.jsonl is never edited).
        "leader_leaderboard_auc": float(board.loc[0, "auc"]),
        "n_models": int(len(board)),
        "models_by_algo": board["algo"].value_counts().to_dict(),
        "base_models": base_models, "n_base_models": len(base_models),
        "varimp_model": varimp_model,
        "nfolds_used": _h2o_param(leader, "nfolds"),
        "frames": _h2o_frames(events),
        "validation": _h2o_validation_text(_h2o_param(leader, "nfolds"), _h2o_frames(events)),
        "n_features_in": len(names),
        "n_features_used_by_varimp_model": None if single is None else len(_h2o_model_features(single)),
        "dropped_features": None if single is None else sorted(set(names) - set(_h2o_model_features(single))),
        # H2O models train on the raw columns: a varimp name not in the input would be engineered.
        "new_features_created": _h2o_new_features(paths, names),
        "constant_in_train": constant_columns(X, names),
        "missing_values_in_train": missing_values(X),
        "ignore_const_cols": None if single is None else _h2o_param(single, "ignore_const_cols"),
        "categorical_encoding": None if single is None else _h2o_param(single, "categorical_encoding"),
        "event_log_highlights": _h2o_event_highlights(events),
        "saved_model": _h2o_save(h2o, leader, model_dir),
        "checks": h2o_checks(leader),  # metalearner coefficients: which base models really count
    }
    _write_json(notes, _out(paths, "h2o_preprocessing", ".json"))
    info = {k: notes[k] for k in ("leader_algo", "leader_leaderboard_auc", "n_models", "models_by_algo",
                                  "base_models", "n_base_models", "varimp_model", "version", "saved_model")}
    return {**info, "effective_base_models": notes["checks"]["effective_base_models"]}


def _h2o_frames(events: pd.DataFrame) -> dict[str, int]:
    """Rows of the frames H2O AutoML used ({"training": n, "validation": n, ...}), from its event log."""
    frames: dict[str, int] = {}
    for message in events.get("message", pd.Series(dtype=str)).astype(str):
        match = re.match(r"(\w+) frame: .*rows: (\d+)", message)
        if match:
            frames[match.group(1)] = int(match.group(2))
    return frames


def _h2o_validation_text(nfolds: Any, frames: dict[str, int]) -> str:
    """How H2O validated: k-fold CV, or (its automatic choice for large data / short budgets) a split of train."""
    if isinstance(nfolds, int) and nfolds >= 2:
        return f"{nfolds}-fold cross-validation on train; leaderboard metrics are CV metrics"
    sizes = ", ".join(f"{name} {rows:,}" for name, rows in frames.items() if name != "leaderboard")
    return (f"no cross-validation (H2O's automatic choice, nfolds={nfolds}): it split train into frames of "
            f"{sizes} rows; models train on the training frame only and the leaderboard is scored on the "
            f"validation frame")


def _h2o_model_features(model: Any) -> list[str]:
    """Input columns a trained H2O model kept (its output names minus the response)."""
    return [n for n in model._model_json["output"]["names"] if n != LABEL]


def _h2o_param(model: Any, name: str) -> Any:
    """The value H2O actually used for a parameter (None if the model has no such parameter)."""
    return model.params.get(name, {}).get("actual")


def _h2o_base_models(leader: Any) -> list[str]:
    """Names of the base models a stacked-ensemble leader combines."""
    actual = _h2o_param(leader, "base_models") or []
    return [m["name"] if isinstance(m, dict) else str(m) for m in actual]


def _h2o_varimp(h2o: Any, board: pd.DataFrame, names: list[str], paths: AutomlPaths) -> str | None:
    """Varimp of the best model that has one (stacked ensembles do not); returns that model's id."""
    for model_id in board["model_id"]:
        model = h2o.get_model(model_id)
        varimp = model.varimp(use_pandas=True)
        if varimp is None or len(varimp) == 0:
            continue
        varimp = varimp.rename(columns={"variable": "feature"})
        varimp["family"] = varimp["feature"].map(family_of)
        varimp.insert(0, "model_id", model_id)
        varimp.to_csv(_out(paths, "h2o_varimp", ".csv"), index=False)
        return str(model_id)
    return None


def _h2o_new_features(paths: AutomlPaths, names: list[str]) -> list[str] | None:
    """Features in the saved varimp that are not input columns (None if no varimp was saved)."""
    path = _out(paths, "h2o_varimp", ".csv")
    if not path.exists():
        return None
    return sorted(set(pd.read_csv(path)["feature"].astype(str)) - set(names))


def _h2o_event_highlights(events: pd.DataFrame) -> list[str]:
    """Event-log messages about skipped algorithms, time allocation and the final leader."""
    pattern = re.compile(r"[Ss]kip|not available|XGBoost|time|budget|[Ll]eader|[Ee]xclud", re.IGNORECASE)
    messages = events["message"].astype(str) if "message" in events else pd.Series(dtype=str)
    return [m for m in messages if pattern.search(m)][:60]


def _h2o_save(h2o: Any, leader: Any, model_dir: Path) -> dict[str, str | None]:
    """Save the leader as a MOJO (portable) and as an H2O binary model (needs the same H2O version)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str | None] = {"mojo": None, "binary": None}
    try:
        saved["mojo"] = leader.download_mojo(path=str(model_dir))
    except Exception as exc:  # noqa: BLE001 - recorded in the notes; the binary save below still runs
        saved["mojo_error"] = repr(exc)
    saved["binary"] = h2o.save_model(leader, path=str(model_dir), force=True)
    return saved


RUNNERS: dict[str, Callable[..., FitResult]] = {  # (X, y, budget_s, paths, names=None) -> FitResult
    "flaml": run_flaml, "autogluon": run_autogluon, "h2o": run_h2o,
}


# --- Final scoring: the ONLY place the test sets are read ---------------------------------

def _save_predictions(method: str, eval_id: str, scored: dict[str, tuple[np.ndarray, np.ndarray]],
                      paths: AutomlPaths) -> Path:
    """Same file format as tune's saved predictions, so tables.py bootstraps both the same way."""
    paths.predictions.mkdir(parents=True, exist_ok=True)
    path = paths.predictions / f"{method}{RUN_SUFFIX}.npz"
    arrays = {}
    for name, (y_test, p_mal) in scored.items():
        arrays[f"{name}_y"] = np.asarray(y_test, dtype=np.int8)
        arrays[f"{name}_p"] = np.asarray(p_mal, dtype=np.float64)
    np.savez_compressed(path, eval_id=eval_id, **arrays)
    return path


def metrics_with_ci(y: np.ndarray, p_mal: np.ndarray, seed: int) -> dict[str, float]:
    """compute_metrics() plus 95% bootstrap CIs of AUC and TPR@1%FPR (same seed as the tables)."""
    metrics = compute_metrics(y, p_mal)
    samples = paired_bootstrap(y, {"m": p_mal}, seed=seed)["m"]
    for name in CI_METRICS:
        metrics[f"{name}_ci_low"], metrics[f"{name}_ci_high"] = percentile_ci(samples[name])
    return metrics


def evaluate_and_log(fit: FitResult, budget_s: int, train_rows: int,
                     paths: AutomlPaths = AutomlPaths(),
                     timing_check: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score the finished model on both test sets, save the scores and log one final_eval record.

    timing_check is Heartbeat.report() of the fit; a record with timing_valid=False stays in the log
    (nothing is deleted) but the tables skip it, because its timed budget included sleep.
    """
    # The tables' bootstrap seeds, so the CIs logged here equal those in the comparison table.
    # Imported here, not at the top: tables.py pulls in matplotlib, which the runners do not need.
    from src.tables import BOOTSTRAP_SEEDS

    method = f"automl_{fit.tool}"
    scored, metrics = {}, {}
    with timed(fit.timings, "score_s"):
        for name in TEST_SETS:
            X_test, y_test, _ = load_split(name)  # loaded only now, after the fit has finished
            scored[name] = (y_test, fit.predict(X_test))
            metrics[name] = metrics_with_ci(*scored[name], seed=BOOTSTRAP_SEEDS[name])
    eval_id = f"{RUN_ID}-{method}-{uuid.uuid4().hex[:6]}"
    record = {
        "step": "automl", "kind": "final_eval", "method": method, "tool": fit.tool,
        "tool_version": fit.info.get("version"), "eval_id": eval_id,
        "best_model": fit.best_model, "budget_s": budget_s, **fit.timings,
        "train_rows": train_rows, "n_features": len(feature_names()), "seed": SEED, "n_jobs": N_JOBS,
        "metric": "roc_auc", "interpretability": fit.interpretability, "info": fit.info,
        **(timing_check or {}), **metrics, "predictions_path": _save_predictions(method, eval_id, scored, paths),
    }
    log_experiment(record, path=paths.log)
    return record
