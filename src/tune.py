"""Step 2: baselines and manual hyper-parameter tuning of LightGBM (Grid, Random, Optuna TPE).

Equal budget
------------
"Equal budget" means every tuning method gets
  * the same number of configurations (TUNE_TRIALS),
  * evaluated on the same persisted CV folds (get_cv_folds(subset=True)),
  * over the same data (the persisted tuning subset of `train`, tune_subset_indices()),
  * with the same model, the same fixed LightGBM settings (BASE_PARAMS) and the same
    n_estimators cap (MAX_ROUNDS) with early stopping on each fold's validation part.
Wall-clock time is measured and reported, but it is NOT capped: a time cap would make the
result depend on machine load, while a trial count is exactly reproducible.
Caveat for efficiency comparisons: Optuna's TPE sampler starts with TPE_STARTUP_TRIALS = 10
random trials drawn with the same seed as Random Search, so the first 10 configurations of the
two methods are identical. They only differ from trial 11 on, i.e. in 26 of the 36 trials.

Test-set discipline
-------------------
The only function in this module that reads test_random / test_future is final_evaluation().
The search code (run_search and everything it calls) only ever sees the tuning subset of
`train`, so test scores cannot influence which configuration is chosen.

Refit strategy (REFIT_VERSION 2)
--------------------------------
The best configuration of each search is refit once on ALL rows of `train` (no hold-out), with a
FIXED number of boosting rounds taken from the search itself:
    n_estimators = round(mean of the best trial's fold best_iterations)       (refit_rounds())
Why a fixed count from CV:
  * Version 1 refit with early stopping (patience 50) on a 10% hold-out of `train`. That was
    noisy: it stopped the grid winner after 187 trees although its CV folds peaked after 719 on
    average. The test score then measured the noisy refit, not the tuner.
  * The CV best_iterations are what the tuner actually measured for that configuration.
Why no scaling for the larger refit set (160k rows vs 40k per CV training fold):
  * The tuning logs cannot give the right factor, because every fold in them has the same size.
    (It could be measured with an extra early-stopped CV of each winner on larger folds, e.g.
    data.get_cv_folds(subset=False); we did not run that.) Linear scaling (x4) would be a guess,
    and it would ask for 2,400-3,000 trees, beyond the MAX_ROUNDS = 2000 that any CV fold was
    ever allowed to explore.
  * Near its peak the validation AUC curve of boosting is flat: one configuration's fold
    best_iterations often differ by hundreds of rounds (see tables/tuning_trials.csv), so we
    assume being off by some rounds costs little AUC.
  * ASSUMPTION, not tested here: more data moves the optimum towards more trees. If it holds,
    the unscaled count gives a slightly smaller model than optimal. The rule is the same for
    every tuner, so the comparison between tuners stays fair either way.
Every final_eval record carries refit_version. A newer version SUPERSEDES older ones: old records
stay in the log, tables.py reports only the latest, and refuses to mix versions across methods.

Flow (driven by scripts/run_tuning.py)
--------------------------------------
1. run_baselines(): LightGBM with library defaults and a scaled LogisticRegression; both are
   cross-validated on the tuning folds (so they get a CV AUC like the tuners), refit on
   `train` and scored on both test sets.
2. run_search(method): evaluates TUNE_TRIALS configurations and logs every trial.
3. refit_and_evaluate(): refits the best configuration and scores it on both test sets.
   With --refit-only, search_from_log() rebuilds a finished search from the log instead of
   searching again (the searches take hours; the refits take minutes).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from config import EXPERIMENT_LOG, MODELS, N_JOBS, PREDICTIONS, QUICK, RUN_SUFFIX, SEED, TUNE_TRIALS
from src.eval import compute_metrics
from src.logging_utils import RUN_ID, log_experiment, read_records

Folds = list[tuple[np.ndarray, np.ndarray]]
Params = dict[str, Any]
Predictor = Callable[[np.ndarray], np.ndarray]  # X -> P(malicious)

TUNING_METHODS: tuple[str, ...] = ("grid", "random", "optuna")

# --- Fixed training settings (shared by every method) ------------------------------------
MAX_ROUNDS: int = 2000             # n_estimators cap; early stopping normally ends far earlier
EARLY_STOPPING_ROUNDS: int = 50    # stop when validation AUC has not improved for 50 rounds
TPE_STARTUP_TRIALS: int = 10       # Optuna's default, made explicit: random trials before TPE starts
# Bump when the way final models are fit changes; tables.py reports only the latest version.
# 1 = early stopping on a 10% hold-out of train, 2 = all of train with fixed rounds from CV.
REFIT_VERSION: int = 2
LOGREG_MAX_ITER: int = 2000
# None = fit LogReg on the full train split. Set an int to subsample if it is too slow;
# the number of rows actually used is always logged.
LOGREG_MAX_ROWS: int | None = None

# Settings every LightGBM model gets, tuned or not.
# deterministic + force_col_wise + a fixed thread count make reruns bit-identical on the same
# machine; `seed` derives LightGBM's bagging, feature-fraction and data seeds from SEED.
BASE_PARAMS: Params = {
    "objective": "binary",
    "metric": "auc",
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": N_JOBS,
    "verbosity": -1,
    # Lets min_child_samples change between trials while the binned fold Datasets are reused.
    "feature_pre_filter": False,
}
# All 2568 columns are treated as numeric (no categorical_feature): it is the same matrix the
# AutoML tools get, so every method is compared on identical inputs.

# --- One search space for all three tuners ------------------------------------------------
# name -> (low, high, log scale?). Random and Optuna sample from exactly these ranges.
SEARCH_SPACE: dict[str, tuple[float, float, bool]] = {
    "num_leaves": (16, 512, True),
    "learning_rate": (0.02, 0.3, True),
    "min_child_samples": (5, 200, True),
    "feature_fraction": (0.3, 1.0, False),
    "bagging_fraction": (0.5, 1.0, False),
    "lambda_l2": (1e-8, 10.0, True),  # 1e-8 is effectively "no L2" (LightGBM's default is 0)
}
INT_PARAMS: frozenset[str] = frozenset({"num_leaves", "min_child_samples"})

# Grid: 3 x 3 x 2 x 2 = 36 points over the four parameters that usually matter most.
# The four varied values lie inside SEARCH_SPACE. The other two stay at LightGBM's defaults
# (bagging off, lambda_l2 = 0.0). Note 0.0 is just below SEARCH_SPACE's lambda_l2 floor of 1e-8,
# which Random/Optuna need because a log-scale range cannot start at 0; the two are equivalent.
GRID: dict[str, list[float]] = {
    "num_leaves": [31, 127, 511],
    "learning_rate": [0.03, 0.1, 0.3],
    "min_child_samples": [20, 100],
    "feature_fraction": [0.5, 1.0],
}
GRID_FIXED: Params = {"bagging_fraction": 1.0, "lambda_l2": 0.0}


@dataclass(frozen=True)
class OutputPaths:
    """Where logs and models go. Tests and smoke runs point these at a temp dir."""

    log: Path = EXPERIMENT_LOG
    models: Path = MODELS
    predictions: Path = PREDICTIONS


@dataclass(frozen=True)
class TuningData:
    """The tuning subset of train and its CV folds (row positions within the subset)."""

    X: np.ndarray
    y: np.ndarray
    folds: Folds


@dataclass(frozen=True)
class FoldData:
    """One CV fold, binned once for LightGBM. X_val/y_val are kept for scoring."""

    dtrain: lgb.Dataset
    dvalid: lgb.Dataset
    X_val: np.ndarray
    y_val: np.ndarray


@dataclass(frozen=True)
class CvResult:
    """Cross-validation outcome of one configuration."""

    fold_aucs: list[float]
    best_iterations: list[int]
    fit_seconds: float

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def std_auc(self) -> float:
        return float(np.std(self.fold_aucs))


@dataclass
class SearchResult:
    """Everything one search produced."""

    method: str
    search_id: str
    folds_hash: str
    trials: list[tuple[Params, CvResult]] = field(default_factory=list)
    search_seconds: float = 0.0

    @property
    def best(self) -> tuple[Params, CvResult]:
        # max() keeps the first maximum, so ties go to the earlier trial (deterministic).
        return max(self.trials, key=lambda t: t[1].mean_auc)


# --- Data loading: the only place this module touches src.data ----------------------------
# src.data is imported inside the loaders so tune.py can be imported (and unit-tested with
# monkeypatched loaders) without the vectorised data cache existing.

def load_train() -> tuple[np.ndarray, np.ndarray]:
    """The full train split (X, y)."""
    from src import data

    X, y, _meta = data.load_split("train")
    return X, y


def load_tuning_data(X_train: np.ndarray, y_train: np.ndarray) -> TuningData:
    """The persisted tuning subset of train and its persisted folds."""
    from src import data

    subset = data.tune_subset_indices()
    folds = data.get_cv_folds(subset=True)
    tuning = TuningData(X_train[subset], y_train[subset], folds)
    check_folds(tuning.folds, len(tuning.y))
    return tuning


def load_test_sets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Both test sets. Called ONLY from final_evaluation()."""
    from src import data

    out = {}
    for name in ("test_random", "test_future"):
        X, y, _meta = data.load_split(name)
        out[name] = (X, y)
    return out


def check_folds(folds: Folds, n_rows: int) -> None:
    """Fail loudly if the folds do not partition the tuning subset (a silent bug would bias CV)."""
    all_val = np.concatenate([val for _tr, val in folds])
    if len(all_val) != n_rows or len(np.unique(all_val)) != n_rows:
        raise ValueError("CV validation folds must cover every tuning row exactly once")
    for tr, val in folds:
        if np.intersect1d(tr, val).size:
            raise ValueError("A CV fold has rows in both its train and validation part")


def folds_fingerprint(folds: Folds) -> str:
    """Short hash of the fold indices, logged so a reader can check all methods used the same folds."""
    h = hashlib.sha256()
    for tr, val in folds:
        h.update(np.asarray(tr, dtype=np.int64).tobytes())
        h.update(np.asarray(val, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


# --- Cross-validation of one LightGBM configuration ---------------------------------------

def lgb_params(params: Params) -> Params:
    """Full LightGBM parameter dict: the shared base settings plus one configuration."""
    full = {**BASE_PARAMS, **params}
    if "bagging_fraction" in params:
        # bagging_fraction only takes effect when bagging_freq > 0 (bag every iteration).
        full.setdefault("bagging_freq", 1)
    return full


def build_fold_data(tuning: TuningData) -> list[FoldData]:
    """Bin each fold once and reuse it for every trial of a search.

    Binning 2568 features is a large share of a trial's cost. Bin edges depend only on the
    fold's training rows, not on the hyper-parameters, so reuse changes no result.
    """
    fold_data = []
    for tr, val in tuning.folds:
        dtrain = lgb.Dataset(tuning.X[tr], tuning.y[tr], params=lgb_params({}))
        X_val, y_val = tuning.X[val], tuning.y[val]
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain, params=lgb_params({}))  # same params: no override warning
        fold_data.append(FoldData(dtrain, dvalid, X_val, y_val))
    return fold_data


def cross_validate_lgbm(params: Params, fold_data: list[FoldData], early_stopping: bool = True) -> CvResult:
    """Validation ROC-AUC of one configuration on each shared fold.

    With early stopping each fold trains up to MAX_ROUNDS and stops on its own validation part.
    That makes the fold AUC slightly optimistic (the stopping point saw the validation labels),
    but the bias is identical for every method, so the comparison stays fair.
    Without early stopping (the library-default baseline) LightGBM's default 100 rounds are used.
    """
    start = time.perf_counter()
    aucs, iters = [], []
    for fold in fold_data:
        if early_stopping:
            booster = lgb.train(
                lgb_params(params), fold.dtrain, num_boost_round=MAX_ROUNDS,
                valid_sets=[fold.dvalid], valid_names=["valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            n_iter = booster.best_iteration
        else:
            booster = lgb.train(lgb_params(params), fold.dtrain)
            n_iter = booster.current_iteration()
        p_val = booster.predict(fold.X_val, num_iteration=n_iter)
        aucs.append(float(roc_auc_score(fold.y_val, p_val)))
        iters.append(int(n_iter))
    return CvResult(aucs, iters, time.perf_counter() - start)


def folds_at_round_cap(best_iterations: list[int], max_rounds: int = MAX_ROUNDS) -> int:
    """How many folds ended at the round cap instead of by early stopping.

    Early stopping fires EARLY_STOPPING_ROUNDS rounds after the best round. A best round later
    than max_rounds - EARLY_STOPPING_ROUNDS therefore means the cap ended training, and a higher
    cap might have given that configuration a higher CV score.
    """
    return sum(int(it) > max_rounds - EARLY_STOPPING_ROUNDS for it in best_iterations)


# --- The three tuners ---------------------------------------------------------------------

def grid_configs(n_trials: int = TUNE_TRIALS) -> list[Params]:
    """The grid as a list of full configurations, in itertools.product order.

    If n_trials is smaller than the grid (QUICK mode), a seeded random subset of grid points is
    used, kept in grid order, so the method still searches the same grid.
    """
    names = list(GRID)
    full = [{**dict(zip(names, values)), **GRID_FIXED} for values in itertools.product(*GRID.values())]
    if n_trials > len(full):
        raise ValueError(f"Grid has {len(full)} points but {n_trials} trials were requested")
    if n_trials < len(full):
        keep = np.sort(np.random.default_rng(SEED).choice(len(full), size=n_trials, replace=False))
        full = [full[i] for i in keep]
    return full


def suggest_params(trial: optuna.Trial) -> Params:
    """Sample one configuration from SEARCH_SPACE (used by both Random Search and Optuna)."""
    params: Params = {}
    for name, (low, high, log) in SEARCH_SPACE.items():
        if name in INT_PARAMS:
            params[name] = trial.suggest_int(name, int(low), int(high), log=log)
        else:
            params[name] = trial.suggest_float(name, low, high, log=log)
    return params


def make_sampler(method: str) -> optuna.samplers.BaseSampler:
    """Random Search and Optuna share one objective; only the sampler differs.

    Note: TPESampler starts with n_startup_trials=10 random trials drawn by an internal
    RandomSampler with the same seed, so Optuna's first 10 configurations equal Random Search's
    first 10. The methods only diverge once TPE starts modelling (trial 11 onwards).
    """
    if method == "random":
        return optuna.samplers.RandomSampler(seed=SEED)
    if method == "optuna":
        return optuna.samplers.TPESampler(n_startup_trials=TPE_STARTUP_TRIALS, seed=SEED)
    raise ValueError(f"No sampler for method {method!r}")


def run_search(method: str, tuning: TuningData, n_trials: int = TUNE_TRIALS,
               paths: OutputPaths = OutputPaths()) -> SearchResult:
    """Run one tuning method for n_trials configurations on the shared folds; log every trial."""
    if method not in TUNING_METHODS:
        raise ValueError(f"Unknown tuning method {method!r}; choose from {TUNING_METHODS}")
    result = SearchResult(method, search_id=uuid.uuid4().hex[:12], folds_hash=folds_fingerprint(tuning.folds))
    start = time.perf_counter()
    fold_data = build_fold_data(tuning)  # inside the timer: every method pays the same binning cost

    def evaluate(params: Params) -> float:
        cv = cross_validate_lgbm(params, fold_data)
        result.trials.append((params, cv))
        _log_trial(result, len(result.trials) - 1, params, cv, len(tuning.y), paths)
        return cv.mean_auc

    if method == "grid":
        for params in grid_configs(n_trials):
            evaluate(params)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=make_sampler(method))
        study.optimize(lambda trial: evaluate(suggest_params(trial)), n_trials=n_trials)

    result.search_seconds = time.perf_counter() - start
    _log_search(result, tuning, paths)
    return result


def _log_trial(result: SearchResult, trial_no: int, params: Params, cv: CvResult,
               n_rows: int, paths: OutputPaths) -> None:
    log_experiment({
        "step": "tuning", "kind": "tune_trial", "method": result.method, "search_id": result.search_id,
        "trial": trial_no, "params": params,
        "cv_auc_mean": cv.mean_auc, "cv_auc_std": cv.std_auc, "cv_auc_folds": cv.fold_aucs,
        "best_iterations": cv.best_iterations, "fit_seconds": cv.fit_seconds,
        "max_rounds": MAX_ROUNDS, "n_folds_at_round_cap": folds_at_round_cap(cv.best_iterations),
        "n_rows": n_rows, "folds_hash": result.folds_hash,
    }, path=paths.log)


def _log_search(result: SearchResult, tuning: TuningData, paths: OutputPaths) -> None:
    best_params, best_cv = result.best
    log_experiment({
        "step": "tuning", "kind": "tune_search", "method": result.method, "search_id": result.search_id,
        "n_trials": len(result.trials), "search_seconds": result.search_seconds,
        "best_trial": result.trials.index(result.best), "best_params": best_params,
        "best_cv_auc_mean": best_cv.mean_auc, "best_cv_auc_std": best_cv.std_auc,
        "n_rows": len(tuning.y), "n_features": tuning.X.shape[1], "folds_hash": result.folds_hash,
    }, path=paths.log)


def search_from_log(method: str, log: Path = EXPERIMENT_LOG) -> SearchResult:
    """Rebuild the latest finished search of `method` (current quick mode) from its logged trials.

    Lets a changed refit rule be applied without re-running a multi-hour search. It fails
    loudly if the trials are incomplete or do not reproduce the logged winner.
    """
    records = [r for r in read_records(log) if r.get("step") == "tuning" and r.get("quick") == QUICK]
    searches = [r for r in records if r["kind"] == "tune_search" and r["method"] == method]
    if not searches:
        raise RuntimeError(f"No finished {method!r} search (quick={QUICK}) in {log}")
    search = searches[-1]  # the log is append-only, so the last record is the newest
    trials = sorted((r for r in records if r["kind"] == "tune_trial" and r["search_id"] == search["search_id"]),
                    key=lambda r: r["trial"])
    if [r["trial"] for r in trials] != list(range(search["n_trials"])):
        raise RuntimeError(f"Search {search['search_id']} has missing or repeated trials in the log")
    result = SearchResult(method, search["search_id"], search["folds_hash"], search_seconds=search["search_seconds"])
    result.trials = [(r["params"], CvResult(r["cv_auc_folds"], r["best_iterations"], r["fit_seconds"]))
                     for r in trials]
    if result.trials.index(result.best) != search["best_trial"] or result.best[0] != search["best_params"]:
        raise RuntimeError(f"Logged trials of search {search['search_id']} do not reproduce its logged winner")
    return result


# --- Final fits and the ONLY test-set evaluation ------------------------------------------

def refit_rounds(cv: CvResult) -> int:
    """Fixed number of boosting rounds for the refit: the mean fold best_iteration (see Refit strategy)."""
    return max(1, int(round(float(np.mean(cv.best_iterations)))))


def fit_lgbm_fixed_rounds(params: Params, X: np.ndarray, y: np.ndarray, n_rounds: int) -> lgb.Booster:
    """Fit on every row of X with exactly n_rounds trees. No early stopping, so no hold-out is needed."""
    dtrain = lgb.Dataset(X, y, params=lgb_params({}))
    return lgb.train(lgb_params(params), dtrain, num_boost_round=n_rounds)


def _save_predictions(method: str, eval_id: str, scored: dict[str, tuple[np.ndarray, np.ndarray]],
                      paths: OutputPaths) -> Path:
    """Save labels and scores of both test sets, so tables.py can bootstrap CIs without refitting.

    eval_id ties the file to its final_eval record; tables.py refuses a file whose id does not match.
    """
    paths.predictions.mkdir(parents=True, exist_ok=True)
    path = paths.predictions / f"{method}{RUN_SUFFIX}.npz"
    arrays = {}
    for name, (y_test, p_mal) in scored.items():
        arrays[f"{name}_y"] = np.asarray(y_test, dtype=np.int8)
        arrays[f"{name}_p"] = np.asarray(p_mal, dtype=np.float64)
    np.savez_compressed(path, eval_id=eval_id, **arrays)
    return path


def final_evaluation(method: str, predict: Predictor, record: dict[str, Any],
                     paths: OutputPaths = OutputPaths()) -> dict[str, Any]:
    """Score a finished model on test_random and test_future, save the scores, log one final_eval record.

    This is the only function that loads the test sets.
    """
    scored, metrics = {}, {}
    for name, (X_test, y_test) in load_test_sets().items():
        scored[name] = (y_test, predict(X_test))
        metrics[name] = compute_metrics(*scored[name])
    eval_id = f"{RUN_ID}-{method}-{uuid.uuid4().hex[:6]}"
    predictions_path = _save_predictions(method, eval_id, scored, paths)
    full = {"step": "tuning", "kind": "final_eval", "method": method, "refit_version": REFIT_VERSION,
            "eval_id": eval_id, **record, **metrics, "predictions_path": predictions_path}
    log_experiment(full, path=paths.log)
    return full


def _save_lgbm(booster: lgb.Booster, method: str, params: Params, paths: OutputPaths) -> Path:
    """Save the booster as text plus its full parameter dict as JSON."""
    paths.models.mkdir(parents=True, exist_ok=True)
    model_path = paths.models / f"lgbm_{method}{RUN_SUFFIX}.txt"
    booster.save_model(str(model_path))  # all trees: final models are fit without early stopping
    params_path = model_path.with_name(f"lgbm_{method}{RUN_SUFFIX}_params.json")
    params_path.write_text(json.dumps(
        {"params": lgb_params(params), "num_trees": booster.num_trees()}, indent=2), encoding="utf-8")
    return model_path


def refit_and_evaluate(result: SearchResult, X_train: np.ndarray, y_train: np.ndarray,
                       paths: OutputPaths = OutputPaths()) -> dict[str, Any]:
    """Refit a search's best configuration on train, save it, and score both test sets."""
    best_params, best_cv = result.best
    n_rounds = refit_rounds(best_cv)
    start = time.perf_counter()
    booster = fit_lgbm_fixed_rounds(best_params, X_train, y_train, n_rounds)
    fit_seconds = time.perf_counter() - start
    model_path = _save_lgbm(booster, result.method, best_params, paths)
    record = {
        "model": "lightgbm", "search_id": result.search_id, "params": best_params,
        "cv_auc_mean": best_cv.mean_auc, "cv_auc_std": best_cv.std_auc,
        "n_trials": len(result.trials), "search_seconds": result.search_seconds,
        "fit_seconds": fit_seconds, "train_rows": len(y_train),
        "refit_strategy": "all of train, fixed n_estimators = round(mean CV fold best_iteration), no early stopping",
        "refit_n_estimators": n_rounds, "num_trees": booster.num_trees(),
        "cv_best_iterations": best_cv.best_iterations,
        "cv_mean_best_iteration": float(np.mean(best_cv.best_iterations)),
        "cv_folds_at_round_cap": folds_at_round_cap(best_cv.best_iterations),
        "folds_hash": result.folds_hash, "model_path": model_path,
    }
    return final_evaluation(result.method, lambda X: booster.predict(X), record, paths)


# --- Baselines ----------------------------------------------------------------------------

def run_lgbm_default(X_train: np.ndarray, y_train: np.ndarray, tuning: TuningData,
                     paths: OutputPaths = OutputPaths()) -> dict[str, Any]:
    """LightGBM with library defaults (100 trees, 31 leaves, lr 0.1): CV, fit on all of train, score."""
    cv = cross_validate_lgbm({}, build_fold_data(tuning), early_stopping=False)
    start = time.perf_counter()
    booster = lgb.train(lgb_params({}), lgb.Dataset(X_train, y_train, params=lgb_params({})))
    fit_seconds = time.perf_counter() - start
    model_path = _save_lgbm(booster, "default", {}, paths)
    record = {
        "model": "lightgbm", "search_id": None, "params": {},
        "cv_auc_mean": cv.mean_auc, "cv_auc_std": cv.std_auc, "cv_seconds": cv.fit_seconds,
        "n_trials": None, "search_seconds": None, "fit_seconds": fit_seconds,
        "train_rows": len(y_train), "refit_strategy": "library defaults on all of train, no early stopping",
        "folds_hash": folds_fingerprint(tuning.folds), "model_path": model_path,
    }
    return final_evaluation("lgbm_default", lambda X: booster.predict(X), record, paths)


def make_logreg() -> Pipeline:
    """Standardise then logistic regression: a linear sanity floor for the tree models."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=SEED))


def _fit_logreg(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, bool]:
    """Fit a LogReg pipeline; return it and whether the solver converged (flagged, not hidden)."""
    model = make_logreg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # recorded via `converged` instead
        model.fit(X, y)
    converged = bool(model[-1].n_iter_.max() < LOGREG_MAX_ITER)
    return model, converged


def run_logreg(X_train: np.ndarray, y_train: np.ndarray, tuning: TuningData,
               paths: OutputPaths = OutputPaths()) -> dict[str, Any]:
    """LogisticRegression on standardised features: CV on the tuning folds, fit on train, score."""
    start = time.perf_counter()
    fold_aucs, fold_converged = [], []
    for tr, val in tuning.folds:
        model, converged = _fit_logreg(tuning.X[tr], tuning.y[tr])
        fold_aucs.append(float(roc_auc_score(tuning.y[val], model.predict_proba(tuning.X[val])[:, 1])))
        fold_converged.append(converged)
    cv_seconds = time.perf_counter() - start

    rows = np.arange(len(y_train))
    if LOGREG_MAX_ROWS is not None and LOGREG_MAX_ROWS < len(rows):
        rows, _ = train_test_split(rows, train_size=LOGREG_MAX_ROWS, stratify=y_train, random_state=SEED)
        X_fit, y_fit = X_train[rows], y_train[rows]
    else:
        X_fit, y_fit = X_train, y_train  # no fancy-index copy of the full 1.6 GB matrix
    start = time.perf_counter()
    model, converged = _fit_logreg(X_fit, y_fit)
    fit_seconds = time.perf_counter() - start

    paths.models.mkdir(parents=True, exist_ok=True)
    model_path = paths.models / f"logreg{RUN_SUFFIX}.joblib"
    joblib.dump(model, model_path)
    record = {
        "model": "logistic_regression", "search_id": None,
        "params": {"C": 1.0, "max_iter": LOGREG_MAX_ITER, "scaler": "StandardScaler"},
        "cv_auc_mean": float(np.mean(fold_aucs)), "cv_auc_std": float(np.std(fold_aucs)),
        "cv_auc_folds": fold_aucs, "cv_seconds": cv_seconds, "cv_converged": fold_converged,
        "n_trials": None, "search_seconds": None, "fit_seconds": fit_seconds,
        "train_rows": len(rows), "subsampled": len(rows) < len(y_train), "converged": converged,
        "refit_strategy": "fit on all of train" if len(rows) == len(y_train) else f"stratified subsample of {len(rows)} train rows",
        "folds_hash": folds_fingerprint(tuning.folds), "model_path": model_path,
    }
    return final_evaluation("logreg", lambda X: model.predict_proba(X)[:, 1], record, paths)


def run_baselines(X_train: np.ndarray, y_train: np.ndarray, tuning: TuningData,
                  paths: OutputPaths = OutputPaths()) -> list[dict[str, Any]]:
    """Both baselines, one after the other."""
    return [run_lgbm_default(X_train, y_train, tuning, paths), run_logreg(X_train, y_train, tuning, paths)]
