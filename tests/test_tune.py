"""Fast checks of the Step 2 tuning code on tiny synthetic data (no real data needed).

Data loading is monkeypatched, and every log, model, table and figure goes to pytest's tmp_path,
so these tests never touch data/, models/ or reports/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import without installing the project

from config import SEED  # noqa: E402
from src import tables, tune  # noqa: E402
from src.eval import compute_metrics, paired_bootstrap, percentile_ci, threshold_free_metrics  # noqa: E402
from src.logging_utils import read_experiments  # noqa: E402

N_TRIALS = 3
N_TRAIN, N_TEST = 600, 200


@pytest.fixture(scope="module")
def synthetic() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A small, learnable but not trivial binary problem split into train and two test sets."""
    X, y = make_classification(n_samples=N_TRAIN + 2 * N_TEST, n_features=30, n_informative=8,
                               flip_y=0.05, random_state=SEED)
    X, y = X.astype(np.float32), y.astype(np.int8)
    return {
        "train": (X[:N_TRAIN], y[:N_TRAIN]),
        "test_random": (X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]),
        "test_future": (X[N_TRAIN + N_TEST:], y[N_TRAIN + N_TEST:]),
    }


@pytest.fixture
def paths(monkeypatch: pytest.MonkeyPatch, synthetic, tmp_path: Path) -> tune.OutputPaths:
    """Patch the three data loaders and send all outputs to tmp_path."""
    X_train, y_train = synthetic["train"]

    def fake_tuning_data(X: np.ndarray, y: np.ndarray) -> tune.TuningData:
        subset = np.arange(0, len(y), 2)  # every other row stands in for the persisted subset
        skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
        folds = list(skf.split(np.zeros(len(subset)), y[subset]))
        return tune.TuningData(X[subset], y[subset], folds)

    monkeypatch.setattr(tune, "load_train", lambda: (X_train, y_train))
    monkeypatch.setattr(tune, "load_tuning_data", fake_tuning_data)
    monkeypatch.setattr(tune, "load_test_sets", lambda: {k: synthetic[k] for k in ("test_random", "test_future")})
    monkeypatch.setitem(tune.BASE_PARAMS, "num_threads", 2)  # tiny data: thread start-up dominates
    return tune.OutputPaths(log=tmp_path / "experiments.jsonl", models=tmp_path / "models",
                            predictions=tmp_path / "predictions")


def _tuning() -> tune.TuningData:
    X, y = tune.load_train()
    return tune.load_tuning_data(X, y)


@pytest.mark.parametrize("method", tune.TUNING_METHODS)
def test_each_tuner_runs_configured_trials_and_logs_them(method: str, paths: tune.OutputPaths) -> None:
    result = tune.run_search(method, _tuning(), n_trials=N_TRIALS, paths=paths)

    assert len(result.trials) == N_TRIALS
    log = read_experiments(paths.log)
    trials = log[(log["kind"] == "tune_trial") & (log["method"] == method)]
    assert list(trials["trial"]) == list(range(N_TRIALS))
    assert (trials["search_id"] == result.search_id).all()
    assert trials["cv_auc_mean"].between(0.5, 1.0).all()  # the toy problem is learnable
    assert set(tune.SEARCH_SPACE) <= {c.removeprefix("params.") for c in trials.columns}
    searches = log[log["kind"] == "tune_search"]
    assert len(searches) == 1 and int(searches["n_trials"].iloc[0]) == N_TRIALS


@pytest.mark.parametrize("method", tune.TUNING_METHODS)
def test_same_seed_gives_identical_cv_scores(method: str, paths: tune.OutputPaths) -> None:
    first = tune.run_search(method, _tuning(), n_trials=N_TRIALS, paths=paths)
    second = tune.run_search(method, _tuning(), n_trials=N_TRIALS, paths=paths)

    assert [p for p, _ in first.trials] == [p for p, _ in second.trials]
    assert [cv.fold_aucs for _, cv in first.trials] == [cv.fold_aucs for _, cv in second.trials]
    assert [cv.best_iterations for _, cv in first.trials] == [cv.best_iterations for _, cv in second.trials]


def test_search_never_reads_the_test_sets(paths: tune.OutputPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> None:
        raise AssertionError("search code loaded a test set")

    monkeypatch.setattr(tune, "load_test_sets", forbidden)
    for method in tune.TUNING_METHODS:
        tune.run_search(method, _tuning(), n_trials=2, paths=paths)


def test_all_methods_use_the_same_folds(paths: tune.OutputPaths) -> None:
    hashes = {tune.run_search(m, _tuning(), n_trials=1, paths=paths).folds_hash for m in tune.TUNING_METHODS}
    assert len(hashes) == 1


def test_grid_has_36_distinct_points_inside_the_shared_search_space() -> None:
    grid = tune.grid_configs(36)
    assert len(grid) == 36
    assert len({tuple(sorted(p.items())) for p in grid}) == 36
    varied = [name for name in tune.SEARCH_SPACE if len({p[name] for p in grid}) > 1]
    assert len(varied) == 4
    for name in varied:
        low, high, _log = tune.SEARCH_SPACE[name]
        assert all(low <= p[name] <= high for p in grid)
    assert len(tune.grid_configs(4)) == 4
    assert all(p in grid for p in tune.grid_configs(4))
    with pytest.raises(ValueError):
        tune.grid_configs(37)


def test_baselines_refit_and_tables_end_to_end(paths: tune.OutputPaths, tmp_path: Path) -> None:
    X_train, y_train = tune.load_train()
    tuning = tune.load_tuning_data(X_train, y_train)
    finals = tune.run_baselines(X_train, y_train, tuning, paths)
    for method in tune.TUNING_METHODS:
        result = tune.run_search(method, tuning, n_trials=N_TRIALS, paths=paths)
        finals.append(tune.refit_and_evaluate(result, X_train, y_train, paths))

    assert [f["method"] for f in finals] == ["lgbm_default", "logreg", "grid", "random", "optuna"]
    for rec in finals:
        assert rec["test_random"]["n"] == N_TEST and rec["test_future"]["n"] == N_TEST
        assert Path(rec["model_path"]).exists()
    assert any(paths.models.glob("lgbm_optuna*.txt")) and any(paths.models.glob("lgbm_optuna*_params.json"))

    written = tables.build_tuning_outputs(_table_paths(paths, tmp_path))
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    summary = pd.read_csv(written[0])
    assert list(summary["method"]) == tables.METHOD_ORDER
    assert summary.loc[summary["method"].isin(tune.TUNING_METHODS), "n_trials"].eq(N_TRIALS).all()
    assert summary["test_future_roc_auc"].notna().all()
    assert summary["train_rows"].eq(N_TRAIN).all()  # every final model is fit on all of train
    assert summary["refit_version"].eq(tune.REFIT_VERSION).all()
    for test in tables.TEST_SETS:  # each point estimate lies inside its own bootstrap CI
        for metric in tables.CI_METRICS:
            col = f"{test}_{metric}"
            assert (summary[f"{col}_ci_low"] <= summary[col] + 1e-12).all()
            assert (summary[col] <= summary[f"{col}_ci_high"] + 1e-12).all()
    pairwise = pd.read_csv(next(p for p in written if p.name.startswith("tuning_pairwise")))
    assert len(pairwise) == len(tables.TEST_SETS) * len(tables.CI_METRICS) * len(tables.PAIRS)
    trials = pd.read_csv(next(p for p in written if p.name.startswith("tuning_trials")))
    assert len(trials) == len(tune.TUNING_METHODS) * N_TRIALS and "hit_round_cap" in trials
    assert any(p.suffix == ".md" for p in written)


def _table_paths(paths: tune.OutputPaths, tmp_path: Path) -> tables.TablePaths:
    return tables.TablePaths(log=paths.log, tables=tmp_path / "tables", figures=tmp_path / "figures",
                             predictions=paths.predictions, reports=tmp_path)


def test_refit_uses_all_rows_and_fixed_rounds_from_cv(paths: tune.OutputPaths) -> None:
    X_train, y_train = tune.load_train()
    result = tune.run_search("random", _tuning(), n_trials=2, paths=paths)
    rec = tune.refit_and_evaluate(result, X_train, y_train, paths)

    best_cv = result.best[1]
    assert rec["refit_n_estimators"] == round(np.mean(best_cv.best_iterations))
    assert rec["num_trees"] == rec["refit_n_estimators"]  # no early stopping in the refit
    assert rec["train_rows"] == len(y_train)
    assert rec["refit_version"] == tune.REFIT_VERSION


def test_search_from_log_rebuilds_the_logged_search(paths: tune.OutputPaths) -> None:
    result = tune.run_search("optuna", _tuning(), n_trials=N_TRIALS, paths=paths)
    rebuilt = tune.search_from_log("optuna", paths.log)

    assert rebuilt.search_id == result.search_id and rebuilt.folds_hash == result.folds_hash
    assert [p for p, _ in rebuilt.trials] == [p for p, _ in result.trials]
    assert rebuilt.best[1].best_iterations == result.best[1].best_iterations
    with pytest.raises(RuntimeError):
        tune.search_from_log("grid", paths.log)  # no grid search was logged


def test_tables_use_only_the_latest_refit_version() -> None:
    df = pd.DataFrame({
        "kind": ["final_eval"] * 3, "method": ["grid"] * 3,
        "refit_version": [np.nan, 2, 1],  # NaN = written before the field existed = version 1
        "timestamp_utc": ["t1", "t2", "t3"],
    })
    latest = tables.latest_final_evals(df)
    assert len(latest) == 1 and latest["timestamp_utc"].iloc[0] == "t2"


def test_tables_refuse_to_mix_refit_versions() -> None:
    # A refit run that died after the baselines: baselines at the current version, grid still at 1.
    df = pd.DataFrame({
        "kind": ["final_eval"] * 3, "method": ["lgbm_default", "grid", "grid"],
        "refit_version": [tune.REFIT_VERSION, 1, np.nan],
        "timestamp_utc": ["t3", "t1", "t2"],
    })
    with pytest.raises(RuntimeError, match="Refit incomplete for: grid"):
        tables.check_refit_complete(tables.latest_final_evals(df))
    tables.check_refit_complete(tables.latest_final_evals(df[df["method"] == "lgbm_default"]))  # complete: no error


def test_tables_refuse_a_missing_method() -> None:
    df = pd.DataFrame({"kind": ["final_eval"], "method": ["lgbm_default"],
                       "refit_version": [tune.REFIT_VERSION], "timestamp_utc": ["t1"]})
    with pytest.raises(RuntimeError, match="No final_eval record for: logreg, grid, random, optuna"):
        tables.check_refit_complete(tables.latest_final_evals(df), required=tables.METHOD_ORDER)


def test_model_file_points_superseded_versions_to_their_archive(tmp_path: Path) -> None:
    current = tmp_path / "lgbm_grid.txt"
    assert tables.model_file(str(current), tune.REFIT_VERSION) == current.as_posix()
    assert tables.model_file(str(current), 1) == "(not kept)"  # archive folder does not exist yet
    (tmp_path / "refit_v1").mkdir()
    (tmp_path / "refit_v1" / "lgbm_grid.txt").write_text("")
    assert tables.model_file(str(current), 1) == (tmp_path / "refit_v1" / "lgbm_grid.txt").as_posix()


def test_rows_fit_reports_the_version_1_holdout() -> None:
    assert tables.rows_fit(160_000, 1) == 144_000  # 10% early-stopping hold-out, rounded up
    assert tables.rows_fit(160_000, 2) == 160_000


def test_round_cap_flag() -> None:
    cap, patience = tune.MAX_ROUNDS, tune.EARLY_STOPPING_ROUNDS
    assert tune.folds_at_round_cap([10, cap - patience, cap - patience + 1, cap]) == 2


def test_paired_bootstrap_is_seeded_and_paired() -> None:
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400), 0, None)
    first = paired_bootstrap(y, {"a": p, "b": p}, n_resamples=50)
    second = paired_bootstrap(y, {"a": p, "b": p}, n_resamples=50)
    assert np.array_equal(first["a"]["roc_auc"], second["a"]["roc_auc"])  # same seed, same resamples
    assert np.array_equal(first["a"]["roc_auc"], first["b"]["roc_auc"])  # same rows for every model
    low, high = percentile_ci(first["a"]["roc_auc"])
    assert low < compute_metrics(y, p)["roc_auc"] < high
    fast, full = threshold_free_metrics(y, p), compute_metrics(y, p)
    assert fast["roc_auc"] == pytest.approx(full["roc_auc"], abs=1e-12)
    assert fast["tpr_at_1pct_fpr"] == full["tpr_at_1pct_fpr"]
