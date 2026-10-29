"""Checks of the Step 4 leakage code (src/leakage.py, src/leakage_report.py).

Most tests use synthetic data in pytest's tmp_path and never touch reports/ or models/.
test_variants_on_real_splits needs the feature cache and skips without it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import without installing the project

from config import QUICK, SEED  # noqa: E402
from src import leakage, leakage_report, tune  # noqa: E402
from src.automl_runs import feature_names  # noqa: E402
from src.logging_utils import log_experiment  # noqa: E402

from conftest import require_persisted_splits  # noqa: E402

D = 2568


def _meta(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    seconds = 1_700_000_000 + rng.integers(0, 10_000_000, n)
    return pd.DataFrame({"detection_ratio_float": rng.random(n),
                         "first_submission_date": pd.to_datetime(seconds, unit="s")})


# --- Time proxies ---------------------------------------------------------------------------

def test_time_proxy_columns_are_the_documented_ones() -> None:
    assert leakage.time_proxy_columns() == {"header__0": 696, "authenticode__6": 2478, "authenticode__7": 2479}


def test_time_proxy_positions_match_thrember_source() -> None:
    """Re-reads thrember's process_raw_features(): each proxy sits where TIME_PROXIES says."""
    for row in leakage.time_proxy_provenance():
        assert row["source_matches_offset"], row
    keys, _, _ = leakage.vector_order("AuthenticodeSignature")
    assert len(keys) == 8  # the family is 8 wide, so every value was matched


# --- Building variants ----------------------------------------------------------------------

def test_apply_variant_appends_or_removes_the_right_columns() -> None:
    n = 50
    X = np.random.default_rng(SEED).random((n, D), dtype=np.float32)
    names, meta = feature_names(), _meta(n)

    X_clean, names_clean = leakage.apply_variant(X, names, meta, leakage.CLEAN)
    assert X_clean is X and names_clean == names

    X_dr, names_dr = leakage.apply_variant(X, names, meta, leakage.LEAK_DR)
    assert X_dr.shape == (n, D + 1) and names_dr[-1] == leakage.LEAK_COLUMN[leakage.LEAK_DR]
    np.testing.assert_array_equal(X_dr[:, :D], X)
    np.testing.assert_allclose(X_dr[:, -1], meta["detection_ratio_float"].to_numpy(np.float32))

    X_date, _ = leakage.apply_variant(X, names, meta, leakage.LEAK_DATE)
    seconds = meta["first_submission_date"].astype("int64").to_numpy() // 10**9
    np.testing.assert_allclose(X_date[:, -1], seconds, rtol=0, atol=128)  # float32 keeps ~128 s at 1.7e9

    X_abl, names_abl = leakage.apply_variant(X, names, meta, leakage.ABLATE)
    assert X_abl.shape == (n, D - 3)
    assert not set(leakage.time_proxy_columns()) & set(names_abl)
    np.testing.assert_array_equal(X_abl[:, 696], X[:, 697])  # columns after 696 shift left by one


def test_unix_seconds_round_trip() -> None:
    dates = pd.Series(pd.to_datetime([0, 1_727_000_000], unit="s"))
    assert leakage.unix_seconds(dates).tolist() == [0, 1_727_000_000]


def test_watched_columns() -> None:
    names = [*feature_names(), leakage.LEAK_COLUMN[leakage.LEAK_DR]]
    assert leakage.watched_columns(names) == [leakage.LEAK_COLUMN[leakage.LEAK_DR], "header__0",
                                              "authenticode__6", "authenticode__7"]


# --- Importance ranks and SHAP ------------------------------------------------------------

def test_rank_of_handles_ties_and_missing_columns() -> None:
    imp = pd.Series({"a": 5.0, "b": 5.0, "c": 1.0, "d": np.nan, "e": 0.0, "f": 0.0})
    assert leakage.rank_of(imp, "b")["rank"] == 1  # tie with "a": both get the best rank
    assert leakage.rank_of(imp, "c") == {"in_model": True, "rank": 3, "value": 1.0, "share": 1 / 11, "n_features": 5}
    assert leakage.rank_of(imp, "e")["rank"] == 5  # unused columns are tied last, not 4th
    assert leakage.rank_of(imp, "d")["rank"] is None  # in the model but not computed (AutoGluon time cap)
    assert leakage.rank_of(imp, "zzz")["in_model"] is False  # dropped by the tool


def _toy_booster(n_rounds: int = 20) -> tuple[lgb.Booster, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    X = rng.random((400, 5))
    y = (X[:, 4] > 0.5).astype(int)  # the last column is a perfect "leak"
    params = {"objective": "binary", "verbosity": -1, "seed": SEED, "num_threads": 1, "min_data_in_leaf": 5}
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_rounds), X, y


def test_tree_shap_adds_up_to_the_raw_score_and_finds_the_leak() -> None:
    booster, X, _ = _toy_booster()
    contrib = leakage.tree_shap(booster, X)
    full = booster.predict(X, pred_contrib=True)
    np.testing.assert_allclose(contrib.sum(axis=1) + full[:, -1], booster.predict(X, raw_score=True), atol=1e-6)
    imp = leakage.mean_abs_shap(contrib, list("abcde"))
    assert leakage.rank_of(imp, "e")["rank"] == 1


def test_leak_split_thresholds_reads_the_date_column() -> None:
    booster, _, _ = _toy_booster()
    names = ["f0", "f1", "f2", "f3", leakage.LEAK_COLUMN[leakage.LEAK_DATE]]
    out = leakage.leak_split_thresholds(booster, names)
    assert out["n_splits"] > 0 and 0.0 < out["max_threshold"] < 1.0
    assert leakage.leak_split_thresholds(booster, ["f0"] * 5) is None


def test_relied_on_leak_rule() -> None:
    base = {"leak_column": "leak__x", "leak_in_model": True}
    assert leakage_report.relied_on_leak({**base, "leak_rank_native": 1, "leak_rank_shap": 3}) is True
    assert leakage_report.relied_on_leak({**base, "leak_rank_native": 2, "leak_rank_shap": 1}) is True
    assert leakage_report.relied_on_leak({**base, "leak_rank_native": 2, "leak_rank_shap": None}) is False
    assert leakage_report.relied_on_leak({**base, "leak_in_model": False, "leak_rank_native": 1}) is False
    assert leakage_report.relied_on_leak({"leak_column": None}) is None
    # A split-count importance (FLAML) ranking the leak first is not reliance; only SHAP counts then
    flaml = {**base, "importance_native_kind": "FLAML feature_importances_ of lgbm"}
    assert leakage_report.relied_on_leak({**flaml, "leak_rank_native": 1, "leak_rank_shap": 23}) is False
    assert leakage_report.relied_on_leak({**flaml, "leak_rank_native": 2, "leak_rank_shap": 1}) is True
    assert leakage_report.relied_on_leak({**flaml, "leak_rank_native": 1, "leak_rank_shap": None}) is True


# --- Config lookup, deltas and log scan ------------------------------------------------------

def test_tuned_lgbm_config_picks_best_cv_tuner(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    for method, cv, trees in (("grid", 0.990, 100), ("optuna", 0.995, 300), ("random", 0.993, 200)):
        log_experiment({"step": "tuning", "kind": "final_eval", "method": method, "refit_version": tune.REFIT_VERSION,
                        "cv_auc_mean": cv, "params": {"num_leaves": trees}, "refit_n_estimators": trees,
                        "eval_id": f"e-{method}", "test_random": {"roc_auc": 0.9}, "test_future": {"roc_auc": 0.8}},
                       path=log)
    config = leakage.tuned_lgbm_config(log)
    assert config["tuner"] == "optuna" and config["n_estimators"] == 300


def _fake_record(tmp_path: Path, variant: str, y: np.ndarray, p: np.ndarray) -> dict:
    path = tmp_path / f"{variant}.npz"
    np.savez_compressed(path, eval_id=f"id-{variant}", test_random_y=y, test_random_p=p, test_future_y=y,
                        test_future_p=p)
    from src.eval import compute_metrics

    return {"method": variant, "eval_id": f"id-{variant}", "predictions_path": str(path),
            "test_random": compute_metrics(y, p), "test_future": compute_metrics(y, p)}


def test_delta_vs_clean_detects_a_real_jump(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, 600)
    noisy = np.clip(y * 0.6 + rng.normal(0.2, 0.3, len(y)), 0, 1)
    clean = _fake_record(tmp_path, "clean", y, noisy)
    leaky = _fake_record(tmp_path, "leak", y, y * 0.98 + 0.01)  # perfect separation
    delta = leakage_report.delta_vs_clean(leaky, clean)
    assert delta["delta_test_random_roc_auc"] > 0 and delta["delta_test_random_roc_auc_significant"]
    same = leakage_report.delta_vs_clean(clean, clean)
    assert same["delta_test_future_roc_auc"] == 0 and not same["delta_test_future_roc_auc_significant"]


def test_load_predictions_finds_a_file_logged_on_another_machine(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 4 records hold /home/user/automl/... paths; the file is then looked up by name here."""
    y = np.array([0, 1, 0, 1])
    rec = _fake_record(tmp_path, "clean", y, np.array([0.1, 0.9, 0.2, 0.8]))
    monkeypatch.setattr(leakage, "LEAKAGE_PREDICTIONS", tmp_path)
    moved = {**rec, "predictions_path": "/home/user/automl/reports/predictions/leakage/clean.npz"}
    assert np.array_equal(leakage_report.load_predictions(moved)["test_future"][0], y)
    with pytest.raises(RuntimeError):  # same name, different evaluation: refused
        leakage_report.load_predictions({**moved, "eval_id": "another-run"})


def test_scan_logs_blanks_the_column_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The leak column's own name contains "leak"; only OTHER flag words may count as a flag."""
    monkeypatch.setattr(leakage_report, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(leakage, "LEAKAGE_LEADERBOARDS", tmp_path / "lb")
    monkeypatch.setattr(leakage_report, "REPORTS", tmp_path)
    variant, column = leakage.LEAK_DR, leakage.LEAK_COLUMN[leakage.LEAK_DR]
    log = leakage_report.worker_log_path("flaml", variant)
    log.parent.mkdir(parents=True)
    log.write_text(f"[leakage] flaml on {variant}\nusing {column}\nUserWarning: something\n", encoding="utf-8")
    notes = tmp_path / "lb" / variant / f"flaml_preprocessing{'_quick' if QUICK else ''}.json"
    notes.parent.mkdir(parents=True)
    notes.write_text(json.dumps({"dropped_features": ["header__36"]}), encoding="utf-8")
    scan = leakage_report.scan_logs("flaml", variant)
    assert scan["n_lines_mentioning_leak"] == 1
    assert scan["n_flag_word_lines"] == 0  # "leak" inside the column name does not count
    assert scan["n_warning_lines"] == 1
    assert scan["leak_dropped_by_tool"] is False


def test_require_disk_refuses_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(leakage, "free_disk_gb", lambda path=None: 1.0)
    with pytest.raises(RuntimeError, match="GB free"):
        leakage.require_disk(5.0)
    leakage.require_disk(0.5)  # enough space: no error


def test_machine_info_names_cpu_and_ram() -> None:
    info = leakage.machine_info()
    assert info["cpu_count"] and info["ram_gb"] and info["os"]


def test_h2o_varimp_counts_unused_columns_as_zero(tmp_path: Path) -> None:
    """H2O's varimp lists only used columns; an unused leak column is rank-last, not 'dropped'."""
    from src.automl_runs import AutomlPaths, FitResult

    paths = AutomlPaths(log=tmp_path / "log.jsonl", models=tmp_path / "m", leaderboards=tmp_path / "lb",
                        predictions=tmp_path / "p")
    paths.leaderboards.mkdir()
    names = ["a", "b", "c", leakage.LEAK_COLUMN[leakage.LEAK_DATE]]
    pd.DataFrame({"feature": ["a", "b"], "relative_importance": [3.0, 1.0]}).to_csv(
        paths.leaderboards / f"h2o_varimp{'_quick' if QUICK else ''}.csv", index=False)
    (paths.leaderboards / f"h2o_preprocessing{'_quick' if QUICK else ''}.json").write_text(
        json.dumps({"constant_in_train": ["c"]}), encoding="utf-8")
    fit = FitResult(tool="h2o", best_model="x", predict=lambda X: X, timings={}, info={})  # no varimp model
    out = leakage.h2o_importance(fit, np.zeros((3, 4)), names, paths)
    rank = out["native"]["ranks"][leakage.LEAK_COLUMN[leakage.LEAK_DATE]]
    assert rank["in_model"] is True and rank["rank"] == 3 and rank["value"] == 0.0  # 3 kept columns, tied last


def test_autogluon_importance_computes_watched_columns_first(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """The leak column is last in X; with fi_first it gets a value even if the time cap stops after one chunk."""
    from src import automl_runs

    leak = leakage.LEAK_COLUMN[leakage.LEAK_DR]
    names = [f"general__{i}" for i in range(6)] + [leak]

    class FakePredictor:
        def features(self, feature_stage: str = "original") -> list[str]:
            return names

        def feature_importance(self, features: list, **_: object) -> pd.DataFrame:
            if isinstance(features[0], tuple):  # the family stage
                return pd.DataFrame({"importance": [1.0] * len(features)}, index=[f for f, _ in features])
            monkeypatch.setattr(automl_runs, "AG_FI_TIME_LIMIT_S", -1.0)  # cap reached after this chunk
            return pd.DataFrame({"importance": [1.0] * len(features)}, index=features)

    monkeypatch.setattr(automl_runs, "AG_FI_FEATURE_CHUNK", 2)
    monkeypatch.setattr(automl_runs, "AG_FI_TIME_LIMIT_S", 1e9)
    paths = automl_runs.AutomlPaths(log=tmp_path / "l.jsonl", models=tmp_path, leaderboards=tmp_path,
                                    predictions=tmp_path)
    info = automl_runs._ag_feature_importance(FakePredictor(), names, paths, fi_first=(leak,))
    assert info["features_covered"] == 2
    fi = pd.read_csv(tmp_path / f"autogluon_feature_importance{'_quick' if QUICK else ''}.csv").set_index("feature")
    assert fi.loc[leak, "importance"] == 1.0 and fi["importance"].isna().sum() == 5


def test_run_plan_covers_every_required_comparison() -> None:
    plan = set(leakage.RUN_PLAN)
    for model in leakage.MODEL_NAMES:
        assert {(model, leakage.CLEAN), (model, leakage.LEAK_DR), (model, leakage.LEAK_DATE)} <= plan
    assert (leakage.LGBM, leakage.ABLATE) in plan
    assert leakage.RUN_PLAN[:4] == tuple((leakage.LGBM, v) for v in leakage.VARIANTS)  # cheap runs first


# --- On the real splits (skips without the cache) -------------------------------------------

def test_variants_on_real_splits() -> None:
    require_persisted_splits()
    X, y, names, meta = leakage.load_variant("test_future", leakage.LEAK_DR)
    assert names[-1] == leakage.LEAK_COLUMN[leakage.LEAK_DR] and X.shape[1] == D + 1
    benign, malicious = X[y == 0, -1], X[y == 1, -1]
    assert benign.max() < malicious.min()  # the leak separates the classes perfectly (AUC 1.0)
    from src.data import load_cache_meta, split_rows

    train_meta = load_cache_meta("train").iloc[split_rows("train")]  # metadata only: no 1.6 GB X
    # Every future date lies after every training date: the date leak is out of range on test_future.
    assert meta["first_submission_date"].min() > train_meta["first_submission_date"].max()
