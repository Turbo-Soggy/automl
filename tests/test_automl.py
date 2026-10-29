"""Fast checks of the Step 3 code on synthetic inputs (no real data, no AutoML fit).

A fake experiment log and fake saved test scores go to pytest's tmp_path, so these tests never
touch reports/ or models/. They check the AutoML-vs-manual tables builder and the FLAML
leaderboard parser, which turn logs into the reported numbers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import without installing the project

from config import RUN_SUFFIX, SEED  # noqa: E402
from src import automl_checks, automl_runs, tables, tune  # noqa: E402
from src.eval import compute_metrics  # noqa: E402
from src.logging_utils import log_experiment  # noqa: E402

N_TEST = 400
# CV AUC per tuner: "random" has the best CV score, so it must be the one reported as lgbm_tuned
# even though "optuna" gets the best (noise-free) test scores below. Selection is by CV only.
CV_AUC = {"grid": 0.990, "random": 0.995, "optuna": 0.993}
NOISE = {"lgbm_default": 0.8, "logreg": 1.5, "grid": 0.6, "random": 0.5, "optuna": 0.3,
         "automl_flaml": 0.5, "automl_autogluon": 0.4, "automl_h2o": 0.7}


def _labels() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return {test: rng.integers(0, 2, N_TEST) for test in tables.TEST_SETS}


def _scores(y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Probabilities that separate the classes, more noise = worse model."""
    logit = (2 * y - 1) * 2.0 + rng.normal(0.0, noise * 2.0, len(y))
    return 1.0 / (1.0 + np.exp(-logit))


def _save(predictions: Path, method: str, eval_id: str, scored: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    arrays = {}
    for test, (y, p) in scored.items():
        arrays[f"{test}_y"], arrays[f"{test}_p"] = y.astype(np.int8), p
    predictions.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions / f"{method}{RUN_SUFFIX}.npz", eval_id=eval_id, **arrays)


def _write_fake_run(tmp: Path, tools: list[str]) -> tables.TablePaths:
    """A log with all five Step 2 final_evals and one final_eval per AutoML tool, plus saved scores."""
    paths = tables.TablePaths(log=tmp / "experiments.jsonl", tables=tmp / "tables", figures=tmp / "figures",
                              predictions=tmp / "predictions", reports=tmp, leaderboards=tmp / "leaderboards")
    labels, rng = _labels(), np.random.default_rng(SEED + 7)
    for method in tables.METHOD_ORDER:
        scored = {t: (labels[t], _scores(labels[t], NOISE[method], rng)) for t in tables.TEST_SETS}
        eval_id = f"fake-{method}"
        _save(paths.predictions, method, eval_id, scored)
        is_tuner = method in tables.TUNERS
        log_experiment({
            "step": "tuning", "kind": "final_eval", "method": method, "refit_version": tune.REFIT_VERSION,
            "eval_id": eval_id, "model": "lightgbm" if method != "logreg" else "logistic_regression",
            "cv_auc_mean": CV_AUC.get(method, 0.98), "n_trials": 36 if is_tuner else None,
            "search_seconds": 3600.0 if is_tuner else None, "fit_seconds": 60.0, "train_rows": 160_000,
            "refit_n_estimators": 500 if is_tuner else None,
            **{t: compute_metrics(*scored[t]) for t in tables.TEST_SETS},
        }, path=paths.log)
    for tool in tools:
        method = f"automl_{tool}"
        scored = {t: (labels[t], _scores(labels[t], NOISE[method], rng)) for t in tables.TEST_SETS}
        eval_id = f"fake-{method}"
        _save(paths.predictions, method, eval_id, scored)
        log_experiment({
            "step": "automl", "kind": "final_eval", "method": method, "tool": tool, "eval_id": eval_id,
            "best_model": f"{tool}_best", "budget_s": 1200, "init_s": 5.0, "convert_s": 10.0, "fit_s": 1250.0,
            "wall_s": 1265.0, "train_rows": 160_000, "interpretability": "fake",
            **{t: compute_metrics(*scored[t]) for t in tables.TEST_SETS},
        }, path=paths.log)
    return paths


def test_automl_tables_from_synthetic_log(tmp_path: Path) -> None:
    paths = _write_fake_run(tmp_path, list(tables.AUTOML_TOOLS))
    written = tables.build_automl_outputs(paths)
    assert all(p.exists() for p in written)

    table = pd.read_csv(paths.tables / f"automl_vs_manual{RUN_SUFFIX}.csv")
    assert table["method"].tolist() == [*tables.AUTOML_TOOLS, tables.TUNED_LABEL, "lgbm_default", "logreg"]
    tuned = table.set_index("method").loc[tables.TUNED_LABEL]
    assert "random" in tuned["best_model"]  # best CV AUC wins, not the best test score (optuna)
    assert tuned["wall_seconds"] == pytest.approx(3600.0 + 60.0)
    assert tuned["all_manual_seconds"] == pytest.approx(3 * 3600.0 + 3 * 60.0)
    flaml = table.set_index("method").loc["flaml"]
    assert flaml["over_budget_seconds"] == pytest.approx(65.0)  # wall 1265 - budget 1200
    assert flaml["fit_over_budget_seconds"] == pytest.approx(50.0)  # fit 1250 - budget 1200 (no set-up)
    for test in tables.TEST_SETS:  # each logged point estimate lies inside its bootstrap CI
        for metric in tables.CI_METRICS:
            col = f"{test}_{metric}"
            assert (table[f"{col}_ci_low"] <= table[col] + 1e-12).all()
            assert (table[col] <= table[f"{col}_ci_high"] + 1e-12).all()

    pairwise = pd.read_csv(paths.tables / f"automl_pairwise{RUN_SUFFIX}.csv")
    # Per test set and metric: 3 tools x 2 references, default vs tuned, 3 tool pairs = 10 pairs;
    # plus per metric one gap comparison against tuned for the 3 tools and default (4).
    assert len(pairwise) == 2 * 2 * 10 + 2 * 4
    assert set(pairwise.loc[pairwise.test_set == tables.GAP_SET, "method_b"]) == {tables.TUNED_LABEL}
    gap = pairwise[(pairwise.test_set == tables.GAP_SET) & (pairwise.method_a == "h2o")
                   & (pairwise.metric == "roc_auc")].iloc[0]
    by = table.set_index("method")["auc_gap_random_minus_future"]
    assert gap["diff_a_minus_b"] == pytest.approx(by["h2o"] - by[tables.TUNED_LABEL])
    row = pairwise[(pairwise.method_a == "flaml") & (pairwise.method_b == tables.TUNED_LABEL)
                   & (pairwise.test_set == "test_future") & (pairwise.metric == "roc_auc")].iloc[0]
    expected = table.set_index("method").loc["flaml", "test_future_roc_auc"] - tuned["test_future_roc_auc"]
    assert row["diff_a_minus_b"] == pytest.approx(expected)
    assert row["ci_low"] <= row["diff_a_minus_b"] <= row["ci_high"]

    text = (paths.reports / f"automl_comparison{RUN_SUFFIX}.md").read_text(encoding="utf-8")
    assert "not an apples-to-apples" in text
    assert "Leaderboard not available" in text  # no leaderboards in this fake run: says so, no crash


def test_real_run_requires_all_tools() -> None:
    two_tools = pd.DataFrame({"tool": ["flaml", "autogluon"]})
    with pytest.raises(RuntimeError, match="h2o"):
        tables.check_tools_complete(two_tools, require_all=True)
    tables.check_tools_complete(two_tools, require_all=False)  # smoke runs may skip a tool


def test_flaml_leaderboard_from_search_log(tmp_path: Path) -> None:
    trials = [
        {"record_id": 0, "learner": "lgbm", "validation_loss": 0.02, "wall_clock_time": 1.0, "sample_size": 10_000,
         "trial_time": 0.5, "config": {"n_estimators": 4}},
        {"record_id": 1, "learner": "rf", "validation_loss": 0.05, "wall_clock_time": 2.0, "sample_size": 10_000,
         "trial_time": 0.7, "config": {"n_estimators": 8}},
        {"record_id": 2, "learner": "lgbm", "validation_loss": 0.01, "wall_clock_time": 3.0, "sample_size": 40_000,
         "trial_time": 1.5, "config": {"n_estimators": 16}},
    ]
    log = tmp_path / "search.json"
    lines = [json.dumps(t) for t in trials] + [json.dumps({"curr_best_record_id": 2})]  # FLAML's last line
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    board = automl_runs.flaml_leaderboard(automl_runs.read_flaml_log(log))
    assert board["estimator"].tolist() == ["lgbm", "rf"]
    lgbm = board.iloc[0]
    assert (lgbm["n_trials"], lgbm["best_record_id"], lgbm["best_sample_size"]) == (2, 2, 40_000)
    assert lgbm["best_val_auc"] == pytest.approx(0.99)
    assert json.loads(lgbm["best_config"]) == {"n_estimators": 16}


def test_feature_names_follow_groups() -> None:
    names = automl_runs.feature_names()
    assert len(names) == len(set(names)) == sum(end - start for _, start, end in automl_runs.feature_groups())
    assert all(automl_runs.family_of(n) in {g for g, _, _ in automl_runs.feature_groups()} for n in names)


def test_max_gap_flags_a_sleep() -> None:
    assert automl_runs.max_gap([]) == 0.0 and automl_runs.max_gap([5.0]) == 0.0
    assert automl_runs.max_gap([0.0, 30.0, 60.0, 400.0, 430.0]) == pytest.approx(340.0)  # slept ~5 min


def test_heartbeat_reports_valid_timing_for_a_short_block() -> None:
    with automl_runs.Heartbeat("test", interval=0.05, max_gap_s=5.0) as beat:
        time.sleep(0.3)
    report = beat.report()
    assert report["timing_valid"] is True
    assert len(beat.beats) >= 3  # start, some beats, end
    assert 0.25 <= report["heartbeat_seconds"] < 5.0


def test_tables_skip_runs_that_slept(tmp_path: Path) -> None:
    paths = _write_fake_run(tmp_path, list(tables.AUTOML_TOOLS))
    # A LATER flaml record that the sleep guard flagged must not replace the valid one.
    log_experiment({"step": "automl", "kind": "final_eval", "method": "automl_flaml", "tool": "flaml",
                    "eval_id": "slept", "best_model": "slept_model", "budget_s": 1200, "init_s": 0.0,
                    "convert_s": 1.0, "fit_s": 5000.0, "wall_s": 5001.0, "train_rows": 160_000,
                    "interpretability": "fake", "heartbeat_max_gap_s": 3600.0, "timing_valid": False},
                   path=paths.log)
    records = tables.automl_records(paths.log).set_index("tool")
    assert records.loc["flaml", "eval_id"] == "fake-automl_flaml"
    assert records.loc["flaml", "invalid_attempts"] == 1
    assert records.loc["h2o", "invalid_attempts"] == 0
    tables.build_automl_outputs(paths)
    text = (paths.reports / f"automl_comparison{RUN_SUFFIX}.md").read_text(encoding="utf-8")
    assert "1 earlier attempt(s) discarded" in text


def test_automl_pairs_cover_tools_references_and_tool_pairs() -> None:
    pairs = tables.automl_pairs(["flaml", "autogluon", tables.TUNED_LABEL, "lgbm_default"])
    assert ("flaml", tables.TUNED_LABEL) in pairs and ("autogluon", "lgbm_default") in pairs
    assert ("lgbm_default", tables.TUNED_LABEL) in pairs and ("flaml", "autogluon") in pairs
    assert ("h2o", tables.TUNED_LABEL) not in pairs  # a tool missing from the table gets no rows


def test_minority_rows_counts_rows_off_the_mode() -> None:
    X = np.zeros((10, 2), dtype=np.float32)
    X[[0, 1, 2], 1] = 1.0  # a rare flag set in 3 of 10 rows
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    out = automl_checks.minority_rows(X, y, ["a__0", "a__1"], ["a__1"])
    assert out["a__1"] == {"rows_not_mode": 3, "n_distinct": 2, "malicious_share_of_those": pytest.approx(2 / 3)}


def test_h2o_interpretability_uses_metalearner_weights() -> None:
    text = automl_checks.h2o_interpretability("stackedensemble", ["GBM_1_AutoML_1_x", "GLM_1_AutoML_1_x"],
                                              ["GBM_1_AutoML_1_x"])
    assert "only to GBM_1" in text and "in effect one model" in text
    assert "GLM metalearner over 2 of 2" in automl_checks.h2o_interpretability(
        "stackedensemble", ["A", "B"], ["A", "B"])
    assert automl_checks.interpretability_from_notes("flaml", {"checks": {}}) is None


def test_refresh_records_renames_and_changes() -> None:
    notes = {"leader_cv_auc": 0.9, "importance_type": None}
    out = automl_checks._apply(notes, {"importance_type": "gain", "checks": {"x": 1}}, "h2o")
    assert "leader_cv_auc" not in out and out["leader_leaderboard_auc"] == 0.9
    assert out["refreshed"]["changed"]["leader_cv_auc"] == {"renamed_to": "leader_leaderboard_auc"}
    assert out["refreshed"]["changed"]["importance_type"] == {"old": None, "new": "gain"}
    again = automl_checks._apply(out, {"importance_type": "gain", "checks": {"x": 1}}, "h2o")
    assert again["refreshed"]["changed"]["importance_type"]["old"] is None  # history survives a second refresh


def test_gap_rows_count_the_smaller_gap_as_better() -> None:
    # For a gap a larger value means more overstatement, so "a better" must count diff < 0.
    diff = np.array([0.1, 0.2, -0.1, 0.3])
    assert tables._pair_row("test_future", "roc_auc", "a", "b", 0.1, diff)["share_resamples_a_better"] == 0.75
    gap = tables._pair_row(tables.GAP_SET, "roc_auc", "a", "b", 0.1, diff, larger_is_better=False)
    assert gap["share_resamples_a_better"] == 0.25


def test_autogluon_family_ties_are_grouped() -> None:
    from src import automl_writeup

    df = pd.DataFrame({"family": ["imports", "header", "strings", "datadirectories", "general"],
                       "importance": [0.0035, 0.0022, 0.0018, 0.00175, 0.0001],
                       "stddev": [0.0002, 0.0002, 0.0005, 0.0005, 0.00001], "n": 5})
    text = automl_writeup._ag_family_text(df)
    assert "header, strings and datadirectories (ranks 2-4)" in text
    assert "imports and" not in text  # rank 1 is clearly ahead of rank 2
