"""Fast checks of the Step 4 synthesis (src/critical_eval.py, src/critical_eval_writeup.py).

Synthetic inputs pin down the pieces that turn scores into the reported claims (thresholds fixed on
train, precision on both test sets, the paired gap difference, the AutoML self-estimates, and the
sentences chosen from the numbers). The last test renders both reports from the real tables when
they exist for the current mode, and skips otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import without installing the project

from config import SEED, TABLES  # noqa: E402
from src import critical_eval as ce  # noqa: E402
from src import critical_eval_writeup as cw  # noqa: E402


def _scores(n: int = 4000, future_noise: float = 1.0, seed: int = SEED) -> dict[str, tuple[np.ndarray, dict]]:
    """Two test sets and three models; 'drifty' is much noisier on test_future only."""
    rng = np.random.default_rng(seed)
    out = {}
    for test in ("test_random", "test_future"):
        y = rng.integers(0, 2, n)
        signal = 3 * (y - 0.5)
        noise = rng.normal(size=n)
        drifty_noise = noise * (3.0 if test == "test_future" else future_noise)
        out[test] = (y, {"optuna": 1 / (1 + np.exp(-(signal + noise))),
                         "grid": 1 / (1 + np.exp(-(signal + noise))),   # identical to the tuned model
                         "h2o": 1 / (1 + np.exp(-(signal + drifty_noise)))})
    return out


def test_train_thresholds_only_uses_train_derived_values() -> None:
    facts = {"oof_1pct_fpr_thresholds": {"optuna": 0.6, "logreg": 0.9, "not_scored": 0.1},
             "tuned_tuner": "optuna", "deployment_threshold": 0.01,
             "deployment_threshold_source": "models/threshold.json: cost-minimising"}
    thr = ce.train_thresholds(facts, ["optuna", "logreg", "flaml"])
    assert thr == {"optuna": {"0.5": 0.5, "train_oof_1pct_fpr": 0.6, "deployment": 0.01},
                   "logreg": {"0.5": 0.5, "train_oof_1pct_fpr": 0.9}, "flaml": {"0.5": 0.5}}
    # Without a saved threshold file the 0.5 placeholder source must not become a "deployment" row.
    facts["deployment_threshold_source"] = "0.5 default (threshold.json does not exist yet)"
    assert "deployment" not in ce.train_thresholds(facts, ["optuna"])["optuna"]


def test_precision_table_covers_both_sets_and_matches_the_formula() -> None:
    scores = _scores()
    table = ce.precision_table(scores, {"optuna": {"0.5": 0.5}})
    assert set(table["test_set"]) == {"test_random", "test_future"}
    row = table[(table["test_set"] == "test_future") & (table["prevalence"] == 0.01)].iloc[0]
    y, p = scores["test_future"][0], scores["test_future"][1]["optuna"]
    tpr, fpr = (p[y == 1] >= 0.5).mean(), (p[y == 0] >= 0.5).mean()
    assert row["precision"] == pytest.approx(0.01 * tpr / (0.01 * tpr + 0.99 * fpr))
    assert row["precision_ci_low"] <= row["precision"] <= row["precision_ci_high"]


def test_gap_vs_tuned_flags_only_a_real_difference() -> None:
    table = ce.gap_vs_tuned_table(_scores(), "optuna", n_resamples=200)
    auc = table[table["metric"] == "roc_auc"].set_index("model")
    assert auc.loc["optuna", "gap_minus_tuned"] == 0 and not auc.loc["optuna", "differs_from_tuned"]
    assert auc.loc["grid", "gap_minus_tuned_ci_low"] == 0 == auc.loc["grid", "gap_minus_tuned_ci_high"]
    assert auc.loc["h2o", "differs_from_tuned"] and auc.loc["h2o", "gap_minus_tuned"] > 0


@pytest.mark.parametrize("tool, notes, auc, start", [
    ("flaml", {"best_val_auc": 0.99, "eval_method": "holdout", "holdout_rows_expected": 16000}, 0.99,
     "random stratified holdout"),
    ("autogluon", {"best_val_auc": 0.98, "holdout_rows": 2496}, 0.98, "random holdout"),
    ("h2o", {"leader_leaderboard_auc": 0.97, "nfolds_used": 0, "frames": {"validation": 100}}, 0.97,
     "random validation frame"),
    ("h2o", {"leader_leaderboard_auc": 0.96, "nfolds_used": 5}, 0.96, "5-fold CV"),
])
def test_own_estimate_reads_each_tools_notes(tool: str, notes: dict, auc: float, start: str) -> None:
    got_auc, how = ce._own_estimate(tool, notes)
    assert got_auc == auc and how.startswith(start)


def test_sentences_follow_the_numbers() -> None:
    pr = pd.Series({"precision_at_test_mix": 0.99, "precision": 0.50})
    pf = pd.Series({"precision": 0.45})
    assert "the 50/50 test mix is the larger" in cw._mix_vs_split_sentence(pr, pf)
    pf_big = pd.Series({"precision": 0.0})
    assert "the random split is the larger" in cw._mix_vs_split_sentence(pr.replace(0.99, 0.6), pf_big)
    assert cw._range(pd.Series([0.99999, 1.0])) == "1.0000"
    assert cw._range(pd.Series([0.9, 1.0])) == "0.9000 to 1.0000"
    gap = pd.DataFrame([
        {"model": "flaml", "kind": "automl", "metric": "roc_auc", "gap_minus_tuned": 0.0001,
         "gap_minus_tuned_ci_low": -0.0001, "gap_minus_tuned_ci_high": 0.0003, "differs_from_tuned": False},
        {"model": "h2o", "kind": "automl", "metric": "roc_auc", "gap_minus_tuned": 0.0004,
         "gap_minus_tuned_ci_low": 0.0002, "gap_minus_tuned_ci_high": 0.0007, "differs_from_tuned": True}])
    text = cw._automl_drift_verdict({"gap_vs_tuned": gap})
    assert text.startswith("none of the 2 AutoML tools has a smaller AUC gap") and "H2O AutoML (+0.0004" in text


def test_reports_render_from_the_real_tables() -> None:
    needed = ["precision_both_sets", "gap_vs_tuned", "automl_self_estimate", "leakage", "challenge"]
    missing = [s for s in needed if not (TABLES / f"{s}{cw.RUN_SUFFIX}.csv").exists()]
    if missing:
        pytest.skip(f"Step 4 tables not built for this mode: {missing}")
    t = cw.load_inputs()
    for render in (cw.critical_eval_markdown, cw.limitations_markdown):
        text = render(t)
        assert text.startswith("# ") and " nan" not in text.lower() and "n/a%" not in text
