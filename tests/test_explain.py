"""Checks of the Step 4 part C code (SHAP helpers, calibration, cost threshold) on small synthetic data.

Nothing here reads the data cache; the last test checks the saved threshold file only if a run
has written it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import without installing the project

import config  # noqa: E402
from config import SEED  # noqa: E402
from src import calibration as cal  # noqa: E402
from src import explain  # noqa: E402


@pytest.fixture(scope="module")
def toy() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(n_samples=800, n_features=12, n_informative=5, flip_y=0.05, random_state=SEED)
    return X.astype(np.float32), y.astype(np.int8)


@pytest.fixture(scope="module")
def booster(toy: tuple[np.ndarray, np.ndarray]) -> lgb.Booster:
    X, y = toy
    params = {"objective": "binary", "verbosity": -1, "seed": SEED, "num_leaves": 7, "num_threads": 1}
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=30)


# --- SHAP ---------------------------------------------------------------------------------

def test_lgbm_shap_matches_shap_treeexplainer_and_adds_up(toy, booster) -> None:
    shap = pytest.importorskip("shap")
    X, _y = toy
    values, base = explain.lgbm_shap(booster, X[:50], n_threads=1)
    reference = shap.TreeExplainer(booster).shap_values(X[:50])
    reference = reference[1] if isinstance(reference, list) else reference
    np.testing.assert_allclose(values, reference, atol=1e-6)
    margin = booster.predict(X[:50], raw_score=True)
    np.testing.assert_allclose(base + values.sum(axis=1), margin, atol=1e-6)


def test_xgb_shap_adds_up_to_margin(toy) -> None:
    xgb = pytest.importorskip("xgboost")
    X, y = toy
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=SEED, n_jobs=1).fit(X, y)
    values, base = explain.xgb_shap(model, X[:40], n_threads=1)
    margin = model.predict(X[:40], output_margin=True)
    np.testing.assert_allclose(base + values.sum(axis=1), margin, atol=1e-4)


def test_expand_to_all_places_columns() -> None:
    values = np.array([[1.0, 2.0]])
    out = explain.expand_to_all(values, ["b__0", "a__1"], ["a__0", "a__1", "b__0"])
    np.testing.assert_array_equal(out, [[0.0, 2.0, 1.0]])


def test_family_table_shares_and_grouping() -> None:
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=(100, 5))
    groups = [("a", 0, 2), ("b", 2, 5)]
    table = explain.family_table(values, groups, [f"f{i}" for i in range(5)])
    assert table["share"].sum() == pytest.approx(1.0)
    assert (table["grouped_mean_abs_shap"] <= table["sum_mean_abs_shap"] + 1e-12).all()
    assert list(table["rank"]) == [1, 2]
    assert set(table["family"]) == {"a", "b"}


def test_readable_names_cover_every_column() -> None:
    data = pytest.importorskip("src.data")
    try:
        extractor = data.get_extractor()
    except Exception as exc:  # thrember not importable on this machine
        pytest.skip(f"thrember extractor unavailable: {exc}")
    names = explain.readable_names(extractor)
    groups = data.feature_groups()
    assert len(names) == groups[-1][2]
    start = {fam: s for fam, s, _e in groups}
    # The time proxies named in src/leakage.py and the column the AutoML tools rank first.
    assert names[start["header"]] == "header: coff.timestamp"
    assert names[start["authenticode"] + 6] == "authenticode: latest_signing_time"
    assert names[start["authenticode"] + 7] == "authenticode: signing_time_diff"
    assert names[start["exports"]] == "exports: has_exports(0/128)"
    assert not any(n.split(": ", 1)[1].isdigit() for n in names), "a family fell back to bare indices"


def test_agreement_table_identical_and_reversed() -> None:
    names = [f"a__{i}" for i in range(10)] + [f"b__{i}" for i in range(10)] + [f"c__{i}" for i in range(5)]
    base = pd.Series(np.arange(len(names), 0, -1, dtype=float), index=names)
    feat = {"x": base, "y": base.copy(), "z": base[::-1].copy().set_axis(names)}
    fams = ["a", "b", "c"]
    fam = {t: explain.to_families(s, fams) for t, s in feat.items()}
    table = explain.agreement_table(feat, fam).set_index(["tool_a", "tool_b"])
    assert table.loc[("x", "y"), "top5_overlap"] == 5
    assert table.loc[("x", "y"), "family_spearman"] == pytest.approx(1.0)
    assert table.loc[("x", "z"), "top5_overlap"] == 0


def test_pick_cases() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.99, 0.7, 0.8, 0.1, 0.2, 0.95])
    cases = explain.pick_cases(y, p)
    assert cases == {"tp": 0, "fp": 5, "fn": 4}
    assert explain.pick_cases(np.array([1]), np.array([0.9]))["fp"] is None


def test_stratified_sample_keeps_balance() -> None:
    meta = pd.DataFrame({"label": np.repeat([0, 1], 500), "week": np.tile(np.arange(10), 100)})
    pos = explain.stratified_sample(meta, 200)
    assert len(pos) == 200 and np.all(np.diff(pos) > 0)
    assert meta.iloc[pos]["label"].mean() == pytest.approx(0.5)
    assert len(explain.stratified_sample(meta, 5000)) == 1000


# --- Calibration --------------------------------------------------------------------------

def test_oof_predictions_cover_rows_and_detect_gaps(toy) -> None:
    X, y = toy
    idx = np.arange(len(y))
    folds = [(idx[idx % 2 == 1], idx[idx % 2 == 0]), (idx[idx % 2 == 0], idx[idx % 2 == 1])]
    seen_rows: list[int] = []

    def fit(X_fit: np.ndarray, y_fit: np.ndarray):
        seen_rows.append(len(y_fit))
        return lambda X_val: np.full(len(X_val), y_fit.mean())

    oof = cal.oof_predictions(X, y, folds, fit)
    assert not np.isnan(oof).any() and seen_rows == [400, 400]
    with pytest.raises(ValueError):
        cal.oof_predictions(X, y, folds[:1], fit)


def test_calibration_metrics_known_values() -> None:
    rng = np.random.default_rng(SEED)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)   # calibrated by construction
    m = cal.calibration_metrics(y, p)
    assert m["ece"] < 0.01
    assert m["brier"] == pytest.approx(1 / 6, abs=0.005)   # E[p(1-p)] for p ~ U(0,1)
    constant = cal.calibration_metrics(y, np.full(p.size, 0.9))
    assert constant["ece"] == pytest.approx(0.4, abs=0.01)


def test_calibrators_are_monotone_and_reduce_miscalibration() -> None:
    rng = np.random.default_rng(SEED)
    true_p = rng.uniform(size=20_000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(int)
    skewed = true_p ** 3   # systematically under-confident about malware
    for fit in (cal.fit_isotonic, cal.fit_platt):
        c = fit(y, skewed)
        grid = np.linspace(0, 1, 101)
        assert np.all(np.diff(c.transform(grid)) >= -1e-12)
        assert cal.expected_calibration_error(y, c.transform(skewed)) < cal.expected_calibration_error(y, skewed)


# --- Cost threshold -----------------------------------------------------------------------

def brute_force_cost(y: np.ndarray, p: np.ndarray, t: float, ratio: float) -> float:
    flag = p >= t
    return ratio * np.sum(~flag & (y == 1)) + np.sum(flag & (y == 0))


@pytest.mark.parametrize("ratio", cal.COST_RATIOS)
def test_choose_threshold_is_the_brute_force_minimum(ratio: float) -> None:
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, size=400)
    p = np.round(np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1), 2)   # rounding creates ties
    chosen = cal.choose_threshold(y, p, ratio)
    best = min(brute_force_cost(y, p, t, ratio) for t in np.r_[np.unique(p), np.inf])
    assert chosen["cost"] == best
    assert brute_force_cost(y, p, chosen["threshold"], ratio) == best   # the midway cut keeps the counts
    counts = cal.confusion_at(y, p, chosen["threshold"], ratio)
    assert counts["cost"] == best and counts["fn"] == chosen["fn"] and counts["fp"] == chosen["fp"]


def test_higher_miss_cost_never_raises_threshold() -> None:
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, size=2000)
    p = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    thresholds = [cal.choose_threshold(y, p, r)["threshold"] for r in cal.COST_RATIOS]
    assert thresholds == sorted(thresholds, reverse=True)


def test_saved_threshold_file_is_complete() -> None:
    path = config.MODELS / f"threshold{config.RUN_SUFFIX}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not written yet; run scripts/run_explain.py --stages threshold")
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("threshold", "cost_ratio_fn_to_fp", "how_chosen", "source_of_predictions", "model"):
        assert key in record
    assert 0.0 < record["threshold"] < 1.0
    assert "No test set" in record["how_chosen"]
    assert "oof" in record["source_of_predictions"]
