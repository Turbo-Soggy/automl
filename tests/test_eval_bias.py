"""Fast checks of the Step 4 part B code (temporal, bias, drift, challenge) on synthetic inputs.

No real data, no saved model and no fit of the real size: these tests pin down the pieces that
turn scores into reported numbers (fold layout, metrics, thresholds, CIs, base-rate maths, the
LogReg decomposition and the challenge-set filtering).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import without installing the project

from config import SEED  # noqa: E402
from src import bias, challenge, drift, temporal  # noqa: E402
from src.eval import tpr_at_fpr  # noqa: E402


def _scored(n: int = 2000, noise: float = 1.0, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(3 * (y - 0.5) + noise * rng.normal(size=n))))
    return y, p


# --- temporal -----------------------------------------------------------------------------
def test_forward_folds_are_expanding_and_strictly_forward() -> None:
    weeks = np.repeat(np.arange(52), 5)
    folds = temporal.forward_folds(weeks, n_folds=4)
    assert len(folds) == 4
    prev_fit = 0
    for fit, val, desc in folds:
        assert weeks[fit].max() < weeks[val].min()       # validation strictly after training
        assert len(np.intersect1d(fit, val)) == 0
        assert len(fit) > prev_fit                         # expanding window
        prev_fit = len(fit)
        assert len(np.unique(weeks[val])) == 10            # equal validation blocks of 10 weeks
        assert "validate weeks" in desc
    assert weeks[folds[0][0]].min() == 0 and weeks[folds[-1][1]].max() == 51


def test_operating_metrics_match_sklearn() -> None:
    y, p = _scored()
    m = temporal.operating_metrics(y, p, 0.5)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["tpr_at_1pct_fpr"] == pytest.approx(tpr_at_fpr(y, p, 0.01))
    assert m["f1"] == pytest.approx(f1_score(y, p >= 0.5))
    assert m["fpr"] == pytest.approx(np.mean(p[y == 0] >= 0.5))
    assert m["tpr"] == pytest.approx(np.mean(p[y == 1] >= 0.5))


def test_threshold_at_fpr_respects_the_cap() -> None:
    y, p = _scored(5000)
    thr = temporal.threshold_at_fpr(y, p, 0.01)
    assert np.mean(p[y == 0] >= thr) <= 0.01
    lower = p[p < thr].max()  # the next lower score would break the cap
    assert np.mean(p[y == 0] >= lower) > 0.01


def test_gap_rows_gap_is_random_minus_future() -> None:
    y_r, p_r = _scored(800, noise=0.5, seed=1)
    y_f, p_f = _scored(800, noise=1.5, seed=2)
    scores = {"test_random": (y_r, {"m": p_r}), "test_future": (y_f, {"m": p_f})}
    row = temporal.gap_rows(scores, ["m"], n_resamples=50)[0]
    assert row["roc_auc_gap"] == pytest.approx(row["test_random_roc_auc"] - row["test_future_roc_auc"])
    assert row["roc_auc_gap"] > 0  # the noisier "future" set is harder
    assert row["test_future_roc_auc_ci_low"] <= row["test_future_roc_auc"] <= row["test_future_roc_auc_ci_high"]


def test_logreg_shift_parts_add_up_to_the_logit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-family parts must sum to decision_function minus the intercept."""
    from src import data

    rng = np.random.default_rng(SEED)
    d = 12
    groups = [("a", 0, 4), ("b", 4, 8), ("c", 8, 12)]
    monkeypatch.setattr(data, "feature_groups", lambda: groups)
    monkeypatch.setattr(temporal, "time_proxy_columns", lambda: {"t0": 0, "t1": 5})
    X = rng.normal(size=(400, d))
    y = (X[:, 0] + X[:, 5] + rng.normal(size=400) > 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    X_f = X + 0.3
    table = temporal.logreg_shift_table(model, X, y, X_f, y).set_index("family")
    logit = model.decision_function(X_f) - model[-1].intercept_[0]
    families = table.drop(index="time_proxies")
    for label, cls in (("benign", 0), ("malicious", 1)):
        assert families[f"test_future_{label}_mean_logit"].sum() == pytest.approx(logit[y == cls].mean())


# --- bias ---------------------------------------------------------------------------------
def test_wilson_ci_known_values() -> None:
    lo, hi = bias.wilson_ci(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-3)
    assert bias.wilson_ci(0, 384)[0] == 0.0 and bias.wilson_ci(384, 384)[1] == 1.0  # clipped exactly
    lo, hi = bias.wilson_ci(50, 100)
    assert lo < 0.5 < hi and hi - 0.5 == pytest.approx(0.5 - lo)
    assert np.isnan(bias.wilson_ci(0, 0)[0])


def test_base_rate_effect() -> None:
    """At 1% prevalence a 1% FPR detector raises about as many false alarms as true ones."""
    point = bias.at_prevalence(tpr=0.98, fpr=0.01, prevalence=0.01)
    assert point["precision"] == pytest.approx(0.98 * 0.01 / (0.98 * 0.01 + 0.99 * 0.01))
    assert point["alarms_per_10k"] == pytest.approx(98 + 99)
    assert bias.at_prevalence(0.98, 0.01, 0.5)["precision"] > 0.98


def test_slices_flag_small_and_family_is_recall_only() -> None:
    rng = np.random.default_rng(SEED)
    n = 600
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    meta = pd.DataFrame({"week": rng.integers(52, 55, n), "file_size": rng.integers(1, 10_000, n),
                         "is_packed": (y == 1) & (rng.random(n) < 0.1), "packer": "", "signed": rng.random(n) < 0.5,
                         "family": np.where(y == 1, "fam", None)})
    edges = bias.size_quartile_edges(rng.integers(1, 10_000, 5000))
    masks = bias.slice_masks(meta, rng.random(n) * 8, edges)
    rows = pd.DataFrame(bias.slice_rows("test_future", y, p, masks, 0.5, "0.5")).set_index(["slice_family", "slice"])
    sizes = rows.xs("file_size_quartile")["n"]
    assert sizes.sum() == n                                    # quartiles partition the test set
    packed = rows.loc[("packer_tag", "packed (any)")]
    assert packed["n_benign"] == 0 and bool(packed["fpr_too_small"]) and np.isnan(packed["fpr"])
    family = rows.loc[("family", "known")]
    assert np.isnan(family["fpr"]) and family["n"] == family["n_malicious"]


def test_class_weight_specs() -> None:
    y = np.r_[np.zeros(60, int), np.ones(40, int)]
    specs = {s.name: s for s in bias.class_weight_specs(y)}
    assert specs["class_weight_none"].lgbm_extra == {}
    assert specs["class_weight_balanced"].lgbm_extra["scale_pos_weight"] == pytest.approx(60 / 40)
    assert specs["class_weight_prior_10pct"].lgbm_extra["scale_pos_weight"] == pytest.approx(1 / 9)
    assert all(len(s.fit_pos) == 100 and s.val_pos is None for s in specs.values())


# --- drift --------------------------------------------------------------------------------
def test_drift_specs_sliding_window_and_size_control() -> None:
    weeks = np.repeat(np.arange(52), 20)
    labels = np.tile([0, 1], len(weeks) // 2)
    sliding, control = drift.drift_specs(weeks, labels)
    assert set(np.unique(weeks[sliding.fit_pos])) == set(range(40, 52))
    assert len(control.fit_pos) == len(sliding.fit_pos)
    assert weeks[control.fit_pos].min() < 40  # drawn from all weeks, not just the recent ones


def test_weekly_table_one_row_per_model_and_week() -> None:
    y, p = _scored(1200)
    weeks = np.repeat([52, 53, 54], 400)
    table = drift.weekly_table(y, weeks, {"expanding": p, "sliding_12w": p ** 2}, n_resamples=30)
    assert len(table) == 6 and set(table["week"]) == {52, 53, 54}
    sliding = table[table["model"] == "sliding_12w"]
    assert "roc_auc_minus_expanding" in sliding and sliding["roc_auc_minus_expanding"].abs().max() < 1e-9  # p**2 keeps the ranking


# --- challenge ----------------------------------------------------------------------------
class _FakeExtractor:
    dim = 3

    def process_raw_features(self, row: dict) -> np.ndarray:
        return np.array([row["general"]["size"], 0.0, 1.0], dtype=np.float32)


def _raw(sha: str, file_type: str = "Win64", label: int = 1) -> bytes:
    return orjson.dumps({"sha256": sha, "label": label, "file_type": file_type, "first_submission_date": 1_700_000_000,
                         "week_id": 8, "detection_ratio": "3/70", "general": {"size": 123}, "packer": [],
                         "authenticode": {}}) + b"\n"


def test_challenge_rows_filtered_and_deduped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import data

    monkeypatch.setattr(data, "get_extractor", lambda: _FakeExtractor())
    f = tmp_path / "2023-01-01_2023-01-07_challenge_malicious.jsonl"
    f.write_bytes(_raw("a") + _raw("a") + _raw("b", "Win32") + _raw("c", label=-1) + b"\n" + _raw("d"))
    vectors, metas, counts = challenge.read_rows([f])
    assert [m["sha256"] for m in metas] == ["a", "d"]          # Win64 only, first copy wins, bad label dropped
    assert counts == {"lines": 5, "by_file_type": {"Win64": 4, "Win32": 1}, "duplicates": 1, "dropped_bad_label": 1}
    assert metas[1]["source_offset"] == len(_raw("a")) * 2 + len(_raw("b", "Win32")) + len(_raw("c", label=-1)) + 1
    assert np.vstack(vectors).shape == (2, 3)
