"""Step 4 part C: calibration and the cost-sensitive decision threshold.

Three pieces, all on saved artefacts so every number can be rebuilt:
  * oof_predictions(): out-of-fold scores on TRAIN for the tuned LightGBM configuration
    (persisted 3-fold CV of the full train split, same fixed round count as the refit).
    These are the only scores used to fit a calibrator or choose a threshold, so neither
    test set ever influences a decision.
  * calibration metrics: Brier score, log loss and expected calibration error (ECE), plus
    reliability-curve points, on test_future for the tuned LightGBM and the AutoML winners.
  * cost threshold: the threshold that minimises C_FN*FN + C_FP*FP on the OOF scores.

Why OOF and not a single hold-out: every train row gets a score from a model that never saw
it, so the threshold is chosen on 160k honest scores instead of a small slice.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

COST_RATIOS: tuple[float, ...] = (1.0, 5.0, 10.0, 50.0)   # C_FN / C_FP values for the sensitivity table
DEFAULT_COST_RATIO: float = 10.0                           # a missed malware costs 10 false alarms
N_CAL_BINS: int = 15                                       # equal-width bins for ECE and the reliability curve
EPS: float = 1e-7                                          # clip for log loss / logit of saturated scores


# --- Out-of-fold scores on train ------------------------------------------------------------

def oof_predictions(X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
                    fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """Score every row with a model fitted on the other folds.

    fit(X_fit, y_fit) returns a predict function (X -> P(malicious)). Raises if the folds do
    not cover every row exactly once, because a gap would silently leave scores at zero.
    """
    oof = np.full(len(y), np.nan)
    for fit_idx, val_idx in folds:
        predict = fit(X[fit_idx], y[fit_idx])
        oof[val_idx] = predict(X[val_idx])
    if np.isnan(oof).any():
        raise ValueError("folds do not cover every row: some OOF scores are missing")
    return oof


# --- Calibration metrics ------------------------------------------------------------------

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = N_CAL_BINS) -> float:
    """Weighted mean |observed malicious rate - mean score| over equal-width score bins."""
    curve = reliability_curve(y, p, n_bins)
    return float((curve["n"] * (curve["observed"] - curve["mean_score"]).abs()).sum() / len(y))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = N_CAL_BINS) -> pd.DataFrame:
    """One row per non-empty bin: mean score, observed malicious rate and row count."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)   # p = 1.0 goes into the last bin
    frame = pd.DataFrame({"bin": bins, "y": y, "p": p})
    grouped = frame.groupby("bin").agg(mean_score=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return grouped.reset_index()


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Brier score, log loss and ECE of one score vector (lower is better for all three)."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece": expected_calibration_error(y, p)}


def brier_only(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Metric function for eval.paired_bootstrap (cheap enough for 1000 resamples)."""
    return {"brier": float(np.mean((np.asarray(p, dtype=float) - y) ** 2))}


@dataclass(frozen=True)
class Calibrator:
    """A fitted post-hoc calibrator: name plus the function that maps raw scores to new scores."""

    name: str
    transform: Callable[[np.ndarray], np.ndarray]


def fit_isotonic(y: np.ndarray, p: np.ndarray) -> Calibrator:
    """Monotone step function from score to observed rate (flexible; needs many rows, which OOF has)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return Calibrator("isotonic", lambda s: iso.predict(np.asarray(s, dtype=float)))


def fit_platt(y: np.ndarray, p: np.ndarray) -> Calibrator:
    """Logistic regression on the logit of the score (two parameters: a slope and an offset)."""
    def logit(s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), EPS, 1 - EPS)
        return np.log(s / (1 - s)).reshape(-1, 1)

    lr = LogisticRegression(C=1e6).fit(logit(p), y)   # large C: plain Platt scaling, no shrinkage
    return Calibrator("platt", lambda s: lr.predict_proba(logit(s))[:, 1])


# --- Cost-sensitive threshold -------------------------------------------------------------

def cost_curve(y: np.ndarray, p: np.ndarray, cost_ratio: float, c_fp: float = 1.0) -> pd.DataFrame:
    """Total cost C_FN*FN + C_FP*FP at every distinct score used as threshold (flag if p >= t).

    One sort gives FN and FP for every candidate threshold at once, so the minimum is exact,
    not a grid approximation. The row with threshold = inf flags nothing (all malware missed).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    # Last position of each distinct score: flagging "p >= t" flags everything up to there.
    last = np.r_[np.nonzero(np.diff(p_sorted))[0], len(p_sorted) - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = np.cumsum(1 - y_sorted)[last]
    n_pos = int(y.sum())
    thresholds = np.r_[np.inf, p_sorted[last]]
    tp, fp = np.r_[0, tp], np.r_[0, fp]
    fn = n_pos - tp
    c_fn = cost_ratio * c_fp
    return pd.DataFrame({"threshold": thresholds, "tp": tp, "fp": fp, "fn": fn,
                         "cost": c_fn * fn + c_fp * fp})


def choose_threshold(y: np.ndarray, p: np.ndarray, cost_ratio: float, c_fp: float = 1.0) -> dict[str, float]:
    """The cost-minimising threshold. Ties go to the highest threshold (fewest alarms)."""
    curve = cost_curve(y, p, cost_ratio, c_fp)
    best = curve.loc[curve["cost"] == curve["cost"].min()].iloc[0]   # rows run high -> low threshold
    # Place the cut midway to the next lower score, so a model whose scores move slightly
    # (the deployed refit vs the fold models) is not decided by one exact float.
    lower = curve["threshold"].to_numpy()
    pos = int(best.name)
    t = float(best["threshold"])
    if np.isfinite(t) and pos + 1 < len(lower):
        t = float((t + lower[pos + 1]) / 2.0)
    return {"threshold": t, "cost": float(best["cost"]), "fn": int(best["fn"]), "fp": int(best["fp"]),
            "cost_ratio": float(cost_ratio), "c_fp": float(c_fp), "c_fn": float(cost_ratio * c_fp)}


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float, cost_ratio: float,
                 c_fp: float = 1.0) -> dict[str, float]:
    """Counts, rates and cost when flagging p >= threshold (cost also per 1000 files)."""
    y = np.asarray(y).astype(int)
    flag = np.asarray(p, dtype=float) >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    cost = cost_ratio * c_fp * fn + c_fp * fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "cost": float(cost), "cost_per_1000": 1000.0 * cost / len(y)}
