"""Evaluation metrics shared by every model and every tool, plus bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import N_BOOTSTRAP, SEED

MetricsFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]  # (y_true, p_mal) -> {metric: value}


def tpr_at_fpr(y_true: np.ndarray, p_mal: np.ndarray, max_fpr: float = 0.01) -> float:
    """Highest detection rate (TPR) reachable while flagging at most `max_fpr` of benign files.

    This is the security-relevant number: analysts tolerate few false alarms,
    so what matters is how much malware is caught at a fixed low FPR.
    drop_intermediate=False keeps every threshold, so the best point under max_fpr is never
    thinned away (the default drops collinear points, which can include that point).
    """
    fpr, tpr, _ = roc_curve(y_true, p_mal, drop_intermediate=False)
    return float(tpr[fpr <= max_fpr].max())


def compute_metrics(y_true: np.ndarray, p_mal: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """All headline metrics for binary malware detection (1 = malicious).

    Threshold-free: roc_auc, tpr_at_1pct_fpr.
    At `threshold`: f1, accuracy, precision, recall, fpr (share of benign files flagged).
    """
    y_true = np.asarray(y_true).astype(int)
    p_mal = np.asarray(p_mal, dtype=float)
    y_pred = (p_mal >= threshold).astype(int)
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2  # AUC is undefined on a one-class slice

    return {
        "roc_auc": float(roc_auc_score(y_true, p_mal)) if both_classes else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, p_mal, 0.01) if both_classes else float("nan"),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }


# --- Bootstrap confidence intervals -------------------------------------------------------

def threshold_free_metrics(y_true: np.ndarray, p_mal: np.ndarray, max_fpr: float = 0.01) -> dict[str, float]:
    """ROC-AUC and TPR at `max_fpr`, both read off ONE ROC curve.

    Same numbers as roc_auc_score() and tpr_at_fpr(), but with one sort instead of two, which
    halves the cost of the 1000-resample bootstrap below.
    """
    fpr, tpr, _ = roc_curve(y_true, p_mal, drop_intermediate=False)
    return {"roc_auc": float(auc(fpr, tpr)), "tpr_at_1pct_fpr": float(tpr[fpr <= max_fpr].max())}


def paired_bootstrap(y_true: np.ndarray, scores: dict[str, np.ndarray],
                     metric_fn: MetricsFn = threshold_free_metrics,
                     n_resamples: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, dict[str, np.ndarray]]:
    """Score every model on the same `n_resamples` bootstrap resamples of one test set.

    Returns {model: {metric: array of n_resamples values}}.
    "Paired" means that in resample b every model is scored on exactly the same rows. So
    samples[a][m] - samples[b][m] is a bootstrap sample of the *difference* between two models:
    the luck of which files landed in the test set hits both models alike and cancels out.
    That is what makes "model A beats model B" testable even when both CIs overlap.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = {model: np.asarray(p_mal, dtype=float) for model, p_mal in scores.items()}
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, np.ndarray]] = {model: {} for model in scores}
    for b in range(n_resamples):
        rows = rng.integers(0, len(y_true), size=len(y_true))  # n rows drawn with replacement
        for model, p_mal in scores.items():
            for name, value in metric_fn(y_true[rows], p_mal[rows]).items():
                out[model].setdefault(name, np.empty(n_resamples))[b] = value
    return out


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval, e.g. the 2.5th and 97.5th percentile for level 0.95."""
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)
