"""Step 4 (part B): drift monitor. How does the tuned LightGBM score week by week after training?

Three models of the SAME tuned configuration (same parameters, same fixed number of rounds):
  expanding     all of train, weeks 0-51: the saved Step 2 model (its saved test scores are reused)
  sliding_12w   only the last 12 training weeks (40-51), i.e. a model retrained on recent data
  size_matched  a random (label x week stratified) sample of weeks 0-51 with as many rows as
                sliding_12w: separates "recent data" from "less data" when the two are compared
Each is scored on every future week 52-63 of test_future (about 3,300 files per week), with
bootstrap CIs per week. Nothing is retrained on future weeks and no threshold is picked on them.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from config import FIGURES, RUN_SUFFIX  # noqa: E402
from src import tables  # noqa: E402
from src.eval import paired_bootstrap, percentile_ci  # noqa: E402
from src.temporal import DEFAULT_THRESHOLD, FitSpec, operating_metrics, stratified_sample  # noqa: E402

SLIDING_WEEKS: int = 12
DRIFT_METRICS: list[str] = ["roc_auc", "f1", "tpr_at_1pct_fpr"]
MODEL_ORDER: list[str] = ["expanding", "sliding_12w", "size_matched"]


def drift_specs(weeks: np.ndarray, labels: np.ndarray) -> list[FitSpec]:
    """FitSpecs for the sliding-window model and its size-matched control (the expanding model exists)."""
    last = int(weeks.max())
    recent = np.flatnonzero(weeks > last - SLIDING_WEEKS)
    strata = labels.astype(np.int64) * 1000 + weeks.astype(np.int64)
    control = stratified_sample(np.arange(len(weeks)), strata, len(recent))
    return [
        FitSpec("drift_sliding_12w", "lgbm_tuned", recent, save_model=True,
                note=f"weeks {last - SLIDING_WEEKS + 1}-{last} only"),
        FitSpec("drift_size_matched", "lgbm_tuned", control, save_model=True,
                note=f"random {len(recent)} rows of weeks {int(weeks.min())}-{last} (size control)"),
    ]


def weekly_table(y: np.ndarray, weeks: np.ndarray, preds: dict[str, np.ndarray],
                 n_resamples: int = config.N_BOOTSTRAP, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Long table: one row per (model, week) with AUC, F1 and TPR@1%FPR and their 95% CIs.

    Resamples are paired across the models within a week, so the model-vs-model difference in a
    week can be read off as well (sliding_minus_expanding columns).
    """
    rows = []
    for week in np.unique(weeks):
        mask = weeks == week
        samples = paired_bootstrap(y[mask], {m: p[mask] for m, p in preds.items()},
                                   metric_fn=lambda yy, pp: operating_metrics(yy, pp, threshold),
                                   n_resamples=n_resamples, seed=config.SEED + int(week))
        for model, p in preds.items():
            point = operating_metrics(y[mask], p[mask], threshold)
            row = {"model": model, "week": int(week), "n": int(mask.sum()), "n_benign": int((y[mask] == 0).sum())}
            for metric in DRIFT_METRICS:
                row[metric] = point[metric]
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = percentile_ci(samples[model][metric])
                if model != "expanding" and "expanding" in preds:
                    diff = samples[model][metric] - samples["expanding"][metric]
                    row[f"{metric}_minus_expanding"] = point[metric] - operating_metrics(
                        y[mask], preds["expanding"][mask], threshold)[metric]
                    row[f"{metric}_minus_expanding_ci_low"], row[f"{metric}_minus_expanding_ci_high"] = \
                        percentile_ci(diff)
            rows.append(row)
    return pd.DataFrame(rows)


def overall_rows(y: np.ndarray, preds: dict[str, np.ndarray], test: str) -> pd.DataFrame:
    """Whole-test-set metrics of each drift model (week = 'all'), for the summary in the write-up."""
    rows = []
    for model, p in preds.items():
        point = operating_metrics(y, p)
        rows.append({"model": model, "week": f"all_{test}", "n": len(y), "n_benign": int((y == 0).sum()),
                     **{m: point[m] for m in DRIFT_METRICS}})
    return pd.DataFrame(rows)


def trend_slope(table: pd.DataFrame, metric: str) -> dict[str, float]:
    """Least-squares slope of the weekly metric per model (change per week): a simple drift number."""
    out = {}
    weekly = table[pd.to_numeric(table["week"], errors="coerce").notna()]
    for model, part in weekly.groupby("model"):
        out[model] = float(np.polyfit(part["week"].astype(float), part[metric].astype(float), 1)[0])
    return out


def plot_drift(table: pd.DataFrame, out_dir=FIGURES) -> str:
    """One panel per metric, one line per model, 95% CI bands."""
    weekly = table[pd.to_numeric(table["week"], errors="coerce").notna()].copy()
    weekly["week"] = weekly["week"].astype(int)
    fig, axes = plt.subplots(1, len(DRIFT_METRICS), figsize=(13, 3.8), facecolor=tables.SURFACE)
    titles = {"roc_auc": "ROC-AUC", "f1": "F1 at 0.5", "tpr_at_1pct_fpr": "TPR at 1% FPR"}
    for ax, metric in zip(axes, DRIFT_METRICS):
        for i, model in enumerate(m for m in MODEL_ORDER if m in set(weekly["model"])):
            part = weekly[weekly["model"] == model].sort_values("week")
            color = tables.SERIES_COLORS[i % len(tables.SERIES_COLORS)]
            ax.plot(part["week"], part[metric], tables.LINE_STYLES[i % 3], color=color, marker="o", ms=3, label=model)
            ax.fill_between(part["week"], part[f"{metric}_ci_low"], part[f"{metric}_ci_high"], color=color, alpha=0.15)
        tables._style(ax, titles[metric])
        ax.set_xlabel("future week (test_future)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Drift monitor: tuned LightGBM per future week (bands = 95% bootstrap CI)", color=tables.INK)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"drift{RUN_SUFFIX}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)
