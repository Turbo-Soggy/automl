"""Build the Step 2 tables, figures and write-up from the experiment log and saved predictions only.

Nothing here is typed by hand: every number comes from reports/experiments.jsonl or from the
test-set scores that tune.final_evaluation() saved next to it.

Selection rules, so a log with several runs still gives one clear answer:
  * Only records of the current mode are used (quick runs and real runs never mix).
  * Per method, only final_eval records of the latest refit_version count (records written
    before refit_version existed are version 1), and of those the most recent one wins.
    Older versions are never deleted from the log: a newer version supersedes them.
  * Every reported method must be at the CURRENT version (tune.REFIT_VERSION). If a refit run
    stopped partway, build_tuning_outputs() refuses to mix versions (check_refit_complete()).
  * Convergence curves and the trials table use only the trials of that winning record's search.
Outputs get the "_quick" suffix in QUICK mode so smoke runs never overwrite real results.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed (also safe in tests and on servers)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    EXPERIMENT_LOG, FIGURES, LEADERBOARDS, PREDICTIONS, QUICK, REPORTS, ROOT, RUN_SUFFIX, SEED, TABLES,
)
from src import tune  # noqa: E402
from src.eval import paired_bootstrap, percentile_ci  # noqa: E402
from src.logging_utils import read_experiments  # noqa: E402

METHOD_ORDER: list[str] = ["lgbm_default", "logreg", "grid", "random", "optuna"]
TUNERS: list[str] = ["grid", "random", "optuna"]
TEST_SETS: list[str] = ["test_random", "test_future"]
# One bootstrap seed per test set: paired across methods (same rows in each resample), but a
# separate random stream per test set, so the two sets' resamples are independent; the
# random-vs-future gap CI assumes that. The write-up prints these seeds from here.
BOOTSTRAP_SEEDS: dict[str, int] = {test: SEED + i for i, test in enumerate(TEST_SETS)}
TEST_METRICS: list[str] = ["roc_auc", "f1", "accuracy", "tpr_at_1pct_fpr", "fpr"]
CI_METRICS: list[str] = ["roc_auc", "tpr_at_1pct_fpr"]  # the metrics that get bootstrap CIs
# Pairwise comparisons (a minus b): every tuner against every other, and each against no tuning.
PAIRS: list[tuple[str, str]] = [
    ("optuna", "random"), ("optuna", "grid"), ("random", "grid"),
    ("grid", "lgbm_default"), ("random", "lgbm_default"), ("optuna", "lgbm_default"),
]

# Validated categorical palette (slots 1-3 pass all-pairs CVD checks on a light surface).
SERIES_COLORS: list[str] = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_MUTED, GRID_INK = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LINE_STYLES: list[str] = ["-", "--", ":"]
# Refit version 1 of the tuners held out this share of train for early stopping
# (train_test_split(test_size=0.1), which rounds the held-out part UP), so it fit on fewer rows
# than its logged train_rows says. Used only to report the rows version 1 really fit on.
V1_HOLDOUT_SHARE: float = 0.10

Samples = dict[str, dict[str, dict[str, np.ndarray]]]  # test set -> method -> metric -> bootstrap values


@dataclass(frozen=True)
class TablePaths:
    """Input log and predictions, and output folders. Tests and smoke runs point these at a temp dir."""

    log: Path = EXPERIMENT_LOG
    tables: Path = TABLES
    figures: Path = FIGURES
    predictions: Path = PREDICTIONS
    reports: Path = REPORTS  # where the markdown write-up goes
    leaderboards: Path = LEADERBOARDS  # AutoML leaderboards and preprocessing notes (Step 3 write-up)


# --- Selecting records --------------------------------------------------------------------

def tuning_records(log: Path = EXPERIMENT_LOG) -> pd.DataFrame:
    """All Step 2 records of the current quick/non-quick mode, oldest first."""
    df = read_experiments(log)
    if df.empty or "step" not in df:
        return pd.DataFrame()
    df = df[(df["step"] == "tuning") & (df["quick"] == QUICK)]
    return df.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)


def final_evals(df: pd.DataFrame) -> pd.DataFrame:
    """Every final_eval record, with refit_version filled in (records from before the field existed are 1)."""
    finals = df[df["kind"] == "final_eval"].copy()
    versions = finals["refit_version"] if "refit_version" in finals else pd.Series(np.nan, index=finals.index)
    finals["refit_version"] = versions.fillna(1).astype(int)
    return finals


def latest_final_evals(df: pd.DataFrame) -> pd.DataFrame:
    """Per method: the most recent final_eval of its latest refit_version, in METHOD_ORDER."""
    finals = final_evals(df)
    newest = finals.groupby("method")["refit_version"].transform("max")
    finals = finals[finals["refit_version"] == newest].groupby("method", sort=False).tail(1)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return finals.sort_values("method", key=lambda s: s.map(order).fillna(len(order))).reset_index(drop=True)


def check_refit_complete(finals: pd.DataFrame, required: list[str] | None = None) -> None:
    """Fail with a clear message unless every reported method is at the current refit version.

    latest_final_evals() picks the newest version PER METHOD, so a refit run that died partway
    would otherwise report e.g. version-2 baselines next to version-1 tuners. `required` lists
    methods that must be present at all: without it, a method with no final_eval would simply
    be missing from the summary instead of failing.
    """
    missing = [m for m in (required or []) if m not in set(finals["method"])]
    if missing:
        raise RuntimeError(f"No final_eval record for: {', '.join(missing)}. "
                           "Run scripts/run_tuning.py --methods <those methods> (or --refit-only).")
    stale = finals.loc[finals["refit_version"] != tune.REFIT_VERSION, ["method", "refit_version"]]
    if not stale.empty:
        listing = ", ".join(f"{m} (version {v})" for m, v in stale.itertuples(index=False))
        raise RuntimeError(f"Refit incomplete for: {listing}; current refit version is {tune.REFIT_VERSION}. "
                           "Re-run scripts/run_tuning.py --refit-only (it reuses the logged searches).")


def trials_for(df: pd.DataFrame, search_id: str) -> pd.DataFrame:
    """The trials of one search, in trial order."""
    trials = df[(df["kind"] == "tune_trial") & (df["search_id"] == search_id)]
    return trials.sort_values("trial").reset_index(drop=True)


# --- Saved predictions and the bootstrap --------------------------------------------------

def load_scores(finals: pd.DataFrame, predictions: Path) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """{test set: (y_true, {method: p_malicious})} from the files final_evaluation() saved.

    Fails loudly if a file belongs to a different final_eval record than the one being reported,
    or if two methods were scored on different rows (then a paired comparison would be meaningless).
    """
    out: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for _, rec in finals.iterrows():
        path = predictions / f"{rec['method']}{RUN_SUFFIX}.npz"
        eval_id = rec.get("eval_id")
        if not isinstance(eval_id, str) or not path.exists():
            raise RuntimeError(f"No saved predictions for {rec['method']!r}; re-run scripts/run_tuning.py "
                               "(--refit-only reuses the logged searches)")
        with np.load(path) as saved:
            if str(saved["eval_id"]) != eval_id:
                raise RuntimeError(f"{path} belongs to another evaluation than the logged record {eval_id}")
            for test in TEST_SETS:
                y, p = saved[f"{test}_y"].astype(int), saved[f"{test}_p"]
                if test in out and not np.array_equal(out[test][0], y):
                    raise RuntimeError(f"{rec['method']} was scored on different {test} rows than the others")
                out.setdefault(test, (y, {}))[1][rec["method"]] = p
    return out


def bootstrap_samples(scores: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]) -> Samples:
    """Paired bootstrap of CI_METRICS for every method, per test set."""
    samples: Samples = {}
    for test in TEST_SETS:
        y, preds = scores[test]
        samples[test] = paired_bootstrap(y, preds, seed=BOOTSTRAP_SEEDS[test])  # see BOOTSTRAP_SEEDS
    return samples


# --- Tables -------------------------------------------------------------------------------

def summary_table(df: pd.DataFrame, samples: Samples) -> pd.DataFrame:
    """One row per method: CV AUC, both test sets' metrics with 95% CIs, the gap, timings, refit facts."""
    rows = []
    for _, rec in latest_final_evals(df).iterrows():
        method = rec["method"]
        row = {
            "method": method,
            "model": rec.get("model"),
            "refit_version": rec["refit_version"],
            "n_trials": rec.get("n_trials"),
            "best_cv_auc": rec.get("cv_auc_mean"),
            "cv_auc_std": rec.get("cv_auc_std"),
        }
        for test in TEST_SETS:
            row[f"{test}_n"] = rec.get(f"{test}.n")
            for metric in TEST_METRICS:
                row[f"{test}_{metric}"] = rec.get(f"{test}.{metric}")
                if metric in CI_METRICS:
                    row[f"{test}_{metric}_ci_low"], row[f"{test}_{metric}_ci_high"] = \
                        percentile_ci(samples[test][method][metric])
        # Positive gap = how much the random split overstates performance on future weeks.
        for metric, name in (("roc_auc", "auc_gap"), ("tpr_at_1pct_fpr", "tpr_gap")):
            row[f"{name}_random_minus_future"] = row[f"test_random_{metric}"] - row[f"test_future_{metric}"]
            gap = samples["test_random"][method][metric] - samples["test_future"][method][metric]
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = percentile_ci(gap)
        row["search_seconds"] = rec.get("search_seconds")
        row["seconds_per_trial"] = (rec["search_seconds"] / rec["n_trials"]) if pd.notna(rec.get("n_trials")) else None
        row["fit_seconds"] = rec.get("fit_seconds")
        row["train_rows"] = rec.get("train_rows")
        row["refit_n_estimators"] = rec.get("refit_n_estimators")
        row["cv_mean_best_iteration"] = rec.get("cv_mean_best_iteration")
        row["cv_folds_at_round_cap"] = rec.get("cv_folds_at_round_cap")
        row["folds_hash"] = rec.get("folds_hash")
        row["run_id"] = rec.get("run_id")
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Nullable ints: baselines have no trials, and plain floats would print "36.0".
    for col in ("n_trials", "train_rows", "refit_n_estimators", "cv_folds_at_round_cap", "test_random_n", "test_future_n"):
        summary[col] = pd.to_numeric(summary[col]).round().astype("Int64")
    return summary


def pairwise_table(summary: pd.DataFrame, samples: Samples) -> pd.DataFrame:
    """Paired differences (a minus b) with 95% CIs: is one method really better, or is it test-set luck?"""
    point = summary.set_index("method")
    rows = []
    for test in TEST_SETS:
        for metric in CI_METRICS:
            for a, b in PAIRS:
                if a not in point.index or b not in point.index:
                    continue
                diff = samples[test][a][metric] - samples[test][b][metric]
                low, high = percentile_ci(diff)
                rows.append({
                    "test_set": test, "metric": metric, "method_a": a, "method_b": b,
                    "diff_a_minus_b": point.loc[a, f"{test}_{metric}"] - point.loc[b, f"{test}_{metric}"],
                    "ci_low": low, "ci_high": high,
                    "share_resamples_a_better": float(np.mean(diff > 0)),
                    # Significant at 95% when the interval excludes 0 (no multiple-comparison correction).
                    "significant_95": bool(low > 0 or high < 0),
                })
    return pd.DataFrame(rows)


def trials_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every trial of the reported searches, with the round-cap flag derived from best_iterations."""
    finals = latest_final_evals(df).set_index("method")
    parts = []
    for method in TUNERS:
        if method not in finals.index:
            continue
        trials = trials_for(df, finals.loc[method, "search_id"])
        # Older records did not log max_rounds; they all ran with the current MAX_ROUNDS.
        caps = trials["max_rounds"] if "max_rounds" in trials else pd.Series(tune.MAX_ROUNDS, index=trials.index)
        caps = caps.fillna(tune.MAX_ROUNDS).astype(int)
        n_capped = [tune.folds_at_round_cap(its, cap) for its, cap in zip(trials["best_iterations"], caps)]
        part = pd.DataFrame({
            "method": method,
            "search_id": trials["search_id"],
            "trial_number": trials["trial"].astype(int) + 1,  # 1-based, as in the convergence plot
            "cv_auc_mean": trials["cv_auc_mean"],
            "cv_auc_std": trials["cv_auc_std"],
            "best_so_far": trials["cv_auc_mean"].cummax(),
            "best_iterations": [";".join(str(int(i)) for i in its) for its in trials["best_iterations"]],
            "mean_best_iteration": [float(np.mean(its)) for its in trials["best_iterations"]],
            "n_folds_at_round_cap": n_capped,
            "hit_round_cap": [n > 0 for n in n_capped],
            "fit_seconds": trials["fit_seconds"],
            "tuning_rows": trials["n_rows"].astype(int),
        })
        part["is_best"] = part.index == int(trials["cv_auc_mean"].to_numpy().argmax())  # first max, as in tune
        for name in tune.SEARCH_SPACE:
            part[name] = trials[f"params.{name}"].to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def shared_startup_trials(trials: pd.DataFrame) -> int:
    """How many leading trials Random and Optuna share (identical parameters), read from the log."""
    if trials.empty or not {"random", "optuna"} <= set(trials["method"]):
        return 0
    params = list(tune.SEARCH_SPACE)
    rnd = trials[trials["method"] == "random"][params].to_numpy()
    opt = trials[trials["method"] == "optuna"][params].to_numpy()
    same = [np.array_equal(r, o) for r, o in zip(rnd, opt)]
    return len(same) if all(same) else same.index(False)


def rows_fit(train_rows: int, refit_version: int) -> int:
    """Rows a tuner's final model was really fit on.

    Version 1 logged len(train) as train_rows but fit on train minus its early-stopping hold-out
    (160,000 - ceil(0.1 * 160,000) = 144,000; the leaf counts of the saved version 1 models in
    models/refit_v1/ confirm it). Version 2 fits on all of train, as logged.
    """
    if refit_version == 1:
        return train_rows - math.ceil(V1_HOLDOUT_SHARE * train_rows)
    return train_rows


def model_file(model_path: str, refit_version: int) -> str:
    """Where a final_eval record's model file lives now, relative to the project root if inside it.

    The log is append-only, so a superseded record keeps its original model_path, which now holds
    the newer version's model. Superseded model files were moved to models/refit_v<version>/
    (same file name), so for those records the logged path would load the wrong model.
    Returns "(not kept)" when a superseded model was not preserved.
    """
    path = Path(model_path)
    if refit_version != tune.REFIT_VERSION:
        path = path.parent / f"refit_v{refit_version}" / path.name
        if not path.exists():  # e.g. old smoke-run models were overwritten, not moved
            return "(not kept)"
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:  # smoke runs write models outside the project
        return path.as_posix()


def refit_version_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each tuner: the latest test score under every refit_version, to show what the refit rule changed."""
    finals = final_evals(df)
    finals = finals[finals["method"].isin(TUNERS)].groupby(["method", "refit_version"], sort=False).tail(1)
    cols = {"method": "method", "refit_version": "refit_version", "refit_strategy": "refit_strategy",
            "test_random.roc_auc": "test_random_roc_auc", "test_future.roc_auc": "test_future_roc_auc",
            "test_future.tpr_at_1pct_fpr": "test_future_tpr_at_1pct_fpr"}
    out = finals[[c for c in cols if c in finals]].rename(columns=cols)
    out["train_rows_fit"] = [rows_fit(int(n), int(v)) for n, v in zip(finals["train_rows"], finals["refit_version"])]
    # Trees in the final model: version 1 logged the early-stopped count, version 2 the fixed count.
    trees = finals["refit_n_estimators"] if "refit_n_estimators" in finals else pd.Series(np.nan, index=finals.index)
    if "refit_best_iteration" in finals:
        trees = trees.fillna(finals["refit_best_iteration"])
    out["n_trees"] = pd.to_numeric(trees).round().astype("Int64").to_numpy()
    out["model_file"] = [model_file(str(p), int(v)) for p, v in zip(finals["model_path"], finals["refit_version"])]
    out["_order"] = out["method"].map({m: i for i, m in enumerate(TUNERS)})
    return out.sort_values(["_order", "refit_version"]).drop(columns="_order").reset_index(drop=True)


def _write_csv(table: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}{RUN_SUFFIX}.csv"
    table.to_csv(path, index=False)
    return path


# --- Figures ------------------------------------------------------------------------------

def _style(ax: plt.Axes, title: str) -> None:
    """Recessive axes and grid so the data marks carry the figure."""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID_INK, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_INK)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _grouped_bars(ax: plt.Axes, summary: pd.DataFrame, columns: list[str], labels: list[str],
                  title: str, zoom: bool, fmt: str = "%.3f", cap_at_one: bool = False) -> None:
    """Bars per method, one colour per column; values printed above each bar.

    A column with matching "<column>_ci_low/_ci_high" columns also gets 95% CI whiskers.
    cap_at_one (for scores that cannot exceed 1): the zoomed y-axis ends at 1.0 and, with no
    head-room left above the bars, the value labels go inside them.
    """
    x = np.arange(len(summary))
    width = 0.8 / len(columns)
    for i, (col, label) in enumerate(zip(columns, labels)):
        values = summary[col].astype(float).to_numpy()
        yerr = None
        if f"{col}_ci_low" in summary:
            yerr = np.vstack([values - summary[f"{col}_ci_low"].to_numpy(float),
                              summary[f"{col}_ci_high"].to_numpy(float) - values])
        bars = ax.bar(x + (i - (len(columns) - 1) / 2) * width, values, width * 0.92, yerr=yerr,
                      error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK}, label=label, color=SERIES_COLORS[i])
        # Vertical labels: three neighbouring AUC values like 0.977 would overlap if horizontal.
        if cap_at_one:
            ax.bar_label(bars, fmt=fmt, fontsize=7, color="white", label_type="center", rotation=90)
        else:
            ax.bar_label(bars, fmt=fmt, fontsize=7, color=INK_MUTED, padding=3, rotation=90)
    ax.set_xticks(x, summary["method"])
    if zoom:  # AUCs cluster near 1, so a 0-based axis would hide every difference
        finite = summary[columns].astype(float).to_numpy()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low, high = max(0.0, finite.min() - 0.02), min(1.0, finite.max())
            # head-room for the vertical value labels, unless they sit inside the bars
            ax.set_ylim(low, 1.0 if cap_at_one else high + 0.15 * (high - low))
    else:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # head-room for the value labels
    # Legend below the axis so it never covers a bar.
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(columns))
    _style(ax, title)


def plot_compare(summary: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    """Four panels: ROC-AUC (CV and both tests), TPR@1%FPR, F1, and search/fit time."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE)
    _grouped_bars(axes[0, 0], summary, ["best_cv_auc", "test_random_roc_auc", "test_future_roc_auc"],
                  ["CV (tuning subset)", "test_random", "test_future"],
                  "ROC-AUC (y-axis zoomed; whiskers = 95% bootstrap CI)", zoom=True)
    _grouped_bars(axes[0, 1], summary, ["test_random_tpr_at_1pct_fpr", "test_future_tpr_at_1pct_fpr"],
                  ["test_random", "test_future"], "TPR at 1% FPR (whiskers = 95% bootstrap CI)", zoom=False)
    _grouped_bars(axes[1, 0], summary, ["test_random_f1", "test_future_f1"],
                  ["test_random", "test_future"], "F1 at threshold 0.5", zoom=False)
    timing = summary.assign(search_seconds=summary["search_seconds"].fillna(0.0))
    _grouped_bars(axes[1, 1], timing, ["search_seconds", "fit_seconds"],
                  ["search (all trials)", "final fit"], "Wall-clock seconds (baselines: no search)", zoom=False, fmt="%.0f")
    fig.suptitle("Step 2: baselines vs tuned LightGBM", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tuning_compare{RUN_SUFFIX}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_convergence(trials: pd.DataFrame, shared: int, out_dir: Path = FIGURES) -> Path:
    """Best-so-far CV AUC against trial number for the three tuners (dots = individual trials)."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    if shared:
        # Shade the trials Random and Optuna share, so nobody reads them as a TPE result.
        ax.axvspan(0.5, shared + 0.5, color=GRID_INK, alpha=0.6, linewidth=0)
        ax.text(0.7, 0.02, f"trials 1-{shared}: random = optuna\n(TPE start-up samples)", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK_MUTED, va="bottom")
    # Distinct line styles as well as colours: curves coincide in the shared trials,
    # and a hidden line would look like a bug.
    for color, style, method in zip(SERIES_COLORS, LINE_STYLES, TUNERS):
        part = trials[trials["method"] == method] if not trials.empty else trials
        if part.empty:
            continue
        trial_no = part["trial_number"].to_numpy()
        scores = part["cv_auc_mean"].to_numpy(dtype=float)
        ax.scatter(trial_no, scores, s=14, color=color, alpha=0.35, linewidths=0)
        ax.step(trial_no, part["best_so_far"].to_numpy(dtype=float), where="post", color=color, linewidth=2,
                linestyle=style, label=f"{method} (best {scores.max():.4f})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("trial", color=INK_MUTED)
    ax.set_ylabel("mean CV ROC-AUC", color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "Tuning convergence: best-so-far CV AUC (line) and each trial (dots)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tuning_convergence{RUN_SUFFIX}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- Step 3: AutoML versus manual tuning ---------------------------------------------------

AUTOML_TOOLS: list[str] = ["flaml", "autogluon", "h2o"]
BASELINES: list[str] = ["lgbm_default", "logreg"]
TUNED_LABEL: str = "lgbm_tuned"  # the comparison table's name for the tuner Step 2 selected by CV
BASELINE_INTERPRETABILITY: dict[str, str] = {
    "lgbm_default": "single LightGBM model: native importances and TreeSHAP",
    "logreg": "linear model on standardized features: one coefficient per feature",
}


def _timing_valid(df: pd.DataFrame) -> pd.Series:
    """False only where the sleep guard flagged the run; records from before the guard count as valid."""
    if "timing_valid" not in df:
        return pd.Series(True, index=df.index)
    return df["timing_valid"].map(lambda v: v is not False and v != 0)


def automl_records(log: Path = EXPERIMENT_LOG) -> pd.DataFrame:
    """Per AutoML tool, its most recent VALID final_eval of the current mode, in AUTOML_TOOLS order.

    A run whose timed fit included sleep (timing_valid False) stays in the log but is skipped here;
    invalid_attempts counts how many such runs each tool had, so the write-up can say so.
    """
    df = read_experiments(log)
    if df.empty or "step" not in df:
        return pd.DataFrame()
    df = df[(df["step"] == "automl") & (df["kind"] == "final_eval") & (df["quick"] == QUICK)]
    valid = _timing_valid(df)
    invalid_counts = df.loc[~valid, "tool"].value_counts()
    df = df[valid].sort_values("timestamp_utc", kind="stable").groupby("tool", sort=False).tail(1)
    df = df.assign(invalid_attempts=df["tool"].map(invalid_counts).fillna(0).astype(int))
    order = {t: i for i, t in enumerate(AUTOML_TOOLS)}
    return df.sort_values("tool", key=lambda s: s.map(order)).reset_index(drop=True)


def selected_tuner(manual: pd.DataFrame) -> str:
    """The tuner with the best mean CV AUC: chosen on CV only, never on a test set."""
    tuners = manual[manual["method"].isin(TUNERS)]
    if tuners.empty:
        raise RuntimeError("No Step 2 tuner final_eval records; run scripts/run_tuning.py first")
    return str(tuners.loc[tuners["cv_auc_mean"].astype(float).idxmax(), "method"])


def manual_search_totals(manual: pd.DataFrame) -> tuple[float, float]:
    """(seconds of all three manual searches, seconds of all three refits), from the logged final_evals."""
    tuners = manual[manual["method"].isin(TUNERS)]
    return float(tuners["search_seconds"].sum()), float(tuners["fit_seconds"].sum())


def _comparison_row(label: str, kind: str, rec: pd.Series, samples: Samples, method: str) -> dict:
    """Test metrics (logged point values, bootstrap CIs from the saved scores) for one comparison row."""
    row: dict = {"method": label, "kind": kind}
    for test in TEST_SETS:
        for metric in TEST_METRICS:
            row[f"{test}_{metric}"] = rec.get(f"{test}.{metric}")
            if metric in CI_METRICS:
                row[f"{test}_{metric}_ci_low"], row[f"{test}_{metric}_ci_high"] = \
                    percentile_ci(samples[test][method][metric])
    # Positive gap = how much the random split overstates performance on future weeks (as in Step 2).
    for metric, name in (("roc_auc", "auc_gap"), ("tpr_at_1pct_fpr", "tpr_gap")):
        row[f"{name}_random_minus_future"] = row[f"test_random_{metric}"] - row[f"test_future_{metric}"]
        gap = samples["test_random"][method][metric] - samples["test_future"][method][metric]
        row[f"{name}_ci_low"], row[f"{name}_ci_high"] = percentile_ci(gap)
    return row


def check_tools_complete(automl: pd.DataFrame, require_all: bool) -> None:
    """A real run must report all three tools (D4); smoke runs may cover a subset."""
    missing = [t for t in AUTOML_TOOLS if t not in set(automl["tool"])]
    if missing and require_all:
        raise RuntimeError(f"No AutoML final_eval for: {', '.join(missing)}; "
                           f"run scripts/run_automl.py --tools {','.join(missing)}")


def automl_comparison_table(automl: pd.DataFrame, manual: pd.DataFrame, samples: Samples,
                            interpretability: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per AutoML tool plus tuned LightGBM, default LightGBM and LogReg (in that order).

    wall_seconds is "start to model ready": AutoML = cluster start + data conversion + fit (incl.
    anything after the budget); manual = the selected tuner's search + its final refit; baselines =
    their single fit. all_manual_seconds adds up all three manual searches and refits.
    Two overrun columns, because H2O's start-up and data import are not part of its search:
    over_budget_seconds = wall - budget, fit_over_budget_seconds = fit - budget.
    `interpretability` (tool -> text) replaces the logged note where the saved-model checks say
    more (automl_checks.interpretability_from_notes); the logged text stays in interpretability_logged.
    """
    tuner = selected_tuner(manual)
    by_method = manual.set_index("method")
    rows = []
    for _, rec in automl.iterrows():
        row = _comparison_row(rec["tool"], "automl", rec, samples, rec["method"])
        row.update({"best_model": rec["best_model"], "budget_seconds": rec["budget_s"],
                    "wall_seconds": rec["wall_s"], "fit_seconds": rec["fit_s"],
                    "convert_seconds": rec["convert_s"], "init_seconds": rec["init_s"],
                    "over_budget_seconds": rec["wall_s"] - rec["budget_s"],
                    "fit_over_budget_seconds": rec["fit_s"] - rec["budget_s"],
                    "overhead_seconds": rec["init_s"] + rec["convert_s"],
                    "interpretability": (interpretability or {}).get(rec["tool"], rec["interpretability"]),
                    "interpretability_logged": rec["interpretability"], "train_rows": rec["train_rows"],
                    "tool_version": rec.get("tool_version"), "run_id": rec["run_id"],
                    "heartbeat_max_gap_seconds": rec.get("heartbeat_max_gap_s", np.nan),
                    "invalid_attempts": rec.get("invalid_attempts", 0)})
        rows.append(row)
    search_all, refit_all = manual_search_totals(manual)
    for method in [tuner, *BASELINES]:
        if method not in by_method.index:
            continue
        rec = by_method.loc[method]
        label = TUNED_LABEL if method == tuner else method
        row = _comparison_row(label, "manual", rec, samples, method)
        search = float(rec["search_seconds"]) if pd.notna(rec.get("search_seconds")) else 0.0
        trees = rec.get("refit_n_estimators")
        model = (f"LightGBM via {method} search ({int(trees)} trees)" if method == tuner else str(rec.get("model")))
        row.update({"best_model": model, "budget_seconds": np.nan,
                    "wall_seconds": search + float(rec["fit_seconds"]), "fit_seconds": rec["fit_seconds"],
                    "search_seconds": search if method == tuner else np.nan,
                    "all_manual_seconds": search_all + refit_all if method == tuner else np.nan,
                    "interpretability": (f"single LightGBM model ({int(trees)} trees): native importances and TreeSHAP"
                                         if method == tuner else BASELINE_INTERPRETABILITY.get(method, "")),
                    "train_rows": rec.get("train_rows"), "run_id": rec.get("run_id")})
        rows.append(row)
    table = pd.DataFrame(rows)
    table["wall_minutes"] = table["wall_seconds"] / 60.0
    return table


GAP_SET: str = "gap_random_minus_future"  # pairwise rows comparing two models' random-vs-future gaps


def automl_pairs(methods: list[str]) -> list[tuple[str, str]]:
    """(a, b) pairs for automl_pairwise_table, keeping only methods that are in the table.

    Each tool against tuned and default LightGBM, default against tuned (the figure's reference
    bar), and every pair of tools (to rank the tools against each other).
    """
    tools = [t for t in AUTOML_TOOLS if t in methods]
    refs = [m for m in (TUNED_LABEL, "lgbm_default") if m in methods]
    pairs = [(tool, ref) for tool in tools for ref in refs]
    if {TUNED_LABEL, "lgbm_default"} <= set(methods):
        pairs.append(("lgbm_default", TUNED_LABEL))
    pairs += [(a, b) for i, a in enumerate(tools) for b in tools[i + 1:]]
    return pairs


def _pair_row(test_set: str, metric: str, a: str, b: str, point_diff: float, diff: np.ndarray,
              larger_is_better: bool = True) -> dict:
    """One pairwise row: point difference a - b and the percentile CI of its bootstrap values.

    larger_is_better=False for gap rows: a larger random-vs-future gap means a overstates more, which
    is worse, so "a better" there is the share of resamples where a's gap is the smaller one.
    """
    low, high = percentile_ci(diff)
    a_better = diff > 0 if larger_is_better else diff < 0
    return {"test_set": test_set, "metric": metric, "method_a": a, "method_b": b, "diff_a_minus_b": point_diff,
            "ci_low": low, "ci_high": high, "share_resamples_a_better": float(np.mean(a_better)),
            "significant_95": bool(low > 0 or high < 0)}  # no multiple-comparison correction


def automl_pairwise_table(table: pd.DataFrame, samples: Samples, sample_key: dict[str, str]) -> pd.DataFrame:
    """Paired bootstrap 95% CIs of a minus b for every pair of automl_pairs().

    Rows with test_set == GAP_SET compare random-vs-future gaps: (random_a - future_a) minus
    (random_b - future_b), from the same resamples as the gap CIs. They answer "does a overstate
    more than b?" and are made only against tuned LightGBM. In those rows share_resamples_a_better
    is the share where a's gap is smaller (a overstates less).
    """
    point = table.set_index("method")
    pairs = automl_pairs(list(point.index))
    rows = []
    for test in TEST_SETS:
        for metric in CI_METRICS:
            for a, b in pairs:
                diff = samples[test][sample_key[a]][metric] - samples[test][sample_key[b]][metric]
                rows.append(_pair_row(test, metric, a, b,
                                      point.loc[a, f"{test}_{metric}"] - point.loc[b, f"{test}_{metric}"], diff))
    gap_col = {"roc_auc": "auc_gap_random_minus_future", "tpr_at_1pct_fpr": "tpr_gap_random_minus_future"}
    for metric in CI_METRICS:
        for a, b in [(a, b) for a, b in pairs if b == TUNED_LABEL]:
            gap = {m: samples["test_random"][sample_key[m]][metric] - samples["test_future"][sample_key[m]][metric]
                   for m in (a, b)}
            rows.append(_pair_row(GAP_SET, metric, a, b,
                                  point.loc[a, gap_col[metric]] - point.loc[b, gap_col[metric]], gap[a] - gap[b],
                                  larger_is_better=False))  # a smaller overstatement is the better one
    return pd.DataFrame(rows)


def _diff_panel(ax: plt.Axes, pairwise: pd.DataFrame, metric: str, title: str, scale: float, fmt: str) -> None:
    """Each model minus tuned LightGBM (paired bootstrap 95% CI) on both test sets; 0 = tied."""
    rows = pairwise[(pairwise["method_b"] == TUNED_LABEL) & (pairwise["metric"] == metric)
                    & (pairwise["test_set"].isin(TEST_SETS))]
    methods = list(dict.fromkeys(rows["method_a"]))
    x, width = np.arange(len(methods)), 0.8 / len(TEST_SETS)
    for i, test in enumerate(TEST_SETS):
        part = rows[rows["test_set"] == test].set_index("method_a").reindex(methods)
        values = part["diff_a_minus_b"].to_numpy(float) * scale
        yerr = np.vstack([values - part["ci_low"].to_numpy(float) * scale,
                          part["ci_high"].to_numpy(float) * scale - values])
        bars = ax.bar(x + (i - (len(TEST_SETS) - 1) / 2) * width, values, width * 0.92, yerr=yerr, label=test,
                      color=SERIES_COLORS[i], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK})
        ax.bar_label(bars, fmt=fmt, fontsize=7, color=INK_MUTED, padding=3)
    ax.axhline(0.0, color=INK, linewidth=0.8)  # zero = no difference to tuned LightGBM
    ax.set_xticks(x, methods)
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.08 * (high - low), high + 0.08 * (high - low))  # room for the value labels
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(TEST_SETS))
    _style(ax, title)


def plot_automl_compare(table: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    """Six panels. Top: ROC-AUC, TPR@1%FPR (with CIs) and F1 on both test sets, y-axes zoomed and
    capped at 1.0. Bottom: AUC and TPR differences to tuned LightGBM (too small to see in the bars,
    but they are what the write-up discusses) and wall-clock minutes. logreg has no difference bar:
    it is far off that scale (its numbers are in the top row)."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 9), facecolor=SURFACE)
    tests = ["test_random", "test_future"]
    _grouped_bars(axes[0, 0], table, ["test_random_roc_auc", "test_future_roc_auc"], tests,
                  "ROC-AUC (y-axis zoomed; whiskers = 95% bootstrap CI)", zoom=True, fmt="%.4f", cap_at_one=True)
    _grouped_bars(axes[0, 1], table, ["test_random_tpr_at_1pct_fpr", "test_future_tpr_at_1pct_fpr"], tests,
                  "TPR at 1% FPR (y-axis zoomed; whiskers = 95% CI)", zoom=True, cap_at_one=True)
    _grouped_bars(axes[0, 2], table, ["test_random_f1", "test_future_f1"], tests,
                  "F1 at threshold 0.5 (y-axis zoomed)", zoom=True, cap_at_one=True)
    _diff_panel(axes[1, 0], pairwise, "roc_auc", f"ROC-AUC minus {TUNED_LABEL} (95% paired CI)", 1.0, "%+.5f")
    _diff_panel(axes[1, 1], pairwise, "tpr_at_1pct_fpr",
                f"TPR@1%FPR minus {TUNED_LABEL}, percentage points (95% paired CI)", 100.0, "%+.2f")
    _time_panel(axes[1, 2], table)
    fig.suptitle("Step 3: AutoML (FLAML, AutoGluon, H2O) vs manual tuning", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"automl_vs_manual{RUN_SUFFIX}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _time_panel(ax: plt.Axes, table: pd.DataFrame) -> None:
    """Wall-clock minutes to a ready model (log scale: 1 min to hours), plus all manual searches."""
    x = np.arange(len(table))
    bars = ax.bar(x, table["wall_minutes"], 0.55, color=SERIES_COLORS[0], label="start to model ready")
    ax.bar_label(bars, fmt="%.1f", fontsize=7, color=INK_MUTED, padding=2)
    if "all_manual_seconds" in table and table["all_manual_seconds"].notna().any():
        i = int(np.flatnonzero(table["all_manual_seconds"].notna())[0])
        minutes = float(table["all_manual_seconds"].iloc[i]) / 60.0
        ax.scatter([i], [minutes], marker="D", s=36, color=SERIES_COLORS[1], zorder=3,
                   label="all 3 manual searches + refits")
        ax.annotate(f"{minutes:.0f}", (i, minutes), textcoords="offset points", xytext=(8, -3),
                    fontsize=7, color=INK_MUTED)
    budgets = table["budget_seconds"].dropna()
    if not budgets.empty:
        ax.axhline(float(budgets.iloc[0]) / 60.0, color=INK_MUTED, linestyle="--", linewidth=1,
                   label=f"AutoML budget ({float(budgets.iloc[0]) / 60:.0f} min)")
    ax.set_yscale("log")
    ax.set_xticks(x, table["method"])
    ax.set_ylabel("minutes (log scale)", color=INK_MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    _style(ax, "Wall-clock minutes (not equal work: see write-up)")


def automl_interpretability(automl: pd.DataFrame, leaderboards: Path) -> dict[str, str]:
    """Interpretability notes refined by the saved-model checks in each tool's notes JSON (if any)."""
    from src.automl_checks import interpretability_from_notes  # local import: no AutoML library is loaded

    out = {}
    for tool in automl["tool"]:
        path = leaderboards / f"{tool}_preprocessing{RUN_SUFFIX}.json"
        text = interpretability_from_notes(tool, json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None
        if text:
            out[tool] = text
    return out


def build_automl_outputs(paths: TablePaths = TablePaths()) -> list[Path]:
    """Regenerate the Step 3 comparison table, pairwise table, figure and write-up from the log."""
    from src import automl_writeup  # local import: automl_writeup imports this module

    automl = automl_records(paths.log)
    if automl.empty:
        raise RuntimeError(f"No Step 3 final_eval records (quick={QUICK}) in {paths.log}; run scripts/run_automl.py")
    check_tools_complete(automl, require_all=not QUICK)
    manual = latest_final_evals(tuning_records(paths.log))
    if manual.empty:
        raise RuntimeError(f"No Step 2 final_eval records (quick={QUICK}) in {paths.log}; run scripts/run_tuning.py")
    check_refit_complete(manual)
    tuner = selected_tuner(manual)
    reported = manual[manual["method"].isin([tuner, *BASELINES])]  # all tuners stay in `manual` for the time totals
    # Bootstrap every model on the same resamples (paired), with the same seeds as Step 2.
    finals = pd.concat([reported[["method", "eval_id"]], automl[["method", "eval_id"]]], ignore_index=True)
    samples = bootstrap_samples(load_scores(finals, paths.predictions))
    table = automl_comparison_table(automl, manual, samples, automl_interpretability(automl, paths.leaderboards))
    sample_key = {**{t: f"automl_{t}" for t in automl["tool"]}, TUNED_LABEL: tuner,
                  **{b: b for b in BASELINES}}
    pairwise = automl_pairwise_table(table, samples, sample_key)
    written = [
        _write_csv(table, paths.tables, "automl_vs_manual"),
        _write_csv(pairwise, paths.tables, "automl_pairwise"),
        plot_automl_compare(table, pairwise, paths.figures),
    ]
    written.append(automl_writeup.write(
        automl_writeup.WriteupInputs(table, pairwise, tuner, paths.leaderboards), paths.reports, paths.figures))
    return written


def build_tuning_outputs(paths: TablePaths = TablePaths()) -> list[Path]:
    """Regenerate every Step 2 table, figure and the write-up. Returns the written paths (summary first)."""
    from src import tuning_writeup  # local import: tuning_writeup imports this module

    df = tuning_records(paths.log)
    if df.empty or "final_eval" not in set(df["kind"]):
        raise RuntimeError(f"No Step 2 final_eval records (quick={QUICK}) in {paths.log}")
    latest = latest_final_evals(df)
    # Never mix refit versions in one report; a real run must report all five methods (D3),
    # while smoke runs may cover a subset of methods.
    check_refit_complete(latest, required=None if QUICK else METHOD_ORDER)
    samples = bootstrap_samples(load_scores(latest, paths.predictions))
    summary = summary_table(df, samples)
    pairwise = pairwise_table(summary, samples)
    trials = trials_table(df)
    shared = shared_startup_trials(trials)
    versions = refit_version_table(df)
    written = [
        _write_csv(summary, paths.tables, "tuning_summary"),
        _write_csv(pairwise, paths.tables, "tuning_pairwise"),
        _write_csv(trials, paths.tables, "tuning_trials"),
        _write_csv(versions, paths.tables, "tuning_refit_versions"),
        plot_compare(summary, paths.figures),
        plot_convergence(trials, shared, paths.figures),
    ]
    written.append(tuning_writeup.write(
        tuning_writeup.WriteupInputs(summary, pairwise, trials, versions, shared), paths.reports, paths.figures))
    return written
