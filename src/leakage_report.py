"""Step 4 part A outputs: reports/tables/leakage.csv, reports/figures/leakage_jump.png, reports/leakage.md.

Built only from the experiment log (records with step "leakage"), the saved test-set scores in
reports/predictions/leakage/, and the saved run logs and tool notes. Nothing is typed by hand.

Selection rule: per (model, variant), the most recent final_eval of the current mode whose
timed fit did not include machine sleep (timing_valid). Older records stay in the log.

"Delta vs clean" is (variant minus clean) of the same model on the same test rows, with a
paired bootstrap 95% CI (the same resampled rows for both runs, seeds as in Steps 2-3).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    AUTOML_BUDGET_S, EXPERIMENT_LOG, FIGURES, LEAKAGE_BUDGET_S, QUICK, REPORTS, RUN_SUFFIX, TABLES,
)
from src import leakage  # noqa: E402
from src.eval import paired_bootstrap, percentile_ci  # noqa: E402
from src.logging_utils import read_records  # noqa: E402
from src.tables import BOOTSTRAP_SEEDS, GRID_INK, INK, INK_MUTED, SURFACE, _style  # noqa: E402

CI_METRICS: tuple[str, ...] = ("roc_auc", "tpr_at_1pct_fpr")
POINT_METRICS: tuple[str, ...] = ("roc_auc", "tpr_at_1pct_fpr", "f1", "accuracy", "fpr")
LOG_DIR: Path = REPORTS / "leakage_logs"
# Categorical slots 1-4 of the validated palette (checked with the dataviz validator: all pass;
# slots 3-4 are below 3:1 contrast on the light surface, so every bar also carries a value label).
VARIANT_COLORS: dict[str, str] = {leakage.CLEAN: "#2a78d6", leakage.LEAK_DR: "#eb6834",
                                  leakage.LEAK_DATE: "#1baf7a", leakage.ABLATE: "#eda100"}
VARIANT_LABELS: dict[str, str] = {leakage.CLEAN: "clean", leakage.LEAK_DR: "+ detection_ratio",
                                  leakage.LEAK_DATE: "+ first_submission_date", leakage.ABLATE: "- time proxies"}
# Words that would show a tool noticed something odd about a column. The leak column's own name,
# the variant names and our "leakage" output folder contain "leak", so they are blanked out first.
FLAG_WORDS: re.Pattern[str] = re.compile(
    r"leak|target|suspicious|correlat|perfect|too good|predictive|dominat|id[- ]like", re.IGNORECASE)
OUR_FOLDER: re.Pattern[str] = re.compile(r"[\\/]leakage[\\/]")
# A warning: the word on its own ("WARN", "Warning:") or a Python warning class ("UserWarning").
# Word boundaries keep feature names such as "pefilewarnings__1" from counting as warnings.
WARNING_WORDS: re.Pattern[str] = re.compile(r"\bwarn(ing|ings)?\b|[a-z]Warning\b", re.IGNORECASE)
TOOL_LOGS: dict[str, tuple[str, ...]] = {"flaml": ("flaml_fit_log.txt",), "autogluon": ("autogluon_fit_log.txt",),
                                         "h2o": ("h2o_event_log.csv",)}


# --- Records and predictions --------------------------------------------------------------

def worker_log_path(model: str, variant: str) -> Path:
    """Where scripts/run_leakage.py writes one run's stdout and stderr."""
    return LOG_DIR / f"{leakage.method_id(model, variant)}{RUN_SUFFIX}.log"


def latest_records(log: Path = EXPERIMENT_LOG) -> dict[tuple[str, str], dict[str, Any]]:
    """{(model, variant): newest valid leakage final_eval of the current mode}."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in read_records(log):  # oldest first, so later records overwrite earlier ones
        if (rec.get("step") == "leakage" and rec.get("kind") == "final_eval" and rec.get("quick") == QUICK
                and rec.get("timing_valid") is not False):
            latest[(rec["model"], rec["variant"])] = rec
    return latest


def load_predictions(rec: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{test set: (y, p)} of one record; refuses a file written by a different evaluation.

    The record's predictions_path is absolute on the machine that ran it (the Step 4 cloud VM
    logged /home/user/automl/...). When that path does not exist here, the file is found by its
    name under this checkout's predictions folder. The eval_id check below proves it is the right file.
    """
    path = Path(rec["predictions_path"])
    if not path.exists():
        path = leakage.LEAKAGE_PREDICTIONS / Path(str(rec["predictions_path"]).replace("\\", "/")).name
    with np.load(path) as saved:
        if str(saved["eval_id"]) != rec["eval_id"]:
            raise RuntimeError(f"{path} belongs to another evaluation than the logged record {rec['eval_id']}")
        return {t: (saved[f"{t}_y"].astype(int), saved[f"{t}_p"]) for t in leakage.TEST_SETS}


def delta_vs_clean(rec: dict[str, Any], clean: dict[str, Any]) -> dict[str, float]:
    """variant minus clean on each test set, AUC and TPR@1%FPR, with paired bootstrap 95% CIs."""
    ours, theirs, out = load_predictions(rec), load_predictions(clean), {}
    for test in leakage.TEST_SETS:
        y, p = ours[test]
        if not np.array_equal(y, theirs[test][0]):
            raise RuntimeError(f"{rec['method']} and its clean run were scored on different {test} rows")
        samples = paired_bootstrap(y, {"variant": p, "clean": theirs[test][1]}, seed=BOOTSTRAP_SEEDS[test])
        for metric in CI_METRICS:
            diff = samples["variant"][metric] - samples["clean"][metric]
            key = f"delta_{test}_{metric}"
            out[key] = rec[test][metric] - clean[test][metric]
            out[f"{key}_ci_low"], out[f"{key}_ci_high"] = percentile_ci(diff)
            out[f"{key}_significant"] = bool(out[f"{key}_ci_low"] > 0 or out[f"{key}_ci_high"] < 0)
    return out


# --- Did the tool notice? -----------------------------------------------------------------

def _log_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix == ".csv":  # H2O event log: "LEVEL: message", so H2O's own WARN level counts
        frame = pd.read_csv(path)
        if "message" not in frame:
            return []
        level = frame["level"].astype(str) if "level" in frame else pd.Series("", index=frame.index)
        return (level + ": " + frame["message"].astype(str)).tolist()
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _notes(model: str, variant: str) -> dict[str, Any]:
    path = leakage.LEAKAGE_LEADERBOARDS / variant / f"{model}_preprocessing{RUN_SUFFIX}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def scan_logs(model: str, variant: str) -> dict[str, Any]:
    """What the tool said about the leak column: warnings, flag words, and whether it dropped it.

    Reads the run's own stdout/stderr log (library warnings land there) and the tool's own log.
    Our driver's lines start with "[" + model, so they are skipped. Returns counts and the lines.
    """
    column = leakage.LEAK_COLUMN.get(variant)
    files = [worker_log_path(model, variant)]
    files += [leakage.LEAKAGE_LEADERBOARDS / variant / f"{Path(n).stem}{RUN_SUFFIX}{Path(n).suffix}"
              for n in TOOL_LOGS.get(model, ())]
    warnings, mentions, flags = [], [], []
    for path in files:
        for line in _log_lines(path):
            if line.startswith(("[leakage]", f"[{model}")):
                continue  # our own progress lines
            blanked = OUR_FOLDER.sub("/<folder>/", line.replace(column, "<col>") if column else line)
            for name in leakage.VARIANTS:
                blanked = blanked.replace(name, "<variant>")
            if WARNING_WORDS.search(line):
                warnings.append(line.strip())
            if column and column in line:
                mentions.append(line.strip())
            if FLAG_WORDS.search(blanked):
                flags.append(line.strip())
    dropped = _notes(model, variant).get("dropped_features")
    return {"files": [p.relative_to(REPORTS.parent).as_posix() for p in files if p.exists()],
            "n_warning_lines": len(warnings), "warning_lines": warnings[:30],
            "n_lines_mentioning_leak": len(mentions), "leak_mention_lines": mentions[:30],
            "n_flag_word_lines": len(flags), "flag_word_lines": flags[:30],
            "leak_dropped_by_tool": (column in dropped) if (column and isinstance(dropped, list)) else None}


# --- The table ----------------------------------------------------------------------------

def _rank(rec: dict[str, Any], source: str, column: str) -> dict[str, Any]:
    """rec["importance"][source]["ranks"][column], or {} if that importance was not computed."""
    return ((rec.get("importance") or {}).get(source) or {}).get("ranks", {}).get(column) or {}


def table_row(rec: dict[str, Any], clean: dict[str, Any] | None, scan: dict[str, Any]) -> dict[str, Any]:
    """One leakage.csv row: metrics with CIs, delta vs clean, watched-column ranks, log scan."""
    row: dict[str, Any] = {"model": rec["model"], "variant": rec["variant"], "leak_column": rec.get("leak_column"),
                           "best_model": rec.get("best_model"), "budget_s": rec.get("budget_s"),
                           "wall_s": rec.get("wall_s"), "n_features": rec.get("n_features"),
                           "train_rows": rec.get("train_rows")}
    for test in leakage.TEST_SETS:
        for metric in POINT_METRICS:
            row[f"{test}_{metric}"] = rec[test][metric]
        for metric in CI_METRICS:
            row[f"{test}_{metric}_ci_low"] = rec[test][f"{metric}_ci_low"]
            row[f"{test}_{metric}_ci_high"] = rec[test][f"{metric}_ci_high"]
    row["auc_gap_random_minus_future"] = rec["test_random"]["roc_auc"] - rec["test_future"]["roc_auc"]
    row["tpr_gap_random_minus_future"] = rec["test_random"]["tpr_at_1pct_fpr"] - rec["test_future"]["tpr_at_1pct_fpr"]
    if clean is not None and clean is not rec:
        row.update(delta_vs_clean(rec, clean))
    importance = rec.get("importance") or {}
    row["importance_native_kind"] = (importance.get("native") or {}).get("kind")
    row["importance_shap_kind"] = (importance.get("shap") or {}).get("kind") or \
        (importance.get("shap") or {}).get("skipped") or (importance.get("shap") or {}).get("error")
    column = rec.get("leak_column")
    if column:
        native, shap = _rank(rec, "native", column), _rank(rec, "shap", column)
        row.update({"leak_in_model": native.get("in_model", shap.get("in_model")),
                    "leak_rank_native": native.get("rank"), "leak_share_native": native.get("share"),
                    "leak_rank_shap": shap.get("rank"), "leak_share_shap": shap.get("share"),
                    "leak_rank_family": _rank(rec, "family", "leak").get("rank"),
                    "n_features_ranked": native.get("n_features")})
    for proxy in leakage.TIME_PROXIES:
        row[f"rank_native_{proxy.name}"] = _rank(rec, "native", proxy.name).get("rank")
        row[f"rank_shap_{proxy.name}"] = _rank(rec, "shap", proxy.name).get("rank")
    row.update({k: scan[k] for k in ("n_warning_lines", "n_lines_mentioning_leak", "n_flag_word_lines",
                                     "leak_dropped_by_tool")})
    row["relied_on_leak"] = relied_on_leak(row)
    row.update({"timing_valid": rec.get("timing_valid"), "heartbeat_max_gap_s": rec.get("heartbeat_max_gap_s"),
                "eval_id": rec["eval_id"], "run_id": rec["run_id"]})
    return row


# Native importances that count splits rather than measure contribution. FLAML's
# feature_importances_ of its LightGBM learner is the split count, which favours continuous
# high-cardinality columns (a raw date is split on often even when it adds little).
SPLIT_COUNT_KINDS: tuple[str, ...] = ("FLAML feature_importances_",)


def relied_on_leak(row: dict[str, Any]) -> bool | None:
    """True when the model kept the leak column AND ranked it first by its own importance or SHAP.

    That is what "fell for it" means here: the tool built its model mainly on the leak. A split-count
    importance is not evidence of reliance, so for those tools only the SHAP rank counts (when SHAP
    exists). None for variants without a leak column.
    """
    if not row.get("leak_column"):
        return None
    shap_rank = row.get("leak_rank_shap")
    split_count = str(row.get("importance_native_kind") or "").startswith(SPLIT_COUNT_KINDS)
    ranks = [shap_rank] if split_count and pd.notna(shap_rank) else [row.get("leak_rank_native"), shap_rank]
    return bool(row.get("leak_in_model")) and 1 in ranks


def build_table(records: dict[tuple[str, str], dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """leakage.csv rows in RUN_PLAN order (model-major), plus the full log scans for the JSON."""
    rows, scans = [], {}
    for model in leakage.MODEL_NAMES:
        for variant in leakage.VARIANTS:
            rec = records.get((model, variant))
            if rec is None:
                continue
            scan = scan_logs(model, variant)
            scans[leakage.method_id(model, variant)] = scan
            rows.append(table_row(rec, records.get((model, leakage.CLEAN)), scan))
    # Clean rows have no delta or leak columns; take the column order from the widest row.
    columns = list(dict.fromkeys(k for row in sorted(rows, key=len, reverse=True) for k in row))
    return pd.DataFrame(rows, columns=columns), scans


# --- Figure -------------------------------------------------------------------------------

def _zoomed_floor(values: np.ndarray) -> float:
    """y-axis floor that keeps differences near 1.0 visible (scores here are all close to 1)."""
    low = float(np.nanmin(values))
    return max(0.0, low - 0.35 * (1.0 - low))


def plot_jump(table: pd.DataFrame, out_dir: Path = FIGURES) -> Path:
    """2 x 2 panels (ROC-AUC, TPR@1%FPR) x (test_random, test_future): bars per model, one colour per variant."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor=SURFACE)
    models = [m for m in leakage.MODEL_NAMES if m in set(table["model"])]
    variants = [v for v in leakage.VARIANTS if v in set(table["variant"])]
    width = 0.8 / len(variants)
    for r, (metric, label) in enumerate((("roc_auc", "ROC-AUC"), ("tpr_at_1pct_fpr", "TPR at 1% FPR"))):
        values = table[[f"{t}_{metric}" for t in leakage.TEST_SETS]].to_numpy(float)
        floor = _zoomed_floor(values)
        for c, test in enumerate(leakage.TEST_SETS):
            ax = axes[r, c]
            for i, variant in enumerate(variants):
                part = table[table["variant"] == variant].set_index("model").reindex(models)
                y = part[f"{test}_{metric}"].to_numpy(float)
                err = np.vstack([y - part[f"{test}_{metric}_ci_low"].to_numpy(float),
                                 part[f"{test}_{metric}_ci_high"].to_numpy(float) - y])
                x = np.arange(len(models)) + (i - (len(variants) - 1) / 2) * width
                bars = ax.bar(x, y - floor, width * 0.9, bottom=floor, yerr=err, color=VARIANT_COLORS[variant],
                              label=VARIANT_LABELS[variant], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK})
                ax.bar_label(bars, labels=[("" if np.isnan(v) else f"{v:.4f}") for v in y], fontsize=6.5,
                             color=INK_MUTED, padding=2, rotation=90)
            ax.set_xticks(np.arange(len(models)), models)
            ax.set_ylim(floor, 1.0 + 0.45 * (1.0 - floor))  # head-room for the vertical value labels
            ax.set_yticks([t for t in ax.get_yticks() if floor <= t <= 1.0])  # no ticks above the 1.0 ceiling
            ax.axhline(1.0, color=GRID_INK, linewidth=1.0, linestyle="--")  # the ceiling of both metrics
            _style(ax, f"{label} on {test} (y-axis zoomed; whiskers = 95% bootstrap CI)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    budget = f"AutoML {LEAKAGE_BUDGET_S} s per run; LightGBM = Step 2 tuned config, fixed trees"
    fig.suptitle(f"Step 4: what one leaky column does to each model ({budget})", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"leakage_jump{RUN_SUFFIX}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- Entry point ----------------------------------------------------------------------------

def date_ranges() -> dict[str, str]:
    """First-submission date range of train and test_future (from the split metadata, no model)."""
    from src.data import load_cache_meta, split_rows

    out = {}
    for split, official in (("train", "train"), ("test_future", "test")):
        dates = load_cache_meta(official).iloc[split_rows(split)]["first_submission_date"]
        out[f"{split}_min"], out[f"{split}_max"] = str(dates.min()), str(dates.max())
        out[f"{split}_min_unix"] = int(leakage.unix_seconds(dates).min())
    return out


def build_outputs(log: Path = EXPERIMENT_LOG) -> list[Path]:
    """Write leakage.csv, the log-scan JSON, leakage_jump.png and leakage.md from the log."""
    from src import leakage_writeup

    records = latest_records(log)
    if not records:
        raise RuntimeError(f"No leakage final_eval records (quick={QUICK}) in {log}; run scripts/run_leakage.py")
    table, scans = build_table(records)
    TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES / f"leakage{RUN_SUFFIX}.csv"
    table.to_csv(csv_path, index=False, lineterminator="\n")
    scan_path = leakage.LEAKAGE_LEADERBOARDS / f"log_scan{RUN_SUFFIX}.json"
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    scan_path.write_text(json.dumps(scans, indent=2), encoding="utf-8", newline="\n")
    figure = plot_jump(table)
    context = {"records": records, "scans": scans, "dates": date_ranges(),
               "provenance": leakage.time_proxy_provenance(), "step3_budget_s": AUTOML_BUDGET_S}
    writeup = leakage_writeup.write(table, context, csv_path, figure)
    return [csv_path, scan_path, figure, writeup]
