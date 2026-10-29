"""Step 4 synthesis: three tables that answer the research question head-on.

  precision_table()        every Step 2-3 model at every threshold fixed on train: TPR, FPR and
                           precision at 1%, 5% and 10% malware prevalence on BOTH test sets
                           (prevalence.csv has test_future only), so "random vs real" can be read
                           at a realistic base rate and not only at the 50/50 mix of the test sets.
  gap_vs_tuned_table()     each model's random-minus-future gap minus the tuned LightGBM's gap,
                           with a paired bootstrap CI: does any AutoML tool lose less (or more)
                           to drift than the manually tuned model?
  self_estimate_table()    each AutoML run's own validation AUC (the number its leaderboard shows,
                           measured on a random split of `train`) against its test_random and
                           test_future AUC: does the tool's own estimate warn about drift or a leak?

Test-set discipline: nothing here selects, tunes or picks a threshold. The thresholds are 0.5,
the 1%-FPR threshold from TRAIN out-of-fold scores and models/threshold.json (also from TRAIN
out-of-fold scores); the test sets are only scored, from the saved predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import EXPERIMENT_LOG, LEADERBOARDS, N_BOOTSTRAP, PREDICTIONS, QUICK, RUN_SUFFIX, TABLES
from src import bias, tables, temporal
from src.eval import paired_bootstrap, percentile_ci, threshold_free_metrics
from src.logging_utils import read_experiments

AUTOML_TOOLS: tuple[str, ...] = ("flaml", "autogluon", "h2o")
GAP_METRICS: tuple[str, ...] = ("roc_auc", "tpr_at_1pct_fpr")
LEAKAGE_VARIANTS: tuple[str, ...] = ("clean", "leak_detection_ratio", "leak_first_submission")


def _facts() -> dict[str, Any]:
    """Facts that src/eval_bias_report.py saved (train-OOF thresholds, the deployment threshold)."""
    path = TABLES / f"eval_bias_facts{RUN_SUFFIX}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/run_eval_bias.py --stages tables first")
    return json.loads(path.read_text(encoding="utf-8"))


def train_thresholds(facts: dict[str, Any], models: list[str]) -> dict[str, dict[str, float]]:
    """{model: {threshold name: value}}, every value fixed on train before any test set is scored.

    0.5 for every model; the 1%-FPR threshold on TRAIN out-of-fold scores where such scores exist;
    the cost-sensitive deployment threshold for the tuned model when models/threshold.json exists.
    """
    out = {m: {"0.5": temporal.DEFAULT_THRESHOLD} for m in models}
    for model, thr in facts.get("oof_1pct_fpr_thresholds", {}).items():
        if model in out:
            out[model]["train_oof_1pct_fpr"] = float(thr)
    tuned = facts.get("tuned_tuner")
    if tuned in out and str(facts.get("deployment_threshold_source", "")).startswith("models"):
        out[tuned]["deployment"] = float(facts["deployment_threshold"])
    return out


# --- 1. Precision at a realistic base rate, on both test sets ------------------------------
def precision_table(scores: temporal.Scores, thresholds: dict[str, dict[str, float]]) -> pd.DataFrame:
    """bias.prevalence_table() for test_random AND test_future, one row per model/threshold/set/prevalence."""
    parts = [bias.prevalence_table({model: (scores[test][0], scores[test][1][model])}, thresholds[model], test)
             for test in tables.TEST_SETS for model in thresholds if model in scores[test][1]]
    return pd.concat(parts, ignore_index=True)


# --- 2. Is any model's drift gap different from the tuned LightGBM's? ----------------------
def gap_vs_tuned_table(scores: temporal.Scores, tuned: str, n_resamples: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Per model and metric: its random-minus-future gap, and that gap minus the tuned model's gap.

    The resamples are paired across models within a test set (same rows in resample b, seeds as in
    random_vs_future.csv), so the difference of two gaps is itself a bootstrap sample and its CI
    tests "model A loses more to drift than model B" directly.
    """
    models = [m for m in temporal.STEP23_MODELS if m in scores["test_future"][1]]
    samples = {test: paired_bootstrap(scores[test][0], {m: scores[test][1][m] for m in models},
                                      metric_fn=threshold_free_metrics, n_resamples=n_resamples,
                                      seed=tables.BOOTSTRAP_SEEDS[test])
               for test in tables.TEST_SETS}
    point = {test: {m: threshold_free_metrics(scores[test][0], scores[test][1][m]) for m in models}
             for test in tables.TEST_SETS}
    rows = []
    for model in models:
        for metric in GAP_METRICS:
            def gap(m: str, source: dict) -> Any:
                return source["test_random"][m][metric] - source["test_future"][m][metric]
            diff = gap(model, samples) - gap(tuned, samples)
            low, high = percentile_ci(diff)
            rows.append({"model": model, "kind": temporal.KIND[model], "metric": metric,
                         "gap": gap(model, point), "tuned_gap": gap(tuned, point),
                         "gap_minus_tuned": gap(model, point) - gap(tuned, point),
                         "gap_minus_tuned_ci_low": low, "gap_minus_tuned_ci_high": high,
                         "differs_from_tuned": bool(model != tuned and (low > 0 or high < 0))})
    return pd.DataFrame(rows)


# --- 3. The AutoML tools' own estimates ----------------------------------------------------
def _own_estimate(tool: str, notes: dict[str, Any]) -> tuple[float, str]:
    """(validation AUC the tool reports, how it measured it), from its saved preprocessing notes."""
    if tool == "h2o":
        frames = notes.get("frames") or {}
        nfolds = notes.get("nfolds_used")
        how = (f"{nfolds}-fold CV on train" if isinstance(nfolds, int) and nfolds >= 2
               else f"random validation frame of train ({frames.get('validation', 'n/a')} rows)")
        return float(notes.get("leader_leaderboard_auc", np.nan)), how
    if tool == "flaml":
        if notes.get("eval_method") == "holdout":
            how = f"random stratified holdout of train ({notes.get('holdout_rows_expected', 'n/a')} rows)"
        else:
            how = f"{notes.get('n_splits')}-fold CV on train"
        return float(notes.get("best_val_auc", np.nan)), how
    return float(notes.get("best_val_auc", np.nan)), f"random holdout of train ({notes.get('holdout_rows', 'n/a')} rows)"


def _notes(folder: Path, tool: str) -> dict[str, Any] | None:
    path = folder / f"{tool}_preprocessing{RUN_SUFFIX}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _machine_of(step: str, log: Path) -> dict[str, str]:
    """{tool or model: 'N threads, machine'} for the latest records of one step, from the log."""
    df = read_experiments(log)
    if df.empty or "step" not in df:
        return {}
    df = df[(df["step"] == step) & (df["quick"] == QUICK)].sort_values("timestamp_utc", kind="stable")
    key = "tool" if step == "automl" else "model"
    out = {}
    for _, rec in df.groupby(key).tail(1).iterrows():
        cpus, system = rec.get("machine.cpu_count"), rec.get("machine.os")
        # Step 3 records predate the machine field: they all ran on the Steps 2-3 laptop.
        where = f"{int(cpus)}-CPU {str(system).split()[0]} VM" if pd.notna(cpus) else "Steps 2-3 laptop"
        out[rec[key]] = f"{int(rec['n_jobs'])} threads, {where}"
    return out


def same_scores_as_step3(tool: str) -> bool | None:
    """Did the Step 4 clean rerun give exactly the Step 3 test scores? None if either file is missing.

    A deterministic winner (same model, seed and data) reproduces bit for bit even on another
    machine; a time-budgeted search that ends elsewhere does not.
    """
    step3 = PREDICTIONS / f"automl_{tool}{RUN_SUFFIX}.npz"
    rerun = PREDICTIONS / "leakage" / f"{tool}__clean{RUN_SUFFIX}.npz"
    if not (step3.exists() and rerun.exists()):
        return None
    with np.load(step3) as a, np.load(rerun) as b:
        return all(np.array_equal(a[k], b[k]) for k in ("test_random_p", "test_future_p", "test_future_y"))


def self_estimate_table(log: Path = EXPERIMENT_LOG) -> pd.DataFrame:
    """One row per AutoML run (Step 3 at its budget, and the Step 4 leakage variants at theirs)."""
    rows = []
    step3 = pd.read_csv(TABLES / f"automl_vs_manual{RUN_SUFFIX}.csv").set_index("method")
    machines3 = _machine_of("automl", log)
    for tool in AUTOML_TOOLS:
        notes = _notes(LEADERBOARDS, tool)
        if notes is None or tool not in step3.index:
            continue
        auc, how = _own_estimate(tool, notes)
        rec = step3.loc[tool]
        rows.append({"tool": tool, "run": "step3", "variant": "clean", "budget_s": rec["budget_seconds"],
                     "machine": machines3.get(tool, "n/a"), "own_validation": how, "own_auc": auc,
                     "test_random_roc_auc": rec["test_random_roc_auc"],
                     "test_future_roc_auc": rec["test_future_roc_auc"]})
    leak_csv = TABLES / f"leakage{RUN_SUFFIX}.csv"
    if leak_csv.exists():
        leak = pd.read_csv(leak_csv).set_index(["model", "variant"])
        machines4 = _machine_of("leakage", log)
        for variant in LEAKAGE_VARIANTS:
            for tool in AUTOML_TOOLS:
                notes = _notes(LEADERBOARDS / "leakage" / variant, tool)
                if notes is None or (tool, variant) not in leak.index:
                    continue
                auc, how = _own_estimate(tool, notes)
                rec = leak.loc[(tool, variant)]
                rows.append({"tool": tool, "run": "step4_leakage", "variant": variant, "budget_s": rec["budget_s"],
                             "machine": machines4.get(tool, "n/a"), "own_validation": how, "own_auc": auc,
                             "test_random_roc_auc": rec["test_random_roc_auc"],
                             "test_future_roc_auc": rec["test_future_roc_auc"],
                             "same_scores_as_step3": same_scores_as_step3(tool) if variant == "clean" else None})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["own_minus_random"] = table["own_auc"] - table["test_random_roc_auc"]
        table["own_minus_future"] = table["own_auc"] - table["test_future_roc_auc"]
    return table


def build_tables(n_resamples: int = N_BOOTSTRAP) -> dict[str, Path]:
    """Write the three CSVs to reports/tables/ (with the _quick suffix in quick mode)."""
    facts = _facts()
    scores = temporal.load_step23_scores()
    tuned = facts["tuned_tuner"]
    thresholds = train_thresholds(facts, list(scores["test_future"][1]))
    out = {"precision_both_sets": temporal.write_csv(precision_table(scores, thresholds), "precision_both_sets")}
    print("[tables] precision at prevalence on both test sets", flush=True)
    out["gap_vs_tuned"] = temporal.write_csv(gap_vs_tuned_table(scores, tuned, n_resamples), "gap_vs_tuned")
    print("[tables] drift gap of every model against the tuned LightGBM's", flush=True)
    out["automl_self_estimate"] = temporal.write_csv(self_estimate_table(), "automl_self_estimate")
    print("[tables] AutoML own validation estimates", flush=True)
    return out
