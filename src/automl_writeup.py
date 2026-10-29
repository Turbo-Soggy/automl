"""Render reports/automl_comparison.md, the Step 3 write-up, from the generated tables and saved artefacts.

Inputs are only files the pipeline wrote: the comparison and pairwise tables (built by tables.py from
the experiment log and the saved test-set scores) and each tool's leaderboard, preprocessing notes
(including the saved-model "checks" from automl_checks.py), fit log, event log and importances in
reports/leaderboards/. Sentences that make a claim ("better", "tied", "dropped") are chosen by code
from those numbers, so the text follows the results when they change. The few library facts that
cannot be read from an artefact (e.g. "FLAML does not scale features") are stated with the library
version they were checked for. A missing artefact prints "not available" instead of failing, so a
partial (quick) run still renders.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import CV_FOLDS, N_BOOTSTRAP, N_JOBS, QUICK_BANNER, RUN_SUFFIX, SEED, TUNE_SUBSET_N, TUNE_TRIALS
from src.automl_checks import importance_type, short_model_id
from src.automl_runs import read_flaml_log
from src.tables import AUTOML_TOOLS, BOOTSTRAP_SEEDS, GAP_SET, TUNED_LABEL

TOOL_NAMES: dict[str, str] = {"flaml": "FLAML", "autogluon": "AutoGluon", "h2o": "H2O AutoML"}
METRIC_NAMES: dict[str, str] = {"roc_auc": "ROC-AUC", "tpr_at_1pct_fpr": "TPR@1%FPR"}
TOP_FAMILIES: int = 3
CHANCE_AUC: float = 0.6  # a learner whose best validation AUC is below this did no better than chance
# FLAML learners whose fit(budget) checks a deadline inside the fit (flaml/automl/model.py, FLAML 2.7.0).
FLAML_DEADLINE_LEARNERS: frozenset[str] = frozenset({"lgbm", "xgboost", "xgb_limitdepth", "catboost"})

# What happened while producing these results (crashes, fixes, re-runs). Written by us, not
# measured; kept here so the generated write-up always carries it.
RUN_NOTES: list[str] = [
    "Smoke test (`--quick`): AutoGluon's `feature_importance(data=None)` raised "
    "\"dataset cannot be None if feature_stage=='original'\". AutoGluon keeps its validation rows only "
    "after preprocessing, so the call now uses `feature_stage='transformed'` (same column names here, "
    "because all features are numeric). The smoke test was then re-run for all three tools.",
    "First full run (run id 20261028T105459Z): FLAML started, then the laptop went to sleep and the run "
    "stalled; it was killed before any tool logged a result. Its partial FLAML output was deleted and a "
    "sleep guard was added (a heartbeat every 30 s; a gap over 60 s marks the run invalid and it is re-run), "
    "because a timed budget that includes sleep is not the same budget as the other tools got. All three "
    "tools were then run again from scratch.",
]


@dataclass(frozen=True)
class WriteupInputs:
    """The tables (from src/tables.py), the tuner Step 2 selected by CV, and the artefact folder."""

    table: pd.DataFrame
    pairwise: pd.DataFrame
    tuner: str
    leaderboards: Path


# --- Reading artefacts --------------------------------------------------------------------

def _artifact(folder: Path, stem: str, ext: str) -> Path:
    return folder / f"{stem}{RUN_SUFFIX}{ext}"


def _csv(folder: Path, stem: str) -> pd.DataFrame | None:
    path = _artifact(folder, stem, ".csv")
    return pd.read_csv(path) if path.exists() else None


def _text(folder: Path, stem: str) -> str:
    path = _artifact(folder, stem, ".txt")
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _notes(folder: Path, tool: str) -> dict[str, Any]:
    path = _artifact(folder, f"{tool}_preprocessing", ".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _checks(notes: dict[str, Any]) -> dict[str, Any]:
    """The saved-model checks (automl_checks.py); {} for notes written before they existed."""
    return notes.get("checks") or {}


# --- Formatting helpers -------------------------------------------------------------------

def _f4(x: Any) -> str:
    return "n/a" if pd.isna(x) else f"{float(x):.4f}"


def _pct(x: Any) -> str:
    return "n/a" if pd.isna(x) else f"{100 * float(x):.2f}%"


def _value_ci(row: pd.Series, col: str, as_pct: bool) -> str:
    fmt = _pct if as_pct else _f4
    return f"{fmt(row[col])} [{fmt(row[col + '_ci_low'])}, {fmt(row[col + '_ci_high'])}]"


def _diff(x: float, metric: str) -> str:
    """AUC differences with 5 decimals (often < 0.0001), TPR differences in percentage points."""
    return f"{100 * x:+.2f} pp" if metric == "tpr_at_1pct_fpr" else f"{x:+.5f}"


def _span(pair: pd.Series, metric: str) -> str:
    return f"{_diff(pair['diff_a_minus_b'], metric)} [{_diff(pair['ci_low'], metric)}, {_diff(pair['ci_high'], metric)}]"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _minutes(seconds: Any) -> str:
    return "n/a" if pd.isna(seconds) else f"{float(seconds) / 60:.1f}"


def _tools(inp: WriteupInputs) -> list[str]:
    return [t for t in AUTOML_TOOLS if t in set(inp.table["method"])]


def _row(inp: WriteupInputs, method: str) -> pd.Series:
    return inp.table.set_index("method").loc[method]


def _pair(inp: WriteupInputs, a: str, b: str, test: str, metric: str) -> pd.Series | None:
    p = inp.pairwise
    hit = p[(p.method_a == a) & (p.method_b == b) & (p.test_set == test) & (p.metric == metric)]
    return None if hit.empty else hit.iloc[0]


def _status(pair: pd.Series) -> str:
    """'better' / 'worse' when the paired CI excludes 0, else 'tied'."""
    if not pair["significant_95"]:
        return "tied"
    return "better" if pair["diff_a_minus_b"] > 0 else "worse"


def _seed_text() -> str:
    return ", ".join(f"{seed} for `{test}`" for test, seed in BOOTSTRAP_SEEDS.items())


# --- Setup --------------------------------------------------------------------------------

def _setup_section(inp: WriteupInputs) -> str:
    tools = _tools(inp)
    first = _row(inp, tools[0])
    n_train = int(first["train_rows"])
    validation = [f"* {TOOL_NAMES[t]}: {_validation_text(t, _notes(inp.leaderboards, t), n_train)}" for t in tools]
    seeds = [f"* {TOOL_NAMES[t]}: {_seeds_text(t, _notes(inp.leaderboards, t))}" for t in tools]
    return f"""## Setup: what was the same and what was not

Same for every tool: the persisted `train` split ({int(first['train_rows']):,} rows x 2,568 float32 features,
named `family__i` so importances group by EMBER feature family), optimisation metric ROC-AUC,
{N_JOBS} threads, a search budget of {int(first['budget_seconds'])} s, and the tools ran one after another in separate
processes with no other heavy job running. Seed {SEED} was passed wherever a tool takes one, but not every
random choice inside the tools follows it (see "Seeds actually used"). Neither test set was loaded before a
tool's fit had finished; every leaderboard and importance below comes from the tool's own internal validation
on `train`.

Each tool chose its own validation inside `train`, and so used the 160,000 rows differently:

{chr(10).join(validation)}

Seeds actually used (read back from the saved models by `src/automl_checks.py`):

{chr(10).join(seeds)}

Manual tuning (Step 2) is different in kind: {TUNE_TRIALS} LightGBM configurations per tuner, each scored by
{CV_FOLDS}-fold CV on a {TUNE_SUBSET_N:,}-row subset of `train`; the best configuration (here `{inp.tuner}`,
picked by mean CV AUC, never by a test score) was then refit on all of `train`. In the tables below this
model is called `{TUNED_LABEL}`."""


def _validation_text(tool: str, notes: dict[str, Any], n_train: int) -> str:
    if not notes:
        return "notes not available"
    if tool == "flaml":
        if notes.get("eval_method") == "holdout":
            return (f"holdout of {notes.get('split_ratio')} of the rows (about {notes.get('holdout_rows_expected'):,} rows), "
                    f"then the winner is retrained on all rows (`retrain_full={notes.get('retrain_full')}`).")
        return f"{notes.get('n_splits')}-fold cross-validation (`eval_method={notes.get('eval_method')}`)."
    if tool == "autogluon":
        rows = notes.get("holdout_rows")
        if not rows:
            return str(notes.get("validation"))
        refit = _checks(notes).get("refit_on_all_rows")
        refit_txt = {True: "; the models were then refit on all rows", False: "; no refit on all rows"}.get(refit, "")
        return (f"{notes.get('validation')}; {rows:,} validation rows, so its models trained on the other "
                f"{n_train - rows:,} rows{refit_txt}.")
    return f"{notes.get('validation')}; no refit on all rows." if notes.get("validation") else "n/a"


def _seeds_text(tool: str, notes: dict[str, Any]) -> str:
    seeds = _checks(notes).get("seeds")
    if tool == "h2o":
        # H2O derives its models' seeds from the AutoML seed; nothing to read back per model here.
        return (f"`seed={SEED}` passed to `H2OAutoML`, which seeds its models and its frame split from it. H2O "
                "documents that a run limited by `max_runtime_secs` (as here) is not exactly reproducible even "
                "with a seed, because how many models fit in the time depends on the machine.")
    if not seeds:
        return "not checked (notes written before the saved-model checks; run `scripts/run_automl.py --refresh-notes`)."
    if tool == "flaml":
        return (f"search `seed={seeds.get('search (seed=)')}`; FLAML's holdout split uses its own fixed "
                f"`flaml.config.RANDOM_SEED={seeds.get('holdout split (flaml.config.RANDOM_SEED)')}`, and the winning "
                f"model kept FLAML's default `random_state={seeds.get('winner model random_state')}`.")
    models = seeds.get("base models (random_seed)", {})
    distinct = sorted({str(v) for v in models.values()})
    return (f"train/validation split `random_state={seeds.get('train/validation split (learner random_state)')}`; "
            f"its {len(models)} models used `random_seed` {', '.join(distinct)} (AutoGluon's default). AutoGluon "
            f"takes a per-model `random_seed` only inside `hyperparameters`, which this run did not set.")


# --- What each tool tried -----------------------------------------------------------------

def _flaml_section(folder: Path, budget_s: float) -> str:
    board, notes = _csv(folder, "flaml_leaderboard"), _notes(folder, "flaml")
    if board is None or not notes:
        return "### FLAML\n\nLeaderboard not available."
    full = int(board["max_sample_size"].max())  # the largest sample any learner reached = FLAML's full train part
    rows = [[r.estimator, int(r.n_trials), _f4(r.best_val_auc), f"{int(r.best_sample_size):,}",
             f"{r.total_trial_time_s:.0f}", _flaml_note(r, full)] for r in board.itertuples()]
    untried = sorted(set(notes.get("estimator_list", [])) - set(board["estimator"]))
    untried_txt = (f" Estimators in FLAML's list that never got a trial: {', '.join(f'`{e}`' for e in untried)}."
                   if untried else " Every estimator in FLAML's list got at least one trial.")
    config = json.dumps(notes.get("best_config", {}), sort_keys=True)
    return f"""### FLAML

FLAML searched {len(board)} estimator types with {notes.get('n_trials_total')} trials in total (from
`leaderboards/flaml_search_log{RUN_SUFFIX}.json`). It starts each learner on a small sample of rows and grows
the sample for promising ones, so "sample" below is the number of rows of the best trial.{untried_txt}

{_table(["estimator", "trials", "best validation AUC", "sample (rows)", "seconds in trials", "note"], rows)}

{_flaml_budget_text(folder, board, full, budget_s)}

Winner: `{notes.get('best_estimator')}` (validation AUC {_f4(notes.get('best_val_auc'))}, found after
{float(notes.get('time_to_find_best_model_s', float('nan'))):.0f} s), config `{config}`.

{_flaml_overrun_text(folder, notes, budget_s)}"""


def _flaml_note(r: Any, full: int) -> str:
    """Why a learner's row says little: cut to small samples, a single trial, or chance-level AUC."""
    notes = []
    if int(r.max_sample_size) < full:
        notes.append(f"never past {int(r.max_sample_size):,}-row samples")
    if int(r.n_trials) == 1:
        notes.append("1 trial only")
    if float(r.best_val_auc) < CHANCE_AUC:
        notes.append("no better than chance")
    return "; ".join(notes) or "-"


def _flaml_budget_text(folder: Path, board: pd.DataFrame, full: int, budget_s: float) -> str:
    """FLAML's own estimate of the budget it needed (first lines of its fit log)."""
    match = re.search(r"Estimated sufficient time budget=(\d+)s\. Estimated necessary time budget=(\d+)s",
                      _text(folder, "flaml_fit_log"))
    reached = [f"`{e}`" for e, n in zip(board["estimator"], board["max_sample_size"]) if int(n) == full]
    small = len(board) - len(reached)
    if not match:
        return f"Only {', '.join(reached)} reached the full {full:,}-row sample; {small} learners never did."
    return (f"FLAML's own log estimated at the start: \"Estimated sufficient time budget={match.group(1)}s. "
            f"Estimated necessary time budget={match.group(2)}s.\" The {budget_s:.0f} s budget is "
            f"{int(match.group(2)) / budget_s:.1f} times below the \"necessary\" estimate, so only "
            f"{', '.join(reached)} reached the full {full:,}-row sample; the other {small} learners were only ever "
            f"compared on small samples. Their rows say little about how they would do on this data with more time.")


def _flaml_trials(folder: Path) -> pd.DataFrame | None:
    path = _artifact(folder, "flaml_search_log", ".json")
    return read_flaml_log(path) if path.exists() else None


def _flaml_overrun_text(folder: Path, notes: dict[str, Any], budget_s: float) -> str:
    """Why FLAML ran past its budget, read from its search log and fit log (not assumed).

    FLAML passes the remaining budget to each trial. Its LightGBM, XGBoost and CatBoost wrappers stop
    at that deadline (flaml/automl/model.py, FLAML 2.7.0), but a scikit-learn learner such as lrl1 is
    one uninterruptible fit, so a trial of it started just before the deadline runs to its end. After
    the search FLAML retrains the winner on all rows.
    """
    trials = _flaml_trials(folder)
    if trials is None:
        return ""
    last = trials.loc[trials["wall_clock_time"].idxmax()]
    started = float(last["wall_clock_time"]) - float(last["trial_time"])
    longest = trials.loc[trials["trial_time"].idxmax()]
    retrain = next((re.search(r"retrain (\S+) for ([\d.]+)s", line) for line in notes.get("fit_log_highlights", [])
                    if re.search(r"retrain \S+ for [\d.]+s", line)), None)
    retrain_txt = (f" Then it retrained `{retrain.group(1)}` on all rows for {float(retrain.group(2)):.0f} s."
                   if retrain else "")
    kind = "" if str(last["learner"]) in FLAML_DEADLINE_LEARNERS else ", a fit that cannot be interrupted"
    return (f"Overrun: FLAML passes the remaining budget to each trial; its boosting learners (LightGBM, XGBoost, "
            f"CatBoost) stop at that deadline, but scikit-learn learners such as logistic regression cannot be "
            f"interrupted inside one fit. The last trial (`{last['learner']}`, "
            f"{float(last['trial_time']):.0f} s{kind}) started at {started:.0f} s, before the {budget_s:.0f} s budget was "
            f"used up, and ended at {float(last['wall_clock_time']):.0f} s; this one trial is most of FLAML's overrun. "
            f"The longest trial of the search was `{longest['learner']}` ({float(longest['trial_time']):.0f} s, "
            f"validation AUC {_f4(1 - float(longest['validation_loss']))}).{retrain_txt}")


def _ag_truncated(log_text: str) -> dict[str, str]:
    """Models AutoGluon stopped because their time ran out ("Ran out of time, early stopping on iteration N")."""
    out, current = {}, None
    for line in log_text.splitlines():
        start = re.match(r"Fitting model: (\S+)", line)
        if start:
            current = start.group(1)
        stop = re.search(r"Ran out of time, early stopping on iteration (\d+)", line)
        if stop and current:
            out[current] = f"stopped by the time limit at iteration {stop.group(1)}"
    return out


def _ag_preset(log_text: str) -> str | None:
    match = re.search(r"Using hyperparameters preset: (hyperparameters='\w+')", log_text)
    return match.group(1) if match else None


def _ag_section(folder: Path) -> str:
    board, notes = _csv(folder, "autogluon_leaderboard"), _notes(folder, "autogluon")
    if board is None or not notes:
        return "### AutoGluon\n\nLeaderboard not available."
    log_text = _text(folder, "autogluon_fit_log")
    truncated = _ag_truncated(log_text)
    weights = notes.get("ensemble_weights") or {}
    own = _ag_own_fit_seconds(board, weights)
    rows = [[r.model, _f4(r.score_val), f"{r.fit_time:.0f}",
             truncated.get(r.model) or (f"fit seconds include its base models; own fit {own[r.model]:.2f} s"
                                        if r.model in own else "-")] for r in board.itertuples()]
    log = notes.get("fit_log_highlights", [])
    planned = _ag_planned_models(log)
    started = [m for m in (re.match(r"Fitting model: (\S+)", line) for line in log) if m]
    started_names = [m.group(1) for m in started if not m.group(1).startswith("WeightedEnsemble")]
    failed = [m for m in started_names if m not in set(board["model"])]
    weights_txt = ", ".join(f"`{m}` {w:.2f}" for m, w in weights.items()) if weights else "single model"
    if planned is None:
        never = "the number of planned models is not in the log"
    elif planned > len(started_names):
        never = f"{planned - len(started_names)} of the {planned} planned base models were never started (no time left)"
    else:
        never = f"all {planned} planned base models were started"
    fail_txt = f" Started but not in the leaderboard (failed or stopped): {', '.join(failed)}." if failed else ""
    trunc_txt = (f" {', '.join(f'`{m}`' for m in truncated)} got only the time that was left and stopped early, so "
                 "its score says little about that model family on this data." if truncated else "")
    preset = _ag_preset(log_text)
    xt = "`extra_trees=True`" if "'extra_trees': True" in log_text else ""  # from the hyperparameter dump in the log
    preset_txt = (f" AutoGluon fitted every model with its preset hyperparameters (`{preset}`; medium_quality does no "
                  f"hyperparameter search), so `LightGBMXT` is LightGBM with {xt or 'extra trees'} and otherwise "
                  "AutoGluon's default settings." if preset else "")
    return f"""### AutoGluon

`medium_quality` preset: one holdout split, no bagging or stacking, base models fitted one after another until
the time runs out, then a weighted ensemble of them. {len(started_names)} base models were started;
{never}.{fail_txt}{trunc_txt}

{_table(["model", "validation AUC", "fit seconds (cumulative)", "note"], rows)}

Winner: `{notes.get('best_model')}` (validation AUC {_f4(notes.get('best_val_auc'))}); ensemble weights:
{weights_txt}.{preset_txt}"""


def _ag_own_fit_seconds(board: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    """Own fit time of each WeightedEnsemble row, whose leaderboard fit_time also counts its base models.

    Uses AutoGluon's fit_time_marginal when the leaderboard has it; older leaderboards did not keep it,
    so then: cumulative fit_time minus the fit_time of the base models with non-zero weight.
    """
    out = {}
    fit = board.set_index("model")["fit_time"]
    members = [m for m, w in weights.items() if w > 0]
    for r in board.itertuples():
        if not str(r.model).startswith("WeightedEnsemble"):
            continue
        if "fit_time_marginal" in board:
            out[r.model] = float(r.fit_time_marginal)
        elif members and all(m in fit.index for m in members):
            out[r.model] = float(r.fit_time) - float(fit[members].sum())
    return out


def _ag_planned_models(log: list[str]) -> int | None:
    """Base models AutoGluon planned to fit ("Fitting 11 L1 models"), read from its fit log."""
    for line in log:
        match = re.search(r"Fitting (\d+) L1 models", line)
        if match:
            return int(match.group(1))
    return None


def _h2o_events(folder: Path) -> list[str]:
    events = _csv(folder, "h2o_event_log")
    return [] if events is None or "message" not in events else events["message"].astype(str).tolist()


def _h2o_skipped(events: list[str], trained: set[str]) -> tuple[list[str], int]:
    """(planned non-ensemble steps H2O skipped for lack of time, number of skipped stacked ensembles)."""
    skipped = list(dict.fromkeys(m.group(1).split(" [")[0].strip()
                                 for m in (re.search(r"out of time; skipping (.+)", e) for e in events) if m))
    skipped = [s for s in skipped if s not in trained]  # some steps were retried and trained in the end
    ensembles = [s for s in skipped if s.startswith("StackedEnsemble")]
    return [s for s in skipped if s not in ensembles], len(ensembles)


def _h2o_training_times(events: list[str]) -> dict[str, int]:
    """Training seconds per model ("Adding model X to leaderboard ... Training time: model=519s")."""
    out = {}
    for e in events:
        match = re.search(r"Adding model (\S+) to leaderboard .*Training time: model=(\d+)s", e)
        if match:
            out[match.group(1)] = int(match.group(2))
    return out


def _h2o_section(folder: Path, budget_s: float) -> str:
    board, notes = _csv(folder, "h2o_leaderboard"), _notes(folder, "h2o")
    if board is None or not notes:
        return "### H2O AutoML\n\nLeaderboard not available."
    rows = [[short_model_id(r.model_id), r.algo, _f4(r.auc), f"{r.training_time_ms / 1000:.0f}"]
            for r in board.head(12).itertuples()]
    by_algo = ", ".join(f"{algo} {n}" for algo, n in notes.get("models_by_algo", {}).items())
    events = _h2o_events(folder)
    xgb = [e for e in events if "xgboost" in e.lower()]
    xgb_txt = f" Event log on XGBoost: \"{xgb[0].strip()}\"" if xgb else ""
    skipped, n_se_skipped = _h2o_skipped(events, set(board["model_id"]))
    skip_txt = (f" Out of time, H2O also skipped these planned steps: {', '.join(f'`{s}`' for s in skipped)}, "
                f"and {n_se_skipped} further stacked ensembles." if skipped else "")
    times = sorted(_h2o_training_times(events).items(), key=lambda kv: -kv[1])[:2]
    time_txt = (" Most of the budget went to " + " and ".join(
        f"`{short_model_id(m)}` ({s} s, {100 * s / budget_s:.0f}% of the budget)" for m, s in times) + "."
        if times else "")
    more = f" (top 12 of {len(board)} shown; full list in `leaderboards/h2o_leaderboard{RUN_SUFFIX}.csv`)" if len(board) > 12 else ""
    leader_auc = notes.get("leader_leaderboard_auc", notes.get("leader_cv_auc"))  # old notes used the CV name
    return f"""### H2O AutoML

H2O trained {notes.get('n_models')} models ({by_algo}). Validation: {notes.get('validation')}.
XGBoost is not available in H2O on Windows, so H2O's default plan ran without it.{xgb_txt}{skip_txt}{time_txt}

{_table(["model (suffix _AutoML_... dropped)", "algorithm", "leaderboard AUC", "training seconds"], rows)}{more}

Leader: `{notes.get('leader')}` ({notes.get('leader_algo')}, leaderboard AUC {_f4(leader_auc)}).
{_h2o_metalearner_text(board, notes)}
{_h2o_saved_text(notes.get('saved_model', {}))}"""


def _h2o_metalearner_text(board: pd.DataFrame, notes: dict[str, Any]) -> str:
    """Which base models the stacked-ensemble leader really uses (metalearner coefficients)."""
    checks = _checks(notes)
    coef = checks.get("metalearner_coefficients")
    if not coef:
        return "Metalearner weights: not checked."
    weights = ", ".join(f"`{short_model_id(k)}` {v:.3f}" for k, v in coef.items() if k != "Intercept")
    effective = checks.get("effective_base_models") or []
    se = board[board["algo"].astype(str).str.lower() == "stackedensemble"]
    same = se[(se["auc"] == board.loc[0, "auc"]) & (se["logloss"] == board.loc[0, "logloss"])]
    same_txt = (f" All {len(same)} stacked ensembles have the same leaderboard AUC and logloss to full precision, "
                "which fits them all ending up with the same single base model." if len(same) > 1 else "")
    if len(effective) == 1:
        return (f"Its GLM metalearner (non-negative weights: {checks.get('metalearner_non_negative')}) gives the base "
                f"models the weights {weights}, so the leader is in effect `{short_model_id(effective[0])}` passed "
                f"through a logistic calibration, not a blend of models.{same_txt}")
    return f"Its GLM metalearner gives the base models the weights {weights}.{same_txt}"


def _h2o_saved_text(saved: dict[str, Any]) -> str:
    """Which model formats were saved; the MOJO export can be refused by this H2O build."""
    if saved.get("mojo"):
        return f"Saved as MOJO and as an H2O binary model in `models/h2o{RUN_SUFFIX}/`."
    reason = f" (H2O said: {saved['mojo_error']})" if saved.get("mojo_error") else ""
    return (f"Saved as an H2O binary model in `models/h2o{RUN_SUFFIX}/`, which needs the same H2O version to "
            f"load; the MOJO export failed{reason}.")


# --- Preprocessing ------------------------------------------------------------------------

def _preprocessing_section(inp: WriteupInputs) -> str:
    rows = []
    for tool in _tools(inp):
        n = _notes(inp.leaderboards, tool)
        if not n:
            rows.append([TOOL_NAMES[tool]] + ["n/a"] * 6)
            continue
        used = n.get("n_features_used", n.get("n_features_used_by_varimp_model"))
        new = n.get("new_features_created")
        rows.append([TOOL_NAMES[tool], n.get("n_features_in"), used if used is not None else "n/a",
                     "n/a" if n.get("dropped_features") is None else len(n["dropped_features"]),
                     len(n.get("constant_in_train", [])), n.get("missing_values_in_train", "n/a"),
                     "none" if new == [] else ("n/a" if new is None else f"{len(new)} new columns")])
    return f"""## Preprocessing and feature engineering (read from the saved notes)

{_table(["tool", "features in", "features used", "dropped", "constant in train", "missing values in train",
         "engineered features"], rows)}

"engineered features" compares the column names after each tool's preprocessing with the input names:
"none" means every column the models saw is one of our input features.

{_ag_preprocessing_text(inp.leaderboards, int(_row(inp, _tools(inp)[0])["train_rows"]))}

{_flaml_preprocessing_text(inp.leaderboards)}

{_h2o_preprocessing_text(inp.leaderboards)}"""


def _ag_preprocessing_text(folder: Path, n_train: int) -> str:
    n = _notes(folder, "autogluon")
    if not n:
        return "AutoGluon: notes not available."
    log = n.get("fit_log_highlights", [])
    useless = next((line.strip() for line in log if "Useless Original Features" in line), None)
    unused = next((line.strip() for line in log if "Unused Original Features" in line), None)
    count = lambda line: re.search(r"Count: (\d+)", line).group(1) if line and "Count" in line else "0"  # noqa: E731
    stages = "; ".join(n.get("feature_generators", [])) or "n/a"
    return (f"AutoGluon: its feature generators ({stages}) dropped {count(useless)} \"useless\" features "
            f"(a single value in train) and {count(unused)} \"unused\" ones, keeping "
            f"{n.get('n_features_used')} of {n.get('n_features_in')}, so AutoGluon's models saw "
            f"{int(n.get('n_features_in', 0)) - int(n.get('n_features_used', 0)) - len(n.get('constant_in_train', []))} "
            f"non-constant features fewer than FLAML's and H2O's.{_ag_unused_text(n, n_train)} Model-specific preprocessing "
            f"(e.g. the binning of feature values that LightGBM and CatBoost do internally) happens inside each "
            f"model and is not visible at this level.")


def _ag_unused_text(n: dict[str, Any], n_train: int) -> str:
    """What the non-constant columns AutoGluon dropped are: exact copies, or rare flags."""
    dups = n.get("dropped_nonconstant_duplicate_of", {})
    if not dups:
        return ""
    copies = {c: d for c, d in dups.items() if d is not None}
    copy_txt = ", ".join(f"`{c}` = `{d}`" for c, d in copies.items())
    text = f" Our own check over all rows finds {len(copies)} of the {len(dups)} to be an exact copy of another column ({copy_txt})."
    rare = _checks(n).get("dropped_nonconstant_minority_rows") or {}
    others = {c: v for c, v in rare.items() if c not in copies}
    if not others:
        return text
    counts = [v["rows_not_mode"] for v in others.values()]
    binary = all(v["n_distinct"] == 2 for v in others.values())
    top = max(others, key=lambda c: others[c]["rows_not_mode"])
    sizes = _checks(n).get("duplicate_check_sample_sizes") or {}
    sample = f" on a sample of {sizes['sample_size_final']:,} rows" if sizes.get("sample_size_final") else ""
    share = others[top]["malicious_share_of_those"]
    return (text + f" The other {len(others)} are not copies but near-constant {'binary ' if binary else ''}flags: "
            f"only {min(counts)} to {max(counts)} of the {n_train:,} rows differ from the most common value. "
            f"AutoGluon's duplicate check (DropDuplicatesFeatureGenerator) compares columns only{sample}, where such "
            f"rare flags look like copies of each other, so it dropped them; over all rows they are not duplicates. "
            f"Some carry signal (e.g. `{top}`: {others[top]['rows_not_mode']} rows, "
            f"{100 * share:.0f}% of them malicious), but they are set in too few rows to matter much.")


def _flaml_preprocessing_text(folder: Path) -> str:
    n = _notes(folder, "flaml")
    if not n:
        return "FLAML: notes not available."
    same = set(n.get("dropped_features", [])) == set(n.get("constant_in_train", []))
    which = "exactly the constant columns" if same else "not exactly the constant columns"
    scale = _checks(n).get("feature_scale")
    scale_txt = (f" The raw features are far from equal in scale: the median column's largest |value| is "
                 f"{scale['median_column_max_abs']:g}, but {scale['n_columns_max_abs_over_1e6']} columns exceed 10^6 "
                 f"and the largest reaches {scale['largest_column_max_abs']:.2g}." if scale else "")
    board = _csv(folder, "flaml_leaderboard")
    linear = board[board["estimator"].isin(["lrl1", "sgd"])] if board is not None else pd.DataFrame()
    linear_txt = ", ".join(f"`{r.estimator}` reached validation AUC {_f4(r.best_val_auc)} in {int(r.n_trials)} "
                           f"trial(s) ({r.total_trial_time_s:.0f} s)" for r in linear.itertuples())
    return (f"FLAML: its DataTransformer dropped {len(n.get('dropped_features', []))} columns ({which}) and fits "
            f"{n.get('imputation')}; with {n.get('missing_values_in_train')} missing values in train the imputer "
            f"has nothing to fill. FLAML does not scale features for any learner (checked in the FLAML "
            f"{n.get('version')} source: the scikit-learn wrappers only turn category columns into codes, and "
            f"`lrl1` only sets `solver='saga'` and `tol`), so the linear learners got the raw values.{scale_txt} "
            f"That plausibly explains why they failed: {linear_txt}. A linear model on these features needs "
            f"scaling, as our own `logreg` baseline has (standardized features)."
            if linear_txt else
            f"FLAML: its DataTransformer dropped {len(n.get('dropped_features', []))} columns ({which}) and fits "
            f"{n.get('imputation')}; FLAML does not scale features for any learner.{scale_txt}")


def _h2o_preprocessing_text(folder: Path) -> str:
    n = _notes(folder, "h2o")
    if not n:
        return "H2O: notes not available."
    dropped = n.get("dropped_features") or []
    extra = sorted(set(dropped) - set(n.get("constant_in_train", [])))
    extra_txt = (f" Besides the constant columns it also dropped {len(extra)} that are "
                 f"constant (or \"bad\") inside H2O's own training frame, which is only part of train "
                 f"({', '.join(f'`{c}`' for c in extra[:5])})." if extra else "")
    return (f"H2O: `ignore_const_cols={n.get('ignore_const_cols')}` (read from the model `{n.get('varimp_model')}`), "
            f"so constant columns are skipped inside each model.{extra_txt} Also, `categorical_encoding={n.get('categorical_encoding')}` "
            f"has no effect because every column is numeric. H2O does no feature engineering in AutoML.")


# --- Results ------------------------------------------------------------------------------

def _results_section(inp: WriteupInputs, figures_rel: str) -> str:
    rows = []
    for r in inp.table.itertuples():
        row = inp.table.loc[r.Index]
        rows.append([f"`{r.method}`", r.best_model, _value_ci(row, "test_random_roc_auc", False),
                     _value_ci(row, "test_future_roc_auc", False), _value_ci(row, "test_future_tpr_at_1pct_fpr", True),
                     _f4(r.test_random_f1), _f4(r.test_future_f1), _f4(r.test_future_accuracy),
                     _pct(r.test_future_fpr), _minutes(r.wall_seconds)])
    return f"""## Results on the two test sets

95% CIs are percentile bootstrap intervals ({N_BOOTSTRAP} resamples; seed {_seed_text()}), with every model
scored on the same resamples. F1, accuracy and FPR use the threshold 0.5 (classes are 50/50, so accuracy and
F1 move together). Full table, including test_random accuracy, TPR and FPR:
`tables/automl_vs_manual{RUN_SUFFIX}.csv`.

{_table(["model", "winner", "test_random AUC", "test_future AUC", "test_future TPR@1%FPR", "test_random F1",
         "test_future F1", "test_future accuracy", "test_future FPR", "wall min"], rows)}

![AutoML vs manual]({figures_rel}/automl_vs_manual{RUN_SUFFIX}.png)

The bottom row of the figure shows each model minus `{TUNED_LABEL}` with its paired CI: at the scale of the
top row the differences between the tree models are too small to see."""


# --- Better, worse or different -----------------------------------------------------------

def _comparison_section(inp: WriteupInputs) -> str:
    lines = []
    for tool in _tools(inp):
        parts = [f"{test} {METRIC_NAMES[metric]}: {_verdict(pair, metric)}"
                 for test in ("test_future", "test_random") for metric in ("roc_auc", "tpr_at_1pct_fpr")
                 if (pair := _pair(inp, tool, TUNED_LABEL, test, metric)) is not None]
        lines.append(f"* **{TOOL_NAMES[tool]}** vs `{TUNED_LABEL}`: " + "; ".join(parts) + ".")
    default_lines = [f"{TOOL_NAMES[t]} {_verdict(pair, 'roc_auc')}" for t in _tools(inp)
                     if (pair := _pair(inp, t, "lgbm_default", "test_future", "roc_auc")) is not None]
    default_txt = ("Against untuned LightGBM on test_future AUC: " + "; ".join(default_lines) + "."
                   if default_lines else "")
    return f"""## Better, worse or just different?

Paired bootstrap differences (AutoML tool minus `{TUNED_LABEL}`), 95% CI, no correction for the number of
comparisons (`tables/automl_pairwise{RUN_SUFFIX}.csv`). "Better" or "worse" only when the CI excludes zero.
In that CSV's `{GAP_SET}` rows, `share_resamples_a_better` is the share of resamples where tool a's gap
is the smaller one (a smaller random-split overstatement counts as better).

{chr(10).join(lines)}

{default_txt}

{_tool_ranking_text(inp)}

{_gap_text(inp)}

{_verdict_section(inp)}"""


def _verdict(pair: pd.Series, metric: str) -> str:
    """'better' / 'worse' when the paired CI excludes 0, else 'no clear difference'."""
    status = _status(pair)
    if status == "tied":
        return f"no clear difference ({_span(pair, metric)}, CI includes 0)"
    return f"{status} ({_span(pair, metric)})"


def _tool_ranking_text(inp: WriteupInputs) -> str:
    """The tools against each other on test_future AUC (paired)."""
    tools = _tools(inp)
    parts = [f"{TOOL_NAMES[a]} minus {TOOL_NAMES[b]} {_span(pair, 'roc_auc')} ({_status(pair)})"
             for i, a in enumerate(tools) for b in tools[i + 1:]
             if (pair := _pair(inp, a, b, "test_future", "roc_auc")) is not None]
    return "The tools against each other, test_future AUC: " + "; ".join(parts) + "." if parts else ""


def _gap_text(inp: WriteupInputs) -> str:
    """Random-vs-future gap of every model, and whether any AutoML gap differs from tuned LightGBM's."""
    rows = [[f"`{r.method}`", _diff(r.auc_gap_random_minus_future, "roc_auc"),
             f"[{_diff(r.auc_gap_ci_low, 'roc_auc')}, {_diff(r.auc_gap_ci_high, 'roc_auc')}]",
             _diff(r.tpr_gap_random_minus_future, "tpr_at_1pct_fpr")] for r in inp.table.itertuples()]
    return ("Random-split overstatement (score on test_random minus test_future; positive = the random split "
            "looks better than the future weeks):\n\n"
            + _table(["model", "AUC gap", "95% CI", "TPR@1%FPR gap"], rows)
            + "\n\n" + _gap_answer(inp))


def _gap_answer(inp: WriteupInputs) -> str:
    """Do the AutoML winners overstate more than manual tuning? Answered by the paired gap difference."""
    pairs = {t: _pair(inp, t, TUNED_LABEL, GAP_SET, "roc_auc") for t in _tools(inp)}
    pairs = {t: p for t, p in pairs.items() if p is not None}
    if not pairs:
        return ""
    listing = "; ".join(f"{TOOL_NAMES[t]} {_span(p, 'roc_auc')}" for t, p in pairs.items())
    differ = [t for t, p in pairs.items() if p["significant_95"]]
    logreg = inp.table.set_index("method")["auc_gap_random_minus_future"].get("logreg")
    logreg_txt = (f" Only the linear baseline overstates much more (`logreg` AUC gap {_diff(logreg, 'roc_auc')})."
                  if logreg is not None and not pd.isna(logreg) else "")
    gaps = inp.table[inp.table["method"] != "logreg"]["auc_gap_random_minus_future"].astype(float)
    if not differ:
        return (f"Does AutoML overstate more than manual tuning? No: the paired CI of each tool's AUC gap minus "
                f"`{TUNED_LABEL}`'s gap includes 0 ({listing}). AutoML neither reduced nor added to the random-split "
                f"overstatement; every tree model here loses between {gaps.min():.4f} and {gaps.max():.4f} AUC on "
                f"the future weeks.{logreg_txt}")
    same = [TOOL_NAMES[t] for t in pairs if t not in differ]
    same_txt = (f" The gaps of {_join(same)} cannot be told apart from `{TUNED_LABEL}`'s." if same else "")
    return (f"Does AutoML overstate more than manual tuning? Paired difference of each tool's AUC gap minus "
            f"`{TUNED_LABEL}`'s gap: {listing}. "
            + " ".join(_gap_differs_text(inp, t, pairs[t]) for t in differ)
            + f" The marginal gap CIs in the table overlap, but the paired difference is the sharper test (and is "
            f"not corrected for the number of comparisons).{same_txt} Every tree model "
            f"loses between {gaps.min():.4f} and {gaps.max():.4f} AUC on the future weeks.{logreg_txt}")


def _join(items: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _more_or_less(pair: pd.Series) -> str:
    """Direction of a gap difference: a positive gap minus tuned's gap means the tool overstates more."""
    return "more" if float(pair["diff_a_minus_b"]) > 0 else "less"


def _gap_differs_text(inp: WriteupInputs, tool: str, pair: pd.Series) -> str:
    """One tool whose AUC-gap difference is significant, with its TPR@1%FPR gap difference next to it."""
    direction = _more_or_less(pair)
    text = (f"The CI excludes 0 for {TOOL_NAMES[tool]}: the random split overstates that winner's future-week AUC "
            f"by {abs(float(pair['diff_a_minus_b'])):.4f} {direction} than it does for `{TUNED_LABEL}`.")
    tpr = _pair(inp, tool, TUNED_LABEL, GAP_SET, "tpr_at_1pct_fpr")
    if tpr is None:
        return text
    verdict = ("also measurably " + _more_or_less(tpr) if tpr["significant_95"]
               else "not measurably different (CI includes 0)")
    return f"{text} Its TPR@1%FPR gap difference, {_span(tpr, 'tpr_at_1pct_fpr')}, is {verdict}."


def _verdict_section(inp: WriteupInputs) -> str:
    """Plain-language conclusion per tool: quality, cost and what the winner is, all from the tables."""
    tuned = _row(inp, TUNED_LABEL)
    lines = [f"* **{TOOL_NAMES[t]}**: {_quality_text(inp, t)} It needed {_minutes(_row(inp, t)['wall_seconds'])} "
             f"wall minutes, against {_minutes(tuned['wall_seconds'])} for the selected manual search "
             f"({_minutes(tuned['all_manual_seconds'])} for all three). Winner: {_winner_text(inp, t)}."
             for t in _tools(inp)]
    return ("### Verdict\n\n" + "\n".join(lines) + "\n\n" + _verdict_summary(inp)) if lines else ""


def _verdict_summary(inp: WriteupInputs) -> str:
    """Better / worse / merely different in one paragraph, each clause computed from the tables and notes."""
    tools, tuned = _tools(inp), _row(inp, TUNED_LABEL)
    share = max(float(_row(inp, t)["wall_seconds"]) for t in tools) / float(tuned["wall_seconds"])
    better = (f"Better: cost. Every tool reached its result in at most {100 * share:.0f}% of the selected manual "
              "search's wall time, with no search space to design.")
    status = {t: _status(p) for t in tools if (p := _pair(inp, t, TUNED_LABEL, "test_future", "roc_auc")) is not None}
    worse_tools = [t for t, s in status.items() if s == "worse"]
    worse = ("Worse: " + "; ".join(f"{TOOL_NAMES[t]} did not reach `{TUNED_LABEL}` on test_future AUC"
                                    + (f" ({h})" if (h := _handicap_text(inp, t)) else "") for t in worse_tools) + "."
             if worse_tools else f"Worse: no tool is below `{TUNED_LABEL}` on test_future AUC.")
    tied = [TOOL_NAMES[t] for t, s in status.items() if s == "tied"]
    tied_txt = f" {', '.join(tied)} is statistically tied with `{TUNED_LABEL}`." if tied else ""
    single = [t for t in tools if _single_model(inp, t)]
    single_txt = (" after checking the ensemble weights, every winner is in effect one tree model, so in principle "
                  f"none is harder to explain than `{TUNED_LABEL}`.{_h2o_explain_caveat(inp)}"
                  if len(single) == len(tools) else "")
    gaps = {t: g for t in tools if (g := _pair(inp, t, TUNED_LABEL, GAP_SET, "roc_auc")) is not None}
    differ = [f"{TOOL_NAMES[t]}, whose AUC gap is slightly but measurably "
              f"{'larger' if _more_or_less(g) == 'more' else 'smaller'}" for t, g in gaps.items() if g["significant_95"]]
    gap_txt = (" No tool changed how much the random split overstates performance." if gaps and not differ else
               f" The random-split overstatement is about the same as `{TUNED_LABEL}`'s for every tool except "
               f"{_join(differ)} (see above)." if differ else "")
    return f"{better} {worse}{tied_txt} Merely different:{single_txt}{gap_txt}"


def _h2o_explain_caveat(inp: WriteupInputs) -> str:
    """The practical catch for H2O: no MOJO, so explaining needs a running cluster and the base model."""
    if "h2o" not in _tools(inp):
        return ""
    n = _notes(inp.leaderboards, "h2o")
    saved = n.get("saved_model") or {}
    if not saved or saved.get("mojo"):
        return ""
    base = n.get("varimp_model")
    base_txt = f"its base model `{short_model_id(base)}`" if base else "its base model"
    return (f" In practice the H2O winner must be explained through {base_txt} inside a running H2O cluster, "
            "since no MOJO could be exported (only an H2O binary model was saved).")


def _handicap_text(inp: WriteupInputs, tool: str) -> str:
    """A possible cause of the tool's lower score, read from its artefacts, or ''. Untested, so hedged."""
    n = _notes(inp.leaderboards, tool)
    if tool == "flaml":
        match = re.search(r"Estimated necessary time budget=(\d+)s", _text(inp.leaderboards, "flaml_fit_log"))
        budget = float(_row(inp, tool)["budget_seconds"])
        if match and int(match.group(1)) > budget:
            return f"possibly because its budget was below its own \"necessary\" estimate of {match.group(1)} s"
    if tool == "h2o" and n.get("frames", {}).get("training"):
        share = n["frames"]["training"] / float(_row(inp, tool)["train_rows"])
        return f"plausibly in part because its models trained on {100 * share:.0f}% of train"
    return ""


def _single_model(inp: WriteupInputs, tool: str) -> bool:
    """True when the tool's winner is in effect one tree model (by its notes and checks)."""
    from src.automl_runs import TREE_LEARNERS

    n = _notes(inp.leaderboards, tool)
    if tool == "flaml":
        return n.get("best_estimator") in TREE_LEARNERS
    if tool == "autogluon":
        weights = n.get("ensemble_weights") or {}
        return len([w for w in weights.values() if w > 0]) == 1 or not weights
    return len(_checks(n).get("effective_base_models") or []) == 1


def _quality_text(inp: WriteupInputs, tool: str) -> str:
    """'tied with lgbm_tuned on test_future (AUC ..., TPR ...) and better than lgbm_default (...)'."""
    parts = []
    for ref in (TUNED_LABEL, "lgbm_default"):
        auc, tpr = (_pair(inp, tool, ref, "test_future", m) for m in ("roc_auc", "tpr_at_1pct_fpr"))
        if auc is None or tpr is None:
            continue
        word = {"tied": "statistically tied with", "better": "better than", "worse": "worse than"}
        status = _status(auc) if _status(auc) == _status(tpr) else None
        head = word[status] if status else f"{word[_status(auc)]} (AUC) / {word[_status(tpr)]} (TPR) "
        parts.append(f"{head} `{ref}` on test_future (AUC {_span(auc, 'roc_auc')}, "
                     f"TPR@1%FPR {_span(tpr, 'tpr_at_1pct_fpr')})")
    return ("; ".join(parts) + ".") if parts else "comparison not available."


def _winner_text(inp: WriteupInputs, tool: str) -> str:
    """What the winning model really is, from the notes and checks (not from its name)."""
    n = _notes(inp.leaderboards, tool)
    if not n:
        return str(_row(inp, tool)["best_model"])
    if tool == "flaml":
        return (f"one `{n.get('best_estimator')}` (XGBoost) model found by FLAML's search of "
                f"{n.get('n_trials_total')} trials, retrained on all rows")
    if tool == "autogluon":
        members = [m for m, w in (n.get("ensemble_weights") or {}).items() if w > 0]
        preset = _ag_preset(_text(inp.leaderboards, "autogluon_fit_log"))
        if len(members) == 1:
            pair = _pair(inp, tool, TUNED_LABEL, "test_future", "roc_auc")
            matched = (": AutoGluon matched the tuned model with one LightGBM at preset settings, trained on nearly "
                       "all rows" if pair is not None and _status(pair) == "tied" else "")
            return (f"a single `{members[0]}` (all ensemble weight) with AutoGluon's preset hyperparameters"
                    f"{f' (`{preset}`)' if preset else ''}, i.e. no search{matched}")
        return f"`{n.get('best_model')}`"
    effective = _checks(n).get("effective_base_models") or []
    frames = n.get("frames", {})
    share = f" ({100 * frames['training'] / int(_row(inp, tool)['train_rows']):.0f}% of train)" if frames.get("training") else ""
    rows_txt = f", trained on the {frames['training']:,}-row training frame{share} without a refit on all rows" if share else ""
    last = min(_tools(inp), key=lambda t: float(_row(inp, t)["test_future_roc_auc"])) == tool
    last_txt = "; training on fewer rows is a plausible part of why H2O ranks last among the tools" if last and share else ""
    if len(effective) == 1:
        return (f"a stacked ensemble that is in effect one `{short_model_id(effective[0])}` (metalearner weight 0 on "
                f"the other base model){rows_txt}{last_txt}")
    return f"`{n.get('leader')}`{rows_txt}{last_txt}"


# --- Time ---------------------------------------------------------------------------------

def _time_section(inp: WriteupInputs) -> str:
    rows = []
    for tool in _tools(inp):
        r = _row(inp, tool)
        rows.append([TOOL_NAMES[tool], int(r["budget_seconds"]), f"{r['init_seconds']:.0f}",
                     f"{r['convert_seconds']:.0f}", f"{r['fit_seconds']:.0f}", f"{r['wall_seconds']:.0f}",
                     f"{r['fit_seconds'] - r['budget_seconds']:+.0f}", f"{r['over_budget_seconds']:+.0f}"])
    tuned = _row(inp, TUNED_LABEL)
    all_manual = float(tuned["all_manual_seconds"])
    automl_total = sum(float(_row(inp, t)["wall_seconds"]) for t in _tools(inp))
    budget_min = float(_row(inp, _tools(inp)[0])["budget_seconds"]) / 60
    return f"""## Time: a {budget_min:.0f}-minute budget per tool versus hours of manual search

{_table(["tool", "budget s", "cluster start s", "data conversion s", "fit s", "wall s", "fit over budget s",
         "wall over budget s"], rows)}

Wall = cluster start + data conversion + fit, i.e. start to "model ready"; saving artefacts and scoring the
test sets are logged separately (`artifacts_s`, `score_s` in the log). "Fit over budget" is how far the tool's
own fit call ran past its search budget (for FLAML, see its section: a trial started before the deadline and the
final retrain). "Wall over budget" also counts set-up that is not search: {_overhead_text(inp)}

{_cost_vs_quality_text(inp)}

Manual tuning: the selected tuner `{inp.tuner}` took {float(tuned['search_seconds']) / 3600:.2f} h of search plus
{float(tuned['fit_seconds']):.0f} s for the final refit; all three manual searches with their refits took
{all_manual / 3600:.2f} h. The three AutoML tools together took {automl_total / 3600:.2f} h.

This is **not an apples-to-apples comparison**. The manual searches were not time-capped (equal budget meant
equal configurations: {TUNE_TRIALS} trials x {CV_FOLDS} folds), they searched one model family on a
{TUNE_SUBSET_N:,}-row subset with {CV_FOLDS}-fold CV, and three of them were run to compare search methods, not
to get one model fast. The AutoML tools had a hard search budget, each received all
{int(_row(inp, _tools(inp)[0])['train_rows']):,} rows but used them its own way (see Setup), and spread the
budget over several model families. A fair reading is:
"this is the quality each approach reached at these very different costs" (see the paired differences above),
not "AutoML is N times faster at the same job"."""


def _overhead_text(inp: WriteupInputs) -> str:
    """Start-up and data conversion per tool, next to how far its fit itself ran over."""
    parts = [f"{TOOL_NAMES[t]} {float(r['init_seconds']) + float(r['convert_seconds']):.0f} s of start-up and data "
             f"conversion against {float(r['fit_seconds']) - float(r['budget_seconds']):+.0f} s of fit over budget"
             for t in _tools(inp) if (r := _row(inp, t)) is not None
             and float(r["init_seconds"]) + float(r["convert_seconds"]) >= 1.0]
    return ("; ".join(parts) + ".") if parts else "none of the tools needed any."


def _cost_vs_quality_text(inp: WriteupInputs) -> str:
    """Tools that used more wall time than the best tool and still scored lower (test_future AUC, paired)."""
    tools = _tools(inp)
    best = max(tools, key=lambda t: float(_row(inp, t)["test_future_roc_auc"]))
    parts = []
    for tool in tools:
        if tool == best or float(_row(inp, tool)["wall_seconds"]) <= float(_row(inp, best)["wall_seconds"]):
            continue
        pair = _pair(inp, best, tool, "test_future", "roc_auc")  # tool pairs are stored in one order only
        if pair is None:
            pair = _pair(inp, tool, best, "test_future", "roc_auc")
        extra = 100 * (float(_row(inp, tool)["wall_seconds"]) / float(_row(inp, best)["wall_seconds"]) - 1)
        diff = "" if pair is None else f" (paired test_future AUC difference {_span(pair, 'roc_auc')}, {pair['method_a']} minus {pair['method_b']})"
        parts.append(f"{TOOL_NAMES[tool]} used {extra:.0f}% more wall time than {TOOL_NAMES[best]} "
                     f"({float(_row(inp, tool)['wall_seconds']):.0f} s vs {float(_row(inp, best)['wall_seconds']):.0f} s) "
                     f"and still scored lower{diff}")
    return ("More time did not buy more quality here: " + "; ".join(parts) + ".") if parts else ""


# --- Interpretability ---------------------------------------------------------------------

def _family_ranking(df: pd.DataFrame | None, value: str) -> str:
    """Top families by summed importance, e.g. "strings (31%), header (18%)"."""
    if df is None or df.empty or value not in df:
        return "not available"
    totals = df.groupby("family")[value].sum().clip(lower=0)
    if totals.sum() <= 0:
        return "not available"
    share = (totals / totals.sum()).sort_values(ascending=False).head(TOP_FAMILIES)
    return ", ".join(f"{fam} ({100 * s:.0f}%)" for fam, s in share.items())


def _ag_family_text(df: pd.DataFrame | None) -> str:
    """Top AutoGluon families by AUC drop, and which neighbouring ranks are ties (within one stddev)."""
    if df is None or "importance" not in df:
        return "not available"
    ranked = df.sort_values("importance", ascending=False).reset_index(drop=True)
    has_sd = "stddev" in ranked
    top = ranked.head(TOP_FAMILIES)
    text = ", ".join(f"{r.family} (AUC drop {r.importance:.4f}" + (f" ± {r.stddev:.4f})" if has_sd else ")")
                     for r in top.itertuples())
    if not has_sd:
        return text
    # Neighbours whose difference is below the larger of their two stddevs are tied: their order is noise.
    # Chains of such neighbours are reported as one group, as far as rank TOP_FAMILIES + 1.
    groups, current = [], [0]
    for i in range(min(TOP_FAMILIES, len(ranked) - 1)):
        if ranked.importance[i] - ranked.importance[i + 1] <= max(ranked.stddev[i], ranked.stddev[i + 1]):
            current.append(i + 1)
        else:
            groups.append(current)
            current = [i + 1]
    groups.append(current)
    ties = [f"{_join(list(ranked.family[g]))} (ranks {g[0] + 1}-{g[-1] + 1})" for g in groups if len(g) > 1]
    shuffles = f" over {int(ranked['n'].iloc[0])} shuffles" if "n" in ranked else ""
    tie_txt = (f"; tied within one stddev{shuffles}, so their order is not meaningful: {'; '.join(ties)}"
               if ties else "")
    return text + tie_txt


def _flaml_importance_type(notes: dict[str, Any]) -> str:
    """Notes written before the fix stored None for XGBoost; resolve it the same way the runner now does."""
    if not notes:
        return "n/a"
    return importance_type(str(notes.get("best_estimator", "")), notes.get("importance_type")) or "n/a"


def _interpretability_section(inp: WriteupInputs) -> str:
    rows = [[f"`{r.method}`", r.interpretability] for r in inp.table.itertuples()]
    folder = inp.leaderboards
    ag_family_txt = _ag_family_text(_csv(folder, "autogluon_family_importance"))
    flaml_notes, h2o_notes = _notes(folder, "flaml"), _notes(folder, "h2o")
    flaml_type = _flaml_importance_type(flaml_notes)
    flaml_kind = (f"{'XGBoost' if str(flaml_notes.get('best_estimator', '')).startswith('xgb') else 'native'} "
                  f"{flaml_type} of FLAML's `{flaml_notes.get('best_estimator', 'n/a')}` winner")
    return f"""## Interpretability

{_table(["model", "what you can explain"], rows)}

Top {TOP_FAMILIES} EMBER feature families by each tool's own importance (share of summed importance):

* FLAML winner, native importance (type `{flaml_type}`; `flaml_feature_importance{RUN_SUFFIX}.csv`): {_family_ranking(_csv(folder, 'flaml_feature_importance'), 'importance')}.
* AutoGluon, permutation importance of whole families on its validation rows (`autogluon_family_importance{RUN_SUFFIX}.csv`): {ag_family_txt}.
* H2O, variable importance of `{h2o_notes.get('varimp_model', 'n/a')}`, the best model that has one (`h2o_varimp{RUN_SUFFIX}.csv`): {_family_ranking(_csv(folder, 'h2o_varimp'), 'scaled_importance')}.

The three importances measure different things ({flaml_kind}; drop in validation AUC for AutoGluon;
H2O's scaled importance), so only the ranking of families is comparable, not the numbers. Step 4 compares
them with SHAP on the tuned LightGBM. A real ensemble winner would be harder to explain than one LightGBM
model: there would be no single tree model to run TreeSHAP on, only model-agnostic methods such as
permutation importance, which is slower and does not give a per-file explanation for free. Here the checks
above show that both ensemble winners put their weight on one model."""


# --- Run notes ----------------------------------------------------------------------------

def _sleep_guard_note(inp: WriteupInputs) -> str:
    """Generated from the log: each reported tool's longest heartbeat gap and any discarded attempts."""
    parts = []
    for tool in _tools(inp):
        r = _row(inp, tool)
        gap = r.get("heartbeat_max_gap_seconds")
        gap_txt = "not recorded" if pd.isna(gap) else f"longest heartbeat gap {float(gap):.0f} s"
        invalid = int(r.get("invalid_attempts", 0) or 0)
        parts.append(f"{TOOL_NAMES[tool]} {gap_txt}"
                     + (f", {invalid} earlier attempt(s) discarded because the machine slept" if invalid else ""))
    return "Sleep guard for the reported runs: " + "; ".join(parts) + "."


def _refresh_note(inp: WriteupInputs) -> str:
    """Which notes fields were recomputed from the saved models after the run (automl_checks.refresh_notes)."""
    parts = []
    for tool in _tools(inp):
        refreshed = _notes(inp.leaderboards, tool).get("refreshed")
        if not refreshed:
            continue
        changes = [f"`{k}` renamed to `{v['renamed_to']}`" if "renamed_to" in v else f"`{k}` {v['old']} -> {v['new']}"
                   for k, v in refreshed.get("changed", {}).items()]
        parts.append(f"{TOOL_NAMES[tool]} ({'; '.join(changes + ['saved-model checks added'])})")
    if not parts:
        return ""
    return ("The code that writes the notes JSON in `reports/leaderboards/` changed after the full run (it now "
            "resolves XGBoost's importance type, names H2O's leaderboard AUC correctly and adds the saved-model "
            "checks). Instead of refitting, `scripts/run_automl.py --refresh-notes` recomputed those fields from "
            "the saved models and `train`: " + "; ".join(parts) + ". The fits, timings and test scores are "
            "unchanged; the H2O record in `experiments.jsonl` still has the old key `leader_cv_auc`, because the "
            "log is never edited.")


def _notes_section(inp: WriteupInputs) -> str:
    notes = [*RUN_NOTES, _sleep_guard_note(inp), _refresh_note(inp)]
    return "## Run notes\n\n" + "\n".join(f"* {note}" for note in notes if note)


def render(inp: WriteupInputs, figures_rel: str = "figures") -> str:
    """The whole write-up as markdown."""
    run_ids = ", ".join(sorted(set(inp.table["run_id"].dropna().astype(str))))
    tools = _tools(inp)
    budget = float(_row(inp, tools[0])["budget_seconds"]) if tools else float("nan")
    return "\n\n".join([
        "# Step 3 write-up: AutoML (FLAML, AutoGluon, H2O) versus manual tuning" + QUICK_BANNER,
        f"_Generated by `src/automl_writeup.py` from the Step 3 tables and the artefacts in `reports/leaderboards/` "
        f"(runs: {run_ids}). Do not edit by hand: rerun `scripts/run_automl.py --tables-only` to regenerate it._",
        _setup_section(inp),
        "## What each tool tried and what won",
        _flaml_section(inp.leaderboards, budget),
        _ag_section(inp.leaderboards),
        _h2o_section(inp.leaderboards, budget),
        _preprocessing_section(inp),
        _results_section(inp, figures_rel),
        _comparison_section(inp),
        _time_section(inp),
        _interpretability_section(inp),
        _notes_section(inp),
    ]) + "\n"


def write(inp: WriteupInputs, reports_dir: Path, figures_dir: Path) -> Path:
    """Write reports/automl_comparison[_quick].md; image links are relative so it renders on GitHub."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"automl_comparison{RUN_SUFFIX}.md"
    figures_rel = Path(os.path.relpath(figures_dir, reports_dir)).as_posix()
    path.write_text(render(inp, figures_rel), encoding="utf-8")
    return path
