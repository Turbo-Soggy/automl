"""Render reports/tuning_writeup.md, the Step 2 write-up, from the generated tables.

Every number in the write-up is read from the DataFrames that tables.py builds from the log and
the saved predictions. Sentences that make a claim ("X wins", "the gap is significant") are chosen
by code from those numbers, so the text cannot drift away from the results when they change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import CV_FOLDS, N_BOOTSTRAP, RUN_SUFFIX, SEED
from src import tune
from src.tables import BOOTSTRAP_SEEDS

TUNERS: list[str] = ["grid", "random", "optuna"]
TREE_MODELS: list[str] = ["lgbm_default", "grid", "random", "optuna"]
METRIC_NAMES: dict[str, str] = {"roc_auc": "ROC-AUC", "tpr_at_1pct_fpr": "TPR@1%FPR"}
TEST_NAMES: dict[str, str] = {"test_random": "test_random", "test_future": "test_future"}


@dataclass(frozen=True)
class WriteupInputs:
    """The tables the write-up is built from (all produced by src/tables.py)."""

    summary: pd.DataFrame
    pairwise: pd.DataFrame
    trials: pd.DataFrame
    versions: pd.DataFrame
    shared_trials: int


# --- Small formatting helpers -------------------------------------------------------------

def _auc(x: float) -> str:
    return f"{x:.4f}"


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _ci(low: float, high: float, as_pct: bool = False) -> str:
    return f"[{_pct(low)}, {_pct(high)}]" if as_pct else f"[{_auc(low)}, {_auc(high)}]"


def _diff(x: float, metric: str) -> str:
    """AUC differences as signed numbers, TPR differences as signed percentage points.

    AUC differences get 5 decimals: they are often below 0.0001, and with 4 decimals a CI
    bound such as 0.00004 would print as +0.0000 and look as if the CI touched zero.
    """
    return f"{100 * x:+.2f} pp" if metric == "tpr_at_1pct_fpr" else f"{x:+.5f}"


def _diff_ci(low: float, high: float, metric: str) -> str:
    return f"[{_diff(low, metric)}, {_diff(high, metric)}]"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _row(summary: pd.DataFrame, method: str) -> pd.Series:
    return summary.set_index("method").loc[method]


# --- Sections -----------------------------------------------------------------------------

def _budget(inp: WriteupInputs) -> tuple[int, int]:
    """(trials per tuner, rows in the tuning subset), read from the logged trials."""
    return int(inp.trials.groupby("method").size().max()), int(inp.trials["tuning_rows"].iloc[0])


def _methods_section(inp: WriteupInputs) -> str:
    n_trials, tuning_rows = _budget(inp)
    grid = ", ".join(f"`{k}` in {v}" for k, v in tune.GRID.items())
    space = ", ".join(f"`{k}` {lo:g} to {hi:g}{' (log)' if log else ''}" for k, (lo, hi, log) in tune.SEARCH_SPACE.items())
    return f"""## Methods

* **lgbm_default**: LightGBM with library defaults (100 trees, 31 leaves, learning rate 0.1). The "no tuning" reference.
* **logreg**: logistic regression on standardized features. A linear sanity floor.
* **grid**: Grid Search over {grid}; `bagging_fraction` and `lambda_l2` stay at LightGBM's defaults.
* **random**: Random Search, {n_trials} seeded samples from {space}.
* **optuna**: Optuna TPE, {n_trials} trials with a seeded sampler over the same space as Random Search.

All LightGBM models share the same fixed settings (`tune.BASE_PARAMS`: deterministic,
{tune.BASE_PARAMS['num_threads']} threads, seed {SEED}).

### Equal budget

"Equal budget" means every tuner gets exactly {n_trials} configurations, each scored by the
same {CV_FOLDS} persisted CV folds of the same {tuning_rows:,}-row stratified tuning subset of `train`,
with the same cap of {tune.MAX_ROUNDS} boosting rounds and early stopping after
{tune.EARLY_STOPPING_ROUNDS} rounds without improvement on each fold's validation part.
Wall-clock time is measured but not capped, because a time cap would make the result depend
on machine load while a trial count is exactly reproducible. Equal budget is therefore equal
*trials*, not equal *time* (see Timing). Random Search and Optuna also share their first
{inp.shared_trials} trials (TPE's random start-up phase, same seed), so they differ in only
{n_trials - inp.shared_trials} of their {n_trials} configurations.

Neither test set is used for any choice: the tuners only see the CV folds, and each method's
winner is scored once on `test_random` and once on `test_future`."""


def _shared_note(shared: int, n_trials: int) -> str:
    if not shared:
        return "Random Search and Optuna share no leading trials in this log."
    return (f"**Caveat for every efficiency comparison:** Optuna's TPE sampler starts with "
            f"{tune.TPE_STARTUP_TRIALS} random start-up trials drawn with the same seed as Random Search. "
            f"The log confirms the first {shared} configurations of the two methods are identical, so they "
            f"only differ in the remaining {n_trials - shared} of {n_trials} trials. Any difference "
            f"between Random and Optuna comes from those {n_trials - shared} trials alone.")


def _refit_section(inp: WriteupInputs) -> str:
    rows = []
    for _, r in inp.versions.iterrows():
        rows.append([r["method"], int(r["refit_version"]), r.get("refit_strategy", ""), r["n_trees"],
                     f"{int(r['train_rows_fit']):,}", _auc(r["test_future_roc_auc"]),
                     _pct(r["test_future_tpr_at_1pct_fpr"]), f"`{r['model_file']}`"])
    tuned = inp.summary[inp.summary["method"].isin(TUNERS)]
    rule_rows = [[r["method"], f"{r['cv_mean_best_iteration']:.1f}", r["refit_n_estimators"], f"{int(r['train_rows']):,}"]
                 for _, r in tuned.iterrows()]
    train_rows = int(tuned["train_rows"].iloc[0])
    fold_rows = _budget(inp)[1] * (CV_FOLDS - 1) // CV_FOLDS
    v1 = inp.versions[inp.versions["refit_version"] == 1]
    v1_rows = f"{int(v1['train_rows_fit'].iloc[0]):,}" if not v1.empty else "fewer"
    return f"""## Refit rule

Each tuner's best configuration is refit once on **all {train_rows:,} rows of `train`**,
with a **fixed** number of trees: `n_estimators = round(mean of the winning trial's {CV_FOLDS} fold best_iterations)`.
No hold-out and no early stopping are used in the refit.

{_table(["method", "mean CV best_iteration", "n_estimators used", "train rows"], rule_rows)}

**Why no scaling for the larger training set.** The CV training folds have
{fold_rows:,} rows, the refit set {train_rows / fold_rows:.0f} times more. The tuning logs cannot tell us the
right scale factor, because every fold they contain has the same size. Measuring it would need an extra
early-stopped CV of each winner on larger folds, which we did not run. Linear scaling would be a guess, and
it would ask for far more trees than any fold was allowed to explore ({tune.MAX_ROUNDS}). Near its optimum
the validation AUC curve is flat (the fold best_iterations of one configuration often differ by hundreds of
rounds, see `tables/tuning_trials.csv`), so we assume the exact count matters little. We also *assume*, without
having tested it here, that more data moves the optimum towards more trees; if so, this rule gives slightly
too few trees. Either way the rule is the same for every tuner, so the comparison between tuners stays fair.

**What changed.** Refit version 1 used early stopping (patience {tune.EARLY_STOPPING_ROUNDS}) on a 10% hold-out
of `train`, so it fit on {v1_rows} rows (column `train rows fit` below) although its log said {train_rows:,}.
That stop was noisy: compare its `trees` column below with the mean CV best_iteration above. The test scores then measured the noisy refit, not the
tuner. Version 2 supersedes version 1: the version 1 records stay in the log and its models in
`models/refit_v1/` for transparency, and all other tables use version 2 only. The log is append-only, so a
version 1 record's `model_path` still names the file that now holds the version 2 model; column `model file`
below says where each version's model really is.

{_table(["method", "refit version", "strategy", "trees", "train rows fit", "test_future AUC",
         "test_future TPR@1%FPR", "model file"], rows)}"""


def _seeds_text() -> str:
    """The bootstrap seeds, read from tables.BOOTSTRAP_SEEDS so the text cannot drift from the code."""
    return ", ".join(f"seed {seed} for `{test}`" for test, seed in BOOTSTRAP_SEEDS.items())


def _cv_bias_note(inp: WriteupInputs) -> str:
    """Why the tuners' CV AUC must not be compared with the baselines' CV AUC."""
    n_trials = _budget(inp)[0]
    return (f"**The tuners' CV AUC is slightly optimistic.** Each fold's early stopping picked its round on the same "
            f"validation part that is then scored, and the reported value is the best of {n_trials} trials. The "
            f"baselines' CV AUC has neither bias (fixed number of rounds, one configuration). So the CV column is "
            f"comparable among the three tuners, but not between tuners and baselines; the test scores are the "
            f"unbiased comparison.")


def _results_section(inp: WriteupInputs) -> str:
    rows = []
    for _, r in inp.summary.iterrows():
        rows.append([
            r["method"],
            f"{_auc(r['best_cv_auc'])} ± {r['cv_auc_std']:.4f}",
            f"{_auc(r['test_random_roc_auc'])} {_ci(r['test_random_roc_auc_ci_low'], r['test_random_roc_auc_ci_high'])}",
            f"{_auc(r['test_future_roc_auc'])} {_ci(r['test_future_roc_auc_ci_low'], r['test_future_roc_auc_ci_high'])}",
            f"{_pct(r['test_random_tpr_at_1pct_fpr'])} "
            f"{_ci(r['test_random_tpr_at_1pct_fpr_ci_low'], r['test_random_tpr_at_1pct_fpr_ci_high'], True)}",
            f"{_pct(r['test_future_tpr_at_1pct_fpr'])} "
            f"{_ci(r['test_future_tpr_at_1pct_fpr_ci_low'], r['test_future_tpr_at_1pct_fpr_ci_high'], True)}",
            f"{r['test_future_f1']:.4f}",
        ])
    n_test = int(inp.summary["test_future_n"].iloc[0])
    return f"""## Results

CV AUC is the mean ± standard deviation over the {CV_FOLDS} tuning folds. Test values are point estimates on
{n_test:,} rows per test set, with **95% paired bootstrap confidence intervals** ({N_BOOTSTRAP} resamples;
{_seeds_text()}, independent streams so the gap CI below is valid; `src/eval.py: paired_bootstrap`). The CIs only capture test-set sampling noise: which files happened to land
in the test set. They do not capture training randomness (one seed, one fold split).

{_table(["method", "CV AUC", "test_random AUC [95% CI]", "test_future AUC [95% CI]",
         "test_random TPR@1%FPR [95% CI]", "test_future TPR@1%FPR [95% CI]", "test_future F1"], rows)}

{_cv_bias_note(inp)}

Full numbers: `tables/tuning_summary.csv`. Comparison figure: `figures/tuning_compare.png`."""


def _pair(inp: WriteupInputs, test: str, metric: str, a: str, b: str) -> pd.Series | None:
    p = inp.pairwise
    hit = p[(p["test_set"] == test) & (p["metric"] == metric) & (p["method_a"] == a) & (p["method_b"] == b)]
    return None if hit.empty else hit.iloc[0]


def _winner_section(inp: WriteupInputs) -> str:
    s = inp.summary.set_index("method")
    tuned = s.loc[[m for m in TUNERS if m in s.index]]
    cv_best = tuned["best_cv_auc"].idxmax()
    cv_range = tuned["best_cv_auc"].max() - tuned["best_cv_auc"].min()
    fut_best = tuned["test_future_roc_auc"].idxmax()
    tpr_best = tuned["test_future_tpr_at_1pct_fpr"].idxmax()

    rows, n_sig, n_all = [], 0, 0
    for test in ("test_future", "test_random"):
        for metric in ("roc_auc", "tpr_at_1pct_fpr"):
            for a, b in (("optuna", "random"), ("optuna", "grid"), ("random", "grid")):
                r = _pair(inp, test, metric, a, b)
                if r is None:
                    continue
                n_all += 1
                n_sig += bool(r["significant_95"])
                rows.append([test, METRIC_NAMES[metric], f"{a} − {b}", _diff(r["diff_a_minus_b"], metric),
                             _diff_ci(r["ci_low"], r["ci_high"], metric), f"{r['share_resamples_a_better']:.1%}",
                             "yes" if r["significant_95"] else "no"])
    sig_rows = inp.pairwise[inp.pairwise["method_a"].isin(TUNERS) & inp.pairwise["method_b"].isin(TUNERS)
                            & inp.pairwise["significant_95"]]
    if sig_rows.empty:
        verdict = (f"**No tuner wins significantly.** None of the {n_all} tuner-versus-tuner differences has a 95% CI "
                   f"that excludes zero. `{fut_best}` has the highest test_future AUC point estimate and `{tpr_best}` "
                   f"the highest test_future TPR@1%FPR, but the test set cannot tell the three apart.")
    else:
        claims = "; ".join(
            f"`{r['method_a']}` {'beats' if r['diff_a_minus_b'] > 0 else 'trails'} `{r['method_b']}` on "
            f"{r['test_set']} {METRIC_NAMES[r['metric']]} ({_diff(r['diff_a_minus_b'], r['metric'])}, "
            f"CI {_diff_ci(r['ci_low'], r['ci_high'], r['metric'])})"
            for _, r in sig_rows.iterrows())
        verdict = (f"**{n_sig} of {n_all} tuner-versus-tuner differences are significant at 95%:** {claims}. "
                   f"The other differences are within test-set noise. With {n_all} uncorrected comparisons, about "
                   f"{0.05 * n_all:.1f} would come out 'significant' by chance alone even if all tuners were equal")
        verdict += (", so one or two significant differences are weak evidence." if n_sig <= 2 else ".")
        verdict += _effect_size_note(sig_rows, tuned["cv_auc_std"].max())

    default_rows = []
    for m in TUNERS:
        for metric in ("roc_auc", "tpr_at_1pct_fpr"):
            r = _pair(inp, "test_future", metric, m, "lgbm_default")
            if r is not None:
                default_rows.append([f"{m} − lgbm_default", METRIC_NAMES[metric], _diff(r["diff_a_minus_b"], metric),
                                     _diff_ci(r["ci_low"], r["ci_high"], metric), "yes" if r["significant_95"] else "no"])
    n_default_sig = sum(row[-1] == "yes" for row in default_rows)

    cv_std = tuned.loc[cv_best, "cv_auc_std"]
    cv_cmp = "smaller" if cv_range < cv_std else "larger"
    return f"""## Which tuner won, and is it significant?

* **On CV** (what the tuners optimised), `{cv_best}` is best. The three best CV AUCs lie within
  {cv_range:.5f} of each other, which is {cv_cmp} than the fold-to-fold standard deviation of the winner
  ({cv_std:.5f}).
* **On the test sets**, the paired bootstrap compares the methods on the same resampled rows, so a difference
  can be significant even when two CIs overlap.

{_table(["test set", "metric", "difference", "value", "95% CI", "resamples where a is better", "significant"], rows)}

{verdict}

**Tuning versus no tuning.** On test_future, {n_default_sig} of {len(default_rows)} tuned-versus-default differences
are significant:

{_table(["comparison", "metric", "difference", "95% CI", "significant"], default_rows)}

Full table, both test sets: `tables/tuning_pairwise.csv`."""


def _effect_size_note(sig_rows: pd.DataFrame, cv_std: float) -> str:
    """Put significant tuner differences next to their size: significant is not the same as large.

    The bootstrap CIs only cover test-set sampling, not training randomness (one seed), so a
    difference smaller than the CV fold-to-fold spread may not survive a re-run with another seed.
    """
    auc = sig_rows[sig_rows["metric"] == "roc_auc"]["diff_a_minus_b"].abs()
    if auc.empty:
        return ""
    size = "smaller" if auc.max() < cv_std else "not smaller"
    return (f" The significant AUC differences are at most {auc.max():.5f}, {size} than the largest CV fold-to-fold "
            f"standard deviation of the tuners ({cv_std:.5f}). Because the CIs do not include training randomness "
            f"(one seed), a re-run with another seed could reorder the tuners; we read these as "
            f"{'practically negligible' if size == 'smaller' else 'small'} differences.")


def _timing_section(inp: WriteupInputs) -> str:
    s = inp.summary.set_index("method")
    rows = []
    for m in [m for m in TUNERS if m in s.index]:
        best_trial = inp.trials[(inp.trials["method"] == m) & inp.trials["is_best"]]["trial_number"].iloc[0]
        rows.append([m, f"{s.loc[m, 'search_seconds']:.0f}", f"{s.loc[m, 'search_seconds'] / 3600:.2f}",
                     f"{s.loc[m, 'seconds_per_trial']:.0f}", int(best_trial), f"{s.loc[m, 'fit_seconds']:.0f}"])
    base = [m for m in ("lgbm_default", "logreg") if m in s.index]
    base_txt = ", ".join(f"`{m}` {s.loc[m, 'fit_seconds']:.0f} s" for m in base)
    fastest = min(rows, key=lambda r: float(r[1]))[0]
    slowest = max(rows, key=lambda r: float(r[1]))[0]
    n_trials = _budget(inp)[0]
    return f"""## Timing

Procedure: all methods ran on the same machine, one after another, with LightGBM on
{tune.BASE_PARAMS['num_threads']} threads (`config.N_JOBS`), and we started the timed runs only when no other
heavy job was running. Machine load itself was not logged, so this is the procedure we followed, not a
measurement. The search times come from the original search run; a later `--refit-only` run re-measures
only the final fits.

{_table(["method", "search seconds", "search hours", "seconds per trial", "best trial (1-based)", "final fit seconds"], rows)}

Baselines have no search; their final fits took {base_txt}. Equal trials did not mean equal time:
`{slowest}` took longest, `{fastest}` was fastest. Time per trial depends on which configurations a method
visits (many leaves and a low learning rate are slow), not only on the method's own overhead, which is
negligible next to LightGBM training. {_shared_note(inp.shared_trials, n_trials)}"""


def _convergence_section(inp: WriteupInputs, figures_rel: str) -> str:
    t = inp.trials
    lines = []
    for m in [m for m in TUNERS if m in set(t["method"])]:
        part = t[t["method"] == m]
        best = part[part["is_best"]].iloc[0]
        capped = int(part["hit_round_cap"].sum())
        cap_txt = (f"{capped} of {len(part)} trials hit the {tune.MAX_ROUNDS}-round cap in at least one fold"
                   if capped else "no trial hit the round cap")
        winner_cap = " (including the winner)" if best["hit_round_cap"] else ""
        within = part["cv_auc_mean"].max() - part["cv_auc_mean"].min()
        lines.append(f"* `{m}`: best CV AUC {_auc(best['cv_auc_mean'])} at trial {int(best['trial_number'])}; "
                     f"range over all trials {within:.4f}; {cap_txt}{winner_cap}.")
    return f"""## Convergence

![Tuning convergence]({figures_rel}/tuning_convergence{RUN_SUFFIX}.png)

The line is the best CV AUC found so far, the dots are individual trials. The grey band marks the trials that
Random Search and Optuna share. For Grid the curve only shows when the fixed grid order happens to reach good
points; it is not a search effect.

{chr(10).join(lines)}

A trial "hit the round cap" when a fold's best round is later than {tune.MAX_ROUNDS} − {tune.EARLY_STOPPING_ROUNDS},
i.e. the cap and not early stopping ended training, so a higher cap might have scored that configuration higher
(column `hit_round_cap` in `tables/tuning_trials.csv`)."""


def _gap_section(inp: WriteupInputs) -> str:
    s = inp.summary.set_index("method")
    rows = []
    for m in [m for m in s.index]:
        r = s.loc[m]
        rows.append([m, _diff(r["auc_gap_random_minus_future"], "roc_auc"),
                     _diff_ci(r["auc_gap_ci_low"], r["auc_gap_ci_high"], "roc_auc"),
                     _diff(r["tpr_gap_random_minus_future"], "tpr_at_1pct_fpr"),
                     _diff_ci(r["tpr_gap_ci_low"], r["tpr_gap_ci_high"], "tpr_at_1pct_fpr")])
    trees = s.loc[[m for m in TREE_MODELS if m in s.index]]
    auc_lo, auc_hi = trees["auc_gap_random_minus_future"].min(), trees["auc_gap_random_minus_future"].max()
    tpr_lo, tpr_hi = trees["tpr_gap_random_minus_future"].min(), trees["tpr_gap_random_minus_future"].max()
    all_sig = bool((trees["auc_gap_ci_low"] > 0).all())
    sig_txt = ("Every tree model's AUC gap CI excludes zero, so the overstatement is real, just small."
               if all_sig else "Not every tree model's AUC gap CI excludes zero.")
    lr_txt = ""
    if "logreg" in s.index:
        lr = s.loc["logreg"]
        lr_txt = (f" For logistic regression the AUC gap is {lr['auc_gap_random_minus_future']:+.4f} and the "
                  f"TPR@1%FPR gap {100 * lr['tpr_gap_random_minus_future']:+.1f} percentage points.")
    return f"""## Random split versus future weeks

Gap = score on `test_random` minus score on `test_future` (positive = the random split overstates). The two test
sets are resampled independently, so the gap CI is the spread of the difference of two independent estimates.

{_table(["method", "AUC gap", "95% CI", "TPR@1%FPR gap", "95% CI"], rows)}

For the LightGBM models the AUC gap is {auc_lo:+.4f} to {auc_hi:+.4f}, and the TPR@1%FPR gap is
{100 * tpr_lo:+.2f} to {100 * tpr_hi:+.2f} percentage points. {sig_txt}{lr_txt}

{_gap_reading(trees, s)}"""


def _gap_reading(trees: pd.DataFrame, s: pd.DataFrame) -> str:
    """Interpretation paragraph, chosen by the size of the measured gaps."""
    auc_hi = trees["auc_gap_random_minus_future"].max()
    tpr_mean = trees["tpr_gap_random_minus_future"].mean()
    if auc_hi >= 0.01 or not (trees["auc_gap_ci_low"] > 0).all():
        return ("The gaps are either large or not clearly different from zero, so no simple reading applies here. "
                "Step 4 looks at where any loss comes from (per-week drift plot, week-blocked forward CV, "
                "challenge set).")
    linear = ""
    if "logreg" in s.index and s.loc["logreg", "auc_gap_random_minus_future"] > 5 * auc_hi:
        linear = (" Boosted trees on 2,568 static features seem to generalise across those weeks much better "
                  "than a linear model does.")
    return (f"We expected a much larger drift penalty for the tree models, and we report the small one as it is "
            f"rather than hunting for a setup that makes it bigger. Plausible reasons, not tested yet: "
            f"`test_future` covers only the 12 weeks right after training, so most malware families are still "
            f"the ones seen in training.{linear} The drop is easier to see at a low false-alarm rate: on average "
            f"{100 * tpr_mean:.2f} percentage points of TPR@1%FPR, which is the operating region that matters in "
            f"practice. Step 4 checks this further with a per-week drift plot, a week-blocked forward CV, and the "
            f"challenge set of malware that antivirus engines initially missed.")


def render(inp: WriteupInputs, figures_rel: str = "figures") -> str:
    """The whole write-up as markdown."""
    run_ids = ", ".join(sorted(set(inp.summary["run_id"].dropna())))
    return "\n\n".join([
        "# Step 2 write-up: baselines and manual tuning of LightGBM",
        f"_Generated by `src/tuning_writeup.py` from the Step 2 tables (final evaluation run(s): {run_ids}). "
        "Do not edit by hand: rerun `scripts/run_tuning.py --tables-only` to regenerate it._",
        _methods_section(inp),
        _refit_section(inp),
        _results_section(inp),
        _winner_section(inp),
        _timing_section(inp),
        _convergence_section(inp, figures_rel),
        _gap_section(inp),
    ]) + "\n"


def write(inp: WriteupInputs, reports_dir: Path, figures_dir: Path) -> Path:
    """Write the write-up next to the tables; image links are relative so the file renders on GitHub."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"tuning_writeup{RUN_SUFFIX}.md"
    figures_rel = Path(os.path.relpath(figures_dir, reports_dir)).as_posix()
    path.write_text(render(inp, figures_rel), encoding="utf-8")
    return path
