"""Render reports/leakage.md from the leakage table and the logged records (no number typed by hand).

Every number below is read from reports/tables/leakage.csv (built by src/leakage_report.py), the
experiment log records it was built from, or the split metadata. Sentences that judge a result
("helped", "fell for it") are chosen by explicit rules on those numbers (see the helpers), so a
re-run with different results produces different wording, never stale claims.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from config import LEAKAGE_BUDGET_S, QUICK, REPORTS, ROOT, RUN_SUFFIX, TABLES
from src import leakage

MODEL_LABEL: dict[str, str] = {leakage.LGBM: "tuned LightGBM", "flaml": "FLAML", "autogluon": "AutoGluon",
                               "h2o": "H2O"}


# --- Small formatting helpers -------------------------------------------------------------

def _num(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _signed(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def _rank(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"#{int(value)}"


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def _ci(row: pd.Series, key: str, digits: int = 4) -> str:
    """'0.9991 [0.9989, 0.9993]' for a metric with _ci_low / _ci_high columns."""
    return f"{_num(row[key], digits)} [{_num(row[key + '_ci_low'], digits)}, {_num(row[key + '_ci_high'], digits)}]"


def _delta(row: pd.Series, test: str, metric: str = "roc_auc", digits: int = 4) -> str:
    key = f"delta_{test}_{metric}"
    if key not in row or pd.isna(row[key]):
        return "-"
    star = " *" if bool(row.get(f"{key}_significant")) else ""
    return f"{_signed(row[key], digits)} [{_signed(row[key + '_ci_low'], digits)}, {_signed(row[key + '_ci_high'], digits)}]{star}"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _row(table: pd.DataFrame, model: str, variant: str) -> pd.Series | None:
    part = table[(table["model"] == model) & (table["variant"] == variant)]
    return None if part.empty else part.iloc[0]


def _effect(row: pd.Series, test: str, metric: str = "roc_auc") -> str:
    """'raised' / 'lowered' / 'did not measurably change' (rule: the paired 95% CI excludes 0)."""
    key = f"delta_{test}_{metric}"
    if key not in row or pd.isna(row[key]):
        return "was not compared"
    if not bool(row[f"{key}_significant"]):
        return "did not measurably change"
    return "raised" if row[key] > 0 else "lowered"


# --- Sections -----------------------------------------------------------------------------

def _setup(table: pd.DataFrame, context: dict[str, Any]) -> str:
    lgbm = context["records"].get((leakage.LGBM, leakage.CLEAN))
    config = (lgbm or {}).get("lgbm_config") or {}
    lines = [
        "## What was run",
        "",
        "| Variant | Change to the 2,568 standard features |",
        "|---|---|",
        f"| `clean` | none (the inputs of Steps 2 and 3) |",
        f"| `{leakage.LEAK_DR}` | + column `{leakage.LEAK_COLUMN[leakage.LEAK_DR]}`: share of AV engines that flagged "
        "the file (`detection_ratio`, parsed to a float) |",
        f"| `{leakage.LEAK_DATE}` | + column `{leakage.LEAK_COLUMN[leakage.LEAK_DATE]}`: VirusTotal first-submission "
        "date as raw unix seconds |",
        f"| `{leakage.ABLATE}` | - the {len(leakage.TIME_PROXIES)} time-stamp columns already in the vector "
        "(see the ablation section); tuned LightGBM only |",
        "",
        f"* **Tuned LightGBM:** the configuration of the Step 2 tuner with the best CV AUC "
        f"(`{config.get('tuner', 'n/a')}`), refit on all of `train` with its fixed "
        f"{config.get('n_estimators', 'n/a')} trees, no early stopping, for every variant.",
        f"* **AutoML:** FLAML, AutoGluon (`medium_quality`) and H2O through the Step 3 runners, with "
        f"**{LEAKAGE_BUDGET_S} s per run**. This is a stated deviation from the {context['step3_budget_s']} s "
        "of Step 3, made to save time. The `clean` variant is re-run at the same budget, so in every "
        "with/without comparison the leak column is the only intended difference. The budget caps the "
        "search only, so the effective run times differed (see *Run times* below).",
        f"* Same `train` ({_rows(lgbm)} rows), seed 42, {(lgbm or {}).get('n_jobs', 'n/a')} threads, ROC-AUC as the "
        "tools' metric. Importance and SHAP use rows of `train` or the tool's own validation rows, never a test set.",
        f"* **Machine:** {_machine(context)}. Steps 2 and 3 ran on a different machine (a 16-core Intel Core "
        "Ultra 9 285H laptop, 63 GB RAM, Windows, 16 threads). Every run in this step, clean and leaky, ran on "
        "the machine above with the same settings, one run at a time, so the with/without comparisons are "
        "equal-hardware and equal-budget (not equal-compute, see *Run times*); "
        "times here must not be compared with the times of Steps 2-3. With 4 cores instead of 16, an AutoML "
        "tool searches far fewer configurations in the same budget.",
        f"* **Memory (15 GB RAM) forced runner settings that differ from Step 3, the same for every variant:** {_tool_settings(context)}",
        "* `Delta` columns are *variant minus clean* for the same model on the same test rows, with a "
        "paired bootstrap 95% CI (1000 resamples); `*` marks a CI that excludes 0.",
    ]
    if lgbm is not None and config:
        diff_r = lgbm["test_random"]["roc_auc"] - config["step2_test_random_auc"]
        diff_f = lgbm["test_future"]["roc_auc"] - config["step2_test_future_auc"]
        lines.append(f"* Check: the clean LightGBM refit gives test AUC {_num(lgbm['test_random']['roc_auc'], 6)} / "
                     f"{_num(lgbm['test_future']['roc_auc'], 6)} (random / future); the Step 2 record "
                     f"`{config['step2_eval_id']}` has {_num(config['step2_test_random_auc'], 6)} / "
                     f"{_num(config['step2_test_future_auc'], 6)} (difference {diff_r:+.1e} / {diff_f:+.1e}). "
                     + ("The refit reproduces Step 2." if max(abs(diff_r), abs(diff_f)) < 1e-6 else
                        "LightGBM runs with `deterministic=True`, but results are bit-identical only for the same "
                        "thread count; this machine used a different one than Step 2, so tiny differences are "
                        "expected."))
    step3 = TABLES / f"automl_vs_manual{RUN_SUFFIX}.csv"
    if step3.exists():
        ref = pd.read_csv(step3).set_index("method")
        rows = []
        for tool in leakage.TOOLS:
            row = _row(table, tool, leakage.CLEAN)
            if row is None or tool not in ref.index:
                continue
            rows.append([MODEL_LABEL[tool], _num(ref.loc[tool, "test_random_roc_auc"]),
                         _num(row["test_random_roc_auc"]), _num(ref.loc[tool, "test_future_roc_auc"]),
                         _num(row["test_future_roc_auc"]), str(ref.loc[tool, "best_model"])[:40],
                         str(row["best_model"])[:40]])
        if rows:
            lines += ["", f"Effect of the shorter budget **and the smaller machine** on the clean runs (Step 3 values "
                      f"from `{_rel(step3)}`, on the 16-core laptop):", "",
                      _md_table(["Tool", f"AUC random, {context['step3_budget_s']} s", f"AUC random, {LEAKAGE_BUDGET_S} s",
                                 f"AUC future, {context['step3_budget_s']} s", f"AUC future, {LEAKAGE_BUDGET_S} s",
                                 f"winner, {context['step3_budget_s']} s", f"winner, {LEAKAGE_BUDGET_S} s"], rows)]
    lines += ["", _run_times(table, context)]
    return "\n".join(lines)


def _tool_settings(context: dict[str, Any]) -> str:
    """The Step 4 runner settings that differ from Step 3, read back from the logged records."""
    recs = context["records"]
    flaml = {(r.get("info") or {}).get("preprocessing") for (m, _), r in recs.items() if m == "flaml"}
    h2o = {tuple((r.get("info") or {}).get("exclude_algos") or ()) for (m, _), r in recs.items() if m == "h2o"}
    heap = {str((r.get("info") or {}).get("max_mem")) for (m, _), r in recs.items() if m == "h2o"}
    parts = []
    if flaml:
        parts.append(f"FLAML preprocessing: {', '.join(sorted(str(f) for f in flaml))}. FLAML's own DataTransformer "
                     "ran out of memory on the 160k-row train (its median imputer makes float64 copies of X); "
                     "there are no missing values, so the data FLAML sees is the same.")
    if h2o:
        excluded = ", ".join(sorted({a for t in h2o for a in t})) or "none"
        parts.append(f"H2O excluded algorithms: {excluded}. XGBoost is available on Linux (it was not on the "
                     "Windows laptop of Step 3) and uses memory outside the H2O heap, which killed the H2O "
                     "server; leaving it out also keeps H2O's algorithm set the same as in Step 3. "
                     f"H2O heap: {', '.join(sorted(heap))} (Step 3: 24G); with 9G the JVM plus the Python "
                     "worker passed the machine's memory limit and the kernel killed the H2O server.")
    ag = [r for (m, _), r in recs.items() if m == "autogluon"]
    if ag:
        trained = sorted({mdl for r in ag for mdl in (r.get("info") or {}).get("models_trained", [])})
        skipped = {v: sum("Not enough memory to safely train" in line
                          for line in context["scans"].get(leakage.method_id("autogluon", v), {}).get("warning_lines", []))
                   for (m, v) in recs if m == "autogluon"}
        skipped_text = ", ".join(f"`{v}` {n}" for v, n in skipped.items())
        parts.append(f"AutoGluon ran unchanged; across its runs it trained only: {', '.join(trained)}. It decides "
                     "itself which models fit in the time and memory left; lines \"Not enough memory to safely train "
                     f"model\" (a model it skipped) per run: {skipped_text}."
                     + (" On this machine AutoGluon is therefore in effect one LightGBM model."
                        if len([t for t in trained if not t.startswith("WeightedEnsemble")]) == 1
                        and all(t.startswith(("LightGBM", "WeightedEnsemble")) for t in trained) else ""))
    return " ".join(parts) or "n/a"


def _rows(rec: dict[str, Any] | None) -> str:
    return "n/a" if not rec else f"{int(rec['train_rows']):,}"


def _machine(context: dict[str, Any]) -> str:
    """'Intel Xeon ..., 4 logical CPUs, 15.6 GB RAM, Linux ...' from the logged records (all must agree)."""
    machines = {tuple(sorted((r.get("machine") or {}).items())) for r in context["records"].values()}
    if len(machines) != 1:
        return f"the records name {len(machines)} different machines (see the `machine` field of each record)"
    m = dict(next(iter(machines)))
    return f"{m.get('cpu')}, {m.get('cpu_count')} logical CPUs, {m.get('ram_gb')} GB RAM, {m.get('os')}"


def _run_times(table: pd.DataFrame, context: dict[str, Any]) -> str:
    """Wall time of every run on this machine, and what happened to the AutoGluon predictor folders."""
    rows = []
    for _, row in table.iterrows():
        rec = context["records"][(row["model"], row["variant"])]
        over = None if row["budget_s"] is None or pd.isna(row["budget_s"]) else rec.get("wall_s", 0) - row["budget_s"]
        rows.append([MODEL_LABEL[row["model"]], f"`{row['variant']}`", _num(rec.get("fit_s"), 0),
                     _num(rec.get("wall_s"), 0), "-" if over is None else _signed(over, 0),
                     _num(rec.get("score_s"), 0), "yes" if rec.get("timing_valid") else "NO"])
    deleted = [r for r in context["records"].values() if r.get("model_folder_deleted_after_scoring")]
    text = ["Run times on this machine, seconds (`fit` = the tool's fit call; `wall` = everything the budget "
            "covers; `over budget` = wall minus budget; `timing valid` = the sleep guard saw no gap):", "",
            _md_table(["Model", "Variant", "fit", "wall", "over budget", "scoring", "timing valid"], rows),
            "", _compute_spread(table, context)]
    if deleted:
        text += ["", f"Disk: the machine had little free disk, so each AutoGluon predictor folder (about 2 GB) was "
                 f"deleted right after its run was scored, once its leaderboard, importances, preprocessing notes "
                 f"and test scores had been saved ({len(deleted)} folders, e.g. "
                 f"`{deleted[0]['model_folder_deleted_after_scoring']}`). The AutoGluon rows can therefore be "
                 "re-scored only by re-running them."]
    return "\n".join(text)


def _compute_spread(table: pd.DataFrame, context: dict[str, Any]) -> str:
    """Why the variants of one AutoML tool did not get equal compute, with the spread per tool."""
    spreads, clean_most = [], []
    for model in table.loc[table["budget_s"].notna(), "model"].unique():
        walls = {row["variant"]: context["records"][(model, row["variant"])].get("wall_s")
                 for _, row in table[table["model"] == model].iterrows()}
        walls = {v: w for v, w in walls.items() if w is not None}
        if len(walls) < 2:
            continue
        spreads.append(f"{MODEL_LABEL[model]} {max(walls.values()) - min(walls.values()):.0f} s")
        if max(walls, key=walls.get) == leakage.CLEAN:
            clean_most.append(MODEL_LABEL[model])
    if not spreads:
        return ""
    return ("**The pairs are equal-budget, not equal-compute.** The budget is each tool's soft limit on its "
            "*search*: FLAML's final refit on all rows, H2O's data import and its post-search ensemble and scoring "
            "come on top, and a tool may also stop before the budget. So the wall times of one tool's variants "
            f"differ by up to {', '.join(spreads)}; the clean run had the most time for "
            f"{', '.join(clean_most) or 'no tool'}. This cannot explain the `{leakage.LEAK_DR}` jump (where the "
            "clean run had more time, a leak gain is, if anything, understated), but it is one more reason to "
            f"read the deltas of about 1e-4 AUC for `{leakage.LEAK_DATE}` as run-to-run noise.")


def _results(table: pd.DataFrame) -> str:
    rows = []
    for _, row in table.iterrows():
        leak = f"{_rank(row.get('leak_rank_native'))} / {_rank(row.get('leak_rank_shap'))}" \
            if isinstance(row.get("leak_column"), str) else "-"
        rows.append([MODEL_LABEL[row["model"]], f"`{row['variant']}`", _ci(row, "test_random_roc_auc"),
                     _ci(row, "test_future_roc_auc"), _num(row["test_random_tpr_at_1pct_fpr"]),
                     _num(row["test_future_tpr_at_1pct_fpr"]), _delta(row, "test_random"), _delta(row, "test_future"),
                     leak])
    return "\n".join([
        "## Results",
        "",
        _md_table(["Model", "Variant", "AUC test_random [95% CI]", "AUC test_future [95% CI]",
                   "TPR@1%FPR random", "TPR@1%FPR future", "Delta AUC random vs clean", "Delta AUC future vs clean",
                   "Leak rank (tool's own / SHAP)"], rows),
        "",
        "Leak rank: 1 = the most important of all columns, by the tool's own importance (LightGBM gain, "
        "FLAML `feature_importances_`, which for its LightGBM learner is the split count and so favours "
        "continuous high-cardinality columns such as a date, AutoGluon permutation importance, H2O varimp) and by mean |TreeSHAP| "
        "on 2,000 `train` rows where the model is a tree model (n/a = not computed; the reason is in the "
        "`importance_shap_kind` column of the CSV). AutoGluon's own ranks are among the features its "
        "permutation importance reached before its time cap, and its shares can exceed 100% (see Caveats). "
        "The full table, with F1, FPR, TPR CIs, time-proxy ranks and timings, is the CSV.",
    ])


def _leak_dr(table: pd.DataFrame) -> str:
    part = table[table["variant"] == leakage.LEAK_DR]
    if part.empty:
        return "## Leak 1: `detection_ratio`\n\nNot run."
    lines = ["## Leak 1: `detection_ratio` (post-hoc label information)", ""]
    for _, row in part.iterrows():
        clean = _row(table, row["model"], leakage.CLEAN)
        before = f"{_num(clean['test_random_roc_auc'])} -> " if clean is not None else ""
        before_f = f"{_num(clean['test_future_roc_auc'])} -> " if clean is not None else ""
        dropped = row.get("leak_dropped_by_tool")
        dropped_text = {True: "it **dropped** the column", False: "it kept the column"}.get(
            None if pd.isna(dropped) else bool(dropped), "it kept the column" if row.get("leak_in_model") else "n/a")
        lines.append(
            f"* **{MODEL_LABEL[row['model']]}** ({row['best_model']}): AUC on test_random {before}"
            f"{_num(row['test_random_roc_auc'])}, on test_future {before_f}{_num(row['test_future_roc_auc'])}; "
            f"TPR@1%FPR {_num(row['test_random_tpr_at_1pct_fpr'])} / {_num(row['test_future_tpr_at_1pct_fpr'])}. "
            f"The leak column ranks {_rank(row.get('leak_rank_native'))} by its own importance "
            f"({_pct(row.get('leak_share_native'))} of the total) and {_rank(row.get('leak_rank_shap'))} by SHAP "
            f"({_pct(row.get('leak_share_shap'))}); {dropped_text}; "
            f"log lines naming it: {int(row['n_lines_mentioning_leak'])}, warning lines: {int(row['n_warning_lines'])}.")
    fell = int(sum(v is True or v == 1.0 for v in part["relied_on_leak"]))
    lines += [
        "",
        f"**{fell} of {len(part)} models fell for the leak**: they kept the column and ranked it first "
        "(rule: rank 1 by the tool's own importance or by SHAP; for FLAML, whose own importance is a split "
        "count, by SHAP only). No tool can know from the data alone that "
        "this column is post-hoc: a column with AUC 1.0 against the label looks like the best feature there "
        "is, and every tool's job is to find exactly such features.",
        "",
        f"Note what temporal validation does **not** show here: the detection ratio of a future file is just "
        f"as post-hoc as that of a past one, so the leaky models score {_num(part['test_future_roc_auc'].min())} "
        f"to {_num(part['test_future_roc_auc'].max())} AUC on test_future too: the future split cannot expose "
        "this leak. Only asking *when would this value be known?* exposes it: a detector scores a file when it "
        "first appears, before (or instead of) the antivirus scan the ratio comes from.",
    ]
    return "\n".join(lines)


def _leak_date(table: pd.DataFrame, context: dict[str, Any]) -> str:
    part = table[table["variant"] == leakage.LEAK_DATE]
    if part.empty:
        return "## Leak 2: `first_submission_date`\n\nNot run."
    dates = context["dates"]
    lines = ["## Leak 2: raw `first_submission_date` (a time stamp, not a file property)", "",
             f"`train` dates run from {dates['train_min']} to {dates['train_max']}; test_future dates from "
             f"{dates['test_future_min']} to {dates['test_future_max']}. Every future date is outside the training "
             "range. Classes are 50/50 in every week by construction, so the date alone carries almost no label "
             "information; any gain must come from combining it with other columns.", ""]
    for _, row in part.iterrows():
        lines.append(
            f"* **{MODEL_LABEL[row['model']]}**: on test_random the date {_effect(row, 'test_random')} AUC "
            f"({_delta(row, 'test_random')}) and {_effect(row, 'test_random', 'tpr_at_1pct_fpr')} TPR@1%FPR "
            f"({_delta(row, 'test_random', 'tpr_at_1pct_fpr')}); on test_future it "
            f"{_effect(row, 'test_future')} AUC ({_delta(row, 'test_future')}) and "
            f"{_effect(row, 'test_future', 'tpr_at_1pct_fpr')} TPR@1%FPR "
            f"({_delta(row, 'test_future', 'tpr_at_1pct_fpr')}). Leak rank {_rank(row.get('leak_rank_native'))} "
            f"(own) / {_rank(row.get('leak_rank_shap'))} (SHAP).")
    rec = context["records"].get((leakage.LGBM, leakage.LEAK_DATE)) or {}
    thresholds = rec.get("date_leak_thresholds")
    if thresholds and thresholds.get("n_splits"):
        max_t = pd.Timestamp(int(thresholds["max_threshold"]), unit="s")
        beyond = thresholds["max_threshold"] < dates["test_future_min_unix"]
        lines += ["", f"What the date can and cannot do on test_future, shown on the tuned LightGBM: the model "
                  f"splits on the date {thresholds['n_splits']} times, and its largest threshold is {max_t}. "
                  + ("That is before the first test_future date, so all future rows take the same branch at every "
                     "date split: for this model the column is a constant on test_future. The model cannot tell "
                     "one future file from another by date; on the date column it treats every future file like "
                     f"the training files submitted after {max_t}."
                     if beyond else
                     "Some thresholds fall inside the test_future range, so the column is not fully constant there.")]
        lgbm_row = _row(table, leakage.LGBM, leakage.LEAK_DATE)
        if beyond and lgbm_row is not None and _effect(lgbm_row, "test_future") == "raised":
            lines.append(
                "So the measured gain on test_future cannot come from the future dates themselves. It is "
                "consistent with a side effect: with the date, the trees can fit week-specific patterns of other "
                "columns, and every future file is then scored with the patterns of the latest training weeks, "
                "which are the closest in time to the future. That is a crude form of adapting to drift, not "
                "information about the future file; we did not test this explanation further.")
        elif beyond:
            lines.append("Whatever the date did on the random split, it carries no information to test_future.")
    elif thresholds is not None:
        lines += ["", "The tuned LightGBM never split on the date column."]
    weak = [MODEL_LABEL[r["model"]] for _, r in part.iterrows()
            if r["model"] != leakage.LGBM and "did not" not in _effect(r, "test_future")
            and min(_rank_or_inf(r.get("leak_rank_native")), _rank_or_inf(r.get("leak_rank_shap"))) > 10]
    if weak:
        lines += ["", f"For {', '.join(weak)} the date changed the score measurably although it is not among the "
                  "model's 10 most important columns by either measure. A time-budgeted AutoML search is not "
                  "repeatable: with one more column it tries other configurations and can end at a different "
                  "winner, so these deltas are more likely the search taking another path than an effect of the "
                  "date itself."]
    helped = [MODEL_LABEL[r["model"]] for _, r in part.iterrows() if _effect(r, "test_random") == "raised"]
    lines += ["", ("On the random split the date measurably raised AUC for: " + ", ".join(helped) + "."
                   if helped else "No model got a measurable AUC gain from the date, even on the random split "
                   "(every paired CI includes 0 or is negative). With 50/50 classes in every week the raw date "
                   "is a weak leak in this dataset; it would be a strong one wherever the share of malware "
                   "changes over time.")]
    return "\n".join(lines)


def _rank_or_inf(value: Any) -> float:
    return math.inf if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _ablation(table: pd.DataFrame, context: dict[str, Any]) -> str:
    prov = context["provenance"]
    rows = [[str(p["column"]), f"`{p['name']}`", p["thrember_class"], f"`{p['raw_key']}`", p["meaning"],
             f"{p['source_file']}:{p['process_raw_features_line']}", "yes" if p["source_matches_offset"] else "NO"]
            for p in prov]
    lines = ["## Time proxies already in the feature vector: ablation", "",
             "Located in thrember's source (`process_raw_features()` of each class lists the values in vector "
             "order; `src.leakage.time_proxy_provenance()` re-reads the source and checks each position):", "",
             _md_table(["Column", "Name", "thrember class", "Value", "Meaning", "Source (method start)",
                        "Position verified"], rows),
             "",
             "The other authenticode columns (2472-2477: certificate count, self-signed, empty program name, "
             "no countersigner, parse error, chain depth) and the rest of the header block are not times.", ""]
    proxy_rows = []
    for model in leakage.MODEL_NAMES:
        row = _row(table, model, leakage.CLEAN)
        if row is None:
            continue
        proxy_rows.append([MODEL_LABEL[model]] + [f"{_rank(row.get(f'rank_native_{p.name}'))} / "
                                                  f"{_rank(row.get(f'rank_shap_{p.name}'))}"
                                                  for p in leakage.TIME_PROXIES])
    if proxy_rows:
        lines += ["Rank of each proxy in the clean runs (own importance / SHAP, out of all features; AutoGluon's own "
                  "ranks are out of the features its permutation importance reached, see Caveats):", "",
                  _md_table(["Model"] + [f"`{p.name}`" for p in leakage.TIME_PROXIES], proxy_rows), ""]
    clean, ablated = _row(table, leakage.LGBM, leakage.CLEAN), _row(table, leakage.LGBM, leakage.ABLATE)
    if clean is not None and ablated is not None:
        lines += [
            f"Removing them from the tuned LightGBM {_effect(ablated, 'test_random')} AUC on test_random "
            f"({_delta(ablated, 'test_random')}) and {_effect(ablated, 'test_future')} it on test_future "
            f"({_delta(ablated, 'test_future')}); TPR@1%FPR: {_delta(ablated, 'test_random', 'tpr_at_1pct_fpr')} "
            f"random, {_delta(ablated, 'test_future', 'tpr_at_1pct_fpr')} future. The random-minus-future AUC gap "
            f"is {_signed(clean['auc_gap_random_minus_future'], 5)} with the proxies and "
            f"{_signed(ablated['auc_gap_random_minus_future'], 5)} without them (TPR@1%FPR gap "
            f"{_signed(clean['tpr_gap_random_minus_future'], 4)} vs {_signed(ablated['tpr_gap_random_minus_future'], 4)}).",
            "",
            f"So removing the proxies changes the random-minus-future AUC gap by "
            f"{_signed(ablated['auc_gap_random_minus_future'] - clean['auc_gap_random_minus_future'], 5)}: "
            + ("it does not shrink it. " if ablated["auc_gap_random_minus_future"] >= clean["auc_gap_random_minus_future"]
               else "it shrinks it. "),
            "",
            "Reading: if the proxies were a time *leak* that the random split rewards, removing them would cost "
            "more on test_random than on test_future and shrink the gap. If the cost is similar on both sets, "
            "the model uses them as ordinary file properties (for example, whether a file is signed at all, or "
            "has a zeroed or implausible compile date), not as a calendar."]
    return "\n".join(lines)


def _warnings(table: pd.DataFrame, context: dict[str, Any]) -> str:
    rows, examples = [], []
    for _, row in table.iterrows():
        if row["model"] == leakage.LGBM:
            continue  # LightGBM is our own code: there is no tool that could warn
        dropped = row.get("leak_dropped_by_tool")
        rows.append([MODEL_LABEL[row["model"]], f"`{row['variant']}`", str(int(row["n_warning_lines"])),
                     str(int(row["n_lines_mentioning_leak"])), str(int(row["n_flag_word_lines"])),
                     "n/a" if pd.isna(dropped) else ("yes" if dropped else "no")])
        scan = context["scans"].get(leakage.method_id(row["model"], row["variant"]), {})
        examples += [f"  * {MODEL_LABEL[row['model']]} `{row['variant']}`: `{line[:160]}`"
                     for line in scan.get("flag_word_lines", [])[:3] + scan.get("leak_mention_lines", [])[:3]]
    if not rows:
        return ""
    lines = ["## Did any tool warn about, flag or drop the leak column?", "",
             _md_table(["Tool", "Variant", "Warning lines", "Lines naming the leak column", "Flag-word lines",
                        "Leak column dropped"], rows), "",
             "Scanned: each run's own stdout/stderr (`reports/leakage_logs/`) and the tool's log (FLAML fit log, "
             "AutoGluon predictor log, H2O event log); \"dropped\" comes from the tool's preprocessing notes. "
             "Flag words: leak, target, suspicious, correlated, perfect, too good, predictive, dominant, id-like "
             "(the leak column's own name and our output folder name are blanked first). Warning lines: the word "
             "warn/warning or a Python warning class; H2O's event-log WARN lines are its notes on dropping "
             "constant columns. Full lines: "
             f"`{_rel(leakage.LEAKAGE_LEADERBOARDS / f'log_scan{RUN_SUFFIX}.json')}`."]
    leaky = table[table["leak_column"].notna() & (table["model"] != leakage.LGBM)]
    named = int(leaky["n_lines_mentioning_leak"].sum())
    dropped_any = any(v is True for v in leaky["leak_dropped_by_tool"])
    lines += ["", f"**Verdict:** across the {len(leaky)} leaky AutoML runs, {named} log lines name the leak column "
              f"and {'at least one tool dropped it' if dropped_any else 'no tool dropped it'}. "
              + ("So no tool singled the column out: no warning or note in any log names it. "
                 if named == 0 and not dropped_any else "")
              + (("The flag-word and warning lines below do not name the leak column; they are the tools' "
                  "generic messages (constant or useless columns, the evaluation metric)"
                  + ("; each tool prints at least as many flag-word lines in its clean run." if _generic_in_clean(table)
                    else "."))
                 if named == 0 else "The lines that name it are listed below; read them before concluding.")]
    lines += ["", "Lines that matched (first few per run):", *examples] if examples else \
        ["", "No line in any log matched a flag word or named the leak column."]
    return "\n".join(lines)


def _generic_in_clean(table: pd.DataFrame) -> bool:
    """True if each tool's clean run has at least as many flag-word lines as any of its leaky runs."""
    tools = table[table["model"] != leakage.LGBM]
    for _, part in tools.groupby("model"):
        clean = part[part["variant"] == leakage.CLEAN]
        if clean.empty or (part["n_flag_word_lines"] > int(clean["n_flag_word_lines"].iloc[0])).any():
            return False
    return True


STATIC_TAIL = """## Why this matters

* A leaked score is not a slightly optimistic score: it measures a different task. With
  `detection_ratio` the model is no longer a detector, it reads the antivirus verdict that the
  label was made from. Deployed, that column does not exist yet when a new file arrives, and the
  model's real performance is unknown.
* AutoML makes this easier to miss, not harder: the search is rewarded for exactly the column a
  human should distrust, and the result arrives as a polished leaderboard with a near-perfect score.
* The date leak is the subtle kind. It looks harmless on its own and its importance can be modest,
  but it is a calendar, not a file property, and a tree cannot extrapolate it past the last training
  date.

## What protects against it

1. **A domain review of feature provenance, before any training.** For every column ask: what is
   it, who produces it, and is it known at the moment the model must decide? `detection_ratio`
   (made by the same AV scans as the label) and `first_submission_date` (a VirusTotal bookkeeping
   time) both fail this test. No tool did this review for us; it has to be a human step.
2. **Temporal validation for time-like leaks.** Training on earlier weeks and testing on later
   ones (our `test_future`) puts every date out of range, so a model cannot use the future dates
   themselves, and a validation score taken that way shows what a date column is really worth
   after deployment. It does not catch post-hoc label proxies such as `detection_ratio`, which
   stay available (and perfect) in every period.
3. **Treat a too-good score as a bug report.** An AUC of 1.0 on malware detection, or one feature
   with most of the total importance, should trigger a look at that feature before anything else.
"""


def _autogluon_permutation_coverage() -> str:
    """How much of AutoGluon's permutation importance was computed per run, from the saved importance files.

    AutoGluon stops the importance at a time cap (the watched columns go first, see leakage.run_tool), so
    the features it did not reach have no value; `n` is the number of shuffles per feature.
    """
    parts, all_shuffles = [], set()
    for variant in leakage.VARIANTS:
        path = leakage.LEAKAGE_LEADERBOARDS / variant / f"autogluon_feature_importance{RUN_SUFFIX}.csv"
        if not path.exists():
            continue
        imp = pd.read_csv(path)
        done = imp["importance"].notna()
        all_shuffles |= {int(n) for n in imp.loc[done, "n"].dropna()}
        parts.append(f"`{variant}` {int(done.sum()):,} of the {len(imp):,} features it kept")
    if not parts:
        return ""
    shuffle_txt = (" Each feature was shuffled once, so the values carry shuffle noise." if all_shuffles == {1}
                   else f" Shuffles per feature: {sorted(all_shuffles)}.")
    return ("* AutoGluon's permutation importance runs under a time cap (the watched columns are computed "
            "first) and was computed for: " + "; ".join(parts) + ". Its ranks are among the computed "
            "features only." + shuffle_txt)


def _caveats(table: pd.DataFrame) -> str:
    coverage = _autogluon_permutation_coverage()
    return "\n".join([
        "## Caveats",
        "",
        "* One seed and one training sample per run; the CIs cover test-set sampling noise only, not "
        "training randomness. Small deltas (around 1e-4 AUC) are within run-to-run variation of the AutoML "
        "searches, which are not deterministic in time-budgeted mode.",
        f"* The AutoML runs had {LEAKAGE_BUDGET_S} s each, so their clean scores are not the Step 3 scores; "
        "compare each leak run only with the clean run at the same budget.",
        "* Importances are not comparable across tools (gain, permutation, H2O varimp measure different "
        "things), so only ranks are compared. H2O's varimp and SHAP come from the best H2O model that has "
        "them (a GBM or DRF), which may not be the stacked-ensemble leader. AutoGluon's permutation "
        "importance uses its own internal validation rows (a few thousand rows of `train`). Permutation "
        "importances can be negative, so a column's share of their sum can exceed 100% (or be negative "
        "when the sum is negative).",
        *([coverage] if coverage else []),
        "* `relied_on_leak` is our rule (kept the column and ranked it first; a split-count importance, "
        "FLAML's, does not count, only its SHAP rank), not a statement by the tool.",
    ])


def write(table: pd.DataFrame, context: dict[str, Any], csv_path: Path, figure: Path) -> Path:
    """Write reports/leakage[_quick].md and return its path."""
    header = [
        "# Step 4A: leakage experiments" + (" (QUICK smoke run: tiny data, not results)" if QUICK else ""),
        "",
        "Generated by `scripts/run_leakage.py` (`src/leakage_writeup.py`) from `reports/experiments.jsonl` "
        f"(records with step `leakage`), the saved test scores and run logs. Table: `{_rel(csv_path)}`. "
        f"Figure: `{_rel(figure)}`. Do not edit by hand.",
        "",
        f"![leakage jump]({figure.relative_to(REPORTS).as_posix()})",
    ]
    parts = [*header, "", _setup(table, context), "", _results(table), "", _leak_dr(table), "",
             _leak_date(table, context), "", _ablation(table, context), "", _warnings(table, context), "",
             STATIC_TAIL, _caveats(table), ""]
    path = REPORTS / f"leakage{RUN_SUFFIX}.md"
    path.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    return path
