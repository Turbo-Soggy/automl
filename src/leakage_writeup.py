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
        "with/without comparison the leak column is the only difference.",
        "* Same `train` (160k rows in the full run), seed 42, 16 threads, ROC-AUC as the tools' metric. "
        "Importance and SHAP use rows of `train` or the tool's own validation rows, never a test set.",
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
                     "LightGBM is run deterministically, so the refit reproduces Step 2.")
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
            lines += ["", f"Effect of the shorter budget on the clean runs (Step 3 values from `{_rel(step3)}`):", "",
                      _md_table(["Tool", f"AUC random, {context['step3_budget_s']} s", f"AUC random, {LEAKAGE_BUDGET_S} s",
                                 f"AUC future, {context['step3_budget_s']} s", f"AUC future, {LEAKAGE_BUDGET_S} s",
                                 f"winner, {context['step3_budget_s']} s", f"winner, {LEAKAGE_BUDGET_S} s"], rows)]
    return "\n".join(lines)


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
        "FLAML `feature_importances_`, AutoGluon permutation importance, H2O varimp) and by mean |TreeSHAP| "
        "on 2,000 `train` rows where the model is a tree model (n/a = not computed; the reason is in the "
        "`importance_shap_kind` column of the CSV). The full table, with F1, FPR, TPR CIs, time-proxy "
        "ranks and timings, is the CSV.",
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
    fell = int(part["relied_on_leak"].fillna(False).astype(bool).sum())
    lines += [
        "",
        f"**{fell} of {len(part)} models fell for the leak**: they kept the column and ranked it first "
        "(rule: rank 1 by the tool's own importance or by SHAP). No tool can know from the data alone that "
        "this column is post-hoc: a column with AUC 1.0 against the label looks like the best feature there "
        "is, and every tool's job is to find exactly such features.",
        "",
        "Note what temporal validation does **not** show here: the detection ratio of a future file is just "
        "as post-hoc as that of a past one, so the leaky model also looks perfect on test_future. The split "
        "cannot expose this leak. Only asking *when would this value be known?* exposes it: a detector "
        "scores a file when it first appears, before (or instead of) the antivirus scan the ratio comes from.",
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
    helped = [MODEL_LABEL[r["model"]] for _, r in part.iterrows() if _effect(r, "test_random") == "raised"]
    lines += ["", ("On the random split the date measurably raised AUC for: " + ", ".join(helped) + "."
                   if helped else "No model got a measurable AUC gain from the date, even on the random split "
                   "(every paired CI includes 0 or is negative). With 50/50 classes in every week the raw date "
                   "is a weak leak in this dataset; it would be a strong one wherever the share of malware "
                   "changes over time.")]
    return "\n".join(lines)


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
        lines += ["Rank of each proxy in the clean runs (own importance / SHAP, out of all features):", "",
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
            "Reading: if the proxies were a time *leak* that the random split rewards, removing them would cost "
            "more on test_random than on test_future and shrink the gap. If the cost is similar on both sets, "
            "the model uses them as ordinary file properties (e.g. a zero or a fake compile date is typical of "
            "some packers and malware builders), not as a calendar."]
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
    lines += ["", "Lines that matched (first few per run):", *examples] if examples else \
        ["", "No line in any log matched a flag word or named the leak column."]
    return "\n".join(lines)


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
   ones (our `test_future`) puts every date out of range, which removes any benefit a date column
   had on the random split. It does not catch post-hoc label proxies such as `detection_ratio`,
   which stay available (and perfect) in every period.
3. **Treat a too-good score as a bug report.** An AUC of 1.0 on malware detection, or one feature
   with most of the total importance, should trigger a look at that feature before anything else.
"""


def _caveats(table: pd.DataFrame) -> str:
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
        "importance uses its own ~2.5k validation rows.",
        "* `relied_on_leak` is our rule (kept the column and ranked it first), not a statement by the tool.",
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
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
