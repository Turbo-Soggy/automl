"""Render reports/critical_eval.md and reports/limitations.md (Step 4 synthesis) from tables only.

Inputs: the CSVs and JSON facts that Step 4 wrote to reports/tables/ (src/critical_eval.py,
src/eval_bias_report.py, src/leakage_report.py, scripts/run_explain.py), plus the data checks of
Step 0 (reports/data_check.json, data/splits/splits_info.json). Every number in both reports is read
from those files, and the sentences that make a claim ("falls", "differs", "no tool") are chosen by
code from the numbers, so the text follows the results if they change. Do not edit the .md by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import config
from config import REPORTS, RUN_SUFFIX, TABLES
from src.logging_utils import read_experiments
from src.temporal_writeup import MODEL_NAMES

# Steps 2-3 ran here; the experiment log of that time has no machine field, so it is described once.
LAPTOP: str = "a 16-core Intel Core Ultra 9 285H laptop (63 GB RAM, Windows)"
CHALLENGE_ALL: str = "all Win64 challenge"
OOF_THRESHOLD: str = "train_oof_1pct_fpr"
HEADLINE_PREVALENCE: float = 0.01
# Our rule for calling an AUC gap "small" in the text (the measured gaps are in the tables).
SMALL_AUC_GAP: float = 0.005
METRIC_NAMES: dict[str, str] = {"roc_auc": "AUC", "tpr_at_1pct_fpr": "TPR@1%FPR"}
LEAK_NAMES: dict[str, str] = {"leak_detection_ratio": "`detection_ratio`",
                              "leak_first_submission": "`first_submission_date`"}
TIME_PROXY_COLUMNS: tuple[int, ...] = (696, 2478, 2479)  # COFF timestamp, two signing times (src/leakage.py)
TOOL_NAMES: dict[str, str] = {"lgbm_tuned": "tuned LightGBM", "flaml": "FLAML", "autogluon": "AutoGluon", "h2o": "H2O"}


# --- Reading ------------------------------------------------------------------------------
def _csv(stem: str) -> pd.DataFrame:
    path = TABLES / f"{stem}{RUN_SUFFIX}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the Step 4 scripts first (see scripts/run_critical_eval.py)")
    return pd.read_csv(path, dtype={"week": str})


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict[str, Any]:
    """Every table and fact the two reports use, read once."""
    stems = ["random_vs_future", "precision_both_sets", "prevalence", "challenge", "leakage", "gap_vs_tuned",
             "automl_self_estimate", "cv_comparison", "drift", "gap_adversarial", "gap_adversarial_top_features",
             "class_weight", "calibration", "bias_slices", "class_balance", "shap_family_importance",
             "cross_tool_agreement", "cross_tool_top20"]
    inputs: dict[str, Any] = {stem: _csv(stem) for stem in stems}
    inputs["facts"] = _json(TABLES / f"eval_bias_facts{RUN_SUFFIX}.json")
    inputs["data_check"] = _json(REPORTS / f"data_check{RUN_SUFFIX}.json")
    inputs["splits_info"] = _json(config.split_file("splits_info", ".json"))
    inputs["vm"] = _vm_machine(inputs["facts"])
    return inputs


def _vm_machine(facts: dict[str, Any]) -> dict[str, Any]:
    """CPU, cores, RAM and OS of the Step 4 machine, from the latest leakage record (else eval facts)."""
    df = read_experiments(config.EXPERIMENT_LOG)
    if not df.empty and "step" in df:
        df = df[(df["step"] == "leakage") & (df["quick"] == config.QUICK)]
        if not df.empty and "machine.cpu" in df:
            rec = df.sort_values("timestamp_utc").iloc[-1]
            return {"cpu": rec["machine.cpu"], "cpu_count": int(rec["machine.cpu_count"]),
                    "ram_gb": float(rec["machine.ram_gb"]), "os": rec["machine.os"], "n_jobs": int(rec["n_jobs"])}
    m = facts["machine"]
    return {"cpu": m.get("processor", "n/a"), "cpu_count": m["cpu_count"], "ram_gb": m["ram_gib"], "os": m["os"],
            "n_jobs": config.N_JOBS}


def _one(df: pd.DataFrame, **match: Any) -> pd.Series:
    """The single row of `df` whose columns equal `match`; anything else is a bug in the inputs."""
    mask = pd.Series(True, index=df.index)
    for col, value in match.items():
        mask &= df[col] == value
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {match}, found {len(rows)}")
    return rows.iloc[0]


# --- Formatting ---------------------------------------------------------------------------
def _f4(x: Any) -> str:
    return "n/a" if pd.isna(x) else f"{float(x):.4f}"


def _pct(x: Any, digits: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{100 * float(x):.{digits}f}%"


def _pp(x: Any) -> str:
    return "n/a" if pd.isna(x) else f"{100 * float(x):+.2f} pp"


def _int(x: Any) -> str:
    return "n/a" if pd.isna(x) else f"{float(x):,.0f}"


def _ci(row: pd.Series, col: str, fmt=_f4) -> str:
    return f"{fmt(row[col])} [{fmt(row[col + '_ci_low'])}, {fmt(row[col + '_ci_high'])}]"


def _detection_ci(row: pd.Series) -> str:
    """A challenge.csv row's detection rate with its Wilson CI (columns ci_low / ci_high)."""
    return f"{_pct(row['detection_rate'])} [{_pct(row['ci_low'])}, {_pct(row['ci_high'])}]"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _name(model: str) -> str:
    return MODEL_NAMES.get(model, TOOL_NAMES.get(model, model))


def _src(stem: str) -> str:
    return f"`reports/tables/{stem}{RUN_SUFFIX}.csv`"


def _range(values: pd.Series) -> str:
    """'a to b', or one value when both round to the same 4 decimals."""
    lo, hi = _f4(values.min()), _f4(values.max())
    return lo if lo == hi else f"{lo} to {hi}"


def _moves(before: float, after: float) -> str:
    return "rises" if after > before else "falls" if after < before else "stays"


def _excludes_zero(row: pd.Series, col: str) -> bool:
    return bool(row[f"{col}_ci_low"] > 0 or row[f"{col}_ci_high"] < 0)


# --- Shared numbers -----------------------------------------------------------------------
def _tuned(t: dict[str, Any]) -> str:
    return t["facts"]["tuned_tuner"]


def _automl_drift_verdict(t: dict[str, Any], metric: str = "roc_auc") -> str:
    """One sentence on the AutoML tools' drift gaps against the tuned model's, from gap_vs_tuned.csv."""
    gap = t["gap_vs_tuned"]
    auto = gap[(gap["kind"] == "automl") & (gap["metric"] == metric)]
    differs = auto["differs_from_tuned"].astype(bool)
    worse, better = auto[differs & (auto["gap_minus_tuned"] > 0)], auto[differs & (auto["gap_minus_tuned"] < 0)]

    def listed(rows: pd.DataFrame) -> str:
        return ", ".join(f"{_name(r['model'])} ({r['gap_minus_tuned']:+.4f} [{r['gap_minus_tuned_ci_low']:+.4f}, "
                         f"{r['gap_minus_tuned_ci_high']:+.4f}])" for _, r in rows.iterrows())
    name = METRIC_NAMES.get(metric, metric)
    smaller = f"a smaller {name} gap than the tuned LightGBM: {listed(better)}" if len(better) else \
        f"none of the {len(auto)} AutoML tools has a smaller {name} gap than the tuned LightGBM"
    larger = f"a larger one: {listed(worse)}" if len(worse) else "none has a larger one"
    return f"{smaller}; {larger} (paired bootstrap CI of the gap difference)"


def _prec(t: dict[str, Any], model: str, threshold: str, test: str,
          prevalence: float = HEADLINE_PREVALENCE) -> pd.Series:
    df = t["precision_both_sets"]
    return _one(df[(df["prevalence"] - prevalence).abs() < 1e-9], model=model, threshold_name=threshold, test_set=test)


def _challenge(t: dict[str, Any], model: str, threshold: str) -> pd.Series | None:
    df = t["challenge"]
    rows = df[(df["model"] == model) & (df["threshold_name"] == threshold) & (df["subset"] == CHALLENGE_ALL)]
    return rows.iloc[0] if len(rows) == 1 else None


def _machine_text(t: dict[str, Any]) -> str:
    vm = t["vm"]
    return (f"Steps 2-3 (the tuning searches and the {config.AUTOML_BUDGET_S} s AutoML runs, and the saved models "
            f"scored here) ran on {LAPTOP} with 16 threads. Step 4 ran on a cloud VM: {vm['cpu']}, "
            f"{vm['cpu_count']} logical CPUs, {vm['ram_gb']:.1f} GB RAM, {vm['os']}; the leakage runs used "
            f"{vm['n_jobs']} threads (the part B and C fits {t['facts']['eval_n_jobs']}, because those two jobs "
            "shared the machine). Every with/without comparison inside Step 4 was run on the VM with the same "
            "settings. No wall-clock time is compared across the two machines, and the AutoML scores at the "
            "Step 4 budget are not the Step 3 scores (fewer cores search fewer configurations in the same time)."
            + _challenge_machine_clause(t))


def _challenge_machine_clause(t: dict[str, Any]) -> str:
    """Where the challenge-set scoring ran, from its own record (it needs every saved model)."""
    sm = t["facts"].get("challenge_scoring", {}).get("machine") or {}
    if not sm or sm.get("cpu_count") == t["vm"]["cpu_count"]:
        return ""
    return (f" The challenge-set scores (sections 1, 2 and the verdict) were computed on a {sm['cpu_count']}-CPU "
            f"{sm['os'].split('-')[0]} machine (the laptop), which holds every saved Step 2-3 model; scoring "
            "fits nothing, so the hardware does not change those numbers.")


# =============================================================================
# reports/critical_eval.md
# =============================================================================
def _header(t: dict[str, Any]) -> str:
    stems = ["random_vs_future", "precision_both_sets", "challenge", "leakage", "gap_vs_tuned",
             "automl_self_estimate", "cv_comparison", "drift", "gap_adversarial", "class_weight", "calibration"]
    return "\n".join([
        "# Critical evaluation: how far do random-split scores overstate real-world performance, "
        "and does AutoML protect against leakage and drift?", "",
        "Generated by `scripts/run_critical_eval.py` (`src/critical_eval_writeup.py`) from "
        + ", ".join(_src(s) for s in stems) + " and `eval_bias_facts.json`. Nothing here is typed by hand; "
        "the detail behind each section is in [`leakage.md`](leakage.md), "
        "[`temporal_bias.md`](temporal_bias.md) and [`explainability.md`](explainability.md), and the "
        "threats to validity are in [`limitations.md`](limitations.md).", "",
        f"**Machine.** {_machine_text(t)}", "",
        "**Test-set discipline.** No model, configuration, early-stopping point or threshold was chosen on "
        "test_random or test_future. The thresholds used below are 0.5, the 1%-FPR threshold set on TRAIN "
        "out-of-fold scores (`train_oof_1pct_fpr`) and the cost-sensitive deployment threshold in "
        "`models/threshold.json` (also set on TRAIN out-of-fold scores). CIs are 95%: bootstrap "
        f"({config.N_BOOTSTRAP} resamples, seed {config.SEED}) or Wilson for rates.", ""])


def _short_answer(t: dict[str, Any]) -> str:
    tuned = _tuned(t)
    rvf = t["random_vs_future"]
    r = _one(rvf, model=tuned)
    trees = rvf[rvf["model"] != "logreg"]
    lr = _one(rvf, model="logreg")
    pr, pf = _prec(t, tuned, OOF_THRESHOLD, "test_random"), _prec(t, tuned, OOF_THRESHOLD, "test_future")
    ch = _challenge(t, tuned, OOF_THRESHOLD)
    leak = t["leakage"]
    dr = leak[leak["variant"] == "leak_detection_ratio"]
    fell = int(dr["relied_on_leak"].astype(bool).sum())
    named = int(dr["n_lines_mentioning_leak"].fillna(0).sum())
    dropped = int(dr["leak_dropped_by_tool"].astype(str).str.lower().eq("true").sum())
    gap = t["gap_vs_tuned"]
    auto_auc = gap[(gap["kind"] == "automl") & (gap["metric"] == "roc_auc")]
    est = t["automl_self_estimate"]
    lines = ["## The answer in short", ""]
    lines.append(
        f"1. **{'Ranking metrics barely move for the tree models.' if trees['roc_auc_gap'].max() < SMALL_AUC_GAP else 'Ranking metrics fall on future weeks.'}** From test_random to test_future the tuned "
        f"LightGBM goes from AUC {_f4(r['test_random_roc_auc'])} to {_f4(r['test_future_roc_auc'])} (gap "
        f"{_ci(r, 'roc_auc_gap')}) and from TPR@1%FPR {_pct(r['test_random_tpr_at_1pct_fpr'])} to "
        f"{_pct(r['test_future_tpr_at_1pct_fpr'])} (gap {_ci(r, 'tpr_at_1pct_fpr_gap', _pp)}). The AUC gaps of all "
        f"{len(trees)} tree models lie between {_f4(trees['roc_auc_gap'].min())} and "
        f"{_f4(trees['roc_auc_gap'].max())}; only LogReg, which cannot cap out-of-range feature values, loses "
        f"much more ({_f4(lr['roc_auc_gap'])} AUC)."
        + (" By AUC alone, the random split overstates the tuned model's real-world performance only slightly."
           if 0 <= r["roc_auc_gap"] < SMALL_AUC_GAP else ""))
    fpr_rel = pf["fpr"] / pr["fpr"] - 1 if pr["fpr"] else float("nan")
    head2 = ("The overstatement is in the operating point and the base rate, which AUC hides."
             if pf["fpr"] > pr["fpr"] else "The base rate matters more than the split.")
    lines.append(
        f"2. **{head2}** At the "
        f"threshold fixed on train (t = {pf['threshold']:.4f}) the false-positive rate {_moves(pr['fpr'], pf['fpr'])} from "
        f"{_pct(pr['fpr'], 2)} on test_random to {_pct(pf['fpr'], 2)} on test_future ({fpr_rel:+.0%} relative). "
        f"On the 50/50 test sets precision is {_pct(pr['precision_at_test_mix'])} / "
        f"{_pct(pf['precision_at_test_mix'])}; at a realistic {HEADLINE_PREVALENCE:.0%} malware prevalence it "
        f"is {_ci(pr, 'precision', _pct)} on test_random and {_ci(pf, 'precision', _pct)} on test_future, "
        f"i.e. {_int(pf['false_alarms_per_10k'])} false alarms per 10,000 files scanned "
        f"(random split: {_int(pr['false_alarms_per_10k'])}). "
        + _mix_vs_split_sentence(pr, pf))
    if ch is not None:
        lines.append(
            f"3. **The hard tail is invisible to both test sets.** On the {int(ch['n'])} Win64 challenge-set "
            f"files (malware that antivirus engines missed when first seen) the tuned model detects "
            f"{_detection_ci(ch)} at the same threshold, against "
            f"{_pct(ch['test_future_tpr_same_threshold'])} of ordinary test_future malware.")
    auto_dr = dr[dr["model"] != "lgbm_tuned"]
    auto_fell = int(auto_dr["relied_on_leak"].astype(bool).sum())
    head = ("No AutoML tool protects against the label leak." if auto_fell == len(auto_dr)
            else f"{len(auto_dr) - auto_fell} of {len(auto_dr)} AutoML tools did not rely on the label leak.")
    lines.append(
        f"4. **{head}** With `detection_ratio` added, {fell} of "
        f"{len(dr)} models (tuned LightGBM and the three AutoML tools) ranked it first, by their own importance "
        f"or by SHAP, and scored test_future AUC "
        f"{_range(dr['test_future_roc_auc'])}; {named} log lines named the column and {dropped} "
        "tools dropped it. The future split does not expose this leak either: a post-hoc label proxy is just "
        "as available (and perfect) in later weeks.")
    clean_own = est[est["variant"] == "clean"]
    leak_own = est[est["variant"] == "leak_detection_ratio"]
    random_val = bool(len(est)) and bool(est["own_validation"].str.startswith("random").all())
    head = ("AutoML does not protect against drift." if not (auto_auc["gap_minus_tuned_ci_high"] < 0).any()
            else "Some AutoML tool lost less to drift than the tuned model.")
    verdict = _automl_drift_verdict(t)
    lines.append(
        f"5. **{head}** {verdict[0].upper() + verdict[1:]}, {_src('gap_vs_tuned')}. "
        + ("Every tool validated on a random split of `train` (the same weeks as its training rows)"
           if random_val else "The tools' validation schemes are listed in section 5")
        + f": its own validation AUC was {_range(clean_own['own_auc'])} on the clean runs, against test_future "
        f"{_range(clean_own['test_future_roc_auc'])}"
        + (f", and {_range(leak_own['own_auc'])} with the leak: a near-perfect score that no "
           "tool questioned." if len(leak_own) else "."))
    return "\n".join(lines) + "\n"


def _mix_vs_split_sentence(pr: pd.Series, pf: pd.Series) -> str:
    """Which costs more precision: reading a 50/50 test set as real traffic, or the random split?"""
    mix = pr["precision"] - pr["precision_at_test_mix"]    # 50/50 -> 1% prevalence, on test_random
    split = pf["precision"] - pr["precision"]              # test_random -> test_future, at 1% prevalence
    bigger = "the 50/50 test mix" if abs(mix) > abs(split) else "the random split"
    return (f"Reading the 50/50 test_random as {HEADLINE_PREVALENCE:.0%}-prevalence traffic changes precision "
            f"by {_pp(mix)}; moving from test_random to test_future at that prevalence changes it by "
            f"{_pp(split)}: {bigger} is the larger source of optimism.")


def _ladder(t: dict[str, Any]) -> str:
    tuned = _tuned(t)
    cv, rvf = t["cv_comparison"], t["random_vs_future"]
    r = _one(rvf, model=tuned)
    rows = []
    for est, label in (("stratified_3fold_cv", "stratified 3-fold CV on train (out-of-fold)"),
                       ("forward_cv_4fold", "forward 4-fold CV on train (week-blocked)")):
        c = _one(cv, estimate=est)
        rows.append([label, _f4(c["roc_auc"]), _pct(c["tpr_at_1pct_fpr"]), "", "", "", ""])
    weeks = {s: t["splits_info"]["splits"][s]["n_weeks"] for s in ("test_random", "test_future")}
    for test, label in (("test_random", f"test_random: same {weeks['test_random']} weeks as train, 50/50"),
                        ("test_future", f"test_future: {weeks['test_future']} later weeks, 50/50")):
        p = _prec(t, tuned, OOF_THRESHOLD, test)
        rows.append([label, _f4(r[f"{test}_roc_auc"]), _pct(r[f"{test}_tpr_at_1pct_fpr"]), _pct(p["tpr"]),
                     _pct(p["fpr"], 2), _pct(p["precision_at_test_mix"]), _ci(p, "precision", _pct)])
    ch = _challenge(t, tuned, OOF_THRESHOLD)
    if ch is not None:
        rows.append([f"challenge set: {int(ch['n'])} Win64 files AV missed at first (all malicious)", "", "",
                     _detection_ci(ch), "", "", ""])
    pr, pf = _prec(t, tuned, OOF_THRESHOLD, "test_random"), _prec(t, tuned, OOF_THRESHOLD, "test_future")
    table = _table(["evaluation", "AUC", "TPR@1%FPR", "TPR at t", "FPR at t", "precision, 50/50 mix",
                    f"precision at {HEADLINE_PREVALENCE:.0%} prevalence [CI]"], rows)
    steps = (f"From test_random to test_future: AUC {r['test_future_roc_auc'] - r['test_random_roc_auc']:+.4f}, "
             f"TPR@1%FPR {_pp(r['test_future_tpr_at_1pct_fpr'] - r['test_random_tpr_at_1pct_fpr'])}, "
             f"FPR at t {_pp(pf['fpr'] - pr['fpr'])}, precision at {HEADLINE_PREVALENCE:.0%} prevalence "
             f"{_pp(pf['precision'] - pr['precision'])}.")
    if ch is not None:
        steps += (f" From ordinary test_future malware to the challenge set: recall at t "
                  f"{_pp(ch['detection_rate'] - pf['tpr'])}.")
    strat, fwd = _one(cv, estimate="stratified_3fold_cv"), _one(cv, estimate="forward_cv_4fold")
    future_auc = r["test_future_roc_auc"]
    if strat["roc_auc"] > future_auc > fwd["roc_auc"]:
        cv_line = ("* The two CV schemes bracket test_future AUC: stratified CV (every fold mixes all 52 weeks, "
                   "like test_random) is optimistic, the forward CV (validate on later weeks only) pessimistic, "
                   "partly because its folds fit on fewer rows (`temporal_bias.md`, section 2).")
    else:
        cv_line = (f"* Stratified CV AUC {_f4(strat['roc_auc'])} and forward CV AUC {_f4(fwd['roc_auc'])} against "
                   f"test_future {_f4(future_auc)} (`temporal_bias.md`, section 2).")
    mix_bigger = abs(pr["precision"] - pr["precision_at_test_mix"]) > abs(pf["precision"] - pr["precision"])
    return "\n".join([
        "## 1. The evaluation ladder: the same tuned model, scored more and more realistically", "",
        f"The tuned LightGBM (`{tuned}`, chosen by CV AUC in Step 2) at the threshold t = {pf['threshold']:.4f} "
        "that gives 1% FPR on its TRAIN out-of-fold scores. Precision at a prevalence pi is "
        "pi*TPR / (pi*TPR + (1-pi)*FPR); its CI combines the Wilson bounds of TPR and FPR in the worst and best "
        f"direction. Sources: {_src('cv_comparison')}, {_src('random_vs_future')}, "
        f"{_src('precision_both_sets')}, {_src('challenge')}.", "",
        table, "",
        steps, "",
        cv_line,
        "* AUC and TPR@1%FPR are per-class rates, so they do not depend on the 50/50 mix; precision does"
        + (". That is why the base rate moves precision far more than the random-to-future step does." if mix_bigger
           else "."), ""])


def _all_models(t: dict[str, Any]) -> str:
    prec = t["precision_both_sets"]
    keys = prec[["model", "threshold_name"]].drop_duplicates().itertuples(index=False)
    rows, drops = [], []
    for model, thr in keys:
        pr, pf = _prec(t, model, thr, "test_random"), _prec(t, model, thr, "test_future")
        ch = _challenge(t, model, thr)
        drops.append((pf["precision"] - pr["precision"], model, thr))
        rows.append([_name(model), f"`{thr}`", f"{pf['threshold']:.4f}", f"{_pct(pr['fpr'], 2)} / {_pct(pf['fpr'], 2)}",
                     f"{_pct(pr['precision'])} / {_ci(pf, 'precision', _pct)}",
                     f"{_int(pr['false_alarms_per_10k'])} / {_int(pf['false_alarms_per_10k'])}",
                     "not scored" if ch is None else _detection_ci(ch)])
    n_fall = sum(d < 0 for d, _, _ in drops)
    worst = min(drops)
    not_scored = t["facts"].get("challenge_scoring", {}).get("not_scored", {})
    note = "".join(f" {_name(m)} has no challenge score: {why}." for m, why in not_scored.items())
    return "\n".join([
        f"## 2. Every model at {HEADLINE_PREVALENCE:.0%} malware prevalence, both test sets, and the challenge set", "",
        f"Columns are *test_random / test_future*. Sources: {_src('precision_both_sets')} (also 5% and 10% "
        f"prevalence), {_src('challenge')}.{note}", "",
        _table(["model", "threshold", "t", "FPR", f"precision at {HEADLINE_PREVALENCE:.0%} [CI on future]",
                "false alarms / 10k files", "challenge detection [CI]"], rows), "",
        f"* Precision at {HEADLINE_PREVALENCE:.0%} prevalence is lower on test_future than on test_random for "
        f"{n_fall} of {len(drops)} model/threshold pairs; the largest fall is {_name(worst[1])} at `{worst[2]}` "
        f"({_pp(worst[0])}).",
        _challenge_sentence(t), ""])


def _challenge_sentence(t: dict[str, Any]) -> str:
    ch = t["challenge"]
    ch = ch[ch["subset"] == CHALLENGE_ALL]
    if ch.empty:
        return "* The challenge set was not scored."
    usual = ch[ch["threshold_name"] != "deployment"]
    best = usual.loc[usual["detection_rate"].idxmax()]
    text = (f"* At 0.5 or the train 1%-FPR threshold the best challenge-set detection is "
            f"{_pct(best['detection_rate'])} ({_name(best['model'])}, `{best['threshold_name']}`).")
    dep = ch[ch["threshold_name"] == "deployment"]
    if len(dep):
        d = dep.iloc[0]
        pf = _prec(t, d["model"], "deployment", "test_future")
        text += (f" The low cost-sensitive deployment threshold (t = {d['threshold']:.4f}) catches "
                 f"{_pct(d['detection_rate'])}, at the price of a {_pct(pf['fpr'], 2)} FPR on test_future: "
                 "a lower threshold buys recall on the hard tail with false alarms.")
    return text


def _why_small(t: dict[str, Any]) -> str:
    adv = t["gap_adversarial"]
    control = adv[adv["comparison"].str.contains("control")].iloc[0]
    future = adv[~adv["comparison"].str.contains("control")].iloc[0]
    leak = t["leakage"]
    clean = _one(leak, model="lgbm_tuned", variant="clean")
    ablate = _one(leak, model="lgbm_tuned", variant="ablate_time_proxies")
    bal = t["class_balance"].set_index("split")["malicious_share"]
    slope = t["facts"]["drift_slopes_per_week"]["roc_auc"]["expanding"]
    drift = t["drift"]
    exp = _one(drift, model="expanding", week="all_test_future")
    slid = _one(drift, model="sliding_12w", week="all_test_future")
    # Size-matched control: same row count as sliding_12w, random weeks 0-51 (separates recency from size)
    size_m = _one(drift, model="size_matched", week="all_test_future")
    weekly = drift[drift["week"] != "all_test_future"]
    weekly = weekly[weekly["week"] != "all_test_random"]
    sl_w = weekly[weekly["model"] == "sliding_12w"].set_index("week")["roc_auc"]
    sm_w = weekly[weekly["model"] == "size_matched"].set_index("week")["roc_auc"]
    sl_beats = int((sl_w > sm_w.reindex(sl_w.index)).sum())
    top = t["gap_adversarial_top_features"]
    top = top[~top["comparison"].str.contains("control")]
    proxy_share = top.loc[top["column"].isin(TIME_PROXY_COLUMNS), "gain_share"].sum()
    shrinks = ablate["auc_gap_random_minus_future"] < clean["auc_gap_random_minus_future"]
    return "\n".join([
        "## 3. Why the random-vs-future gap is small for trees (and not a bug)", "",
        f"* **The inputs do drift.** A classifier separates train files from future files with AUC "
        f"{_f4(future['auc'])} (control, same weeks: {_f4(control['auc'])}); the three time-stamp columns "
        f"carry {_pct(proxy_share)} of its gain ({_src('gap_adversarial')}, "
        f"{_src('gap_adversarial_top_features')}). Trees cannot extrapolate: a value beyond the last split "
        "lands in the same leaf as the newest training values, which limits what drift in such columns can do "
        "to the detector.",
        f"* **Removing the time proxies inside X {'shrinks' if shrinks else 'does not shrink'} the gap.** "
        f"Without the COFF timestamp and the two signing-time columns the tuned model's AUC gap goes from "
        f"{clean['auc_gap_random_minus_future']:+.5f} to {ablate['auc_gap_random_minus_future']:+.5f} "
        f"({_src('leakage')}, `ablate_time_proxies`), so they are not a hidden time leak that flatters test_random.",
        f"* **No label shift.** The malicious share is {_pct(bal['train'], 2)} / {_pct(bal['test_random'], 2)} / "
        f"{_pct(bal['test_future'], 2)} in train / test_random / test_future ({_src('class_balance')}), by "
        "construction, so the score cannot move through the class mix.",
        f"* **Short horizon, slow decay.** Over the 12 future weeks the weekly AUC slope is {slope:+.5f} per "
        f"week ({_src('drift')}); a model retrained on only the last 12 training weeks scores "
        f"{_f4(slid['roc_auc'])} on test_future, against {_f4(exp['roc_auc'])} for the model trained on all 52"
        + (": at this horizon, more data beats recency" if slid["roc_auc"] < exp["roc_auc"] else "")
        + f". At equal size, recent weeks are {'worth slightly more' if slid['roc_auc'] > size_m['roc_auc'] else 'not worth more'}"
        f" than random weeks 0-51: {_f4(slid['roc_auc'])} vs {_f4(size_m['roc_auc'])} for a size-matched random "
        f"sample (higher in {sl_beats} of {len(sl_w)} future weeks).",
        "* **Weekly stratification** makes test_random an interpolation test: every test_random file has "
        "training files from its own week, which is the main reason it is the optimistic one.", ""])


def _leakage(t: dict[str, Any]) -> str:
    leak = t["leakage"].set_index(["model", "variant"])
    rows = []
    for variant, leak_name in LEAK_NAMES.items():
        for model in TOOL_NAMES:
            if (model, variant) not in leak.index or (model, "clean") not in leak.index:
                continue
            r, c = leak.loc[(model, variant)], leak.loc[(model, "clean")]
            star = " *" if _excludes_zero(r, "delta_test_future_roc_auc") else ""
            rows.append([leak_name, _name(model), f"{_f4(c['test_future_roc_auc'])} -> {_f4(r['test_future_roc_auc'])}",
                         f"{r['delta_test_future_roc_auc']:+.4f}{star}",
                         f"#{_int(r['leak_rank_native'])} / #{_int(r['leak_rank_shap'])}",
                         _int(r["n_lines_mentioning_leak"]),
                         "n/a (our model)" if pd.isna(r["leak_dropped_by_tool"]) else str(r["leak_dropped_by_tool"]).lower(),
                         "yes" if str(r["relied_on_leak"]).lower() == "true" else "no"])
    fs = t["leakage"][t["leakage"]["variant"] == "leak_first_submission"]
    dr = t["leakage"][t["leakage"]["variant"] == "leak_detection_ratio"]
    all_took = bool(dr["relied_on_leak"].astype(bool).all())
    leak_auc = t["data_check"]["detection_ratio"]["train"]["roc_auc_detection_ratio_float_vs_label"]
    n_sig = int(sum(_excludes_zero(r, "delta_test_future_roc_auc") for _, r in fs.iterrows()))
    est = t["automl_self_estimate"]
    own_leak = est[est["variant"] == "leak_detection_ratio"]
    # Did the date inflate test_random, as the plan expected of a time leak? (CI above 0)
    raised_random = [_name(r["model"]) for _, r in fs.iterrows() if r["delta_test_random_roc_auc_ci_low"] > 0]
    h2o_spread = _h2o_search_spread(t)
    lines = [
        "## 4. Does AutoML protect against leakage?", "",
        f"Each model was trained with and without one leak column at the same budget "
        f"({config.LEAKAGE_BUDGET_S} s per AutoML run, on the VM). Delta = leak minus clean on the same "
        f"test_future rows, `*` = paired-bootstrap CI excludes 0. Rank = the leak column's rank by the tool's "
        f"own importance / by mean |SHAP|. Source: {_src('leakage')}; the log scan is in "
        "`reports/leaderboards/leakage/log_scan.json`.", "",
        _table(["leak", "model", "AUC test_future clean -> leak", "delta", "leak rank (own / SHAP)",
                "log lines naming it", "dropped by tool", "relied on it (rank 1)"], rows), "",
        "*Relied on it* = kept the column and ranked it first by its own importance or by SHAP. FLAML's own "
        "importance (`feature_importances_` of its LightGBM learner) is the split count, which favours "
        "continuous high-cardinality columns such as a date, so for FLAML only its SHAP rank counts.", "",
        f"* **`detection_ratio` is a label leak, and {'every tool took it' if all_took else 'not every tool took it'}.** "
        "It is the share of antivirus engines that flagged the file, i.e. the thing the label was made from. "
        f"A column with AUC {leak_auc:.4f} against the label (`reports/data_check{RUN_SUFFIX}.json`) is exactly what a search for the best score is built to find, so no tool can be "
        "expected to reject it from the data alone; only a human question about provenance ('is this known "
        "when the file first arrives?') catches it.",
        f"* **`first_submission_date` is mostly harmless here, but for a fragile reason.** It moved "
        f"test_future AUC by at most {fs['delta_test_future_roc_auc'].abs().max():.4f}. For {n_sig} of {len(fs)} "
        "models that move excludes 0, but only under test-set resampling: the CIs do not cover the run-to-run "
        "variation of a time-budgeted AutoML search"
        + (f", and H2O's move ({h2o_spread['delta']:+.4f}) is smaller than the spread between H2O's own clean "
           f"searches (test_future AUC {_f4(h2o_spread['step3'])} in Step 3 vs {_f4(h2o_spread['clean'])} in the "
           "Step 4 clean run)" if h2o_spread else "")
        + ", so no effect of the date column on the AutoML tools is shown. "
        "Contrary to the plan's expectation (a time leak that helps under the random split and fails on "
        "test_future), the date did not inflate test_random AUC"
        + (f" except for {', '.join(raised_random)}, within the same search noise" if raised_random else "")
        + ", so this run does not demonstrate a time leak under the random split. "
        "Every future date lies after the last training "
        "date, so for a tree the column is a constant on test_future; with a 50/50 class mix in every week it "
        "carries almost no label information. In data where the malware share changes over time, the same "
        "column would let a model learn the calendar.",
    ]
    if len(own_leak):
        lines.append(
            f"* **The tools' own validation did not raise the alarm either:** with `detection_ratio` their "
            f"internal validation AUC was {_range(own_leak['own_auc'])} "
            f"({_src('automl_self_estimate')}). A near-perfect AUC on malware detection should itself be "
            "treated as a bug report.")
    return "\n".join(lines) + "\n"


def _h2o_search_spread(t: dict[str, Any]) -> dict[str, float] | None:
    """H2O's date-leak delta next to its Step 3 and Step 4 clean test_future AUCs (two searches)."""
    leak = t["leakage"]
    fs = leak[(leak["model"] == "h2o") & (leak["variant"] == "leak_first_submission")]
    clean = leak[(leak["model"] == "h2o") & (leak["variant"] == "clean")]
    est = t["automl_self_estimate"]
    step3 = est[(est["tool"] == "h2o") & (est["run"] == "step3")]
    if not (len(fs) and len(clean) and len(step3)):
        return None
    return {"delta": float(fs.iloc[0]["delta_test_future_roc_auc"]),
            "clean": float(clean.iloc[0]["test_future_roc_auc"]), "step3": float(step3.iloc[0]["test_future_roc_auc"])}


def _drift(t: dict[str, Any]) -> str:
    gap = t["gap_vs_tuned"]
    tuned = _tuned(t)
    rows = []
    for model in gap["model"].unique():
        a = _one(gap, model=model, metric="roc_auc")
        p = _one(gap, model=model, metric="tpr_at_1pct_fpr")
        fmt = lambda r, f: "-" if r["model"] == tuned else (  # noqa: E731
            f"{f(r['gap_minus_tuned'])} [{f(r['gap_minus_tuned_ci_low'])}, {f(r['gap_minus_tuned_ci_high'])}]"
            + (" *" if bool(r["differs_from_tuned"]) else ""))
        rows.append([_name(model), a["kind"], _f4(a["gap"]), fmt(a, lambda x: f"{x:+.4f}"), _pp(p["gap"]), fmt(p, _pp)])
    est = t["automl_self_estimate"]
    est_rows = [[TOOL_NAMES.get(r["tool"], r["tool"]), "Step 3" if r["run"] == "step3" else f"Step 4 `{r['variant']}`",
                 f"{r['budget_s']:.0f} s", r["machine"], r["own_validation"], _f4(r["own_auc"]),
                 _f4(r["test_random_roc_auc"]), _f4(r["test_future_roc_auc"]), f"{r['own_minus_future']:+.4f}"]
                for _, r in est.iterrows()]
    clean = est[est["variant"] == "clean"]
    n_over = int((clean["own_minus_future"] > 0).sum())
    return "\n".join([
        "## 5. Does AutoML protect against temporal drift?", "",
        "Gap = test_random minus test_future, for the saved Step 2-3 models. *Minus tuned* = the model's gap "
        "minus the tuned LightGBM's gap, with a paired bootstrap CI (both models scored on the same resampled "
        f"rows); `*` = the CI excludes 0. Source: {_src('gap_vs_tuned')}.", "",
        _table(["model", "kind", "AUC gap", "AUC gap minus tuned [CI]", "TPR@1%FPR gap", "TPR gap minus tuned [CI]"], rows), "",
        f"* AUC: {_automl_drift_verdict(t, 'roc_auc')}. TPR@1%FPR: {_automl_drift_verdict(t, 'tpr_at_1pct_fpr')}.",
        "* The AutoML winners are gradient-boosted trees or ensembles of them, so they share the tuned model's "
        "robustness to out-of-range values; a tool's search adds no protection of its own, because it only "
        "ever sees `train`.",
        "",
        f"What each AutoML tool told us about its own model ({_src('automl_self_estimate')}): its validation "
        "AUC (the leaderboard number), measured on its own split of `train`, against the two test sets.", "",
        _table(["tool", "run", "budget", "machine", "the tool's own validation", "own AUC", "AUC test_random",
                "AUC test_future", "own minus future"], est_rows), "",
        f"* {'All three tools' if bool(est['own_validation'].str.startswith('random').all()) else 'The tools'}, at "
        "their default settings, validate on a **random** split of `train`, which holds the same weeks as "
        "their training rows: the same design as test_random. Their own estimate was above the test_future "
        f"AUC in {n_over} of {len(clean)} clean runs. Each tool can be given a time-ordered "
        "validation instead (FLAML `split_type='time'`, an H2O `fold_column` or `validation_frame`, AutoGluon "
        "`tuning_data`); we tested the defaults, which is what a user who trusts the tool gets.",
        *_same_scores_lines(est),
        *(["* FLAML's own AUC is *below* its test scores in some runs: it is the holdout score of the best "
           "configuration during the search, which FLAML may have fitted on a subsample, before it retrains "
           "that configuration on all of `train`."]
          if ((est["tool"] == "flaml") & (est["own_minus_random"] < 0)).any() else []), ""])


def _same_scores_lines(est: pd.DataFrame) -> list[str]:
    """Note any Step 4 clean rerun whose test scores equal Step 3's exactly (column same_scores_as_step3)."""
    if "same_scores_as_step3" not in est:
        return []
    same = est[est["same_scores_as_step3"].astype(str).str.lower() == "true"]
    if same.empty:
        return []
    lines = []
    for _, r in same.iterrows():
        step3 = est[(est["tool"] == r["tool"]) & (est["run"] == "step3")]
        budget3 = f"{step3.iloc[0]['budget_s']:.0f} s" if len(step3) else "the Step 3 budget"
        lines.append(
            f"* {TOOL_NAMES.get(r['tool'], r['tool'])}: the Step 4 `clean` rerun ({r['budget_s']:.0f} s, VM) gave "
            f"exactly the same test scores as Step 3 ({budget3}, laptop): every saved score is equal. Its winner "
            "is a single seeded LightGBM, and AutoGluon keeps the boosting iteration that was best on its "
            "holdout; both runs reached that iteration (the shorter run's fit log reports running out of time "
            "after it), so the budget did not change the model.")
    return lines


def _bias_explain(t: dict[str, Any]) -> str:
    """Worst bias slices, top SHAP families and tool agreement: do the AutoML winners share blind spots?"""
    slices = t["bias_slices"]
    fut = slices[(slices["test_set"] == "test_future") & (slices["threshold_name"] == OOF_THRESHOLD)
                 & (slices["slice_family"] != "week")]
    # Only slices with enough files of the class the rate is about (Wilson CIs are wide below 100)
    worst_fpr = fut[fut["n_benign"] >= 100].sort_values("fpr").iloc[-1]
    worst_tpr = fut[fut["n_malicious"] >= 100].sort_values("tpr").iloc[0]
    fam_imp = t["shap_family_importance"]
    fams = ", ".join(f"{r.family} ({_pct(r.share)})" for r in fam_imp.sort_values("rank").head(3).itertuples())
    # The summed per-column share favours large families; the grouped |sum of SHAP| does not
    grouped = ", ".join(f"{r.family} ({r.grouped_mean_abs_shap:.2f}, {int(r.n_features)} columns)"
                        for r in fam_imp.sort_values("grouped_mean_abs_shap", ascending=False).head(3).itertuples())
    agree = t["cross_tool_agreement"].sort_values("family_spearman")
    most, least = agree.iloc[-1], agree.iloc[0]
    top5 = t["cross_tool_top20"].head(5)
    shared = sorted(set.intersection(*(set(top5[c]) for c in top5.columns if c != "rank")))
    return "\n".join([
        "## 5b. Bias slices, explanations and cross-tool agreement", "",
        f"Sources: {_src('bias_slices')} (tuned LightGBM, train 1%-FPR threshold), "
        f"{_src('shap_family_importance')}, {_src('cross_tool_agreement')}, {_src('cross_tool_top20')}; "
        "details in `temporal_bias.md` section 4a and `explainability.md`.", "",
        f"* **Weakest slices on test_future** (slices with at least 100 files of the class): the highest benign "
        f"FPR is `{worst_fpr['slice_family']}` = {worst_fpr['slice']}, {_pct(worst_fpr['fpr'], 2)} "
        f"[{_pct(worst_fpr['fpr_ci_low'], 2)}, {_pct(worst_fpr['fpr_ci_high'], 2)}] of "
        f"{int(worst_fpr['n_benign']):,} benign files; the lowest detection rate is "
        f"`{worst_tpr['slice_family']}` = {worst_tpr['slice']}, {_pct(worst_tpr['tpr'])} "
        f"[{_pct(worst_tpr['tpr_ci_low'])}, {_pct(worst_tpr['tpr_ci_high'])}] of "
        f"{int(worst_tpr['n_malicious']):,} malicious files.",
        f"* **What the tuned model uses** (test_future): by the summed per-column mean |SHAP| share, which "
        f"favours families with many columns, {fams}; by the grouped measure (mean |sum of the family's SHAP|), "
        f"{grouped}. The ranking depends on the measure; `explainability.md` explains both.",
        f"* **Do the tools agree?** By the rank correlation of feature-group importance, the most similar pair "
        f"is {_name(most['tool_a'])} / {_name(most['tool_b'])} (Spearman {most['family_spearman']:.2f}, "
        f"{int(most['top5_overlap'])} of their top 5 features shared) and the least similar "
        f"{_name(least['tool_a'])} / {_name(least['tool_b'])} ({least['family_spearman']:.2f}). Features in "
        "the top 5 of every tool: " + (", ".join(f"`{f}`" for f in shared) if shared else "none")
        + ". The measures differ by tool (SHAP, permutation importance, H2O varimp), so these are rough "
        "comparisons.",
        "* **What this means for AutoML:** the AutoML winners are tree ensembles on the same features and, "
        "where the measures are comparable, they rank the same signals high. So they are likely to share the "
        "same blind spots, such as the weaker detection of signed malware: AutoML changes how a model is "
        "searched for, not what it can see.", ""])


def _what_protects(t: dict[str, Any]) -> str:
    tuned = _tuned(t)
    dep = t["prevalence"]
    dep = dep[(dep["model"] == tuned) & (dep["threshold_name"] == "deployment")]
    cw = t["class_weight"]
    cw_f = cw[cw["test_set"] == "test_future"]
    cal = t["calibration"]
    iso = cal[cal["model"] == "lgbm_tuned+isotonic"]
    drift = t["drift"]
    exp = _one(drift, model="expanding", week="all_test_future")
    slid = _one(drift, model="sliding_12w", week="all_test_future")
    lines = ["## 6. What protects, and what does not", "",
             "* **A provenance review of every column, before training** (human): the only defence against "
             "`detection_ratio` in this project. It is cheap and no tool did it.",
             "* **A temporal hold-out** (train on earlier weeks, test on later ones): it measures drift and "
             "makes a date column useless, but it cannot catch a post-hoc label proxy (section 4)."]
    if len(dep):
        d = dep[(dep["prevalence"] - HEADLINE_PREVALENCE).abs() < 1e-9].iloc[0]
        lines.append(
            f"* **Thresholds and costs re-derived for the real base rate.** The 10:1 cost threshold in "
            f"`models/threshold.json` (t = {d['threshold']:.4f}), chosen on 50/50 train data, has FPR "
            f"{_pct(d['fpr'], 2)} on test_future and precision {_pct(d['precision'])} at "
            f"{HEADLINE_PREVALENCE:.0%} prevalence ({_src('prevalence')}): fine on a balanced test set, "
            "costly in real traffic.")
    if len(cw_f):
        moved = [r["model"] for _, r in cw_f.iterrows() if r["model"].startswith("class_weight_")
                 and r["model"] != "class_weight_none" and r["roc_auc_minus_baseline_ci_low"] > 0]
        lines.append(
            f"* **Class weights did not help:** {len(moved)} class-weighted refit(s) improved test_future AUC "
            f"beyond the CI ({_src('class_weight')}). With 50/50 classes there is no imbalance to correct, and "
            "reweighting only shifts scores, which a threshold does as well.")
    if len(iso):
        i = iso.iloc[0]
        helps = i["brier_diff_ci_high"] < 0
        lines.append(
            f"* **Calibration fitted on train out-of-fold scores {'carries over' if helps else 'does not clearly help'} "
            f"to later weeks:** isotonic calibration changes the test_future Brier score by "
            f"{i['brier_diff_vs_lgbm_tuned']:+.4f} [{i['brier_diff_ci_low']:+.4f}, {i['brier_diff_ci_high']:+.4f}] "
            f"({_src('calibration')}). It makes the probabilities more trustworthy, not the ranking.")
    lines.append(
        f"* **Retraining on recent weeks only {'did not help' if slid['roc_auc'] < exp['roc_auc'] else 'helped'} "
        f"at this horizon:** {_f4(slid['roc_auc'])} vs "
        f"{_f4(exp['roc_auc'])} test_future AUC ({_src('drift')}). The real remedy, retraining with labelled "
        "files from the deployment period, cannot be tested without using test_future for training.")
    lines.append("* **A stress set of hard files** (the challenge set) is what showed the largest real-world "
                 "weakness; neither random nor temporal splits of ordinary data show it.")
    return "\n".join(lines) + "\n"


def _verdict(t: dict[str, Any]) -> str:
    tuned = _tuned(t)
    r = _one(t["random_vs_future"], model=tuned)
    pf = _prec(t, tuned, OOF_THRESHOLD, "test_future")
    ch = _challenge(t, tuned, OOF_THRESHOLD)
    tail = (f", and it misses {_pct(1 - ch['detection_rate'], 0)} of the malware that antivirus engines missed "
            "at first" if ch is not None else "")
    share = (f"only {_pct(pf['precision'])}, fewer than half," if pf["precision"] < 0.5
             else f"{_pct(pf['precision'])}")
    auto = t["random_vs_future"]
    auto = auto[auto["kind"] == "automl"]
    dr = t["leakage"][(t["leakage"]["variant"] == "leak_detection_ratio") & (t["leakage"]["model"] != "lgbm_tuned")]
    all_fell = bool(dr["relied_on_leak"].astype(bool).all()) and int(dr["n_lines_mentioning_leak"].fillna(0).sum()) == 0
    return "\n".join([
        "## 7. Verdict", "",
        f"For a tree model on EMBER2024 Win64, a random split overstates the ranking quality "
        f"{'only a little' if r['roc_auc_gap'] < SMALL_AUC_GAP else 'clearly'} "
        f"(AUC {_f4(r['test_random_roc_auc'])} vs {_f4(r['test_future_roc_auc'])} on later weeks). The bigger "
        f"overstatement comes from reading a 50/50 test set as if it were real traffic: at "
        f"{HEADLINE_PREVALENCE:.0%} prevalence {share} of the tuned model's alarms on "
        f"test_future are real{tail}. The AutoML tools' Step 3 models reach test_future AUC "
        f"{_f4(auto['test_future_roc_auc'].min())} to {_f4(auto['test_future_roc_auc'].max())} (tuned LightGBM "
        f"{_f4(r['test_future_roc_auc'])}), and they lose no less to drift: {_automl_drift_verdict(t)}. "
        + ("They do not protect against leakage: given a post-hoc label column they all use it, report a "
           "near-perfect validation score, and say nothing. " if all_fell else
           "Their response to the label leak is mixed (section 4). ")
        + "The protections that worked here were human ones: asking where each column comes from, testing "
        "on later data, and checking the operating point at a realistic base rate.", ""])


def critical_eval_markdown(t: dict[str, Any]) -> str:
    return "\n".join([_header(t), _short_answer(t), _ladder(t), _all_models(t), _why_small(t), _leakage(t),
                      _drift(t), _bias_explain(t), _what_protects(t), _verdict(t)])


# =============================================================================
# reports/limitations.md
# =============================================================================
def _week_overlap(t: dict[str, Any]) -> str:
    """Whether the first and last future weeks' AUC CIs overlap (expanding model, drift.csv)."""
    weekly = t["drift"]
    weekly = weekly[(weekly["model"] == "expanding") & weekly["week"].str.isdigit()]
    weekly = weekly.assign(week_n=weekly["week"].astype(int)).sort_values("week_n")
    first, last = weekly.iloc[0], weekly.iloc[-1]
    overlap = first["roc_auc_ci_low"] <= last["roc_auc_ci_high"] and last["roc_auc_ci_low"] <= first["roc_auc_ci_high"]
    return (f"comes with week-level CIs that {'overlap' if overlap else 'do not overlap'} between the first "
            f"(week {first['week_n']}) and the last (week {last['week_n']}) future week")


def limitations_markdown(t: dict[str, Any]) -> str:
    dc, si, facts = t["data_check"], t["splits_info"], t["facts"]
    tuned = _tuned(t)
    cache = dc["cache_info"]["splits"]
    groups = dc["cache"]["feature_groups"]
    det = dc["detection_ratio"]
    slices = t["bias_slices"]

    def slice_rate(test: str, family: str, name: str, col: str) -> str:
        rows = slices[(slices["test_set"] == test) & (slices["threshold_name"] == OOF_THRESHOLD)
                      & (slices["slice_family"] == family) & (slices["slice"] == name)]
        return _pct(rows.iloc[0][col], 2) if len(rows) else "n/a"

    pf = _prec(t, tuned, OOF_THRESHOLD, "test_future")
    ch = _challenge(t, tuned, OOF_THRESHOLD)
    leak = t["leakage"]
    auto_leak = leak[leak["model"] != "lgbm_tuned"]
    over = auto_leak["wall_s"] - auto_leak["budget_s"]
    h2o_clean = leak[(leak["model"] == "h2o") & (leak["variant"] == "clean")]
    step3 = t["automl_self_estimate"]
    h2o3 = step3[(step3["tool"] == "h2o") & (step3["run"] == "step3")]
    vm = t["vm"]
    n_train, n_rand, n_fut = (si["splits"][s]["n"] for s in ("train", "test_random", "test_future"))
    tr_dates, fu_dates = dc["dates"]["train"], dc["dates"]["test"]
    lines = [
        "# Limitations and threats to validity", "",
        "Generated by `scripts/run_critical_eval.py` (`src/critical_eval_writeup.py`) from the Step 0 data "
        f"checks (`reports/data_check{RUN_SUFFIX}.json`, `data/splits/splits_info{RUN_SUFFIX}.json`), "
        "`config.py` and the Step 4 tables; no number is typed by hand. The findings these limits apply to "
        "are in [`critical_eval.md`](critical_eval.md).", "",
        "## Data and features", "",
        f"* **Static features only.** Every model sees the {dc['cache']['feature_dim']:,} thrember features "
        f"in {len(groups)} groups ({', '.join(g['name'] for g in groups)}), computed from the file's bytes "
        "and PE structure without running it. Behaviour that only shows at run time (unpacking, network "
        "traffic, injected code) is invisible, and so is anything outside a Win64 PE file.",
        f"* **Packed and obfuscated files.** Packer tags in EMBER2024 exist only on malicious files, so the "
        "false-positive rate on packed *benign* software cannot be measured here; the model may simply have "
        "learned 'packed means malicious'. The feature-based proxy (whole-file entropy > 7) shows the cost: "
        f"benign FPR on test_future is {slice_rate('test_future', 'entropy_proxy', 'entropy > 7', 'fpr')} for "
        f"high-entropy files against {slice_rate('test_future', 'entropy_proxy', 'entropy <= 7', 'fpr')} for "
        f"the rest (`bias_slices{RUN_SUFFIX}.csv`, 1%-FPR train threshold).",
        f"* **Dataset age and recency.** Training files are from {tr_dates['date_min'][:10]} to "
        f"{tr_dates['date_max'][:10]}, test_future files from {fu_dates['date_min'][:10]} to "
        f"{fu_dates['date_max'][:10]}. The model has never seen anything newer, and the drift measured over "
        f"{fu_dates['n_weeks']} weeks says little about drift over the years between the data and any use of "
        "this model today.",
        f"* **Drift horizon.** All temporal results cover {fu_dates['n_weeks']} future weeks. The weekly AUC "
        f"slope ({facts['drift_slopes_per_week']['roc_auc']['expanding']:+.5f} per week, `drift{RUN_SUFFIX}.csv`) "
        f"{_week_overlap(t)}; a longer horizon could show more decay.",
        f"* **50/50 balance vs real prevalence.** The splits were built with {_pct(si['splits']['train']['pos_rate'], 2)} / "
        f"{_pct(si['splits']['test_random']['pos_rate'], 2)} / {_pct(si['splits']['test_future']['pos_rate'], 2)} "
        "malware in train / test_random / test_future. Real traffic is mostly benign; at "
        f"{HEADLINE_PREVALENCE:.0%} prevalence the tuned model's precision on test_future would be "
        f"{_pct(pf['precision'])} (`precision_both_sets{RUN_SUFFIX}.csv`). The real prevalence of the place of "
        "deployment is unknown, so 1%, 5% and 10% are scenarios, not measurements.",
        f"* **Subsampling.** After deduplication the official data has {cache['train']['rows']:,} training-period "
        f"and {cache['test']['rows']:,} test-period files. We used {n_train:,} for train and {n_rand:,} for "
        f"test_random (from the training weeks) and {n_fut:,} for test_future, stratified by label x week; "
        f"Step 2 tuned on a {config.TUNE_SUBSET_N:,}-row subset with {config.CV_FOLDS} folds. Larger samples "
        "would narrow the CIs, and could change close rankings between tuners and tools.",
        f"* **Label noise from AV-derived labels.** A label is a verdict of antivirus engines: in our cache no "
        f"benign file has more than {det['train']['max_detections_benign']:.0f} detections and no malicious "
        f"file fewer than {det['train']['min_detections_malicious']:.0f} "
        f"({det['train']['benign_with_any_detection']:,} benign training-period files have at least one), so "
        "files in the ambiguous middle are not in the data at all, which makes the task easier than real "
        "triage. Labels can also be wrong in ways the engines share, and they change as engines catch up: "
        "the challenge set consists of files the engines first missed.",
        "* **Near-duplicates.** Deduplication removes exact copies only (same sha256); variants of one malware "
        "family remain on both sides of the random split, which is part of why test_random is easy "
        f"(per-family slices in `bias_slices{RUN_SUFFIX}.csv`).",
        f"* **The duplicate rows in the raw data.** Each weekly file lists every sample twice "
        f"({cache['train']['duplicates_in_file']:,} and {cache['test']['duplicates_in_file']:,} in-file "
        f"duplicates in the training and test periods, plus {cache['test']['duplicates_across_files']} across "
        "test files). We deduplicate globally by sha256 before any split; without that, a random split would "
        "put copies of the same file into train and test and inflate test_random further. Anyone comparing "
        "with other EMBER2024 results should check whether they did the same.", "",
        "## Threats from the evaluation itself", "",
        "* **Adversarial evasion was not tested.** Static features can be changed without changing behaviour "
        "(appended bytes, added imports or sections, a valid signature). The models lean on signature columns "
        f"(`explainability{RUN_SUFFIX}.md`); detection of signed malware on test_future is "
        f"{slice_rate('test_future', 'signed', 'signed', 'tpr')} at the train 1%-FPR threshold, against "
        f"{slice_rate('test_future', 'signed', 'unsigned', 'tpr')} for unsigned malware"
        + (f", and the challenge set is detected at {_pct(ch['detection_rate'])}" if ch is not None else "")
        + ". An attacker who knows this has an easy target.",
        f"* **One seed ({config.SEED}), one split per scheme.** The CIs cover test-set sampling only, not "
        "training randomness or another draw of the splits; time-budgeted AutoML searches are not "
        "repeatable, so deltas of about 1e-4 AUC between AutoML runs are within run-to-run variation.",
        "* **Many slices, no multiple-comparison correction.** Slice CIs are per slice; with dozens of slices "
        "some differences will look significant by chance. A test_random week has only about 385 benign "
        "files, so a single week's FPR there is very noisy.",
        "* **Thresholds and calibrators come from smaller models.** They are fitted on train out-of-fold "
        "scores of fold models trained on two thirds of train, while the deployed model is trained on all "
        f"of it (stated, not corrected; `threshold{RUN_SUFFIX}.csv` shows the effect at high cost ratios, "
        "where the test_future FPR is above the out-of-fold FPR).",
        "* **The forward CV confounds recency with training-set size:** later folds are both closer in time "
        f"and trained on more weeks. A size-matched control exists only for drift (`drift{RUN_SUFFIX}.csv`).",
        "* **The AutoGluon leakage predictors were deleted** after scoring (about 2 GB each, little disk on "
        "the VM); their leaderboards, importances and test scores are saved, but those rows can be "
        "re-scored only by re-running them.",
        f"* **The challenge set is small and malicious only:** {int(ch['n']) if ch is not None else 'n/a'} Win64 "
        "files, so it measures misses, not false alarms" + "".join(
            f"; {_name(m)} was not scored on it ({why})"
            for m, why in facts.get("challenge_scoring", {}).get("not_scored", {}).items()) + ".", "",
        "## Compute, hardware and budgets", "",
        f"* **CPU only, and two machines.** Steps 2-3 ran on {LAPTOP} with 16 threads; Step 4 on a cloud VM "
        f"({vm['cpu']}, {vm['cpu_count']} logical CPUs, {vm['ram_gb']:.1f} GB RAM, {vm['os']}). Every "
        "comparison inside a step ran on one machine with the same settings, but times are not comparable "
        "across steps, and an AutoML tool searches fewer configurations per second on the VM."
        + (f" The challenge-set scoring alone was re-run on a {sm['cpu_count']}-CPU {sm['os'].split('-')[0]} "
           "machine (the laptop), which holds every saved Step 2-3 model including AutoGluon's; scoring "
           "fits nothing, so the hardware does not change its result."
           if (sm := facts.get("challenge_scoring", {}).get("machine")) and sm.get("cpu_count") != vm["cpu_count"]
           else ""),
        "* **Memory forced runner changes on the VM.** The runner settings were the same for every leakage "
        "variant: a lean FLAML preprocessing and a smaller H2O heap without XGBoost. AutoGluon itself skipped "
        "memory-heavy models (only in the detection_ratio run; in the clean and date runs its first LightGBM used the whole "
        "budget), so on the VM it was in effect a single LightGBM model (per-run counts in `leakage.md`).",
        f"* **The {config.LEAKAGE_BUDGET_S} s leakage budget** is a stated deviation from the "
        f"{config.AUTOML_BUDGET_S} s of Step 3, to keep nine AutoML runs within a few hours. The clean "
        "reruns used the same budget and hardware, so each with/without-leak pair is equal-budget (not "
        "equal-compute: see below), but the clean scores are not "
        "the Step 3 scores"
        + (f" (H2O: test_future AUC {_f4(h2o_clean.iloc[0]['test_future_roc_auc'])} at "
           f"{config.LEAKAGE_BUDGET_S} s on the VM vs {_f4(h2o3.iloc[0]['test_future_roc_auc'])} at "
           f"{config.AUTOML_BUDGET_S} s on the laptop)" if len(h2o_clean) and len(h2o3) else "")
        + f". Budgets are not hard limits: the AutoML runs ended {over.min():+.0f} s to {over.max():+.0f} s "
        f"from their budget (wall minus budget, `leakage{RUN_SUFFIX}.csv`): the budget caps the search, and "
        "FLAML's final refit and H2O's data import, ensemble and scoring come on top, so the effective compute "
        "of one tool's variants differed by minutes (`leakage.md`, *Run times*).",
        "* **Default AutoML settings.** We tested what a user gets by default (random-split validation, "
        "default presets); other settings, such as time-ordered validation, could behave differently.", ""]
    return "\n".join(lines)


def write_reports() -> list[Path]:
    t = load_inputs()
    out = []
    for stem, render in (("critical_eval", critical_eval_markdown), ("limitations", limitations_markdown)):
        path = REPORTS / f"{stem}{RUN_SUFFIX}.md"
        path.write_text(render(t), encoding="utf-8", newline="\n")
        out.append(path)
    return out
