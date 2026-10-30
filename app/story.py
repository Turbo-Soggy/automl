"""The presentation sections of the demo page (app/streamlit_app.py), for a non-specialist audience.

One function per section, in the order of the talk: the question, the data, the contenders, how
they were compared, the results, the full scorecard of metrics, the winner and why, what we
learned, and what to keep in mind. The live demo (section 8) stays in streamlit_app.py because it
calls the API.

Every number shown is read at run time from reports/tables/, reports/data_check.json,
data/splits/splits_info.json, models/final/ or config.py; none is typed here. A missing file
shows a one-line note instead of breaking the page (a slim deployment may ship only app/ and
models/).
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

import config

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPO_URL = "https://github.com/Turbo-Soggy/automl"
DATASET_URL = "https://huggingface.co/datasets/joyce8/EMBER2024"

# Plain-language name and one-line meaning of each thrember feature family.
FAMILY_TEXT: dict[str, tuple[str, str]] = {
    "general": ("Basic file facts", "size, and how many functions it imports and exports"),
    "histogram": ("Byte make-up", "how often each byte value appears in the file"),
    "byteentropy": ("Randomness of the content", "packed or encrypted code looks random"),
    "strings": ("Readable text inside", "web addresses, file paths, registry keys and other text"),
    "header": ("File header", "what kind of program it says it is and how it was built"),
    "section": ("Section layout", "how the code and data blocks are arranged and sized"),
    "imports": ("Windows functions it uses", "for example networking, registry or process access"),
    "exports": ("Functions it offers", "what other programs can call in it"),
    "datadirectories": ("Internal tables", "sizes of the file's resource, import and signature tables"),
    "richheader": ("Build-tool fingerprint", "traces of the compiler and linker that built it"),
    "authenticode": ("Digital signature", "whether and how the file is signed by a publisher"),
    "pefilewarnings": ("Structural oddities", "malformed or unusual parts of the file format"),
}

# Model ids used in the tables -> (name on the page, team).
MODELS: dict[str, tuple[str, str]] = {
    "optuna": ("Tuned LightGBM (Optuna search)", "Tuned by hand"),
    "grid": ("LightGBM, grid search", "Tuned by hand"),
    "random": ("LightGBM, random search", "Tuned by hand"),
    "autogluon": ("AutoGluon", "AutoML"),
    "flaml": ("FLAML", "AutoML"),
    "h2o": ("H2O AutoML", "AutoML"),
    "lgbm_default": ("LightGBM, default settings", "Baseline"),
    "logreg": ("Logistic regression", "Baseline"),
}
TEAM_COLOURS: dict[str, str] = {"Tuned by hand": "#2563eb", "AutoML": "#ea580c", "Baseline": "#94a3b8"}
TUNED = "lgbm_tuned"   # the selected hand-tuned model (Optuna) in the AutoML comparison tables


# --- Reading the saved results ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame | None:
    """reports/tables/<name>.csv, or None when it is not shipped."""
    path = TABLES / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def json_file(rel: str) -> dict[str, Any] | None:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def missing(what: str) -> None:
    st.caption(f"({what} is not available in this copy of the project.)")


def pairwise(metric: str, a: str, b: str = TUNED) -> pd.Series | None:
    """test_future paired-bootstrap difference a - b for one metric (automl_pairwise.csv)."""
    pw = table("automl_pairwise")
    if pw is None:
        return None
    rows = pw[(pw.test_set == "test_future") & (pw.metric == metric) & (pw.method_a == a) & (pw.method_b == b)]
    return rows.iloc[0] if len(rows) else None


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def pts(x: float) -> str:
    """A difference of two shares, in percentage points."""
    return f"{100 * x:+.2f} points"


# --- Small layout helpers ---------------------------------------------------------------------

def step(n: int, title: str) -> None:
    st.markdown(f'<div class="step"><div class="n">{n}</div><div class="t">{html.escape(title)}</div></div>',
                unsafe_allow_html=True)


def cards(items: list[tuple[str, str]]) -> None:
    """A row of big-number cards: (big text, label)."""
    st.markdown('<div class="stats">' + "".join(
        f'<div class="stat"><div class="big">{html.escape(b)}</div><div class="lbl">{html.escape(t)}</div></div>'
        for b, t in items) + "</div>", unsafe_allow_html=True)


def finding(title: str, body: str) -> None:
    """One boxed finding; body is trusted HTML built in this module."""
    st.markdown(f'<div class="finding"><div class="ft">{html.escape(title)}</div><div>{body}</div></div>',
                unsafe_allow_html=True)


# --- Section 1: the question ------------------------------------------------------------------

def question() -> None:
    st.markdown(
        '<div class="lead">Every day new Windows programs appear, and some are malware. Can a model tell them '
        "apart <b>by reading the file, without running it</b>? And does an <b>AutoML</b> tool, which picks and "
        "tunes models by itself, do better than a data scientist tuning a model by hand?</div>",
        unsafe_allow_html=True)
    st.markdown("We also ask an awkward question most demos skip: **is the usual way of testing a model "
                "too optimistic** when tomorrow's malware is not like yesterday's?")


# --- Section 2: the data ----------------------------------------------------------------------

def data_section(schema: dict[str, Any]) -> None:
    st.markdown(f"**EMBER2024** is a public research dataset of Windows programs, labelled as malware or safe "
                f"(Joyce et al., KDD 2025). It is on Hugging Face as [`joyce8/EMBER2024`]({DATASET_URL}), under "
                "the Apache-2.0 license. We use its **64-bit Windows** part. The dataset contains measurements "
                "of each file, not the files themselves, so nothing here can be run.")
    balance, info = table("class_balance"), json_file("data/splits/splits_info.json")
    if balance is None or info is None:
        missing("The split summary")
        return
    n = balance.set_index("split")
    share = n["malicious_share"].mean()
    cards([(f"{int(n.loc['train', 'n']):,}", "programs to learn from"),
           (f"{int(n.loc['test_random', 'n']):,}", "for the random test"),
           (f"{int(n.loc['test_future', 'n']):,}", "for the future test")])
    st.caption(f"About {pct(share, 0)} of the files in every group are malware, by design. Duplicate files "
               "were removed first.")
    split = info["splits"]
    future = split["test_future"]
    st.markdown(
        '<div class="timeline">'
        f'<div class="seg past" style="flex:{split["train"]["n_weeks"]}"><b>{split["train"]["n_weeks"]} weeks</b> '
        f'{split["train"]["date_min"][:10]} to {split["train"]["date_max"][:10]}<br>learning files and the random '
        "test are drawn from here</div>"
        f'<div class="seg future" style="flex:{future["n_weeks"]}"><b>{future["n_weeks"]} later weeks</b> '
        f'{future["date_min"][:10]} to {future["date_max"][:10]}<br>the future test</div></div>',
        unsafe_allow_html=True)
    groups = [g["name"] for g in schema["feature_groups"]]
    st.markdown(f"**What the model sees.** Each program is described by **{schema['feature_dim']:,} numbers**, "
                f"in {len(groups)} groups:")
    st.markdown('<div class="chips">' + "".join(
        f'<span class="chip" title="{html.escape(FAMILY_TEXT.get(g, (g, ""))[1])}">'
        f'{html.escape(FAMILY_TEXT.get(g, (g, ""))[0])}</span>' for g in groups) + "</div>", unsafe_allow_html=True)


# --- Section 3: the contenders ----------------------------------------------------------------

def contenders() -> None:
    minutes = config.AUTOML_BUDGET_S / 60
    st.markdown(
        '<div class="teams">'
        '<div class="team" style="border-color:#2563eb"><div class="tt">Team 1: tuned by hand</div>'
        "<p><b>LightGBM</b>, a fast model made of many small decision trees, the usual choice for this dataset.</p>"
        f"<p>We tried <b>{config.TUNE_TRIALS} settings</b> with each of three search strategies:</p><ul>"
        "<li><b>Grid</b>: a fixed checklist of combinations</li>"
        "<li><b>Random</b>: random combinations</li>"
        "<li><b>Optuna</b>: learns from earlier tries where to look next</li></ul>"
        "<p class='small'>Two reference points: LightGBM with its default settings, and "
        "<b>logistic regression</b>, a simple straight-line model.</p></div>"
        '<div class="team" style="border-color:#ea580c"><div class="tt">Team 2: AutoML</div>'
        "<p>Tools that choose the model type and its settings on their own. Each got the same "
        f"<b>{minutes:g}-minute</b> time budget:</p><ul>"
        "<li><b>FLAML</b> (Microsoft): a fast, cost-aware search</li>"
        "<li><b>AutoGluon</b> (Amazon): trains many models and combines the best</li>"
        "<li><b>H2O AutoML</b>: trains many models and stacks them</li></ul></div>"
        "</div>", unsafe_allow_html=True)


# --- Section 4: how they were compared --------------------------------------------------------

def fairness() -> None:
    info = json_file("data/splits/splits_info.json")
    subset = f"{info['tune_subset']['n']:,}" if info else "a subset of"
    train = f"{info['splits']['train']['n']:,}" if info else "the same"
    st.markdown(
        f"- **Same learning files for everyone** ({train} programs). Hand-tuning searched on {subset} of them to "
        "save time, then retrained its best settings on all of them.\n"
        f"- **Same random seed ({config.SEED})** everywhere, so the data splits and the hand-tuning repeat "
        "exactly (AutoML searches under a time limit can still vary a little from run to run).\n"
        f"- **No peeking.** Hand-tuning chose its settings by {config.CV_FOLDS}-fold cross-validation on the "
        "learning files; the AutoML tools used their own internal checks. The test files were only used for the "
        "final scores.\n"
        "- **Two exams.** The *random test* comes from the same weeks as the learning files: the usual, "
        "easy setup. The *future test* comes from later weeks: closer to real use.")
    st.markdown(
        '<div class="metric">The score we lead with: <b>share of malware caught when only 1 in 100 safe '
        "programs may be flagged</b>. A security team cannot handle many false alarms, so this matters more than "
        "plain accuracy.</div>", unsafe_allow_html=True)
    st.caption(f"Also reported: AUC (0.5 = coin flip, 1.0 = perfect ranking) and wall-clock time. Error bars are "
               f"95% confidence intervals from resampling the test files {config.N_BOOTSTRAP:,} times.")


# --- Section 5: the results -------------------------------------------------------------------

def results() -> None:
    rvf = table("random_vs_future")
    if rvf is None:
        missing("The results table")
        return
    df = rvf[rvf.model.isin(MODELS)].copy()
    df["name"] = df.model.map(lambda m: MODELS[m][0])
    df["team"] = df.model.map(lambda m: MODELS[m][1])
    for col in ("test_future_tpr_at_1pct_fpr", "test_future_tpr_at_1pct_fpr_ci_low",
                "test_future_tpr_at_1pct_fpr_ci_high"):
        df[col.replace("test_future_tpr_at_1pct_fpr", "caught")] = 100 * df[col]
    shown = df[df.model != "logreg"]   # far below the rest: on the same axis it would squash the top models
    order = shown.sort_values("caught", ascending=False)["name"].tolist()
    y = alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13))
    colour = alt.Color("team:N", title=None, legend=alt.Legend(orient="top", labelFontSize=13),
                       scale=alt.Scale(domain=list(TEAM_COLOURS), range=list(TEAM_COLOURS.values())))
    x_title = "Malware caught on the future test, at 1 false alarm per 100 safe programs (%)"
    x = alt.X("caught:Q", title=x_title, scale=alt.Scale(zero=False))
    base = alt.Chart(shown)
    ci = alt.X("caught_ci_low:Q", title=x_title, scale=alt.Scale(zero=False))
    chart = (base.mark_rule(strokeWidth=2).encode(x=ci, x2="caught_ci_high:Q", y=y, color=colour)
             + base.mark_circle(size=170, opacity=1).encode(
                 x=x, y=y, color=colour,
                 tooltip=["name", alt.Tooltip("caught:Q", format=".2f"), alt.Tooltip("test_future_roc_auc:Q", format=".4f")]))
    st.altair_chart(chart.properties(height=300), width="stretch")
    st.caption("Dots are the scores, lines the 95% confidence intervals. The axis does not start at 0, so small "
               "differences look large; logistic regression is left off because it is far below (see below). "
               "Source: reports/tables/random_vs_future.csv.")
    result_sentences(df)


def result_sentences(df: pd.DataFrame) -> None:
    """The plain-language reading of the chart, from the paired comparisons against tuned LightGBM."""
    by = df.set_index("model")
    tuned = by.loc["optuna"]
    lines = [f"**Tuned LightGBM** caught **{tuned['caught']:.1f}%** of future malware."]
    for tool in ("autogluon", "flaml", "h2o"):
        d = pairwise("tpr_at_1pct_fpr", tool)
        if d is None:
            continue
        verdict = ("**a tie**: the difference is smaller than the uncertainty" if not d.significant_95
                   else ("**ahead**" if d.diff_a_minus_b > 0 else "**behind**"))
        lines.append(f"**{MODELS[tool][0]}**: {by.loc[tool, 'caught']:.1f}%, {verdict} "
                     f"({pts(d.diff_a_minus_b)}, 95% CI {100 * d.ci_low:+.2f} to {100 * d.ci_high:+.2f}).")
    hand = by.loc[["grid", "random", "optuna"], "caught"]
    lines.append(f"The three hand-tuning strategies ended within **{hand.max() - hand.min():.2f} points** of each "
                 "other: how you search mattered much less than searching at all "
                 f"(default settings: {by.loc['lgbm_default', 'caught']:.1f}%).")
    lines.append(f"The straight-line model is far behind: **{by.loc['logreg', 'caught']:.1f}%**.")
    st.markdown("\n".join(f"- {s}" for s in lines))


# --- Section 6: the full scorecard ------------------------------------------------------------

# What each metric means, in plain words, for the glossary under the tables.
METRIC_TEXT: list[tuple[str, str]] = [
    ("ROC-AUC", "how well the scores put malware above safe files, over every possible flag line "
                "(0.5 = coin flip, 1.0 = perfect)"),
    ("Caught at 1% false alarms", "TPR at 1% FPR, our headline score: the share of malware caught when the flag "
                                  "line lets only 1 in 100 safe files be flagged"),
    ("Accuracy", "the share of all files given the right label"),
    ("Precision", "of the files flagged as malware, the share that really are malware. It depends on how common "
                  "malware is: our tests are half malware, so it is high here, and much lower when malware is "
                  "rare (section 9)"),
    ("Recall", "of the malware files, the share that were flagged (also called detection rate or TPR)"),
    ("F1", "one number that balances precision and recall (their harmonic mean)"),
    ("False alarms", "FPR: the share of safe files wrongly flagged"),
    ("Brier score", "the average squared gap between the malware probability and the truth; lower is better, "
                    "and it rewards honest probabilities"),
]


def with_ci(row: pd.Series, col: str, as_pct: bool) -> str:
    """'value (low–high)' for a metric and its 95% confidence interval."""
    fmt = (lambda x: f"{100 * x:.2f}%") if as_pct else (lambda x: f"{x:.4f}")
    return f"{fmt(row[col])} ({fmt(row[col + '_ci_low'])}–{fmt(row[col + '_ci_high'])})"


def accuracies(test_set: str) -> dict[str, float]:
    """Accuracy at the 0.5 cut-off per model id: hand-tuned and baselines from tuning_summary.csv, AutoML from
    automl_vs_manual.csv (random_vs_future.csv does not carry accuracy)."""
    out: dict[str, float] = {}
    ts, avm = table("tuning_summary"), table("automl_vs_manual")
    if ts is not None:
        out.update(ts.set_index("method")[f"{test_set}_accuracy"].to_dict())
    if avm is not None:
        out.update(avm[avm.kind == "automl"].set_index("method")[f"{test_set}_accuracy"].to_dict())
    return out


def metric_table(rvf: pd.DataFrame, prec: pd.DataFrame, test_set: str, order: list[str]) -> pd.DataFrame:
    """Every standard metric for every model on one test set, as display strings."""
    r = rvf.set_index("model")
    at_half = prec[(prec.test_set == test_set) & (prec.threshold_name == "0.5")].drop_duplicates("model")
    precision = at_half.set_index("model")["precision_at_test_mix"]   # at the test sets' own ~50/50 mix
    recall = at_half.set_index("model")["tpr"]
    acc = accuracies(test_set)
    rows = []
    for m in order:
        row = r.loc[m]
        rows.append({
            "Model": MODELS[m][0],
            "ROC-AUC": with_ci(row, f"{test_set}_roc_auc", as_pct=False),
            "Caught at 1% false alarms": with_ci(row, f"{test_set}_tpr_at_1pct_fpr", as_pct=True),
            "Accuracy": pct(acc[m], 2) if m in acc else "n/a",
            "Precision": pct(precision[m], 2) if m in precision else "n/a",
            "Recall": pct(recall[m], 2) if m in recall else "n/a",
            "F1": f"{row[f'{test_set}_f1']:.4f}",
            "False alarms": pct(row[f"{test_set}_fpr"], 2),
        })
    return pd.DataFrame(rows)


def scores_html(df: pd.DataFrame) -> str:
    """A compact HTML table of metric_table() output; a 'value (low–high)' score shows its range small below."""
    def cell(v: str) -> str:
        value, _, ci = v.partition(" (")
        return html.escape(value) + (f'<span class="ci">{html.escape(ci.rstrip(")"))}</span>' if ci else "")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    body = "".join(f"<tr><td>{html.escape(str(row[0]))}</td>"   # the model name is not split
                   + "".join(f"<td>{cell(str(v))}</td>" for v in row[1:]) + "</tr>"
                   for row in df.itertuples(index=False))
    return f'<div class="scores"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def confusion(tp: int, fn: int, fp: int, tn: int) -> pd.DataFrame:
    """A 2x2 confusion matrix with plain row and column names, as display strings."""
    return pd.DataFrame({"Flagged as malware": [f"{tp:,}", f"{fp:,}"], "Let through": [f"{fn:,}", f"{tn:,}"]},
                        index=["Really malware", "Really safe"])


def deployed_confusion(schema: dict[str, Any]) -> None:
    """The deployed model's confusion matrix on the future test, at its flag line and at the plain 0.5 cut-off."""
    th = table("threshold")
    if th is None:
        return
    rows = th[th.cost_ratio == schema["threshold"]["cost_ratio_fn_to_fp"]]
    if not len(rows):
        return
    t = rows.iloc[0]
    malware, safe = int(t.test_future_tp + t.test_future_fn), int(t.test_future_fp + t.test_future_tn)
    fn_half, fp_half = int(t["test_future_fn_at_0.5"]), int(t["test_future_fp_at_0.5"])
    st.markdown(f"#### The deployed model, file by file (future test: {malware:,} malware, {safe:,} safe)")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**At its flag line ({schema['threshold']['threshold']:.4f})**, as deployed")
        st.table(confusion(int(t.test_future_tp), int(t.test_future_fn), int(t.test_future_fp),
                           int(t.test_future_tn)))
    with right:
        st.markdown("**At the plain 0.5 cut-off**")
        st.table(confusion(malware - fn_half, fn_half, fp_half, safe - fp_half))
    st.caption(f"The low flag line trades {int(t.test_future_fp) - fp_half:,} extra false alarms for "
               f"{fn_half - int(t.test_future_fn):,} fewer missed malware files: a miss counts "
               f"{schema['threshold']['cost_ratio_fn_to_fp']:g} times a false alarm. The line was chosen on the "
               "learning files only. Source: reports/tables/threshold.csv.")


def plain_check(how: str) -> str:
    """An AutoML tool's validation note in the page's words: 'train' -> 'learning files', 2496 -> 2,496."""
    how = how.replace("of train", "of the learning files")
    return re.sub(r"\d{4,}", lambda m: f"{int(m.group()):,}", how)


def before_the_test() -> None:
    """The scores each model reached on the learning files alone, before any test file was used."""
    ts, own = table("tuning_summary"), table("automl_self_estimate")
    lines = []
    if ts is not None:
        s = ts.set_index("method")
        lines += [f"**{MODELS[m][0]}**: {s.loc[m, 'best_cv_auc']:.4f} ± {s.loc[m, 'cv_auc_std']:.4f}"
                  for m in ("optuna", "grid", "random", "lgbm_default", "logreg") if m in s.index]
    if own is not None:
        o = own[(own.run == "step3") & (own.variant == "clean")].set_index("tool")
        lines += [f"**{MODELS[m][0]}**: {o.loc[m, 'own_auc']:.4f}, on its {plain_check(o.loc[m, 'own_validation'])}"
                  for m in ("autogluon", "flaml", "h2o") if m in o.index]
    if not lines:
        return
    st.markdown("#### Before the test: validation on the learning files only")
    st.markdown(f"Hand-tuned models and baselines: {config.CV_FOLDS}-fold cross-validation AUC (mean ± spread "
                f"across folds) on the {config.TUNE_SUBSET_N:,}-file tuning subset. AutoML tools: the AUC on the "
                "part of the learning files each tool held out to check itself. None of these used a test file. "
                "The checks differ in size and data, so compare each number with that model's test score, not "
                "with each other.")
    st.markdown("\n".join(f"- {s}" for s in lines))


def calibration_line() -> None:
    """The Brier score of each model whose future-test probabilities were checked (calibration.csv)."""
    cal = table("calibration")
    if cal is None:
        return
    c = cal[cal.test_set == "test_future"].set_index("model")
    names = {TUNED: "optuna", "autogluon": "autogluon", "flaml": "flaml", "h2o": "h2o", "logreg": "logreg"}
    parts = [f"{MODELS[m][0]} {c.loc[k, 'brier']:.4f}" for k, m in names.items() if k in c.index]
    st.markdown("**How trustworthy the probabilities are** (Brier score on the future test, lower is better): "
                + "; ".join(parts) + ".")


def scorecard(schema: dict[str, Any]) -> None:
    """Section 6: every evaluation metric, for when someone asks for more than the headline score."""
    rvf, prec = table("random_vs_future"), table("precision_both_sets")
    if rvf is None or prec is None:
        missing("The full metrics table")
        return
    st.markdown("Section 5 led with one score. Here is every standard metric for every model, on both tests, for "
                "when someone asks. Accuracy, precision, recall, F1 and false alarms use the usual 0.5 cut-off "
                "on the malware probability. The small grey numbers under a score are its 95% confidence interval. "
                "The tests are half malware, so precision here is far higher than it would be where malware is rare "
                "(section 9).")
    order = (rvf[rvf.model.isin(MODELS)].sort_values("test_future_tpr_at_1pct_fpr", ascending=False)
             ["model"].tolist())
    for test_set, label in (("test_future", "Future test (later weeks: the one that matters)"),
                            ("test_random", "Random test (same weeks as the learning files)")):
        st.markdown(f"#### {label}")
        st.markdown(scores_html(metric_table(rvf, prec, test_set, order)), unsafe_allow_html=True)
    st.caption("Rows in the order of section 5's chart. Sources: reports/tables/random_vs_future.csv, "
               "precision_both_sets.csv, tuning_summary.csv and automl_vs_manual.csv.")
    calibration_line()
    deployed_confusion(schema)
    before_the_test()
    with st.expander("What each metric means"):
        st.markdown("\n".join(f"- **{name}**: {text}." for name, text in METRIC_TEXT))


# --- Section 7: the winner and why ------------------------------------------------------------

def plain_pick(note: str) -> str | None:
    """Plain words for an AutoML tool's final model, from its interpretability note; None if not a tree model."""
    if "xgb" in note:
        return "XGBoost, a boosted-tree model"
    if "all weight on LightGBM" in note:
        return "an ensemble that, in the end, put all its weight on one LightGBM model"
    if "weight only to GBM" in note:
        return "an ensemble that, in the end, used only one boosted-tree model (GBM)"
    return None


def winner(schema: dict[str, Any], model_file: Path) -> None:
    avm, rvf = table("automl_vs_manual"), table("random_vs_future")
    if avm is None or rvf is None:
        missing("The comparison table")
        return
    a = avm.set_index("method")
    tied = [MODELS[t][0] for t in ("autogluon", "flaml", "h2o")
            if (d1 := pairwise("tpr_at_1pct_fpr", t)) is not None and (d2 := pairwise("roc_auc", t)) is not None
            and not d1.significant_95 and not d2.significant_95]
    if tied:
        head = f"A tie at the top: tuned LightGBM and {', '.join(tied)}"
    else:
        head = "Tuned LightGBM came out on top"
    st.markdown(f'<div class="winner">{html.escape(head)}</div>', unsafe_allow_html=True)
    if "autogluon" in a.index:
        st.markdown(f"AutoGluon got there in **{a.loc['autogluon', 'wall_minutes']:.0f} minutes** with no human "
                    f"input. The hand-tuned winner took **{a.loc[TUNED, 'wall_minutes']:.0f} minutes** of search and "
                    f"training, and all three hand searches together "
                    f"**{a.loc[TUNED, 'all_manual_seconds'] / 60:.0f} minutes**.")

    st.markdown("#### Why these models do well")
    picks = {m: plain_pick(str(r["interpretability"])) for m, r in a[a.kind == "automl"].iterrows()}
    if picks and all(p is not None for p in picks.values()):
        st.markdown("**Every AutoML tool ended up with the same kind of model we tuned by hand: boosted decision "
                    "trees.** What each tool finally used:")
        st.markdown("\n".join(f"- **{MODELS.get(m, (m,))[0]}**: {p}" for m, p in picks.items()))
        st.caption("From the interpretability notes in reports/tables/automl_vs_manual.csv.")
    st.markdown("Trees ask simple yes/no questions about single measurements (*is the file signed? is the "
                "content very random?*) and combine thousands of them. They cope with measurements on wildly "
                f"different scales without any rescaling, which suits these {schema['feature_dim']:,} mixed numbers.")
    r = rvf.set_index("model")
    tree_gaps = r.loc[["optuna", "grid", "random", "flaml", "autogluon", "h2o"], "roc_auc_gap"]
    st.markdown(f"A straight-line model is not enough: logistic regression's AUC drops by "
                f"**{r.loc['logreg', 'roc_auc_gap']:.4f}** from the random to the future test, the tree models' by "
                f"at most **{tree_gaps.max():.4f}**.")

    st.markdown("#### Why we deploy tuned LightGBM")
    size_mb = model_file.stat().st_size / 2**20 if model_file.exists() else float("nan")
    st.markdown(
        ("With accuracy tied, we chose on everything else:\n" if tied else "Besides scoring best:\n") +
        "- Its settings were **chosen by cross-validation on the learning files**, without looking at any test "
        "score.\n"
        f"- It is **one file** ({schema['num_trees']} trees, {size_mb:.1f} MB) that needs no AutoML library to run.\n"
        "- It can **explain each decision** with SHAP, as the live demo shows next.")


# --- Section 9: what we learned ---------------------------------------------------------------

def learned() -> None:
    rvf = table("random_vs_future")
    if rvf is not None:
        r = rvf.set_index("model")
        finding("1. Random test vs future test: by AUC, the gap is small for trees",
                f"Tuned LightGBM's AUC is {r.loc['optuna', 'test_random_roc_auc']:.4f} on the random test and "
                f"{r.loc['optuna', 'test_future_roc_auc']:.4f} on the future test: a small gap for trees, a large "
                f"one ({r.loc['logreg', 'roc_auc_gap']:.4f}) for the straight-line model.")
    prev = table("precision_both_sets")
    if prev is not None:
        rows = prev[(prev.model == "optuna") & (prev.test_set == "test_future") & (prev.threshold_name == "deployment")]
        low = rows[rows.prevalence == rows.prevalence.min()]
        if len(low):
            p = low.iloc[0]
            finding("2. In the real world, most alarms would be false",
                    f"Our tests are half malware, so {pct(p.precision_at_test_mix, 0)} of the deployed model's alarms "
                    f"are real. If only {pct(p.prevalence, 0)} of files were malware, about "
                    f"<b>{pct(p.precision, 0)}</b> of its alarms would be: about "
                    f"{p.false_alarms_per_10k / p.true_alarms_per_10k:.0f} false alarms for every real catch. "
                    "That is why it is a triage aid, not an automatic blocker.")
    leak = table("leakage")
    if leak is not None:
        d = leak[leak.variant == "leak_detection_ratio"]
        tools = d[d.model != TUNED]
        ranks = d["leak_rank_shap"].dropna().astype(int)
        rank_text = "their most important input" if len(ranks) and ranks.max() == 1 else f"a top-{ranks.max()} input"
        kept = int((tools["leak_dropped_by_tool"] == False).sum())  # noqa: E712 (the column also holds NaN)
        silent = int((tools["n_lines_mentioning_leak"] == 0).sum())
        finding("3. No AutoML tool noticed a planted cheat",
                "We added a column that nearly gives away the answer: how many antivirus engines already flag the "
                f"file. {int(d['relied_on_leak'].sum())} of {len(d)} models made it {rank_text}. {kept} of "
                f"{len(tools)} AutoML tools kept it, and {silent} of {len(tools)} never mentioned it in their logs. "
                "Catching leaks is still a human's job.")
    ch = table("challenge")
    if ch is not None:
        row = ch[(ch.model == "optuna") & (ch.threshold_name == "deployment") & (ch.subset == "all Win64 challenge")]
        if len(row):
            c = row.iloc[0]
            finding("4. The hardest malware is still hard",
                    f"On {int(c.n)} malware files that antivirus engines first missed, the deployed model catches "
                    f"<b>{pct(c.detection_rate, 0)}</b>, against {pct(c.test_future_tpr_same_threshold, 1)} of "
                    "ordinary future malware.")


# --- Section 10: keep in mind and explore -----------------------------------------------------

def keep_in_mind() -> None:
    st.markdown('<ul class="keep">'
                "<li>It is a <b>triage aid</b> for analysts, not a final verdict for blocking files.</li>"
                "<li>It reads the file without running it, so malware built to look harmless can slip through.</li>"
                "<li>It only knows 64-bit Windows programs, and it is weaker on brand-new kinds of malware.</li>"
                "</ul>", unsafe_allow_html=True)
    st.markdown(f"**Explore further:** the full report `reports/report.pdf`, the model card `MODEL_CARD.md`, and "
                f"the code and every result at [{REPO_URL}]({REPO_URL}).")
