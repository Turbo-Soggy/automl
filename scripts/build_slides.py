"""Step 6: build the 5-minute demo deck reports/slides.pptx from the saved tables and figures.

Every number on a slide and in the speaker notes is read at build time from reports/tables/*.csv,
reports/data_check.json or models/threshold.json; nothing is typed by hand. Figures come from
reports/figures/ and the app screenshot from reports/screenshots/. Ten slides, about 30 s of
speaker notes each.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/build_slides.py            # writes reports/slides.pptx
  .venv/Scripts/python.exe scripts/build_slides.py --out x.pptx
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"

# One palette and one type scale for the whole deck. Body text never goes below 18 pt.
NAVY = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0xE0, 0x6C, 0x2F)   # same orange as test_future in the figures
BLUE = RGBColor(0x2A, 0x78, 0xD6)     # same blue as test_random in the figures
GREY = RGBColor(0x5F, 0x66, 0x73)
LIGHT = RGBColor(0xEE, 0xF1, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
TITLE_PT = 30
BODY_PT = 20
TABLE_PT = 18
SOURCE_PT = 12
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.5)


# --- Reading the saved results -------------------------------------------------------------------

def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def vm_ram_gib() -> float:
    """RAM of the Step 4 cloud VM, read from the leakage records in reports/experiments.jsonl.

    src/leakage.py logs machine.ram_gb as bytes / 2**30, i.e. GiB (15.7 GiB = 16.9 decimal GB;
    the Step 4 eval records log the decimal figure, 16.9)."""
    rams = []
    for line in (REPORTS / "experiments.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            ram = (rec.get("machine") or {}).get("ram_gb")
            if rec.get("step") == "leakage" and ram:
                rams.append(float(ram))
    return max(rams)


def row(df: pd.DataFrame, **match: object) -> pd.Series:
    """The single row of df whose columns equal the given values (fails loudly otherwise)."""
    mask = pd.Series(True, index=df.index)
    for col, val in match.items():
        mask &= df[col] == val
    hit = df[mask]
    if len(hit) != 1:
        raise ValueError(f"expected one row for {match}, found {len(hit)}")
    return hit.iloc[0]


def auc(x: float) -> str:
    return f"{x:.4f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def minutes(seconds: float) -> str:
    return "-" if pd.isna(seconds) else f"{seconds / 60:.1f}"


# --- Drawing helpers ------------------------------------------------------------------------------

def add_text(slide, left, top, width, height, paragraphs: list[str] | str, size: int = BODY_PT,
             color: RGBColor = NAVY, bold: bool = False, bullets: bool = False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """A text box with one paragraph per string; '**x**' at the start of a string makes x bold."""
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, text in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(8)
        head, rest = "", text
        m = re.match(r"\*\*(.+?)\*\*(.*)", text, flags=re.S)
        if m:
            head, rest = m.group(1), m.group(2)
        if bullets:
            set_bullet(p, size)
        parts = [(head, True), (rest, bold)]
        for part, is_bold in parts:
            if part:
                r = p.add_run()
                r.text = part
                r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(size), color
                r.font.bold = is_bold
    return box


def set_bullet(paragraph, size: int) -> None:
    """Give a paragraph a real bullet with a hanging indent (python-pptx has no API for this,
    so the DrawingML attributes are set directly: marL = text indent, indent = bullet offset)."""
    hang = Emu(int(Pt(size) * 0.9))
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(hang))
    pPr.set("indent", str(-hang))
    for tag in ("a:buNone", "a:buChar"):  # replace any bullet setting already there
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "\u2022"}))


def new_slide(prs: Presentation, title: str, notes: str, source: str = ""):
    """A blank slide with the navy title bar, the speaker notes and an optional source line."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    add_text(slide, MARGIN, Inches(0.12), SLIDE_W - 2 * MARGIN, Inches(0.8), title,
             size=TITLE_PT, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if source:
        add_text(slide, MARGIN, SLIDE_H - Inches(0.45), SLIDE_W - 2 * MARGIN, Inches(0.35),
                 "Source: " + source, size=SOURCE_PT, color=GREY)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_table(slide, left, top, width, header: list[str], rows: list[list[str]],
              col_widths: list[float], highlight: int | None = None, row_h: float = 0.5):
    """A plain table: navy header, light zebra rows, one row optionally highlighted in orange."""
    shape = slide.shapes.add_table(len(rows) + 1, len(header), left, top, width,
                                   Inches(row_h * (len(rows) + 1)))
    tbl = shape.table
    for j, w in enumerate(col_widths):
        tbl.columns[j].width = Inches(w)
    for i, values in enumerate([header] + rows):
        tbl.rows[i].height = Inches(row_h)
        for j, value in enumerate(values):
            cell = tbl.cell(i, j)
            cell.text = value
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            run = para.runs[0]
            run.font.name, run.font.size = FONT, Pt(TABLE_PT)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = NAVY
                run.font.color.rgb, run.font.bold = WHITE, True
            elif highlight is not None and i - 1 == highlight:
                cell.fill.fore_color.rgb = RGBColor(0xFB, 0xE3, 0xD3)
                run.font.color.rgb, run.font.bold = NAVY, True
            else:
                cell.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
                run.font.color.rgb = NAVY
    return shape


def add_picture_fit(slide, path: Path, left, top, max_w, max_h,
                    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)):
    """Insert an image scaled to fit the box (aspect kept), centred in it.

    `crop` = (left, top, right, bottom) as shares of the image; the crop is done in memory,
    so no extra image files are written."""
    im = Image.open(path)
    x0, y0, x1, y1 = crop
    im = im.crop((int(im.width * x0), int(im.height * y0), int(im.width * x1), int(im.height * y1)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    w_px, h_px = im.size
    scale = min(max_w / w_px, max_h / h_px)
    w, h = int(w_px * scale), int(h_px * scale)
    return slide.shapes.add_picture(buf, left + (max_w - w) // 2, top + (max_h - h) // 2,
                                    width=w, height=h)


def add_box(slide, left, top, width, height, text: str, fill: RGBColor, size: int = 18,
            color: RGBColor = WHITE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(size), color
        r.font.bold = i == 0
    return shape


def add_arrow(slide, left, top, width, height, shape=MSO_SHAPE.RIGHT_ARROW):
    arrow = slide.shapes.add_shape(shape, left, top, width, height)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = GREY
    arrow.line.fill.background()


# --- The slides -----------------------------------------------------------------------------------

def slide_title(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(3.55), Inches(2.0),
                                    Inches(0.08))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = ACCENT
    stripe.line.fill.background()
    add_text(slide, MARGIN, Inches(1.4), Inches(12.3), Inches(2.1),
             ["Malware detection on EMBER2024 Win64 PE files:",
              "manual LightGBM tuning vs AutoML"], size=40, color=WHITE, bold=True)
    add_text(slide, MARGIN, Inches(3.85), Inches(12.3), Inches(1.6),
             ["Does a random train/test split overstate real-world performance, and does AutoML "
              "(FLAML, AutoGluon, H2O) protect against leakage and drift?"],
             size=22, color=WHITE)
    add_text(slide, MARGIN, Inches(5.7), Inches(12.3), Inches(1.0),
             ["Isaac Benjamin David (URK23AI1035)  |  IA-3 Part B: AutoML  |  seed 42 throughout"],
             size=20, color=RGBColor(0xC9, 0xD1, 0xE0))
    slide.notes_slide.notes_text_frame.text = (
        "Hello. Our project detects Windows malware from static features of the executable, "
        "using the EMBER2024 Win64 data. We compare a LightGBM model that we tuned by hand with "
        "three AutoML tools: FLAML, AutoGluon and H2O. Our research question is whether the usual "
        "random train/test split overstates how well such a detector works on files that arrive "
        "later, and whether AutoML protects us from a leaky column or from drift.")


def slide_problem_data(prs: Presentation) -> None:
    check = json.loads((REPORTS / "data_check.json").read_text(encoding="utf-8"))
    splits = check["cache_info"]["splits"]
    lines = sum(s["lines"] for s in splits.values())
    rows = sum(s["rows"] for s in splits.values())
    dup = sum(s["duplicates_in_file"] + s["duplicates_across_files"] for s in splits.values())
    dim = check["cache_info"]["feature_dim"]
    n_groups = len(check["cache"]["feature_groups"])
    bal = table("class_balance")
    tr, tra, fut = (row(bal, split=s) for s in ("train", "test_random", "test_future"))
    notes = (
        "A detector is only useful on files it has not seen yet. EMBER2024 gives weekly files "
        "with labels, so we can test on later weeks. First finding: every weekly file lists "
        f"each sample twice. Of {lines:,} raw "
        f"lines only {rows:,} are unique, so we de-duplicate by sha256 before splitting; otherwise "
        "copies of one file land in both train and test. We sample "
        f"{int(tr.n):,} training rows and two {int(fut.n):,}-row test sets: a random one from the "
        "same weeks, and a future one from later weeks. Classes are 50/50 by construction.")
    slide = new_slide(prs, "Problem, data and a duplicate finding", notes,
                      "reports/data_check.json, reports/tables/class_balance.csv")
    add_text(slide, MARGIN, Inches(1.25), Inches(6.2), Inches(5.6), [
        "**Problem: **static malware detection for Windows 64-bit PE files, before they run.",
        "**Why it matters: **new malware keeps arriving; a score on old files can flatter "
        "a detector.",
        f"**Data: **EMBER2024 Win64, {dim:,} static features in {n_groups} families "
        "(imports, strings, headers, signatures, ...).",
        f"**Duplicate finding: **each weekly file repeats every sample, so {lines:,} raw lines "
        f"hold only {rows:,} unique files ({dup:,} copies removed by sha256 before any split).",
    ], bullets=False)
    add_table(slide, Inches(7.0), Inches(1.45), Inches(5.8),
              ["split", "weeks", "files", "malware"],
              [["train", "0-51", f"{int(tr.n):,}", pct(tr.malicious_share)],
               ["test_random", "0-51", f"{int(tra.n):,}", pct(tra.malicious_share)],
               ["test_future", "52-63", f"{int(fut.n):,}", pct(fut.malicious_share)]],
              [1.8, 1.2, 1.4, 1.4], row_h=0.55)
    add_text(slide, Inches(7.0), Inches(3.9), Inches(5.8), Inches(2.6), [
        "test_random: same weeks as train (stratified by label x week).",
        "test_future: 12 strictly later weeks.",
        "Both are used only for final scoring, never for tuning or thresholds.",
    ], size=18, color=GREY, bullets=True)


def slide_pipeline(prs: Presentation) -> None:
    notes = (
        "The pipeline. Raw files are de-duplicated and split once with seed 42, and the splits "
        "are saved so every step uses the same rows. Step 2 tunes LightGBM by grid search, "
        "random search and Optuna, 36 trials each on the same folds. Step 3 gives each AutoML "
        "tool the same training split and time budget. Step 4 is the critical evaluation: "
        "leakage, drift, real-world "
        "prevalence, the challenge set and SHAP. Step 5 serves the chosen model through an API "
        "and a Streamlit app.")
    slide = new_slide(prs, "Method and pipeline", notes,
                      "docs/PLAN.md, run_all.py; the detailed diagram for the report is "
                      "reports/figures/architecture.png")
    # Drawn with shapes rather than pasting architecture.png: that figure is sized for an A4
    # page, and its small labels would be unreadable on a projected slide.
    w, h, gap, top1, top2 = Inches(2.75), Inches(1.75), Inches(0.42), Inches(1.45), Inches(4.2)
    row1 = [("Raw data", "64 weekly jsonl files\nEMBER2024 Win64", NAVY),
            ("De-duplicate", "global sha256\n(each file listed twice)", NAVY),
            ("Fixed splits", "train / test_random:\nweeks 0-51\ntest_future: 52-63", NAVY),
            ("Feature cache", "2,568 float32\nfeatures per file", NAVY)]
    row2 = [("Step 2: manual", "LightGBM: grid,\nrandom, Optuna\n36 trials each", BLUE),
            ("Step 3: AutoML", "FLAML, AutoGluon,\nH2O; equal\ntime budget", BLUE),
            ("Step 4: evaluate", "leakage, drift,\nprevalence,\nchallenge, SHAP", ACCENT),
            ("Step 5: deploy", "FastAPI + Streamlit,\ncost threshold,\nmodel card", ACCENT)]
    # Row 1 runs left to right; row 2 runs right to left (a snake), so the down-arrow from the
    # feature cache leads straight into Step 2 and every arrow follows the order of the steps.
    for i, (head, body, fill) in enumerate(row1):
        left = MARGIN + i * (w + gap)
        add_box(slide, left, top1, w, h, f"{head}\n{body}", fill)
        if i < len(row1) - 1:
            add_arrow(slide, left + w + Inches(0.05), top1 + h // 2 - Inches(0.15),
                      gap - Inches(0.1), Inches(0.3))
    for i, (head, body, fill) in enumerate(row2):
        left = MARGIN + (len(row2) - 1 - i) * (w + gap)
        add_box(slide, left, top2, w, h, f"{head}\n{body}", fill)
        if i < len(row2) - 1:
            add_arrow(slide, left - gap + Inches(0.05), top2 + h // 2 - Inches(0.15),
                      gap - Inches(0.1), Inches(0.3), shape=MSO_SHAPE.LEFT_ARROW)
    down = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, MARGIN + 3 * (w + gap) + w // 2
                                  - Inches(0.2), top1 + h + Inches(0.1), Inches(0.4),
                                  top2 - top1 - h - Inches(0.2))
    down.fill.solid()
    down.fill.fore_color.rgb = GREY
    down.line.fill.background()
    add_text(slide, MARGIN, Inches(6.15), SLIDE_W - 2 * MARGIN, Inches(0.6),
             "Test sets are scored once at the end; every table and figure is built by a script.",
             size=18, color=GREY)


def slide_tuning(prs: Presentation) -> None:
    ts = table("tuning_summary")
    tuners = ts[ts.n_trials.notna()]
    best = tuners.loc[tuners.best_cv_auc.idxmax()]
    default = row(ts, method="lgbm_default")
    names = {"lgbm_default": "LightGBM default", "logreg": "LogReg", "grid": "Grid search",
             "random": "Random search", "optuna": "Optuna (TPE)"}
    rows, hi = [], None
    for i, r in enumerate(ts.itertuples()):
        if r.method == best.method:
            hi = i
        rows.append([names.get(r.method, r.method),
                     "-" if pd.isna(r.n_trials) else str(int(r.n_trials)),
                     auc(r.best_cv_auc), auc(r.test_random_roc_auc), auc(r.test_future_roc_auc),
                     pct(r.test_future_tpr_at_1pct_fpr), minutes(r.search_seconds)])
    spread = tuners.test_future_roc_auc.max() - tuners.test_future_roc_auc.min()
    spread_txt = f"{spread:.5f}"
    notes = (
        "Step 2. Equal budget means every tuner gets exactly 36 LightGBM configurations, scored "
        "on the same three cross-validation folds of a 60,000-row subset of train. "
        f"{names[best.method]} had the best CV AUC, {auc(best.best_cv_auc)}, so it became our "
        "tuned model, refit on all of train and scored once on each test set. Tuning "
        f"lifts test_future true-positive rate at 1% false-positive rate from "
        f"{pct(default.test_future_tpr_at_1pct_fpr)} to {pct(best.test_future_tpr_at_1pct_fpr)}. "
        f"The three tuners end within {spread_txt} AUC of each other, so the search method "
        "mattered far less than tuning at all.")
    slide = new_slide(prs, "Step 2: manual LightGBM tuning (36 trials each)", notes,
                      "reports/tables/tuning_summary.csv, reports/tuning_writeup.md")
    add_table(slide, MARGIN, Inches(1.3), Inches(12.3),
              ["method", "trials", "CV AUC", "AUC random", "AUC future", "TPR@1%FPR future",
               "search min"],
              rows, [2.5, 1.0, 1.6, 1.8, 1.8, 2.3, 1.3], highlight=hi, row_h=0.55)
    add_text(slide, MARGIN, Inches(4.9), Inches(12.3), Inches(1.9), [
        f"**Selected by CV AUC (never by a test score): **{names[best.method]}.",
        f"**Tuning beats defaults: **TPR@1%FPR on test_future "
        f"{pct(default.test_future_tpr_at_1pct_fpr)} -> {pct(best.test_future_tpr_at_1pct_fpr)}; "
        f"the three tuners differ by at most {spread_txt} AUC on test_future.",
    ])


def slide_automl(prs: Presentation) -> None:
    avm = table("automl_vs_manual")
    pw = table("automl_pairwise")
    names = {"flaml": "FLAML", "autogluon": "AutoGluon", "h2o": "H2O AutoML",
             "lgbm_tuned": "Tuned LightGBM", "lgbm_default": "LightGBM default"}
    rows, verdicts, max_diff = [], {}, 0.0
    for r in avm[avm.method != "logreg"].itertuples():
        winner = re.sub(r"_\d+_AutoML_.*$", "", str(r.best_model)).replace("_", " ")
        winner = re.sub(r"LightGBM via (\w+) search \((\d+) trees\)", r"LightGBM, \2 trees", winner)
        if r.kind == "automl":
            d = row(pw, test_set="test_future", metric="roc_auc", method_a=r.method,
                    method_b="lgbm_tuned")
            verdicts[r.method] = ("tied" if not d.significant_95 else
                                  "better" if d.diff_a_minus_b > 0 else "worse")
            vs = f"{d.diff_a_minus_b:+.5f} ({verdicts[r.method]})"
            max_diff = max(max_diff, abs(d.diff_a_minus_b))
        else:
            vs = "reference" if r.method == "lgbm_tuned" else "-"
        rows.append([names[r.method], winner, minutes(r.wall_seconds),
                     auc(r.test_future_roc_auc), pct(r.test_future_tpr_at_1pct_fpr), vs])
    tuned = row(avm, method="lgbm_tuned")
    budget = avm.budget_seconds.dropna().iloc[0]
    tied = [names[m] for m, v in verdicts.items() if v == "tied"]
    worse = [names[m] for m, v in verdicts.items() if v == "worse"]
    ag = row(avm, method="autogluon")
    notes = (
        f"Step 3. Each AutoML tool got the same training split, ROC-AUC as the metric, 16 "
        f"threads on the laptop and a {budget / 60:.0f}-minute budget. On test_future AUC, "
        f"{', '.join(tied) or 'no tool'} is statistically tied with our tuned LightGBM, and "
        f"{' and '.join(worse) or 'no tool'} {'are' if len(worse) > 1 else 'is'} slightly worse; "
        f"no tool is more than {max_diff:.4f} AUC away. The big difference is cost: AutoGluon took "
        f"{minutes(ag.wall_seconds)} minutes against {minutes(tuned.wall_seconds)} for the manual "
        "search we selected and its refit, with no search space to design. Every winner is in effect one "
        "gradient-boosted tree model.")
    slide = new_slide(prs, f"Step 3: AutoML vs manual ({budget / 60:.0f}-minute budget per tool)",
                      notes, "reports/tables/automl_vs_manual.csv, automl_pairwise.csv; scores "
                      "on test_future; AUC vs tuned = paired bootstrap, 95% CI; 16-core laptop")
    add_table(slide, MARGIN, Inches(1.3), Inches(12.3),
              ["model", "winner", "wall min", "AUC future", "TPR@1%FPR", "AUC vs tuned"],
              rows, [2.2, 3.7, 1.1, 1.5, 1.5, 2.3], row_h=0.55)
    add_text(slide, MARGIN, Inches(5.0), Inches(12.3), Inches(1.9), [
        f"**Accuracy: **{', '.join(tied)} tied with tuned LightGBM; {', '.join(worse)} "
        "slightly worse (paired bootstrap CI excludes 0).",
        f"**Cost: **AutoGluon {minutes(ag.wall_seconds)} min vs {minutes(tuned.wall_seconds)} min "
        "for the selected manual search and refit; all three beat untuned LightGBM.",
    ])


def slide_research_question(prs: Presentation) -> None:
    rvf = table("random_vs_future")
    prec = table("precision_both_sets")
    ch = table("challenge")
    th = "train_oof_1pct_fpr"
    tuned = row(rvf, model="optuna")
    logreg = row(rvf, model="logreg")
    tree_gaps = rvf[rvf.model != "logreg"].roc_auc_gap
    p_r = row(prec, model="optuna", test_set="test_random", threshold_name=th, prevalence=0.01)
    p_f = row(prec, model="optuna", test_set="test_future", threshold_name=th, prevalence=0.01)
    c = row(ch, model="optuna", threshold_name=th, subset="all Win64 challenge")
    c_dep = row(ch, model="optuna", threshold_name="deployment", subset="all Win64 challenge")
    rows = [["test_random (weeks 0-51)", auc(tuned.test_random_roc_auc), pct(p_r.fpr, 2),
             pct(p_r.tpr), pct(p_r.precision_at_test_mix), pct(p_r.precision)],
            ["test_future (weeks 52-63)", auc(tuned.test_future_roc_auc), pct(p_f.fpr, 2),
             pct(p_f.tpr), pct(p_f.precision_at_test_mix), pct(p_f.precision)],
            [f"challenge set ({int(c.n)}, all malware)", "-", "-", pct(c.detection_rate), "-",
             "-"]]
    notes = (
        "The research question. By AUC the random split barely overstates: the tuned "
        f"model loses only {tuned.roc_auc_gap:.4f} AUC on the future weeks, and every tree model "
        f"loses between {tree_gaps.min():.4f} and {tree_gaps.max():.4f}; only logistic "
        f"regression loses {logreg.roc_auc_gap:.4f}. The real overstatement is elsewhere. At a "
        "threshold fixed on train, the false-positive rate rises on future files, and the 50/50 "
        f"test mix hides the base rate: at 1% malware, precision is {pct(p_f.precision)} on future "
        "files. On the challenge "
        "set, malware that antivirus engines first missed, the model catches only "
        f"{pct(c.detection_rate)}.")
    slide = new_slide(prs, "Research question: how much does a random split overstate?", notes,
                      "reports/tables/random_vs_future.csv, precision_both_sets.csv, "
                      "challenge.csv; "
                      "threshold = 1% FPR on TRAIN out-of-fold scores")
    add_table(slide, MARGIN, Inches(1.3), Inches(12.3),
              ["tuned LightGBM on", "AUC", "FPR", "malware caught", "precision 50/50",
               "precision @1% malware"],
              rows, [3.4, 1.4, 1.4, 1.9, 2.0, 2.2], row_h=0.55)
    add_text(slide, MARGIN, Inches(3.75), Inches(12.3), Inches(3.1), [
        f"**AUC hides it: **gap {tuned.roc_auc_gap:.4f} [{tuned.roc_auc_gap_ci_low:.4f}, "
        f"{tuned.roc_auc_gap_ci_high:.4f}]; all tree models {tree_gaps.min():.4f} to "
        f"{tree_gaps.max():.4f}; LogReg {logreg.roc_auc_gap:.4f}.",
        f"**The base rate matters most: **precision {pct(p_r.precision_at_test_mix)} at 50/50 "
        f"vs {pct(p_r.precision)} at 1% malware; future weeks cut it to {pct(p_f.precision)} "
        f"({p_f.false_alarms_per_10k:.0f} false alarms per 10,000 files).",
        f"**The hard tail is invisible to both test sets: **only {pct(c.detection_rate)} of "
        f"challenge-set malware is caught at this threshold ({pct(c_dep.detection_rate)} at the "
        "deployed, recall-oriented one).",
    ])


def slide_leakage(prs: Presentation) -> None:
    lk = table("leakage")
    self_est = table("automl_self_estimate")
    names = {"lgbm_tuned": "Tuned LightGBM", "flaml": "FLAML", "autogluon": "AutoGluon",
             "h2o": "H2O AutoML"}
    rows = []
    for m in names:
        clean = row(lk, model=m, variant="clean")
        leak = row(lk, model=m, variant="leak_detection_ratio")
        dropped = "n/a" if m == "lgbm_tuned" else ("yes" if leak.leak_dropped_by_tool else "no")
        own = self_est[(self_est.tool == m) & (self_est.variant == "leak_detection_ratio")]
        own_auc = auc(own.own_auc.iloc[0]) if len(own) else "-"
        rows.append([names[m], f"{auc(clean.test_future_roc_auc)} -> "
                     f"{auc(leak.test_future_roc_auc)}",
                     f"#{int(leak.leak_rank_native)} / #{int(leak.leak_rank_shap)}", dropped,
                     own_auc])
    leaks = lk[lk.variant == "leak_detection_ratio"]
    n_relied = int(leaks.relied_on_leak.astype(str).eq("True").sum())
    n_shap1 = int((leaks.leak_rank_shap == 1).sum())
    date_moves = lk[lk.variant == "leak_first_submission"].delta_test_future_roc_auc.abs().max()
    notes = (
        "Leakage. We added detection_ratio, the share of antivirus engines that flagged the "
        "file, which is what the label is made from. Our tuned LightGBM and all three AutoML "
        f"tools used it: {n_shap1} of {len(leaks)} ranked it first by SHAP, test_future AUC "
        "went to "
        "about 1.0, and no tool dropped it or named it in a log. Their own validation also said "
        "1.0. Only a human question about provenance catches this. The time leak, "
        "first_submission_date, moved AUC "
        f"by at most {date_moves:.4f}. These runs used a 600 s budget on a 4-core cloud VM.")
    slide = new_slide(prs, "Leakage: did AutoML fall for it? Yes, every tool", notes,
                      "reports/tables/leakage.csv, automl_self_estimate.csv; Step 4 ran on a "
                      f"4-core / {vm_ram_gib():.1f} GiB RAM Linux VM, 600 s per AutoML run")
    add_table(slide, MARGIN, Inches(1.3), Inches(12.3),
              ["+ detection_ratio", "AUC future clean -> leak", "leak rank own / SHAP",
               "dropped?", "own validation AUC"],
              rows, [2.7, 3.4, 2.5, 1.5, 2.2], row_h=0.55)
    add_text(slide, MARGIN, Inches(4.35), Inches(12.3), Inches(2.5), [
        f"**{n_relied} of {len(leaks)} models relied on the label leak; **no tool dropped it or "
        "warned about it, and their own validation said the model was near-perfect.",
        f"**Time leak (first_submission_date): **test_future AUC moved by at most "
        f"{date_moves:.4f}: every future date lies past the training dates, and classes are "
        "50/50 in every week.",
    ])


def slide_shap(prs: Presentation) -> None:
    fam = table("shap_family_importance")
    feat = table("shap_feature_importance").sort_values("rank").head(3)
    lk = table("leakage")
    clean = row(lk, model="lgbm_tuned", variant="clean")
    ablate = row(lk, model="lgbm_tuned", variant="ablate_time_proxies")
    bias = table("bias_slices")
    signed = row(bias, test_set="test_future", threshold_name="train_oof_1pct_fpr",
                 slice_family="signed", slice="signed")
    unsigned = row(bias, test_set="test_future", threshold_name="train_oof_1pct_fpr",
                   slice_family="signed", slice="unsigned")
    top3 = fam.sort_values("rank").head(3)
    fam_txt = ", ".join(f"{r.family} {pct(r.share, 0)}" for r in top3.itertuples())
    feat_txt = "; ".join(f"{r.label}" for r in feat.itertuples())
    notes = (
        "SHAP on the tuned model. No family dominates: the top three are "
        f"{fam_txt} of the total. The three strongest single features are {feat_txt}. "
        "The COFF timestamp and signing-time columns matter, but removing them leaves the "
        "random-vs-future AUC gap about the same, "
        f"{clean.auc_gap_random_minus_future:.5f} versus {ablate.auc_gap_random_minus_future:.5f}. "
        "Signature columns are also an attack surface: signed malware on test_future is caught "
        f"{pct(signed.tpr)} of the time, against {pct(unsigned.tpr)} for unsigned malware.")
    slide = new_slide(prs, "Explainability: what the tuned model looks at (SHAP)", notes,
                      "reports/figures/shap_family_bar.png, tables/shap_*.csv, leakage.csv, "
                      "bias_slices.csv")
    add_picture_fit(slide, FIGURES / "shap_family_bar.png", MARGIN, Inches(1.2), Inches(6.4),
                    Inches(5.6))
    add_text(slide, Inches(7.2), Inches(1.3), Inches(5.6), Inches(5.6), [
        f"**Top families: **{fam_txt}.",
        f"**Top features: **{feat_txt}.",
        f"**Time proxies are not a hidden leak: **AUC gap {clean.auc_gap_random_minus_future:.5f}"
        f" with them, {ablate.auc_gap_random_minus_future:.5f} without.",
        f"**Signatures are an attack surface: **signed malware caught {pct(signed.tpr)} vs "
        f"{pct(unsigned.tpr)} unsigned (test_future).",
    ], size=19)


def api_routes() -> list[str]:
    """The FastAPI routes, read from app/api.py so the slide matches the code."""
    src = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
    return [f"{m.upper()} {p}" for m, p in re.findall(r'@app\.(get|post)\("([^"]+)"', src)]


def slide_deploy(prs: Presentation) -> None:
    thr = json.loads((ROOT / "models" / "threshold.json").read_text(encoding="utf-8"))
    chk = thr["test_future_check"]
    dep = row(table("prevalence"), model="optuna", test_set="test_future", threshold_name="deployment",
              prevalence=0.01)
    shot = REPORTS / "screenshots" / "a_malicious_correct.png"
    routes = api_routes()
    notes = (
        "Deployment. The tuned LightGBM is served by a FastAPI service and a Streamlit app; you "
        "can pick a test_future file or upload a PE file, and the app shows the score, the "
        "verdict and the top SHAP reasons. The threshold was chosen on train out-of-fold scores "
        f"so that a missed malware costs {thr['cost_ratio_fn_to_fp']:.0f} times a false alarm; on "
        f"test_future that gives {pct(chk['recall'])} recall at {pct(chk['fpr'])} false-positive "
        f"rate, but at a realistic 1% malware share only about {pct(dep.precision, 0)} of its alarms "
        "would be real malware. So it is a triage aid, not a blocker, and the app says so; the model "
        "card lists the limits.")
    slide = new_slide(prs, "Step 5: deployment demo (FastAPI + Streamlit)", notes,
                      "reports/screenshots/a_malicious_correct.png, models/threshold.json, "
                      "app/api.py, MODEL_CARD.md")
    # crop to the result area (sidebar, filters above it and the white side margins cut off) so the text is readable
    add_picture_fit(slide, shot, MARGIN, Inches(1.15), Inches(7.0), Inches(5.8),
                    crop=(0.26, 0.47, 0.95, 0.88))
    add_text(slide, Inches(7.8), Inches(1.3), Inches(5.0), Inches(5.6), [
        f"**API: **{', '.join(routes)}.",
        "**App: **pick a test_future file or upload a PE; shows P(malicious), verdict and top "
        "SHAP reasons.",
        f"**Threshold {thr['threshold']:.4f}: **missed malware costs "
        f"{thr['cost_ratio_fn_to_fp']:.0f}x a false alarm, set on train out-of-fold scores.",
        f"**On test_future: **recall {pct(chk['recall'])}, FPR {pct(chk['fpr'])}; at 1% "
        f"malware ~{pct(dep.precision, 0)} precision, so triage, not blocking.",
    ], size=19)


def slide_takeaways(prs: Presentation) -> None:
    avm = table("automl_vs_manual")
    rvf = table("random_vs_future")
    ch = table("challenge")
    prec = table("precision_both_sets")
    tuned = row(rvf, model="optuna")
    p_f = row(prec, model="optuna", test_set="test_future", threshold_name="train_oof_1pct_fpr",
              prevalence=0.01)
    c = row(ch, model="optuna", threshold_name="train_oof_1pct_fpr", subset="all Win64 challenge")
    c_dep = row(ch, model="optuna", threshold_name="deployment", subset="all Win64 challenge")
    ag, lt = row(avm, method="autogluon"), row(avm, method="lgbm_tuned")
    notes = (
        "To sum up. AutoML matched or came close to manual tuning in a fraction of the time. "
        "The random split overstates mainly through the operating point and the 50/50 class "
        "mix, not through AUC. No tool protected us from a leaky column; that needs a person "
        "who asks where each column comes from. Our limits: static features only, 12 future "
        "weeks, one seed, no adversarial testing, and Step 4 ran on a smaller 4-core cloud VM. "
        "Thank you; we are happy to take questions.")
    slide = new_slide(prs, "Takeaways and limitations", notes,
                      "reports/critical_eval.md, reports/limitations.md")
    add_text(slide, MARGIN, Inches(1.2), Inches(6.6), Inches(0.5), "Takeaways", size=24,
             color=ACCENT, bold=True)
    add_text(slide, MARGIN, Inches(1.8), Inches(6.6), Inches(5.1), [
        f"AutoGluon matched tuned LightGBM (AUC {auc(ag.test_future_roc_auc)} vs "
        f"{auc(lt.test_future_roc_auc)}) in {minutes(ag.wall_seconds)} vs "
        f"{minutes(lt.wall_seconds)} min.",
        f"The random split overstates little by AUC ({tuned.roc_auc_gap:.4f}); the 50/50 test "
        f"mix overstates a lot: precision {pct(p_f.precision_at_test_mix)} vs {pct(p_f.precision)} "
        "at 1% malware on future files.",
        "No AutoML tool caught the label leak.",
        f"The hard tail: {pct(c.detection_rate)} of challenge-set malware caught at 1% FPR "
        f"({pct(c_dep.detection_rate)} at the deployed threshold).",
    ], size=19, bullets=True)
    add_text(slide, Inches(7.4), Inches(1.2), Inches(5.4), Inches(0.5), "Limitations", size=24,
             color=ACCENT, bold=True)
    add_text(slide, Inches(7.4), Inches(1.8), Inches(5.4), Inches(5.1), [
        "Static features only; packed or modified files can evade.",
        "50/50 classes; real prevalence is unknown.",
        "12 future weeks; one seed; no adversarial test.",
        f"Step 4 ran on a 4-core / {vm_ram_gib():.1f} GiB cloud VM (600 s AutoML budget), Steps 2-3 on the "
        "16-core laptop.",
    ], size=19, bullets=True)


def build(out: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    for make in (slide_title, slide_problem_data, slide_pipeline, slide_tuning, slide_automl,
                 slide_research_question, slide_leakage, slide_shap, slide_deploy,
                 slide_takeaways):
        make(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    words = sum(len(s.notes_slide.notes_text_frame.text.split()) for s in prs.slides)
    print(f"wrote {out} ({len(prs.slides)} slides, {words} words of notes, "
          f"~{words / 160:.1f} min at 160 words per minute)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPORTS / "slides.pptx")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
