"""Build the Step 6 PDF report (reports/report.pdf) from the saved tables, figures and write-ups.

Every number in the PDF is read at build time from reports/tables/*.csv, the JSON artefacts
(reports/data_check.json, data/splits/splits_info.json, models/threshold.json,
models/final/schema.json) or the generated Step 4 write-ups (reports/critical_eval.md,
reports/limitations.md). Nothing is typed by hand, so after any fix just re-run:

    .venv/Scripts/python.exe scripts/build_report.py        (Windows)
    .venv/bin/python scripts/build_report.py                 (Linux / macOS)

It takes a few seconds, trains nothing and never touches the test sets beyond reading the
already-computed scores. It also redraws reports/figures/architecture.png (matplotlib).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no window, works on a server
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
SHOTS = REPORTS / "screenshots"
OUT_PDF = REPORTS / "report.pdf"
ARCH_PNG = FIGURES / "architecture.png"

STUDENT = "Isaac Benjamin David"
REG_NO = "URK23AI1035"
REPO_URL = "https://github.com/Turbo-Soggy/automl"

# Display names used in every table (internal method id -> readable name).
NAMES = {
    "lgbm_default": "LightGBM default",
    "logreg": "LogReg",
    "grid": "LightGBM grid",
    "random": "LightGBM random",
    # `optuna` (Step 2 tables) and `lgbm_tuned` (Step 3-4 tables) are the same deployed model.
    "optuna": "LightGBM tuned (Optuna)",
    "lgbm_tuned": "LightGBM tuned (Optuna)",
    "flaml": "FLAML",
    "autogluon": "AutoGluon",
    "h2o": "H2O AutoML",
}

MISSING: list[str] = []  # figures/screenshots that were not found (reported at the end)


# --------------------------------------------------------------------------------------
# Number formatting: one place, so every table uses the same precision.
# --------------------------------------------------------------------------------------
def f4(x: float) -> str:
    return f"{x:.4f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def pp(x: float, nd: int = 2) -> str:
    return f"{100 * x:+.{nd}f} pp"


def ci4(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def cipct(lo: float, hi: float, nd: int = 1) -> str:
    return f"[{100 * lo:.{nd}f}%, {100 * hi:.{nd}f}%]"


def intc(x: float) -> str:
    return f"{int(round(x)):,}"


def name(m: str) -> str:
    return NAMES.get(m, m)


def short_model(m: str) -> str:
    """A winning-model name short enough for a table cell, cut at a word boundary, not mid-word.

    H2O ids end in '_AutoML_<run>_<timestamp>' (no information for the reader); the manual
    entry 'LightGBM via optuna search (609 trees)' becomes 'LightGBM, 609 trees'."""
    m = m.split("_AutoML")[0]
    hit = re.match(r"LightGBM via \w+ search \((\d+) trees\)", m)
    return f"LightGBM, {hit.group(1)} trees" if hit else m


def table_csv(stem: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{stem}.csv")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


# --------------------------------------------------------------------------------------
# Fonts and styles. DejaVu ships inside matplotlib, so the same fonts exist on every OS
# (the base-14 PDF fonts cannot draw characters such as an en dash or a "≈").
# --------------------------------------------------------------------------------------
def register_fonts() -> tuple[str, str]:
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        pdfmetrics.registerFont(TTFont("DV", str(ttf / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DV-B", str(ttf / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DV-I", str(ttf / "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFont(TTFont("DV-BI", str(ttf / "DejaVuSans-BoldOblique.ttf")))
        pdfmetrics.registerFont(TTFont("DVM", str(ttf / "DejaVuSansMono.ttf")))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
        return "DV", "DVM"
    except Exception as exc:  # fall back to built-in fonts rather than fail the build
        print(f"[report] DejaVu fonts not found ({exc}); using Helvetica", file=sys.stderr)
        return "Helvetica", "Courier"


BODY_FONT, MONO_FONT = register_fonts()
_base = getSampleStyleSheet()
ST = {
    "body": ParagraphStyle("body", parent=_base["Normal"], fontName=BODY_FONT, fontSize=9.5, leading=13.2, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", parent=_base["Normal"], fontName=BODY_FONT, fontSize=9.5, leading=13, spaceAfter=2),
    "h1": ParagraphStyle("h1", parent=_base["Heading1"], fontName=BODY_FONT + "-B" if BODY_FONT == "DV" else "Helvetica-Bold",
                         fontSize=16, leading=20, spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#1f3b5c")),
    "h2": ParagraphStyle("h2", parent=_base["Heading2"], fontName=BODY_FONT + "-B" if BODY_FONT == "DV" else "Helvetica-Bold",
                         fontSize=12, leading=15, spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#1f3b5c")),
    "h3": ParagraphStyle("h3", parent=_base["Heading3"], fontName=BODY_FONT + "-B" if BODY_FONT == "DV" else "Helvetica-Bold",
                         fontSize=10, leading=13, spaceBefore=6, spaceAfter=3),
    "caption": ParagraphStyle("caption", parent=_base["Normal"], fontName=BODY_FONT, fontSize=8, leading=10.5,
                              textColor=colors.HexColor("#333333"), spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", parent=_base["Normal"], fontName=BODY_FONT, fontSize=7.4, leading=9),
    "cellh": ParagraphStyle("cellh", parent=_base["Normal"], fontName=BODY_FONT + "-B" if BODY_FONT == "DV" else "Helvetica-Bold",
                            fontSize=7.4, leading=9, textColor=colors.white),
    "title": ParagraphStyle("title", parent=_base["Title"], fontName=BODY_FONT + "-B" if BODY_FONT == "DV" else "Helvetica-Bold",
                            fontSize=22, leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#1f3b5c")),
    "subtitle": ParagraphStyle("subtitle", parent=_base["Normal"], fontName=BODY_FONT, fontSize=12, leading=16, alignment=TA_CENTER),
    "small": ParagraphStyle("small", parent=_base["Normal"], fontName=BODY_FONT, fontSize=8, leading=10.5),
    "code": ParagraphStyle("code", parent=_base["Normal"], fontName=MONO_FONT, fontSize=7.6, leading=9.8,
                           backColor=colors.HexColor("#f3f3f3"), borderPadding=4, spaceBefore=2, spaceAfter=8),
}
for _h in ("h1", "h2", "h3"):
    ST[_h].keepWithNext = 1  # never leave a heading alone at the bottom of a page
TOC_STYLES = [
    ParagraphStyle("toc1", fontName=ST["h2"].fontName, fontSize=9.5, leading=12.5, leftIndent=0),
    ParagraphStyle("toc2", fontName=BODY_FONT, fontSize=8.5, leading=10.6, leftIndent=14),
]

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
TEXT_W = PAGE_W - 2 * MARGIN


# --------------------------------------------------------------------------------------
# Markdown -> reportlab markup, for the generated write-ups (paragraphs and bullets only).
# --------------------------------------------------------------------------------------
def md_inline(text: str) -> str:
    """Convert inline markdown (bold, italics, code, links) to reportlab's mini-HTML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # keep link text only
    text = re.sub(r"`([^`]+)`", rf'<font face="{MONO_FONT}" size="8">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # single *italic*: only when the stars hug words, so "pi*TPR" stays as it is
    text = re.sub(r"(?<![\w*])\*(\S[^*]*?\S|\S)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def md_blocks(md: str) -> list[tuple[str, str]]:
    """Split markdown into ('h1'|'h2'|'h3'|'p'|'li', text) blocks; tables and code are skipped."""
    blocks: list[tuple[str, str]] = []
    para: list[str] = []
    kind = "p"
    in_code = False

    def flush() -> None:
        nonlocal para, kind
        if para:
            blocks.append((kind, " ".join(s.strip() for s in para)))
        para, kind = [], "p"

    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            flush()
            continue
        if in_code or line.strip().startswith("|"):
            flush()
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush()
            blocks.append((f"h{len(m.group(1))}", m.group(2)))
            continue
        m = re.match(r"^\s*(?:[*-]|\d+\.)\s+(.*)", line)
        if m:
            flush()
            kind, para = "li", [m.group(1)]
            continue
        if not line.strip():
            flush()
            continue
        para.append(line)
    flush()
    return blocks


def md_section(path: Path, heading: str) -> list[tuple[str, str]]:
    """Blocks under the heading that starts with `heading`, up to the next heading of the same level."""
    blocks = md_blocks(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    level = None
    for kind, text in blocks:
        if level is None:
            if kind.startswith("h") and text.startswith(heading):
                level = kind
            continue
        if kind.startswith("h") and int(kind[1]) <= int(level[1]):
            break
        out.append((kind, text))
    if level is None:
        raise KeyError(f"heading {heading!r} not found in {path}")
    return out


def md_table(path: Path, heading: str) -> tuple[list[str], list[list[str]]]:
    """The first markdown table under `heading`: (header, rows), cells converted to reportlab markup."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("#") and l.lstrip("#").strip().startswith(heading))
    tab: list[list[str]] = []
    for line in lines[start + 1:]:
        if line.startswith("#"):
            break
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):  # skip the |---| separator row
                tab.append([md_inline(c) for c in cells])
        elif tab:
            break
    return tab[0], tab[1:]


def render_blocks(blocks: Iterable[tuple[str, str]], h3_style: str = "h3") -> list[Flowable]:
    """Turn md blocks into flowables, grouping consecutive bullets into one list."""
    out: list[Flowable] = []
    items: list[str] = []

    def flush_items() -> None:
        if items:
            out.append(bullets(items, raw=True))
            items.clear()

    for kind, text in blocks:
        if kind == "li":
            items.append(md_inline(text))
            continue
        flush_items()
        if kind == "p":
            out.append(Paragraph(md_inline(text), ST["body"]))
        else:
            out.append(Paragraph(md_inline(text), ST[h3_style]))
    flush_items()
    return out


# --------------------------------------------------------------------------------------
# Flowable helpers: paragraphs, bullets, tables, figures.
# --------------------------------------------------------------------------------------
def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ST[style])


def bullets(items: Sequence[str], raw: bool = False) -> ListFlowable:
    """A bulleted list. `raw=True`: items are already reportlab markup."""
    return ListFlowable(
        [ListItem(Paragraph(t if raw else t, ST["bullet"]), leftIndent=12) for t in items],
        bulletType="bullet", start="•", leftIndent=12, bulletFontName=BODY_FONT, bulletFontSize=8,
        spaceAfter=6,
    )


class Counter:
    """Running figure and table numbers for the captions."""

    def __init__(self) -> None:
        self.fig = 0
        self.tab = 0


NUM = Counter()


def data_table(header: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[float] | None = None,
               caption: str | None = None, font_size: float = 7.4) -> list[Flowable]:
    """A zebra-striped table with wrapped cells and a numbered caption above it."""
    cell = ParagraphStyle("c", parent=ST["cell"], fontSize=font_size, leading=font_size * 1.22)
    cellh = ParagraphStyle("ch", parent=ST["cellh"], fontSize=font_size, leading=font_size * 1.22)
    data = [[Paragraph(str(h), cellh) for h in header]]
    data += [[Paragraph(str(c), cell) for c in r] for r in rows]
    if widths is None:
        widths = [TEXT_W / len(header)] * len(header)
    else:  # widths are relative weights: scale to the text width
        s = sum(widths)
        widths = [TEXT_W * w / s for w in widths]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b5c")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c2cc")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef2f6")))
    t.setStyle(TableStyle(style))
    out: list[Flowable] = []
    if caption:
        NUM.tab += 1
        out.append(Paragraph(f"<b>Table {NUM.tab}.</b> {caption}", ST["caption"]))
    out.append(t)
    out.append(Spacer(1, 8))
    return [KeepTogether(out)] if len(rows) <= 22 else out


def figure(path: Path, caption: str, width: float = 1.0, max_h: float = 11.5 * cm) -> list[Flowable]:
    """An image scaled to `width` (share of the text width) with a numbered caption below.

    A missing file becomes a clearly labelled placeholder box, and is listed in MISSING."""
    NUM.fig += 1
    cap = Paragraph(f"<b>Figure {NUM.fig}.</b> {caption}", ST["caption"])
    if not path.exists():
        MISSING.append(str(path.relative_to(ROOT)))
        box = Table([[Paragraph(f"<b>PLACEHOLDER: missing file</b> {path.relative_to(ROOT).as_posix()}", ST["body"])]],
                    colWidths=[TEXT_W * width], rowHeights=[4 * cm])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.red), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                 ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        return [KeepTogether([box, cap])]
    w_px, h_px = PILImage.open(path).size
    w = TEXT_W * width
    h = w * h_px / w_px
    if h > max_h:  # tall images: shrink to the height limit, keep the aspect ratio
        h = max_h
        w = h * w_px / h_px
    img = Image(str(path), width=w, height=h)
    img.hAlign = "CENTER"
    return [KeepTogether([img, cap])]


def figure_row(paths: Sequence[Path], caption: str, max_h: float = 10 * cm,
               crop_bottom: float = 0.0, crop_left: float = 0.0, crop_top: float = 0.0) -> list[Flowable]:
    """Several images side by side under one numbered caption (screenshots, SHAP waterfalls).

    `crop_bottom` / `crop_left` / `crop_top` trim that share of each image (empty space, the parts of
    a screenshot that are not the point of the figure)."""
    NUM.fig += 1
    cap = Paragraph(f"<b>Figure {NUM.fig}.</b> {caption}", ST["caption"])
    cell_w = TEXT_W / len(paths) - 6
    cells: list[Flowable] = []
    for path in paths:
        if not path.exists():
            MISSING.append(str(path.relative_to(ROOT)))
            cells.append(Paragraph(f"<b>PLACEHOLDER: missing file</b> {path.relative_to(ROOT).as_posix()}", ST["body"]))
            continue
        src: str | io.BytesIO = str(path)
        im = PILImage.open(path)
        if crop_bottom > 0 or crop_left > 0 or crop_top > 0:  # crop in memory: no extra files in the repo
            im = im.crop((int(im.width * crop_left), int(im.height * crop_top), im.width,
                          int(im.height * (1 - crop_bottom))))
            src = io.BytesIO()
            im.save(src, format="PNG")
            src.seek(0)
        w_px, h_px = im.size
        w, h = cell_w, cell_w * h_px / w_px
        if h > max_h:
            h, w = max_h, max_h * w_px / h_px
        cells.append(Image(src, width=w, height=h))
    t = Table([cells], colWidths=[TEXT_W / len(paths)] * len(paths))
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return [KeepTogether([t, cap])]


class Doc(BaseDocTemplate):
    """A4 document with a running footer and a table of contents fed by the h1/h2 headings."""

    def __init__(self, filename: str, **kw) -> None:
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=1.8 * cm, bottomMargin=1.8 * cm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([
            PageTemplate(id="title", frames=[frame]),
            PageTemplate(id="body", frames=[frame], onPage=self._footer),
        ])

    def _footer(self, canv, doc) -> None:
        canv.saveState()
        canv.setFont(BODY_FONT, 7.5)
        canv.setFillColor(colors.HexColor("#555555"))
        canv.drawString(MARGIN, 1.1 * cm, f"Malware detection on EMBER2024: manual tuning vs AutoML  |  {STUDENT} ({REG_NO})")
        canv.drawRightString(PAGE_W - MARGIN, 1.1 * cm, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            level = 0 if flowable.style.name == "h1" else 1
            text = flowable.getPlainText()
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def H1(text: str) -> Paragraph:
    return Paragraph(text, ST["h1"])


def H2(text: str) -> Paragraph:
    return Paragraph(text, ST["h2"])


def H3(text: str) -> Paragraph:
    return Paragraph(text, ST["h3"])


# --------------------------------------------------------------------------------------
# Architecture / workflow diagram (matplotlib).
# --------------------------------------------------------------------------------------
def draw_architecture(facts: dict) -> None:
    """Data -> dedupe/vectorise -> splits -> tuning / AutoML -> critical eval -> export -> API/app."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=200)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 70)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=color, ec="#1f3b5c", lw=1.2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9.6, weight="bold", color="#1f3b5c")
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=8.0, color="#222222", linespacing=1.3)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#1f3b5c"))

    blue, green, amber, rose, grey = "#dbe8f5", "#dff0e0", "#fdf0d5", "#f6dede", "#ececec"
    # Row 1: data pipeline
    box(1, 46, 30, 18, "1. Raw data", f"EMBER2024 Win64 weekly jsonl\n(Hugging Face joyce8/EMBER2024)\n"
        f"{facts['raw_lines']:,} raw lines, weeks 0-63\nthrember features, Apache-2.0", blue)
    box(40, 46, 30, 18, "2. Dedupe + vectorise", f"global sha256 dedup:\n{facts['unique_train']:,} train-period +\n"
        f"{facts['unique_test']:,} test-period files\n{facts['dim']:,} float32 features, 12 families", blue)
    box(79, 46, 30, 18, "3. Splits (seed 42)", f"train {facts['n_train']:,}  (weeks 0-51)\n"
        f"test_random {facts['n_random']:,}  (weeks 0-51)\ntest_future {facts['n_future']:,}  (weeks 52-63)\n"
        f"stratified by label x week, 50/50", blue)
    arrow(31.8, 55, 39.2, 55)
    arrow(70.8, 55, 78.2, 55)
    # Row 2: modelling
    box(1, 24, 33, 17, "Step 2: manual tuning", f"LightGBM default, LogReg baselines\nGrid / Random / Optuna, "
        f"{facts['n_trials']} configs each\n{facts['cv_folds']}-fold CV on a {facts['tune_n']:,}-row subset\nrefit best on all of train", green)
    box(38.5, 24, 33, 17, "Step 3: AutoML", f"FLAML, AutoGluon, H2O AutoML\n{facts['budget']:.0f} s each, one at a time\n"
        "same train split, ROC-AUC\nleaderboards saved", green)
    box(76, 24, 33, 17, "Step 4: critical evaluation", "random vs temporal split, leakage,\nbias / prevalence, SHAP, calibration,\n"
        "cost threshold, drift, challenge set", amber)
    arrow(94, 45.2, 94, 41.8)  # splits -> eval
    arrow(88, 45.2, 58, 41.8)  # splits -> automl
    arrow(82, 45.2, 20, 41.8)  # splits -> tuning
    arrow(34.6, 32.5, 37.9, 32.5)
    arrow(72.1, 32.5, 75.4, 32.5)
    # Row 3: deployment and reporting
    box(76, 2, 33, 17, "Step 5: export", "tuned LightGBM -> models/final/\nmodel.txt + schema.json (sha256)\n"
        "cost threshold, model card", rose)
    box(38.5, 2, 33, 17, "Serving", "FastAPI: /health, /predict,\n/predict_batch, /predict_file\nStreamlit demo page (via the API)", rose)
    box(1, 2, 33, 17, "Logs -> report", "every run -> experiments.jsonl\nscripts -> tables/*.csv, figures/*.png\n"
        "-> write-ups, model card, this PDF", grey)
    arrow(92, 23.2, 92, 19.8)
    arrow(75.4, 10.5, 72.1, 10.5)
    arrow(37.9, 10.5, 34.6, 10.5)
    ax.text(55, 69.5, "Workflow and architecture of the project", ha="center", va="top", fontsize=12, weight="bold",
            color="#1f3b5c")
    fig.tight_layout()
    fig.savefig(ARCH_PNG, dpi=200, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Facts read from the artefacts once, used across sections.
# --------------------------------------------------------------------------------------
def load_facts() -> dict:
    dc = load_json(REPORTS / "data_check.json")
    si = load_json(ROOT / "data" / "splits" / "splits_info.json")
    ci = dc["cache_info"]
    ts = table_csv("tuning_summary")
    avm = table_csv("automl_vs_manual")
    return {
        "dc": dc,
        "si": si,
        "raw_lines": ci["splits"]["train"]["lines"] + ci["splits"]["test"]["lines"],
        "unique_train": ci["splits"]["train"]["rows"],
        "unique_test": ci["splits"]["test"]["rows"],
        "dim": ci["feature_dim"],
        "n_train": si["splits"]["train"]["n"],
        "n_random": si["splits"]["test_random"]["n"],
        "n_future": si["splits"]["test_future"]["n"],
        "tune_n": si["tune_subset"]["n"],
        "cv_folds": si["cv_folds"],
        "n_trials": int(ts.loc[ts.method == "optuna", "n_trials"].iloc[0]),
        "budget": float(avm.loc[avm.kind == "automl", "budget_seconds"].iloc[0]),
        "threshold": load_json(ROOT / "models" / "threshold.json"),
        "schema": load_json(ROOT / "models" / "final" / "schema.json"),
        "facts4": load_json(TABLES / "eval_bias_facts.json"),
    }


def row_of(df: pd.DataFrame, **eq) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in eq.items():
        m &= df[k] == v
    sub = df[m]
    if len(sub) != 1:
        raise ValueError(f"expected one row for {eq}, got {len(sub)}")
    return sub.iloc[0]


def _join_names(methods: Iterable[str]) -> str:
    """'A', 'A and B', 'A, B and C' from internal method ids."""
    names = [name(m) for m in methods]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def automl_vs_tuned_verdict() -> str:
    """Say which AutoML tools tied, lost to or beat the tuned LightGBM on test_future AUC.

    Built from the paired bootstrap rows in automl_pairwise.csv, so the abstract and section 8
    can never claim more than the significance tests show.
    """
    avm = table_csv("automl_vs_manual")
    pw = table_csv("automl_pairwise")
    vs = pw[(pw.test_set == "test_future") & (pw.metric == "roc_auc") & (pw.method_b == "lgbm_tuned")
            & pw.method_a.isin(avm.loc[avm.kind == "automl", "method"])]
    sig = vs.significant_95.astype(bool)
    tied, worse, better = vs[~sig], vs[sig & (vs.diff_a_minus_b < 0)], vs[sig & (vs.diff_a_minus_b > 0)]
    parts = []
    if not tied.empty:
        parts.append(f"{_join_names(tied.method_a)} statistically tied the tuned LightGBM")
    if not worse.empty:
        parts.append(f"{_join_names(worse.method_a)} {'was' if len(worse) == 1 else 'were'} slightly but "
                     f"significantly worse (by at most {abs(worse.diff_a_minus_b.min()):.4f} AUC)")
    if not better.empty:
        parts.append(f"{_join_names(better.method_a)} significantly beat it")
    tail = "none beat it" if better.empty else ""
    return "; ".join(parts + ([tail] if tail else []))


def max_auc_shortfall() -> float:
    """Largest test_future AUC shortfall of an AutoML tool against the tuned LightGBM (0 if none is behind)."""
    avm = table_csv("automl_vs_manual")
    tuned = row_of(avm, method="lgbm_tuned").test_future_roc_auc
    return max(0.0, float(tuned - avm.loc[avm.kind == "automl", "test_future_roc_auc"].min()))


def drift_vs_tuned_verdict() -> tuple[str, str]:
    """Section 8.1 headline on drift: which AutoML gaps match the tuned model's (gap_vs_tuned.csv)."""
    g = table_csv("gap_vs_tuned")
    g = g[(g.metric == "roc_auc") & (g.kind == "automl")]
    diff = g[g.differs_from_tuned.astype(bool)]
    same = g[~g.differs_from_tuned.astype(bool)]
    if diff.empty:
        return "Same robustness to drift as the manual model.", "Their random-vs-future gaps matched the tuned model's"
    head = "Mostly the same robustness to drift as the manual model."
    extra = "; ".join(f"{name(r.model)}'s is slightly larger ({r.gap_minus_tuned:+.4f} "
                      f"{ci4(r.gap_minus_tuned_ci_low, r.gap_minus_tuned_ci_high)}, CI excludes 0)"
                      for _, r in diff.iterrows())
    body = (f"{_join_names(same.model)} had random-vs-future gaps that match the tuned model's" if not same.empty
            else "No tool matched the tuned model's gap") + f"; {extra}"
    return head, body


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------
def title_page(F: dict) -> list[Flowable]:
    today = dt.date.today().isoformat()
    s: list[Flowable] = [Spacer(1, 3.2 * cm)]
    s.append(P("Malware Detection on EMBER2024:<br/>Manual Tuning versus AutoML", "title"))
    s.append(Spacer(1, 0.4 * cm))
    s.append(P("How much do random-split scores overstate real-world performance, and do AutoML tools "
               "protect against, or fall for, leakage and temporal drift?", "subtitle"))
    s.append(Spacer(1, 1.6 * cm))
    info = [
        ["Course", "AutoML Skill-Based Project, Evaluation IA-3, Part B (30 marks)"],
        ["Programme", "Final-year B.Tech Artificial Intelligence and Data Science, "
                      "Karunya Institute of Technology and Sciences"],
        ["Student", f"{STUDENT}"],
        ["Register number", REG_NO],
        ["Team members", "TODO: team members to be added by the student"],
        ["Code", REPO_URL],
        ["Report built", f"{today} by scripts/build_report.py (all numbers read from reports/ at build time)"],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", ST["body"]), Paragraph(b, ST["body"])] for a, b in info],
              colWidths=[4 * cm, TEXT_W - 4 * cm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cc")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s.append(t)
    s.append(Spacer(1, 1.8 * cm))
    avm = table_csv("automl_vs_manual")
    tuned = row_of(avm, method="lgbm_tuned")
    s.append(P(f"Dataset: EMBER2024 Win64 PE files, {F['n_train']:,} training, {F['n_random']:,} random-test and "
               f"{F['n_future']:,} future-test files, {F['dim']:,} static features. Best model: tuned LightGBM, "
               f"test_future ROC-AUC {f4(tuned.test_future_roc_auc)}.", "small"))
    return s


def abstract(F: dict) -> list[Flowable]:
    rvf = table_csv("random_vs_future")
    opt = row_of(rvf, model="optuna")
    avm = table_csv("automl_vs_manual")
    am = avm[avm.kind == "automl"]
    prev = table_csv("precision_both_sets")
    p1 = row_of(prev, model="optuna", test_set="test_future", threshold_name="train_oof_1pct_fpr", prevalence=0.01)
    ch = table_csv("challenge")
    c = row_of(ch, model="optuna", threshold_name="train_oof_1pct_fpr", subset="all Win64 challenge")
    lk = table_csv("leakage")
    dr = lk[lk.variant == "leak_detection_ratio"]
    s = [H1("Abstract")]
    s.append(P(
        f"We train malware detectors on static features of Windows 64-bit PE files from EMBER2024 and compare "
        f"manual hyperparameter tuning of LightGBM (grid, random and Bayesian search with Optuna) against three "
        f"AutoML tools (FLAML, AutoGluon and H2O AutoML) under one fixed time budget of {F['budget']:.0f} s each. "
        f"Every model is scored on two held-out sets: <i>test_random</i> (the same 52 weeks as training) and "
        f"<i>test_future</i> (12 strictly later weeks). The tuned LightGBM reaches ROC-AUC "
        f"{f4(opt.test_random_roc_auc)} on test_random and {f4(opt.test_future_roc_auc)} on test_future "
        f"(gap {f4(opt.roc_auc_gap)} {ci4(opt.roc_auc_gap_ci_low, opt.roc_auc_gap_ci_high)}); the AutoML winners "
        f"reach {f4(am.test_future_roc_auc.min())} to {f4(am.test_future_roc_auc.max())} on test_future: "
        f"{automl_vs_tuned_verdict()} (paired bootstrap). The random split overstates ranking quality only slightly; the larger "
        f"overstatement comes from the 50/50 class mix: at a realistic 1% malware prevalence only "
        f"{pct(p1.precision)} of the tuned model's alarms on test_future are real. On the EMBER2024 challenge set "
        f"(malware that antivirus engines first missed) it detects only {pct(c.detection_rate)}. When a post-hoc "
        f"label proxy (<i>detection_ratio</i>) is added as a feature, every tool uses it "
        f"({int(dr.relied_on_leak.sum())} of {len(dr)} models rank it first), reports a near-perfect validation "
        f"score and gives no warning. AutoML came within {max_auc_shortfall():.4f} test_future AUC of expert "
        f"tuning in a fraction of the effort, but it did not "
        f"protect against leakage or drift: those protections were human ones (column provenance, a temporal "
        f"hold-out, a realistic base rate). The best model is served with FastAPI and Streamlit."))
    return s


def problem(F: dict) -> list[Flowable]:
    s = [H1("1. Problem statement and research question")]
    s.append(P(
        "Static malware detection decides from a file's bytes and PE structure, without running it, whether a "
        "Windows executable is malicious. It is a natural binary classification task, but its published scores are "
        "often measured on random splits, where training and test files come from the same weeks. In deployment a "
        "detector always faces files that are newer than anything it was trained on, and the malware share of real "
        "traffic is far below the 50% of a benchmark."))
    s.append(P("<b>Research question.</b> How much do random-split scores overstate real-world performance for "
               "malware detection, and do AutoML tools protect against, or fall for, leakage and temporal drift?"))
    s.append(P("We answer it in four steps that follow the course rubric:"))
    s.append(bullets([
        "<b>Step 2 (baseline and tuning):</b> LightGBM default and logistic regression baselines, then the same "
        "LightGBM tuned by grid search, random search and Optuna with an equal number of configurations.",
        "<b>Step 3 (AutoML):</b> FLAML, AutoGluon and H2O AutoML on the same training split with the same budget, "
        "their leaderboards, and a comparison with manual tuning on score, time and interpretability.",
        "<b>Step 4 (critical evaluation):</b> random vs temporal split, two deliberate leaky features, bias and "
        "prevalence, SHAP explanations, calibration, a cost-sensitive threshold, drift and a hard challenge set.",
        "<b>Step 5 (deployment):</b> export of the best model, a FastAPI service, a Streamlit demo and a model card.",
    ]))
    s.append(P("Ground rules: one seed (42) everywhere; no model is trained, tuned, early-stopped or thresholded on "
               "test_random or test_future, and no configuration is selected on them (the one decision that reads "
               "test scores, which finished model to deploy, is stated openly in section 7); every table and figure "
               "is generated by a script from the experiment log or saved artefacts."))
    return s


def dataset(F: dict) -> list[Flowable]:
    """Section 2: dataset, dedup, labels and the three splits, from data_check.json and splits_info.json."""
    dc, si = F["dc"], F["si"]
    ci = dc["cache_info"]
    tr, te = ci["splits"]["train"], ci["splits"]["test"]
    s = [H1("2. Dataset")]
    s.append(P(
        "We use <b>EMBER2024</b> (Joyce et al., KDD 2025), the successor of the original EMBER benchmark, "
        "restricted to its Win64 PE slice. Each weekly file holds static features extracted by the "
        f"<i>thrember</i> extractor (version {ci['thrember_version']}, commit <font face='{MONO_FONT}' size='8'>"
        f"{ci['thrember_commit'][:12]}</font>). The data are published on Hugging Face "
        "(<font face='%s' size='8'>joyce8/EMBER2024</font>) under the Apache-2.0 license." % MONO_FONT))
    rows = [
        ["Source", "Hugging Face joyce8/EMBER2024: Win64_train.zip, Win64_test.zip, challenge.zip"],
        ["Feature extractor", f"thrember {ci['thrember_version']} (commit {ci['thrember_commit'][:12]})"],
        ["License", "Apache-2.0"],
        ["Training period", f"{tr['files']} weekly files, {si['splits']['train']['date_min'][:10]} to "
                            f"{si['splits']['train']['date_max'][:10]} (weeks 0-51)"],
        ["Test period", f"{te['files']} weekly files, {si['splits']['test_future']['date_min'][:10]} to "
                        f"{si['splits']['test_future']['date_max'][:10]} (weeks 52-63)"],
        ["Raw lines", f"{tr['lines']:,} (training period) + {te['lines']:,} (test period)"],
        ["Unique files after dedup", f"{tr['rows']:,} + {te['rows']:,}"],
        ["Feature vector", f"{ci['feature_dim']:,} float32 columns in {len(dc['cache']['feature_groups'])} families"],
        ["Labels", "0 = benign, 1 = malicious (only 0/1 present; no unlabelled rows)"],
    ]
    s += data_table(["item", "value"], rows, widths=[1, 3], caption="Dataset facts (from reports/data_check.json).")
    s.append(H2("2.1 The duplicate finding"))
    s.append(P(
        f"While building the feature cache we found that every weekly file lists each sample twice: "
        f"{tr['duplicates_in_file']:,} of {tr['lines']:,} training-period lines and {te['duplicates_in_file']:,} of "
        f"{te['lines']:,} test-period lines are in-file sha256 duplicates, plus {te['duplicates_across_files']} "
        f"duplicates across test files. We deduplicate globally by sha256 (first occurrence kept) <i>before</i> any "
        f"split. Without that, a random split would put copies of the same file into train and test and inflate "
        f"test_random further. The checks in data_check.json confirm zero sha256 overlap between the splits."))
    s.append(H2("2.2 Splits and class balance"))
    s.append(P(
        f"The brief asks for a stratified subsample of 150k-250k rows, the same for every method. From the "
        f"training period we draw <b>train</b> ({F['n_train']:,}) and <b>test_random</b> ({F['n_random']:,}), "
        f"stratified by label x week; from the 12 later weeks we draw <b>test_future</b> ({F['n_future']:,}). "
        f"test_future starts strictly after the last training file. Step 2 tunes on a {F['tune_n']:,}-row subset of "
        f"train with {F['cv_folds']} persisted CV folds."))
    cb = table_csv("class_balance")
    rows = []
    for _, r in cb.iterrows():
        sp = si["splits"][r["split"]]
        rows.append([r["split"], intc(r.n), intc(r.n_benign), intc(r.n_malicious), pct(r.malicious_share, 2),
                     str(sp["n_weeks"]), f"{sp['date_min'][:10]} to {sp['date_max'][:10]}"])
    s += data_table(["split", "rows", "benign", "malicious", "malicious share", "weeks", "dates"], rows,
                    widths=[1.2, 0.8, 0.8, 0.8, 0.9, 0.6, 1.8],
                    caption="Splits and class balance (reports/tables/class_balance.csv, data/splits/splits_info.json). "
                            "Classes are 50/50 by construction, because each week of EMBER2024 holds equal numbers.")
    s.append(H2("2.3 Feature families"))
    fg = dc["cache"]["feature_groups"]
    rows = [[g["name"], str(g["width"]), f"{g['start']}-{g['end'] - 1}"] for g in fg]
    desc = {
        "general": "file size, virtual size, imports/exports counts and similar",
        "histogram": "byte value histogram (256 bins)",
        "byteentropy": "joint byte/entropy histogram",
        "strings": "printable strings: counts, character frequencies, special patterns",
        "header": "COFF and optional header fields (incl. COFF timestamp)",
        "section": "section names, sizes, entropy, characteristics",
        "imports": "imported libraries and functions (hashed)",
        "exports": "exported functions (hashed)",
        "datadirectories": "PE data directory sizes and addresses",
        "richheader": "Rich header (compiler/linker) entries",
        "authenticode": "code-signing information (incl. signing times)",
        "pefilewarnings": "warnings raised by the pefile parser",
    }
    rows = [r + [desc.get(r[0], "")] for r in rows]
    s += data_table(["family", "columns", "column range", "content"], rows, widths=[1.2, 0.6, 0.9, 3.4],
                    caption="The 12 thrember feature families (offsets from reports/data_check.json).")
    d = dc["detection_ratio"]["train"]
    s.append(P(
        "Two metadata columns are <b>not</b> features: <i>detection_ratio</i> (the share of antivirus engines that "
        f"flagged the file; its ROC-AUC against the label is {d['roc_auc_detection_ratio_float_vs_label']:.4f}, "
        "because the label is derived from it) and <i>first_submission_date</i>. Step 4 adds each of them as a "
        "deliberate leak. Time proxies already inside X are the COFF timestamp (column 696) and the authenticode "
        "signing-time columns."))
    return s


def methodology(F: dict) -> list[Flowable]:
    s = [H1("3. Methodology and workflow")]
    s += figure(ARCH_PNG, "Workflow and architecture. Every run appends a record to reports/experiments.jsonl; "
                          "scripts turn the log and the saved artefacts into the tables, figures and this report.",
                width=1.0, max_h=12 * cm)
    s.append(H2("3.1 Design decisions"))
    s.append(bullets([
        "<b>Model family for manual tuning: LightGBM.</b> Gradient-boosted trees are the standard strong baseline "
        "on EMBER, handle 2,568 unscaled numeric features, and train fast enough for 36 CV configurations per tuner. "
        "Logistic regression (on standardised features) is the linear reference.",
        f"<b>Equal budget for the tuners:</b> grid, random and Optuna each evaluate {F['n_trials']} configurations, "
        f"scored by the same {F['cv_folds']}-fold CV on the same {F['tune_n']:,}-row subset of train, with early "
        "stopping inside the folds; the best configuration by mean CV AUC is refit on all of train.",
        f"<b>Equal budget for the AutoML tools:</b> {F['budget']:.0f} s of search each (the brief's 20 minutes), "
        "the same train split, ROC-AUC as the metric, run one after another on an otherwise idle machine. "
        "Wall-clock time is logged, including overruns.",
        "<b>Metrics:</b> ROC-AUC, F1 and accuracy at 0.5 (the rubric), plus TPR at 1% FPR (the operating point a "
        "malware analyst cares about). 95% confidence intervals come from a bootstrap with 1000 resamples (seeds 42 and 43, one per test set) "
        "or Wilson intervals for rates; model differences use paired bootstraps on the same rows.",
        "<b>Test discipline:</b> test_random and test_future are scored once per final model. Thresholds are chosen "
        "on train out-of-fold scores, never on a test set.",
        "<b>Reproducibility:</b> SEED = 42 everywhere; persisted splits and CV folds; every table and figure is "
        "rebuilt from the log by a <font face='%s' size='8'>--tables-only</font> entry point." % MONO_FONT,
    ]))
    return s


def tuning(F: dict) -> list[Flowable]:
    ts = table_csv("tuning_summary")
    s = [H1("4. Step 2: baseline and hyperparameter tuning")]
    s.append(P(
        f"Two baselines (LightGBM with default parameters, logistic regression) and three tuners over one LightGBM "
        f"search space. Each tuner ran {F['n_trials']} configurations; the time column is the whole search. Scores "
        f"are for the refit of the best configuration on all {F['n_train']:,} training rows."))
    rows = []
    for _, r in ts.iterrows():
        rows.append([
            name(r.method), "-" if pd.isna(r.n_trials) else str(int(r.n_trials)), f4(r.best_cv_auc),
            f4(r.test_random_roc_auc),
            f"{f4(r.test_future_roc_auc)} {ci4(r.test_future_roc_auc_ci_low, r.test_future_roc_auc_ci_high)}",
            f4(r.test_future_f1), f4(r.test_future_accuracy), pct(r.test_future_tpr_at_1pct_fpr),
            ("-" if pd.isna(r.search_seconds) else f"{r.search_seconds / 60:.1f} min") + f" + {r.fit_seconds:.0f} s",
        ])
    s += data_table(["method", "trials", "CV AUC", "AUC random", "AUC future [95% CI]", "F1 future",
                     "acc. future", "TPR@1% FPR future", "search + refit time"], rows,
                    widths=[1.6, 0.65, 0.75, 0.75, 1.75, 0.7, 0.75, 0.85, 1.0],
                    caption="Step 2 results (reports/tables/tuning_summary.csv). CV AUC is the mean over the 3 folds "
                            "of the 60k tuning subset; test columns are for the refit on all of train.")
    pw = table_csv("tuning_pairwise")
    fut = pw[(pw.test_set == "test_future") & (pw.metric == "roc_auc")]
    tuners = fut[fut.method_a.isin(["grid", "random", "optuna"]) & fut.method_b.isin(["grid", "random", "optuna"])]
    n_sig = int(tuners.significant_95.sum())
    vs_def = fut[(fut.method_b == "lgbm_default") & fut.method_a.isin(["grid", "random", "optuna"])]
    best = ts.loc[ts.best_cv_auc.idxmax()]
    s.append(P(
        f"<b>Findings.</b> All three tuners beat the default LightGBM on test_future AUC "
        f"({int(vs_def.significant_95.sum())} of {len(vs_def)} paired differences significant, by "
        f"{f4(vs_def.diff_a_minus_b.min())} to {f4(vs_def.diff_a_minus_b.max())}). Between the tuners the differences "
        f"are tiny: {n_sig} of {len(tuners)} pairwise test_future AUC differences excludes zero "
        f"(reports/tables/tuning_pairwise.csv), so with this budget the search method matters far less than "
        f"tuning at all. {name(best.method)} had the best CV AUC ({f4(best.best_cv_auc)}) and was therefore chosen "
        f"as <i>the tuned model</i> for the rest of the project, a choice made on CV only. Search times: " +
        ", ".join(f"{name(r.method)} {r.search_seconds / 60:.1f} min"
                  for _, r in ts.dropna(subset=["search_seconds"]).iterrows()) +
        f" (Figure {NUM.fig + 2} shows the best CV AUC against the number of configurations tried)."))
    s += figure(FIGURES / "tuning_compare.png", "Step 2 comparison of the tuners and baselines: scores with 95% CIs "
                "on both test sets and the search time (reports/figures/tuning_compare.png).", width=0.92)
    s += figure(FIGURES / "tuning_convergence.png", "Best CV AUC so far against the number of configurations "
                "evaluated, per tuner (reports/figures/tuning_convergence.png).", width=0.72)
    return s


def automl(F: dict) -> list[Flowable]:
    avm = table_csv("automl_vs_manual")
    s = [H1("5. Step 3: AutoML and tool comparison")]
    s.append(P(
        f"FLAML, AutoGluon (medium_quality preset) and H2O AutoML were each given the same train split "
        f"({F['n_train']:,} rows x {F['dim']:,} features), ROC-AUC as the metric, {F['budget']:.0f} s of search "
        f"and 16 threads on the laptop, one after another in separate processes. Each tool chose its own internal "
        f"validation inside train; neither test set was loaded before a tool's fit had finished."))
    rows = []
    for _, r in avm.iterrows():
        budget = "-" if pd.isna(r.budget_seconds) else f"{r.budget_seconds:.0f} s"
        wall = f"{r.wall_minutes:.1f} min"
        rows.append([
            name(r.method), short_model(str(r.best_model)), budget, wall,
            f4(r.test_random_roc_auc),
            f"{f4(r.test_future_roc_auc)} {ci4(r.test_future_roc_auc_ci_low, r.test_future_roc_auc_ci_high)}",
            f4(r.test_future_f1), f4(r.test_future_accuracy), pct(r.test_future_tpr_at_1pct_fpr),
        ])
    s += data_table(["method", "winning model", "budget", "wall time", "AUC random", "AUC future [95% CI]",
                     "F1 future", "acc. future", "TPR@1% FPR future"], rows,
                    widths=[1.4, 1.75, 0.8, 0.75, 0.75, 1.7, 0.7, 0.75, 0.9],
                    caption="AutoML vs manual (reports/tables/automl_vs_manual.csv). For the tuned LightGBM the wall "
                            "time covers the whole Step 2 search of its tuner plus the refit.")
    s += figure(FIGURES / "automl_vs_manual.png", "AutoML tools against the manual models on both test sets, with 95% "
                "bootstrap CIs (reports/figures/automl_vs_manual.png).", width=1.0)
    pw = table_csv("automl_pairwise")
    fut = pw[(pw.test_set == "test_future") & (pw.metric == "roc_auc") & (pw.method_b == "lgbm_tuned")]
    parts = []
    for _, r in fut.iterrows():
        verdict = "significant" if r.significant_95 else "not significant"
        parts.append(f"{name(r.method_a)} {r.diff_a_minus_b:+.4f} {ci4(r.ci_low, r.ci_high)} ({verdict})")
    s.append(P("<b>Paired test_future AUC difference against the tuned LightGBM</b> "
               "(reports/tables/automl_pairwise.csv): " + "; ".join(parts) + "."))
    over = avm[avm.kind == "automl"]
    s.append(P(
        "<b>Time.</b> Each tool's search stopped at the budget, but FLAML's final refit on all rows and H2O's stacked "
        "ensemble came on top: wall time minus budget was " +
        ", ".join(f"{name(r.method)} {r.over_budget_seconds:+.0f} s" for _, r in over.iterrows()) +
        f". The manual route needed {avm.loc[avm.method == 'lgbm_tuned', 'all_manual_seconds'].iloc[0] / 3600:.1f} h "
        "of search and refit for all three tuners together, against well under half an hour per AutoML tool."))
    s.append(H2("5.1 Leaderboards"))
    s.append(P("Summaries of the saved leaderboards (reports/leaderboards/); the full lists are in Appendix A."))
    lb = pd.read_csv(REPORTS / "leaderboards" / "flaml_leaderboard.csv")
    rows = [[str(r.estimator), str(int(r.n_trials)), f4(r.best_val_auc), intc(r.best_sample_size)] for _, r in lb.head(4).iterrows()]
    s += data_table(["FLAML estimator", "trials", "best validation AUC", "rows in best trial"], rows,
                    widths=[1.5, 0.6, 1, 1], caption="FLAML: top 4 estimators by validation AUC.")
    lb = pd.read_csv(REPORTS / "leaderboards" / "autogluon_leaderboard.csv")
    rows = [[str(r.model), f4(r.score_val), f"{r.fit_time:.0f} s", str(int(r.stack_level))] for _, r in lb.head(4).iterrows()]
    s += data_table(["AutoGluon model", "validation AUC", "fit time", "stack level"], rows,
                    widths=[1.5, 1, 0.8, 0.6], caption="AutoGluon: top 4 models by validation AUC.")
    lb = pd.read_csv(REPORTS / "leaderboards" / "h2o_leaderboard.csv")
    rows = [[str(r.model_id).split("_AutoML")[0], str(r.algo), f4(r.auc), f"{r.training_time_ms / 1000:.0f} s"]
            for _, r in lb.head(4).iterrows()]
    s += data_table(["H2O model", "algorithm", "leaderboard AUC", "training time"], rows,
                    widths=[1.8, 1, 0.8, 0.8], caption="H2O AutoML: top 4 models (leaderboard on its validation frame).")
    s.append(H2("5.2 Preprocessing and interpretability"))
    s.append(P(
        "None of the tools engineered new features. FLAML's DataTransformer dropped the constant columns, fitted a "
        "median imputer (nothing to fill: no missing values) and scaled nothing; AutoGluon's feature generators "
        "dropped constant, duplicate and near-constant columns; H2O ignores constant columns inside each model. "
        "Read back from the saved models (reports/automl_comparison.md, reports/leaderboards/*_preprocessing.json):"))
    hdr, prows = md_table(REPORTS / "automl_comparison.md", "Preprocessing and feature engineering")
    s += data_table(hdr, prows, caption="Preprocessing applied by each tool (table from reports/automl_comparison.md).")
    rows = [[name(r.method), str(r.interpretability)] for _, r in avm.iterrows()]
    s += data_table(["method", "what the final model is, and how it can be explained"], rows, widths=[1, 4],
                    caption="Interpretability of each final model (automl_vs_manual.csv, column interpretability).")
    return s


def critical(F: dict) -> list[Flowable]:
    """Section 6: the Step 4 critical evaluation, from the Step 4 tables, figures and critical_eval.md."""
    ce = REPORTS / "critical_eval.md"
    s = [H1("6. Step 4: critical evaluation")]
    s.append(P(
        "<b>Hardware note.</b> Steps 2-3 ran on the laptop (16-core Intel Core Ultra 9 285H, 63 GB RAM, Windows, "
        f"16 threads). Step 4 ran on a 4-core, {vm_ram_gib():.1f} GiB RAM Linux cloud VM; its leakage runs used a "
        "600 s AutoML budget, "
        "and every with/without comparison inside Step 4 used the same machine and settings. Step 4 timings and "
        "AutoML scores are therefore not comparable with Step 3 timings. The challenge set was scored on the laptop "
        "(scoring fits nothing, so hardware does not change those numbers)."))
    s.append(H2("6.1 The answer in short"))
    s += render_blocks(md_section(ce, "The answer in short"))

    # ---- random vs temporal
    s.append(H2("6.2 Random vs temporal split"))
    rvf = table_csv("random_vs_future")
    rows = []
    for _, r in rvf.iterrows():
        rows.append([name(r.model), f4(r.test_random_roc_auc), f4(r.test_future_roc_auc),
                     f"{f4(r.roc_auc_gap)} {ci4(r.roc_auc_gap_ci_low, r.roc_auc_gap_ci_high)}",
                     pct(r.test_random_tpr_at_1pct_fpr), pct(r.test_future_tpr_at_1pct_fpr),
                     pp(r.tpr_at_1pct_fpr_gap), pct(r.test_random_fpr, 2), pct(r.test_future_fpr, 2)])
    s += data_table(["model", "AUC random", "AUC future", "AUC gap [95% CI]", "TPR@1% FPR random",
                     "TPR@1% FPR future", "TPR gap", "FPR random (t=0.5)", "FPR future (t=0.5)"], rows,
                    widths=[1.6, 0.7, 0.7, 1.6, 0.85, 0.85, 0.75, 0.75, 0.75],
                    caption="Random vs future, saved Step 2-3 models at threshold 0.5 (reports/tables/random_vs_future.csv). "
                            "Gap = test_random minus test_future.")
    cv = table_csv("cv_comparison")
    rows = [[str(r.estimate), f4(r.roc_auc), "-" if pd.isna(r.tpr_at_1pct_fpr) else pct(r.tpr_at_1pct_fpr),
             f4(r.roc_auc_abs_error_vs_future), str(r.note)]
            for _, r in cv.iterrows()]
    s += data_table(["estimate", "AUC", "TPR@1%FPR", "|AUC error| vs future", "how it was computed"], rows,
                    widths=[1.5, 0.6, 0.85, 0.85, 3.0], font_size=7,
                    caption="Which internal estimate predicts test_future best? Tuned LightGBM (reports/tables/cv_comparison.csv).")
    s += render_blocks(md_section(ce, "3. Why the random-vs-future gap"))

    # ---- leakage
    s.append(H2("6.3 Deliberate leakage"))
    s.append(P(
        "Two leaky columns were added, one at a time, to tuned LightGBM and to each AutoML tool, which were "
        "retrained with and without the column at the same budget (600 s per AutoML run on the VM). A third "
        "variant removes the time proxies already inside X from the tuned model."))
    lk = table_csv("leakage")
    rows = []
    for _, r in lk.iterrows():
        delta = "" if pd.isna(r.delta_test_future_roc_auc) else f"{r.delta_test_future_roc_auc:+.4f}" + (
            " *" if r.delta_test_future_roc_auc_significant is True or str(r.delta_test_future_roc_auc_significant) == "True" else "")
        rank = "" if pd.isna(r.leak_rank_native) else f"#{int(r.leak_rank_native)} / " + (
            "-" if pd.isna(r.leak_rank_shap) else f"#{int(r.leak_rank_shap)}")
        relied = "" if pd.isna(r.relied_on_leak) else ("yes" if str(r.relied_on_leak) == "True" else "no")
        rows.append([name(r.model), str(r.variant), f4(r.test_random_roc_auc), f4(r.test_future_roc_auc), delta,
                     rank, relied, f"{r.wall_s:.0f} s"])
    s += data_table(["model", "variant", "AUC random", "AUC future", "delta future vs clean", "leak rank (own / SHAP)",
                     "relied on leak", "wall time (VM)"], rows,
                    widths=[1.4, 1.6, 0.7, 0.7, 0.9, 0.9, 0.7, 0.7],
                    caption="Leakage experiments (reports/tables/leakage.csv). * = paired bootstrap CI excludes 0. "
                            "'Relied on leak' = the model ranked the column first by its own importance or by SHAP "
                            "(for FLAML only the SHAP rank counts, because its own importance is a split count).")
    s += figure(FIGURES / "leakage_jump.png", "Score with and without each leak column, per model, both test sets "
                "(reports/figures/leakage_jump.png).", width=0.9)
    s += render_blocks(md_section(ce, "4. Does AutoML protect against leakage"))

    # ---- AutoML drift + self estimate
    s.append(H2("6.4 Do AutoML tools protect against drift?"))
    gv = table_csv("gap_vs_tuned")
    g = gv[gv.metric == "roc_auc"]
    rows = [[name(r.model), r.kind, f4(r.gap), f"{r.gap_minus_tuned:+.4f} {ci4(r.gap_minus_tuned_ci_low, r.gap_minus_tuned_ci_high)}",
             "yes" if r.differs_from_tuned else "no"] for _, r in g.iterrows()]
    s += data_table(["model", "kind", "AUC gap", "gap minus tuned gap [95% CI]", "differs"], rows,
                    widths=[1.6, 0.8, 0.8, 1.8, 0.6],
                    caption="Random-vs-future AUC gap compared with the tuned LightGBM's gap, paired bootstrap "
                            "(reports/tables/gap_vs_tuned.csv).")
    se = table_csv("automl_self_estimate")
    rows = [[name(r.tool), str(r.run).replace("_leakage", "").replace("_automl", ""), str(r.variant), f"{r.budget_s:.0f} s", str(r.machine), f4(r.own_auc),
             f4(r.test_random_roc_auc), f4(r.test_future_roc_auc), f"{r.own_minus_future:+.4f}"] for _, r in se.iterrows()]
    s += data_table(["tool", "run", "variant", "budget", "machine", "own valid. AUC", "AUC random", "AUC future",
                     "own minus future"], rows, widths=[0.9, 0.9, 1.35, 0.65, 1.45, 0.7, 0.65, 0.65, 0.7], font_size=6.9,
                    caption="What each tool's own validation (a random split of train) said, against the test sets "
                            "(reports/tables/automl_self_estimate.csv).")
    s += render_blocks([b for b in md_section(ce, "5. Does AutoML protect against temporal drift") if b[0] == "li"][:2])

    # ---- bias / prevalence
    s.append(H2("6.5 Bias, imbalance and base rate"))
    prev = table_csv("precision_both_sets")
    sub = prev[(prev.model == "optuna") & (prev.threshold_name == "train_oof_1pct_fpr")]
    rows = [[r.test_set, pct(r.prevalence, 0), f"{r.threshold:.4f}", pct(r.tpr), pct(r.fpr, 2),
             f"{pct(r.precision)} {cipct(r.precision_ci_low, r.precision_ci_high)}", f"{r.false_alarms_per_10k:.0f}",
             f"{r.missed_per_10k:.0f}", pct(r.precision_at_test_mix)] for _, r in sub.iterrows()]
    s += data_table(["test set", "prevalence", "threshold", "TPR", "FPR", "precision [95% CI]",
                     "false alarms / 10k", "missed / 10k", "precision at 50/50"], rows,
                    widths=[0.85, 0.85, 0.8, 0.6, 0.6, 1.45, 0.8, 0.7, 0.8],
                    caption="Tuned LightGBM at its train 1%-FPR threshold, re-weighted to realistic malware prevalence "
                            "(reports/tables/precision_both_sets.csv).")
    s += figure(FIGURES / "bias_slices.png", "Per-slice detection rate and benign false-positive rate of the tuned "
                "model on test_future, with Wilson CIs (reports/figures/bias_slices.png).", width=0.95)
    s += render_blocks([b for b in md_section(ce, "5b. Bias slices") if b[0] == "li"][:1])
    cw = table_csv("class_weight")
    cwf = cw[cw.test_set == "test_future"]
    rows = [[str(r.model), f4(r.roc_auc), f"{r.roc_auc_minus_baseline:+.4f} {ci4(r.roc_auc_minus_baseline_ci_low, r.roc_auc_minus_baseline_ci_high)}",
             pct(r.tpr_at_1pct_fpr), f4(r.f1), pct(r.fpr, 2)] for _, r in cwf.iterrows()]
    s += data_table(["refit", "AUC future", "AUC minus no weights [95% CI]", "TPR@1%FPR", "F1 @0.5", "FPR @0.5"], rows,
                    widths=[1.6, 0.7, 1.6, 0.85, 0.6, 0.6],
                    caption="Class weights on test_future (reports/tables/class_weight.csv). With 50/50 classes there "
                            "is no imbalance to correct; reweighting only shifts the scores.")

    # ---- SHAP
    s.append(H2("6.6 Explainability (SHAP)"))
    sf = table_csv("shap_family_importance")
    rows = [[str(int(r["rank"])), r.family, str(int(r.n_features)), pct(r.share), f"{r.grouped_mean_abs_shap:.2f}",
             str(r.top_feature)[:60]] for _, r in sf.iterrows()]
    s += data_table(["rank", "family", "columns", "share of summed |SHAP|", "grouped |SHAP|", "top feature"], rows,
                    widths=[0.4, 1.1, 0.6, 0.9, 0.8, 3.0],
                    caption="TreeSHAP importance of the tuned LightGBM on test_future, grouped by EMBER feature family "
                            "(reports/tables/shap_family_importance.csv). The summed share favours families with many "
                            "columns; the grouped measure (mean |sum of the family's SHAP|) does not.")
    s += figure(FIGURES / "shap_family_bar.png", "Mean |SHAP| by feature family (reports/figures/shap_family_bar.png).",
                width=0.6)
    s += figure(FIGURES / "shap_beeswarm.png", "SHAP summary (beeswarm) of the top features of the tuned model "
                "(reports/figures/shap_beeswarm.png).", width=0.9)
    cases = table_csv("shap_cases").set_index("case")
    c = cases.loc["tp"]
    s += figure(FIGURES / "shap_waterfall_tp.png", f"Local explanation of a true positive (week {int(c.week)}, "
                f"P(malicious) = {c.p:.4f}): SHAP waterfall in log-odds (reports/figures/shap_waterfall_tp.png).",
                width=0.72)
    # the two error types, one full-width waterfall each (side by side the labels were too small to read)
    for k, lab in (("fp", "a false positive, a benign file flagged"), ("fn", "a false negative, malware the model missed")):
        s += figure(FIGURES / f"shap_waterfall_{k}.png", f"Local explanation of {lab} (week "
                    f"{int(cases.loc[k, 'week'])}, P(malicious) = {cases.loc[k, 'p']:.4f}): SHAP waterfall in log-odds "
                    f"(reports/figures/shap_waterfall_{k}.png).", width=0.72)
    ag = table_csv("cross_tool_agreement")
    rows = [[name(r.tool_a), name(r.tool_b), f"{int(r.top5_overlap)} of 5", f"{int(r.top20_overlap)} of 20",
             f"{r.family_spearman:.2f}"] for _, r in ag.iterrows()]
    s += data_table(["model A", "model B", "top-5 overlap", "top-20 overlap", "family rank correlation"], rows,
                    widths=[1.3, 1.3, 0.8, 0.8, 1.0],
                    caption="Cross-tool agreement of feature importance (reports/tables/cross_tool_agreement.csv). "
                            "The measures differ by tool (SHAP, permutation importance, H2O varimp).")
    s += figure(FIGURES / "cross_tool_agreement.png", "Feature-family importance shares across the tuned model and "
                "the AutoML winners (reports/figures/cross_tool_agreement.png).", width=0.95)

    # ---- calibration and threshold
    s.append(H2("6.7 Calibration and cost-sensitive threshold"))
    cal = table_csv("calibration")
    rows = [[name(r.model) if r.model in NAMES else str(r.model), r.test_set, f4(r.brier), f4(r.log_loss), f4(r.ece),
             f4(r.roc_auc)] for _, r in cal.iterrows()]
    s += data_table(["model", "scored on", "Brier", "log loss", "ECE", "AUC"], rows, widths=[1.8, 0.9, 0.7, 0.7, 0.7, 0.7],
                    caption="Calibration (reports/tables/calibration.csv). Isotonic and Platt calibrators were fitted on "
                            "train out-of-fold scores.")
    s += figure(FIGURES / "calibration.png", "Reliability curves on test_future (reports/figures/calibration.png).",
                width=0.9)
    th = table_csv("threshold")
    rows = [[f"{r.cost_ratio:.0f}:1", f"{r.threshold:.4f}", pct(r.oof_recall), pct(r.oof_fpr, 2),
             pct(r.test_future_recall), pct(r.test_future_fpr, 2), f"{r.test_future_cost_per_1000:.1f}",
             f"{r['test_future_cost_per_1000_at_0.5']:.1f}"] for _, r in th.iterrows()]
    s += data_table(["cost FN:FP", "threshold (train OOF)", "recall OOF", "FPR OOF", "recall future", "FPR future",
                     "cost / 1000 future", "cost / 1000 at 0.5"], rows,
                    widths=[0.7, 0.9, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8],
                    caption="Cost-minimising thresholds chosen on train out-of-fold scores, checked on test_future "
                            "(reports/tables/threshold.csv).")
    t = F["threshold"]
    s.append(P(
        f"The deployed threshold is the {t['cost_ratio_fn_to_fp']:.0f}:1 one (t = {t['threshold']:.4f}, "
        f"models/threshold.json): a missed malware sample is assumed {t['cost_ratio_fn_to_fp']:.0f} times as costly "
        f"as a false alarm. On test_future it gives recall {pct(t['test_future_check']['recall'])} at FPR "
        f"{pct(t['test_future_check']['fpr'], 2)}. Caveat: it was chosen at 50/50 class balance and must be "
        f"re-derived for a deployment with another prevalence."))
    s += figure(FIGURES / "pr_cost.png", "Precision-recall trade-off and expected cost against the threshold "
                "(reports/figures/pr_cost.png).", width=0.95)

    # ---- drift
    s.append(H2("6.8 Drift over the 12 future weeks"))
    fx = F["facts4"]
    dr = table_csv("drift")
    weekly = dr[dr.week.astype(str).str.isdigit()]  # the table also has whole-test-set rows ("all_test_future")
    weeks = weekly.week.astype(int)
    exp = weekly[weekly.model == "expanding"]
    whole = dr[dr.week == "all_test_future"].set_index("model")
    s.append(P(
        f"The tuned configuration was retrained three ways and scored week by week on test_future: on all train "
        f"weeks (expanding, {fx['drift_train_rows']['expanding']:,} rows), on the last 12 training weeks only "
        f"(sliding, {fx['drift_train_rows']['sliding_12w']:,} rows) and on a random sample of the same size from all "
        f"weeks (size-matched). The weekly AUC slope of the expanding model is "
        f"{fx['drift_slopes_per_week']['roc_auc']['expanding']:+.5f} per week; its weekly AUC ranges from "
        f"{f4(exp.roc_auc.min())} to {f4(exp.roc_auc.max())} (weeks {weeks.min()}-{weeks.max()}). On the whole of "
        f"test_future the AUC is {f4(whole.loc['expanding', 'roc_auc'])} (expanding), "
        f"{f4(whole.loc['sliding_12w', 'roc_auc'])} (sliding) and {f4(whole.loc['size_matched', 'roc_auc'])} "
        f"(size-matched): at this short horizon more data beats recency. At equal size recent weeks may help a little, "
        f"but that is only suggestive: there is no paired CI on the sliding minus size-matched difference, and one seed."))
    s += figure(FIGURES / "drift.png", "Weekly ROC-AUC, F1 and TPR@1%FPR on test_future for the expanding, sliding "
                "12-week and size-matched models (reports/figures/drift.png).", width=1.0)

    # ---- challenge
    s.append(H2("6.9 The challenge set"))
    ch = table_csv("challenge")
    sub = ch[ch.subset == "all Win64 challenge"]
    rows = [[name(r.model), str(r.threshold_name), f"{r.threshold:.4f}", intc(r.n), intc(r.detected),
             f"{pct(r.detection_rate)} {cipct(r.ci_low, r.ci_high)}", pct(r.test_future_tpr_same_threshold)]
            for _, r in sub.iterrows()]
    n_ch = int(sub.n.iloc[0])
    s.append(P(
        f"EMBER2024's challenge set holds malware that antivirus engines missed when it was first seen. Its "
        f"{n_ch} Win64 files (all malicious, none in our splits) measure the hard tail that neither test set shows."))
    s += data_table(["model", "threshold", "t", "files", "detected", "detection rate [95% CI]",
                     "TPR on test_future at same t"], rows, widths=[1.6, 1.2, 0.6, 0.5, 0.75, 1.5, 1.0],
                    caption="Detection on the Win64 challenge set (reports/tables/challenge.csv).")
    s.append(H2("6.10 What protects, and the verdict"))
    # critical_eval.md's "section 4" (leakage) is section 6.3 of this report
    s += render_blocks([(k, t.replace("(section 4)", "(section 6.3)")) for k, t in md_section(ce, "6. What protects")])
    s += render_blocks(md_section(ce, "7. Verdict"))
    return s


def deployment(F: dict) -> list[Flowable]:
    """Section 7: the deployed model, why it was picked, the API and the demo screenshots."""
    sc = F["schema"]
    m = sc["metrics"]
    avm = table_csv("automl_vs_manual")
    # Paired test_future AUC differences against the tuned model: which AutoML tools are tied with it?
    pw = table_csv("automl_pairwise")
    vs = pw[(pw.test_set == "test_future") & (pw.metric == "roc_auc") & (pw.method_b == "lgbm_tuned")
            & pw.method_a.isin(avm.loc[avm.kind == "automl", "method"])]
    tied = vs[~vs.significant_95.astype(bool)]
    better = vs[vs.significant_95.astype(bool) & (vs.diff_a_minus_b > 0)]
    tie_txt = ("; ".join(f"{name(r.method_a)} {r.diff_a_minus_b:+.5f} [{r.ci_low:+.5f}, {r.ci_high:+.5f}]"
                         for _, r in tied.iterrows()) or "none")
    s = [H1("7. Step 5: deployment")]
    s.append(P(
        "<b>Which model and why.</b> The tuned LightGBM is deployed. Its configuration was fixed in Step 2 by CV AUC "
        "on train, before any test score existed. The final pick among the finished, frozen models is the one "
        "decision in this project that read test scores, so it is stated here: on test_future " +
        ("no AutoML tool was significantly better than the tuned model" if better.empty else
           "significantly better than the tuned model: " + ", ".join(name(x) for x in better.method_a)) +
        f", and the tools statistically tied with it (paired AUC difference, tool minus tuned, CI includes 0; "
        f"reports/tables/automl_pairwise.csv) were: {tie_txt}. The test scores therefore do not decide between the candidates. Practical reasons do: "
        f"the tuned LightGBM is one plain model file with no framework runtime (no Java for H2O, no AutoGluon "
        f"install), it can be explained exactly with TreeSHAP, and its threshold was fitted on its own train "
        f"out-of-fold scores. Nothing was trained, tuned or thresholded on a test set."))
    rows = [
        ["model version", sc["model_version"]],
        ["file / sha256", f"models/final/{sc['model_file']} / {sc['model_sha256'][:16]}..."],
        ["model", f"{sc['model_type']}, {sc['num_trees']} trees, LightGBM {sc['lightgbm_version']}"],
        ["input", f"{sc['feature_dim']:,} float32 thrember features (raw vector, no scaler needed)"],
        ["threshold", f"{sc['threshold']['threshold']:.6f} (cost ratio FN:FP = {sc['threshold']['cost_ratio_fn_to_fp']:.0f}, from models/threshold.json)"],
        ["export check", f"reproduces the saved Step 2 test_future scores on {sc['parity_check']['rows']:,} rows, "
                         f"max |diff| = {sc['parity_check']['max_abs_diff']:.1e}"],
        ["test_future", f"ROC-AUC {f4(m['test_future_roc_auc'])}, TPR@1%FPR {pct(m['test_future_tpr_at_1pct_fpr'])}, "
                        f"F1 {f4(m['test_future_f1'])}"],
    ]
    s += data_table(["item", "value"], rows, widths=[1, 4], caption="The exported model (models/final/schema.json).")
    s.append(P("<b>Serving.</b> app/api.py is a FastAPI service. It verifies the model file against the sha256 in "
               "schema.json before serving, and offers:"))
    s.append(bullets([
        "<font face='%s' size='8'>GET /health</font>: status, model version, sha256 and threshold;" % MONO_FONT,
        "<font face='%s' size='8'>POST /predict</font>: one 2,568-value feature vector (validated with pydantic: "
        "length, finite values) returns P(malicious), the label at the threshold and, on request, the top TreeSHAP "
        "contributions;" % MONO_FONT,
        "<font face='%s' size='8'>POST /predict_batch</font>: several vectors at once;" % MONO_FONT,
        "<font face='%s' size='8'>POST /predict_file</font>: upload a PE file; it is vectorised in memory with "
        "thrember and never stored or executed." % MONO_FONT,
    ]))
    s.append(P("app/streamlit_app.py (with app/story.py) is the presentation page: one page that reads top to bottom "
               "as the whole talk, for a non-specialist audience, in ten sections: the question, the data, the "
               "contenders, how they were compared, the results, a full scorecard of evaluation metrics, the winner "
               "and why, a live demo, what we learned, and the limits. Every number on it is read from reports/tables/. In the live demo the user picks an "
               "example (a caught malware file, a safe program, a missed malware file, a false alarm) or uploads a PE "
               "file; the API scores it, and the page shows the verdict and the four feature families that pushed "
               "the decision most (SHAP sums, in plain words). The raw score, the threshold and the top-10 SHAP "
               "columns are in a technical-detail box. present.py starts the API and the page together. MODEL_CARD.md (generated from the same tables) documents data, metrics, "
               "intended and out-of-scope uses and known limits. reports/talk_script.md is the presenter's script for the page (generated from the same tables) and DEMO.md the setup notes. Deployment is "
               "local; cloud hosting is left to the student's own account."))
    shots = [
        ("a_malicious_correct.png", "Streamlit demo: a malicious test_future file, correctly flagged, with the "
                                    "feature families that pushed the decision most."),
        ("b_benign_correct.png", "Streamlit demo: a benign test_future file, correctly passed."),
        ("c_upload_notepad.png", "Streamlit demo: uploading a benign Windows binary (notepad.exe); the file is parsed "
                                 "in memory by the API and never executed."),
    ]
    # one screenshot per figure: the live-demo section of the page (app/take_screenshots.py captures only it)
    for fname, cap in shots:
        s += figure_row([SHOTS / fname], f"{cap} (reports/screenshots/{fname}, the live-demo section of the page)",
                        max_h=17.5 * cm)
    return s


def reflections(F: dict) -> list[Flowable]:
    avm = table_csv("automl_vs_manual")
    am = avm[avm.kind == "automl"]
    tuned = row_of(avm, method="lgbm_tuned")
    lk = table_csv("leakage")
    dr = lk[lk.variant == "leak_detection_ratio"]
    se = table_csv("automl_self_estimate")
    clean = se[se.variant == "clean"]
    drift_head, drift_body = drift_vs_tuned_verdict()
    s = [H1("8. Reflections: what AutoML did well, and where it failed")]
    s.append(H2("8.1 What AutoML did well"))
    s.append(bullets([
        f"<b>Expert-level scores for little effort.</b> With one call and {F['budget']:.0f} s each, the three tools "
        f"reached test_future AUC {f4(am.test_future_roc_auc.min())} to {f4(am.test_future_roc_auc.max())}, against "
        f"{f4(tuned.test_future_roc_auc)} for the tuned LightGBM, whose three searches and refits took "
        f"{tuned.all_manual_seconds / 3600:.1f} h together.",
        "<b>Sensible model choices.</b> All three converged on gradient-boosted trees (LightGBM, XGBoost or H2O GBM), "
        "the family a human would pick for this data, and FLAML grew its sample sizes so that cheap trials came first.",
        "<b>Transparent artefacts.</b> Every tool left a leaderboard, a log and importances that could be audited, "
        "which made the leakage and agreement analyses possible.",
        f"<b>{drift_head}</b> {drift_body}; the winners are the same kind of model as the tuned one "
        "(reports/tables/gap_vs_tuned.csv).",
    ]))
    s.append(H2("8.2 Where AutoML failed or misled"))
    s.append(bullets([
        f"<b>It fell for the label leak.</b> With detection_ratio added, {int(dr.relied_on_leak.sum())} of {len(dr)} "
        "models ranked it first and scored a near-perfect AUC; no tool warned or dropped the column. An optimiser for "
        "validation score cannot know that a column is only available after the fact.",
        f"<b>Its own validation was a random split.</b> By default every tool validated on random rows of train (the "
        f"same weeks), so its own estimate was above test_future AUC in "
        f"{int((clean.own_minus_future > 0).sum())} of {len(clean)} clean runs, and it reported 1.0000 with the leak.",
        "<b>Budgets are soft.</b> FLAML's refit and H2O's ensemble ran minutes past the budget, so 'equal budget' "
        "is equal search time, not equal compute.",
        "<b>Hardware sensitivity and repeatability.</b> On the 4-core VM the same tools searched fewer models in "
        "600 s; H2O's test_future AUC dropped noticeably. A time-budgeted search is not exactly repeatable, even with "
        "a seed.",
        "<b>Interpretability depends on the winner.</b> FLAML and AutoGluon ended with one tree model, but H2O "
        "returned a stacked ensemble whose metalearner used one base model; explaining it took extra work.",
    ]))
    s.append(H2("8.3 What we would do differently"))
    s.append(bullets([
        "Give each AutoML tool a time-ordered validation set (FLAML split_type='time', an H2O fold column, "
        "AutoGluon tuning_data) and compare it with the defaults.",
        "Review the provenance of every column before training; this is the only thing that stopped the label leak.",
        "Evaluate at a realistic prevalence and on a hard stress set from the start, not only on 50/50 test splits.",
        "Repeat the AutoML runs with several seeds on one machine to measure their run-to-run variation.",
    ]))
    return s


def limitations(F: dict) -> list[Flowable]:
    lim = REPORTS / "limitations.md"
    blocks = md_blocks(lim.read_text(encoding="utf-8"))
    body = [b for b in blocks if b[0] != "h1"]
    body = [b for b in body if not b[1].startswith("Generated by")]
    s = [H1("9. Limitations")]
    s.append(P("This section is taken from reports/limitations.md, which scripts/run_critical_eval.py generates from "
               "the data checks and the Step 4 tables."))
    s += render_blocks([("h3" if k == "h2" else k, t) for k, t in body])
    return s


def reproducibility(F: dict) -> list[Flowable]:
    s = [H1("10. Reproducibility")]
    s.append(P(f"All code is in {REPO_URL}. One seed (42) is used for numpy, random, every model library and every "
               "split. Splits and CV folds are persisted under data/splits/. Every run appends a record (method, "
               "parameters, scores, time; Step 4 records also log the machine) to reports/experiments.jsonl, and every table, figure and "
               "write-up is regenerated from that log and the saved models."))
    s.append(H2("10.1 Hardware"))
    s.append(bullets([
        "<b>Steps 2-3 and Step 5</b> (tuning, AutoML, export and app): the student's laptop, 16-core Intel Core "
        "Ultra 9 285H, 63 GB RAM, Windows 11, CPU only, 16 threads.",
        f"<b>Step 4</b> (critical evaluation): a 4-core, {vm_ram_gib():.1f} GiB RAM Linux cloud VM. The leakage "
        "runs used a 600 s "
        "AutoML budget and memory-saving runner settings, identical for every with/without pair. The challenge-set "
        "scoring was re-run on the laptop (no fitting).",
        "Timings are only compared within one step (one machine).",
    ]))
    s.append(H2("10.2 Commands"))
    cmds = [
        "python run_all.py --quick                  # smoke run on tiny samples (*_quick files)",
        "python run_all.py                          # full run: data, tune, automl, eval",
        "python scripts/run_tuning.py --tables-only # rebuild Step 2 tables, figures, write-up",
        "python scripts/run_automl.py --tables-only # rebuild Step 3 tables, figure, write-up",
        "python scripts/run_leakage.py --tables-only",
        "python scripts/run_critical_eval.py        # Step 4 synthesis and limitations",
        "python scripts/export_model.py             # Step 5 export + model card",
        "python scripts/build_report.py             # this PDF (a few seconds)",
        "python -m pytest -q                        # tests on splits, leakage, API",
    ]
    s.append(Paragraph("<br/>".join(c.replace(" ", "&nbsp;") for c in cmds), ST["code"]))
    s.append(P("Here python is the project's virtual environment interpreter (.venv/Scripts/python.exe on Windows, "
               ".venv/bin/python on Linux). Python 3.12; all packages are pinned in requirements.txt (LightGBM, Optuna, "
               "FLAML, AutoGluon, H2O, SHAP, FastAPI, Streamlit, reportlab). Quick runs set EMBER_QUICK=1 and never "
               "overwrite full results."))
    avm = table_csv("automl_vs_manual")
    rows = [[name(r.method), str(r.tool_version)] for _, r in avm[avm.kind == "automl"].iterrows()]
    rows.append(["LightGBM (manual, deployed)", F["schema"]["lightgbm_version"]])
    s += data_table(["tool", "version"], rows, widths=[2, 1], caption="Tool versions logged with the runs.")
    return s


def references() -> list[Flowable]:
    refs = [
        "R. J. Joyce, G. Miller, P. Roth, R. Zak, E. Zaresky-Williams, H. Anderson, E. Raff, J. Holt. "
        "<i>EMBER2024 - A Benchmark Dataset for Holistic Evaluation of Malware Classifiers.</i> KDD 2025. "
        "arXiv:2506.05074.",
        "H. S. Anderson, P. Roth. <i>EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models.</i> "
        "arXiv:1804.04637, 2018.",
        "G. Ke et al. <i>LightGBM: A Highly Efficient Gradient Boosting Decision Tree.</i> NeurIPS 2017.",
        "T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama. <i>Optuna: A Next-generation Hyperparameter Optimization "
        "Framework.</i> KDD 2019.",
        "C. Wang, Q. Wu, M. Weimer, E. Zhu. <i>FLAML: A Fast and Lightweight AutoML Library.</i> MLSys 2021.",
        "N. Erickson et al. <i>AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.</i> arXiv:2003.06505, 2020.",
        "E. LeDell, S. Poirier. <i>H2O AutoML: Scalable Automatic Machine Learning.</i> 7th ICML Workshop on AutoML, 2020.",
        "S. M. Lundberg, S.-I. Lee. <i>A Unified Approach to Interpreting Model Predictions.</i> NeurIPS 2017.",
        "S. M. Lundberg et al. <i>From local explanations to global understanding with explainable AI for trees.</i> "
        "Nature Machine Intelligence 2, 2020.",
        "F. Pedregosa et al. <i>Scikit-learn: Machine Learning in Python.</i> JMLR 12, 2011.",
        "M. Mitchell et al. <i>Model Cards for Model Reporting.</i> FAT* 2019.",
        "F. Pendlebury, F. Pierazzi, R. Jordaney, J. Kinder, L. Cavallaro. <i>TESSERACT: Eliminating Experimental "
        "Bias in Malware Classification across Space and Time.</i> USENIX Security 2019.",
        "FastAPI (https://fastapi.tiangolo.com) and Streamlit (https://streamlit.io), software used for serving.",
    ]
    s = [H1("11. References")]
    # one KeepTogether block (the list fits on a page), so glue_headings keeps the heading with it
    s.append(KeepTogether([ListFlowable([ListItem(Paragraph(r, ST["bullet"]), leftIndent=24) for r in refs],
                                        bulletType="1", bulletFormat="[%s]", leftIndent=24,
                                        bulletFontName=BODY_FONT, bulletFontSize=8.5)]))
    return s


def appendix(F: dict) -> list[Flowable]:
    s = [H1("Appendix")]
    s.append(H2("A. Full AutoML leaderboards"))
    lb = pd.read_csv(REPORTS / "leaderboards" / "flaml_leaderboard.csv")
    rows = [[str(int(r["rank"])), str(r.estimator), str(int(r.n_trials)), f4(r.best_val_auc), intc(r.best_sample_size),
             f"{r.best_found_at_s:.0f} s", f"{r.total_trial_time_s:.0f} s"] for _, r in lb.iterrows()]
    s += data_table(["rank", "estimator", "trials", "best val. AUC", "rows (best)", "found at", "time in trials"], rows,
                    caption="FLAML leaderboard (reports/leaderboards/flaml_leaderboard.csv).")
    lb = pd.read_csv(REPORTS / "leaderboards" / "autogluon_leaderboard.csv")
    rows = [[str(r.model), f4(r.score_val), f"{r.fit_time:.0f} s", f"{r.pred_time_val:.2f} s", str(int(r.stack_level)),
             str(int(r.num_features))] for _, r in lb.iterrows()]
    s += data_table(["model", "val. AUC", "fit time", "predict time (val.)", "stack level", "features"], rows,
                    widths=[1.6, 0.7, 0.7, 0.9, 0.7, 0.7], caption="AutoGluon leaderboard (reports/leaderboards/autogluon_leaderboard.csv).")
    lb = pd.read_csv(REPORTS / "leaderboards" / "h2o_leaderboard.csv")
    rows = [[str(r.model_id).split("_AutoML")[0], str(r.algo), f4(r.auc), f4(r.logloss), f4(r.aucpr),
             f"{r.training_time_ms / 1000:.0f} s"] for _, r in lb.iterrows()]
    s += data_table(["model", "algorithm", "AUC", "log loss", "AUC-PR", "training time"], rows,
                    widths=[2.2, 1, 0.7, 0.7, 0.7, 0.8],
                    caption="H2O AutoML leaderboard (reports/leaderboards/h2o_leaderboard.csv).")
    s.append(H2("B. Tuned hyperparameters"))
    rows = []
    for m in ("grid", "random", "optuna"):
        p = load_json(ROOT / "models" / f"lgbm_{m}_params.json")
        keep = ["num_leaves", "learning_rate", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l2"]
        vals = [f"{p['params'][k]:.4g}" if k in p["params"] else "-" for k in keep]
        rows.append([name(m)] + vals + [str(p.get("num_trees", "-"))])
    s += data_table(["tuner", "leaves", "learning rate", "min child samples", "feature fraction", "bagging fraction",
                     "lambda l2", "trees"], rows, widths=[1.5, 0.8, 0.9, 1, 0.9, 0.9, 0.9, 0.6],
                    caption="Best configuration of each tuner, refit on all of train (models/lgbm_*_params.json; "
                            "LightGBM names num_leaves, learning_rate, min_child_samples, feature_fraction, "
                            "bagging_fraction, lambda_l2).")
    s.append(H2("C. Experiment log"))
    counts: dict[str, int] = {}
    for line in (REPORTS / "experiments.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            st = json.loads(line).get("step", "?")
            counts[st] = counts.get(st, 0) + 1
    rows = [[k, str(v)] for k, v in counts.items()]
    s += data_table(["step", "records"], rows, widths=[1, 1],
                    caption="Records in reports/experiments.jsonl by step (every tuning trial, AutoML run, leakage run "
                            "and evaluation job).")
    s.append(H2("D. Where to find everything"))
    s.append(bullets([
        "Write-ups: reports/tuning_writeup.md, automl_comparison.md, leakage.md, temporal_bias.md, explainability.md, "
        "critical_eval.md, limitations.md.",
        "Tables: reports/tables/*.csv; figures: reports/figures/*.png; leaderboards: reports/leaderboards/.",
        "Models: models/ (Step 2 LightGBM files, AutoML folders), models/final/ (deployed), models/threshold.json.",
        "App: app/api.py, app/streamlit_app.py; MODEL_CARD.md; DEMO.md.",
    ]))
    return s


def toc_page() -> list[Flowable]:
    toc = TableOfContents()
    toc.levelStyles = TOC_STYLES
    return [Paragraph("Contents", ST["h1"].clone("toch", textColor=colors.HexColor("#1f3b5c"))), toc]


def glue_headings(story: list[Flowable]) -> list[Flowable]:
    """Put each heading inside the KeepTogether block that follows it, so a heading is never left
    alone at the bottom of a page above a table or figure that moved to the next page."""
    out: list[Flowable] = []
    for fl in story:
        heads: list[Flowable] = []
        while out and isinstance(out[-1], Paragraph) and out[-1].style.name in ("h1", "h2", "h3"):
            heads.insert(0, out.pop())
        if heads and isinstance(fl, KeepTogether):
            out.append(KeepTogether(heads + list(fl._content)))
        else:
            out.extend(heads)
            out.append(fl)
    return out


def build() -> None:
    facts = load_facts()
    draw_architecture(facts)
    story: list[Flowable] = []
    story += title_page(facts)
    from reportlab.platypus.doctemplate import NextPageTemplate
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    story += toc_page()
    story.append(PageBreak())
    sections: list[Callable[[], list[Flowable]]] = [
        lambda: abstract(facts), lambda: problem(facts), lambda: dataset(facts), lambda: methodology(facts),
        lambda: tuning(facts), lambda: automl(facts), lambda: critical(facts), lambda: deployment(facts),
        lambda: reflections(facts), lambda: limitations(facts), lambda: reproducibility(facts), references,
        lambda: appendix(facts),
    ]
    for i, sec in enumerate(sections):
        story += sec()
        if i in (1, 5, 6, 11):  # new page before the big sections
            story.append(PageBreak())
    story = glue_headings(story)
    doc = Doc(str(OUT_PDF), title="Malware detection on EMBER2024: manual tuning vs AutoML",
              author=f"{STUDENT} ({REG_NO})", subject="IA-3 Part B AutoML project report")
    doc.multiBuild(story)
    print(f"[report] wrote {OUT_PDF.relative_to(ROOT)} ({OUT_PDF.stat().st_size / 1e6:.1f} MB), "
          f"{NUM.fig} figures, {NUM.tab} tables")
    if MISSING:
        print(f"[report] WARNING: placeholders for missing files: {', '.join(MISSING)}")


if __name__ == "__main__":
    build()
