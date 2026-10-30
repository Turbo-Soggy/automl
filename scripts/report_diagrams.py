"""UML diagrams for the skill-based evaluation report (scripts/build_skill_report.py).

Draws a class diagram (the project's modules as component classes) and a sequence diagram
(one prediction through the presentation page and the API) with matplotlib, in the same
light-purple box style as the sample report. The names in the boxes are the real module,
class and function names of this repository.

  .venv/Scripts/python.exe scripts/report_diagrams.py     # writes reports/figures/uml_*.png
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
CLASS_PNG = FIGURES / "uml_class.png"
SEQUENCE_PNG = FIGURES / "uml_sequence.png"

FILL, EDGE, TEXT = "#ECECFF", "#9370DB", "#222222"
NOTE_FILL, NOTE_EDGE = "#FFF5AD", "#AAAA33"
FONT = "DejaVu Sans"


@dataclass
class Box:
    """One class box: title, attribute lines and method lines, at (x, y) = top-left, in data units."""
    title: str
    attrs: list[str]
    methods: list[str]
    x: float
    y: float
    w: float
    stereotype: str = ""

    LINE = 2.1   # height of one text line
    PAD = 1.2

    @property
    def h(self) -> float:
        head = self.LINE * (2 if self.stereotype else 1) + self.PAD
        return head + self.LINE * (len(self.attrs) + len(self.methods)) + 3 * self.PAD

    def anchor(self, side: str) -> tuple[float, float]:
        """Middle of one side of the box: top, bottom, left or right."""
        return {"top": (self.x + self.w / 2, self.y), "bottom": (self.x + self.w / 2, self.y - self.h),
                "left": (self.x, self.y - self.h / 2), "right": (self.x + self.w, self.y - self.h / 2)}[side]


def draw_box(ax: plt.Axes, b: Box) -> None:
    ax.add_patch(Rectangle((b.x, b.y - b.h), b.w, b.h, facecolor=FILL, edgecolor=EDGE, linewidth=1.2))
    y = b.y - b.PAD
    if b.stereotype:
        ax.text(b.x + b.w / 2, y - b.LINE / 2, f"«{b.stereotype}»", ha="center", va="center", fontsize=7,
                family=FONT, color=TEXT)
        y -= b.LINE
    ax.text(b.x + b.w / 2, y - b.LINE / 2, b.title, ha="center", va="center", fontsize=8.5, weight="bold",
            family=FONT, color=TEXT)
    y -= b.LINE + b.PAD
    for lines in (b.attrs, b.methods):   # two compartments, each under a divider line
        ax.plot([b.x, b.x + b.w], [y, y], color=EDGE, linewidth=0.8)
        y -= b.PAD / 2
        for line in lines:
            ax.text(b.x + 0.8, y - b.LINE / 2, line, ha="left", va="center", fontsize=7, family=FONT, color=TEXT)
            y -= b.LINE
        y -= b.PAD / 2


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], label: str = "",
          dashed: bool = False, rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color="#555555", linewidth=0.9,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.5, family=FONT, color=TEXT,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})


def class_diagram(path: Path = CLASS_PNG) -> Path:
    """The modules of the project as component classes, in five rows from configuration to serving."""
    fig, ax = plt.subplots(figsize=(8.6, 10.0), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(-11, 108)
    ax.axis("off")
    cfg = Box("config", ["+SEED = 42", "+TUNE_TRIALS, CV_FOLDS", "+AUTOML_BUDGET_S", "+N_BOOTSTRAP, N_JOBS"],
              ["+set_seed()"], 37, 107, 26)
    data = Box("src.data", ["+feature cache (X, meta)", "+splits, CV folds"],
               ["+build_cache()", "+load_split(name)", "+feature_groups()"], 35, 85, 30)
    tune = Box("src.tune", ["+Grid / Random / Optuna", "+TuningData, SearchResult"],
               ["+run_baselines()", "+run_search(method)", "+refit_and_evaluate()"], 1, 62, 30)
    automl = Box("src.automl_runs", ["+FLAML, AutoGluon, H2O", "+Heartbeat (sleep guard)"],
                 ["+run_flaml(budget)", "+run_autogluon(budget)", "+run_h2o(budget)", "+evaluate_and_log()"], 35, 62, 30)
    evalb = Box("Step 4 evaluation", ["src.leakage, temporal, bias,", "drift, explain, calibration"],
                ["+run_lgbm(variant), run_tool()", "+random_vs_future_table()", "+lgbm_shap()",
                 "+choose_threshold()"], 69, 62, 30)
    export = Box("scripts.export_model", ["+models/final/ (model.txt,", " schema.json, demo samples)"],
                 ["+export()", "+parity_check()"], 1, 36, 30)
    log = Box("src.logging_utils / src.eval", ["+reports/experiments.jsonl"],
              ["+log_experiment(record)", "+compute_metrics()", "+paired_bootstrap()"], 35, 36, 30)
    api = Box("FastAPI service (app.api)", ["+FeatureVector (2,568 finite)"],
              ["+GET /health", "+POST /predict", "+POST /predict_batch", "+POST /predict_file"], 69, 36, 30)
    pred = Box("Predictor", ["-booster: lightgbm.Booster", "-threshold: float", "-families: array"],
               ["+predict(X)", "+shap(x)"], 37, 12, 26, stereotype="app.api")
    page = Box("Streamlit page", ["app.streamlit_app + app.story", "+nine presentation sections"],
               ["+live_demo()", "+call(features | file)", "+results(), winner()"], 69, 9, 30)
    for b in (cfg, data, tune, automl, evalb, export, log, api, pred, page):
        draw_box(ax, b)
    arrow(ax, data.anchor("top"), cfg.anchor("bottom"), "reads", dashed=True)
    arrow(ax, tune.anchor("top"), data.anchor("left"), "loads splits", dashed=True)
    arrow(ax, automl.anchor("top"), data.anchor("bottom"), "loads splits", dashed=True)
    arrow(ax, evalb.anchor("top"), data.anchor("right"), "loads splits", dashed=True)
    arrow(ax, tune.anchor("bottom"), export.anchor("top"), "tuned model")
    arrow(ax, (tune.x + 24, tune.y - tune.h), (log.x + 6, log.y), "logs every trial")
    arrow(ax, automl.anchor("bottom"), (log.x + 15, log.y), "logs scores")
    arrow(ax, export.anchor("bottom"), pred.anchor("left"), "saves")
    arrow(ax, page.anchor("top"), api.anchor("bottom"), "calls (HTTP)")
    arrow(ax, (api.x, api.y - api.h + 3), (pred.x + pred.w, pred.y - 4), "uses")
    fallback_y = pred.y - pred.h + 4
    arrow(ax, (page.x, fallback_y), (pred.x + pred.w, fallback_y), dashed=True)
    ax.text(page.x - 3, fallback_y - 1.2, "in-process\nfallback", ha="center", va="top", fontsize=6.5, family=FONT,
            color=TEXT)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def sequence_diagram(path: Path = SEQUENCE_PNG) -> Path:
    """One prediction: the user picks a file on the page, the API validates and scores it with SHAP."""
    actors = ["User", "Streamlit page", "FastAPI service", "Predictor (LightGBM)"]
    xs = [8, 36, 64, 90]
    fig, ax = plt.subplots(figsize=(8.6, 9.0), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    for name, x in zip(actors, xs):
        for y in (96, 2):
            ax.add_patch(Rectangle((x - 9, y - 2.2), 18, 4.4, facecolor=FILL, edgecolor=EDGE, linewidth=1.1))
            ax.text(x, y, name, ha="center", va="center", fontsize=7.5, family=FONT, color=TEXT)
        ax.plot([x, x], [4.2, 93.8], color=EDGE, linewidth=0.9)
    ax.add_patch(Rectangle((27, 89.5), 72, 3, facecolor=NOTE_FILL, edgecolor=NOTE_EDGE, linewidth=0.8))
    ax.text(63, 91, "The API loads models/final/model.txt and checks its sha256 against schema.json at start",
            ha="center", va="center", fontsize=6.5, family=FONT, color=TEXT)

    def msg(y: float, a: int, b: int, text: str, reply: bool = False) -> None:
        ax.add_patch(FancyArrowPatch((xs[a], y), (xs[b], y), arrowstyle="-|>", mutation_scale=9, color="#333333",
                                     linewidth=0.9, linestyle="--" if reply else "-"))
        ax.text((xs[a] + xs[b]) / 2, y + 1.1, text, ha="center", va="bottom", fontsize=6.5, family=FONT, color=TEXT)

    def self_msg(y: float, a: int, text: str) -> None:
        x = xs[a]
        ax.add_patch(FancyArrowPatch((x, y), (x, y - 2.4), arrowstyle="-|>", mutation_scale=8, color="#333333",
                                     linewidth=0.9, connectionstyle="arc3,rad=-1.6"))
        ax.text(x - 1.5, y + 0.6, text, ha="right", va="bottom", fontsize=6.5, family=FONT, color=TEXT)

    msg(85, 0, 1, "Open the page")
    self_msg(81.5, 1, "Read reports/tables/*.csv for sections 1-6 and 8")
    msg(75, 0, 1, "Pick an example file (or upload a .exe)")
    msg(70, 1, 2, "POST /predict?top_k=10  {features: 2,568 numbers}")
    self_msg(66.5, 2, "Validate: length 2,568, all finite (else 422)")
    msg(60, 2, 3, "predict(x), shap(x)")
    msg(55, 3, 2, "P(malicious), SHAP values", reply=True)
    self_msg(51.5, 2, "Apply the threshold; sum SHAP by feature family")
    msg(45, 2, 1, "JSON: p_malicious, verdict, threshold, contributions", reply=True)
    msg(40, 1, 0, "Verdict card, flag line, top-4 reasons in plain words", reply=True)
    ax.add_patch(Rectangle((2, 7.5), 96, 27.5, facecolor="none", edgecolor="#777777", linewidth=0.8, linestyle=":"))
    ax.text(3, 33.6, "alt  [upload, or no API server]", fontsize=6.5, family=FONT, color=TEXT, va="center")
    msg(29, 1, 2, "POST /predict_file  (file bytes, at most 50 MB)")
    self_msg(25.5, 2, "Parse the PE in memory with thrember; never saved or run")
    msg(19, 2, 1, "Same JSON, plus sha256 and warnings", reply=True)
    self_msg(14.5, 1, "No API: score in-process with app.api.score_vector()")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for p in (class_diagram(), sequence_diagram()):
        print(f"[diagrams] wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
