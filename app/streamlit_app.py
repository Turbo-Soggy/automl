"""Step 5: the presentation page for the malware detector served by app/api.py.

One page that reads top to bottom as the whole talk, for an audience that knows nothing about
malware detection or AutoML, in nine numbered sections: the question, the data, the contenders,
how they were compared, the results, the winner and why, a live demo, what we learned, and what
to keep in mind. Sections other than the live demo live in app/story.py and read every number
from the saved results in reports/. The live demo sends a file to the API and shows the verdict
and the feature families that pushed it (SHAP); the raw numbers are in a "Technical detail" box.

Run from the project root (or use present.py, which also starts the API):
  streamlit run app/streamlit_app.py
The page calls the API at env EMBER_API_URL (default http://127.0.0.1:8000). With
EMBER_IN_PROCESS=1, or when the API cannot be reached, it scores inside this process with the
same functions instead (one-process deployments such as Streamlit Cloud).

test_future files come from the feature cache when it exists (all 40,000 rows, with the Step 2
scores used to pick examples), otherwise, or when EMBER_DEMO_SAMPLES=1, from
models/final/demo_samples.npz (a few hundred rows).
"""
from __future__ import annotations

import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # streamlit puts app/ on sys.path, not the project root

from app import api, story  # noqa: E402
from app.story import FAMILY_TEXT, step  # noqa: E402

API_URL = os.environ.get("EMBER_API_URL", "http://127.0.0.1:8000").rstrip("/")
TOP_K = 10
TOP_FAMILIES = 4   # reasons shown in plain words; more than this is hard to follow live

CHOICES: dict[str, str] = {
    "Caught malware": "a malware file the model flags",
    "Safe program": "a safe program the model lets through",
    "Missed malware": "a malware file the model misses",
    "False alarm": "a safe program the model flags by mistake",
    "Upload a file": "your own Windows program",
}

CSS = """
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {display: none;}
.block-container {max-width: 860px; padding-top: 2.2rem;}
.lead {font-size: 1.2rem; opacity: .85; margin-bottom: .4rem;}
.step {display: flex; align-items: center; gap: .7rem; margin: 2.2rem 0 .6rem;}
.step .n {flex: none; width: 2.1rem; height: 2.1rem; border-radius: 50%; background: #2563eb; color: #fff;
          display: flex; align-items: center; justify-content: center; font-weight: 700;}
.step .t {font-size: 1.55rem; font-weight: 700;}
.stats {display: flex; gap: 1rem; flex-wrap: wrap;}
.stat {flex: 1 1 200px; border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 1rem 1.1rem;}
.stat .big {font-size: 2.3rem; font-weight: 800; line-height: 1.1;}
.stat .lbl {opacity: .8;}
.verdict {border-radius: 14px; padding: 1.3rem 1.5rem; border: 2px solid;}
.verdict.bad {background: rgba(220,38,38,.12); border-color: #dc2626;}
.verdict.good {background: rgba(22,163,74,.12); border-color: #16a34a;}
.verdict .head {font-size: 2rem; font-weight: 800;}
.verdict .score {font-size: 1.15rem; margin-top: .3rem;}
.gauge {position: relative; height: 14px; border-radius: 7px; background: rgba(128,128,128,.25); margin: .9rem 0 .3rem;}
.gauge .fill {height: 100%; border-radius: 7px;}
.gauge .line {position: absolute; top: -6px; width: 4px; height: 26px; border-radius: 2px; background: currentColor;
              box-shadow: 0 0 0 2px rgba(128,128,128,.6);}
.verdict .truth {font-size: 1.15rem; margin-top: .8rem; font-weight: 600;}
.reason {display: grid; grid-template-columns: 1fr 180px; gap: .2rem 1rem; align-items: center;
         padding: .7rem 0; border-bottom: 1px solid rgba(128,128,128,.2);}
.reason .name {font-weight: 700; font-size: 1.1rem;}
.reason .what {opacity: .75; grid-column: 1;}
.reason .bar {grid-row: 1 / span 2; grid-column: 2; display: flex; flex-direction: column; gap: .25rem;}
.reason .bar .track {height: 12px; border-radius: 6px; background: rgba(128,128,128,.2);}
.reason .bar .val {height: 100%; border-radius: 6px;}
.reason .bar .dir {font-size: .9rem; font-weight: 600;}
.keep li {margin-bottom: .35rem; font-size: 1.05rem;}
.timeline {display: flex; gap: 6px; margin: 1rem 0 1.2rem; font-size: .92rem;}
.timeline .seg {border-radius: 10px; padding: .6rem .8rem; color: #fff; line-height: 1.35;}
.timeline .past {background: #2563eb;}
.timeline .future {background: #7c3aed;}
.chips {display: flex; flex-wrap: wrap; gap: .4rem;}
.chip {border: 1px solid rgba(128,128,128,.45); border-radius: 999px; padding: .25rem .75rem; font-size: .95rem;}
.teams {display: flex; gap: 1rem; flex-wrap: wrap;}
.team {flex: 1 1 300px; border: 2px solid; border-radius: 14px; padding: .9rem 1.1rem;}
.team .tt {font-size: 1.2rem; font-weight: 800; margin-bottom: .4rem;}
.team p, .team li {margin-bottom: .3rem;}
.team .small {font-size: .92rem; opacity: .85;}
.metric {border-left: 5px solid #2563eb; background: rgba(37,99,235,.08); padding: .8rem 1rem; border-radius: 6px;
         font-size: 1.1rem; margin: .4rem 0 .3rem;}
.winner {font-size: 1.6rem; font-weight: 800; margin: .2rem 0 .6rem;}
.finding {border: 1px solid rgba(128,128,128,.35); border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .8rem;}
.finding .ft {font-weight: 800; font-size: 1.15rem; margin-bottom: .3rem;}
/* The example picker as large pills: the presenter clicks them in front of an audience. */
div[role="radiogroup"] {gap: .5rem; flex-wrap: wrap;}
div[role="radiogroup"] label {border: 1px solid rgba(128,128,128,.45); border-radius: 999px; padding: .45rem 1rem;
                              margin: 0;}
div[role="radiogroup"] label p {font-size: 1.05rem;}
div[role="radiogroup"] label:has(input:checked) {border-color: #2563eb; background: rgba(37,99,235,.18);}
</style>
"""


# --- Data -------------------------------------------------------------------------------------

@st.cache_resource
def schema() -> dict[str, Any]:
    return json.loads((api.model_dir() / "schema.json").read_text(encoding="utf-8"))


@st.cache_resource
def test_future() -> dict[str, Any]:
    """X, y, offline score p and test_future position of every selectable file, plus the source."""
    cache = ROOT / "data" / "cache" / "X_test.npy"
    if cache.exists() and os.environ.get("EMBER_DEMO_SAMPLES") != "1":
        from src import data
        X, y, meta = data.load_split("test_future")
        saved = np.load(ROOT / schema()["parity_check"]["against"].split(" ")[0], allow_pickle=False)
        return {"X": X, "y": y, "p": saved["test_future_p"], "position": np.arange(len(y)),
                "sha256": meta["sha256"].to_numpy(), "week": meta["week"].to_numpy(),
                "source": f"feature cache: all {len(y):,} test_future files"}
    demo = np.load(api.model_dir() / "demo_samples.npz", allow_pickle=False)
    out = {k: demo[k] for k in ("X", "y", "p", "position", "sha256", "week")}
    out["source"] = f"models/final/demo_samples.npz: {len(out['y'])} test_future files (no feature cache here)"
    return out


@st.cache_resource
def deployed_rates() -> dict[str, float] | None:
    """Recall and false-alarm rate on test_future at the deployed threshold (reports/tables/threshold.csv).

    None when the table is not shipped (a slim deployment with only app/ and models/final/).
    """
    path = ROOT / "reports" / "tables" / "threshold.csv"
    if not path.exists():
        return None
    target = float(schema()["threshold"]["threshold"])
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if abs(float(row["threshold"]) - target) < 1e-12:
                tp, fn = float(row["test_future_tp"]), float(row["test_future_fn"])
                fp, tn = float(row["test_future_fp"]), float(row["test_future_tn"])
                return {"recall": tp / (tp + fn), "fpr": fp / (fp + tn), "n": tp + fn + fp + tn}
    return None


def pick_example(choice: str, k: int) -> int | None:
    """Row of the k-th test_future file of the chosen kind (outcome judged by the Step 2 scores)."""
    tf = test_future()
    flagged = tf["p"] >= float(schema()["threshold"]["threshold"])
    malware = tf["y"] == 1
    mask = {"Caught malware": malware & flagged, "Safe program": ~malware & ~flagged,
            "Missed malware": malware & ~flagged, "False alarm": ~malware & flagged}[choice]
    rows = np.flatnonzero(mask)
    return int(rows[k % len(rows)]) if len(rows) else None


# --- Scoring: through the API, or in-process when there is no API -----------------------------

@st.cache_data(ttl=30, show_spinner=False)
def api_reachable(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=3).json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def call(in_process: bool, *, features: np.ndarray | None = None, file: bytes | None = None) -> dict:
    """POST to /predict or /predict_file with top_k=10; in-process mode calls the same functions."""
    if in_process:
        result = api.score_vector(features, TOP_K) if file is None else api.score_file(file, TOP_K)
        return result.model_dump(exclude_none=True)
    if file is None:
        r = requests.post(f"{API_URL}/predict", params={"top_k": TOP_K},
                          json={"features": features.astype(float).tolist()}, timeout=60)
    else:
        r = requests.post(f"{API_URL}/predict_file", params={"top_k": TOP_K}, data=file,
                          headers={"content-type": "application/octet-stream"}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"API returned {r.status_code}: {r.text[:300]}")
    return r.json()


# --- The live demo (section 7) ---------------------------------------------------------------

def how_good() -> None:
    """The deployed model's operating point, before the audience tries it."""
    rates = deployed_rates()
    if rates is None:   # slim deployment without reports/tables
        return
    story.cards([(f"{rates['recall']:.1%}", "of future malware caught"),
                 (f"{rates['fpr']:.1%}", "of safe programs flagged for a second look")])
    st.caption("This is the model as deployed, at a deliberately low flag line: it misses as little malware as "
               "possible and accepts more false alarms in return (section 5 compared models at 1 false alarm "
               "in 100).")


def score_text(p: float) -> str:
    """P(malicious) as a score out of 100; the extremes are not rounded to a misleading 0 or 100."""
    if p > 0.999:
        return "over 99.9"
    if p < 0.001:
        return "under 0.1"
    return f"{100 * p:.1f}"


def verdict_card(res: dict, true_label: int | None) -> None:
    p, threshold = res["p_malicious"], res["threshold"]
    bad = res["verdict"] == "malicious"
    colour = "#dc2626" if bad else "#16a34a"
    head = "Flagged as malware" if bad else "Looks safe"
    if true_label is None:
        truth = "The true answer is not known for an uploaded file."
    else:
        actual = "malware" if true_label == 1 else "a safe program"
        right = (true_label == 1) == bad
        truth = f"Actually: {actual}. " + ("✔ The model was right." if right else "✘ The model got this one wrong.")
    st.markdown(
        f'<div class="verdict {"bad" if bad else "good"}">'
        f'<div class="head">{"⚠" if bad else "✔"} {head}</div>'
        f'<div class="score">Malware score <b>{score_text(p)}</b> out of 100. '
        f'Anything above <b>{100 * threshold:.1f}</b> is flagged.</div>'
        f'<div class="gauge"><div class="fill" style="width:{max(100 * p, 1):.1f}%;background:{colour}"></div>'
        f'<div class="line" style="left:{100 * threshold:.2f}%"></div></div>'
        f'<div class="truth">{html.escape(truth)}</div></div>', unsafe_allow_html=True)
    st.caption("The flag line is low on purpose: missing a malware file is treated as 10 times worse than a false alarm.")


def reasons(res: dict) -> None:
    st.markdown("The model looks at 2,568 measurements of the file, in 12 groups. "
                "These groups pushed its decision the most:")
    top = list(res["family_shap"].items())[:TOP_FAMILIES]   # the API sorts families by |SHAP sum|
    biggest = max(abs(v) for _, v in top) or 1.0
    rows = []
    for family, value in top:
        name, what = FAMILY_TEXT.get(family, (family, ""))
        colour, direction = ("#dc2626", "points to malware") if value > 0 else ("#16a34a", "points to safe")
        rows.append(f'<div class="reason"><div class="name">{html.escape(name)}</div>'
                    f'<div class="what">{html.escape(what)}</div><div class="bar">'
                    f'<div class="track"><div class="val" style="width:{100 * abs(value) / biggest:.0f}%;'
                    f'background:{colour}"></div></div><div class="dir" style="color:{colour}">{direction}</div>'
                    f"</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("Measured with SHAP, which splits the model's score among its inputs. Bar length is relative to "
               "the strongest group for this file.")


def technical(res: dict, s: dict[str, Any], scored_by: str, sample: str | None) -> None:
    with st.expander("Technical detail (for questions)"):
        p = res["p_malicious"]
        st.markdown(f"**Model** `{s['model_version']}`: tuned LightGBM ({s['num_trees']} trees), {scored_by}.  \n"
                    f"**P(malicious)** {p:.3e}, **threshold** {res['threshold']:.4g}, **verdict** {res['verdict']}.  \n"
                    f"**test_future** ROC-AUC {s['metrics']['test_future_roc_auc']:.4f}.")
        if sample:
            st.markdown(sample)
        for w in res.get("warnings", []):
            st.warning(w)
        contrib = pd.DataFrame(res["contributions"])[["family", "feature", "value", "shap"]]
        st.markdown(f"**Top-{TOP_K} SHAP columns** (log-odds; positive pushes towards malicious; "
                    f"base value {res['base_value']:+.3f})")
        st.dataframe(contrib.style.format({"shap": "{:+.3f}", "value": "{:.4g}"}), hide_index=True)
        st.markdown("**SHAP sum per family (all 2,568 columns)**")
        st.bar_chart(pd.Series(res["family_shap"]), horizontal=True, height=320, sort=False)


def live_demo(s: dict[str, Any], in_process: bool, scored_by: str) -> None:
    """Pick a program, then show the verdict, the reasons and the technical detail."""
    st.markdown("Now the deployed model itself. Pick a kind of file from the future test, or upload a "
                "Windows program.")
    how_good()
    choice = st.radio("Example", list(CHOICES), horizontal=True, label_visibility="collapsed")
    if st.session_state.get("choice") != choice:   # a new kind starts again at its first example
        st.session_state.update(choice=choice, k=0)
    try:
        if choice == "Upload a file":
            up = st.file_uploader("A Windows program (.exe or .dll). It is read in memory, never run or saved.")
            if up is None:
                return
            res, truth = call(in_process, file=up.getvalue()), None
            sample = f"Uploaded **{html.escape(up.name)}**, {len(up.getvalue()):,} bytes, sha256 `{res['sha256']}`"
        else:
            left, right = st.columns([3, 1])
            left.markdown(f"Showing {CHOICES[choice]}, from the future test.")
            if right.button("Show another", width="stretch"):
                st.session_state.k += 1
            i = pick_example(choice, st.session_state.k)
            if i is None:
                st.info("There is no file of this kind in the available sample.")
                return
            tf = test_future()
            res, truth = call(in_process, features=tf["X"][i]), int(tf["y"][i])
            sample = (f"Source: {tf['source']}. test_future index {int(tf['position'][i])}, "
                      f"week {int(tf['week'][i])}, sha256 `{tf['sha256'][i]}`")
    except (RuntimeError, ValueError, requests.RequestException) as err:
        st.error(str(err))
        return
    with st.container(key="demo_result"):   # one block, so the screenshot script can capture just this part
        st.markdown("#### The verdict")
        verdict_card(res, truth)
        st.markdown("#### Why?")
        reasons(res)
    technical(res, s, scored_by, sample)


def main() -> None:
    st.set_page_config(page_title="Malware detection: manual tuning vs AutoML", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    s = schema()
    in_process = os.environ.get("EMBER_IN_PROCESS") == "1" or not api_reachable(API_URL)
    scored_by = "scored in this process (no API server)" if in_process else f"scored by the FastAPI service at {API_URL}"

    st.title("Can a model spot malware without running it?")
    st.caption("Malware detection on EMBER2024: manual tuning vs AutoML. Isaac Benjamin David (URK23AI1035), "
               "IA-3 Part B.")
    step(1, "The question")
    story.question()
    step(2, "The data")
    story.data_section(s)
    step(3, "The contenders")
    story.contenders()
    step(4, "How we compared them fairly")
    story.fairness()
    step(5, "The results")
    story.results()
    step(6, "The winner, and why")
    story.winner(s, api.model_dir() / s["model_file"])
    step(7, "Try it live")
    live_demo(s, in_process, scored_by)
    step(8, "What we learned")
    story.learned()
    step(9, "Keep in mind")
    story.keep_in_mind()
    st.caption(f"Model {s['model_version']}, {scored_by}. Details: MODEL_CARD.md.")


main()
