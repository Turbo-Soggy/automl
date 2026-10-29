"""Step 5: Streamlit demo for the malware detector served by app/api.py.

Pick a test_future file (by filter or by index) or upload a PE file; the page sends it to the
API and shows P(malicious), the threshold, the verdict, the true label for test files and the
sample's top-10 TreeSHAP contributions grouped by feature family.

Run from the project root, with the API already running (see DEMO.md):
  streamlit run app/streamlit_app.py
The API address comes from the sidebar (default: env EMBER_API_URL or http://127.0.0.1:8000).
"In-process" mode scores inside this process instead (one-process deployments); the page
falls back to it by itself when the API cannot be reached.

test_future files come from the feature cache when it exists (all 40,000 rows, with the Step 2
scores used for filtering), otherwise, or when EMBER_DEMO_SAMPLES=1, from
models/final/demo_samples.npz (a few hundred rows).
"""
from __future__ import annotations

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

from app import api  # noqa: E402

DEFAULT_URL = os.environ.get("EMBER_API_URL", "http://127.0.0.1:8000")
TOP_K = 10


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


# --- Calling the API --------------------------------------------------------------------------

def call(url: str, in_process: bool, *, features: np.ndarray | None = None, file: bytes | None = None) -> dict:
    """POST to /predict or /predict_file with top_k=10; in-process mode calls the same functions."""
    if in_process:
        result = api.score_vector(features, TOP_K) if file is None else api.score_file(file, TOP_K)
        return result.model_dump(exclude_none=True)
    if file is None:
        r = requests.post(f"{url}/predict", params={"top_k": TOP_K},
                          json={"features": features.astype(float).tolist()}, timeout=60)
    else:
        r = requests.post(f"{url}/predict_file", params={"top_k": TOP_K}, data=file,
                          headers={"content-type": "application/octet-stream"}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"API returned {r.status_code}: {r.text[:300]}")
    return r.json()


# --- Display ----------------------------------------------------------------------------------

def banner(s: dict[str, Any]) -> None:
    m, t = s["metrics"], s["threshold"]
    st.info(
        f"**Model** `{s['model_version']}`: tuned LightGBM ({s['num_trees']} trees) on {s['feature_dim']:,} static "
        f"thrember features of Win64 PE files (EMBER2024, trained on weeks 0-51).  \n"
        f"**test_future** (weeks 52-63): ROC-AUC {m['test_future_roc_auc']:.4f}, TPR at 1% FPR "
        f"{m['test_future_tpr_at_1pct_fpr']:.1%} (95% CI {m['test_future_tpr_at_1pct_fpr_ci_low']:.1%}-"
        f"{m['test_future_tpr_at_1pct_fpr_ci_high']:.1%}).  \n"
        f"**Threshold** {t['threshold']:.4g}: minimises cost with a missed malware {t['cost_ratio_fn_to_fp']:g}x "
        f"as costly as a false alarm, chosen on train out-of-fold scores. It favours recall, so false alarms are common."
    )
    st.warning(
        "**Limits.** A triage aid for analysts and research, not a sole verdict for blocking files. Win64 PE only; "
        "static features only, so adversarially modified files may evade it (untested) and packed benign files "
        "may be flagged (unmeasured); trained at 50/50 "
        "class balance, so precision at real-world prevalence is much lower; malware the AV engines first "
        "missed (the EMBER2024 challenge set) is detected far less often. See MODEL_CARD.md."
    )


def show_result(res: dict, true_label: int | None = None) -> None:
    cols = st.columns(4)
    p = res["p_malicious"]
    cols[0].metric("P(malicious)", f"{p:.2e}" if p < 0.001 else f"{p:.4f}")
    cols[1].metric("Threshold", f"{res['threshold']:.4g}")
    cols[2].metric("Predicted", res["verdict"].upper())
    if true_label is not None:
        truth = "malicious" if true_label == 1 else "benign"
        cols[3].metric("True label", truth.upper(), "correct" if truth == res["verdict"] else "misclassified",
                       delta_color="normal" if truth == res["verdict"] else "inverse")
    for w in res.get("warnings", []):
        st.warning(w)
    show_shap(res)


def show_shap(res: dict) -> None:
    contrib = pd.DataFrame(res["contributions"])
    family_total = contrib.groupby("family")["shap"].sum()
    # Group the 10 rows by family (families ordered by their summed contribution), then by |SHAP|.
    contrib["family_order"] = contrib["family"].map(-family_total.abs())
    contrib = contrib.assign(abs_shap=contrib["shap"].abs()).sort_values(["family_order", "abs_shap"],
                                                                         ascending=[True, False])
    st.subheader(f"Top-{TOP_K} SHAP contributions, grouped by feature family")
    st.caption("SHAP values are in log-odds: positive pushes towards malicious. "
               f"Base value {res['base_value']:+.3f}; base + all SHAP values = logit(P(malicious)). "
               "Table rows are grouped by family, families ordered by the |sum| of their rows shown.")
    left, right = st.columns([3, 2])
    left.dataframe(contrib[["family", "feature", "value", "shap"]].style.format({"shap": "{:+.3f}", "value": "{:.4g}"}),
                   hide_index=True, column_config={"feature": st.column_config.TextColumn(width="medium")})
    totals = pd.Series(res["family_shap"], name="SHAP sum (all columns)")
    right.markdown("**SHAP sum per family (all 2,568 columns)**")
    right.bar_chart(totals, horizontal=True, height=320, sort=False)   # keep the order by |sum|


def sample_tab(url: str, in_process: bool, threshold: float) -> None:
    tf = test_future()
    scored_by = "is scored in-process, without the API" if in_process else "comes from the API"
    st.caption(f"Source: {tf['source']}. Filters use the Step 2 offline scores; the prediction shown {scored_by}.")
    how = st.radio("Choose by", ["filter", "test_future index"], horizontal=True)
    pred = (tf["p"] >= threshold).astype(int)
    if how == "filter":
        c1, c2 = st.columns(2)
        label = c1.selectbox("True label", ["any", "malicious", "benign"])
        outcome = c2.selectbox("Outcome at the threshold", ["any", "correct", "misclassified"])
        mask = np.ones(len(tf["y"]), dtype=bool)
        if label != "any":
            mask &= tf["y"] == (1 if label == "malicious" else 0)
        if outcome != "any":
            mask &= (pred == tf["y"]) == (outcome == "correct")
        matches = np.flatnonzero(mask)
        if len(matches) == 0:
            st.error("No file matches this filter.")
            return
        k = st.number_input(f"File number among the {len(matches):,} matches", 0, len(matches) - 1, 0)
        i = int(matches[k])
    else:
        positions = tf["position"]
        pos = st.number_input(f"test_future index (0-{int(positions.max())})", int(positions.min()),
                              int(positions.max()), int(positions[0]))
        hits = np.flatnonzero(positions == pos)
        if len(hits) == 0:
            st.error("That index is not in the demo sample; use the filter instead.")
            return
        i = int(hits[0])
    st.markdown(f"test_future index **{int(tf['position'][i])}**, week {int(tf['week'][i])}, "
                f"sha256 `{tf['sha256'][i]}`")
    res = call(url, in_process, features=tf["X"][i])
    show_result(res, int(tf["y"][i]))


def upload_tab(url: str, in_process: bool) -> None:
    where = "scored in this process" if in_process else "sent to the API"
    st.caption(f"The file is {where}, parsed in memory with thrember. It is never executed "
               "or written to disk. Limit 50 MB. Only Win64 PE files are in the model's training distribution.")
    up = st.file_uploader("PE file (.exe, .dll, ...)")
    if up is None:
        return
    data = up.getvalue()
    st.markdown(f"**{up.name}**, {len(data):,} bytes")
    res = call(url, in_process, file=data)
    st.markdown(f"sha256 `{res['sha256']}`")
    show_result(res)


def main() -> None:
    st.set_page_config(page_title="EMBER2024 malware detector", layout="wide")
    st.title("EMBER2024 Win64 malware detector")
    s = schema()
    with st.sidebar:
        st.header("API")
        in_process = st.checkbox("In-process (no API server)", value=False)
        url = st.text_input("API URL", DEFAULT_URL, disabled=in_process).rstrip("/")
        if not in_process:
            try:
                health = requests.get(f"{url}/health", timeout=5).json()
                st.success(f"Connected: {health['model_version']}")
            except (requests.RequestException, ValueError, KeyError) as err:
                # One-process deployments (Streamlit Cloud, a Space) have no API server: score here instead.
                st.warning(f"API not reachable at {url} ({type(err).__name__}); scoring in-process instead. "
                           "Start the API to use it (DEMO.md).")
                in_process = True
    banner(s)
    tab_sample, tab_upload = st.tabs(["test_future sample", "Upload a PE file"])
    try:
        with tab_sample:
            sample_tab(url, in_process, float(s["threshold"]["threshold"]))
        with tab_upload:
            upload_tab(url, in_process)
    except (RuntimeError, ValueError, requests.RequestException) as err:
        st.error(str(err))


main()
