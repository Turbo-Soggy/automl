"""Step 5: the FastAPI service in app/api.py, through FastAPI's TestClient (no server needed).

Needs models/final/ from scripts/export_model.py; the whole module skips without it. The
parity test against the feature cache skips when the cache is absent, and the notepad.exe
upload tests skip off Windows.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app import api

MODEL_DIR = api.model_dir()
if not (MODEL_DIR / "schema.json").exists():
    pytest.skip(f"{MODEL_DIR} has no schema.json: run scripts/export_model.py first", allow_module_level=True)

import lightgbm as lgb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

NOTEPAD = Path(r"C:\Windows\System32\notepad.exe")
D = api.FEATURE_DIM
TOL = 1e-6


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(api.app) as c:   # the context runs the lifespan, which loads the model
        yield c


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads((MODEL_DIR / "schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def offline() -> lgb.Booster:
    """The exported model loaded directly, as offline scoring would."""
    return lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))


@pytest.fixture(scope="module")
def demo() -> np.lib.npyio.NpzFile:
    return np.load(MODEL_DIR / "demo_samples.npz", allow_pickle=False)


def post_raw_json(client: TestClient, text: str):
    """Send JSON text as is: json.dumps would refuse NaN, but a client can still send the tokens."""
    return client.post("/predict", content=text, headers={"content-type": "application/json"})


def vector_text(token: str, position: int = 0) -> str:
    values = ["0.0"] * D
    values[position] = token
    return '{"features": [' + ",".join(values) + "]}"


# --- /health ------------------------------------------------------------------------------------

def test_health_reports_version_and_hash(client, schema) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == schema["model_version"]
    assert body["model_sha256"] == hashlib.sha256((MODEL_DIR / "model.txt").read_bytes()).hexdigest()
    assert body["feature_dim"] == D == schema["feature_dim"]
    assert body["threshold"] == schema["threshold"]["threshold"]


# --- /predict: parity and labels ----------------------------------------------------------------

def test_predict_matches_offline_booster(client, offline, demo) -> None:
    X = demo["X"][:20]
    expected = offline.predict(X)
    for x, p_off, p_saved in zip(X, expected, demo["p"][:20]):
        body = client.post("/predict", json={"features": x.tolist()}).json()
        assert abs(body["p_malicious"] - p_off) <= TOL
        assert abs(body["p_malicious"] - p_saved) <= TOL   # the score saved at export time


def test_predict_matches_saved_step2_scores_on_cache_rows(client, schema) -> None:
    """20 real test_future rows from the cache score as in reports/predictions (Step 2)."""
    import config
    if not (config.DATA_CACHE / "X_test.npy").exists():
        pytest.skip("feature cache absent")
    from src import data
    rows = data.split_rows("test_future")[:20]
    X = np.asarray(data.open_cache_X("test")[rows], dtype=np.float32)
    saved = np.load(config.ROOT / schema["parity_check"]["against"].split(" ")[0])["test_future_p"][:20]
    got = [client.post("/predict", json={"features": x.tolist()}).json()["p_malicious"] for x in X]
    assert np.max(np.abs(np.array(got) - saved)) <= TOL


def test_label_follows_threshold(client, demo, schema) -> None:
    threshold = schema["threshold"]["threshold"]
    for x, p in zip(demo["X"][:30], demo["p"][:30]):
        body = client.post("/predict", json={"features": x.tolist()}).json()
        assert body["label"] == int(p >= threshold)
        assert body["verdict"] == ("malicious" if p >= threshold else "benign")
        assert body["threshold"] == threshold and body["model_version"] == schema["model_version"]


def test_shap_contributions_add_up_to_the_logit(client, demo) -> None:
    body = client.post("/predict?top_k=10", json={"features": demo["X"][0].tolist()}).json()
    assert len(body["contributions"]) == 10
    shaps = [abs(c["shap"]) for c in body["contributions"]]
    assert shaps == sorted(shaps, reverse=True)
    p = body["p_malicious"]
    logit = np.log(p / (1 - p))
    assert abs(body["base_value"] + sum(body["family_shap"].values()) - logit) < 1e-6
    assert len(body["family_shap"]) == 12


def test_batch_matches_single(client, demo) -> None:
    X = demo["X"][:5]
    batch = client.post("/predict_batch", json={"rows": X.tolist()}).json()
    single = [client.post("/predict", json={"features": x.tolist()}).json()["p_malicious"] for x in X]
    assert np.allclose([b["p_malicious"] for b in batch], single, atol=TOL)


# --- /predict: validation -----------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, D - 1, D + 1])
def test_wrong_length_is_422(client, n: int) -> None:
    assert client.post("/predict", json={"features": [0.0] * n}).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_is_422(client, token: str) -> None:
    """Python's json parser turns these tokens into nan / inf, so the API does receive them."""
    response = post_raw_json(client, vector_text(token, position=7))
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "features", 7]


@pytest.mark.parametrize("token", ['"1.5"', '"abc"', "null", "true", "[1.0]", "{}"])
def test_non_numeric_is_422(client, token: str) -> None:
    assert post_raw_json(client, vector_text(token)).status_code == 422


@pytest.mark.parametrize("text", ["", "not json", "{}", '{"rows": []}', '{"features": "0"}'])
def test_malformed_body_is_422(client, text: str) -> None:
    assert post_raw_json(client, text).status_code == 422


def test_422_does_not_echo_the_input(client) -> None:
    body = post_raw_json(client, '{"features": [' + ",".join(["NaN"] * D) + "]}").json()
    assert body["n_errors"] == D and len(body["detail"]) == api.MAX_ERRORS_SHOWN
    assert all("input" not in e for e in body["detail"])


def test_top_k_out_of_range_is_422(client, demo) -> None:
    x = demo["X"][0].tolist()
    assert client.post(f"/predict?top_k={api.MAX_TOP_K + 1}", json={"features": x}).status_code == 422
    assert client.post("/predict?top_k=-1", json={"features": x}).status_code == 422


# --- /predict_file ------------------------------------------------------------------------------

def octet(client: TestClient, data: bytes):
    return client.post("/predict_file?top_k=10", content=data, headers={"content-type": "application/octet-stream"})


@pytest.mark.skipif(not NOTEPAD.exists(), reason="notepad.exe exists on Windows only")
def test_predict_file_notepad(client, offline) -> None:
    import thrember
    data = NOTEPAD.read_bytes()
    body = octet(client, data).json()
    expected = offline.predict(thrember.PEFeatureExtractor().feature_vector(data).reshape(1, -1))[0]
    assert abs(body["p_malicious"] - expected) <= TOL   # same vector as thrember's own extractor
    assert body["sha256"] == hashlib.sha256(data).hexdigest() and body["size_bytes"] == len(data)
    assert body["warnings"] == []   # notepad.exe is a Win64 (AMD64) PE
    assert len(body["contributions"]) == 10
    multipart = client.post("/predict_file", files={"file": ("notepad.exe", data)})
    assert multipart.status_code == 200 and multipart.json()["p_malicious"] == body["p_malicious"]


@pytest.mark.skipif(not NOTEPAD.exists(), reason="notepad.exe exists on Windows only")
def test_upload_is_never_executed(client, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the API tried to start a process")
    monkeypatch.setattr(subprocess, "Popen", refuse)
    assert octet(client, NOTEPAD.read_bytes()).status_code == 200


def test_random_bytes_are_400(client) -> None:
    response = octet(client, np.random.default_rng(42).bytes(10_000))
    assert response.status_code == 400
    assert "not a valid PE" in response.json()["detail"]


def test_truncated_pe_is_400_not_500(client) -> None:
    data = NOTEPAD.read_bytes()[:300] if NOTEPAD.exists() else b"MZ" + b"\0" * 300
    assert octet(client, data).status_code == 400


def test_empty_upload_is_400(client) -> None:
    assert octet(client, b"").status_code == 400
    assert client.post("/predict_file", files={"file": ("empty.exe", b"")}).status_code == 400


def test_multipart_without_file_is_400(client) -> None:
    assert client.post("/predict_file", data={"note": "no file"},
                       files={"other": (None, "text")}).status_code == 400


@pytest.mark.parametrize("multipart", [False, True])
def test_oversize_upload_is_rejected(client, multipart: bool) -> None:
    data = b"\0" * (api.MAX_UPLOAD_BYTES + 1)
    response = (client.post("/predict_file", files={"file": ("big.exe", data)}) if multipart
                else octet(client, data))
    assert response.status_code == 413


@pytest.mark.parametrize("route", sorted(api.JSON_ROUTES))
def test_oversize_json_body_is_413(client, route: str) -> None:
    """The JSON routes refuse a huge body from its Content-Length, before parsing it."""
    body = b"[" + b"0," * (api.MAX_JSON_BYTES // 2) + b"0]"
    response = client.post(route, content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_full_batch_fits_under_the_json_cap(demo) -> None:
    """A legal MAX_BATCH request, written the way json.dumps writes floats, is under the cap."""
    rows = np.resize(demo["X"], (api.MAX_BATCH, api.FEATURE_DIM)).tolist()
    assert len(json.dumps({"rows": rows}).encode()) < api.MAX_JSON_BYTES


def test_upload_never_touches_disk(client, monkeypatch) -> None:
    """A 2 MB multipart upload would normally spill to a temp file (starlette spools at 1 MB)."""
    def refuse(*args, **kwargs):
        raise AssertionError("the upload was written to a temporary file")
    for name in ("TemporaryFile", "NamedTemporaryFile", "mkstemp"):
        monkeypatch.setattr(tempfile, name, refuse)
    data = np.random.default_rng(1).bytes(2 * 1024 * 1024)
    response = client.post("/predict_file", files={"file": ("blob.bin", data)})
    assert response.status_code == 400   # parsed in memory and rejected as not-a-PE, not a 500


# --- The Streamlit page ---------------------------------------------------------------------------

@pytest.mark.parametrize("demo_only", [False, True])
def test_streamlit_app_runs_in_process(demo_only: bool, monkeypatch) -> None:
    """The page renders all nine sections and every example kind headless (streamlit's AppTest),
    scoring in-process, with the full test_future from the cache (when present) and with only
    the shipped demo samples."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()   # the page caches its data source per process
    st.cache_data.clear()
    monkeypatch.setenv("EMBER_IN_PROCESS", "1")   # no API server needed
    if demo_only:
        monkeypatch.setenv("EMBER_DEMO_SAMPLES", "1")
    at = AppTest.from_file(str(Path(api.ROOT) / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    for n, title in enumerate(["The question", "The data", "The contenders", "How we compared them fairly",
                               "The results", "The winner, and why", "Try it live", "What we learned",
                               "Keep in mind"], start=1):
        assert f'<div class="n">{n}</div><div class="t">{title}</div>' in text, title
    right, wrong = "The model was right", "The model got this one wrong"
    expected = {"Caught malware": ("Flagged as malware", right), "Safe program": ("Looks safe", right),
                "Missed malware": ("Looks safe", wrong), "False alarm": ("Flagged as malware", wrong)}
    for choice, (head, outcome) in expected.items():
        at.radio[0].set_value(choice).run()
        assert not at.exception, at.exception
        text = " ".join(m.value for m in at.markdown)
        assert head in text and outcome in text, choice
        assert "points to" in text                     # the plain-language reasons
        assert "What we learned" in text               # the sections after the demo still render
        assert len(at.expander) == 1 and len(at.dataframe) == 1   # technical detail with the SHAP table
    source = " ".join(m.value for m in at.markdown if m.value.startswith("Source:"))
    assert ("demo_samples.npz" in source) == (demo_only or not (Path(api.ROOT) / "data" / "cache" / "X_test.npy").exists())


def test_presentation_numbers_come_from_the_tables(monkeypatch) -> None:
    """The results and winner sections state the numbers and verdicts in the saved tables."""
    import pandas as pd
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    monkeypatch.setenv("EMBER_IN_PROCESS", "1")
    monkeypatch.setenv("EMBER_DEMO_SAMPLES", "1")
    at = AppTest.from_file(str(Path(api.ROOT) / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    text = " ".join(m.value for m in at.markdown)
    tables = Path(api.ROOT) / "reports" / "tables"
    rvf = pd.read_csv(tables / "random_vs_future.csv").set_index("model")
    assert f"caught **{100 * rvf.loc['optuna', 'test_future_tpr_at_1pct_fpr']:.1f}%** of future malware" in text
    pw = pd.read_csv(tables / "automl_pairwise.csv")
    for tool, name in (("autogluon", "AutoGluon"), ("flaml", "FLAML"), ("h2o", "H2O AutoML")):
        row = pw[(pw.test_set == "test_future") & (pw.metric == "tpr_at_1pct_fpr")
                 & (pw.method_a == tool) & (pw.method_b == "lgbm_tuned")].iloc[0]
        verdict = "**a tie**" if not row.significant_95 else ("**ahead**" if row.diff_a_minus_b > 0 else "**slightly behind**")
        assert f"**{name}**: {100 * rvf.loc[tool, 'test_future_tpr_at_1pct_fpr']:.1f}%, {verdict}" in text, tool
    avm = pd.read_csv(tables / "automl_vs_manual.csv").set_index("method")
    assert f"AutoGluon got there in **{avm.loc['autogluon', 'wall_minutes']:.0f} minutes**" in text
    leak = pd.read_csv(tables / "leakage.csv")
    leak = leak[leak.variant == "leak_detection_ratio"]
    assert f"{int(leak['relied_on_leak'].sum())} of {len(leak)} models made it" in text


def test_present_launcher_commands() -> None:
    """present.py starts uvicorn on the API and streamlit on the page, with this interpreter."""
    import sys
    sys.path.insert(0, str(api.ROOT))
    import present
    assert present.api_command()[:4] == [sys.executable, "-m", "uvicorn", "app.api:app"]
    assert present.page_command()[:5] == [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
    out = subprocess.run([sys.executable, str(Path(api.ROOT) / "present.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--no-api" in out.stdout
