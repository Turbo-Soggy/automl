"""Step 5: FastAPI service for the exported malware detector (models/final/).

Endpoints
  GET  /health         model version, sha256 of the model file, feature dimension, threshold.
  POST /predict        {"features": [2568 finite numbers]} -> p_malicious, label, threshold.
  POST /predict_batch  {"rows": [[...], ...]} (at most MAX_BATCH rows) -> one prediction per row.
  POST /predict_file   the bytes of a PE file (raw body, or multipart field "file"), at most
                       50 MB. thrember extracts the features from the bytes in memory.
  Add ?top_k=10 to /predict or /predict_file to also get the sample's largest TreeSHAP
  contributions (log-odds) and the SHAP total of each feature family.

Safety of uploads: the bytes are only parsed (pefile, signify and thrember read them as
data). They are never written to disk and never executed. The multipart parser normally
spills parts over 1 MB to a temporary file, so it is given a spool size above the body size.

Run from the project root:  uvicorn app.api:app --port 8000
EMBER_MODEL_DIR overrides the model folder (default models/final, or models/final_quick
when EMBER_QUICK=1).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIM: int = 2568            # thrember 0.1.0 vector length; checked against schema.json on load
MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024
MULTIPART_SLACK: int = 64 * 1024   # room for multipart boundaries and headers around a 50 MB file
MAX_BATCH: int = 256
MAX_TOP_K: int = 50
MAX_ERRORS_SHOWN: int = 20     # validation errors listed in a 422 (a bad vector can have thousands)
AMD64_MACHINE: int = 0x8664        # IMAGE_FILE_MACHINE_AMD64: the only machine type in the training data


def model_dir() -> Path:
    default = "final_quick" if os.environ.get("EMBER_QUICK") == "1" else "final"
    return Path(os.environ.get("EMBER_MODEL_DIR", ROOT / "models" / default))


# --- The model ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Predictor:
    """The booster plus what is needed to label and explain its scores."""
    booster: Any
    schema: dict[str, Any]
    threshold: float
    version: str
    feature_names: list[str]
    families: np.ndarray   # family name of each column
    lock: threading.Lock   # one prediction at a time: simple, and safe whatever LightGBM's threading

    def predict(self, X: np.ndarray) -> np.ndarray:
        with self.lock:
            return self.booster.predict(X, num_threads=1)

    def shap(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """(TreeSHAP values [D], base value) in log-odds for one row; they sum to logit(p)."""
        with self.lock:
            contrib = self.booster.predict(x.reshape(1, -1), pred_contrib=True, num_threads=1)[0]
        return contrib[:-1], float(contrib[-1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load models/final once; refuse to serve a model whose sha256 differs from schema.json."""
    import lightgbm as lgb

    folder = model_dir()
    schema = json.loads((folder / "schema.json").read_text(encoding="utf-8"))
    model_path = folder / schema["model_file"]
    if _sha256(model_path) != schema["model_sha256"]:
        raise RuntimeError(f"{model_path} does not match the sha256 in schema.json; re-run scripts/export_model.py")
    if schema["feature_dim"] != FEATURE_DIM:
        raise RuntimeError(f"schema.json says {schema['feature_dim']} features, the API expects {FEATURE_DIM}")
    families = np.empty(FEATURE_DIM, dtype=object)
    for group in schema["feature_groups"]:
        families[group["start"]:group["end"]] = group["name"]
    return Predictor(booster=lgb.Booster(model_file=str(model_path)), schema=schema,
                     threshold=float(schema["threshold"]["threshold"]), version=schema["model_version"],
                     feature_names=schema["feature_names"], families=families, lock=threading.Lock())


@lru_cache(maxsize=1)
def get_extractor() -> Any:
    import thrember  # slow import, only needed for /predict_file
    return thrember.PEFeatureExtractor()


# --- Request and response bodies -------------------------------------------------------------

# strict: reject strings and booleans instead of coercing them; allow_inf_nan=False: reject
# NaN and +-Infinity (Python's json module accepts the bare tokens NaN and Infinity).
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector = Annotated[list[FiniteFloat], Field(min_length=FEATURE_DIM, max_length=FEATURE_DIM)]


class FeatureVector(BaseModel):
    features: Vector


class FeatureBatch(BaseModel):
    rows: Annotated[list[Vector], Field(min_length=1, max_length=MAX_BATCH)]


class Contribution(BaseModel):
    column: int
    feature: str
    family: str
    value: float | None   # the feature value (None if the extractor gave a non-finite number)
    shap: float           # log-odds contribution


class Prediction(BaseModel):
    p_malicious: float
    label: int                                  # 1 = malicious: p_malicious >= threshold
    verdict: Literal["malicious", "benign"]
    threshold: float
    model_version: str
    base_value: float | None = None             # SHAP base value (log-odds), with top_k > 0
    contributions: list[Contribution] | None = None
    family_shap: dict[str, float] | None = None


class FilePrediction(Prediction):
    sha256: str
    size_bytes: int
    warnings: list[str]


# --- Scoring (plain functions, so the Streamlit app can also call them in-process) ---------------

def _finite_or_none(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def explain(pred: Predictor, x: np.ndarray, top_k: int) -> dict[str, Any]:
    """The top_k largest |SHAP| columns of one row, and the SHAP sum of every family."""
    values, base = pred.shap(x)
    order = np.argsort(-np.abs(values), kind="mergesort")[:top_k]
    contributions = [Contribution(column=int(i), feature=pred.feature_names[i], family=str(pred.families[i]),
                                  value=_finite_or_none(x[i]), shap=float(values[i])) for i in order]
    totals = {str(f): float(values[pred.families == f].sum()) for f in dict.fromkeys(pred.families)}
    family_shap = dict(sorted(totals.items(), key=lambda kv: -abs(kv[1])))
    return {"base_value": base, "contributions": contributions, "family_shap": family_shap}


def score_vector(x: np.ndarray, top_k: int = 0) -> Prediction:
    pred = get_predictor()
    p = float(pred.predict(x.reshape(1, -1))[0])
    label = int(p >= pred.threshold)
    extra = explain(pred, x, top_k) if top_k > 0 else {}
    return Prediction(p_malicious=p, label=label, verdict="malicious" if label else "benign",
                      threshold=pred.threshold, model_version=pred.version, **extra)


def extract_features(bytez: bytes) -> tuple[np.ndarray, list[str]]:
    """thrember feature vector of a PE file given as bytes; ValueError if it is not a parsable PE.

    Same steps as thrember's PEFeatureExtractor.feature_vector(), except that a file pefile
    cannot parse is rejected: thrember would silently score it as a header-less blob.
    """
    import pefile

    try:
        pe = pefile.PE(data=bytez)
    except (pefile.PEFormatError, AttributeError) as err:
        raise ValueError(f"not a valid PE file: {err}") from err
    warnings: list[str] = []
    if pe.FILE_HEADER.Machine != AMD64_MACHINE:
        warnings.append(f"machine type 0x{pe.FILE_HEADER.Machine:04x} is not AMD64: the model was trained "
                        "on Win64 PE files only, so this score is out of distribution")
    extractor = get_extractor()
    try:
        raw = {"sha256": hashlib.sha256(bytez).hexdigest()}
        raw.update({fe.name: fe.raw_features(bytez, pe) for fe in extractor.features})
        x = extractor.process_raw_features(raw)
    except Exception as err:  # a malformed PE can break any of the 12 feature parsers
        raise ValueError(f"feature extraction failed: {type(err).__name__}: {err}") from err
    if not np.all(np.isfinite(x)):
        warnings.append("the extractor produced non-finite values; LightGBM treats them as missing")
    return x.astype(np.float32), warnings


def score_file(bytez: bytes, top_k: int = 0) -> FilePrediction:
    x, warnings = extract_features(bytez)
    base = score_vector(x, top_k)
    return FilePrediction(**base.model_dump(), sha256=hashlib.sha256(bytez).hexdigest(),
                          size_bytes=len(bytez), warnings=warnings)


# --- Upload handling --------------------------------------------------------------------------

async def _read_body(request: Request, limit: int) -> bytes:
    """The request body, read into memory, refusing more than `limit` bytes (413)."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:   # also catches a chunked body or a false Content-Length
            raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


async def _file_from_multipart(headers: Headers, body: bytes) -> bytes:
    """The first file part of a multipart body, kept in RAM."""
    async def stream() -> AsyncIterator[bytes]:
        yield body
        yield b""

    parser = MultiPartParser(headers, stream(), max_files=1, max_fields=10)
    # SpooledTemporaryFile moves a part to a temp file on disk once it exceeds spool_max_size
    # (1 MB by default). Above the body size, the upload never leaves memory.
    parser.spool_max_size = len(body) + 1
    try:
        form = await parser.parse()
    except MultiPartException as err:
        raise HTTPException(400, f"bad multipart body: {err.message}") from err
    upload = next((v for v in form.values() if isinstance(v, UploadFile)), None)
    if upload is None:
        raise HTTPException(400, "multipart body has no file part")
    data = await upload.read()
    await form.close()
    return data


async def read_upload(request: Request) -> bytes:
    body = await _read_body(request, MAX_UPLOAD_BYTES + MULTIPART_SLACK)
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        body = await _file_from_multipart(request.headers, body)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not body:
        raise HTTPException(400, "empty upload")
    return body


# --- The app ----------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_predictor()   # fail at start-up, not on the first request, if the model is missing or altered
    yield


app = FastAPI(title="EMBER2024 Win64 malware detector", version="1.0", lifespan=lifespan,
              description="Static-feature triage aid (tuned LightGBM). Not a sole blocking verdict; see MODEL_CARD.md.")

@app.exception_handler(RequestValidationError)
async def compact_422(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the input: the default echoes every bad value, and a NaN in it
    cannot be written as JSON (the error response itself would fail with a 500)."""
    errors = exc.errors()
    shown = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in errors[:MAX_ERRORS_SHOWN]]
    return JSONResponse(status_code=422, content={"detail": shown, "n_errors": len(errors)})


TopK = Annotated[int, Query(ge=0, le=MAX_TOP_K, description="also return the top-k SHAP contributions")]


@app.get("/health")
def health() -> dict[str, Any]:
    pred = get_predictor()
    return {"status": "ok", "model_version": pred.version, "model_sha256": pred.schema["model_sha256"],
            "feature_dim": FEATURE_DIM, "threshold": pred.threshold, "num_trees": pred.schema["num_trees"],
            "thrember_commit": pred.schema["thrember"]["commit"]}


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
def predict(body: FeatureVector, top_k: TopK = 0) -> Prediction:
    return score_vector(np.asarray(body.features, dtype=np.float64), top_k)


@app.post("/predict_batch", response_model=list[Prediction], response_model_exclude_none=True)
def predict_batch(body: FeatureBatch) -> list[Prediction]:
    pred = get_predictor()
    p = pred.predict(np.asarray(body.rows, dtype=np.float64))
    return [Prediction(p_malicious=float(v), label=int(v >= pred.threshold),
                       verdict="malicious" if v >= pred.threshold else "benign",
                       threshold=pred.threshold, model_version=pred.version) for v in p]


@app.post("/predict_file", response_model=FilePrediction, response_model_exclude_none=True,
          openapi_extra={"requestBody": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}})
async def predict_file(request: Request, top_k: TopK = 0) -> FilePrediction:
    bytez = await read_upload(request)
    try:
        return await run_in_threadpool(score_file, bytez, top_k)   # parsing is CPU work: keep it off the event loop
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
