"""Append-only experiment log (one JSON object per line).

Every trial and every final evaluation goes through log_experiment(), and every
table or figure in reports/ is rebuilt from this file. Nothing is typed by hand.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import EXPERIMENT_LOG, QUICK

# One id per Python process. run_all.py can set EMBER_RUN_ID so that all its
# child processes share one id and their records can be grouped together.
RUN_ID: str = os.environ.get("EMBER_RUN_ID") or (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
)


def _to_jsonable(obj: Any) -> Any:
    """json.dumps fallback for NumPy scalars/arrays and Paths, which json cannot encode."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot JSON-encode {type(obj).__name__}")


def log_experiment(record: dict[str, Any], path: Path = EXPERIMENT_LOG) -> None:
    """Append one record to the experiment log, stamped with utc time, run_id and quick flag."""
    stamped = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "quick": QUICK,
        **record,
    }
    line = json.dumps(stamped, default=_to_jsonable, allow_nan=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A single write of a whole line keeps records intact when processes append concurrently.
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_records(path: Path = EXPERIMENT_LOG) -> list[dict[str, Any]]:
    """Load the log as a list of the original (nested) records, oldest first."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_experiments(path: Path = EXPERIMENT_LOG) -> pd.DataFrame:
    """Load the log as a DataFrame. Nested dicts are flattened to dotted columns (e.g. "metrics.roc_auc")."""
    records = read_records(path)
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep=".")
