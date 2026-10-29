"""EMBER2024 challenge set: malicious files that antivirus engines initially MISSED.

data/raw/challenge/*.jsonl holds 6,315 malicious rows of six file types; only the Win64 rows
match our models. build_challenge_cache() vectorises them with the same thrember extractor and
the same conventions as the main cache (src/data.py): first occurrence of a sha256 wins, labels
other than 0/1 are dropped, and `week` uses the main cache's anchor. It writes NEW files
  data/cache/X_challenge_win64.npy, meta_challenge_win64.parquet, challenge_win64_info.json
and never touches X_{train,test}.npy or their metadata.

Every challenge file is malicious, so the only metric is the detection rate (recall). Rows whose
sha256 is also in our train split are flagged (in_train_split): a model has seen those.
data/raw is untrusted: its files are only read as JSON (orjson.loads), never imported or run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd

import config
from src import data

FILE_TYPE: str = "Win64"
STEM: str = "challenge_win64"


def challenge_dir() -> Path:
    return config.DATA_RAW / "challenge"


def challenge_files() -> list[Path]:
    """Weekly challenge files, sorted by name (= by date)."""
    files = sorted(challenge_dir().glob("*_challenge_malicious.jsonl"))
    if not files:
        raise FileNotFoundError(f"No challenge files in {challenge_dir()}; run scripts/download_data.py")
    return files


def cache_paths(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or config.DATA_CACHE
    return {"X": out_dir / f"X_{STEM}.npy", "meta": out_dir / f"meta_{STEM}.parquet",
            "info": out_dir / f"{STEM}_info.json"}


def read_rows(files: list[Path], file_type: str = FILE_TYPE) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Vectorise the rows of `file_type`; return (vectors, metadata rows, counts)."""
    extractor = data.get_extractor()
    seen: set[str] = set()
    vectors, metas = [], []
    counts: dict[str, Any] = {"lines": 0, "by_file_type": {}, "duplicates": 0, "dropped_bad_label": 0}
    for path in files:
        offset = 0
        with path.open("rb") as f:
            for line in f:
                line_offset, offset = offset, offset + len(line)
                if not line.strip():
                    continue
                counts["lines"] += 1
                row = orjson.loads(line)
                kind = str(row.get("file_type"))
                counts["by_file_type"][kind] = counts["by_file_type"].get(kind, 0) + 1
                if kind != file_type:
                    continue
                if row["sha256"] in seen:  # same rule as the main cache: the first copy wins
                    counts["duplicates"] += 1
                    continue
                seen.add(row["sha256"])
                if row.get("label") not in (0, 1):
                    counts["dropped_bad_label"] += 1
                    continue
                vectors.append(extractor.process_raw_features(row))
                meta = data.row_metadata(row)
                meta.update(source_file=path.name, source_offset=line_offset)
                metas.append(meta)
    return vectors, metas, counts


def add_overlap_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """Flag challenge files that are also in the main cache or in one of our splits."""
    split_sha = {name: set(data.load_cache_meta(data.SPLIT_SOURCE[name]).iloc[data.split_rows(name)]["sha256"])
                 for name in data.SPLIT_NAMES}
    for official in data.OFFICIAL_SPLITS:
        meta[f"in_{official}_cache"] = meta["sha256"].isin(set(data.load_cache_meta(official)["sha256"]))
    for name, shas in split_sha.items():
        meta[f"in_{name}_split"] = meta["sha256"].isin(shas)
    return meta


def build_challenge_cache(out_dir: Path | None = None) -> dict[str, Any]:
    """Vectorise the Win64 challenge rows into new cache files (written atomically) and return the info."""
    start = time.time()
    paths = cache_paths(out_dir)
    vectors, metas, counts = read_rows(challenge_files())
    X = np.vstack(vectors).astype(np.float32)
    meta = pd.DataFrame(metas)
    # Same week numbering as the main cache (a naive Timestamp's .timestamp() reads it as UTC).
    anchor_s = int(pd.Timestamp(data.cache_info()["week_anchor_utc"]).timestamp())
    meta["week"] = ((meta["first_submission_date"].astype("int64") - anchor_s) // data.SECONDS_PER_WEEK).astype("int16")
    meta["first_submission_date"] = pd.to_datetime(meta["first_submission_date"], unit="s")
    meta["label"] = meta["label"].astype("int8")
    meta = add_overlap_columns(meta)

    paths["X"].parent.mkdir(parents=True, exist_ok=True)
    tmp_x = paths["X"].with_name(paths["X"].stem + ".tmp.npy")
    np.save(tmp_x, X)
    os.replace(tmp_x, paths["X"])
    meta.to_parquet(paths["meta"], index=False)
    info = {
        "built_utc": pd.Timestamp.now(tz="UTC").isoformat(), "seconds": round(time.time() - start, 1),
        "source": challenge_dir().relative_to(config.ROOT).as_posix(), "files": len(challenge_files()),
        "file_type": FILE_TYPE, **counts, "rows": int(len(meta)), "feature_dim": int(X.shape[1]),
        "nonfinite_values": int((~np.isfinite(X)).sum()), "label_counts": meta["label"].value_counts().to_dict(),
        "week_min": int(meta["week"].min()), "week_max": int(meta["week"].max()),
        "week_matches_week_id": bool((meta["week"] == meta["week_id"]).all()),
        "overlap": {col: int(meta[col].sum()) for col in meta.columns if col.startswith("in_")},
    }
    paths["info"].write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


def load_challenge(out_dir: Path | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """(X, meta) of the vectorised Win64 challenge rows (raises if the cache was never built)."""
    paths = cache_paths(out_dir)
    if not paths["info"].exists():
        raise FileNotFoundError(f"{paths['info']} missing: run scripts/run_eval_bias.py --stages challenge")
    return np.load(paths["X"]), pd.read_parquet(paths["meta"])
