"""Data layer: raw EMBER2024 jsonl -> feature cache -> fixed, persisted splits.

Pipeline
  1. build_cache()  streams the raw Win64 jsonl, vectorises every unique sample with
                    thrember's extractor and writes, per official split ("train" = weeks
                    0-51, "test" = weeks 52-63):
                      DATA_CACHE/X_{split}.npy       float32 [n, 2568]
                      DATA_CACHE/meta_{split}.parquet row-aligned metadata
                      DATA_CACHE/cache_info.json      versions, dims, counts
  2. make_splits()  draws the experiment splits from the cache and saves the row
                    indices to SPLITS_DIR, so every method sees identical data.
  3. load_split()   is all that modelling code needs: (X, y, meta) for
                    "train", "test_random" or "test_future".

Index conventions (the source of most bugs in this kind of code, so be precise):
  * cache row      = row number in X_{train,test}.npy / meta_{train,test}.parquet.
  * train position = row number inside the loaded "train" split (0 .. len(train)-1).
  get_cv_folds() and tune_subset_indices() return train positions.
  get_cv_folds(subset=True) returns positions inside the tuning subset.

Security: data/raw is untrusted. It is only ever *read as data* (orjson.loads);
nothing from it is imported or executed, and it is never put on sys.path.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
from config import CV_FOLDS, SEED, split_file

# Paths are read from `config` at call time (not bound here) so tests can redirect them.
OFFICIAL_SPLITS: tuple[str, str] = ("train", "test")
SPLIT_NAMES: tuple[str, str, str] = ("train", "test_random", "test_future")
# Experiment split -> official cache it is drawn from.
SPLIT_SOURCE: dict[str, str] = {"train": "train", "test_random": "train", "test_future": "test"}
SECONDS_PER_WEEK: int = 7 * 24 * 3600
DETECTION_RATIO_RE = re.compile(r"^(\d+)/(\d+)$")


# =============================================================================
# 1. Raw row -> (feature vector, metadata)
# =============================================================================
@lru_cache(maxsize=1)
def get_extractor() -> Any:
    """One thrember PEFeatureExtractor per process (constructing it reads a warnings file)."""
    import thrember  # imported lazily: it is slow to import and only needed here

    return thrember.PEFeatureExtractor()


def parse_detection_ratio(value: Any) -> tuple[float, float, float]:
    """'21/74' -> (21.0, 74.0, 0.2838). Anything malformed gives NaNs instead of crashing."""
    if isinstance(value, str):
        match = DETECTION_RATIO_RE.match(value.strip())
        if match:
            positives, total = float(match.group(1)), float(match.group(2))
            return positives, total, positives / total if total > 0 else float("nan")
    return float("nan"), float("nan"), float("nan")


def _join_tags(value: Any) -> str:
    """List of tags -> 'a|b' string. Parquet handles plain strings more simply than list columns."""
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return "" if value is None else str(value)


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Everything we keep about a sample besides its feature vector.

    detection_ratio is kept for the planned leak experiment ONLY: it is an almost
    perfect label proxy and must never be part of X in the main runs.
    """
    positives, total, ratio = parse_detection_ratio(row.get("detection_ratio"))
    authenticode = row.get("authenticode") or {}
    packers = row.get("packer") or []
    return {
        "sha256": row["sha256"],
        "label": int(row["label"]),
        "first_submission_date": int(row["first_submission_date"]),  # unix s; converted later
        "last_analysis_date": int(row.get("last_analysis_date") or 0),
        "week_id": int(row["week_id"]),  # official 0-based week of the source file
        "detection_ratio": row.get("detection_ratio"),
        "detection_positives": positives,
        "detection_total": total,
        "detection_ratio_float": ratio,
        "family": row.get("family"),
        "family_confidence": row.get("family_confidence"),
        "file_size": int((row.get("general") or {}).get("size", -1)),
        "packer": _join_tags(packers),
        "is_packed": len(packers) > 0,
        "signed": int(authenticode.get("num_certs", 0)) > 0,
        "self_signed": bool(authenticode.get("self_signed", 0)),
        "behavior": _join_tags(row.get("behavior")),
        "file_property": _join_tags(row.get("file_property")),
        "file_type": row.get("file_type"),
    }


# =============================================================================
# 2. One raw file -> one cache part (runs in a worker process)
# =============================================================================
@dataclass
class PartResult:
    """What a worker reports back about one raw file."""

    source_file: str
    lines: int
    kept: int
    duplicates_in_file: int
    dropped_bad_label: int
    nonfinite_values: int
    seconds: float


def vectorize_file(raw_path: Path, part_dir: Path) -> PartResult:
    """Vectorise one weekly jsonl into part_dir/<stem>.npy + <stem>.parquet.

    Why dedupe here: every EMBER2024 weekly file holds each sample twice (lines
    10,001-20,000 repeat lines 1-10,000). The two copies share sha256, feature
    fields and label; only the capa outputs (caps, mbc, ttps) can differ, and those
    are not part of X. Keeping both copies would put identical feature vectors on
    both sides of the random split, i.e. direct train/test leakage.
    """
    start = time.time()
    extractor = get_extractor()
    seen: set[str] = set()
    vectors: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    lines = duplicates = bad_label = 0
    offset = 0
    with raw_path.open("rb") as f:
        for line in f:
            line_offset, offset = offset, offset + len(line)
            if not line.strip():
                continue
            lines += 1
            row = orjson.loads(line)
            if row["sha256"] in seen:  # keep the first occurrence only
                duplicates += 1
                continue
            seen.add(row["sha256"])
            if row.get("label") not in (0, 1):  # EMBER uses -1 for "unlabelled"
                bad_label += 1
                continue
            vectors.append(extractor.process_raw_features(row))
            meta = row_metadata(row)
            # Byte offset lets verify_data.py re-read this exact line for the alignment check.
            meta.update(source_file=raw_path.name, source_offset=line_offset)
            metas.append(meta)

    X = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, extractor.dim), np.float32)
    np.save(part_dir / f"{raw_path.stem}.npy", X)
    pd.DataFrame(metas).to_parquet(part_dir / f"{raw_path.stem}.parquet", index=False)
    return PartResult(
        source_file=raw_path.name,
        lines=lines,
        kept=len(metas),
        duplicates_in_file=duplicates,
        dropped_bad_label=bad_label,
        nonfinite_values=int((~np.isfinite(X)).sum()),
        seconds=round(time.time() - start, 1),
    )


def raw_files(raw_dir: Path, official: str) -> list[Path]:
    """Sorted weekly files of one official split. Sorted names = chronological order."""
    return sorted(raw_dir.glob(f"*_Win64_{official}*.jsonl"))


# =============================================================================
# 3. Parts -> final cache (main process)
# =============================================================================
def _merge_parts(
    files: list[Path], part_dir: Path, official: str, out_dir: Path, already_seen: set[str]
) -> dict[str, Any]:
    """Concatenate the parts of one official split in file order, dropping sha256s already seen.

    The cross-file dedupe matters: a handful of samples appear in two different weeks.
    `already_seen` holds the sha256s of earlier official splits, so a sample that is
    in both train and test is kept only in train (dedupe is global, earliest wins).
    X is written through a memmap so RAM use stays at one part (~100 MB) at a time.
    """
    metas = [pd.read_parquet(part_dir / f"{f.stem}.parquet") for f in files]
    meta = pd.concat(metas, ignore_index=True)
    in_earlier_split = meta["sha256"].isin(already_seen).to_numpy()
    keep = ~meta["sha256"].duplicated(keep="first").to_numpy() & ~in_earlier_split
    n_dim = get_extractor().dim

    tmp_path = out_dir / f"X_{official}.tmp.npy"
    X = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(int(keep.sum()), n_dim))
    row, start = 0, 0
    for f, part_meta in zip(files, metas):
        part_keep = keep[start:start + len(part_meta)]
        start += len(part_meta)
        part_X = np.load(part_dir / f"{f.stem}.npy")
        if len(part_X) != len(part_meta):
            raise RuntimeError(f"{f.name}: {len(part_X)} vectors but {len(part_meta)} metadata rows")
        block = part_X[part_keep]
        X[row:row + len(block)] = block
        row += len(block)
    X.flush()
    del X  # release the file handle before renaming (Windows refuses otherwise)
    os.replace(tmp_path, out_dir / f"X_{official}.npy")

    meta = meta.loc[keep].reset_index(drop=True)
    meta.insert(0, "official_split", official)
    already_seen.update(meta["sha256"])
    return {
        "meta": meta,
        "rows": int(keep.sum()),
        "duplicates_across_files": int((~keep & ~in_earlier_split).sum()),
        "duplicates_of_earlier_split": int(in_earlier_split.sum()),
    }


def _add_time_columns(metas: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Convert unix seconds to datetime64 and derive `week`, 0-based from the earliest date.

    The anchor is midnight (UTC) of the earliest first_submission_date over both
    official splits, so weeks start on the same day as EMBER's weekly files and
    `week` should equal the official `week_id` (verify_data.py checks this).
    """
    earliest = min(int(m["first_submission_date"].min()) for m in metas.values() if len(m))
    anchor_s = earliest - earliest % 86400
    for meta in metas.values():
        seconds = meta["first_submission_date"].astype("int64")
        meta["week"] = ((seconds - anchor_s) // SECONDS_PER_WEEK).astype("int16")
        meta["first_submission_date"] = pd.to_datetime(seconds, unit="s")
        meta["last_analysis_date"] = pd.to_datetime(meta["last_analysis_date"].astype("int64"), unit="s")
        meta["label"] = meta["label"].astype("int8")
    return pd.Timestamp(anchor_s, unit="s")


def raw_win64_dir() -> Path:
    return config.DATA_RAW / "win64"


def build_cache(
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    workers: int = config.N_JOBS,
) -> dict[str, Any]:
    """Build X_{train,test}.npy + meta_{train,test}.parquet + cache_info.json from raw jsonl.

    Files are vectorised in parallel (one file per task), then merged in sorted file
    order, so the result is identical whatever the number of workers.
    cache_info.json is written last: its presence means the cache is complete.
    """
    raw_dir = raw_dir or raw_win64_dir()
    out_dir = out_dir or config.DATA_CACHE
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "cache_info.json"
    info_path.unlink(missing_ok=True)  # an old marker must not vouch for a half-rebuilt cache
    part_dir = out_dir / "_parts"
    part_dir.mkdir(exist_ok=True)
    files = {official: raw_files(raw_dir, official) for official in OFFICIAL_SPLITS}
    all_files = [f for fs in files.values() for f in fs]
    if not all_files:
        raise FileNotFoundError(f"No *_Win64_{{train,test}}*.jsonl files in {raw_dir}")

    start = time.time()
    # Biggest files first so the slowest task is not the one scheduled last.
    order = sorted(all_files, key=lambda p: p.stat().st_size, reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(vectorize_file, order, [part_dir] * len(order)))
        by_file = {r.source_file: r for r in results}
    for r in results:
        print(f"  {r.source_file}: {r.lines} lines -> {r.kept} kept ({r.seconds}s)", flush=True)

    seen: set[str] = set()  # filled train-first, so OFFICIAL_SPLITS order decides who keeps a duplicate
    merged = {o: _merge_parts(files[o], part_dir, o, out_dir, seen) for o in OFFICIAL_SPLITS if files[o]}
    metas = {o: m["meta"] for o, m in merged.items()}
    anchor = _add_time_columns(metas)
    for official, meta in metas.items():
        meta.to_parquet(out_dir / f"meta_{official}.parquet", index=False)

    extractor = get_extractor()
    info = {
        "built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "seconds": round(time.time() - start, 1),
        "raw_dir": str(raw_dir),
        "thrember_version": metadata.version("thrember"),
        "thrember_commit": _thrember_commit(),
        "feature_dim": int(extractor.dim),
        "feature_groups": [list(g) for g in feature_groups()],
        "week_anchor_utc": anchor.isoformat(),
        "splits": {
            official: {
                "files": len(files[official]),
                "lines": sum(by_file[f.name].lines for f in files[official]),
                "duplicates_in_file": sum(by_file[f.name].duplicates_in_file for f in files[official]),
                "duplicates_across_files": merged[official]["duplicates_across_files"],
                "duplicates_of_earlier_split": merged[official]["duplicates_of_earlier_split"],
                "dropped_bad_label": sum(by_file[f.name].dropped_bad_label for f in files[official]),
                "nonfinite_values": sum(by_file[f.name].nonfinite_values for f in files[official]),
                "rows": merged[official]["rows"],
            }
            for official in merged
        },
        "per_file": [vars(by_file[f.name]) for f in all_files],
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    shutil.rmtree(part_dir)  # ~6.5 GB of intermediates, all reproducible from raw
    _read_parquet.cache_clear()  # never serve metadata read before this rebuild
    return info


def _thrember_commit() -> str | None:
    """Commit of the installed thrember, from pip's PEP 610 record (direct_url.json).

    A git install records the commit directly; an install from the local clone in
    third_party/ records only its path, so we ask git for that clone's HEAD.
    """
    try:
        direct_url = json.loads(metadata.distribution("thrember").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        return None
    if commit := direct_url.get("vcs_info", {}).get("commit_id"):
        return commit
    url = direct_url.get("url", "")
    if url.startswith("file:///"):
        result = subprocess.run(["git", "-C", url.removeprefix("file:///"), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() or None
    return None


# =============================================================================
# 4. Cache readers
# =============================================================================
def feature_groups() -> list[tuple[str, int, int]]:
    """(family_name, start, end) column ranges, in the extractor's own order. Sum of widths = D."""
    groups, start = [], 0
    for fe in get_extractor().features:
        groups.append((fe.name, start, start + fe.dim))
        start += fe.dim
    return groups


def cache_info() -> dict[str, Any]:
    """Contents of cache_info.json; raises if the cache was never completed."""
    path = config.DATA_CACHE / "cache_info.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/build_cache.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def open_cache_X(official: str) -> np.ndarray:
    """Read-only memmap of a whole official split ("train" or "test"). Nothing is read until indexed."""
    return np.load(config.DATA_CACHE / f"X_{official}.npy", mmap_mode="r")


def load_cache_meta(official: str) -> pd.DataFrame:
    """Metadata of a whole official split (row i describes X_{official}[i]). Do not modify in place."""
    return _read_parquet(config.DATA_CACHE / f"meta_{official}.parquet")


@lru_cache(maxsize=4)
def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# =============================================================================
# 5. Experiment splits
# =============================================================================
def _strata(meta: pd.DataFrame) -> np.ndarray:
    """label x week strata, so every sample keeps the weekly mix and the 50/50 class balance."""
    return meta["label"].to_numpy().astype(np.int64) * 1000 + meta["week"].to_numpy().astype(np.int64)


def _stratified_sample(rows: np.ndarray, strata: np.ndarray, n: int) -> np.ndarray:
    """Stratified draw of n of `rows` (seeded); all rows if n >= len(rows)."""
    if n >= len(rows):
        return np.sort(rows)
    picked, _ = train_test_split(rows, train_size=n, stratify=strata, random_state=SEED)
    return np.sort(picked)  # sorted cache rows make memmap reads sequential


def _cv_fold_ids(y: np.ndarray, n_folds: int = CV_FOLDS) -> np.ndarray:
    """fold_ids[i] = the fold in which position i is the validation row."""
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for k, (_, val) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_ids[val] = k
    return fold_ids


def make_splits(
    subsample_n: int = config.SUBSAMPLE_N,
    test_future_n: int = config.TEST_FUTURE_N,
    test_fraction: float = config.TEST_FRACTION,
    tune_subset_n: int = config.TUNE_SUBSET_N,
) -> dict[str, Any]:
    """Draw and persist every split. Deterministic given SEED and the cache.

    train / test_random : a stratified (label x week) pool of subsample_n rows from the
                          52 official training weeks, split (1 - f) / f, stratified.
    test_future         : a stratified sample of test_future_n rows from the 12 later
                          official test weeks - the "real deployment" test.
    tune subset + folds : tuning uses a stratified subset of train and fixed CV folds,
                          saved so Grid, Random and Optuna see exactly the same data.
    """
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    meta_tr = load_cache_meta("train")
    meta_te = load_cache_meta("test")
    strata_tr = _strata(meta_tr)

    pool = _stratified_sample(np.arange(len(meta_tr)), strata_tr, subsample_n)
    train_rows, test_random_rows = train_test_split(
        pool, test_size=test_fraction, stratify=strata_tr[pool], random_state=SEED
    )
    test_future_rows = _stratified_sample(np.arange(len(meta_te)), _strata(meta_te), test_future_n)
    rows = {
        "train": np.sort(train_rows),
        "test_random": np.sort(test_random_rows),
        "test_future": test_future_rows,
    }

    y_train = meta_tr["label"].to_numpy()[rows["train"]]
    train_strata = strata_tr[rows["train"]]
    tune_pos = _stratified_sample(np.arange(len(y_train)), train_strata, tune_subset_n)
    artefacts = {
        **{f"{name}_rows": r.astype(np.int64) for name, r in rows.items()},
        "tune_subset_idx": tune_pos.astype(np.int64),
        "cv_fold_ids": _cv_fold_ids(y_train),
        "cv_fold_ids_subset": _cv_fold_ids(y_train[tune_pos]),
    }
    for stem, arr in artefacts.items():
        np.save(split_file(stem), arr)

    info = _splits_summary(rows, tune_pos, {"train": meta_tr, "test": meta_te})
    split_file("splits_info", ".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _clear_split_caches()
    return info


def _splits_summary(
    rows: dict[str, np.ndarray], tune_pos: np.ndarray, metas: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """Sizes, class balance and date ranges of each split, for splits_info.json."""
    summary: dict[str, Any] = {"seed": SEED, "quick": config.QUICK, "cv_folds": CV_FOLDS, "splits": {}}
    for name, r in rows.items():
        meta = metas[SPLIT_SOURCE[name]].iloc[r]
        summary["splits"][name] = {
            "source_cache": SPLIT_SOURCE[name],
            "n": int(len(r)),
            "pos_rate": float(meta["label"].mean()),
            "n_weeks": int(meta["week"].nunique()),
            "date_min": str(meta["first_submission_date"].min()),
            "date_max": str(meta["first_submission_date"].max()),
        }
    y_train = metas["train"]["label"].to_numpy()[rows["train"]]
    summary["tune_subset"] = {"n": int(len(tune_pos)), "pos_rate": float(y_train[tune_pos].mean())}
    return summary


@lru_cache(maxsize=None)
def _load_split_artefact(stem: str) -> np.ndarray:
    path = split_file(stem)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/build_cache.py (it calls make_splits)")
    return np.load(path)


def _clear_split_caches() -> None:
    _load_split_artefact.cache_clear()


def split_rows(name: str) -> np.ndarray:
    """Cache rows (into X_{train|test}.npy) of an experiment split."""
    if name not in SPLIT_SOURCE:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
    return _load_split_artefact(f"{name}_rows")


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(X float32 [n, D], y int8 [n], meta) of "train", "test_random" or "test_future".

    Only the split's own rows are read from the memmap, so loading train (160k rows,
    ~1.6 GB) never pulls the whole 5 GB cache into RAM.
    """
    rows = split_rows(name)
    official = SPLIT_SOURCE[name]
    X = np.asarray(open_cache_X(official)[rows], dtype=np.float32)
    meta = load_cache_meta(official).iloc[rows].reset_index(drop=True)
    y = meta["label"].to_numpy().astype(np.int8)
    return X, y, meta


def tune_subset_indices() -> np.ndarray:
    """Positions within the train split of the stratified tuning subset (TUNE_SUBSET_N rows)."""
    return _load_split_artefact("tune_subset_idx")


def get_cv_folds(subset: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed StratifiedKFold(CV_FOLDS, shuffle, SEED) folds as (fit_idx, val_idx) pairs.

    subset=False: positions within the train split.
    subset=True : positions within the tuning subset (i.e. index X[tune_subset_indices()]).
    """
    fold_ids = _load_split_artefact("cv_fold_ids_subset" if subset else "cv_fold_ids")
    positions = np.arange(len(fold_ids))
    return [(positions[fold_ids != k], positions[fold_ids == k]) for k in range(int(fold_ids.max()) + 1)]
