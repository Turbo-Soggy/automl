"""Build the feature cache and the experiment splits.

Steps: (1) check the download finished and every weekly file is present,
(2) build_cache() unless the cache is already complete (or --force),
(3) make_splits() with the sizes in config.py (EMBER_QUICK=1 -> small "_quick" splits).

Usage (from the project root):
  .venv/Scripts/python.exe scripts/build_cache.py               # cache (if needed) + splits
  .venv/Scripts/python.exe scripts/build_cache.py --force       # rebuild cache from raw
  EMBER_QUICK=1 .venv/Scripts/python.exe scripts/build_cache.py --splits-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # project root only; data/raw is never on sys.path

import config  # noqa: E402
from src import data  # noqa: E402

EXPECTED_FILES: dict[str, int] = {"train": 52, "test": 12}


def check_download() -> None:
    """Fail early, with a clear message, if the raw download is missing or incomplete."""
    if (config.DATA_RAW / "DOWNLOAD_FAILED").exists():
        raise RuntimeError("data/raw/DOWNLOAD_FAILED exists: rerun scripts/download_data.py")
    if not (config.DATA_RAW / "DOWNLOAD_DONE").exists():
        raise RuntimeError("data/raw/DOWNLOAD_DONE missing: run scripts/download_data.py first")
    for official, expected in EXPECTED_FILES.items():
        found = len(data.raw_files(data.raw_win64_dir(), official))
        if found != expected:
            raise RuntimeError(f"Expected {expected} Win64 {official} files in {data.raw_win64_dir()}, found {found}")
    print("Download check passed: 52 train + 12 test weekly files present.", flush=True)


def cache_complete() -> bool:
    """The cache is complete only if cache_info.json and all four data files exist."""
    names = ["cache_info.json"] + [f"{p}_{o}.{e}" for o in data.OFFICIAL_SPLITS for p, e in (("X", "npy"), ("meta", "parquet"))]
    return all((config.DATA_CACHE / n).exists() for n in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-only", action="store_true", help="skip the cache build (cache must exist)")
    parser.add_argument("--force", action="store_true", help="rebuild the cache even if it is complete")
    parser.add_argument("--workers", type=int, default=config.N_JOBS)
    args = parser.parse_args()
    config.set_seed()

    if not args.splits_only:
        check_download()
        if cache_complete() and not args.force:
            print("Cache already complete; skipping build (use --force to rebuild).", flush=True)
        else:
            start = time.time()
            print(f"Building cache with {args.workers} workers ...", flush=True)
            info = data.build_cache(workers=args.workers)
            print(json.dumps(info["splits"], indent=2))
            print(f"Cache built in {time.time() - start:.0f}s", flush=True)
    elif not cache_complete():
        raise RuntimeError("--splits-only given but the cache is incomplete: run without it first")

    print(f"Making splits (quick={config.QUICK}) ...", flush=True)
    splits = data.make_splits()
    print(json.dumps(splits, indent=2))
    print("BUILD_CACHE_DONE", flush=True)


if __name__ == "__main__":  # required on Windows: worker processes re-import this module
    main()
