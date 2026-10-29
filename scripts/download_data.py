"""Download and extract the EMBER2024 Win64 train/test raw features (plus the small challenge set).

Why not thrember.download_dataset()? It os.chdir()s into the download folder, which would
make the untrusted data folder the working directory of our Python process, and it deletes
the zip without checking its hash. This script does the same downloads from the same
Hugging Face repo, but:
  * never changes directory and never executes anything from the data,
  * checks each zip's sha256 against the hash published on Hugging Face,
  * extracts only flat *.jsonl members (no paths, so no zip-slip),
  * is resumable: finished zips and already-extracted files are skipped on re-run,
  * with --keep-zipped, leaves the Win64 zips packed (32 GB less disk); src.data then
    streams the weekly files straight out of the zips, byte for byte the same.

Layout written under data/raw/:
  _zips/                       the downloaded zip files (kept for re-extraction)
  win64/*_Win64_train.jsonl    52 weekly training files
  win64/*_Win64_test.jsonl     12 weekly test files
  challenge/*.jsonl            challenge set (all file types mixed; filter on file_type)
  download.log                 progress log
  DOWNLOAD_DONE                JSON summary, written only after full success
  DOWNLOAD_FAILED              error + traceback, written on failure

Usage (from the project root): .venv/Scripts/python.exe scripts/download_data.py [--no-challenge] [--keep-zipped]
(on Linux or macOS the interpreter is .venv/bin/python)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` imports when run as a plain script

from config import DATA_RAW  # noqa: E402
from src.data import flat_jsonl_name  # noqa: E402

REPO_ID = "joyce8/EMBER2024"
# zip file name on Hugging Face -> sub-folder of data/raw it is extracted into
WIN64_ZIPS: dict[str, str] = {"Win64_test.zip": "win64", "Win64_train.zip": "win64"}
CHALLENGE_ZIPS: dict[str, str] = {"challenge.zip": "challenge"}

ZIP_DIR = DATA_RAW / "_zips"
LOG_FILE = DATA_RAW / "download.log"
DONE_MARKER = DATA_RAW / "DOWNLOAD_DONE"
FAILED_MARKER = DATA_RAW / "DOWNLOAD_FAILED"

log = logging.getLogger("download_data")


def setup_logging() -> None:
    """Log to data/raw/download.log and to the console."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def expected_hashes(filenames: list[str]) -> dict[str, tuple[str, int]]:
    """Ask the Hugging Face API for each file's published sha256 and size."""
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(REPO_ID, filenames, repo_type="dataset")
    return {info.path: (info.lfs.sha256, info.size) for info in infos}


def sha256_of(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    """Stream a file through sha256 (the zips are several GB, so never read them whole)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_zip(filename: str, sha256: str, size: int) -> Path:
    """Download one zip into data/raw/_zips (resumable) and verify its hash."""
    from huggingface_hub import hf_hub_download

    log.info("Downloading %s (%.2f GB)", filename, size / 1e9)
    t0 = time.time()
    path = Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=ZIP_DIR))
    log.info("Downloaded %s in %.0f s, verifying sha256", filename, time.time() - t0)
    actual = sha256_of(path)
    if actual != sha256:
        raise RuntimeError(f"sha256 mismatch for {filename}: expected {sha256}, got {actual}")
    return path


def extract_jsonl(zip_path: Path, dest: Path) -> list[str]:
    """Extract the .jsonl members of a zip into dest (flat); skip files already complete."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            out_name = flat_jsonl_name(info)
            if out_name is None:
                log.warning("Skipping non-jsonl member %s", info.filename)
                continue
            out_path = dest / out_name
            if out_path.exists() and out_path.stat().st_size == info.file_size:
                written.append(out_name)
                continue
            tmp = out_path.with_suffix(".jsonl.part")  # rename at the end, so partial files never look complete
            with zf.open(info) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
            tmp.replace(out_path)
            written.append(out_name)
            log.info("Extracted %s (%.0f MB)", out_name, info.file_size / 1e6)
    return written


def count_lines(f: BinaryIO, chunk: int = 64 * 1024 * 1024) -> int:
    """Number of rows (newlines) in an open jsonl stream, read in chunks."""
    n = 0
    while block := f.read(chunk):
        n += block.count(b"\n")
    return n


def count_zip_lines(zip_path: Path) -> dict[str, int]:
    """Rows per .jsonl member, read straight from the zip (used with --keep-zipped)."""
    rows: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if (name := flat_jsonl_name(info)) is not None:
                with zf.open(info) as f:
                    rows[name] = count_lines(f)
    return rows


def run(zips: dict[str, str], keep_zipped: set[str]) -> dict:
    """Download, verify and extract every zip; return a summary for the DONE marker.

    Zips named in keep_zipped are not extracted: src.data reads their members directly.
    """
    hashes = expected_hashes(list(zips))
    summary: dict = {"repo_id": REPO_ID, "zips": {}}
    for filename, subdir in zips.items():
        sha256, size = hashes[filename]
        zip_path = download_zip(filename, sha256, size)
        if filename in keep_zipped:
            rows = count_zip_lines(zip_path)
        else:
            rows = {}
            for m in extract_jsonl(zip_path, DATA_RAW / subdir):
                with (DATA_RAW / subdir / m).open("rb") as f:
                    rows[m] = count_lines(f)
        log.info("%s: %d jsonl files, %d rows", filename, len(rows), sum(rows.values()))
        summary["zips"][filename] = {"sha256": sha256, "bytes": size, "dir": subdir,
                                     "extracted": filename not in keep_zipped, "rows_per_file": rows}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-challenge", action="store_true", help="skip the 32 MB challenge set")
    parser.add_argument("--keep-zipped", action="store_true",
                        help="do not extract the Win64 zips (saves 32 GB; the cache build reads the zips)")
    args = parser.parse_args()

    setup_logging()
    for marker in (DONE_MARKER, FAILED_MARKER):
        marker.unlink(missing_ok=True)

    # Small files first so the challenge and test sets are usable while train is still downloading.
    zips = {**({} if args.no_challenge else CHALLENGE_ZIPS), **WIN64_ZIPS}
    try:
        summary = run(zips, set(WIN64_ZIPS) if args.keep_zipped else set())
    except BaseException as exc:  # noqa: BLE001 - we want a marker even on Ctrl+C
        FAILED_MARKER.write_text(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8")
        log.exception("Download failed")
        raise SystemExit(1) from exc
    DONE_MARKER.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("All downloads finished: %s", DONE_MARKER)


if __name__ == "__main__":
    main()
