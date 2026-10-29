"""Fetch the first few MB of one EMBER2024 zip and decode its first jsonl rows.

Why: the full Win64 download is ~7 GB. Before writing the cache builder we want to
see the real raw schema. A zip's first member starts at byte 0, so an HTTP range
request plus raw-deflate decompression gives us the first rows without the full file.

Only standard-library code is used, and nothing from the downloaded bytes is executed:
rows are parsed as JSON data and written to data/raw/_probe/.

Usage: python -I scripts/probe_raw.py [--file Win64_test.zip] [--mb 4] [--rows 200]
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request
import zlib
from pathlib import Path

HF_URL = "https://huggingface.co/datasets/joyce8/EMBER2024/resolve/main/{name}"
ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "data" / "raw" / "_probe"


def fetch_prefix(name: str, n_bytes: int) -> bytes:
    """Download only the first n_bytes of a remote file via an HTTP Range request."""
    req = urllib.request.Request(HF_URL.format(name=name), headers={"Range": f"bytes=0-{n_bytes - 1}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def first_member(prefix: bytes) -> tuple[str, bytes]:
    """Parse the first zip local file header and return (member name, decompressed prefix)."""
    sig, _version, _flags, method = struct.unpack("<IHHH", prefix[:10])
    if sig != 0x04034B50:
        raise ValueError("Not a zip local file header")
    name_len, extra_len = struct.unpack("<HH", prefix[26:30])
    name = prefix[30:30 + name_len].decode("utf-8", errors="replace")
    data = prefix[30 + name_len + extra_len:]
    if method == 0:  # stored
        return name, data
    if method != 8:
        raise ValueError(f"Unsupported zip compression method {method}")
    # wbits=-15 means raw deflate stream (zip members have no zlib header).
    return name, zlib.decompressobj(-15).decompress(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="Win64_test.zip")
    parser.add_argument("--mb", type=int, default=4)
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    member, text = first_member(fetch_prefix(args.file, args.mb * 1024 * 1024))
    lines = text.split(b"\n")[:-1]  # last line is almost surely truncated
    rows = [json.loads(line) for line in lines[: args.rows]]
    out = PROBE_DIR / f"{Path(member).stem}_head{len(rows)}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"member={member} decoded_bytes={len(text)} rows_written={len(rows)} -> {out}")


if __name__ == "__main__":
    main()
