"""Checks on reading the weekly raw jsonl straight from a zip (scripts/download_data.py --keep-zipped).

The cache, the byte offsets in meta (`source_offset`) and so the splits must be the
same whether a week is read from an extracted file or from its zip member. These
tests build a tiny zip in a temp folder, so they need no real data and never skip.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest

WEEK = "2023-09-24_2023-09-30_Win64_train.jsonl"
LINES = [b'{"sha256": "a", "label": 0}\n', b'{"sha256": "b", "label": 1}\n', b'{"sha256": "c", "label": 1}\n']


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Write a zip with the given member names and contents."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


@pytest.mark.parametrize("hostile", ["../x.jsonl", "/x.jsonl", "C:x.jsonl", "a/../../x.jsonl", "..\\x.jsonl"])
def test_flat_jsonl_name_refuses_paths_outside_raw(data: Any, hostile: str) -> None:
    """Zip-slip guard: a member that points outside data/raw is refused, not flattened."""
    with pytest.raises(RuntimeError, match="suspicious"):
        data.flat_jsonl_name(zipfile.ZipInfo(hostile))


def test_flat_jsonl_name_flattens_and_skips_non_jsonl(data: Any) -> None:
    assert data.flat_jsonl_name(zipfile.ZipInfo("sub/ok.jsonl")) == "ok.jsonl"
    assert data.flat_jsonl_name(zipfile.ZipInfo("readme.txt")) is None
    assert data.flat_jsonl_name(zipfile.ZipInfo("folder/")) is None


def test_zip_member_reads_like_extracted_file(data: Any, tmp_path: Path) -> None:
    """Same bytes, and seek(offset)+readline gives the same line (verify_data.py relies on it)."""
    content = b"".join(LINES)
    zip_path = make_zip(tmp_path / "Win64_train.zip", {f"train/{WEEK}": content})
    extracted = tmp_path / "win64"
    extracted.mkdir()

    zipped = data.raw_files(extracted, "train", zip_path=zip_path)
    assert [r.name for r in zipped] == [WEEK] and zipped[0].zip_path == zip_path
    (extracted / WEEK).write_bytes(content)
    plain = data.raw_files(extracted, "train", zip_path=zip_path)
    assert plain[0].path == extracted / WEEK  # extracted files take precedence over the zip
    assert zipped[0].size == plain[0].size == len(content)

    offset = len(LINES[0])
    for raw in (zipped[0], plain[0]):
        with raw.open() as f:
            assert f.read() == content
        with raw.open() as f:
            f.seek(offset)
            assert f.readline() == LINES[1]


def test_partial_extraction_is_an_error(data: Any, tmp_path: Path) -> None:
    """A half-extracted folder must not silently hide the weeks that are only in the zip."""
    week2 = WEEK.replace("2023-09-24_2023-09-30", "2023-10-01_2023-10-07")
    zip_path = make_zip(tmp_path / "Win64_train.zip", {WEEK: LINES[0], week2: LINES[1]})
    extracted = tmp_path / "win64"
    extracted.mkdir()
    (extracted / WEEK).write_bytes(LINES[0])
    with pytest.raises(RuntimeError, match="partial extraction"):
        data.raw_files(extracted, "train", zip_path=zip_path)


def test_duplicate_flat_names_are_an_error(data: Any, tmp_path: Path) -> None:
    zip_path = make_zip(tmp_path / "Win64_train.zip", {f"a/{WEEK}": LINES[0], f"b/{WEEK}": LINES[1]})
    with pytest.raises(RuntimeError, match="same file name"):
        data.raw_files(tmp_path, "train", zip_path=zip_path)
