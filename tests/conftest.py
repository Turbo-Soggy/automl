"""Shared pytest setup: make the project importable and load each split once.

Split tests need the cache and the persisted split files, which take a long
time to build. When those files are missing, the tests SKIP with a reason
instead of failing (or, worse, silently building the splits inside pytest).
Set EMBER_QUICK=1 before running pytest to check the quick-mode splits instead.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so `config` and `src` import without installing the project

import config  # noqa: E402

SPLIT_NAMES: tuple[str, ...] = ("train", "test_random", "test_future")


def data_module() -> Any:
    """Import src.data, or skip if that module has not been written yet.

    Only a missing src.data is a skip; an ImportError raised *inside* it is a real
    bug and must fail the test run.
    """
    try:
        from src import data
    except ModuleNotFoundError as exc:
        if exc.name in ("src.data", "src"):
            pytest.skip("src/data.py does not exist yet")
        raise
    return data


def _run_files(folder: Path) -> list[Path]:
    """Files in `folder` that belong to the current mode (quick files carry the _quick suffix)."""
    if not folder.is_dir():
        return []
    files = [p for p in folder.rglob("*") if p.is_file() and not p.name.startswith(".")]
    return [p for p in files if ("_quick" in p.stem) == config.QUICK]


def require_persisted_splits() -> None:
    """Skip unless the feature cache and the split files for this mode exist on disk.

    Checked before calling src.data, so pytest never triggers an expensive build.
    """
    mode = "quick" if config.QUICK else "full"
    cache_files = list(config.DATA_CACHE.glob("[!.]*")) if config.DATA_CACHE.is_dir() else []
    if not cache_files:
        pytest.skip(f"feature cache {config.DATA_CACHE} is empty; run `run_all.py --steps data` first")
    if not _run_files(config.SPLITS_DIR):
        pytest.skip(f"no {mode}-mode split files in {config.SPLITS_DIR}; run `run_all.py --steps data` first")


@pytest.fixture(scope="session")
def data() -> Any:
    """The src.data module (skips if missing)."""
    return data_module()


@pytest.fixture(scope="session")
def splits(data: Any) -> dict[str, tuple[Any, Any, Any]]:
    """{name: (X, y, meta)} for all three splits, loaded once per test session."""
    require_persisted_splits()
    try:
        return {name: data.load_split(name) for name in SPLIT_NAMES}
    except FileNotFoundError as exc:
        pytest.skip(f"split files not built yet: {exc}")
