"""Checks on the persisted splits: no leakage, correct time order, correct sizes.

These are the guarantees the whole comparison rests on. If any of them breaks,
test scores are not trustworthy, so the checks run against the files on disk
(the exact data every method uses), not against a freshly computed split.
All tests skip cleanly while the cache or split files do not exist yet.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import pytest

import config

# A stratified split of 50/50 data should keep both classes within this margin.
BALANCE_TOLERANCE = 0.01  # 1 percentage point
REQUIRED_META_COLUMNS = (
    "sha256", "label", "first_submission_date", "week",
    "detection_ratio", "detection_ratio_float", "family", "file_size",
)

Splits = dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]


def expected_sizes() -> dict[str, int]:
    """Split sizes implied by config (quick mode included)."""
    n_test_random = int(round(config.SUBSAMPLE_N * config.TEST_FRACTION))
    return {
        "train": config.SUBSAMPLE_N - n_test_random,
        "test_random": n_test_random,
        "test_future": config.TEST_FUTURE_N,
    }


def positive_rate(y: np.ndarray) -> float:
    return float(np.mean(y == 1))


def assert_partition(folds: list[tuple[np.ndarray, np.ndarray]], n: int) -> None:
    """Every row is validated exactly once, and never trained on in its own fold."""
    assert len(folds) == config.CV_FOLDS
    all_val = np.concatenate([val for _, val in folds])
    assert np.array_equal(np.sort(all_val), np.arange(n)), "validation folds do not cover 0..n-1 exactly once"
    for k, (tr, val) in enumerate(folds):
        assert np.intersect1d(tr, val).size == 0, f"fold {k}: train and validation overlap"
        assert tr.size + val.size == n, f"fold {k}: train + validation != {n} rows"


def assert_stratified(folds: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> None:
    """Each validation fold keeps the class balance of the data it was cut from."""
    overall = positive_rate(y)
    for k, (_, val) in enumerate(folds):
        assert abs(positive_rate(y[val]) - overall) <= BALANCE_TOLERANCE, f"fold {k} is not stratified"


# --- Shape and alignment ---------------------------------------------------

@pytest.mark.parametrize("name", ["train", "test_random", "test_future"])
def test_split_is_row_aligned(splits: Splits, name: str) -> None:
    """X, y and meta describe the same rows in the same order."""
    X, y, meta = splits[name]
    assert X.ndim == 2 and X.dtype == np.float32
    assert y.dtype == np.int8
    assert X.shape[0] == len(y) == len(meta)
    assert np.array_equal(y, meta["label"].to_numpy().astype(np.int8)), "y and meta.label disagree"
    assert set(np.unique(y)) <= {0, 1}, "labels other than 0/1 (e.g. -1 unlabelled) present"
    missing = [c for c in REQUIRED_META_COLUMNS if c not in meta.columns]
    assert not missing, f"meta lacks contract columns {missing}"
    assert pd.api.types.is_datetime64_any_dtype(meta["first_submission_date"])


@pytest.mark.parametrize("name", ["train", "test_random", "test_future"])
def test_split_sizes_match_config(splits: Splits, name: str) -> None:
    assert len(splits[name][1]) == expected_sizes()[name]


# --- No leakage between splits ---------------------------------------------

@pytest.mark.parametrize("name", ["train", "test_random", "test_future"])
def test_no_duplicate_sha256_within_split(splits: Splits, name: str) -> None:
    """Raw weekly files repeat every sample twice; the cache must have dropped the copies."""
    sha = splits[name][2]["sha256"]
    assert not sha.duplicated().any(), f"{sha.duplicated().sum()} repeated sha256 in {name}"


@pytest.mark.parametrize("a,b", list(combinations(["train", "test_random", "test_future"], 2)))
def test_no_sha256_overlap_between_splits(splits: Splits, a: str, b: str) -> None:
    """The same file must never be in two splits, or test scores measure memorisation."""
    shared = set(splits[a][2]["sha256"]) & set(splits[b][2]["sha256"])
    assert not shared, f"{len(shared)} sha256 shared by {a} and {b}, e.g. {sorted(shared)[:3]}"


def test_no_cache_row_overlap_train_test_random(data: Any, splits: Splits) -> None:
    """train and test_random are cut from the same training-week cache, so their rows must be disjoint.

    test_future comes from the separate test-week cache, so its row numbers live in
    another index space; the sha256 and date checks cover it instead.
    """
    assert data.SPLIT_SOURCE["train"] == data.SPLIT_SOURCE["test_random"] != data.SPLIT_SOURCE["test_future"]
    train_rows, test_rows = data.split_rows("train"), data.split_rows("test_random")
    assert len(np.unique(train_rows)) == len(train_rows), "train repeats a cache row"
    assert len(np.unique(test_rows)) == len(test_rows), "test_random repeats a cache row"
    assert np.intersect1d(train_rows, test_rows).size == 0, "train and test_random share cache rows"


# --- Temporal order ---------------------------------------------------------

def test_future_is_strictly_after_train(splits: Splits) -> None:
    """Every test_future sample was first seen after every training sample."""
    train_meta, future_meta = splits["train"][2], splits["test_future"][2]
    latest_train = train_meta["first_submission_date"].max()
    earliest_future = future_meta["first_submission_date"].min()
    assert earliest_future > latest_train, f"test_future starts {earliest_future}, train ends {latest_train}"
    assert future_meta["week"].min() > train_meta["week"].max()


# --- Class balance -----------------------------------------------------------

def test_class_balance_train_vs_test_random(splits: Splits) -> None:
    """The stratified 80/20 split keeps both sides at the same (50/50) balance."""
    p_train, p_test = positive_rate(splits["train"][1]), positive_rate(splits["test_random"][1])
    assert abs(p_train - p_test) <= BALANCE_TOLERANCE, f"train {p_train:.4f} vs test_random {p_test:.4f}"
    for name, p in (("train", p_train), ("test_random", p_test)):
        assert abs(p - 0.5) <= BALANCE_TOLERANCE, f"{name} malicious share {p:.4f} is not ~50%"


# --- Tuning subset and CV folds ---------------------------------------------

def test_tune_subset_is_valid_and_stratified(data: Any, splits: Splits) -> None:
    y_train = splits["train"][1]
    idx = np.asarray(data.tune_subset_indices())
    assert len(idx) == config.TUNE_SUBSET_N
    assert len(np.unique(idx)) == len(idx), "tuning subset repeats rows"
    assert idx.min() >= 0 and idx.max() < len(y_train), "tuning subset indexes outside train"
    assert abs(positive_rate(y_train[idx]) - positive_rate(y_train)) <= BALANCE_TOLERANCE


def test_cv_folds_partition_tune_subset(data: Any, splits: Splits) -> None:
    """Folds used by every tuning method: an exact, stratified partition of the tuning subset."""
    y_subset = splits["train"][1][np.asarray(data.tune_subset_indices())]
    folds = data.get_cv_folds(subset=True)
    assert_partition(folds, config.TUNE_SUBSET_N)
    assert_stratified(folds, y_subset)


def test_cv_folds_partition_full_train(data: Any, splits: Splits) -> None:
    y_train = splits["train"][1]
    folds = data.get_cv_folds()
    assert_partition(folds, len(y_train))
    assert_stratified(folds, y_train)


def test_cv_folds_are_persisted_and_identical(data: Any, splits: Splits) -> None:
    """Two calls give the same folds, so Grid, Random and Optuna really share them."""
    first, second = data.get_cv_folds(subset=True), data.get_cv_folds(subset=True)
    for (tr1, val1), (tr2, val2) in zip(first, second, strict=True):
        assert np.array_equal(tr1, tr2) and np.array_equal(val1, val2)
    assert np.array_equal(data.tune_subset_indices(), data.tune_subset_indices())


# --- Feature groups ----------------------------------------------------------

def test_feature_groups_cover_all_columns_without_gaps(data: Any) -> None:
    """Families tile [0, D) exactly, with D the thrember extractor's dimension."""
    import thrember

    groups = data.feature_groups()
    names = [name for name, _, _ in groups]
    assert len(names) == len(set(names)), "duplicate family names"
    assert groups[0][1] == 0, "first family does not start at column 0"
    for (prev_name, _, prev_end), (name, start, end) in zip(groups, groups[1:]):
        assert start == prev_end, f"gap or overlap between {prev_name} and {name}"
    for name, start, end in groups:
        assert end > start, f"family {name} is empty"
    assert groups[-1][2] == thrember.PEFeatureExtractor().dim


def test_feature_groups_match_split_width(data: Any, splits: Splits) -> None:
    D = data.feature_groups()[-1][2]
    for name, (X, _, _) in splits.items():
        assert X.shape[1] == D, f"{name} has {X.shape[1]} columns, feature groups end at {D}"
