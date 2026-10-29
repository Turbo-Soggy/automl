"""Project-wide constants: seed, paths and experiment sizes.

Every module imports from here so that one edit changes the whole pipeline.
Set the environment variable EMBER_QUICK=1 for a fast smoke run on tiny samples;
quick runs write to "_quick" files so they never overwrite real results.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --- Reproducibility -------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy RNGs.

    Libraries (LightGBM, Optuna, sklearn, AutoML tools) still get `seed` passed
    explicitly, because they keep their own RNGs and ignore the global ones.
    PYTHONHASHSEED only affects child processes started after this call
    (the current interpreter's hash seed is fixed at start-up).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --- Run mode --------------------------------------------------------------
QUICK: bool = os.environ.get("EMBER_QUICK") == "1"
# Appended to split and log file names so quick runs live in separate files.
RUN_SUFFIX: str = "_quick" if QUICK else ""

# --- Paths -----------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_CACHE: Path = ROOT / "data" / "cache"
SPLITS_DIR: Path = ROOT / "data" / "splits"
REPORTS: Path = ROOT / "reports"
FIGURES: Path = REPORTS / "figures"
TABLES: Path = REPORTS / "tables"
LEADERBOARDS: Path = REPORTS / "leaderboards"
PREDICTIONS: Path = REPORTS / "predictions"  # saved test-set scores, so CIs are computed from artefacts
MODELS: Path = ROOT / "models"
EXPERIMENT_LOG: Path = REPORTS / f"experiments{RUN_SUFFIX}.jsonl"


def split_file(stem: str, ext: str = ".npy") -> Path:
    """Path of a persisted split artefact, e.g. split_file("cv_folds") -> data/splits/cv_folds[_quick].npy."""
    return SPLITS_DIR / f"{stem}{RUN_SUFFIX}{ext}"


# --- Experiment sizes ------------------------------------------------------
SUBSAMPLE_N: int = 200_000      # random-split pool drawn from the 52 training weeks
TEST_FUTURE_N: int = 40_000     # stratified sample of the 12 future test weeks
TEST_FRACTION: float = 0.2      # share of the pool held out as test_random
TUNE_SUBSET_N: int = 60_000     # tuning runs on this subset of train, then refits on all of train
TUNE_TRIALS: int = 36           # configurations per tuning method (equal budget)
CV_FOLDS: int = 3
AUTOML_BUDGET_S: int = 1200     # seconds per AutoML tool
# Step 4 leakage reruns: 600 s per AutoML run, a stated deviation from the 1200 s of Step 3
# (9 reruns at 1200 s would take about 4 hours). Clean reruns get the same 600 s, so every
# with/without-leak comparison is at an equal budget.
LEAKAGE_BUDGET_S: int = 600
# All tools use the same cores so timings are comparable. Steps 2-3 ran on a 16-core laptop;
# a smaller machine (Step 4 onwards ran on a 4-core cloud VM) uses all of its cores instead.
N_JOBS: int = min(16, os.cpu_count() or 1)
N_BOOTSTRAP: int = 1000         # resamples for the paired bootstrap confidence intervals

if QUICK:
    SUBSAMPLE_N = 10_000
    TEST_FUTURE_N = 2_000
    TUNE_SUBSET_N = 5_000
    TUNE_TRIALS = 4
    AUTOML_BUDGET_S = 60
    LEAKAGE_BUDGET_S = 30
