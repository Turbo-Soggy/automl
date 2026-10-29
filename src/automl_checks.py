"""Facts about each AutoML run read back from the fitted models (no refit, no test data).

The runners in automl_runs.py write these into each tool's notes JSON ("checks" key) right
after the fit. refresh_notes() recomputes them from the SAVED models and `train`, so notes
written by an older version of the code can be brought up to date without re-running a
20-minute search. It only reads models and `train`; it never touches a test set.

What is checked, and why:
  * seeds      : which seed each tool really used. SEED is passed where a tool takes one, but
                 some parts use the library's own fixed seed (FLAML's estimators and holdout
                 split, AutoGluon's base models), and the write-up must say so.
  * FLAML      : the importance type of the winner and how large the raw feature values are
                 (FLAML does not scale features, which matters for its linear learners).
  * AutoGluon  : for every non-constant column it dropped, how many rows differ from the most
                 common value, so the write-up can say what kind of columns were lost.
  * H2O        : the stacked-ensemble metalearner's coefficients, i.e. which base models the
                 leader really uses.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import LEADERBOARDS, MODELS, RUN_SUFFIX, SEED

SCALE_CHUNK: int = 256  # columns per chunk when scanning X: avoids a second 1.6 GB copy for np.abs


# --- FLAML --------------------------------------------------------------------------------

def importance_type(learner: str, configured: str | None) -> str | None:
    """The importance a FLAML winner reports: its own setting, else the library default.

    XGBoost leaves importance_type=None and then uses "gain" for tree boosters; LightGBM's
    default is "split" (how often a feature is split on).
    """
    if configured:
        return configured
    if learner.startswith("xgb"):
        return "gain"
    return "split" if learner == "lgbm" else None


def column_max_abs(X: np.ndarray) -> np.ndarray:
    """Largest |value| of every column, scanned in chunks."""
    return np.concatenate([np.abs(X[:, i:i + SCALE_CHUNK]).max(axis=0) for i in range(0, X.shape[1], SCALE_CHUNK)])


def feature_scale(X: np.ndarray) -> dict[str, float | int]:
    """How unequal the raw feature ranges are: unscaled inputs are hard for linear learners."""
    max_abs = column_max_abs(X)
    return {"median_column_max_abs": float(np.median(max_abs)), "largest_column_max_abs": float(max_abs.max()),
            "n_columns_max_abs_over_1e3": int((max_abs > 1e3).sum()),
            "n_columns_max_abs_over_1e6": int((max_abs > 1e6).sum())}


def flaml_checks(automl: Any, X: np.ndarray) -> dict[str, Any]:
    """Seeds and importance type of FLAML's winner, and the scale of the features it saw."""
    from flaml.config import RANDOM_SEED  # FLAML's fixed seed for its own data split

    estimator = getattr(automl.model, "estimator", None)
    return {
        "importance_type": importance_type(automl.best_estimator, getattr(estimator, "importance_type", None)),
        "seeds": {"search (seed=)": SEED, "holdout split (flaml.config.RANDOM_SEED)": RANDOM_SEED,
                  "winner model random_state": getattr(estimator, "random_state", None)},
        "feature_scale": feature_scale(X),
    }


# --- AutoGluon ----------------------------------------------------------------------------

def minority_rows(X: np.ndarray, y: np.ndarray, names: list[str], columns: list[str]) -> dict[str, dict[str, Any]]:
    """Per column: rows NOT at the column's most common value, and the malicious share among them."""
    index = {n: i for i, n in enumerate(names)}
    out: dict[str, dict[str, Any]] = {}
    for col in columns:
        values = X[:, index[col]]
        uniques, counts = np.unique(values, return_counts=True)
        rare = values != uniques[counts.argmax()]
        out[col] = {"rows_not_mode": int(rare.sum()), "n_distinct": int(len(uniques)),
                    "malicious_share_of_those": float(y[rare].mean()) if rare.any() else None}
    return out


def _ag_duplicate_sample_sizes(predictor: Any) -> dict[str, int | None]:
    """Row sample sizes AutoGluon's DropDuplicatesFeatureGenerator compared columns on."""
    for stage in getattr(predictor._learner.feature_generator, "generators", []):  # no public accessor
        for gen in stage:
            if type(gen).__name__ == "DropDuplicatesFeatureGenerator":
                return {"sample_size_init": gen.sample_size_init, "sample_size_final": gen.sample_size_final}
    return {}


def ag_checks(predictor: Any, X: np.ndarray, y: np.ndarray, names: list[str],
              dropped_nonconstant: list[str]) -> dict[str, Any]:
    """Seeds of AutoGluon's split and models, and what the non-constant dropped columns look like."""
    trainer = predictor._trainer  # no public accessor for the fitted models; read-only use
    model_seeds = {name: getattr(trainer.load_model(name), "random_seed", None) for name in predictor.model_names()}
    return {
        "seeds": {"train/validation split (learner random_state)": getattr(predictor._learner, "random_state", None),
                  "base models (random_seed)": model_seeds},
        "duplicate_check_sample_sizes": _ag_duplicate_sample_sizes(predictor),
        "dropped_nonconstant_minority_rows": minority_rows(X, y, names, dropped_nonconstant),
        "refit_on_all_rows": any(name.endswith("_FULL") for name in predictor.model_names()),
    }


# --- H2O ----------------------------------------------------------------------------------

def h2o_checks(leader: Any) -> dict[str, Any]:
    """For a stacked-ensemble leader: metalearner coefficients and the base models with weight > 0."""
    meta = leader.metalearner() if hasattr(leader, "metalearner") else None
    if meta is None:
        return {"metalearner_coefficients": None, "effective_base_models": None}
    coef = {k: float(v) for k, v in meta.coef().items()}
    return {"metalearner": meta.model_id, "metalearner_coefficients": coef,
            "metalearner_non_negative": meta.params.get("non_negative", {}).get("actual"),
            "effective_base_models": [k for k, v in coef.items() if k != "Intercept" and v != 0.0]}


def short_model_id(model_id: str) -> str:
    """"GBM_1_AutoML_1_20261029_80743" -> "GBM_1": the AutoML project suffix carries no information."""
    return re.sub(r"_AutoML_.*$", "", model_id)


def h2o_interpretability(leader_algo: str, base_models: list[str], effective: list[str] | None) -> str:
    """What can be explained about H2O's leader, using the metalearner weights when known."""
    if leader_algo != "stackedensemble":
        return f"single {leader_algo} model with variable importance"
    if effective is not None and len(effective) == 1:
        return (f"stacked ensemble of {len(base_models)} base models, but its GLM metalearner gives weight only "
                f"to {short_model_id(effective[0])}: in effect one model, explained by that model's variable importance")
    if effective:
        return (f"stacked ensemble: GLM metalearner over {len(effective)} of {len(base_models)} base models "
                f"({', '.join(short_model_id(m) for m in effective)}); variable importance only for single base models")
    return (f"stacked ensemble of {len(base_models)} base models with a GLM metalearner; "
            f"variable importance only for single base models")


def interpretability_from_notes(tool: str, notes: dict[str, Any]) -> str | None:
    """A sharper interpretability note than the logged one, when the checks allow it (else None).

    Used by tables.py so records logged before the metalearner check existed still get it.
    """
    effective = (notes.get("checks") or {}).get("effective_base_models")
    if tool != "h2o" or effective is None:
        return None
    return h2o_interpretability(str(notes.get("leader_algo")), notes.get("base_models", []), effective)


# --- Refresh saved notes from the saved models ---------------------------------------------

# Notes keys renamed after the full run. Log records in experiments.jsonl keep the old key
# (the log is never edited); only the notes JSON is updated.
RENAMED_KEYS: dict[str, dict[str, str]] = {
    "h2o": {"leader_cv_auc": "leader_leaderboard_auc"},  # no CV ran: it is the leaderboard-frame AUC
}


def _notes_path(folder: Path, tool: str) -> Path:
    return folder / f"{tool}_preprocessing{RUN_SUFFIX}.json"


def _apply(notes: dict[str, Any], updates: dict[str, Any], tool: str) -> dict[str, Any]:
    """Merge recomputed fields into the notes and record what changed (old and new value)."""
    changed: dict[str, Any] = {}
    for old, new in RENAMED_KEYS.get(tool, {}).items():
        if old in notes:
            notes[new] = notes.pop(old)
            changed[old] = {"renamed_to": new}
    for key, value in updates.items():
        if key != "checks" and notes.get(key) != value:
            changed[key] = {"old": notes.get(key), "new": value}
        notes[key] = value
    history = notes.setdefault("refreshed", {"changed": {}})
    history["changed"].update(changed)
    history["at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history["by"] = "src/automl_checks.refresh_notes (saved models + train; no refit, no test data)"
    return notes


def _flaml_updates(X: np.ndarray) -> dict[str, Any]:
    import joblib

    # Our own file, written by run_flaml() in this project (joblib = pickle: never load untrusted ones).
    automl = joblib.load(MODELS / "flaml" / f"automl{RUN_SUFFIX}.joblib")
    checks = flaml_checks(automl, X)
    return {"importance_type": checks["importance_type"], "checks": checks}


def _ag_updates(X: np.ndarray, y: np.ndarray, names: list[str], notes: dict[str, Any]) -> dict[str, Any]:
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(str(MODELS / f"autogluon{RUN_SUFFIX}"), check_packages=False, verbosity=0)
    dropped = sorted(notes.get("dropped_nonconstant_duplicate_of", {}))
    return {"checks": ag_checks(predictor, X, y, names, dropped)}


def _h2o_updates(notes: dict[str, Any]) -> dict[str, Any]:
    import h2o

    h2o.init(max_mem_size="8G", nthreads=4)  # only loads one saved model: a small cluster is enough
    h2o.no_progress()
    try:
        leader = h2o.load_model(notes["saved_model"]["binary"])
        return {"checks": h2o_checks(leader)}
    finally:
        h2o.cluster().shutdown()


def refresh_notes(tools: list[str], folder: Path = LEADERBOARDS) -> list[Path]:
    """Recompute the derived notes fields of `tools` from their saved models; returns the files written."""
    from src.automl_runs import feature_names  # local import: automl_runs imports this module
    from src.data import load_split

    X, y, _ = load_split("train")
    names = feature_names()
    written = []
    for tool in tools:
        path = _notes_path(folder, tool)
        if not path.exists():
            print(f"[refresh] {path} missing; skipped")
            continue
        notes = json.loads(path.read_text(encoding="utf-8"))
        if tool == "flaml":
            updates = _flaml_updates(X)
        elif tool == "autogluon":
            updates = _ag_updates(X, y, names, notes)
        else:
            updates = _h2o_updates(notes)
        path.write_text(json.dumps(_apply(notes, updates, tool), indent=2, default=str), encoding="utf-8")
        written.append(path)
    return written
