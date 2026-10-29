"""Step 5: export the deployment model to models/final/ with a self-describing schema.json.

The deployed model is the Step 2 tuned LightGBM, the one models/threshold.json was fitted for
(the tuner Step 2 selected on CV). This script does not train anything. It
  1. copies that booster to models/final/model.txt and records its sha256,
  2. checks the copy reproduces the saved test_future scores of Step 2
     (reports/predictions/<method>.npz) to 1e-6, so the API serves the model that was scored,
  3. writes models/final/schema.json: feature dimension, column names, group offsets, the
     thrember commit and signify pin, the threshold, the training data, the metrics row,
  4. writes MODEL_CARD.md from schema.json and the Step 2-4 tables (app/model_card.py),
  5. writes models/final/demo_samples.npz: a few hundred test_future rows (with their scores)
     for the Streamlit demo, so the app also runs where the 5 GB feature cache is absent.

Inference is self-contained: the model reads the raw float32 thrember vector. Tree splits are
invariant to monotone rescaling, so there is no scaler or other preprocessing to ship.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/export_model.py           # Linux: .venv/bin/python
  .venv/Scripts/python.exe scripts/export_model.py --quick   # writes models/final_quick/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import when run as a plain script

DEMO_STRATIFIED_N: int = 300   # label x week stratified rows of test_future for the demo
DEMO_ERRORS_N: int = 50        # plus up to this many false positives and false negatives each
PARITY_TOL: float = 1e-6
# Columns of the comparison table copied into schema.json (generated, never typed).
METRIC_COLUMNS: tuple[str, ...] = (
    "test_random_roc_auc", "test_random_tpr_at_1pct_fpr", "test_random_f1",
    "test_future_roc_auc", "test_future_roc_auc_ci_low", "test_future_roc_auc_ci_high",
    "test_future_tpr_at_1pct_fpr", "test_future_tpr_at_1pct_fpr_ci_low",
    "test_future_tpr_at_1pct_fpr_ci_high", "test_future_f1", "test_future_fpr",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def requirement_line(package: str) -> str | None:
    """The pin of `package` in requirements.txt, e.g. "signify==0.8.1"."""
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.split("==")[0].split(" @")[0].strip().lower() == package:
            return line.strip()
    return None


def tuned_method(model_rel: str) -> str:
    """"models/lgbm_optuna.txt" -> "optuna": the Step 2 method name used in the tables."""
    import config
    stem = Path(model_rel).stem.removesuffix(config.RUN_SUFFIX)
    return stem.removeprefix("lgbm_")


def metrics_row() -> dict[str, float]:
    """The tuned LightGBM row of the Step 3 comparison table (both test sets, bootstrap CIs)."""
    import pandas as pd
    import config
    from src.tables import TUNED_LABEL
    table = pd.read_csv(config.TABLES / f"automl_vs_manual{config.RUN_SUFFIX}.csv")
    row = table.loc[table["method"] == TUNED_LABEL].iloc[0]
    return {c: float(row[c]) for c in METRIC_COLUMNS}


def parity_check(booster: Any, X: np.ndarray, method: str) -> tuple[dict[str, Any], np.ndarray]:
    """(check record, test_future scores): max |exported model - saved Step 2 score| over all rows."""
    import config
    saved = np.load(config.PREDICTIONS / f"{method}{config.RUN_SUFFIX}.npz", allow_pickle=False)
    p = booster.predict(X, num_threads=config.N_JOBS)
    diff = float(np.max(np.abs(p - saved["test_future_p"])))
    if diff > PARITY_TOL:
        raise SystemExit(f"exported model does not reproduce the saved test_future scores (max diff {diff:.2e})")
    return {"against": f"reports/predictions/{method}{config.RUN_SUFFIX}.npz (test_future_p)",
            "rows": int(len(p)), "max_abs_diff": diff, "tolerance": PARITY_TOL}, p


def demo_positions(y: np.ndarray, p: np.ndarray, meta: Any, threshold: float) -> np.ndarray:
    """test_future positions for the demo: a stratified sample plus some errors at the threshold.

    The errors are picked only so the demo can show a misclassified file; nothing is fitted
    or tuned on them.
    """
    from config import SEED
    from src.explain import stratified_sample
    rng = np.random.default_rng(SEED)
    picked = [stratified_sample(meta, DEMO_STRATIFIED_N)]
    for wrong in ((y == 0) & (p >= threshold), (y == 1) & (p < threshold)):
        idx = np.flatnonzero(wrong)
        picked.append(rng.choice(idx, size=min(DEMO_ERRORS_N, len(idx)), replace=False))
    return np.unique(np.concatenate(picked))


def training_data() -> dict[str, Any]:
    import config
    info = json.loads((config.SPLITS_DIR / f"splits_info{config.RUN_SUFFIX}.json").read_text(encoding="utf-8"))
    return {
        "dataset": "EMBER2024 (thrember), Win64 PE files, weekly releases",
        "description": ("trained on the 'train' split: files from weeks 0-51, deduplicated by sha256, "
                        "stratified by label x week, classes about 50/50 by construction. test_random is "
                        "held out from the same weeks; test_future is weeks 52-63, strictly later."),
        "splits": info["splits"],
        "seed": info["seed"],
    }


def export(quick: bool) -> Path:
    import lightgbm as lgb
    import config
    from app.model_card import write_model_card
    from src import data
    from src.explain import column_names, readable_names

    suffix = config.RUN_SUFFIX
    out_dir = config.MODELS / f"final{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = json.loads((config.MODELS / f"threshold{suffix}.json").read_text(encoding="utf-8"))
    source = ROOT / thr["model"]
    method = tuned_method(thr["model"])
    model_path = out_dir / "model.txt"
    shutil.copyfile(source, model_path)   # byte copy: the sha256 matches the Step 2 file
    model_sha = sha256_of(model_path)
    booster = lgb.Booster(model_file=str(model_path))

    groups = data.feature_groups()
    D = groups[-1][2]
    if booster.num_feature() != D:
        raise SystemExit(f"model expects {booster.num_feature()} features, thrember gives {D}")

    X, y, meta = data.load_split("test_future")
    parity, p = parity_check(booster, X, method)
    pos = demo_positions(y, p, meta, float(thr["threshold"]))
    np.savez_compressed(out_dir / "demo_samples.npz", X=X[pos], y=y[pos], p=p[pos], position=pos,
                        sha256=meta["sha256"].to_numpy()[pos].astype("U64"),
                        week=meta["week"].to_numpy()[pos])

    params = json.loads(source.with_name(f"{source.stem}_params.json").read_text(encoding="utf-8"))
    schema = {
        "model_version": f"lgbm-{method}-{model_sha[:12]}",
        "model_file": "model.txt",
        "model_sha256": model_sha,
        "model_type": "LightGBM gradient-boosted trees, binary objective",
        "lightgbm_version": metadata.version("lightgbm"),
        "num_trees": booster.num_trees(),
        "params": params["params"],
        "source_model": thr["model"],
        "selected_by": ("Step 2 tuned LightGBM (tuner chosen by mean CV AUC on train only); "
                        "see MODEL_CARD.md for why it is deployed over the AutoML models"),
        "preprocessing": ("none: the model reads the raw float32 thrember vector; tree splits do not "
                          "need scaling, so no scaler is shipped"),
        "feature_dim": D,
        "feature_groups": [{"name": n, "start": s, "end": e} for n, s, e in groups],
        "column_names": column_names(groups),
        "feature_names": readable_names(data.get_extractor()),
        "thrember": {"version": metadata.version("thrember"), "commit": data._thrember_commit(),
                     "requirement": requirement_line("thrember")},
        "signify": {"version": metadata.version("signify"), "requirement": requirement_line("signify")},
        "threshold": {k: thr[k] for k in ("threshold", "cost_ratio_fn_to_fp", "rule", "how_chosen",
                                           "source_of_predictions", "caveat")},
        "threshold_source": f"models/threshold{suffix}.json",
        "training_data": training_data(),
        "metrics_source": f"reports/tables/automl_vs_manual{suffix}.csv (method=lgbm_tuned)",
        "metrics": metrics_row(),
        "parity_check": parity,
        "demo_samples": {"file": "demo_samples.npz", "rows": int(len(pos)),
                         "how": (f"{DEMO_STRATIFIED_N} label x week stratified test_future rows plus up to "
                                 f"{DEMO_ERRORS_N} false positives and {DEMO_ERRORS_N} false negatives at the "
                                 "threshold (seed 42); 'position' is the row index inside test_future")},
        "seed": config.SEED,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # newline="\n": the same bytes on Windows and Linux, so a re-export does not rewrite every line
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[export] {model_path.relative_to(ROOT)} sha256={model_sha[:12]} trees={booster.num_trees()} "
          f"parity max diff={parity['max_abs_diff']:.1e} demo rows={len(pos)}")
    # The model card is generated from schema.json and the tables, so its numbers are never typed.
    card = write_model_card(schema, config.TABLES, out_dir / "MODEL_CARD.md" if quick else ROOT / "MODEL_CARD.md",
                            suffix)
    print(f"[export] wrote {card.relative_to(ROOT)}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="export the quick model to models/final_quick/")
    args = parser.parse_args(argv)
    if args.quick:
        os.environ["EMBER_QUICK"] = "1"   # before importing config, which reads it at import time
    export(args.quick)


if __name__ == "__main__":
    main()
