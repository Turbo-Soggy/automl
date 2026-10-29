"""Step 4 part B driver: random vs temporal evaluation, bias and imbalance, drift, challenge set.

Stages (run in this order; each one only reads what the stages before it saved):
  fits       the extra LightGBM/LogReg fits on `train` (stratified and forward CV, drift windows,
             class weights, LogReg without time proxies, out-of-fold baselines). Each fit saves its
             scores to reports/predictions/eval/ and logs one record, so a crashed run resumes.
             This is the slow stage (about 1.5 h on 2 threads of the 4-core cloud VM).
  challenge  vectorise the Win64 rows of the EMBER2024 challenge set into data/cache/ (new files;
             the main cache is never touched) and score every saved model on them.
  tables     every CSV and figure, from the saved scores and models only.
  writeup    reports/temporal_bias.md, from the tables only.

Usage (from the project root; python = .venv/Scripts/python.exe on Windows, .venv/bin/python on Linux):
  python scripts/run_eval_bias.py                       # all stages
  python scripts/run_eval_bias.py --quick               # smoke run on the quick splits (_quick files)
  python scripts/run_eval_bias.py --stages tables,writeup
Run it as a detached process for the fits stage; it prints one line per finished fit.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import when run as a plain script

ALL_STAGES: tuple[str, ...] = ("fits", "challenge", "tables", "writeup")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default=",".join(ALL_STAGES), help=f"comma-separated subset of {','.join(ALL_STAGES)}")
    parser.add_argument("--quick", action="store_true", help="smoke-test splits, _quick outputs (sets EMBER_QUICK=1)")
    args = parser.parse_args(argv)
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(args.stages) - set(ALL_STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    return args


def run_fits() -> None:
    """All Step 4 part B fits, most important first (thresholds and CV before the extras)."""
    import numpy as np

    from src import bias, data, drift, temporal

    X, y, meta = data.load_split("train")
    weeks = meta["week"].to_numpy()
    cfg = temporal.tuned_config()
    print(f"[fits] tuned config from {cfg.tuner}: {cfg.n_rounds} rounds, {temporal.EVAL_N_JOBS} threads", flush=True)
    specs = [
        *temporal.forward_cv_specs(weeks),
        *drift.drift_specs(weeks, y),
        *bias.class_weight_specs(y),
        temporal.FitSpec("logreg_no_time_proxies", "logreg", np.arange(len(y)),
                         drop_columns=tuple(temporal.time_proxy_columns().values()),
                         note="LogReg without the 3 time-proxy columns"),
        *temporal.stratified_cv_specs("lgbm_default", "oof_lgbm_default"),
        *temporal.stratified_cv_specs("logreg", "oof_logreg"),
    ]
    test_sets = None
    for spec in specs:
        if spec.val_pos is None and test_sets is None and not temporal.fit_done(spec.name):
            test_sets = {name: data.load_split(name)[:2] for name in ("test_random", "test_future")}
        temporal.run_fit(spec, X, y, cfg, test_sets)
    # Last: the explain job fits the same stratified OOF models of the tuned config (same folds,
    # parameters and rounds). Reuse its scores when they exist and check out; else fit them here.
    if temporal.shared_oof_folds(y, cfg) is not None:
        print(f"[fits] stratified OOF of the tuned config reused from {temporal.SHARED_OOF}", flush=True)
    else:
        for spec in temporal.stratified_cv_specs("lgbm_tuned", "cv_stratified"):
            temporal.run_fit(spec, X, y, cfg, None)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        os.environ["EMBER_QUICK"] = "1"  # must be set before config is imported
    # Thread caps for NumPy's BLAS (LogReg), set before NumPy is first imported: the machine
    # is shared with another Step 4 job.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    from config import set_seed

    set_seed()
    if "fits" in args.stages:
        run_fits()
    if "challenge" in args.stages:
        from src import eval_bias_report

        eval_bias_report.build_challenge()
    if "tables" in args.stages:
        from src import eval_bias_report

        eval_bias_report.build_tables()
    if "writeup" in args.stages:
        from src import temporal_writeup

        print(f"[writeup] {temporal_writeup.write_report()}", flush=True)


if __name__ == "__main__":
    main()
