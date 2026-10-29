"""Step 2 driver: baselines, Grid / Random / Optuna tuning, then the tables and figures.

Methods run one after another (never in parallel) so each one's wall-clock time is measured
on an otherwise idle machine with all N_JOBS cores.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/run_tuning.py                       # everything, real sizes
  .venv/Scripts/python.exe scripts/run_tuning.py --quick               # smoke-test sizes
  .venv/Scripts/python.exe scripts/run_tuning.py --methods grid,optuna
  .venv/Scripts/python.exe scripts/run_tuning.py --tables-only         # rebuild outputs from the log
  .venv/Scripts/python.exe scripts/run_tuning.py --refit-only          # reuse logged searches, refit + score
  ... --output-root DIR   writes the log, models, tables and figures under DIR instead of
                          reports/ and models/ (used for synthetic smoke runs).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import when run as a plain script

ALL_METHODS = ("baseline", "grid", "random", "optuna")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default=",".join(ALL_METHODS),
                        help=f"comma-separated subset of {','.join(ALL_METHODS)}")
    parser.add_argument("--quick", action="store_true", help="smoke-test sizes (sets EMBER_QUICK=1)")
    parser.add_argument("--tables-only", action="store_true", help="skip training; rebuild tables and figures")
    parser.add_argument("--refit-only", action="store_true",
                        help="do not search again: rebuild each tuner's latest search from the log, then refit "
                             "and score it (baselines have no search and are always re-run, ~90 s)")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="write log, models, tables and figures under this folder")
    args = parser.parse_args(argv)
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(args.methods) - set(ALL_METHODS)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        # config reads EMBER_QUICK at import time, so it must be set before the imports below.
        os.environ["EMBER_QUICK"] = "1"

    import config
    from src import tables, tune

    if args.quick and not config.QUICK:
        raise RuntimeError("--quick given but config was imported before EMBER_QUICK was set")
    config.set_seed(config.SEED)

    if args.output_root is None:
        out = tune.OutputPaths()
        table_paths = tables.TablePaths()
    else:
        root = args.output_root
        log = root / config.EXPERIMENT_LOG.name
        out = tune.OutputPaths(log=log, models=root / "models", predictions=root / "predictions")
        table_paths = tables.TablePaths(log=log, tables=root / "tables", figures=root / "figures",
                                        predictions=root / "predictions", reports=root)

    if not args.tables_only:
        run_methods(args.methods, out, refit_only=args.refit_only)

    for path in tables.build_tuning_outputs(table_paths):
        print(f"wrote {path}")


def run_methods(methods: list[str], out, refit_only: bool = False) -> None:
    """Load train once, then run each requested method in order."""
    from config import N_JOBS, QUICK, TUNE_TRIALS
    from src import tune

    print(f"QUICK={QUICK} TUNE_TRIALS={TUNE_TRIALS} refit_only={refit_only} "
          f"refit_version={tune.REFIT_VERSION} N_JOBS={N_JOBS} log={out.log}", flush=True)
    X_train, y_train = tune.load_train()
    tuning = tune.load_tuning_data(X_train, y_train)
    print(f"train {X_train.shape}, tuning subset {tuning.X.shape}, {len(tuning.folds)} folds", flush=True)

    for method in methods:
        # A start line per method: if the process is killed from outside (which no Python code
        # can catch), the last "start" without a matching "done" shows where it stopped.
        print(f"[{method}] start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        start = time.perf_counter()
        try:
            _run_one(method, X_train, y_train, tuning, out, refit_only)
        except Exception:
            # Write the traceback to stdout too, so it lands in the run log even when only
            # stdout was redirected; then re-raise so the exit code still says "failed".
            print(f"[{method}] FAILED after {time.perf_counter() - start:.1f}s", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise
        print(f"[{method}] done in {time.perf_counter() - start:.1f}s", flush=True)
        gc.collect()  # free the previous method's fold Datasets before timing the next one


def _run_one(method: str, X_train, y_train, tuning, out, refit_only: bool) -> None:
    """One method: baselines, or a tuner's search (or its logged search) followed by refit + scoring."""
    from config import TUNE_TRIALS
    from src import tune

    if method == "baseline":
        for rec in tune.run_baselines(X_train, y_train, tuning, out):
            _print_final(rec)
        return
    if refit_only:
        result = tune.search_from_log(method, out.log)
        # The logged search must have run on exactly the folds we have now.
        if result.folds_hash != tune.folds_fingerprint(tuning.folds):
            raise RuntimeError(f"[{method}] logged search used different CV folds; re-run the search")
        print(f"[{method}] reusing logged search {result.search_id}; refit with "
              f"{tune.refit_rounds(result.best[1])} trees", flush=True)
    else:
        result = tune.run_search(method, tuning, TUNE_TRIALS, out)
    _print_final(tune.refit_and_evaluate(result, X_train, y_train, out))


def _print_final(rec: dict) -> None:
    print(f"[{rec['method']}] cv_auc={rec['cv_auc_mean']:.4f} "
          f"test_random_auc={rec['test_random']['roc_auc']:.4f} "
          f"test_future_auc={rec['test_future']['roc_auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
