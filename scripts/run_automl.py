"""Step 3 driver: FLAML, AutoGluon and H2O one after another, then the comparison tables.

Each tool runs in its OWN child process (this script started again with --worker <tool>):
  * the tools run strictly one at a time, so each has the machine to itself,
  * memory a tool leaves behind (AutoGluon models, the H2O JVM) is freed before the next starts,
  * a crash or out-of-memory kill in one tool does not lose the others' results; re-run just
    that tool with --tools <tool>.
A sleep guard (automl_runs.Heartbeat) watches each fit. If the machine slept during it, the worker
logs the result as timing_valid=False, exits with TIMING_INVALID_EXIT, and this driver runs that tool
again (up to MAX_ATTEMPTS times), because a budget that included sleep is not an equal budget.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/run_automl.py                    # all tools, 1200 s each, then tables
  .venv/Scripts/python.exe scripts/run_automl.py --quick            # smoke test: 60 s budgets, _quick files
  .venv/Scripts/python.exe scripts/run_automl.py --tools h2o        # re-run one tool, then rebuild tables
  .venv/Scripts/python.exe scripts/run_automl.py --tables-only      # rebuild tables, figure and write-up
  .venv/Scripts/python.exe scripts/run_automl.py --refresh-notes    # re-derive notes from saved models, then tables

--refresh-notes never refits: it loads each tool's SAVED model and `train`, recomputes the derived
fields of reports/leaderboards/<tool>_preprocessing.json (seeds used, importance type, metalearner
weights, ...; see src/automl_checks.py) and records what changed in the file's "refreshed" key.
It starts a small H2O cluster to load the H2O model, so run it when nothing heavy is running.
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import when run as a plain script

ALL_TOOLS = ("flaml", "autogluon", "h2o")
TIMING_INVALID_EXIT = 3  # worker exit code: finished, but the machine slept during the timed fit
MAX_ATTEMPTS = 3         # per tool, counting only sleep-invalidated attempts; a crash is never retried


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tools", default=",".join(ALL_TOOLS), help=f"comma-separated subset of {','.join(ALL_TOOLS)}")
    parser.add_argument("--quick", action="store_true", help="smoke-test sizes and 60 s budgets (sets EMBER_QUICK=1)")
    parser.add_argument("--tables-only", action="store_true", help="skip the fits; rebuild outputs from the log")
    parser.add_argument("--refresh-notes", action="store_true",
                        help="skip the fits; recompute derived notes from the saved models, then rebuild outputs")
    parser.add_argument("--worker", choices=ALL_TOOLS, help=argparse.SUPPRESS)  # internal: run one tool here
    args = parser.parse_args(argv)
    args.tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    unknown = set(args.tools) - set(ALL_TOOLS)
    if unknown:
        parser.error(f"unknown tools: {sorted(unknown)}")
    return args


def run_worker(tool: str) -> None:
    """Fit one tool on train, save its artefacts, score both test sets and log the result."""
    import config
    from src import automl_runs
    from src.data import load_split

    config.set_seed(config.SEED)
    X, y, _ = load_split("train")  # loading the cache is common to every method, so it is not timed
    print(f"[{tool}] train {X.shape}, budget {config.AUTOML_BUDGET_S} s, n_jobs {config.N_JOBS}, "
          f"quick={config.QUICK}", flush=True)
    # The guard covers the timed part (start-up, conversion, fit) and saving artefacts.
    with automl_runs.Heartbeat(tool) as beat:
        fit = automl_runs.RUNNERS[tool](X, y, config.AUTOML_BUDGET_S, automl_runs.AutomlPaths())
    del X
    gc.collect()  # free train before the test sets are loaded
    try:
        record = automl_runs.evaluate_and_log(fit, config.AUTOML_BUDGET_S, train_rows=len(y),
                                              timing_check=beat.report())
    finally:
        fit.cleanup()  # always shut H2O down, even if scoring failed
    print(f"[{tool}] best={record['best_model']} wall_s={record['wall_s']:.0f} (budget {record['budget_s']}) "
          f"test_random_auc={record['test_random']['roc_auc']:.4f} "
          f"test_future_auc={record['test_future']['roc_auc']:.4f} "
          f"max_heartbeat_gap={record['heartbeat_max_gap_s']:.0f}s timing_valid={record['timing_valid']}",
          flush=True)
    if not record["timing_valid"]:
        raise SystemExit(TIMING_INVALID_EXIT)


def run_tool_once(tool: str, quick: bool) -> int:
    """Run one tool in a child process; returns its exit code."""
    env = dict(os.environ)
    env["EMBER_QUICK"] = "1" if quick else env.get("EMBER_QUICK", "0")
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[{tool}] start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    start = time.perf_counter()
    # cwd is the project root, never data/raw (untrusted data; the cwd goes on sys.path).
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", tool],
                            cwd=ROOT, env=env, check=False)
    status = {0: "done", TIMING_INVALID_EXIT: "INVALID TIMING (machine slept)"}.get(
        result.returncode, f"FAILED (exit code {result.returncode})")
    print(f"[{tool}] {status} after {time.perf_counter() - start:.0f} s", flush=True)
    return result.returncode


def run_tool_process(tool: str, quick: bool) -> bool:
    """Run one tool until it finishes with a valid timing; True on success."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        code = run_tool_once(tool, quick)
        if code != TIMING_INVALID_EXIT:
            return code == 0
        print(f"[{tool}] attempt {attempt} slept during the timed fit; running it again", flush=True)
    return False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        os.environ["EMBER_QUICK"] = "1"  # config reads it at import time, so set it before importing

    if args.worker:
        try:
            run_worker(args.worker)
        except Exception:
            traceback.print_exc(file=sys.stdout)  # into the run log even when only stdout is redirected
            sys.stdout.flush()
            raise
        return

    import config

    if args.quick and not config.QUICK:
        raise RuntimeError("--quick given but config was imported before EMBER_QUICK was set")
    if args.refresh_notes:
        from src import automl_checks
        for path in automl_checks.refresh_notes(args.tools):
            print(f"refreshed {path}")
    elif not args.tables_only:
        failed = [tool for tool in args.tools if not run_tool_process(tool, config.QUICK)]
        if failed:
            raise SystemExit(f"AutoML failed for: {', '.join(failed)}. Fix it and re-run with --tools "
                             f"{','.join(failed)}; the other tools' results are already logged.")

    from src import tables
    for path in tables.build_automl_outputs():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
