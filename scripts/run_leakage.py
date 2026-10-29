"""Step 4 part A driver: leakage experiments (src/leakage.py), then table, figure and write-up.

Every run (one model on one feature variant) is its own child process (this script started
again with --worker model:variant), for the same reasons as scripts/run_automl.py: runs never
overlap, memory (AutoGluon models, the H2O JVM) is freed between runs, and a crash loses only
that run. Each child's output goes to reports/leakage_logs/<model>__<variant>[_quick].log; the
report scans those logs (and the tools' own logs) for warnings about the leak column.
A run whose timed fit included machine sleep is logged as timing_valid=False and re-run.

Usage (from the project root):
  .venv/Scripts/python.exe scripts/run_leakage.py --quick                 # smoke test (30 s budgets)
  .venv/Scripts/python.exe scripts/run_leakage.py                         # all of src.leakage.RUN_PLAN (~3 h)
  .venv/Scripts/python.exe scripts/run_leakage.py --skip-done             # resume: skip runs already logged
  .venv/Scripts/python.exe scripts/run_leakage.py --runs flaml:clean,h2o:leak_detection_ratio
  .venv/Scripts/python.exe scripts/run_leakage.py --tables-only           # rebuild outputs from the log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import when run as a plain script

TIMING_INVALID_EXIT = 3  # worker exit code: finished, but the machine slept during the timed fit
MAX_ATTEMPTS = 3         # per run, counting only sleep-invalidated attempts; a crash is never retried


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smoke-test sizes and budgets (sets EMBER_QUICK=1)")
    parser.add_argument("--runs", default="", help="comma-separated model:variant pairs (default: the whole plan)")
    parser.add_argument("--skip-done", action="store_true", help="skip runs that already have a valid log record")
    parser.add_argument("--tables-only", action="store_true", help="skip the runs; rebuild outputs from the log")
    parser.add_argument("--worker", help=argparse.SUPPRESS)  # internal: model:variant to run in this process
    return parser.parse_args(argv)


def parse_runs(text: str) -> list[tuple[str, str]]:
    """'flaml:clean,lgbm_tuned:ablate_time_proxies' -> validated (model, variant) pairs."""
    from src import leakage

    if not text:
        return list(leakage.RUN_PLAN)
    runs = []
    for item in (t.strip() for t in text.split(",") if t.strip()):
        model, _, variant = item.partition(":")
        if model not in leakage.MODEL_NAMES or variant not in leakage.VARIANTS:
            raise SystemExit(f"bad run {item!r}: models {leakage.MODEL_NAMES}, variants {leakage.VARIANTS}")
        if variant == leakage.ABLATE and model != leakage.LGBM:
            raise SystemExit("the time-proxy ablation is a LightGBM-only experiment")
        runs.append((model, variant))
    return runs


def run_worker(spec: str) -> None:
    """Fit, explain, score and log one model:variant in this process."""
    import config
    from src import leakage

    config.set_seed(config.SEED)
    model, variant = spec.split(":")
    print(f"[leakage] {model} on {variant}: budget {config.LEAKAGE_BUDGET_S} s, quick={config.QUICK}", flush=True)
    record = leakage.run_lgbm(variant) if model == leakage.LGBM else leakage.run_tool(model, variant)
    print(f"[leakage] {model} {variant} best={record['best_model']} wall_s={record['wall_s']:.0f} "
          f"test_random_auc={record['test_random']['roc_auc']:.5f} "
          f"test_future_auc={record['test_future']['roc_auc']:.5f} timing_valid={record['timing_valid']}", flush=True)
    if not record["timing_valid"]:
        raise SystemExit(TIMING_INVALID_EXIT)


def run_once(model: str, variant: str, quick: bool) -> int:
    """One child process; its stdout and stderr go to the run's log file. Returns the exit code."""
    from src.leakage_report import worker_log_path

    env = dict(os.environ, EMBER_QUICK="1" if quick else os.environ.get("EMBER_QUICK", "0"), PYTHONUNBUFFERED="1")
    log_path = worker_log_path(model, variant)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{model}:{variant}] start {time.strftime('%Y-%m-%d %H:%M:%S')} -> {log_path.relative_to(ROOT)}", flush=True)
    start = time.perf_counter()
    # cwd is the project root, never data/raw (untrusted data; the cwd goes on sys.path).
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", f"{model}:{variant}"],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    status = {0: "done", TIMING_INVALID_EXIT: "INVALID TIMING (machine slept)"}.get(
        result.returncode, f"FAILED (exit code {result.returncode})")
    print(f"[{model}:{variant}] {status} after {time.perf_counter() - start:.0f} s", flush=True)
    return result.returncode


def run_until_valid(model: str, variant: str, quick: bool) -> bool:
    """Re-run a sleep-invalidated run up to MAX_ATTEMPTS times; True on success."""
    for _attempt in range(MAX_ATTEMPTS):
        code = run_once(model, variant, quick)
        if code != TIMING_INVALID_EXIT:
            return code == 0
    return False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        os.environ["EMBER_QUICK"] = "1"  # config reads it at import time, so set it before importing
    if args.worker:
        try:
            run_worker(args.worker)
        except Exception:
            traceback.print_exc(file=sys.stdout)  # into the run's log file
            sys.stdout.flush()
            raise
        return

    import config
    from src import leakage_report

    if args.quick and not config.QUICK:
        raise RuntimeError("--quick given but config was imported before EMBER_QUICK was set")
    if not args.tables_only:
        runs = parse_runs(args.runs)
        if args.skip_done:
            done = set(leakage_report.latest_records())  # (model, variant) pairs with a valid record
            runs = [r for r in runs if r not in done]
        failed = [f"{m}:{v}" for m, v in runs if not run_until_valid(m, v, config.QUICK)]
        if failed:
            print(f"FAILED runs: {','.join(failed)} (re-run them with --runs); building outputs from the rest",
                  flush=True)
    for path in leakage_report.build_outputs():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
