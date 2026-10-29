"""Step 4 synthesis driver: the research question answered from the Step 4 tables.

Stages (run in this order, after scripts/run_explain.py, run_eval_bias.py and run_leakage.py):
  tables    reports/tables/precision_both_sets.csv, gap_vs_tuned.csv and automl_self_estimate.csv,
            from the saved test scores, the Step 4 tables and the AutoML tools' saved notes
            (a few minutes: one paired bootstrap over all eight models).
  writeup   reports/critical_eval.md and reports/limitations.md, from the tables only.

Usage (from the project root):
  .venv/bin/python scripts/run_critical_eval.py                    # both stages
  .venv/bin/python scripts/run_critical_eval.py --quick            # smoke run on the _quick outputs
  .venv/bin/python scripts/run_critical_eval.py --stages writeup   # re-render the markdown only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import when run as a plain script

ALL_STAGES: tuple[str, ...] = ("tables", "writeup")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default=",".join(ALL_STAGES), help=f"comma-separated subset of {','.join(ALL_STAGES)}")
    parser.add_argument("--quick", action="store_true", help="read and write the _quick files (sets EMBER_QUICK=1)")
    args = parser.parse_args(argv)
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(args.stages) - set(ALL_STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        os.environ["EMBER_QUICK"] = "1"  # config reads it at import time, so set it before importing
    from config import set_seed

    set_seed()
    if "tables" in args.stages:
        from src import critical_eval

        for path in critical_eval.build_tables().values():
            print(f"wrote {path.relative_to(ROOT)}", flush=True)
    if "writeup" in args.stages:
        from src import critical_eval_writeup

        for path in critical_eval_writeup.write_reports():
            print(f"wrote {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
