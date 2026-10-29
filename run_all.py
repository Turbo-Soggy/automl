"""Run the whole pipeline, or a chosen subset of steps, from the project root.

Usage:
    .venv/Scripts/python.exe run_all.py [--quick] [--steps data,tune,automl,eval,deploy] [--redownload] [--keep-zipped]

On Linux the interpreter is .venv/bin/python. --keep-zipped leaves the Win64 zips packed
(saves 32 GB of disk); src.data then streams the weekly files straight out of the zips.

Each step runs its scripts as separate processes with the venv's Python. A crash
in one script then stops the pipeline with a clear message instead of leaving a
half-initialised interpreter, and each script stays runnable on its own.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
SCRIPTS: Path = ROOT / "scripts"
DOWNLOAD_DONE: Path = ROOT / "data" / "raw" / "DOWNLOAD_DONE"

# Steps always run in this order, whatever order --steps lists them in,
# because each one consumes the outputs of the one before.
ALL_STEPS: tuple[str, ...] = ("data", "tune", "automl", "eval", "deploy")
STEP_SCRIPTS: dict[str, list[str]] = {
    "data": ["download_data.py", "build_cache.py", "verify_data.py"],
    "tune": ["run_tuning.py"],
    "automl": ["run_automl.py"],  # FLAML, AutoGluon, H2O one after another, then the comparison
    # Step 4, in dependency order: explain writes the train out-of-fold scores and models/threshold.json
    # that eval_bias reuses; critical_eval reads the tables of all three.
    "eval": ["run_explain.py", "run_eval_bias.py", "run_leakage.py", "run_critical_eval.py"],
}
# Extra arguments per script. run_leakage.py takes about 3-4 h on a 4-core machine, so a second
# pipeline run reuses every run already logged with valid timing instead of repeating it
# (the eval_bias fits resume the same way by default). Delete its log records to force a rerun.
SCRIPT_ARGS: dict[str, list[str]] = {"run_leakage.py": ["--skip-done"]}
NOT_IMPLEMENTED: dict[str, str] = {
    "deploy": "Step 5 (FastAPI / Streamlit)",
}


def venv_python() -> Path:
    """The project venv's interpreter, so every step uses the pinned packages."""
    candidates = [ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"No venv found at {ROOT / '.venv'}. Create it first (see README, Environment).")


def parse_steps(text: str) -> list[str]:
    """Turn "tune,data" into ["data", "tune"]: validated and in pipeline order."""
    requested = {s.strip() for s in text.split(",") if s.strip()}
    unknown = requested - set(ALL_STEPS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step(s) {sorted(unknown)}; choose from {','.join(ALL_STEPS)}")
    return [s for s in ALL_STEPS if s in requested]


def child_env(quick: bool, run_id: str) -> dict[str, str]:
    """Environment for child scripts: quick flag, one shared run id, project root importable."""
    env = dict(os.environ)
    env["EMBER_QUICK"] = "1" if quick else "0"
    env["EMBER_RUN_ID"] = run_id  # logging_utils uses it, so all records of this run can be grouped
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"  # stream child output live instead of in blocks
    return env


def run_script(name: str, env: dict[str, str], args: list[str] | None = None) -> None:
    """Run scripts/<name> with the venv Python from the project root; stop the pipeline if it fails."""
    script = SCRIPTS / name
    if not script.exists():
        raise SystemExit(f"[run_all] {script.relative_to(ROOT)} does not exist yet; cannot continue.")
    print(f"[run_all] >>> {script.relative_to(ROOT)}", flush=True)
    t0 = time.time()
    # cwd is the project root, never data/raw: the downloaded data is untrusted and
    # Python puts the working directory on sys.path.
    result = subprocess.run([str(venv_python()), str(script), *(args or [])], cwd=ROOT, env=env, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"[run_all] {name} failed with exit code {result.returncode} after {elapsed:.0f} s")
    print(f"[run_all] <<< {name} finished in {elapsed:.0f} s", flush=True)


def scripts_for(step: str, redownload: bool) -> list[str]:
    """Scripts of one step; the 6.6 GB download is skipped once it has completed and been verified."""
    names = list(STEP_SCRIPTS.get(step, []))
    if step == "data" and DOWNLOAD_DONE.exists() and not redownload:
        print(f"[run_all] {DOWNLOAD_DONE.relative_to(ROOT)} exists, skipping download (use --redownload to force)")
        names.remove("download_data.py")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EMBER2024 manual-tuning vs AutoML pipeline.")
    parser.add_argument("--quick", action="store_true", help="smoke run on tiny samples (sets EMBER_QUICK=1)")
    parser.add_argument(
        "--steps", type=parse_steps, default=list(ALL_STEPS),
        help=f"comma-separated subset of {','.join(ALL_STEPS)} (default: all)",
    )
    parser.add_argument("--redownload", action="store_true", help="re-run the download even if DOWNLOAD_DONE exists")
    parser.add_argument("--keep-zipped", action="store_true",
                        help="download without extracting the Win64 zips (32 GB less disk; passed to download_data.py)")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    env = child_env(args.quick, run_id)
    print(f"[run_all] run_id={run_id} quick={args.quick} steps={','.join(args.steps)}", flush=True)

    t0 = time.time()
    for step in args.steps:
        if step in NOT_IMPLEMENTED:
            print(f"[run_all] step '{step}': not implemented yet ({NOT_IMPLEMENTED[step]}); skipping.", flush=True)
            continue
        print(f"[run_all] === step '{step}' ===", flush=True)
        for name in scripts_for(step, args.redownload):
            extra = list(SCRIPT_ARGS.get(name, []))
            if name == "download_data.py" and args.keep_zipped:
                extra.append("--keep-zipped")
            run_script(name, env, extra)
    print(f"[run_all] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
