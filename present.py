"""Start the presentation: the FastAPI service and the Streamlit page, then open the browser.

Usage (from the project root, with the venv's Python):
    .venv/Scripts/python.exe present.py            # Windows
    .venv/bin/python present.py                    # Linux / macOS
    ... present.py --no-api                        # the page alone; it scores in its own process
    ... present.py --no-browser                    # do not open a browser tab

Needs only app/requirements.txt (no dataset download). Press Ctrl+C to stop both servers.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
API_PORT: int = 8000
PAGE_PORT: int = 8501
WAIT_S: float = 90.0   # the API loads the model and SHAP explainer before /health answers


def api_command(port: int = API_PORT) -> list[str]:
    return [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", str(port)]


def page_command(port: int = PAGE_PORT) -> list[str]:
    return [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address", "127.0.0.1",
            "--server.port", str(port), "--server.headless", "true"]


def wait_for(url: str, seconds: float = WAIT_S) -> bool:
    """True once `url` answers with HTTP 200, False after `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-api", action="store_true", help="run the page alone, scoring in-process")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    env = dict(os.environ, EMBER_API_URL=f"http://127.0.0.1:{API_PORT}")
    if args.no_api:
        env["EMBER_IN_PROCESS"] = "1"
    procs: list[subprocess.Popen] = []
    try:
        if not args.no_api:
            print(f"[present] starting the API on port {API_PORT} ...")
            procs.append(subprocess.Popen(api_command(), cwd=ROOT, env=env))
            if not wait_for(f"http://127.0.0.1:{API_PORT}/health"):
                sys.exit("[present] the API did not start; see its messages above")
        print(f"[present] starting the page on port {PAGE_PORT} ...")
        procs.append(subprocess.Popen(page_command(), cwd=ROOT, env=env))
        url = f"http://127.0.0.1:{PAGE_PORT}"
        if not wait_for(url):
            sys.exit("[present] the page did not start; see its messages above")
        print(f"[present] ready: {url}  (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        while all(p.poll() is None for p in procs):   # stop if either server dies
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)
        print("[present] stopped")


if __name__ == "__main__":
    main()
