"""Save the Step 5 demo screenshots to reports/screenshots/ with a headless browser (Playwright).

Needs the API and the Streamlit app running (DEMO.md, steps 1-2). Uses the installed Microsoft
Edge on Windows (no browser download); elsewhere run "playwright install chromium" once.

  .venv/Scripts/python.exe app/take_screenshots.py [--url http://127.0.0.1:8501] [--pe C:/Windows/System32/notepad.exe]

Screens: (a) a malicious test_future file classified correctly, (b) a benign one classified
correctly, (c) an upload of a benign Windows binary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"


def settle(page: Page, text: str) -> None:
    """Wait until Streamlit has finished re-running and `text` is visible."""
    page.wait_for_load_state("networkidle")
    page.get_by_text(text).filter(visible=True).first.wait_for(timeout=60_000)
    page.wait_for_timeout(1500)   # let charts finish drawing


def choose(page: Page, option: str) -> None:
    """Click one of the example pills (a radio option) by its visible text."""
    page.get_by_text(option, exact=True).click()


def sample_shot(page: Page, option: str, path: Path) -> None:
    choose(page, option)
    settle(page, "Keep in mind")
    page.screenshot(path=str(path), full_page=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--pe", type=Path, default=Path(r"C:\Windows\System32\notepad.exe"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge") if sys.platform == "win32" else p.chromium.launch()
        # Streamlit scrolls inside its own container, so full_page cannot extend the shot:
        # the viewport is made tall enough for the whole result instead.
        page = browser.new_page(viewport={"width": 960, "height": 2000})
        page.goto(args.url)
        settle(page, "Keep in mind")
        sample_shot(page, "Caught malware", OUT / "a_malicious_correct.png")
        sample_shot(page, "Safe program", OUT / "b_benign_correct.png")
        choose(page, "Upload a file")
        page.locator("input[type=file]").set_input_files(str(args.pe))
        settle(page, "The true answer is not known")
        page.screenshot(path=str(OUT / "c_upload_notepad.png"), full_page=True)
        browser.close()
    print(f"[screenshots] wrote 3 PNGs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
