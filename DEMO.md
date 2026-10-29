# Presentation script (about 5 minutes)

The talk is one web page, `app/streamlit_app.py` (sections in `app/story.py`), that reads top to bottom in
nine numbered sections. You scroll and talk; there are no slides to switch to, no tabs and no side panels.
Every number on the page is read from `reports/tables/` and `models/final/`, so the page always matches the
report. `reports/slides.pptx` is the backup deck (the brief asks for slides) and the handout.

Run every command from the project root. Interpreter: Windows `.venv/Scripts/python.exe`, Linux/macOS
`.venv/bin/python`; below, `PY` stands for it (PowerShell: `$PY = ".venv\Scripts\python.exe"` and `& $PY`;
Git Bash: `PY=.venv/Scripts/python.exe`).

## Before the talk (2 minutes)

```bash
$PY present.py
```
This starts the API (port 8000) and the page (port 8501) and opens http://127.0.0.1:8501. Wait until the page
has loaded once (about 10 s the first time). Zoom the browser so the section headings are easy to read from
the back of the room. Ctrl+C in that terminal stops both servers at the end.

The last line of the page says "scored by the FastAPI service at ...". If the API did not start, the page
scores in its own process and that line says so; the demo still works (`present.py --no-api` forces this).

## The talk, section by section

| # | Section | Time | Say / show |
|---|---|---|---|
| 1 | The question | 0:00, 20 s | Spot malware without running it; does AutoML beat hand-tuning; and is the usual test too optimistic? |
| 2 | The data | 0:20, 40 s | EMBER2024 from Hugging Face; the three numbers (learning files, random test, future test); point at the timeline: the future test is from later weeks. Read the "What the model sees" line: the model only sees measurements of each file, in the groups listed. |
| 3 | The contenders | 1:00, 30 s | Team 1: LightGBM tuned by hand with three search strategies, plus two reference points. Team 2: three AutoML tools with the same time budget each. |
| 4 | How we compared them fairly | 1:30, 30 s | Same files, same seed, test files never used to choose anything, two exams. Read the blue box: the headline score is "malware caught at 1 false alarm per 100 safe programs". |
| 5 | The results | 2:00, 40 s | The dot chart: top models within a fraction of a point. Read the bullets: AutoGluon ties tuned LightGBM, FLAML and H2O behind, the search strategy barely matters, the straight-line model far behind. |
| 6 | The winner, and why | 2:40, 40 s | "A tie at the top." AutoGluon's time vs the hand search. Why: every AutoML tool ended up with boosted trees too. Why we deploy LightGBM: chosen before seeing test scores, one small file, explainable. |
| 7 | Try it live | 3:20, 60 s | Click **Caught malware**: verdict, flag line (read the caption under the card: it is low on purpose), the red reasons. Then **Upload a file**: `C:\Windows\System32\notepad.exe` ("read in memory, never run"). If time allows, **Missed malware**: be honest about errors. |
| 8 | What we learned | 4:20, 30 s | The four boxes; stress 2 (most real-world alarms would be false) and 3 (no AutoML tool noticed the planted cheat). |
| 9 | Keep in mind | 4:50, 10 s | Triage aid, not a blocker. Point to the repository link. |

**Short on time?** Skim sections 3 and 4 in one sentence each, and in section 7 show only "Caught malware".

**Questions.** In section 7, "Technical detail (for questions)" shows the raw P(malicious), the threshold, the
file's sha256 and test_future index, and the top-10 SHAP columns. `reports/qa_prep.md` has likely questions
with answers, and `reports/report.pdf` has every table.

## Backup: the API by hand

If someone asks to see the API itself (it is what the page calls), open http://127.0.0.1:8000/docs, or:

```bash
curl -s http://127.0.0.1:8000/health
$PY -c "import json, numpy as np; d = np.load('models/final/demo_samples.npz'); json.dump({'features': d['X'][0].tolist()}, open('row.json', 'w')); print('offline score', d['p'][0])"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data "@row.json"
curl -s -X POST "http://127.0.0.1:8000/predict?top_k=3" -H "Content-Type: application/json" --data "@row.json"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"features": [1, 2, 3]}'
```
Expect `"status":"ok"` with the model version, sha256 and threshold; then the same `p_malicious` as the offline
score; then the 3 largest SHAP contributions; then a 422 ("List should have at least 2568 items"). In
PowerShell, call `curl.exe` (not the `curl` alias), keep `"@row.json"` in quotes and write the last body as
`-d '{\"features\": [1, 2, 3]}'`. Delete `row.json` afterwards.

Upload a file (parsed in memory, never stored or run):

```bash
curl.exe -s -X POST http://127.0.0.1:8000/predict_file -F "file=@C:/Windows/System32/notepad.exe"   # Windows
curl -s -X POST http://127.0.0.1:8000/predict_file -F "file=@/bin/ls"                              # Linux: 400, not a PE file
```

The two servers can also be started by hand, each in its own terminal:

```bash
$PY -m uvicorn app.api:app --host 127.0.0.1 --port 8000
$PY -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## Checks before presenting (optional)

```bash
$PY -m pytest tests/test_api.py      # API validation, upload safety, every page section, numbers vs tables
$PY run_all.py --steps deploy        # with the full dataset only: re-export models/final/ and re-check parity
```
`run_all.py --steps deploy` needs the feature cache (`data/cache`); on a clone without it, run only the
pytest line (the cache-only parity test skips). The page itself works without the cache: it then uses the
shipped `models/final/demo_samples.npz` (a few hundred test_future files).

## Regenerate the screenshots

With both servers running:

```bash
$PY app/take_screenshots.py      # writes reports/screenshots/{a_malicious_correct,b_benign_correct,c_upload_notepad}.png
```
Each screenshot is the live-demo section of the page. On Windows the script drives the installed Microsoft
Edge. On Linux run `$PY -m playwright install chromium` once and pass `--pe` a Win64 PE file you have the right
to share (notepad.exe does not exist there). Then rebuild the report and deck (`scripts/build_report.py`,
`scripts/build_slides.py`), which show these screenshots.

## Deploy it yourself (needs your own account)

The page can run as one process: when it cannot reach the API it scores in-process with the same functions
(`app.api.score_vector` / `score_file`), and without the feature cache it uses `models/final/demo_samples.npz`.
`app/requirements.txt` is a slim dependency list for that (no AutoML tools, no torch). Both hosts need `git`
during the build, because thrember and oscrypto are installed from pinned git commits. The presentation
sections also read `config.py`, `reports/tables/`, `data/splits/splits_info.json` and `.streamlit/config.toml`.

### Streamlit Community Cloud
1. Push this repository (with `models/final/`, about 6 MB) to your GitHub account.
2. On https://share.streamlit.io choose "Create app", pick the repository and branch, and set the main file
   path to `app/streamlit_app.py`. In "Advanced settings" choose Python 3.12.
3. Streamlit Cloud installs `app/requirements.txt` (the requirements file next to the entry point is used
   before the root one). The page starts in in-process mode by itself.

### Hugging Face Spaces (Streamlit SDK)
1. Create a Space at https://huggingface.co/new-space with SDK "Streamlit".
2. Copy into the Space repository, keeping the folder layout: `app/` (with `__init__.py`), `models/final/`,
   `config.py`, `reports/tables/`, `data/splits/splits_info.json`, `.streamlit/config.toml`, and
   `app/requirements.txt` as `requirements.txt` at the Space root.
3. Put this header at the top of the Space's `README.md`:
   ```yaml
   ---
   title: EMBER2024 malware detector
   sdk: streamlit
   sdk_version: 1.66.1
   python_version: "3.12"
   app_file: app/streamlit_app.py
   ---
   ```
4. Push; the Space builds and serves the page. To serve the FastAPI too, use a Docker Space instead whose
   command is `uvicorn app.api:app --host 0.0.0.0 --port 7860`.

Before making a deployment public, read MODEL_CARD.md "Ethical considerations": a public scoring endpoint with
SHAP output is an oracle for evasion. Keep it private or add authentication.
