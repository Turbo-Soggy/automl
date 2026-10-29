# Demo script (about 5 minutes)

Step 5 of the project: the tuned LightGBM served by a FastAPI service (`app/api.py`) and a Streamlit page
(`app/streamlit_app.py`). The numbers shown live come from the model and from `MODEL_CARD.md`, which is
generated from `reports/tables/`. Run every command from the project root.

Interpreter: Windows `.venv/Scripts/python.exe`, Linux/macOS `.venv/bin/python`. Below, `PY` stands for it:

```bash
# Linux / macOS (bash)
PY=.venv/bin/python
```
```powershell
# Windows (PowerShell)
$PY = ".venv\Scripts\python.exe"
```
In PowerShell, write `& $PY` wherever the commands below say `$PY`. In Git Bash on Windows use
`PY=.venv/Scripts/python.exe` and the Linux commands.

## Before the demo (once, about 1 minute)

```bash
$PY run_all.py --steps deploy        # export models/final/ + MODEL_CARD.md, then run tests/test_api.py
```
Expect `[export] ... parity max diff=0.0e+00` and all tests passing (the notepad.exe tests skip off Windows).
`models/final/` is committed, so this step only re-checks it. The demo works without the 5 GB feature cache:
the Streamlit page then uses the shipped `models/final/demo_samples.npz` (a few hundred test_future files).

## 1. Start the API (terminal 1) - 0:00

```bash
$PY -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```
Leave it running. Interactive docs: http://127.0.0.1:8000/docs

## 2. Show the API with curl (terminal 2) - 0:30

Say: "The API checks its model file against the sha256 in schema.json before it serves anything."

```bash
curl -s http://127.0.0.1:8000/health
```
Expect `"status":"ok"`, a `model_version` of the form `lgbm-optuna-<first 12 hex digits of the sha256>`, the
sha256 and the threshold.

Score a real test_future file (written from the shipped demo samples), then show input validation:

```bash
$PY -c "import json, numpy as np; d = np.load('models/final/demo_samples.npz'); json.dump({'features': d['X'][0].tolist()}, open('row.json', 'w')); print('offline score', d['p'][0])"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data @row.json
curl -s -X POST "http://127.0.0.1:8000/predict?top_k=3" -H "Content-Type: application/json" --data @row.json
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"features": [1, 2, 3]}'
```
Expect the same `p_malicious` as the offline score, then the 3 largest SHAP contributions, then a 422
("List should have at least 2568 items"). In PowerShell, call `curl.exe` (not the `curl` alias) and write
the last body as `-d '{\"features\": [1, 2, 3]}'`.

Upload a file: the API parses it in memory and never stores or runs it.

```bash
# Windows: a benign Win64 system binary
curl.exe -s -X POST http://127.0.0.1:8000/predict_file -F "file=@C:/Windows/System32/notepad.exe"
# Linux: an ELF binary is not a PE file, so it is rejected with 400
curl -s -X POST http://127.0.0.1:8000/predict_file -F "file=@/bin/ls"
```
Windows: expect `"verdict":"benign"` with `p_malicious` just under the threshold (say: "the threshold is low on
purpose, a missed malware costs 10 false alarms"). Linux: expect `400 not a valid PE file`.
Delete `row.json` afterwards.

## 3. Start the Streamlit page (terminal 2) - 1:30

```bash
$PY -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```
Open http://127.0.0.1:8501. The sidebar shows "Connected: lgbm-optuna-..." (the page calls the API at the URL
in the sidebar; set `EMBER_API_URL` to change the default).

## 4. Walk through the page - 2:00 to 4:30

1. **Banner (20 s).** Model, test_future ROC-AUC and TPR at 1% FPR with CI, the threshold and what it means,
   then the limits box: triage aid only, Win64 only, static features, 50/50 training prevalence, challenge set.
2. **Correct malicious file (40 s).** Tab "test_future sample", True label = malicious, Outcome = correct.
   Show P(malicious), the threshold, Predicted vs True label, then the top-10 SHAP table grouped by family and the
   per-family SHAP sums ("positive pushes towards malicious; base value + all SHAP = logit").
   Screenshot: `reports/screenshots/a_malicious_correct.png`.
3. **Benign file (30 s).** True label = benign, Outcome = correct. Point at the negative SHAP values.
   Screenshot: `reports/screenshots/b_benign_correct.png`.
4. **A mistake (30 s).** Outcome = misclassified, True label = malicious: a missed malware file. Which families
   pulled it down? (Often the authenticode signature: signed malware is missed more often, see MODEL_CARD.md.)
5. **Upload (40 s).** Tab "Upload a PE file", upload `C:\Windows\System32\notepad.exe` (Linux: any Win64 .exe you
   have the right to share). Show the sha256, the verdict and the SHAP view.
   Screenshot: `reports/screenshots/c_upload_notepad.png`.

## 5. Close - 4:30

Say: "The model was fixed in Step 2 and the threshold in Step 4, both on train data only; MODEL_CARD.md lists
the metrics on both test sets, the challenge set and the precision at 1-10% prevalence, every number generated
from reports/tables."

Stop both servers with Ctrl+C in their terminals.

## Regenerate the screenshots

With both servers running:

```bash
$PY app/take_screenshots.py      # writes reports/screenshots/{a_malicious_correct,b_benign_correct,c_upload_notepad}.png
```
On Windows it drives the installed Microsoft Edge. On Linux run `$PY -m playwright install chromium` once and
pass `--pe` a Win64 PE file you have the right to share (notepad.exe does not exist there).

## Deploy it yourself (needs your own account)

The page can run as one process: when it cannot reach the API it scores in-process with the same functions
(`app.api.score_vector` / `score_file`), and without the feature cache it uses `models/final/demo_samples.npz`.
`app/requirements.txt` is a slim dependency list for that (no AutoML tools, no torch). Both hosts need `git`
during the build, because thrember and oscrypto are installed from pinned git commits.

### Streamlit Community Cloud
1. Push this repository (with `models/final/`, about 6 MB) to your GitHub account.
2. On https://share.streamlit.io choose "Create app", pick the repository and branch, and set the main file
   path to `app/streamlit_app.py`. In "Advanced settings" choose Python 3.12.
3. Streamlit Cloud installs `app/requirements.txt` (the requirements file next to the entry point is used
   before the root one). The page starts in in-process mode by itself.

### Hugging Face Spaces (Streamlit SDK)
1. Create a Space at https://huggingface.co/new-space with SDK "Streamlit".
2. Copy into the Space repository: `app/` (with `__init__.py`), `models/final/`, and `app/requirements.txt`
   as `requirements.txt` at the Space root.
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
