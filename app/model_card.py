"""Step 5: write MODEL_CARD.md from models/final/schema.json and the Step 2-4 tables.

Every number in the card is read from a file under reports/ or models/, never typed, so the
card is regenerated whenever scripts/export_model.py runs (run_all.py --steps deploy).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TUNED = "lgbm_tuned"   # name of the tuned LightGBM in the Step 3 comparison tables


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def ci(row: pd.Series, col: str, fmt=pct) -> str:
    """"value [low, high]" for a column with _ci_low / _ci_high companions."""
    return f"{fmt(row[col])} [{fmt(row[col + '_ci_low'])}, {fmt(row[col + '_ci_high'])}]"


def f4(x: float) -> str:
    return f"{x:.4f}"


def size_text(path: Path) -> str | None:
    """Size of a file or folder on disk ("4.7 MB", "1.8 GB"), or None when it is absent
    (the AutoGluon folder is git-ignored, so a fresh clone does not have it)."""
    if not path.exists():
        return None
    n = path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def md_table(rows: list[list[Any]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


class Tables:
    """The generated CSVs the card reads (with the _quick suffix in quick mode)."""

    def __init__(self, tables_dir: Path, suffix: str) -> None:
        def read(stem: str) -> pd.DataFrame:
            return pd.read_csv(tables_dir / f"{stem}{suffix}.csv")
        self.suffix = suffix
        self.comparison = read("automl_vs_manual")
        self.pairwise = read("automl_pairwise")
        self.gap = read("random_vs_future")
        self.precision = read("precision_both_sets")
        self.challenge = read("challenge")
        self.calibration = read("calibration")
        self.threshold = read("threshold")
        self.slices = read("bias_slices")
        self.time_ranks = read("time_proxy_ranks")


def choice_section(t: Tables, method: str, model_size: str, autogluon_size: str | None) -> str:
    comp = t.comparison.set_index("method")
    rows = [[m, ci(r, "test_future_tpr_at_1pct_fpr"), ci(r, "test_future_roc_auc", f4), r["interpretability"]]
            for m, r in comp.iterrows()]
    best_other = comp.drop(index=TUNED)["test_future_tpr_at_1pct_fpr"].idxmax()
    pair = t.pairwise[(t.pairwise.test_set == "test_future") & (t.pairwise.metric == "tpr_at_1pct_fpr")
                      & (t.pairwise[["method_a", "method_b"]].isin([TUNED, best_other]).all(axis=1))]
    pair_text = ""
    if not pair.empty:
        p = pair.iloc[0]
        verdict = "a significant difference" if bool(p["significant_95"]) else "no significant difference"
        pair_text = (f" The paired bootstrap ({p['method_a']} minus {p['method_b']}, `automl_pairwise{t.suffix}.csv`) "
                     f"gives {pct(p['diff_a_minus_b'], 2)} [{pct(p['ci_low'], 2)}, {pct(p['ci_high'], 2)}], "
                     f"i.e. {verdict} at 95%.")
    tuned = comp.loc[TUNED]
    ag = f"no {autogluon_size} predictor folder" if autogluon_size else "no large predictor folder"
    return f"""## Why this model

Candidates on test_future (weeks 52-63), from `reports/tables/automl_vs_manual{t.suffix}.csv` (95% bootstrap CIs):

{md_table(rows, ["model", "TPR at 1% FPR", "ROC-AUC", "interpretability"])}

The deployed model is the Step 2 tuned LightGBM (`{method}`, the tuner Step 2 selected by CV AUC on train).
Its TPR at 1% FPR on test_future is {ci(tuned, "test_future_tpr_at_1pct_fpr")}; the strongest other
candidate, {best_other}, reaches {ci(comp.loc[best_other], "test_future_tpr_at_1pct_fpr")}.{pair_text}
With detection this close, the tie is broken by what matters in deployment:

* **Interpretability:** one LightGBM booster, so TreeSHAP gives exact per-file explanations (the API returns
  them) and the Step 4 SHAP analysis (`reports/explainability.md`) describes exactly this model.
* **Inference simplicity:** one {model_size} text file and the `lightgbm` package. No Java runtime (H2O),
  {ag} (AutoGluon), no preprocessing pipeline, no scaler.
* **A threshold fitted for it:** `models/threshold.json` was chosen on this model's train out-of-fold scores.

The model was fixed in Step 2 and is not retrained here; test_future only confirms that no candidate is
clearly better. Nothing about the model or its threshold was tuned on a test set.
"""


def metrics_section(t: Tables, method: str) -> str:
    r = t.comparison.set_index("method").loc[TUNED]
    rows = [[s, ci(r, f"{s}_roc_auc", f4), ci(r, f"{s}_tpr_at_1pct_fpr"), f4(r[f"{s}_f1"]), pct(r[f"{s}_fpr"], 2)]
            for s in ("test_random", "test_future")]
    g = t.gap[t.gap.model == method].iloc[0]
    return f"""## Metrics

From `reports/tables/automl_vs_manual{t.suffix}.csv` (row `{TUNED}`) and `random_vs_future{t.suffix}.csv`.
F1 and FPR are at the plain 0.5 cut; TPR at 1% FPR is threshold-free (the cut that gives 1% FPR on that set).

{md_table(rows, ["test set", "ROC-AUC [95% CI]", "TPR at 1% FPR [95% CI]", "F1 at 0.5", "FPR at 0.5"])}

Random-minus-future gap: ROC-AUC {g['roc_auc_gap']:.4f} [{g['roc_auc_gap_ci_low']:.4f}, {g['roc_auc_gap_ci_high']:.4f}],
TPR at 1% FPR {pct(g['tpr_at_1pct_fpr_gap'], 2)} [{pct(g['tpr_at_1pct_fpr_gap_ci_low'], 2)}, {pct(g['tpr_at_1pct_fpr_gap_ci_high'], 2)}].
Performance on files from after the training period is slightly but measurably lower.

{challenge_block(t, method)}

{precision_block(t, method)}
"""


def challenge_block(t: Tables, method: str) -> str:
    c = t.challenge[t.challenge.model == method]
    rows = [[r.threshold_name, f"{r.threshold:.4g}", r.subset, r.n, f"{pct(r.detection_rate)} [{pct(r.ci_low)}, {pct(r.ci_high)}]",
             pct(r.test_future_tpr_same_threshold)] for r in c.itertuples()]
    return f"""### Challenge set (malware the AV engines first missed)

From `reports/tables/challenge{t.suffix}.csv`: Win64 files of the EMBER2024 challenge set, all malicious, so
only the detection rate can be measured. The last column is the same model's test_future TPR at that threshold.

{md_table(rows, ["threshold", "value", "subset", "n", "detected [95% CI]", "test_future TPR"])}

Detection drops sharply on these evasive files compared with test_future (last column), at every threshold:
this is the clearest measured limit of the model."""


def precision_block(t: Tables, method: str) -> str:
    p = t.precision[(t.precision.model == method) & t.precision.threshold_name.isin(["deployment", "train_oof_1pct_fpr"])]
    rows = [[r.test_set, r.threshold_name, f"{r.threshold:.4g}", pct(r.prevalence, 0),
             f"{pct(r.precision)} [{pct(r.precision_ci_low)}, {pct(r.precision_ci_high)}]",
             f"{r.false_alarms_per_10k:.0f}", f"{r.missed_per_10k:.1f}"] for r in p.itertuples()]
    return f"""### Precision at realistic prevalence

The test sets are 50/50 by construction; real traffic is mostly benign. `reports/tables/precision_both_sets{t.suffix}.csv`
re-weights each set's TPR and FPR to 1%, 5% and 10% malware (scenarios, not measurements):

{md_table(rows, ["test set", "threshold", "value", "prevalence", "precision [95% CI]", "false alarms / 10k files", "missed / 10k files"])}"""


def threshold_section(t: Tables, schema: dict[str, Any]) -> str:
    thr = schema["threshold"]
    rows = [[f"{r.cost_ratio:g}", f"{r.threshold:.4g}", pct(r.oof_recall, 2), pct(r.oof_fpr, 2),
             pct(r.test_future_recall, 2), pct(r.test_future_fpr, 2), pct(r.test_future_precision, 2)]
            for r in t.threshold.itertuples()]
    dep = t.threshold[t.threshold.cost_ratio == thr["cost_ratio_fn_to_fp"]].iloc[0]   # the deployed ratio
    fpr1 = t.precision[(t.precision.threshold_name == "train_oof_1pct_fpr")
                       & (t.precision.model == schema["model_version"].split("-")[1])]["threshold"].iloc[0]
    return f"""## Threshold and cost ratio

The API labels a file malicious when P(malicious) >= **{thr['threshold']:.6g}** (`{schema['threshold_source']}`).
How it was chosen: {thr['how_chosen']} The cost ratio is {thr['cost_ratio_fn_to_fp']:g}: a missed malware file
is assumed to cost {thr['cost_ratio_fn_to_fp']:g} times a false alarm. From `reports/tables/threshold{t.suffix}.csv`
(cost ratio, the threshold chosen on train OOF scores, then the check on test_future):

{md_table(rows, ["cost ratio FN:FP", "threshold", "OOF recall", "OOF FPR", "test_future recall", "test_future FPR", "test_future precision"])}

At the deployed ratio the threshold is very low, so the model flags {pct(dep.test_future_fpr, 1)} of benign test_future
files while missing {pct(1 - dep.test_future_recall, 2)} of malware. This suits triage (an analyst reviews every flag),
not blocking. For a 1%-FPR operating point use the train-OOF 1%-FPR threshold {fpr1:.4g} (rows `train_oof_1pct_fpr`
in the precision table above). {thr['caveat'].capitalize()}.

The test_future FPR at the deployed ratio ({pct(dep.test_future_fpr, 2)}) is above its OOF FPR ({pct(dep.oof_fpr, 2)}):
the threshold was fitted on out-of-fold scores of fold models trained on two thirds of train, while the deployed model
is trained on all of it, and test_future is also later in time (drift). Both are stated in `reports/limitations.md`,
not corrected.
"""


def calibration_section(t: Tables) -> str:
    c = t.calibration[(t.calibration.test_set == "test_future") & t.calibration.model.str.startswith(TUNED)]
    def diff(r) -> str:   # paired bootstrap difference in Brier vs the raw model ("-" for the raw model itself)
        if pd.isna(r.brier_diff_ci_low):
            return "-"
        return f"{r.brier_diff_vs_lgbm_tuned:+.4f} [{r.brier_diff_ci_low:+.4f}, {r.brier_diff_ci_high:+.4f}]"
    rows = [[r.model, f"{r.brier:.4f} [{r.brier_ci_low:.4f}, {r.brier_ci_high:.4f}]", diff(r), f"{r.ece:.4f}",
             f"{r.log_loss:.4f}"] for r in c.itertuples()]
    return f"""## Calibration

From `reports/tables/calibration{t.suffix}.csv` (test_future; calibrators fitted on train out-of-fold scores):

{md_table(rows, ["model", "Brier [95% CI]", "Brier minus raw [95% CI]", "ECE", "log loss"])}

The served score is the raw LightGBM probability. Platt or isotonic scaling improves calibration a little
(lower Brier, ECE and log loss above). Platt scaling is monotone, so it cannot change the ranking of files or
ROC-AUC and TPR at 1% FPR (isotonic scaling only adds ties). It is not shipped, to keep inference to one artefact whose threshold was fitted on the raw
score; a calibrated variant would need its own threshold. Either way, the score
is calibrated for a 50/50 class mix only: at another prevalence, read P(malicious) as a ranking score and use
the precision table above, not as the probability that this file is malware.
"""


def limits_section(t: Tables, method: str) -> str:
    s = t.slices[(t.slices.test_set == "test_future") & (t.slices.threshold_name == "deployment")].set_index(["slice_family", "slice"])
    signed, unsigned = s.loc[("signed", "signed")], s.loc[("signed", "unsigned")]
    hi, lo = s.loc[("entropy_proxy", "entropy > 7")], s.loc[("entropy_proxy", "entropy <= 7")]
    ranks = t.time_ranks.set_index("label")
    rank_text = ", ".join(f"`{lbl}` rank {int(r['rank_lgbm_tuned'])}" for lbl, r in ranks.iterrows())
    return f"""## Known limits

Measured at the deployed threshold on test_future (`reports/tables/bias_slices{t.suffix}.csv`), with the full list in
`reports/limitations.md` and `reports/critical_eval.md`:

* **Signed malware is missed more often:** TPR {pct(signed.tpr)} for signed malware vs {pct(unsigned.tpr)} unsigned. Benign
  FPR is {pct(signed.fpr)} for signed and {pct(unsigned.fpr)} for unsigned files: unsigned benign software is flagged a lot.
* **High-entropy (packed-looking) files:** benign FPR {pct(hi.fpr)} for whole-file entropy > 7 vs {pct(lo.fpr)} otherwise.
* **Time proxies inside the features:** the model leans on time-like columns (mean-|SHAP| ranks in
  `time_proxy_ranks{t.suffix}.csv`: {rank_text}). They are part of the static file, not leaked metadata, but they
  tie the model to its training period. The leaky metadata (`detection_ratio`, `first_submission_date`) is not an input.
* **Benign system files can score near the threshold:** in the demo, Windows' notepad.exe is labelled benign but
  with a score just under the low threshold (one file observed on the demo laptop's Windows build, see
  `reports/screenshots/c_upload_notepad.png`; another Windows version ships a different notepad.exe).
* **Static features only; evasion untested:** appended bytes, added imports or sections, or a valid signature change
  the features without changing behaviour. No adversarial robustness is claimed.
* **Win64 PE only, data up to 2024-12:** other file types, 32-bit PE and newer malware are out of distribution. The API
  warns when a file's machine type is not AMD64, and rejects files that do not parse as PE.
* **Labels come from antivirus verdicts,** and files with an ambiguous number of detections are not in the data
  (`reports/limitations.md`), so the task is easier than real triage.
"""


def write_model_card(schema: dict[str, Any], tables_dir: Path, out_path: Path, suffix: str = "") -> Path:
    t = Tables(tables_dir, suffix)
    method = schema["model_version"].split("-")[1]
    tr = schema["training_data"]["splits"]
    params = schema["params"]
    keys = ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l2")
    groups = ", ".join(f"{g['name']} ({g['end'] - g['start']})" for g in schema["feature_groups"])
    text = f"""# Model card: EMBER2024 Win64 malware detector

Generated by `scripts/export_model.py` (`app/model_card.py`) from `models/final{suffix}/schema.json` and the tables in
`reports/tables/`. Every measured number (metrics, counts, file sizes) is read from a generated file or measured
from the files, not typed by hand; re-run `run_all.py --steps deploy` to refresh it.

## Model details

* **Version:** `{schema['model_version']}`; file `models/final{suffix}/{schema['model_file']}`, sha256 `{schema['model_sha256']}`.
* **Type:** {schema['model_type']}, {schema['num_trees']} trees, LightGBM {schema['lightgbm_version']}, seed {schema['seed']}.
  Hyperparameters (found by the `{method}` search in Step 2): {", ".join(f"{k}={params[k]:.4g}" if isinstance(params[k], float) else f"{k}={params[k]}" for k in keys if k in params)}.
* **Input:** the {schema['feature_dim']:,}-value thrember feature vector of a PE file, in {len(schema['feature_groups'])} groups: {groups}.
  thrember commit `{schema['thrember']['commit']}`; `{schema['signify']['requirement']}` (thrember needs this older signify).
* **Preprocessing:** {schema['preprocessing']}. Inference is self-contained: model.txt plus `lightgbm`
  (and thrember to vectorise a raw file).
* **Output:** P(malicious) in [0, 1], a label at the threshold below, and on request the file's top TreeSHAP contributions.
* **Serving:** `app/api.py` (FastAPI: `/health`, `/predict`, `/predict_batch`, `/predict_file`) and `app/streamlit_app.py`; see `DEMO.md`.
* **Export check:** the exported file reproduces the saved Step 2 test_future scores on all
  {schema['parity_check']['rows']:,} rows, max abs difference {schema['parity_check']['max_abs_diff']:.1e}; `tests/test_api.py`
  checks the API against offline `booster.predict` to 1e-6.
* **Author:** Isaac Benjamin David (URK23AI1035), IA-3 Part B course project.

{choice_section(t, method, size_text(ROOT / "models" / f"final{suffix}" / schema["model_file"]),
                 size_text(ROOT / "models" / f"autogluon{suffix}"))}
## Intended use

* Triage aid for static analysis of Win64 PE files: rank and flag files for a human analyst to review.
* Research and teaching: comparing manual tuning with AutoML, and studying temporal drift on EMBER2024.

## Out-of-scope uses

* **The sole verdict for blocking, quarantining or deleting files.** The threshold trades many false alarms for few misses.
* **Non-Win64 inputs:** 32-bit PE, .NET-only analysis, scripts, documents, ELF or Mach-O files.
* **Adversarial settings** where an attacker can probe the model or knows its features; robustness was never tested.
* Judging software, vendors or users: a flag is a statistical score, not evidence of intent.

## Training data

* {schema['training_data']['dataset']}: {schema['training_data']['description']}
* train: {tr['train']['n']:,} files, {tr['train']['n_weeks']} weeks, {tr['train']['date_min'][:10]} to {tr['train']['date_max'][:10]},
  {pct(tr['train']['pos_rate'], 2)} malicious.
* test_random: {tr['test_random']['n']:,} files from the same weeks; test_future: {tr['test_future']['n']:,} files,
  {tr['test_future']['date_min'][:10]} to {tr['test_future']['date_max'][:10]}. Source: `data/splits/splits_info{suffix}.json`.
* Labels are EMBER2024's AV-derived labels (label -1, unlabelled, dropped). Raw weekly files list every sample twice;
  the data was deduplicated by sha256 before any split.

{metrics_section(t, method)}
{threshold_section(t, schema)}
{calibration_section(t)}
{limits_section(t, method)}
## Ethical considerations

* **False positives have a cost:** flagging legitimate (often small-vendor, unsigned) software can harm its authors and
  users; the slices above show unsigned benign files are flagged far more often. Keep a human in the loop.
* **False negatives give false comfort:** a "benign" verdict is not a clean bill of health, especially for signed or
  novel malware (challenge set).
* **Dual use:** the SHAP output shows which static features push a score down, which could guide evasion. The API is
  meant for local or authenticated use, not as a public oracle. Its 32 MB JSON-body cap is checked against the
  declared Content-Length only, so a chunked request without one is parsed in full: another reason to keep it private.
* **Uploads:** the API parses uploads in memory and never executes or stores them; the demo uploads only benign
  system binaries. Do not upload files you are not allowed to share.
* **Data:** EMBER2024 contains features and metadata of real malware and benign software, not the binaries; no personal
  data is used.

## How to reproduce

```bash
.venv/Scripts/python.exe run_all.py --steps deploy      # Linux: .venv/bin/python; export + API tests
.venv/Scripts/python.exe -m uvicorn app.api:app --port 8000
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

The model itself comes from Step 2 (`run_all.py --steps tune`) and the threshold from Step 4 (`--steps eval`);
the whole pipeline is `run_all.py` (seed {schema['seed']}). `DEMO.md` has the full demo script.
"""
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return out_path
