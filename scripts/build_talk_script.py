"""The presenter's script for the demo page: what to say in each section, and where to scroll or click.

Writes reports/talk_script.md. Every number in the script is read from the saved results (the same facts
the skill report uses, scripts/build_skill_report.py facts(), plus the scorecard tables), so the script
always matches the page. Run from the project root after the tables exist:
  python scripts/build_talk_script.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_skill_report import TABLES, behind_phrase, facts, pct  # noqa: E402

OUT = ROOT / "reports" / "talk_script.md"


def scorecard_facts() -> dict[str, Any]:
    """The extra numbers the scorecard part of the talk quotes (page section 6)."""
    S: dict[str, Any] = {}
    rvf = pd.read_csv(TABLES / "random_vs_future.csv").set_index("model")
    ts = pd.read_csv(TABLES / "tuning_summary.csv").set_index("method")
    prec = pd.read_csv(TABLES / "precision_both_sets.csv")
    at_half = prec[(prec.test_set == "test_future") & (prec.threshold_name == "0.5")].drop_duplicates("model")
    at_half = at_half.set_index("model")
    t = rvf.loc["optuna"]
    S["auc"], S["f1"] = f"{t.test_future_roc_auc:.4f}", f"{t.test_future_f1:.4f}"
    S["fpr"] = pct(t.test_future_fpr, 2)
    S["acc"] = pct(ts.loc["optuna", "test_future_accuracy"], 2)
    S["precision"], S["recall"] = pct(at_half.loc["optuna", "precision_at_test_mix"], 2), pct(at_half.loc["optuna", "tpr"], 2)
    S["cv_auc"], S["cv_std"] = f"{ts.loc['optuna', 'best_cv_auc']:.4f}", f"{ts.loc['optuna', 'cv_auc_std']:.4f}"
    S["cv_vs_future"] = f"{t.test_future_roc_auc - ts.loc['optuna', 'best_cv_auc']:+.4f}"
    S["cv_vs_random"] = f"{t.test_random_roc_auc - ts.loc['optuna', 'best_cv_auc']:+.4f}"
    S["logreg_auc_r"] = f"{rvf.loc['logreg', 'test_random_roc_auc']:.4f}"
    schema = json.loads((ROOT / "models" / "final" / "schema.json").read_text(encoding="utf-8"))
    th = pd.read_csv(TABLES / "threshold.csv")
    row = th[th.cost_ratio == schema["threshold"]["cost_ratio_fn_to_fp"]].iloc[0]
    S["tp"], S["fn"], S["fp"], S["tn"] = (f"{int(row[c]):,}" for c in
                                          ("test_future_tp", "test_future_fn", "test_future_fp", "test_future_tn"))
    S["fn_half"], S["fp_half"] = f"{int(row['test_future_fn_at_0.5']):,}", f"{int(row['test_future_fp_at_0.5']):,}"
    cal = pd.read_csv(TABLES / "calibration.csv")
    cal = cal[cal.test_set == "test_future"].set_index("model")
    others = cal.loc[["autogluon", "flaml", "h2o", "logreg"]]
    # a model ties tuned LightGBM when the CI of its Brier difference includes 0
    tied = others[(others.brier_diff_ci_low <= 0) & (others.brier_diff_ci_high >= 0)].index.tolist()
    better = others[others.brier_diff_ci_high < 0].index.tolist()
    S["brier_tied"], S["brier_better"] = tied, better
    return S


def tool_line(F: dict[str, Any], tool: str, name: str) -> str:
    """One AutoML tool's headline score and its verdict against tuned LightGBM (paired bootstrap)."""
    d = F["pw"][tool]["tpr_at_1pct_fpr"]
    score = pct(F["rvf"].loc[tool, "test_future_tpr_at_1pct_fpr"])
    if not d.significant_95:
        return f"{name} caught {score}: a tie with tuned LightGBM, because the difference is smaller than the uncertainty"
    return f"{name} caught {score}: {'ahead of' if d.diff_a_minus_b > 0 else 'behind'} tuned LightGBM, and the gap is real"


def script(F: dict[str, Any], S: dict[str, Any]) -> str:
    """The whole script as Markdown, from the report facts F and the scorecard facts S."""
    rvf = F["rvf"]
    logreg_tpr = pct(rvf.loc["logreg", "test_future_tpr_at_1pct_fpr"])
    names = {"autogluon": "AutoGluon", "flaml": "FLAML", "h2o": "H2O", "logreg": "logistic regression"}
    if S["brier_better"]:
        brier = f"{', '.join(names[m] for m in S['brier_better'])} is clearly better than tuned LightGBM"
    elif S["brier_tied"]:
        brier = (f"tuned LightGBM and {', '.join(names[m] for m in S['brier_tied'])} are tied for the best: the "
                 "confidence interval of the difference includes zero")
    else:
        brier = "tuned LightGBM is clearly the best"
    top = (f"It is a tie at the top: tuned LightGBM and AutoGluon. AutoGluon got there in {F['ag_min']} minutes "
           "with no human input." if F["ag_tied"] else "Tuned LightGBM came out on top; no AutoML tool matched it.")
    return f"""# Presentation script

This is a script for presenting the project from the demo page, with notes on where to scroll and what to
click. It follows the page from top to bottom in ten numbered sections. Lines marked **Say** are spoken;
lines marked **Do** are what you do on the screen. Every number below is read from `reports/tables/` by
`scripts/build_talk_script.py`, the same files the page reads, so the script and the page always agree (the
script sometimes gives one more decimal).

**Length.** About 7 minutes in full. For a 5-minute slot, follow the *Short version* notes: they say which
parts to skip.

## Before you start

**Do**
1. In a terminal at the project root, run `python present.py` (on Windows: `.venv\\Scripts\\python.exe present.py`).
   It starts the scoring API and the page, then opens the page in the browser. The first load takes about
   10 seconds.
2. Zoom the browser to 125–150% (Ctrl and +) so people at the back can read it.
3. Scroll back to the top. Keep the terminal open: Ctrl+C there stops everything at the end.

**How to move around the page**
- Scroll with the mouse wheel, or press Page Down. Each section starts with a blue numbered circle.
- To jump to a section during questions, press **Ctrl+F** and type its title, for example "full scorecard".
- Apart from the links to the dataset and the GitHub repository, nothing opens a new page. The only things you
  click are the example buttons, "Show another" and the Upload button in section 8, plus the two fold-out boxes: "What each metric means" (section 6) and "Technical
  detail (for questions)" (section 8).

---

## 1. The question (about 20 s)

**Do** Start at the top of the page, on the big title.

**Say** "Every day new Windows programs appear, and some of them are malware. My question is whether a
model can tell them apart just by reading the file, without running it. And second: does an AutoML tool,
which picks and tunes a model by itself, do better than a person tuning a model by hand? I also checked
whether the usual way of testing a model is too optimistic."

## 2. The data (about 40 s)

**Do** Scroll to section 2. Point at the three number cards, then the blue and purple timeline bar, then the
row of grey chips.

**Say** "The data is EMBER2024, a public research dataset on Hugging Face. I used the 64-bit Windows part.
There are {F['n_train']} programs to learn from and two test sets of {F['n_random']} each. About half are
malware, by design."

"The timeline shows the important part. The learning files and the *random* test come from the same
{F['train_weeks']} weeks. The *future* test comes from the {F['future_weeks']} weeks after that, which is
closer to real life, where tomorrow's malware is new."

"The model never sees the program itself, only {F['dim']} numbers measured from it, in {F['n_families']}
groups, such as its imports, its readable text and how random its content looks."

## 3. The contenders (about 30 s)

**Do** Scroll to section 3. Point at the blue card first, then the orange one.

**Say** "Team 1 is LightGBM tuned by hand. I tried {F['trials']} settings with each of three search methods:
grid, random and Optuna. For reference I also kept LightGBM with default settings, and logistic regression,
a simple straight-line model. Team 2 is three AutoML tools: FLAML, AutoGluon and H2O. Each got the same
{F['budget_min']}-minute budget to choose and tune a model by itself."

*Short version:* say only the two team names and the three AutoML tools.

## 4. How we compared them fairly (about 30 s)

**Do** Scroll to section 4. Point at the blue box.

**Say** "Everyone got the same learning files and the same random seed, {F['seed']}. The test files were only
used for the final scores, never to choose anything. Hand-tuning chose its settings by {F['folds']}-fold
cross-validation on the learning files."

"The main score is in the blue box: the share of malware caught when only 1 in 100 safe programs may be
flagged. A security team cannot handle many false alarms, so this matters more than plain accuracy. Every
score also has a 95% confidence interval from resampling the test files {F['n_boot']} times."

*Short version:* read only the blue box.

## 5. The results (about 40 s)

**Do** Scroll to section 5. Point at the dot chart. Hover over a dot to show its exact value if you want.
Then point at the bullet list under the chart.

**Say** "Each dot is a model and each line is its 95% confidence interval. Note that the axis does not start
at zero, so small differences look big."

"Tuned LightGBM caught {F['tuned_tpr_f']} of future malware. {tool_line(F, 'autogluon', 'AutoGluon')}.
FLAML and H2O were {behind_phrase(F)}. The three hand-tuning methods ended within {F['hand_spread_pts']}
points of each other, so searching at all mattered much more than how you searched: default settings caught
{F['default_tpr_f']}. The straight-line model is far behind at {logreg_tpr}."

## 6. The full scorecard: every evaluation metric (about 60 s, or when asked)

This section answers "show me the evaluation metrics". In a short talk, scroll past it with one sentence and
come back here for questions.

**Do** Scroll to section 6. Point at the first table (the future test), then the row for tuned LightGBM.

**Say** "This is every standard metric for every model. The first table is the future test, the second the
random test. For the tuned LightGBM on the future test: AUC {S['auc']}, accuracy {S['acc']}, precision
{S['precision']}, recall {S['recall']}, F1 {S['f1']}, and {S['fpr']} of safe files flagged. All except the
AUC use the usual 0.5 cut-off. The small grey numbers under a score are its 95% confidence interval. Precision
is this high only because our tests are half malware; section 9 shows how many alarms are false when
malware is rare."

**Do** Point at the logistic regression row in both tables.

**Say** "The straight-line model shows the drift problem: its AUC is {S['logreg_auc_r']} on the random test
but only {F['logreg_auc_f']} on the future test."

**Do** Scroll to "The deployed model, file by file". Point at the left grid, then the right one.

**Say** "This is the confusion matrix of the model we deploy, on the future test. At its flag line it catches
{S['tp']} malware files and misses {S['fn']}, at the cost of {S['fp']} false alarms out of the safe files.
At the plain 0.5 cut-off it would miss {S['fn_half']} and raise {S['fp_half']} false alarms. We moved the
line on purpose, because we count a miss as {F['cost_ratio']} times worse than a false alarm. The line was
chosen on the learning files, not on the test."

**Do** Scroll to "Before the test". Optionally click **What each metric means** to show the glossary.

**Say** "And these are the scores on the learning files alone, before any test. The tuned model's
cross-validation AUC was {S['cv_auc']}, plus or minus {S['cv_std']}. Its test AUCs differ from that by
{S['cv_vs_future']} on the future test and {S['cv_vs_random']} on the random test. (The CV ran on a
{F['tune_subset']}-file subset and was used to choose the settings, so it is a check, not a second exam.)"

*If asked about probabilities:* the line "How trustworthy the probabilities are" gives the Brier score
(lower is better); {brier}.

## 7. The winner, and why (about 40 s)

**Do** Scroll to section 7. Point at the big headline, then the three short parts under it.

**Say** "{top} The hand-tuned winner took {F['tuned_min']} minutes, and
all three hand searches together {F['all_manual_min']} minutes."

"Why these models do well: every AutoML tool ended up choosing boosted decision trees, the same kind of model
I tuned by hand. Trees combine thousands of simple yes/no questions and cope with numbers on very different
scales."

"Why I deploy LightGBM and not AutoGluon: its settings were chosen without looking at any test score, it is a
single file of {F['num_trees']} trees and {F['model_mb']} MB that needs no AutoML library, and it can explain
each decision, which you will see now."

## 8. Try it live (about 60–90 s)

**Do** Scroll to section 8. Point at the two big numbers first.

**Say** "This is the deployed model. At its flag line it catches {F['dep_recall']} of future malware and flags
{F['dep_fpr']} of safe programs for a second look."

**Do** **Caught malware** is already selected (click it if not). Point at the verdict card, then the black line on the bar, then the
"Why?" list.

**Say** "This file is malware, and the model flagged it. The bar is its malware score, and the black line is
the flag line. It is set low on purpose. Under 'Why?' are the groups of measurements that pushed the decision
most. Red means 'points to malware'."

**Do** Click **Upload a file**, then the **Upload** button (or drag the file onto the box), and choose `C:\\Windows\\System32\\notepad.exe`.

**Say** "I can also upload a real program. It is read in memory and never run or saved. Notepad comes out as
safe." (If it does not, say so: the model makes mistakes, and the next example shows one.)

**Do** If time allows, click **Missed malware**.

**Say** "And here is a malware file it missed. I want to show its mistakes too."

*Short version:* show only **Caught malware**.

*If asked for the raw output:* click **Technical detail (for questions)** under the result. It shows the exact
probability, the threshold, the file's fingerprint (sha256) and the top 10 measurements by SHAP.

## 9. What we learned (about 40 s)

**Do** Scroll to section 9. Walk through the four boxes from top to bottom.

**Say**
1. "The random test and the future test: for tree models the AUC barely drops ({F['tuned_auc_r']} to
   {F['tuned_auc_f']}), but the straight-line model drops by {F['logreg_gap']}."
2. "In the real world most alarms would be false. Our tests are half malware. If only {F['prev_rate']} of files
   were malware, only about {F['prec_low']} of the alarms would be real: about {F['false_per_true']} false
   alarms for every real catch. So this is a tool to sort files for an analyst, not to block them
   automatically."
3. "I planted a cheat: a column that almost gives away the answer. {F['leak_relied']} of {F['leak_models']}
   models made it {F['leak_rank_text']}, and {F['leak_verdict']}. Catching leaks is still a human's job."
4. "On {F['chal_n']} malware files that antivirus engines first missed, the model catches {F['chal_rate']},
   against {F['chal_vs']} of ordinary future malware. The hardest malware is still hard."

*Short version:* say only boxes 2 and 3.

## 10. Keep in mind (about 15 s)

**Do** Scroll to section 10, the end of the page.

**Say** "To sum up: it is a triage aid, not a final verdict. It reads the file without running it, so malware
built to look harmless can slip through. And it only knows 64-bit Windows programs. Everything, including the
code and every result, is in the GitHub repository linked here. Thank you, I'm happy to take questions."

---

## Where to go for common questions

| Question | Where on the page | What to show |
|---|---|---|
| "Show me the evaluation metrics." | Section 6 | Both tables; read the tuned LightGBM row |
| "What is the confusion matrix?" | Section 6, "The deployed model, file by file" | The left grid (deployed), the right grid (0.5 cut-off) |
| "Did you cross-validate?" | Section 6, "Before the test" | The {F['folds']}-fold CV AUCs, before any test |
| "What do precision, recall and F1 mean?" | Section 6 | Click **What each metric means** |
| "Is the difference between models real?" | Section 5 | The confidence-interval lines and the "tie" or "behind" wording |
| "Why LightGBM and not AutoGluon?" | Section 7 | "Why we deploy tuned LightGBM" |
| "How was the flag line chosen?" | Section 8 | The caption under the verdict card, then **Technical detail** |
| "Why is the model's decision what it is?" | Section 8 | The "Why?" list, then **Technical detail** (SHAP table) |
| "What is data leakage here?" | Section 9 | Box 3 |
| "Can I see the API?" | A new browser tab | http://127.0.0.1:8000/docs (only when started with `present.py`, not `--no-api`) |

More prepared answers are in `reports/qa_prep.md`. Every table is also in `reports/report.pdf`.

## If something goes wrong

- **The page does not open.** Open http://127.0.0.1:8501 yourself. If the terminal shows an error, run
  `python present.py --no-api`: the page then scores in its own process and the demo still works.
- **The upload fails.** Use the example buttons instead; they cover the same story.
- **No internet.** Not a problem: everything runs on your laptop.
"""


def main() -> None:
    OUT.write_text(script(facts(), scorecard_facts()), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
