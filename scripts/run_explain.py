"""Step 4 part C driver: SHAP, cross-tool agreement, calibration and the cost threshold.

Stages (run in this order; each reads only what earlier stages or earlier steps saved):
  oof         3-fold out-of-fold scores on TRAIN for the tuned LightGBM configuration (the
              only scores used to fit calibrators and choose the threshold). ~10-20 min on a
              4-core VM with 2 threads; run it detached.
  shap        TreeSHAP of the tuned LightGBM (and FLAML's tree winner) on a 5k stratified
              sample of test_future; beeswarm, family bar chart and three waterfall plots.
  agreement   family-level and top-k agreement of LightGBM SHAP, FLAML SHAP, AutoGluon
              permutation importance and H2O varimp.
  calibration reliability curves, Brier / log loss / ECE on test_future from saved predictions;
              isotonic and Platt calibrators fitted on the TRAIN OOF scores.
  threshold   cost-minimising threshold on TRAIN OOF scores (C_FN/C_FP = 1, 5, 10, 50),
              its test_future performance, and models/threshold.json for deployment.
  report      reports/explainability.md, generated from the tables above.

Usage (from the project root):
  .venv/bin/python scripts/run_explain.py --quick                 # smoke run, *_quick outputs
  .venv/bin/python scripts/run_explain.py --stages oof            # the long one, run detached
  .venv/bin/python scripts/run_explain.py --stages shap,agreement,calibration,threshold,report
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `config` and `src` import when run as a plain script

ALL_STAGES = ("oof", "shap", "agreement", "calibration", "threshold", "report")
SHAP_SAMPLE_N = 5000
TUNERS = ("grid", "random", "optuna")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default=",".join(ALL_STAGES), help=f"comma-separated subset of {ALL_STAGES}")
    parser.add_argument("--quick", action="store_true", help="smoke-test sizes (sets EMBER_QUICK=1)")
    parser.add_argument("--n-jobs", type=int, default=2, help="threads (2: another job shares the 4-core VM)")
    args = parser.parse_args(argv)
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(args.stages) - set(ALL_STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    return args


ARGS = parse_args()
if ARGS.quick:
    os.environ["EMBER_QUICK"] = "1"   # before importing config, which reads it at import

import lightgbm as lgb  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from config import FIGURES, LEADERBOARDS, MODELS, PREDICTIONS, REPORTS, RUN_SUFFIX, SEED, TABLES  # noqa: E402
from src import calibration as cal  # noqa: E402
from src import data, explain  # noqa: E402
from src.eval import compute_metrics, paired_bootstrap, percentile_ci  # noqa: E402
from src.logging_utils import log_experiment  # noqa: E402

S = RUN_SUFFIX
OOF_FILE = PREDICTIONS / f"oof_lgbm_tuned{S}.npz"
THRESHOLD_FILE = MODELS / f"threshold{S}.json"
TIME_PROXY_NAMES = ("header__0", "authenticode__6", "authenticode__7")   # see src/leakage.py TIME_PROXIES
AUTOML_TOOLS = ("flaml", "autogluon", "h2o")


def out(folder: Path, stem: str, ext: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}{S}{ext}"


def machine() -> dict[str, Any]:
    """CPU count, RAM and OS: timings below are from this machine, not the Steps 2-3 laptop."""
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 if hasattr(os, "sysconf") else None
    return {"cpu_count": os.cpu_count(), "ram_gb": round(ram, 1) if ram else None,
            "os": platform.platform(), "processor": platform.processor() or platform.machine()}


def best_tuner() -> str:
    """The Step 2 tuner with the best CV AUC (chosen on CV only, as in src.tables.selected_tuner)."""
    summary = pd.read_csv(TABLES / f"tuning_summary{S}.csv")
    tuners = summary[summary["method"].isin(TUNERS)]
    return str(tuners.loc[tuners["best_cv_auc"].idxmax(), "method"])


def tuned_model() -> tuple[str, lgb.Booster, dict[str, Any]]:
    method = best_tuner()
    booster = lgb.Booster(model_file=str(MODELS / f"lgbm_{method}{S}.txt"))
    saved = json.loads((MODELS / f"lgbm_{method}{S}_params.json").read_text(encoding="utf-8"))
    return method, booster, saved


def saved_scores(name: str) -> dict[str, np.ndarray]:
    with np.load(PREDICTIONS / f"{name}{S}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_csv(table: pd.DataFrame, path: Path) -> Path:
    table.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def save_fig(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


# =============================================================================
# Stage: oof
# =============================================================================
def stage_oof(n_jobs: int) -> None:
    """Out-of-fold P(malicious) on train with the tuned params and the refit's fixed round count."""
    method, _booster, saved = tuned_model()
    params = {**saved["params"], "num_threads": n_jobs}
    n_rounds = int(saved["num_trees"])
    X, y, _meta = data.load_split("train")
    folds = data.get_cv_folds(subset=False)
    fold_seconds: list[float] = []

    def fit(X_fit: np.ndarray, y_fit: np.ndarray):
        start = time.perf_counter()
        booster = lgb.train(params, lgb.Dataset(X_fit, y_fit, params=params), num_boost_round=n_rounds)
        fold_seconds.append(time.perf_counter() - start)
        print(f"fold {len(fold_seconds)}/{len(folds)} fitted in {fold_seconds[-1]:.0f} s", flush=True)
        return lambda X_val: booster.predict(X_val, num_threads=n_jobs)

    start = time.perf_counter()
    oof = cal.oof_predictions(X, y, folds, fit)
    seconds = time.perf_counter() - start
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for k, (_fit_idx, val_idx) in enumerate(folds):
        fold_ids[val_idx] = k
    np.savez_compressed(OOF_FILE, y=y, p=oof, fold_ids=fold_ids, method=method, n_rounds=n_rounds)
    print(f"wrote {OOF_FILE.relative_to(ROOT)}")
    metrics = compute_metrics(y, oof)
    log_experiment({"step": "explain_oof", "method": method, "n_rounds": n_rounds, "n_folds": len(folds),
                    "train_rows": len(y), "seconds": seconds, "fold_seconds": fold_seconds, "n_jobs": n_jobs,
                    "seed": SEED, "machine": machine(), "oof_metrics": metrics, "output": OOF_FILE})


def load_oof() -> dict[str, np.ndarray]:
    if not OOF_FILE.exists():
        raise FileNotFoundError(f"{OOF_FILE} missing: run --stages oof first")
    with np.load(OOF_FILE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# =============================================================================
# Stage: shap
# =============================================================================
def shap_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """(positions in test_future, X, y, meta) of the stratified SHAP sample."""
    X_all, y_all, meta_all = data.load_split("test_future")
    pos = explain.stratified_sample(meta_all, SHAP_SAMPLE_N)
    return pos, X_all[pos], y_all[pos], meta_all.iloc[pos].reset_index(drop=True)


def check_scores(p: np.ndarray, saved: np.ndarray, label: str) -> float:
    """Fail if the model we explain does not reproduce the saved test_future scores."""
    diff = float(np.max(np.abs(p - saved)))
    if diff > 1e-5:
        raise RuntimeError(f"{label}: scores differ from saved predictions by {diff:g}")
    return diff


def flaml_shap(X: np.ndarray, names: list[str], n_jobs: int, saved_p: np.ndarray) -> dict[str, Any]:
    """TreeSHAP of FLAML's winner on the same rows, if it is a tree model we can explain."""
    import joblib

    # Our own Step 3 model file (created by src/automl_runs.py), not downloaded data.
    est = joblib.load(MODELS / "flaml" / f"best_estimator{S}.joblib")
    used = [str(c) for c in est.feature_names_in_]
    pos = {n: i for i, n in enumerate(names)}
    X_used = X[:, [pos[c] for c in used]]
    model = est.estimator
    kind = type(model).__name__
    if kind == "XGBClassifier":
        values, base = explain.xgb_shap(model, X_used, n_jobs)
    elif kind == "LGBMClassifier":
        values, base = explain.lgbm_shap(model.booster_, X_used, n_jobs)
    else:
        return {"kind": kind, "values": None}
    p = 1.0 / (1.0 + np.exp(-(base + values.sum(axis=1))))
    diff = check_scores(p, saved_p, f"FLAML {kind}")
    return {"kind": kind, "values": explain.expand_to_all(values, used, names), "base": base,
            "n_used": len(used), "max_abs_diff_vs_saved": diff}


def plot_family_bar(table: pd.DataFrame, n_files: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    t = table.sort_values("sum_mean_abs_shap")
    ax.barh(t["family"], t["sum_mean_abs_shap"], color="#3b6ea5", label="sum of per-feature mean |SHAP|")
    ax.barh(t["family"], t["grouped_mean_abs_shap"], color="#e0a030", height=0.4,
            label="mean |family SHAP| (family as one block)")
    for i, (v, n) in enumerate(zip(t["sum_mean_abs_shap"], t["n_features"])):
        ax.text(v, i, f"  {n} cols", va="center", fontsize=8, color="#444")
    ax.set_xlabel("importance (log-odds units)")
    ax.set_title(f"Tuned LightGBM: SHAP importance by EMBER feature family\n({n_files} stratified test_future files)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, path)


def time_proxy_detail(values: np.ndarray, X: np.ndarray, y: np.ndarray, names: list[str],
                      labels: list[str]) -> pd.DataFrame:
    """Does a time proxy act as a clock, or only as "signed / has a timestamp at all"?

    The signing-time columns are 0 for unsigned files, so a high SHAP rank could come from the
    0 / non-0 split alone. Among the non-zero rows, the Spearman correlation of value and SHAP
    shows whether later times push the score one way, i.e. whether the column works as a clock.
    """
    from scipy.stats import spearmanr

    rows = []
    for name in TIME_PROXY_NAMES:
        j = names.index(name)
        v, s = X[:, j], values[:, j]
        nz = v != 0
        rho = spearmanr(v[nz], s[nz])[0] if nz.sum() > 2 and np.ptp(v[nz]) > 0 else float("nan")
        rows.append({"feature": name, "column": j, "label": labels[j],
                     "share_zero_malicious": float(np.mean(~nz[y == 1])),
                     "share_zero_benign": float(np.mean(~nz[y == 0])),
                     "mean_abs_shap": float(np.abs(s).mean()),
                     "mean_abs_shap_zero_rows": float(np.abs(s[~nz]).mean()) if (~nz).any() else float("nan"),
                     "mean_abs_shap_nonzero_rows": float(np.abs(s[nz]).mean()) if nz.any() else float("nan"),
                     "spearman_value_vs_shap_nonzero": float(rho)})
    return pd.DataFrame(rows)


def case_words(case: str, row: pd.Series, contrib: pd.DataFrame) -> str:
    """Two generated sentences on why the model scored this file as it did."""
    toward = contrib[contrib["shap"] > 0]
    away = contrib[contrib["shap"] < 0]

    def fmt(frame: pd.DataFrame) -> str:
        return "; ".join(f"`{r.feature}` = {r.value:.4g} ({r.shap:+.2f})" for r in frame.itertuples()) or "none"

    truth = "malicious" if row["label"] == 1 else "benign"
    return (f"{case.upper()}: a {truth} file (sha256 {row['sha256'][:12]}..., week {row['week']}) scored "
            f"P(malicious) = {row['p']:.4f}. Largest pushes towards malicious: {fmt(toward)}. "
            f"Largest pushes towards benign: {fmt(away)}.")


def stage_shap(n_jobs: int) -> None:
    import shap

    method, booster, _saved = tuned_model()
    groups = data.feature_groups()
    names = explain.column_names(groups)
    labels = explain.readable_names(data.get_extractor())
    pos, X, y, meta = shap_sample()
    start = time.perf_counter()
    p = booster.predict(X, num_threads=n_jobs)
    saved = saved_scores(method)
    diff = check_scores(p, saved["test_future_p"][pos], "tuned LightGBM")
    values, base = explain.lgbm_shap(booster, X, n_jobs)
    shap_seconds = time.perf_counter() - start
    # Additivity: base + sum(SHAP) is exactly the model's log-odds.
    logit = np.log(p / (1 - p))
    additivity = float(np.max(np.abs(base + values.sum(axis=1) - logit)))

    imp = explain.feature_importance(values, names)
    feat = pd.DataFrame({"column": np.arange(len(names)), "feature": names, "label": labels,
                         "family": [explain.family_of(n) for n in names], "mean_abs_shap": imp.to_numpy()})
    feat["rank"] = feat["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    write_csv(feat.sort_values("rank"), out(TABLES, "shap_feature_importance", ".csv"))
    fam = explain.family_table(values, groups, labels)
    write_csv(fam, out(TABLES, "shap_family_importance", ".csv"))

    write_csv(time_proxy_detail(values, X, y, names, labels), out(TABLES, "shap_time_proxies", ".csv"))

    expl = shap.Explanation(values=values, base_values=np.full(len(y), base), data=X, feature_names=labels)
    plt.figure()
    shap.plots.beeswarm(expl, max_display=21, show=False, plot_size=(9, 8))   # 20 + "sum of the rest"
    plt.title(f"Tuned LightGBM ({method}): top 20 features by mean |SHAP|, {len(y)} test_future files")
    save_fig(plt.gcf(), out(FIGURES, "shap_beeswarm", ".png"))
    plot_family_bar(fam, len(y), out(FIGURES, "shap_family_bar", ".png"))

    cases = []
    for case, i in explain.pick_cases(y, p).items():
        if i is None:
            cases.append({"case": case, "position": None, "words": f"{case.upper()}: no such file in the sample."})
            continue
        row = pd.Series({"label": int(y[i]), "p": float(p[i]), "sha256": meta.loc[i, "sha256"],
                         "week": int(meta.loc[i, "week"])})
        contrib = explain.top_contributions(values[i], X[i], labels)
        plt.figure()
        shap.plots.waterfall(expl[i], max_display=12, show=False)
        plt.title(f"{case.upper()}: label={'malicious' if y[i] else 'benign'}, P(malicious)={p[i]:.4f}")
        save_fig(plt.gcf(), out(FIGURES, f"shap_waterfall_{case}", ".png"))
        cases.append({"case": case, "position": int(pos[i]), **row.to_dict(),
                      "logit": float(logit[i]), "base_value": base,
                      "top_features": "; ".join(contrib["feature"]),
                      "top_shap": "; ".join(f"{v:+.3f}" for v in contrib["shap"]),
                      "top_values": "; ".join(f"{v:.6g}" for v in contrib["value"]),
                      "words": case_words(case, row, contrib)})
    write_csv(pd.DataFrame(cases), out(TABLES, "shap_cases", ".csv"))

    fl = flaml_shap(X, names, n_jobs, saved_scores("automl_flaml")["test_future_p"][pos])
    if fl["values"] is not None:
        fl_imp = explain.feature_importance(fl["values"], names)
        write_csv(pd.DataFrame({"feature": names, "mean_abs_shap": fl_imp.to_numpy()}),
                  out(TABLES, "shap_feature_importance_flaml", ".csv"))
    notes = {"method": method, "sample_n": int(len(y)), "sample_malicious": int(y.sum()),
             "sample_weeks": sorted(int(w) for w in meta["week"].unique()), "base_value": base,
             "max_abs_diff_vs_saved_scores": diff, "additivity_max_abs_error": additivity,
             "shap_seconds": shap_seconds, "n_jobs": n_jobs, "machine": machine(),
             "flaml_model": fl["kind"], "flaml_explained": fl["values"] is not None,
             "flaml_n_features_used": fl.get("n_used"), "flaml_max_abs_diff_vs_saved": fl.get("max_abs_diff_vs_saved"),
             "shap_version": shap.__version__, "lightgbm_version": lgb.__version__}
    out(TABLES, "shap_notes", ".json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    log_experiment({"step": "explain_shap", **notes})


# =============================================================================
# Stage: agreement
# =============================================================================
def tool_importances(names: list[str], families: list[str]) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """Per-feature and per-family importance of every tool, plus what each number measures."""
    feat: dict[str, pd.Series] = {}
    fam: dict[str, pd.Series] = {}
    what: dict[str, str] = {}
    lgbm = pd.read_csv(TABLES / f"shap_feature_importance{S}.csv")
    feat["lgbm_tuned"] = explain.importance_from_csv(lgbm, "mean_abs_shap", names)
    n = json.loads((TABLES / f"shap_notes{S}.json").read_text(encoding="utf-8"))["sample_n"]
    what["lgbm_tuned"] = f"mean |TreeSHAP| on {n} stratified test_future files"
    flaml_shap_csv = TABLES / f"shap_feature_importance_flaml{S}.csv"
    if flaml_shap_csv.exists():
        feat["flaml"] = explain.importance_from_csv(pd.read_csv(flaml_shap_csv), "mean_abs_shap", names)
        what["flaml"] = f"mean |TreeSHAP| of FLAML's tree winner on the same {n} files"
    else:
        flaml_fi = pd.read_csv(LEADERBOARDS / f"flaml_feature_importance{S}.csv")
        feat["flaml"] = explain.importance_from_csv(flaml_fi, "importance", names)
        what["flaml"] = "FLAML winner's native feature_importances_"
    ag = pd.read_csv(LEADERBOARDS / f"autogluon_feature_importance{S}.csv")
    feat["autogluon"] = explain.importance_from_csv(ag, "importance", names)
    what["autogluon"] = ("permutation importance on AutoGluon's internal validation rows (features: 1 shuffle; "
                         "families: shuffled as blocks)")
    h2o = pd.read_csv(LEADERBOARDS / f"h2o_varimp{S}.csv")
    feat["h2o"] = explain.importance_from_csv(h2o, "relative_importance", names)
    what["h2o"] = "H2O varimp (split gain) of the GBM that carries the stacked ensemble"
    for tool, s in feat.items():
        fam[tool] = explain.to_families(s, families)
    # AutoGluon measured families as blocks; that is the better family number for it.
    ag_fam = pd.read_csv(LEADERBOARDS / f"autogluon_family_importance{S}.csv").set_index("family")["importance"]
    fam["autogluon"] = ag_fam.reindex(families).fillna(0.0)
    return feat, fam, what


def plot_agreement(shares: pd.DataFrame, table: pd.DataFrame, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    im = ax1.imshow(shares.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    ax1.set_xticks(range(shares.shape[1]), shares.columns)
    ax1.set_yticks(range(shares.shape[0]), shares.index)
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            v = shares.iat[i, j]
            ax1.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                     color="white" if v > 0.6 * shares.to_numpy().max() else "black")
    ax1.set_title("Share of each tool's importance per feature family")
    fig.colorbar(im, ax=ax1, fraction=0.04)
    tools = list(shares.columns)
    mat = np.eye(len(tools))
    for r in table.itertuples():
        i, j = tools.index(r.tool_a), tools.index(r.tool_b)
        mat[i, j] = mat[j, i] = r.family_spearman
    im2 = ax2.imshow(mat, cmap="RdBu", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(tools)), tools, rotation=30)
    ax2.set_yticks(range(len(tools)), tools)
    for i in range(len(tools)):
        for j in range(len(tools)):
            ax2.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax2.set_title("Spearman rank correlation of family importance")
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    fig.tight_layout()
    return save_fig(fig, path)


def stage_agreement() -> None:
    groups = data.feature_groups()
    names = explain.column_names(groups)
    families = [g[0] for g in groups]
    feat, fam, what = tool_importances(names, families)
    table = explain.agreement_table(feat, fam)
    table["measure_a"] = table["tool_a"].map(what)
    table["measure_b"] = table["tool_b"].map(what)
    write_csv(table, out(TABLES, "cross_tool_agreement", ".csv"))
    shares = pd.DataFrame({t: s / s.sum() for t, s in fam.items()})
    write_csv(shares.rename_axis("family").reset_index(), out(TABLES, "cross_tool_family_shares", ".csv"))
    plot_agreement(shares, table, out(FIGURES, "cross_tool_agreement", ".png"))
    labels = explain.readable_names(data.get_extractor())
    col = {n: i for i, n in enumerate(names)}
    rows = [{"feature": n, "column": col[n], "label": labels[col[n]],
             **{f"rank_{t}": explain.rank_of(s, n) for t, s in feat.items()},
             **{f"importance_{t}": float(s[n]) for t, s in feat.items()}} for n in TIME_PROXY_NAMES]
    write_csv(pd.DataFrame(rows), out(TABLES, "time_proxy_ranks", ".csv"))
    top = pd.DataFrame({t: [labels[col[n]] for n in explain.top_k(s, 20)] for t, s in feat.items()})
    top.insert(0, "rank", np.arange(1, 21))
    write_csv(top, out(TABLES, "cross_tool_top20", ".csv"))


# =============================================================================
# Stage: calibration
# =============================================================================
def model_scores() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{label: (y, p)} on test_future from the saved prediction files."""
    method = best_tuner()
    sources = {"lgbm_tuned": method, "flaml": "automl_flaml", "autogluon": "automl_autogluon",
               "h2o": "automl_h2o", "logreg": "logreg"}
    scored = {}
    for label, stem in sources.items():
        z = saved_scores(stem)
        scored[label] = (z["test_future_y"].astype(int), z["test_future_p"].astype(float))
    y0 = scored["lgbm_tuned"][0]
    for label, (y, _p) in scored.items():
        if not np.array_equal(y, y0):
            raise RuntimeError(f"{label}: test_future labels are in a different order")
    return scored


def plot_reliability(curves: dict[str, pd.DataFrame], cal_curves: dict[str, pd.DataFrame],
                     scores: dict[str, np.ndarray], path: Path) -> Path:
    """Reliability curves (top) and score histograms (bottom).

    The histograms matter: almost every file scores near 0 or 1, so the middle bins of the
    curve rest on few files and zig-zag; the Brier score is dominated by the extreme bins.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), gridspec_kw={"height_ratios": [2.2, 1]})
    for ax, group, title in ((axes[0, 0], curves, "Models as trained (test_future)"),
                             (axes[0, 1], cal_curves, "Tuned LightGBM: raw vs calibrated on TRAIN OOF")):
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
        for label, c in group.items():
            ax.plot(c["mean_score"], c["observed"], marker="o", ms=3, lw=1.2, label=label)
        ax.set_xlabel("mean predicted P(malicious) in bin")
        ax.set_ylabel("observed malicious rate")
        ax.set_title(title)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    bins = np.linspace(0, 1, cal.N_CAL_BINS + 1)
    for ax, group in ((axes[1, 0], curves), (axes[1, 1], cal_curves)):
        for label in group:
            ax.hist(scores[label], bins=bins, histtype="step", lw=1.2, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("predicted P(malicious)")
        ax.set_ylabel("files per bin (log)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Reliability curves ({cal.N_CAL_BINS} equal-width bins) and score histograms")
    fig.tight_layout()
    return save_fig(fig, path)


def stage_calibration() -> None:
    scored = model_scores()
    y = scored["lgbm_tuned"][0]
    oof = load_oof()
    calibrators = [cal.fit_isotonic(oof["y"], oof["p"]), cal.fit_platt(oof["y"], oof["p"])]
    for c in calibrators:
        scored[f"lgbm_tuned+{c.name}"] = (y, c.transform(scored["lgbm_tuned"][1]))
    boot = paired_bootstrap(y, {k: p for k, (_y, p) in scored.items()}, metric_fn=cal.brier_only)
    rows = []
    for label, (_y, p) in scored.items():
        lo, hi = percentile_ci(boot[label]["brier"])
        m = compute_metrics(y, p)
        rows.append({"model": label, "test_set": "test_future", **cal.calibration_metrics(y, p),
                     "brier_ci_low": lo, "brier_ci_high": hi,
                     "brier_diff_vs_lgbm_tuned": float(np.mean((p - y) ** 2) - np.mean((scored["lgbm_tuned"][1] - y) ** 2)),
                     "mean_score": float(np.mean(p)), "pos_rate": float(np.mean(y)),
                     "roc_auc": m["roc_auc"], "f1_at_0.5": m["f1"], "fpr_at_0.5": m["fpr"],
                     "recall_at_0.5": m["recall"]})
    rows.append({"model": "lgbm_tuned (train OOF)", "test_set": "train_oof", **cal.calibration_metrics(oof["y"], oof["p"]),
                 "mean_score": float(np.mean(oof["p"])), "pos_rate": float(np.mean(oof["y"])),
                 "roc_auc": compute_metrics(oof["y"], oof["p"])["roc_auc"]})
    table = pd.DataFrame(rows)
    # Paired difference CI against the tuned LightGBM: is any Brier gap bigger than test-set luck?
    base = boot["lgbm_tuned"]["brier"]
    for i, label in enumerate(table["model"]):
        if label in boot and label != "lgbm_tuned":
            lo, hi = percentile_ci(boot[label]["brier"] - base)
            table.loc[i, "brier_diff_ci_low"], table.loc[i, "brier_diff_ci_high"] = lo, hi
    write_csv(table, out(TABLES, "calibration", ".csv"))
    raw = {k: cal.reliability_curve(y, p) for k, (_y, p) in scored.items() if "+" not in k}
    calibrated = {k: cal.reliability_curve(y, p) for k, (_y, p) in scored.items() if k.startswith("lgbm_tuned")}
    plot_reliability(raw, calibrated, {k: p for k, (_y, p) in scored.items()}, out(FIGURES, "calibration", ".png"))


# =============================================================================
# Stage: threshold
# =============================================================================
def plot_pr_cost(oof: dict[str, np.ndarray], y_f: np.ndarray, p_f: np.ndarray, table: pd.DataFrame, path: Path) -> Path:
    from sklearn.metrics import precision_recall_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for label, (yy, pp) in {"train OOF": (oof["y"], oof["p"]), "test_future": (y_f, p_f)}.items():
        prec, rec, _ = precision_recall_curve(yy, pp)
        ax1.plot(rec, prec, lw=1.4, label=label)
    for r in table.itertuples():
        ax1.scatter(r.test_future_recall, r.test_future_precision, s=40, zorder=3,
                    label=f"C_FN/C_FP={r.cost_ratio:g}: t={r.threshold:.3g} (test_future)")
    # Zoom on the operating region: all models live in the top-right corner.
    low = min(table["test_future_recall"].min(), table["test_future_precision"].min(),
              table["oof_recall"].min(), table["oof_precision"].min())
    low = max(0.0, np.floor((low - 0.02) * 50) / 50)
    ax1.set_xlim(low, 1.0)
    ax1.set_ylim(low, 1.0)
    ax1.set_xlabel("recall (malware caught)")
    ax1.set_ylabel("precision")
    ax1.set_title("Precision-recall (zoomed); points = thresholds chosen on train OOF")
    ax1.legend(fontsize=7, frameon=False, loc="lower left")
    for r in table.itertuples():
        curve = cal.cost_curve(oof["y"], oof["p"], r.cost_ratio)
        curve = curve[np.isfinite(curve["threshold"])]
        ax2.plot(curve["threshold"], 1000 * curve["cost"] / len(oof["y"]), lw=1.2, label=f"C_FN/C_FP={r.cost_ratio:g}")
        ax2.axvline(r.threshold, lw=0.6, ls=":", color=ax2.lines[-1].get_color())
    ax2.set_yscale("log")
    # Logit x-axis: the chosen cuts sit close to 0, where a linear axis squeezes them together.
    ax2.set_xscale("logit")
    ax2.set_xlim(1e-5, 1 - 1e-5)
    ax2.set_xlabel("threshold (flag if P(malicious) >= t), logit scale")
    ax2.set_ylabel("cost per 1000 train files (C_FP = 1)")
    ax2.set_title("Cost on train OOF scores; dotted line = chosen threshold")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, path)


def stage_threshold() -> None:
    oof = load_oof()
    method = best_tuner()
    z = saved_scores(method)
    y_f, p_f = z["test_future_y"].astype(int), z["test_future_p"].astype(float)
    rows = []
    for ratio in cal.COST_RATIOS:
        chosen = cal.choose_threshold(oof["y"], oof["p"], ratio)
        on_oof = cal.confusion_at(oof["y"], oof["p"], chosen["threshold"], ratio)
        on_future = cal.confusion_at(y_f, p_f, chosen["threshold"], ratio)
        at_half = cal.confusion_at(y_f, p_f, 0.5, ratio)
        rows.append({"cost_ratio": ratio, "threshold": chosen["threshold"],
                     **{f"oof_{k}": v for k, v in on_oof.items()},
                     **{f"test_future_{k}": v for k, v in on_future.items()},
                     "test_future_cost_per_1000_at_0.5": at_half["cost_per_1000"],
                     "test_future_fn_at_0.5": at_half["fn"], "test_future_fp_at_0.5": at_half["fp"]})
    table = pd.DataFrame(rows)
    write_csv(table, out(TABLES, "threshold", ".csv"))
    plot_pr_cost(oof, y_f, p_f, table, out(FIGURES, "pr_cost", ".png"))
    pick = table.loc[table["cost_ratio"] == cal.DEFAULT_COST_RATIO].iloc[0]
    record = {
        "threshold": float(pick["threshold"]), "cost_ratio_fn_to_fp": cal.DEFAULT_COST_RATIO,
        "c_fn": cal.DEFAULT_COST_RATIO, "c_fp": 1.0, "rule": "flag as malicious if P(malicious) >= threshold",
        "model": f"models/lgbm_{method}{S}.txt",
        "how_chosen": ("minimises C_FN*FN + C_FP*FP over every distinct score on TRAIN out-of-fold predictions "
                       "(persisted 3-fold CV of the train split, data/splits/cv_fold_ids; tuned params, fixed "
                       f"{int(oof['n_rounds'])} rounds as the refit); cut placed midway to the next lower score. "
                       "No test set was used."),
        "source_of_predictions": str(OOF_FILE.relative_to(ROOT)),
        "oof_rows": int(len(oof["y"])), "oof_cost_per_1000": float(pick["oof_cost_per_1000"]),
        "test_future_check": {k: float(pick[f"test_future_{k}"]) for k in
                              ("precision", "recall", "fpr", "fn", "fp", "cost_per_1000")},
        "caveat": "classes are 50/50 by construction; re-derive the threshold for a deployment with another prevalence",
        "seed": SEED,
    }
    THRESHOLD_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {THRESHOLD_FILE.relative_to(ROOT)}")
    log_experiment({"step": "threshold", **record})


# =============================================================================
# Stage: report
# =============================================================================
def md_table(frame: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and c not in floatfmt and np.isfinite(v) and v.is_integer() and abs(v) >= 1:
                cells.append(str(int(v)))   # counts read back from CSV as floats
            elif isinstance(v, float):
                cells.append("" if np.isnan(v) else format(v, floatfmt.get(c, ".4f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_table(stem: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{stem}{S}.csv")


def fig_link(stem: str) -> str:
    return f"![{stem}](figures/{stem}{S}.png)"


def oof_record() -> dict[str, Any]:
    """The latest explain_oof log record of this mode (timings and machine for the write-up)."""
    from src.logging_utils import read_records

    recs = [r for r in read_records() if r.get("step") == "explain_oof"]
    if not recs:
        raise RuntimeError("no explain_oof record in the experiment log; run --stages oof")
    return recs[-1]


def section_shap(notes: dict[str, Any]) -> list[str]:
    fam = read_table("shap_family_importance")
    feat = read_table("shap_feature_importance").head(20)
    fam_view = fam[["rank", "family", "n_features", "sum_mean_abs_shap", "grouped_mean_abs_shap", "share",
                    "top_feature"]].copy()
    fam_view["share"] = (100 * fam_view["share"]).map(lambda v: f"{v:.1f}%")
    top_fams = ", ".join(f"`{f}`" for f in fam["family"].head(3))
    per_feat = fam.sort_values("mean_abs_shap_per_feature", ascending=False)["family"].head(3)
    return [
        "## 1. SHAP on the tuned LightGBM", "",
        f"* Model: `models/lgbm_{notes['method']}{S}.txt`, the Step 2 tuner with the best CV AUC "
        f"(`{notes['method']}`; chosen on CV, never on a test set).",
        f"* Files explained: a label x week stratified sample of {notes['sample_n']} test_future files "
        f"({notes['sample_malicious']} malicious, weeks {min(notes['sample_weeks'])}-{max(notes['sample_weeks'])}). "
        "Explaining a fixed model is not tuning, so test rows are allowed here; nothing below feeds back into a model.",
        "* Method: TreeSHAP (LightGBM's built-in `pred_contrib`, the same algorithm as `shap.TreeExplainer`; "
        "`tests/test_explain.py` checks they agree). Values are in log-odds.",
        f"* Checks: the model reproduces the saved test_future scores (max abs difference "
        f"{notes['max_abs_diff_vs_saved_scores']:.1e}); base value + sum of SHAP equals the model's log-odds "
        f"for every file (max error {notes['additivity_max_abs_error']:.1e}).", "",
        fig_link("shap_beeswarm"), "",
        "### Importance by EMBER feature family", "",
        "Two numbers per family: `sum_mean_abs_shap` adds up each column's mean |SHAP| (comparable with "
        "summed gain), `grouped_mean_abs_shap` treats the family as one block (mean of |sum of its SHAP "
        "values|), which is smaller when the family's columns push in opposite directions.", "",
        md_table(fam_view, {"sum_mean_abs_shap": ".3f", "grouped_mean_abs_shap": ".3f"}), "",
        fig_link("shap_family_bar"), "",
        f"The three families with the most total SHAP mass are {top_fams}. Per column, the densest families are "
        + ", ".join(f"`{f}`" for f in per_feat) + " (few columns, each one informative). Large families such as "
        "`imports` (1,282 hashed columns) collect their mass from many small contributions.", "",
        "### Top 20 columns", "",
        md_table(feat[["rank", "column", "label", "mean_abs_shap"]]), "",
    ]


def section_time_proxies() -> list[str]:
    ranks = read_table("time_proxy_ranks")
    detail = read_table("shap_time_proxies")
    view = ranks[["column", "label"] + [c for c in ranks.columns if c.startswith("rank_")]]
    d = detail[["label", "share_zero_malicious", "share_zero_benign", "mean_abs_shap_zero_rows",
                "mean_abs_shap_nonzero_rows", "spearman_value_vs_shap_nonzero"]]
    lines = ["## 2. Do the time proxies rank high?", "",
             "Three columns already in the feature vector carry time (located in thrember's source; see "
             "`src/leakage.py` `TIME_PROXIES`): the COFF header timestamp (column 696) and the two authenticode "
             "signing-time columns (2478, 2479). Their rank among all 2,568 columns, per tool "
             "(1 = most important; ties share the best rank):", "",
             md_table(view), ""]
    for r in ranks.itertuples():
        level = "in the top 20" if r.rank_lgbm_tuned <= 20 else ("in the top 100" if r.rank_lgbm_tuned <= 100
                                                                   else "outside the top 100")
        lines.append(f"* `{r.label}` ranks {r.rank_lgbm_tuned} for the tuned LightGBM ({level}).")
    lines += ["", "Whether a proxy acts as a clock or only as a 0 / non-0 flag (unsigned files have signing "
              "time 0). `spearman_value_vs_shap_nonzero` is the rank correlation between the column's value and "
              "its SHAP value over the files where it is non-zero: far from 0 means the model reads the value "
              "itself (a negative sign means larger values push towards benign), near 0 means only the "
              "0 / non-0 split matters.", "", md_table(d), ""]
    for r in detail.itertuples():
        lines.append(f"* `{r.label}`: 0 for {r.share_zero_malicious:.0%} of malicious and {r.share_zero_benign:.0%} "
                     f"of benign files; mean |SHAP| {r.mean_abs_shap_zero_rows:.3f} where 0 and "
                     f"{r.mean_abs_shap_nonzero_rows:.3f} where non-zero; value-vs-SHAP rho "
                     f"{r.spearman_value_vs_shap_nonzero:+.2f} on the non-zero files.")
    rank_cols = [c for c in ranks.columns if c.startswith("rank_")]
    top5_all = ranks.loc[(ranks[rank_cols] <= 5).all(axis=1), "label"].tolist()
    sig = detail.set_index("feature").loc["authenticode__6"]
    lines += ["", "What this means:", ""]
    if top5_all:
        lines.append("* " + ", ".join(f"`{x}`" for x in top5_all) + " is in the top 5 for every tool: the "
                     "signature columns are among the strongest signals in this data, whichever tool learns it.")
    lines += [
        f"* The signing-time column is mostly a 'signed or not' flag ({sig.share_zero_malicious:.0%} of malicious "
        f"vs {sig.share_zero_benign:.0%} of benign files are unsigned), but on signed files its value matters too "
        f"(mean |SHAP| {sig.mean_abs_shap_nonzero_rows:.2f} there; rho {sig.spearman_value_vs_shap_nonzero:+.2f}), "
        "so the model does partly use it as a clock. That is a time proxy inside X. SHAP shows that the model "
        "uses it, not whether it helps or hurts on later files: the ablation that removes these columns and "
        "retrains answers that, in the leakage write-up (`reports/leakage.md`).",
        "* AutoGluon's column ranks come from one permutation per column on its validation rows; columns with "
        "importance at or below 0 (such as the COFF timestamp there) get arbitrary low ranks, so a rank in the "
        "thousands means 'no measurable effect', not 'certainly useless'.", ""]
    return lines


def section_cases() -> list[str]:
    cases = read_table("shap_cases")
    lines = ["## 3. Three local explanations (waterfall plots)", "",
             "Picked at threshold 0.5 on the SHAP sample: the most confident correct malicious file (TP), the "
             "benign file scored most malicious (FP), and the malicious file scored most benign (FN). Each "
             "waterfall starts at the base value (average log-odds) and adds one column's SHAP value per bar. "
             "The sentences are generated from the SHAP values; the reading after them is ours.", ""]
    for r in cases.itertuples():
        lines += [f"### {r.case.upper()}", "", str(r.words), ""]
        if isinstance(r.position, (int, float)) and not pd.isna(r.position):
            lines += [fig_link(f"shap_waterfall_{r.case}"), ""]
        lines += case_reading(r.case, getattr(r, "top_features", "")) + [""]
    return lines


# Our reading of the three full-run waterfalls, written after looking at them. It names features,
# not numbers (those are in the generated sentence). Each reading lists the features it relies on;
# if a re-run picks a file whose top-5 lacks them, the reading is dropped instead of going stale.
CASE_READING: dict[str, tuple[tuple[str, ...], str]] = {
    "tp": (("pefilewarnings: warning[Suspicious flags set for section...]", "strings: count[dos_msg]"),
           "Reading: two positive signals lead: pefile flags a section with suspicious permissions, and the file "
           "contains the 'This program cannot be run in DOS mode' stub more than once, a typical sign of an "
           "embedded second executable (droppers, packers). Absences add to it: not a DLL, no exports, no "
           "authenticode signature. Still, the 'other features' bar is the largest: the score is a sum of "
           "many small pushes, not one rule."),
    "fp": (("datadirectories: EXPORT.size", "header: coff.char.DLL", "authenticode: latest_signing_time"),
           "Reading: this benign file has the profile the model has learned for malware: an executable (not a "
           "DLL) with no exports and no authenticode signature, plus a large heap reserve and a "
           "high-entropy section. The absence of the suspicious-section warning pulls the other way but cannot "
           "outweigh the rest. Unsigned, unusually built benign software is the classic false positive of "
           "static PE detectors."),
    "fn": (("authenticode: latest_signing_time", "header: coff.char.DLL"),
           "Reading: this malware carries the markers the model has learned for benign software: it is an "
           "authenticode-signed DLL with exports and a SECURITY directory. The signing-time column is the "
           "largest single push. Signed malware (stolen or abused certificates) is exactly what a model "
           "that leans on signature presence will miss."),
}


def case_reading(case: str, top_features: str) -> list[str]:
    if config.QUICK or case not in CASE_READING:
        return ["(The hand-written reading is for the full run only.)"]
    needed, text = CASE_READING[case]
    present = set(str(top_features).split("; "))
    if not set(needed) <= present:
        return ["(Reading omitted: the explained file changed since it was written; see the generated sentence.)"]
    return [text]


def section_agreement() -> list[str]:
    table = read_table("cross_tool_agreement")
    shares = read_table("cross_tool_family_shares")
    top = read_table("cross_tool_top20")
    view = table[["tool_a", "tool_b", "top5_overlap", "top20_overlap", "family_spearman", "family_spearman_p",
                  "top_family_a", "top_family_b"]]
    measures = dict(zip(table["tool_a"], table["measure_a"])) | dict(zip(table["tool_b"], table["measure_b"]))
    sh = shares.set_index("family")
    sh = (100 * sh).round(1).astype(str) + "%"
    rho = table["family_spearman"]
    best = table.loc[rho.idxmax()]
    worst = table.loc[rho.idxmin()]
    return [
        "## 4. Do the tools agree on what matters?", "",
        "Each tool's importance, as it measures it:", "",
        *[f"* `{t}`: {m}" for t, m in measures.items()], "",
        "Per-feature importances are summed into the 12 families (AutoGluon's family number is its own block "
        "permutation of each family, which is the more faithful family measure). Top-k overlap = how many of "
        "one tool's k most important columns are also in the other's k; Spearman = rank correlation of the "
        "12 family importances (with 12 points, |rho| below about 0.58 is not significant at 5%).", "",
        md_table(view, {"family_spearman": ".2f", "family_spearman_p": ".3f"}), "",
        f"Most similar pair at family level: `{best.tool_a}` and `{best.tool_b}` (rho = {best.family_spearman:.2f}); "
        f"least similar: `{worst.tool_a}` and `{worst.tool_b}` (rho = {worst.family_spearman:.2f}).", "",
        fig_link("cross_tool_agreement"), "",
        "Share of each tool's importance by family:", "",
        md_table(sh.reset_index()), "",
        "Top 20 columns per tool:", "",
        md_table(top), "",
        "Caveats: the measures differ (SHAP on test_future files; split gain on train; permutation on "
        "AutoGluon's validation rows), and AutoGluon's per-column permutation used one shuffle per column, so "
        "its column ranks are noisy; its family ranks used several shuffles. Gain-based varimp piles credit "
        "onto the first strong split (H2O), so a family can dominate gain while SHAP spreads credit more evenly. "
        "Agreement at family level with disagreement on single hashed columns is expected: correlated or "
        "near-duplicate columns can substitute for each other.", "",
    ]


def section_calibration() -> list[str]:
    t = read_table("calibration")
    view = t[["model", "test_set", "brier", "brier_ci_low", "brier_ci_high", "brier_diff_ci_low",
              "brier_diff_ci_high", "log_loss", "ece", "roc_auc"]]
    raw = t[t["test_set"] == "test_future"].set_index("model")
    lg = raw.loc["lgbm_tuned"]
    best = raw.drop(index=[m for m in raw.index if "+" in m]).sort_values("brier").index[0]
    lines = [
        "## 5. Calibration (test_future)", "",
        "Brier score = mean squared error of the probability (lower is better); ECE = weighted mean gap "
        f"between score and observed malicious rate over {cal.N_CAL_BINS} equal-width bins. The CI columns are "
        "95% paired-bootstrap intervals (1000 resamples); `brier_diff_*` is the CI of (model - tuned LightGBM), "
        "so an interval that excludes 0 is a real difference. Isotonic and Platt calibrators were fitted on the "
        "TRAIN out-of-fold scores (section 6), never on a test set, and then applied to the tuned model's "
        "test_future scores.", "",
        md_table(view, {c: ".4f" for c in view.columns if c not in ("model", "test_set")}), "",
        fig_link("calibration"), "",
        f"* Lowest Brier score among the uncalibrated models: `{best}`. The tuned LightGBM scores "
        f"{lg.brier:.4f} with ECE {lg.ece:.4f}.",
    ]
    for name in ("lgbm_tuned+isotonic", "lgbm_tuned+platt"):
        if name in raw.index:
            r = raw.loc[name]
            sig = "a real improvement" if r.brier_diff_ci_high < 0 else (
                "worse" if r.brier_diff_ci_low > 0 else "not distinguishable from no calibration")
            lines.append(f"* `{name}`: Brier {r.brier:.4f} (difference CI {r.brier_diff_ci_low:+.4f} to "
                         f"{r.brier_diff_ci_high:+.4f}: {sig}).")
    lines += ["* Almost every file scores near 0 or 1 (bottom panels), so the middle of each reliability curve "
              "rests on few files and zig-zags; read the Brier and ECE numbers, not the wiggles.",
              "* The calibrators learn from fold models trained on two thirds of train, while the deployed model "
              "is trained on all of it and is slightly more confident; and test_future is later in time than "
              "train. Both limit how well a train-fitted calibrator transfers.", ""]
    return lines


def section_threshold(rec: dict[str, Any]) -> list[str]:
    t = read_table("threshold")
    th = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
    view = t[["cost_ratio", "threshold", "oof_fn", "oof_fp", "oof_cost_per_1000", "test_future_precision",
              "test_future_recall", "test_future_fpr", "test_future_fn", "test_future_fp",
              "test_future_cost_per_1000", "test_future_cost_per_1000_at_0.5"]]
    pick = t.loc[t["cost_ratio"] == cal.DEFAULT_COST_RATIO].iloc[0]
    better = pick["test_future_cost_per_1000"] < pick["test_future_cost_per_1000_at_0.5"]
    return [
        "## 6. Cost-sensitive threshold", "",
        f"A missed malware file (FN) is taken to cost C_FN = {cal.DEFAULT_COST_RATIO:g} false alarms (C_FP = 1). "
        "The threshold minimising C_FN*FN + C_FP*FP is found exactly (every distinct score is tried) on the "
        f"TRAIN out-of-fold scores: {th['oof_rows']} files, each scored by a model fitted on the other two "
        f"folds (persisted 3-fold CV of train, tuned params, {rec['n_rounds']} rounds). No test set is used to choose it; "
        "test_future only reports how the choice holds up later in time.", "",
        md_table(view, {"threshold": ".4g", "cost_ratio": ".0f", "oof_cost_per_1000": ".2f",
                        "test_future_cost_per_1000": ".2f", "test_future_cost_per_1000_at_0.5": ".2f"}), "",
        fig_link("pr_cost"), "",
        f"* Deployed choice (`models/threshold{S}.json`): threshold {th['threshold']:.4g} at C_FN/C_FP = "
        f"{th['cost_ratio_fn_to_fp']:g}. On test_future it gives precision {pick['test_future_precision']:.4f}, "
        f"recall {pick['test_future_recall']:.4f} and FPR {pick['test_future_fpr']:.4f}, a cost of "
        f"{pick['test_future_cost_per_1000']:.2f} per 1000 files against {pick['test_future_cost_per_1000_at_0.5']:.2f} "
        f"at the default 0.5 ({'lower' if better else 'not lower'}).",
        f"* Later in time the same cut flags more benign files: FPR {pick['oof_fpr']:.4f} on train OOF vs "
        f"{pick['test_future_fpr']:.4f} on test_future, recall {pick['oof_recall']:.4f} vs "
        f"{pick['test_future_recall']:.4f}. The cut sits deep in the tail of the score distribution, where "
        "drift in benign software shows up first.",
        "* Ratios where the OOF-chosen cut did NOT beat the default 0.5 on test_future: "
        + (", ".join(f"{r:g}" for r in t.loc[t["test_future_cost_per_1000"] >= t["test_future_cost_per_1000_at_0.5"],
                                               "cost_ratio"]) or "none") + ".",
        "* A higher miss cost moves the threshold down (more files flagged, fewer misses, more false alarms).",
        "* The OOF scores come from fold models fitted on two thirds of train; the deployed model is fitted on "
        "all of train and may be slightly more confident, which matters most for a cut this far into the tail.",
        "* The classes are 50/50 by construction. In real traffic malware is rarer, so the same threshold gives "
        "more false alarms per detection; the threshold should be re-derived for the deployment's prevalence and "
        "real costs.", ""]


def stage_report() -> None:
    notes = json.loads((TABLES / f"shap_notes{S}.json").read_text(encoding="utf-8"))
    rec = oof_record()
    mach = rec["machine"]
    lines = [
        f"# Explainability, calibration and cost{' (QUICK smoke run: tiny samples, not results)' if S else ''}", "",
        "Step 4 part C. Generated by `scripts/run_explain.py --stages report` from the tables in "
        "`reports/tables/` and the saved models and predictions; no number here is typed by hand.", "",
        f"Machine: {mach['cpu_count']} CPUs, {mach['ram_gb']} GB RAM, {mach['os']}, with "
        f"{rec['n_jobs']} threads (another job shared the machine). Steps 2-3 ran on a 16-core laptop with "
        "63 GB RAM, so the timings below are not comparable with theirs. Train OOF fits took "
        f"{rec['seconds'] / 60:.1f} min in total; SHAP took {notes['shap_seconds']:.0f} s.", "",
    ]
    lines += section_shap(notes) + section_time_proxies() + section_cases() + section_agreement()
    lines += section_calibration() + section_threshold(rec)
    lines += ["## Reproduce", "", "```bash",
              ".venv/bin/python scripts/run_explain.py --stages oof        # long: run detached, writes a log",
              ".venv/bin/python scripts/run_explain.py --stages shap,agreement,calibration,threshold,report",
              "```", ""]
    path = REPORTS / f"explainability{S}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    for stage in ARGS.stages:
        print(f"=== stage {stage} ({'quick' if config.QUICK else 'full'}) ===", flush=True)
        t0 = time.perf_counter()
        {"oof": lambda: stage_oof(ARGS.n_jobs), "shap": lambda: stage_shap(ARGS.n_jobs),
         "agreement": stage_agreement, "calibration": stage_calibration,
         "threshold": stage_threshold, "report": stage_report}[stage]()
        print(f"=== stage {stage} done in {time.perf_counter() - t0:.0f} s ===", flush=True)
