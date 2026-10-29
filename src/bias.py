"""Step 4 (part B): bias, imbalance and the base-rate effect for the tuned LightGBM.

  class_balance_table()  class counts of every split (50/50 by construction).
  slice_table()          per-slice detection rate (TPR) and benign false-positive rate (FPR),
                         with Wilson 95% CIs, on both test sets. Slices: week, file-size quartile
                         (edges from TRAIN), packer tag, high-entropy proxy, signed, family known.
  class_weight_specs()   class-weighted refits (balanced and a deployment-prior weighting).
  prevalence_table()     precision and alarms per 10,000 files at 1%, 5% and 10% malware
                         prevalence, from TPR and FPR (the base-rate effect).
All thresholds are fixed beforehand (0.5, the deployment threshold, or the 1%-FPR threshold
chosen on TRAIN out-of-fold scores); none is chosen on a test set.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIGURES, RUN_SUFFIX  # noqa: E402
from src import tables  # noqa: E402
from src.temporal import FitSpec, operating_metrics  # noqa: E402

MIN_SLICE_N: int = 100        # fewer files of the relevant class -> the slice is marked too small
MIN_PACKER_N: int = 50        # a packer gets its own slice only with this many files in a test set
HIGH_ENTROPY: float = 7.0     # bits per byte; whole-file entropy above this suggests packing/compression
ENTROPY_COLUMN: int = 1       # general group, offset 1 = whole-file entropy (thrember GeneralFileInfo)
Z_95: float = 1.959963984540054
PREVALENCES: tuple[float, ...] = (0.01, 0.05, 0.10)
FILES_PER_ALARM_BASE: int = 10_000


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion: sensible even for 0 or n successes and small n."""
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # For 0 (or n) successes the bound is exactly 0 (or 1); the float formula would leave e.g. 8.7e-19
    low = 0.0 if successes == 0 else max(0.0, float(centre - half))
    high = 1.0 if successes == n else min(1.0, float(centre + half))
    return low, high


def class_balance_table(splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """n, benign, malicious and malicious share of each split."""
    rows = []
    for name, y in splits.items():
        n_mal = int(np.sum(y == 1))
        rows.append({"split": name, "n": len(y), "n_benign": len(y) - n_mal, "n_malicious": n_mal,
                     "malicious_share": n_mal / len(y)})
    return pd.DataFrame(rows)


def class_weight_specs(y_train: np.ndarray, deployment_prevalence: float = 0.10) -> list[FitSpec]:
    """Refits of the tuned configuration on all of train, without and with class weights.

    none         no weights: the reference refit on THIS machine with the same thread count, so the
                 weighted variants are compared with an identical setup (the saved Step 2 model was
                 fit on another machine with 16 threads; LightGBM is only bit-identical for the same
                 thread count, so none-vs-saved also shows how large that numerical noise is).
    balanced     scale_pos_weight = n_benign / n_malicious, the usual imbalance fix. At 50/50 it is
                 almost exactly 1, so it should change (almost) nothing: that is the point to show.
    prior_10pct  weights the classes as if malware were 10% of the data (scale_pos_weight = 1/9),
                 i.e. a model trained for a realistic base rate. It can move scores and the 0.5
                 operating point, but reweighting cannot add information, so ranking metrics
                 (AUC, TPR at 1% FPR) should barely move.
    """
    all_rows = np.arange(len(y_train))
    n_mal = int(np.sum(y_train == 1))
    balanced = (len(y_train) - n_mal) / n_mal
    prior = deployment_prevalence / (1 - deployment_prevalence)
    return [
        FitSpec("class_weight_none", "lgbm_tuned", all_rows, save_model=True,
                note="no class weights (reference refit on this machine)"),
        FitSpec("class_weight_balanced", "lgbm_tuned", all_rows, lgbm_extra={"scale_pos_weight": balanced},
                save_model=True, note=f"scale_pos_weight = n_benign/n_malicious = {balanced:.6f}"),
        FitSpec("class_weight_prior_10pct", "lgbm_tuned", all_rows, lgbm_extra={"scale_pos_weight": prior},
                save_model=True, note=f"scale_pos_weight = {prior:.4f} (malware weighted as 10% of the data)"),
    ]


# --- Slices -------------------------------------------------------------------------------
def slice_masks(meta: pd.DataFrame, entropy: np.ndarray, size_edges: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    """{(slice family, slice value): row mask} for one test set."""
    masks: dict[tuple[str, str], np.ndarray] = {}
    for week in sorted(meta["week"].unique()):
        masks[("week", f"{int(week):02d}")] = (meta["week"] == week).to_numpy()
    quartile = np.searchsorted(size_edges, meta["file_size"].to_numpy(), side="right")  # 0..3
    for q in range(4):
        masks[("file_size_quartile", f"Q{q + 1}")] = quartile == q
    packed = meta["is_packed"].to_numpy(bool)
    masks[("packer_tag", "packed (any)")] = packed
    masks[("packer_tag", "not packed")] = ~packed
    counts = meta.loc[packed, "packer"].value_counts()
    for name in counts[counts >= MIN_PACKER_N].index:
        masks[("packer_name", str(name))] = (meta["packer"] == name).to_numpy()
    masks[("entropy_proxy", f"entropy > {HIGH_ENTROPY:g}")] = entropy > HIGH_ENTROPY
    masks[("entropy_proxy", f"entropy <= {HIGH_ENTROPY:g}")] = entropy <= HIGH_ENTROPY
    signed = meta["signed"].to_numpy(bool)
    masks[("signed", "signed")] = signed
    masks[("signed", "unsigned")] = ~signed
    family = meta["family"].fillna("").astype(str).str.len().to_numpy() > 0
    masks[("family", "known")] = family
    masks[("family", "unknown")] = ~family
    return masks


def slice_rows(test: str, y: np.ndarray, p: np.ndarray, masks: dict[tuple[str, str], np.ndarray],
               threshold: float, threshold_name: str) -> list[dict]:
    """TPR on malicious and FPR on benign files of every slice, with Wilson CIs and size flags.

    family slices report recall only (benign files have no family), as the brief asks.
    """
    flagged = p >= threshold
    rows = []
    for (family, value), mask in masks.items():
        mal, ben = mask & (y == 1), mask & (y == 0)
        n_mal, n_ben = int(mal.sum()), int(ben.sum())
        tp, fp = int(flagged[mal].sum()), int(flagged[ben].sum())
        row = {"test_set": test, "threshold_name": threshold_name, "threshold": threshold,
               "slice_family": family, "slice": value, "n": int(mask.sum()),
               "n_malicious": n_mal, "n_benign": n_ben,
               "tpr": tp / n_mal if n_mal else np.nan, "tpr_ci_low": wilson_ci(tp, n_mal)[0],
               "tpr_ci_high": wilson_ci(tp, n_mal)[1],
               "fpr": fp / n_ben if n_ben else np.nan, "fpr_ci_low": wilson_ci(fp, n_ben)[0],
               "fpr_ci_high": wilson_ci(fp, n_ben)[1],
               "roc_auc": (operating_metrics(y[mask], p[mask])["roc_auc"] if n_mal and n_ben else np.nan)}
        if family == "family":  # malicious-only metric
            row.update(fpr=np.nan, fpr_ci_low=np.nan, fpr_ci_high=np.nan, roc_auc=np.nan, n_benign=0)
            row["n"] = n_mal
        row["tpr_too_small"] = n_mal < MIN_SLICE_N
        row["fpr_too_small"] = (row["n_benign"] < MIN_SLICE_N)
        rows.append(row)
    return rows


def slice_table(per_test: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]],
                size_edges: np.ndarray, thresholds: dict[str, float]) -> pd.DataFrame:
    """Long table over both test sets and every named threshold."""
    rows = []
    for test, (y, p, meta, entropy) in per_test.items():
        masks = slice_masks(meta, entropy, size_edges)
        for name, thr in thresholds.items():
            rows += slice_rows(test, y, p, masks, thr, name)
    return pd.DataFrame(rows)


def size_quartile_edges(train_sizes: np.ndarray) -> np.ndarray:
    """The 25th, 50th and 75th percentile of file size in TRAIN (never fitted on a test set)."""
    return np.percentile(train_sizes, [25, 50, 75])


def plot_slices(table: pd.DataFrame, threshold_name: str, out_dir=FIGURES) -> str:
    """TPR (left) and benign FPR (right) per slice on both test sets, with Wilson CIs; small slices hatched."""
    part = table[(table["threshold_name"] == threshold_name) & (table["slice_family"] != "week")]
    part = part.assign(label=part["slice_family"] + ": " + part["slice"])
    labels = part["label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(labels) + 1.8), facecolor=tables.SURFACE, sharey=True)
    y_pos = np.arange(len(labels))
    # Zoom the TPR axis to the data (all slices sit near 1), and say so in the panel title.
    tpr_low = float(np.nanmin(part["tpr_ci_low"])) if part["tpr_ci_low"].notna().any() else 0.0
    tpr_left = max(0.0, float(np.floor((tpr_low - 0.02) * 20) / 20))
    zoom = f" (axis from {tpr_left:.2f})" if tpr_left > 0 else ""
    for ax, metric, title in ((axes[0], "tpr", f"Detection rate (TPR) on malicious{zoom}"),
                              (axes[1], "fpr", "False-positive rate on benign")):
        for i, test in enumerate(tables.TEST_SETS):
            rows = part[part["test_set"] == test].set_index("label").reindex(labels)
            val = rows[metric].to_numpy(float)
            err = np.vstack([val - rows[f"{metric}_ci_low"].to_numpy(float), rows[f"{metric}_ci_high"].to_numpy(float) - val])
            small = rows[f"{metric}_too_small"].fillna(True).to_numpy(bool)
            offset = (i - 0.5) * 0.36
            color = tables.SERIES_COLORS[i]
            ax.barh(y_pos + offset, np.nan_to_num(val), height=0.34, color=color, alpha=0.85, label=test,
                    hatch=None, edgecolor="none")
            ax.barh(y_pos[small] + offset, np.nan_to_num(val[small]), height=0.34, color="none",
                    edgecolor=tables.INK_MUTED, hatch="///", linewidth=0)
            ax.errorbar(np.nan_to_num(val), y_pos + offset, xerr=np.nan_to_num(err), fmt="none",
                        ecolor=tables.INK_MUTED, elinewidth=0.8, capsize=1.5)
        tables._style(ax, title)
    axes[0].invert_yaxis()  # first slice on top (the y axis is shared, so invert it once)
    axes[0].set_yticks(y_pos, labels, fontsize=8)
    axes[0].set_xlim(tpr_left, 1.0)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Tuned LightGBM per slice at threshold '{threshold_name}' (hatched = fewer than "
                 f"{MIN_SLICE_N} files of that class; bars = 95% Wilson CI)", color=tables.INK, fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bias_slices{RUN_SUFFIX}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


# --- Base rate ----------------------------------------------------------------------------
def at_prevalence(tpr: float, fpr: float, prevalence: float) -> dict[str, float]:
    """Precision and per-10k-file counts if malware were `prevalence` of the files scanned.

    precision = pi*TPR / (pi*TPR + (1-pi)*FPR). TPR and FPR are per-class rates, so they do not
    depend on the test set's 50/50 mix; precision does, strongly.
    """
    detected = prevalence * tpr * FILES_PER_ALARM_BASE
    false_alarms = (1 - prevalence) * fpr * FILES_PER_ALARM_BASE
    alarms = detected + false_alarms
    return {"precision": detected / alarms if alarms else float("nan"), "alarms_per_10k": alarms,
            "true_alarms_per_10k": detected, "false_alarms_per_10k": false_alarms,
            "missed_per_10k": prevalence * (1 - tpr) * FILES_PER_ALARM_BASE}


def prevalence_table(scored: dict[str, tuple[np.ndarray, np.ndarray]], thresholds: dict[str, float],
                     test_set: str = "test_future") -> pd.DataFrame:
    """One row per (model, threshold, prevalence) on one test set (`test_set` only labels the rows).

    precision_ci_low/high combine the Wilson bounds of TPR and FPR in the worst and best direction,
    a conservative interval (it ignores that the two errors are independent).
    """
    rows = []
    for model, (y, p) in scored.items():
        for name, thr in thresholds.items():
            if not np.isfinite(thr):
                continue
            flagged = p >= thr
            tp, n_mal = int(flagged[y == 1].sum()), int((y == 1).sum())
            fp, n_ben = int(flagged[y == 0].sum()), int((y == 0).sum())
            tpr, fpr = tp / n_mal, fp / n_ben
            (tpr_lo, tpr_hi), (fpr_lo, fpr_hi) = wilson_ci(tp, n_mal), wilson_ci(fp, n_ben)
            for prev in PREVALENCES:
                point = at_prevalence(tpr, fpr, prev)
                rows.append({"model": model, "test_set": test_set, "threshold_name": name, "threshold": thr,
                             "prevalence": prev, "tpr": tpr, "fpr": fpr, **point,
                             "precision_ci_low": at_prevalence(tpr_lo, fpr_hi, prev)["precision"],
                             "precision_ci_high": at_prevalence(tpr_hi, fpr_lo, prev)["precision"],
                             "precision_at_test_mix": tp / (tp + fp) if (tp + fp) else float("nan")})
    return pd.DataFrame(rows)
