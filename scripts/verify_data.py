"""Sanity-check the feature cache and the experiment splits; write the evidence to reports/.

Outputs (generated, never hand-edited):
  reports/data_check[_quick].json        every number below, plus "flags" and "hard_checks"
  reports/tables/class_balance_by_week.csv

Hard checks (exit code 1 if any fails): feature dim and group offsets agree, labels are
only 0/1, no sha256 shared between train / test_random / test_future, no duplicate
sha256 inside a split, test_future strictly later than train, and 20 random cached rows
equal a fresh re-vectorisation of their raw jsonl lines.
"Flags" are things a reader must know about but that are not bugs (e.g. the
detection_ratio label proxy).

Usage: .venv/Scripts/python.exe scripts/verify_data.py   (EMBER_QUICK=1 checks the quick splits)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # project root only; data/raw is never on sys.path

import numpy as np  # noqa: E402
import orjson  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import config  # noqa: E402
from src import data  # noqa: E402

N_ALIGNMENT_ROWS = 20
SCAN_CHUNK_ROWS = 20_000  # ~200 MB of X per chunk while scanning for NaN/inf


# ---------------------------------------------------------------------------
# Cache-level checks
# ---------------------------------------------------------------------------
def check_cache_shape(info: dict[str, Any]) -> dict[str, Any]:
    """Feature dim, group offsets and X/meta row counts must all agree."""
    groups = data.feature_groups()
    dim = int(info["feature_dim"])
    contiguous = all(groups[i][2] == groups[i + 1][1] for i in range(len(groups) - 1))
    shapes = {}
    for official in data.OFFICIAL_SPLITS:
        X = data.open_cache_X(official)
        shapes[official] = {"X_shape": list(X.shape), "meta_rows": len(data.load_cache_meta(official))}
    return {
        "feature_dim": dim,
        "feature_groups": [{"name": n, "start": s, "end": e, "width": e - s} for n, s, e in groups],
        "groups_sum_to_dim": groups[0][1] == 0 and contiguous and groups[-1][2] == dim,
        "shapes": shapes,
        "rows_match_meta": all(v["X_shape"][0] == v["meta_rows"] and v["X_shape"][1] == dim for v in shapes.values()),
    }


def check_download_counts(info: dict[str, Any]) -> dict[str, Any]:
    """Lines read per file must equal the row counts the downloader recorded."""
    done = json.loads((config.DATA_RAW / "DOWNLOAD_DONE").read_text(encoding="utf-8"))
    expected = {name: n for z in done["zips"].values() for name, n in z["rows_per_file"].items()}
    mismatched = [p["source_file"] for p in info["per_file"] if expected.get(p["source_file"]) != p["lines"]]
    return {"files_checked": len(info["per_file"]), "line_count_mismatches": mismatched}


def class_balance_by_week(metas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (official split, week): counts, malicious share and date range."""
    frames = []
    for official, meta in metas.items():
        g = meta.groupby("week")
        frames.append(pd.DataFrame({
            "official_split": official,
            "week_id": g["week_id"].min(),
            "n": g.size(),
            "n_benign": g["label"].apply(lambda s: int((s == 0).sum())),
            "n_malicious": g["label"].apply(lambda s: int((s == 1).sum())),
            "pos_rate": g["label"].mean().round(4),
            "date_min": g["first_submission_date"].min(),
            "date_max": g["first_submission_date"].max(),
        }).reset_index())
    return pd.concat(frames, ignore_index=True)


def check_dates(metas: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Date ranges, and whether the derived `week` agrees with EMBER's own week_id."""
    out: dict[str, Any] = {}
    for official, meta in metas.items():
        out[official] = {
            "date_min": str(meta["first_submission_date"].min()),
            "date_max": str(meta["first_submission_date"].max()),
            "weeks": [int(meta["week"].min()), int(meta["week"].max())],
            "n_weeks": int(meta["week"].nunique()),
            "week_ne_week_id": int((meta["week"] != meta["week_id"]).sum()),
        }
    return out


def check_detection_ratio(metas: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """How detection_ratio is formatted and how well it alone predicts the label."""
    out: dict[str, Any] = {}
    for official, meta in metas.items():
        raw = meta["detection_ratio"].astype(str)
        ok = raw.str.fullmatch(r"\d+/\d+")
        ratio = meta["detection_ratio_float"]
        valid = ratio.notna()
        y, pos = meta["label"], meta["detection_positives"]
        out[official] = {
            "format": "string 'positives/total', e.g. " + ", ".join(raw.head(3)),
            "share_matching_k_over_n": float(ok.mean()),
            "unparseable": int((~valid).sum()),
            "denominators": {str(int(k)): int(v) for k, v in meta["detection_total"].value_counts().head(8).items()},
            "roc_auc_detection_ratio_float_vs_label": float(roc_auc_score(y[valid], ratio[valid])),
            "benign_with_any_detection": int(((y == 0) & (pos > 0)).sum()),
            "malicious_with_zero_detections": int(((y == 1) & (pos == 0)).sum()),
            "min_detections_malicious": float(pos[y == 1].min()),
            "max_detections_benign": float(pos[y == 0].max()),
        }
    return out


def scan_nonfinite() -> dict[str, Any]:
    """Count NaN/inf in the whole cache, and columns that are constant (useless to any model)."""
    out = {}
    for official in data.OFFICIAL_SPLITS:
        X = data.open_cache_X(official)
        nonfinite, col_min, col_max = 0, np.full(X.shape[1], np.inf), np.full(X.shape[1], -np.inf)
        for start in range(0, len(X), SCAN_CHUNK_ROWS):
            chunk = np.asarray(X[start:start + SCAN_CHUNK_ROWS])
            nonfinite += int((~np.isfinite(chunk)).sum())
            col_min = np.minimum(col_min, np.nanmin(chunk, axis=0))
            col_max = np.maximum(col_max, np.nanmax(chunk, axis=0))
        out[official] = {"nonfinite_values": nonfinite, "constant_columns": int((col_min == col_max).sum())}
    return out


def check_alignment(metas: dict[str, pd.DataFrame], n_rows: int = N_ALIGNMENT_ROWS) -> dict[str, Any]:
    """Re-read random raw lines (by stored byte offset), re-vectorise them and compare to cached X."""
    rng = np.random.default_rng(config.SEED)
    extractor = data.get_extractor()
    picks = [(o, int(i)) for o in metas for i in rng.choice(len(metas[o]), n_rows // len(metas), replace=False)]
    failures = []
    for official, i in picks:
        meta_row = metas[official].iloc[i]
        with (data.raw_win64_dir() / meta_row["source_file"]).open("rb") as f:
            f.seek(int(meta_row["source_offset"]))
            raw = orjson.loads(f.readline())
        cached = np.asarray(data.open_cache_X(official)[i])
        fresh = extractor.process_raw_features(raw)
        same = (
            raw["sha256"] == meta_row["sha256"]
            and int(raw["label"]) == int(meta_row["label"])
            and np.array_equal(cached, fresh, equal_nan=True)
        )
        if not same:
            failures.append({"official_split": official, "row": i, "sha256": meta_row["sha256"]})
    return {"rows_checked": len(picks), "mismatches": failures}


# ---------------------------------------------------------------------------
# Split-level checks
# ---------------------------------------------------------------------------
def check_splits(metas: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Sizes, balance, sha256 overlap, duplicates, temporal order, tuning subset and folds."""
    split_meta = {n: metas[data.SPLIT_SOURCE[n]].iloc[data.split_rows(n)] for n in data.SPLIT_NAMES}
    shas = {n: set(m["sha256"]) for n, m in split_meta.items()}
    names = list(data.SPLIT_NAMES)
    overlap = {f"{a}&{b}": len(shas[a] & shas[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    tune = data.tune_subset_indices()
    n_train = len(split_meta["train"])
    folds_ok = all(
        len(np.intersect1d(fit, val)) == 0 and len(fit) + len(val) == size
        for subset, size in ((False, n_train), (True, len(tune)))
        for fit, val in data.get_cv_folds(subset=subset)
    )
    return {
        "sizes": {n: len(m) for n, m in split_meta.items()},
        "pos_rate": {n: round(float(m["label"].mean()), 4) for n, m in split_meta.items()},
        "n_weeks": {n: int(m["week"].nunique()) for n, m in split_meta.items()},
        "sha256_overlap": overlap,
        "duplicate_sha256_within": {n: int(m["sha256"].duplicated().sum()) for n, m in split_meta.items()},
        "train_test_random_row_overlap": int(len(np.intersect1d(data.split_rows("train"), data.split_rows("test_random")))),
        "official_train_test_sha256_overlap": len(set(metas["train"]["sha256"]) & set(metas["test"]["sha256"])),
        "test_future_min_date": str(split_meta["test_future"]["first_submission_date"].min()),
        "train_max_date": str(split_meta["train"]["first_submission_date"].max()),
        "test_future_strictly_later": bool(
            split_meta["test_future"]["first_submission_date"].min() > split_meta["train"]["first_submission_date"].max()
        ),
        "tune_subset": {"n": int(len(tune)), "unique": bool(len(np.unique(tune)) == len(tune)),
                        "in_range": bool(tune.min() >= 0 and tune.max() < n_train)},
        "cv_folds_partition_ok": bool(folds_ok),
        "cv_fold_sizes": [len(v) for _, v in data.get_cv_folds()],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def hard_checks(report: dict[str, Any]) -> dict[str, bool]:
    s = report["splits"]
    return {
        "groups_sum_to_dim": report["cache"]["groups_sum_to_dim"],
        "rows_match_meta": report["cache"]["rows_match_meta"],
        "labels_only_0_1": report["labels_only_0_1"],
        "no_line_count_mismatch": not report["download"]["line_count_mismatches"],
        "no_sha256_overlap_across_splits": all(v == 0 for v in s["sha256_overlap"].values()),
        "no_duplicate_sha256_within_splits": all(v == 0 for v in s["duplicate_sha256_within"].values()),
        "test_future_strictly_later": s["test_future_strictly_later"],
        "tune_subset_valid": s["tune_subset"]["unique"] and s["tune_subset"]["in_range"],
        "cv_folds_partition_ok": s["cv_folds_partition_ok"],
        "alignment_ok": not report["alignment"]["mismatches"],
    }


def flags(report: dict[str, Any]) -> list[str]:
    """Things a reader must know about. Not failures: some are the point of the study."""
    out = []
    for official, sp in report["cache_info"]["splits"].items():
        if sp["duplicates_in_file"]:
            out.append(f"{official}: {sp['duplicates_in_file']} of {sp['lines']} raw lines were in-file duplicate "
                       "sha256 (each weekly file repeats its 10k samples); dropped, first occurrence kept.")
        if sp["duplicates_across_files"]:
            out.append(f"{official}: {sp['duplicates_across_files']} sha256 appeared in two different weeks; "
                       "only the earliest week's row is kept.")
        if sp.get("duplicates_of_earlier_split"):
            out.append(f"{official}: {sp['duplicates_of_earlier_split']} sha256 already present in an earlier "
                       "official split; dropped here so train and test never share a sample.")
        if sp["dropped_bad_label"]:
            out.append(f"{official}: {sp['dropped_bad_label']} rows with label not in {{0,1}} dropped.")
    for official, d in report["detection_ratio"].items():
        if d["roc_auc_detection_ratio_float_vs_label"] > 0.99:
            out.append(f"{official}: detection_ratio_float alone gives ROC-AUC "
                       f"{d['roc_auc_detection_ratio_float_vs_label']:.4f} - a near-perfect label proxy. "
                       "Expected (labels are derived from AV detections); it is kept only in meta, never in X.")
    for official, d in report["dates"].items():
        if d["week_ne_week_id"]:
            out.append(f"{official}: {d['week_ne_week_id']} rows whose derived week != official week_id.")
    for official, d in report["nonfinite"].items():
        if d["nonfinite_values"]:
            out.append(f"{official}: {d['nonfinite_values']} NaN/inf values in X.")
        out.append(f"{official}: {d['constant_columns']} constant feature columns (harmless, carry no signal).")
    out.append("Classes are ~50/50 in every week by EMBER2024's construction, not real-world prevalence.")
    out.append("X contains time proxies: header coff.timestamp (col 696, compile time) and authenticode "
               "signing time; they can encode 'when' rather than 'what' under a random split.")
    return out


def main() -> None:
    info = data.cache_info()
    metas = {o: data.load_cache_meta(o) for o in data.OFFICIAL_SPLITS}
    report: dict[str, Any] = {
        "quick": config.QUICK,
        "seed": config.SEED,
        "cache_info": {k: info[k] for k in ("thrember_version", "thrember_commit", "feature_dim",
                                            "week_anchor_utc", "splits", "built_utc")},
        "cache": check_cache_shape(info),
        "download": check_download_counts(info),
        "labels_only_0_1": all(set(m["label"].unique()) <= {0, 1} for m in metas.values()),
        "dates": check_dates(metas),
        "detection_ratio": check_detection_ratio(metas),
        "nonfinite": scan_nonfinite(),
        "alignment": check_alignment(metas),
        "splits": check_splits(metas),
    }
    report["splits_info"] = json.loads(config.split_file("splits_info", ".json").read_text(encoding="utf-8"))
    report["hard_checks"] = hard_checks(report)
    report["ok"] = all(report["hard_checks"].values())
    report["flags"] = flags(report)

    config.TABLES.mkdir(parents=True, exist_ok=True)
    balance = class_balance_by_week(metas)
    balance.to_csv(config.TABLES / "class_balance_by_week.csv", index=False)
    out_path = config.REPORTS / f"data_check{config.RUN_SUFFIX}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print_summary(report, balance)
    print(f"\nWrote {out_path} and {config.TABLES / 'class_balance_by_week.csv'}")
    sys.exit(0 if report["ok"] else 1)


def print_summary(report: dict[str, Any], balance: pd.DataFrame) -> None:
    """Human-readable console version of data_check.json."""
    ci = report["cache_info"]
    print(f"thrember {ci['thrember_version']} @ {ci['thrember_commit']}, feature dim {ci['feature_dim']}")
    for g in report["cache"]["feature_groups"]:
        print(f"  {g['name']:<16} [{g['start']:>4}, {g['end']:>4})  width {g['width']}")
    print(f"groups sum to dim: {report['cache']['groups_sum_to_dim']}")
    print("\nRows per official split:", {o: s["rows"] for o, s in ci["splits"].items()})
    print("Dates:", json.dumps(report["dates"], indent=1))
    with pd.option_context("display.max_rows", 100, "display.width", 160):
        print("\nClass balance per week:\n", balance.to_string(index=False))
    print("\ndetection_ratio:", json.dumps(report["detection_ratio"], indent=1))
    print("NaN/inf:", report["nonfinite"])
    print("Alignment:", report["alignment"])
    print("Splits:", json.dumps(report["splits"], indent=1))
    print("\nHard checks:", json.dumps(report["hard_checks"], indent=1))
    print("Flags:\n  - " + "\n  - ".join(report["flags"]))
    print(f"\nOK = {report['ok']}")


if __name__ == "__main__":
    main()
