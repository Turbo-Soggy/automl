"""Step 4 part C: SHAP explanations of the tuned LightGBM and cross-tool importance agreement.

  * readable_names(): "header__0" -> "header: coff.timestamp", read off thrember's own
    feature classes (hashed buckets keep a bucket number, since a hash has no name).
  * TreeSHAP values come from the libraries' built-in TreeSHAP (LightGBM pred_contrib,
    XGBoost pred_contribs). That is the same algorithm shap.TreeExplainer runs, but it lets
    us cap the threads; tests/test_explain.py checks the two agree.
  * SHAP values are in log-odds (the models' raw margin), so they add up exactly to the
    model's logit for each file: base value + sum(SHAP) = logit(P(malicious)).
  * Family importance: the 12 EMBER feature families from src.data.feature_groups().

Explaining a fixed model on test_future is not tuning, so using test rows here is fine; no
choice made in this module feeds back into a model.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from config import SEED

TOP_K: tuple[int, ...] = (5, 20)   # feature-level overlap sizes for the cross-tool table


# --- Feature names ------------------------------------------------------------------------

def column_names(groups: list[tuple[str, int, int]]) -> list[str]:
    """"{family}__{i}" names (i counts inside the family): the names the AutoML tools saw."""
    return [f"{family}__{i}" for family, start, end in groups for i in range(end - start)]


def family_of(name: str) -> str:
    """"header__0" -> "header"."""
    return name.rsplit("__", 1)[0]


def _hashed(prefix: str, n: int) -> list[str]:
    return [f"{prefix}[hash {i}]" for i in range(n)]


def _family_labels(fe: Any) -> list[str] | None:
    """Readable labels for one thrember feature class, in its process_raw_features() order.

    Returns None for an unknown class, so a thrember update degrades to plain column names
    instead of silently mislabelling columns.
    """
    name = fe.name
    if name == "general":
        return ["size", "entropy", "is_pe"] + [f"start_byte_{i}" for i in range(4)]
    if name == "histogram":
        return [f"byte_0x{i:02x}_freq" for i in range(256)]
    if name == "byteentropy":
        return [f"entropy_bin{h}/byte_hi_nibble{b}" for h in range(16) for b in range(16)]
    if name == "strings":
        printable = [f"char_{chr(0x20 + i)!r}_freq" for i in range(96)]
        regexes = sorted(fe.regex_idxs, key=fe.regex_idxs.get)
        return (["numstrings", "avlength", "printables"] + printable + ["entropy"]
                + [f"count[{r}]" for r in regexes])
    if name == "header":
        scalars = ["coff.timestamp", "coff.number_of_sections", "coff.number_of_symbols",
                   "coff.sizeof_optional_header", "coff.pointer_to_symbol_table", "coff.machine",
                   "optional.subsystem", "optional.major_image_version", "optional.minor_image_version",
                   "optional.major_linker_version", "optional.minor_linker_version",
                   "optional.major_os_version", "optional.minor_os_version",
                   "optional.major_subsystem_version", "optional.minor_subsystem_version",
                   "optional.sizeof_code", "optional.sizeof_headers", "optional.sizeof_image",
                   "optional.sizeof_initialized_data", "optional.sizeof_uninitialized_data",
                   "optional.sizeof_stack_reserve", "optional.sizeof_stack_commit",
                   "optional.sizeof_heap_reserve", "optional.sizeof_heap_commit",
                   "optional.address_of_entrypoint", "optional.base_of_code", "optional.image_base",
                   "optional.section_alignment", "optional.checksum", "optional.number_of_rvas_and_sizes"]
        return (scalars + [f"coff.char.{c}" for c in fe._image_characteristics]
                + [f"optional.dll_char.{c}" for c in fe._dll_characteristics]
                + [f"dos.{m}" for m in fe._dos_members])
    if name == "section":
        general = ["n_sections", "n_zero_size", "n_empty_name", "n_rx", "n_w", "max_entropy",
                   "min_entropy", "max_size_ratio", "min_size_ratio", "max_vsize_ratio", "min_vsize_ratio"]
        return (general + _hashed("size", 50) + _hashed("vsize", 50) + _hashed("entropy", 50)
                + _hashed("characteristics", 50) + _hashed("entry_name", 10)
                + ["overlay.size", "overlay.size_ratio", "overlay.entropy"])
    if name == "imports":
        return ["n_imported_functions", "n_libraries"] + _hashed("library", 256) + _hashed("function", 1024)
    if name == "exports":
        # thrember stores len(hashed vector) = 128 when there are exports, 0 otherwise.
        return ["has_exports(0/128)"] + _hashed("export", 128)
    if name == "datadirectories":
        names = ["EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY", "BASERELOC", "DEBUG",
                 "COPYRIGHT", "GLOBALPTR", "TLS", "LOAD_CONFIG", "BOUND_IMPORT", "IAT",
                 "DELAY_IMPORT", "COM_DESCRIPTOR", "RESERVED"]
        return ([f"{d}.{part}" for d in names for part in ("size", "virtual_address")]
                + ["has_relocs", "has_dynamic_relocs"])
    if name == "richheader":
        return ["n_pairs"] + _hashed("pair", 32)
    if name == "authenticode":
        return ["num_certs", "self_signed", "empty_program_name", "no_countersigner", "parse_error",
                "chain_max_depth", "latest_signing_time", "signing_time_diff"]
    if name == "pefilewarnings":
        by_id = sorted(fe.warning_ids, key=fe.warning_ids.get)
        return [f"warning[{w[:40]}]" for w in by_id] + ["n_warnings"]
    return None


def readable_names(extractor: Any) -> list[str]:
    """"family: label" for every column, in matrix order; falls back to "{family}__{i}"."""
    out: list[str] = []
    for fe in extractor.features:
        labels = _family_labels(fe)
        if labels is None or len(labels) != fe.dim:
            labels = [str(i) for i in range(fe.dim)]   # unknown layout: keep the index only
        out += [f"{fe.name}: {label}" for label in labels]
    return out


# --- Sampling and TreeSHAP ------------------------------------------------------------------

def stratified_sample(meta: pd.DataFrame, n: int, seed: int = SEED) -> np.ndarray:
    """Sorted positions of a label x week stratified sample of n rows (all rows if n >= len)."""
    if n >= len(meta):
        return np.arange(len(meta))
    strata = meta["label"].to_numpy().astype(np.int64) * 1000 + meta["week"].to_numpy().astype(np.int64)
    picked, _ = train_test_split(np.arange(len(meta)), train_size=n, stratify=strata, random_state=seed)
    return np.sort(picked)


def lgbm_shap(booster: Any, X: np.ndarray, n_threads: int) -> tuple[np.ndarray, float]:
    """(SHAP values [n, D], base value) in log-odds, from LightGBM's built-in TreeSHAP."""
    contrib = booster.predict(X, pred_contrib=True, num_threads=n_threads)
    return contrib[:, :-1], float(contrib[0, -1])   # last column is the expected value


def xgb_shap(model: Any, X: np.ndarray, n_threads: int) -> tuple[np.ndarray, float]:
    """(SHAP values, base value) in log-odds for an XGBClassifier, via XGBoost's TreeSHAP."""
    import xgboost as xgb

    booster = model.get_booster()
    booster.set_param({"nthread": n_threads})
    dmat = xgb.DMatrix(X, feature_names=booster.feature_names)
    contrib = booster.predict(dmat, pred_contribs=True)
    return contrib[:, :-1], float(contrib[0, -1])


def expand_to_all(values: np.ndarray, used: list[str], all_names: list[str]) -> np.ndarray:
    """Place SHAP columns of a model fitted on a column subset into the full D-column layout."""
    pos = {n: i for i, n in enumerate(all_names)}
    out = np.zeros((values.shape[0], len(all_names)), dtype=float)
    out[:, [pos[n] for n in used]] = values
    return out


# --- Importance tables ----------------------------------------------------------------------

def feature_importance(values: np.ndarray, names: list[str]) -> pd.Series:
    """Mean |SHAP| per column, as a Series indexed by column name."""
    return pd.Series(np.abs(values).mean(axis=0), index=names)


def family_table(values: np.ndarray, groups: list[tuple[str, int, int]], labels: list[str]) -> pd.DataFrame:
    """Per family: summed mean |SHAP| (comparable with summed gain), grouped |SHAP|, share and top feature.

    grouped_mean_abs_shap = mean over files of |sum of the family's SHAP values|: SHAP values
    are additive, so this is the importance of the family as one block. It is never larger than
    the summed version, because members can push in opposite directions and cancel.
    """
    mean_abs = np.abs(values).mean(axis=0)
    rows = []
    for family, start, end in groups:
        block = values[:, start:end]
        top = start + int(np.argmax(mean_abs[start:end]))
        rows.append({"family": family, "n_features": end - start,
                     "sum_mean_abs_shap": float(mean_abs[start:end].sum()),
                     "grouped_mean_abs_shap": float(np.abs(block.sum(axis=1)).mean()),
                     "top_feature": labels[top], "top_feature_column": top,
                     "top_feature_mean_abs_shap": float(mean_abs[top])})
    table = pd.DataFrame(rows)
    table["share"] = table["sum_mean_abs_shap"] / table["sum_mean_abs_shap"].sum()
    table["mean_abs_shap_per_feature"] = table["sum_mean_abs_shap"] / table["n_features"]
    table = table.sort_values("sum_mean_abs_shap", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def rank_of(importance: pd.Series, name: str) -> int:
    """1-based rank of one column by importance (ties share the best rank); D+1-ish if absent."""
    ranks = importance.rank(ascending=False, method="min")
    return int(ranks.get(name, len(importance) + 1))


def importance_from_csv(frame: pd.DataFrame, value_col: str, names: list[str]) -> pd.Series:
    """A tool's saved per-feature importances on the full column list (absent or NaN -> 0)."""
    s = frame.set_index("feature")[value_col].astype(float)
    s = s[~s.index.duplicated()]
    return s.reindex(names).fillna(0.0)


def to_families(importance: pd.Series, families: list[str]) -> pd.Series:
    """Sum per-feature importances into the 12 families (in feature_groups() order)."""
    return importance.groupby(importance.index.map(family_of)).sum().reindex(families).fillna(0.0)


def top_k(importance: pd.Series, k: int) -> list[str]:
    """Names of the k most important columns (stable order on ties)."""
    return importance.sort_values(ascending=False, kind="mergesort").index[:k].tolist()


def agreement_table(feature_imp: dict[str, pd.Series], family_imp: dict[str, pd.Series],
                    ks: tuple[int, ...] = TOP_K) -> pd.DataFrame:
    """One row per tool pair: top-k feature overlap and family-level Spearman correlation."""
    rows = []
    for a, b in combinations(feature_imp, 2):
        row: dict[str, Any] = {"tool_a": a, "tool_b": b}
        for k in ks:
            shared = set(top_k(feature_imp[a], k)) & set(top_k(feature_imp[b], k))
            row[f"top{k}_overlap"] = len(shared)
            row[f"top{k}_overlap_share"] = len(shared) / k
        rho, p = spearmanr(family_imp[a].to_numpy(), family_imp[b].to_numpy())
        row["family_spearman"] = float(rho)
        row["family_spearman_p"] = float(p)
        top_a, top_b = family_imp[a].idxmax(), family_imp[b].idxmax()
        row["top_family_a"], row["top_family_b"] = top_a, top_b
        rows.append(row)
    return pd.DataFrame(rows)


def pick_cases(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, int | None]:
    """Positions of the three files to explain locally (None when a sample has no such file).

    tp: the most confident correct malicious; fp: the benign file scored most malicious;
    fn: the malicious file scored most benign. The extreme cases show the clearest reasons.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)

    def extreme(mask: np.ndarray, highest: bool) -> int | None:
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return None
        return int(idx[np.argmax(p[idx])] if highest else idx[np.argmin(p[idx])])

    return {"tp": extreme((y == 1) & (p >= threshold), True),
            "fp": extreme((y == 0) & (p >= threshold), True),
            "fn": extreme((y == 1) & (p < threshold), False)}


def top_contributions(shap_row: np.ndarray, x_row: np.ndarray, labels: list[str], n: int = 5) -> pd.DataFrame:
    """The n largest |SHAP| contributions of one file, with the feature value."""
    order = np.argsort(-np.abs(shap_row), kind="mergesort")[:n]
    return pd.DataFrame({"column": order, "feature": [labels[i] for i in order],
                         "value": x_row[order], "shap": shap_row[order]})
