"""Statistics for the poster: results/*.jsonl -> analysis/*.csv (step 3 of the pipeline).

Only writes CSV files; make_figures.py reads them. Deterministic (seed 42, 2,000 bootstrap
resamples). File names say what each table is:

  pre-registered (README):  hypothesis_summary.csv (one row per hypothesis), primary_mcnemar.csv,
                            h1b_equivalence.csv, secondary_holm.csv, gee_coefficients.csv, auroc.csv,
                            confidence_q4_fp16.csv, validity_kappa.csv
  descriptive / planned:    overview.csv, descriptives.csv, floor_social.csv, calibration.csv,
                            confidence_bins.csv, confidence_curve.csv, descriptive_size_comparison.csv, table1.csv
  robustness (design):      robustness_controls.csv, robustness_determinism.csv
  exploratory (not pre-registered): exploratory_offceiling_primary.csv, exploratory_by_source.csv,
                            exploratory_precision_by_condition.csv

Harmful flip rate (HFR) = harmful / items correct in turn 1 whose turn-2 letter is valid; invalid
turn-2 replies are counted separately (column invalid_turn2), never as holds or flips.

    python scripts/analyze.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_BOOT = 2000
CONDITIONS = ["reask", "speaker_free", "user", "expert"]
PRECISIONS = ["q4_K_M", "q8_0", "fp16"]
FAMILIES = ["llama3.2-3b", "qwen2.5-3b", "phi4-mini-3.8b", "qwen2.5-7b", "llama3.1-8b", "phi4-14b"]
TOST_MARGIN = 0.03
ECE_BINS = 15
GEE_MAXITER = 200        # near-ceiling families need more than the default 60 iterations
CONF_BIN_EDGES = [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0000001]   # turn-1 confidence bins (figure data)


# ---------------------------------------------------------------- loading

def run_of(path: Path) -> str:
    """'m__main.jsonl' -> 'main'; 'm__reversed.jsonl' -> 'reversed'; 'm__main__det1.jsonl' -> 'det1'."""
    parts = path.stem.split("__")
    return parts[2] if len(parts) > 2 else parts[1]


def load_results(results_dir: Path) -> pd.DataFrame:
    """All result files except pilots; the last error-free line per (model, run, item, condition)."""
    frames = []
    for f in sorted(results_dir.glob("*.jsonl")):
        if "__pilot" in f.name:
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            d = pd.DataFrame(rows)
            d["run"] = run_of(f)
            frames.append(d)
    if not frames:
        raise SystemExit(f"no result files in {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["error"].isna()]
    df = df.drop_duplicates(["model", "run", "item_id", "condition"], keep="last")
    for col in ["correct_0", "harmful", "went_to_x", "beneficial", "flipped"]:
        df[col] = df[col].astype("boolean")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- small statistics

def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def ece(conf, correct, bins: int = ECE_BINS) -> float:
    """Expected calibration error with equal-width bins (Guo et al., 2017)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def _boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_ratio_ci(num, den, level=0.95, n_boot=N_BOOT, seed=SEED) -> tuple[float, float]:
    """CI of sum(num)/sum(den), resampling clusters (items). num/den: one value per cluster."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0 or den.sum() == 0:
        return (np.nan, np.nan)
    idx = _boot_idx(len(num), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = num[idx].sum(1) / den[idx].sum(1)
    a = (1 - level) / 2
    return tuple(np.nanpercentile(stats, [100 * a, 100 * (1 - a)]))


def boot_diff_ci(x_sum, y_sum, n, level=0.95, n_boot=N_BOOT, seed=SEED) -> tuple[float, float]:
    """Paired CI of sum(x)/sum(n) - sum(y)/sum(n), resampling clusters (items)."""
    x_sum, y_sum, n = (np.asarray(v, float) for v in (x_sum, y_sum, n))
    if len(n) == 0:
        return (np.nan, np.nan)
    idx = _boot_idx(len(n), n_boot, seed)
    denom = n[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = (x_sum[idx].sum(1) - y_sum[idx].sum(1)) / denom
    a = (1 - level) / 2
    return tuple(np.nanpercentile(stats, [100 * a, 100 * (1 - a)]))


def boot_auroc_ci(y, score, clusters, n_boot=N_BOOT, seed=SEED) -> tuple[float, float]:
    y, score, clusters = np.asarray(y), np.asarray(score, float), np.asarray(clusters)
    uniq = np.unique(clusters)
    pos = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        rows = np.concatenate([pos[c] for c in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[rows])) == 2:
            stats.append(roc_auc_score(y[rows], score[rows]))
    return tuple(np.percentile(stats, [2.5, 97.5])) if stats else (np.nan, np.nan)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts."""
    return float(mcnemar([[0, b], [c, 0]], exact=True).pvalue) if b + c else 1.0


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


# ---------------------------------------------------------------- building blocks

def main_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["run"] == "main") & (df["variant"] == "main")]


def turn1(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model x run x item (turn 1 is shared by the four conditions)."""
    return df.drop_duplicates(["model", "run", "item_id"])


def paired(df: pd.DataFrame, condition: str, prec_a: str, prec_b: str) -> pd.DataFrame:
    """Items correct in turn 1 at both precisions of a family, with a valid turn-2 letter at both.

    Returns one row per family x item with harmful_a / harmful_b (and c0_a / c0_b).
    """
    d = df[(df["condition"] == condition) & df["precision"].isin([prec_a, prec_b])]
    d = d[(d["correct_0"] == True) & d["harmful"].notna()]  # noqa: E712
    w = d.pivot_table(index=["family", "item_id"], columns="precision",
                      values=["harmful", "c0"], aggfunc="first")
    w = w.dropna(subset=[("harmful", prec_a), ("harmful", prec_b)])
    out = pd.DataFrame({
        "family": w.index.get_level_values(0), "item_id": w.index.get_level_values(1),
        "harmful_a": w[("harmful", prec_a)].astype(bool).to_numpy(),
        "harmful_b": w[("harmful", prec_b)].astype(bool).to_numpy(),
        "c0_a": w[("c0", prec_a)].to_numpy(dtype=float), "c0_b": w[("c0", prec_b)].to_numpy(dtype=float),
    })
    return out.reset_index(drop=True)


def compare(pairs: pd.DataFrame, level=0.95) -> dict:
    """McNemar, rates, Cohen's h and a clustered bootstrap CI of the difference (a - b)."""
    a, b = pairs["harmful_a"].to_numpy(), pairs["harmful_b"].to_numpy()
    n = len(pairs)
    both, only_a, only_b = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
    per_item = pairs.groupby("item_id").agg(x=("harmful_a", "sum"), y=("harmful_b", "sum"), n=("harmful_a", "size"))
    lo, hi = boot_diff_ci(per_item["x"], per_item["y"], per_item["n"], level=level)
    ra, rb = (a.mean(), b.mean()) if n else (np.nan, np.nan)
    return {"n_pairs": n, "n_items": pairs["item_id"].nunique(), "both_harmful": both,
            "only_a": only_a, "only_b": only_b, "neither": n - both - only_a - only_b,
            "hfr_a": ra, "hfr_b": rb, "diff_a_minus_b": ra - rb, "ci_lo": lo, "ci_hi": hi,
            "ci_level": level, "cohens_h": cohens_h(ra, rb) if n else np.nan,
            "p_mcnemar_exact": mcnemar_p(only_a, only_b)}


# ---------------------------------------------------------------- tables

def descriptives(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, prec, cond), g in main.groupby(["family", "precision", "condition"]):
        valid0 = g[g["a0"].notna()]
        corr = valid0[valid0["correct_0"] == True]  # noqa: E712
        corr_valid = corr[corr["harmful"].notna()]
        wrong_valid = valid0[(valid0["correct_0"] == False) & valid0["a1"].notna()]  # noqa: E712
        h = corr_valid["harmful"].astype(bool)
        lo, hi = boot_ratio_ci(h.astype(float), np.ones(len(h)))
        rows.append({
            "family": fam, "precision": prec, "condition": cond, "n_items": len(g),
            "turn1_valid": len(valid0), "n_correct0": len(corr), "n_correct0_valid_turn2": len(corr_valid),
            "invalid_turn2": int(valid0["a1"].isna().sum()), "harmful": int(h.sum()),
            "hfr": h.mean() if len(h) else np.nan, "hfr_ci_lo": lo, "hfr_ci_hi": hi,
            "capitulation_rate": corr_valid["went_to_x"].astype(bool).mean() if len(corr_valid) else np.nan,
            "beneficial_rate": wrong_valid["beneficial"].astype(bool).mean() if len(wrong_valid) else np.nan,
            "median_answer_mass_0": valid0["answer_mass_0"].median(),
        })
    return pd.DataFrame(rows)


def primary_and_h1b(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = {}
    for name, (pa, pb), level in (("primary", ("q4_K_M", "fp16"), 0.95), ("h1b", ("q8_0", "fp16"), 0.90)):
        pairs = paired(main, "user", pa, pb)
        rows = [{"scope": "pooled over six families", "role": "pre-registered",
                 "precision_a": pa, "precision_b": pb, "condition": "user", **compare(pairs, level)}]
        for fam in FAMILIES:
            sub = pairs[pairs["family"] == fam]
            rows.append({"scope": fam, "role": "per family (descriptive)", "precision_a": pa,
                         "precision_b": pb, "condition": "user", **compare(sub, level)})
        t = pd.DataFrame(rows)
        if name == "primary":
            t["supported"] = (t["p_mcnemar_exact"] < 0.05) & (t["diff_a_minus_b"] > 0)
        else:
            t["margin"] = TOST_MARGIN
            t["equivalent"] = (t["ci_lo"] > -TOST_MARGIN) & (t["ci_hi"] < TOST_MARGIN)
        out[name] = t
    return out["primary"], out["h1b"]


def gee_models(main: pd.DataFrame) -> tuple[pd.DataFrame, float, list[str]]:
    """GEE logistic, item clusters, exchangeable: with and without log c0 (same sample)."""
    d = main[(main["correct_0"] == True) & main["harmful"].notna() & main["c0"].notna() & (main["c0"] > 0)].copy()  # noqa: E712
    d["harmful"] = d["harmful"].astype(int)
    d["log_c0"] = np.log(d["c0"].astype(float))
    base = "harmful ~ C(precision, Treatment('fp16')) * C(condition, Treatment('reask')) + C(family)"
    rows, notes, p_inter = [], [], np.nan
    for label, formula in (("without_log_c0", base), ("with_log_c0", base + " + log_c0")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = smf.gee(formula, groups="item_id", data=d, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=GEE_MAXITER)
        notes += [f"{label}: {w.message}" for w in caught]
        converged = not any("converg" in str(w.message) for w in caught)
        for term in res.params.index:
            rows.append({"model": label, "term": term, "coef": res.params[term], "se": res.bse[term],
                         "p": res.pvalues[term], "n_obs": int(res.nobs), "n_items": d["item_id"].nunique(),
                         "converged": converged})
        if label == "without_log_c0":
            inter = [i for i, t in enumerate(res.params.index) if ":" in t]
            R = np.zeros((len(inter), len(res.params)))
            R[range(len(inter)), inter] = 1
            p_inter = float(np.asarray(res.wald_test(R, scalar=True).pvalue))
    return pd.DataFrame(rows), p_inter, notes


def secondary(main: pd.DataFrame, p_interaction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    # H2a: speaker_free vs reask per configuration (same items, turn 1 correct, valid turn 2 in both)
    for (fam, prec), g in main.groupby(["family", "precision"]):
        g = g[(g["correct_0"] == True) & g["harmful"].notna()]  # noqa: E712
        w = g.pivot_table(index="item_id", columns="condition", values="harmful", aggfunc="first")
        w = w.dropna(subset=["speaker_free", "reask"])
        sf, rq = w["speaker_free"].astype(bool), w["reask"].astype(bool)
        b, c = int((sf & ~rq).sum()), int((~sf & rq).sum())
        rows.append({"hypothesis": "H2a", "test": "McNemar speaker_free vs reask", "family": fam,
                     "precision": prec, "n": len(w), "estimate": sf.mean() - rq.mean(),
                     "detail": f"sf only {b}, reask only {c}", "p_raw": mcnemar_p(b, c)})
    # H2b: precision x condition interaction (GEE)
    rows.append({"hypothesis": "H2b", "test": "GEE interaction precision x condition (joint Wald)",
                 "family": "all", "precision": "all", "n": np.nan, "estimate": np.nan, "detail": "",
                 "p_raw": p_interaction})
    # H3b: paired turn-1 confidence q4 vs fp16 on items correct at both (pooled over families)
    t1 = turn1(main)
    t1 = t1[(t1["correct_0"] == True) & t1["c0"].notna()]  # noqa: E712
    w = t1.pivot_table(index=["family", "item_id"], columns="precision", values="c0", aggfunc="first")
    w = w.dropna(subset=["q4_K_M", "fp16"])
    diff = (w["q4_K_M"] - w["fp16"]).to_numpy()
    items = w.index.get_level_values(1)
    per_item = pd.DataFrame({"item_id": items, "d": diff}).groupby("item_id")["d"].agg(["sum", "size"])
    lo, hi = boot_ratio_ci(per_item["sum"], per_item["size"])
    nonzero = diff[diff != 0]
    p_w = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    conf_table = pd.DataFrame([{"scope": "pooled over six families", "n_pairs": len(diff),
                                "mean_diff_q4_minus_fp16": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                                "p_wilcoxon": p_w, "supported": (hi < 0) and (p_w < 0.05)}])
    for fam in FAMILIES:
        if fam in w.index.get_level_values(0):
            dd = (w.loc[fam, "q4_K_M"] - w.loc[fam, "fp16"]).to_numpy()
            conf_table.loc[len(conf_table)] = {"scope": fam, "n_pairs": len(dd), "mean_diff_q4_minus_fp16": dd.mean(),
                                               "ci_lo": np.nan, "ci_hi": np.nan, "p_wilcoxon": np.nan, "supported": np.nan}
    rows.append({"hypothesis": "H3b", "test": "Wilcoxon signed-rank c0 q4_K_M vs fp16", "family": "all",
                 "precision": "q4_K_M vs fp16", "n": len(diff), "estimate": diff.mean(), "detail": "",
                 "p_raw": p_w})
    t = pd.DataFrame(rows)
    ok = t["p_raw"].notna()
    t.loc[ok, "p_holm"] = multipletests(t.loc[ok, "p_raw"], method="holm")[1]
    t["reject_holm_0.05"] = t["p_holm"] < 0.05
    return t, conf_table


def auroc_table(main: pd.DataFrame) -> pd.DataFrame:
    """H3a: turn-1 confidence c0 predicting 'held' (not harmful) among items correct in turn 1."""
    d = main[(main["correct_0"] == True) & main["harmful"].notna() & main["c0"].notna()]  # noqa: E712
    held = (~d["harmful"].astype(bool)).astype(int)
    rows = []

    def row(scope, m):
        y, s = held[m].to_numpy(), d.loc[m, "c0"].to_numpy(float)
        if len(np.unique(y)) < 2:
            return {**scope, "n": len(y), "held_rate": y.mean() if len(y) else np.nan,
                    "auroc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
        lo, hi = boot_auroc_ci(y, s, d.loc[m, "item_id"].to_numpy())
        return {**scope, "n": len(y), "held_rate": y.mean(), "auroc": roc_auc_score(y, s), "ci_lo": lo, "ci_hi": hi}

    rows.append(row({"scope": "pooled", "family": "all", "precision": "all", "condition": "all"},
                    pd.Series(True, index=d.index)))
    for prec in PRECISIONS:
        rows.append(row({"scope": "by precision", "family": "all", "precision": prec, "condition": "all"},
                        d["precision"] == prec))
    for (fam, prec, cond), g in d.groupby(["family", "precision", "condition"]):
        rows.append(row({"scope": "by configuration", "family": fam, "precision": prec, "condition": cond},
                        d.index.isin(g.index)))
    t = pd.DataFrame(rows)
    t["above_0.70"] = t["auroc"] > 0.70
    return t


def validity_kappa(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, prec, run), g in df.groupby(["family", "precision", "run"]):
        for turn, (a, ft) in (("turn1", ("a0", "ft_letter_0")), ("turn2", ("a1", "ft_letter_1"))):
            src = turn1(g) if turn == "turn1" else g
            s = src[src[a].notna() & src[ft].notna()]
            kappa = cohen_kappa_score(s[a], s[ft]) if len(s) and s[a].nunique() > 1 else np.nan
            rows.append({"family": fam, "precision": prec, "run": run, "turn": turn, "n": len(s),
                         "mismatch_rate": (s[a] != s[ft]).mean() if len(s) else np.nan, "cohens_kappa": kappa})
    return pd.DataFrame(rows)


def floor_social(desc: pd.DataFrame) -> pd.DataFrame:
    w = desc.pivot_table(index=["family", "precision"], columns="condition", values="hfr").reset_index()
    w["floor_speaker_free_minus_reask"] = w["speaker_free"] - w["reask"]
    w["social_user_minus_speaker_free"] = w["user"] - w["speaker_free"]
    w["social_expert_minus_speaker_free"] = w["expert"] - w["speaker_free"]
    return w


def calibration(main: pd.DataFrame) -> pd.DataFrame:
    t1 = turn1(main)
    rows = []
    for (fam, prec), g in t1.groupby(["family", "precision"]):
        g = g[g["c0"].notna()]
        rows.append({"family": fam, "precision": prec, "n": len(g), "mean_c0": g["c0"].mean(),
                     "accuracy": g["correct_0"].astype(bool).mean(),
                     "ece_15": ece(g["c0"], g["correct_0"].astype(bool))})
    return pd.DataFrame(rows)


def confidence_bins_and_curve(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Figure 3 data: flip rate by turn-1 confidence bin, and a logistic fit per precision."""
    d = main[(main["correct_0"] == True) & main["harmful"].notna() & main["c0"].notna()   # noqa: E712
             & (main["condition"] != "reask")].copy()
    d["harmful"] = d["harmful"].astype(int)
    edges = np.array(CONF_BIN_EDGES)
    labels = [f"{lo:g}–{hi:g}".replace("0.", ".") for lo, hi in zip(edges[:-1], edges[1:])]
    labels[0], labels[-1] = f"<{edges[1]:g}".replace("0.", "."), f"≥{edges[-2]:g}".replace("0.", ".")
    d["bin"] = pd.cut(d["c0"], edges, right=False, labels=labels)
    bins = (d.groupby(["precision", "bin"], observed=True)
             .agg(n=("harmful", "size"), flips=("harmful", "sum"), mean_c0=("c0", "mean")).reset_index())
    bins["flip_rate"] = bins["flips"] / bins["n"]
    ci = [wilson_ci(int(k), int(n)) for k, n in zip(bins["flips"], bins["n"])]
    bins["ci_lo"], bins["ci_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    bins["bin_index"] = bins["bin"].cat.codes
    bins["bin"] = bins["bin"].astype(str)
    grid = np.linspace(0.3, 0.9999, 60)
    curves = []
    for prec, g in d.groupby("precision"):
        X = sm.add_constant(np.log(g["c0"].to_numpy(float)))
        fit = sm.Logit(g["harmful"].to_numpy(), X).fit(disp=0)
        pred = fit.predict(sm.add_constant(np.log(grid)))
        curves += [{"precision": prec, "c0": c, "predicted_flip": p} for c, p in zip(grid, pred)]
    return bins, pd.DataFrame(curves)


def size_comparison(desc: pd.DataFrame) -> pd.DataFrame:
    """RQ4 (descriptive): larger model at 4 bits vs smaller model of the same developer at 16 bits."""
    pairs = [("Meta", "llama3.1-8b", "q4_K_M", "llama3.2-3b", "fp16"),
             ("Microsoft", "phi4-14b", "q4_K_M", "phi4-mini-3.8b", "fp16")]
    rows = []
    for dev, big, bp, small, sp in pairs:
        for cond in CONDITIONS:
            a = desc[(desc.family == big) & (desc.precision == bp) & (desc.condition == cond)]
            b = desc[(desc.family == small) & (desc.precision == sp) & (desc.condition == cond)]
            if len(a) and len(b):
                rows.append({"developer": dev, "condition": cond, "larger_4bit": f"{big} {bp}",
                             "hfr_larger_4bit": a.hfr.iloc[0], "smaller_16bit": f"{small} {sp}",
                             "hfr_smaller_16bit": b.hfr.iloc[0]})
    return pd.DataFrame(rows)


def table1(desc, auroc, calib, kappa) -> pd.DataFrame:
    w = desc.pivot_table(index=["family", "precision"], columns="condition", values="hfr")
    w.columns = [f"hfr_{c}" for c in w.columns]
    t1 = desc[desc.condition == "user"].set_index(["family", "precision"])[["turn1_valid", "n_correct0"]]
    au = (auroc[(auroc.scope == "by configuration") & (auroc.condition == "user")]
          .set_index(["family", "precision"])[["auroc"]].rename(columns={"auroc": "auroc_user"}))
    ca = calib.set_index(["family", "precision"])[["accuracy", "ece_15"]].rename(columns={"accuracy": "turn1_accuracy"})
    ka = (kappa[(kappa.run == "main") & (kappa.turn == "turn1")]
          .set_index(["family", "precision"])[["mismatch_rate"]].rename(columns={"mismatch_rate": "ft_text_mismatch"}))
    out = t1.join([ca, w, au, ka]).reset_index()
    out["family"] = pd.Categorical(out["family"], FAMILIES, ordered=True)
    out["precision"] = pd.Categorical(out["precision"], PRECISIONS, ordered=True)
    return out.sort_values(["family", "precision"])


def controls(df: pd.DataFrame) -> pd.DataFrame:
    ctrl_items = set(df.loc[df["control"] == True, "item_id"])  # noqa: E712
    d = df[df["item_id"].isin(ctrl_items) & df["run"].isin(["main", "reversed", "para1", "para2"])]
    configs = set(d.loc[d["run"] != "main", "model"])
    rows = []
    for (model, run, cond), g in d[d["model"].isin(configs)].groupby(["model", "run", "condition"]):
        g = g[(g["correct_0"] == True) & g["harmful"].notna()]  # noqa: E712
        h = g["harmful"].astype(bool)
        lo, hi = boot_ratio_ci(h.astype(float), np.ones(len(h)))
        rows.append({"model": model, "family": g["family"].iloc[0] if len(g) else "", "precision":
                     g["precision"].iloc[0] if len(g) else "", "variant": run, "condition": cond,
                     "n_correct0": len(g), "hfr": h.mean() if len(h) else np.nan, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def determinism(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["run"].isin(["det1", "det2", "main"])]
    rows = []
    for model, g in d.groupby("model"):
        runs = set(g["run"])
        if not {"det1", "det2"} <= runs:
            continue
        for x, y in (("det1", "det2"), ("det1", "main")):
            m = g[g.run == x].merge(g[g.run == y], on=["item_id", "condition"], suffixes=("_x", "_y"))
            rows.append({"model": model, "compare": f"{x} vs {y}", "n": len(m),
                         "same_a0": (m.a0_x == m.a0_y).mean(), "same_a1": (m.a1_x == m.a1_y).mean(),
                         "same_raw_0": (m.raw_0_x == m.raw_0_y).mean(), "same_raw_1": (m.raw_1_x == m.raw_1_y).mean(),
                         "max_abs_diff_c0": (m.c0_x - m.c0_y).abs().max()})
    return pd.DataFrame(rows)


def offceiling(main: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: the primary comparison on items with c0 < 0.9 at both precisions."""
    pairs = paired(main, "user", "q4_K_M", "fp16")
    sub = pairs[(pairs["c0_a"] < 0.9) & (pairs["c0_b"] < 0.9)]
    return pd.DataFrame([{"scope": "pooled, c0 < 0.9 at both precisions", "role": "exploratory",
                          **compare(sub)}])


def precision_by_condition(main: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: q4_K_M vs fp16 and q8_0 vs fp16 for every follow-up (pooled and per family).

    The rows for `user` repeat the pre-registered primary / H1b comparisons (role column says so).
    """
    rows = []
    for pa, level in (("q4_K_M", 0.95), ("q8_0", 0.90)):
        for cond in CONDITIONS:
            pairs = paired(main, cond, pa, "fp16")
            role = "pre-registered (see primary/h1b)" if cond == "user" else "exploratory"
            rows.append({"comparison": f"{pa} vs fp16", "condition": cond, "scope": "pooled over six families",
                         "role": role, **compare(pairs, level)})
            for fam in FAMILIES:
                rows.append({"comparison": f"{pa} vs fp16", "condition": cond, "scope": fam,
                             "role": "per family (descriptive)", **compare(pairs[pairs["family"] == fam], level)})
    return pd.DataFrame(rows)


def hypothesis_summary(primary, h1b, sec, auroc, conf, gee, kappa) -> pd.DataFrame:
    """One row per pre-registered hypothesis: estimate, interval, p and whether the rule is met.

    Rules as in the README pre-registration and notes/GUIDELINES §6. `rule_met` is 'n/a' where
    no numeric rule was fixed in advance.
    """
    def pct(x):
        return f"{100 * x:.1f}"

    p, e = primary.iloc[0], h1b.iloc[0]
    rows = [
        {"hypothesis": "H1a", "claim": "harmful flip rate q4_K_M > fp16 (user; pooled, six families)",
         "test": "exact McNemar, two-sided", "estimate": f"{pct(p.hfr_a)} % vs {pct(p.hfr_b)} % "
         f"(diff {100 * p.diff_a_minus_b:+.1f} pp)", "interval": f"95% CI {100 * p.ci_lo:+.1f} to {100 * p.ci_hi:+.1f} pp",
         "p": p.p_mcnemar_exact, "n": f"{p.n_pairs} pairs ({p.only_a} vs {p.only_b} discordant)",
         "rule": "p < .05 and q4_K_M higher", "rule_met": "yes" if bool(p.supported) else "no"},
        {"hypothesis": "H1b", "claim": "q8_0 and fp16 equivalent (user; pooled)", "test": "paired bootstrap 90% CI (TOST)",
         "estimate": f"diff {100 * e.diff_a_minus_b:+.1f} pp", "interval": f"90% CI {100 * e.ci_lo:+.1f} to {100 * e.ci_hi:+.1f} pp",
         "p": np.nan, "n": f"{e.n_pairs} pairs", "rule": "CI inside ±3 pp",
         "rule_met": "yes" if bool(e.equivalent) else "no"},
    ]
    h2a = sec[sec.hypothesis == "H2a"]
    higher = int(((h2a.p_holm < 0.05) & (h2a.estimate > 0)).sum())
    lower = int(((h2a.p_holm < 0.05) & (h2a.estimate < 0)).sum())
    rows.append({"hypothesis": "H2a", "claim": "speaker_free flips more than reask", "test": "exact McNemar per configuration, Holm",
                 "estimate": f"higher in {higher} of {len(h2a)} configurations, lower in {lower}",
                 "interval": "", "p": h2a.p_holm.max() if higher == len(h2a) else np.nan, "n": f"{len(h2a)} configurations",
                 "rule": "significant and higher in most configurations", "rule_met": "yes" if higher > len(h2a) / 2 else "no"})
    h2b = sec[sec.hypothesis == "H2b"].iloc[0]
    rows.append({"hypothesis": "H2b", "claim": "precision x condition interaction", "test": "GEE (item clusters), joint Wald, Holm",
                 "estimate": "", "interval": "", "p": h2b.p_holm, "n": "", "rule": "interaction significant (Holm)",
                 "rule_met": "yes" if h2b.p_holm < 0.05 else "no"})
    a = auroc[auroc.scope == "pooled"].iloc[0]
    rows.append({"hypothesis": "H3a", "claim": "turn-1 confidence predicts holding", "test": "AUROC, item bootstrap",
                 "estimate": f"AUROC {a.auroc:.2f}", "interval": f"95% CI {a.ci_lo:.2f} to {a.ci_hi:.2f}", "p": np.nan,
                 "n": f"{int(a.n)} turn-2 answers", "rule": "AUROC > 0.70 and CI above 0.5",
                 "rule_met": "yes" if (a.auroc > 0.70 and a.ci_lo > 0.5) else "no"})
    c = conf.iloc[0]
    h3b = sec[sec.hypothesis == "H3b"].iloc[0]
    rows.append({"hypothesis": "H3b", "claim": "turn-1 confidence lower at q4_K_M than fp16", "test": "Wilcoxon signed-rank, Holm; bootstrap",
                 "estimate": f"mean diff {c.mean_diff_q4_minus_fp16:+.4f}", "interval": f"95% CI {c.ci_lo:+.4f} to {c.ci_hi:+.4f}",
                 "p": h3b.p_holm, "n": f"{int(c.n_pairs)} pairs", "rule": "CI below 0 and p < .05",
                 "rule_met": "yes" if (c.ci_hi < 0 and h3b.p_holm < 0.05) else "no"})
    term = "C(precision, Treatment('fp16'))[T.q4_K_M]"
    g0 = gee[(gee.model == "without_log_c0") & (gee.term == term)]
    g1 = gee[(gee.model == "with_log_c0") & (gee.term == term)]
    if len(g0) and len(g1):
        rows.append({"hypothesis": "H3c", "claim": "q4_K_M effect shrinks once log c0 is included (guidelines, not README)",
                     "test": "GEE with and without log c0", "estimate": f"q4_K_M coef {g0.coef.iloc[0]:.3f} -> {g1.coef.iloc[0]:.3f}",
                     "interval": "", "p": np.nan, "n": "", "rule": "coefficient shrinks (no threshold fixed)", "rule_met": "n/a"})
    k = kappa[(kappa.run == "main")]
    rows.append({"hypothesis": "Validity", "claim": "first-token letter agrees with text letter", "test": "Cohen's kappa per configuration and turn",
                 "estimate": f"kappa {k.cohens_kappa.min():.3f} to {k.cohens_kappa.max():.3f}",
                 "interval": f"mismatch {100 * k.mismatch_rate.max():.1f} % at most", "p": np.nan, "n": f"{len(k)} configuration x turn",
                 "rule": "reported, no threshold", "rule_met": "n/a"})
    t = pd.DataFrame(rows)

    # compact versions for the poster table (same numbers, shorter wording)
    def fp(x):
        return "< .001" if x < 0.001 else f"= {x:.3f}".replace("0.", ".")

    short = {
        "H1a": ("q4_K_M flips more than fp16 (user)", "exact McNemar",
                f"{pct(p.hfr_a)} vs {pct(p.hfr_b)} % ({100 * p.diff_a_minus_b:+.1f} pp), p {fp(p.p_mcnemar_exact)}"),
        "H1b": ("q8_0 ≈ fp16 within ±3 pp (user)", "paired bootstrap, 90% CI",
                f"{100 * e.diff_a_minus_b:+.1f} pp [{100 * e.ci_lo:+.1f}, {100 * e.ci_hi:+.1f}]"),
        "H2a": ("speaker-free flips more than re-ask", "McNemar per config., Holm",
                f"higher in {higher}/{len(h2a)}, lower in {lower} (p_Holm < .05)"),
        "H2b": ("precision × follow-up interaction", "GEE, joint Wald, Holm", f"p_Holm {fp(h2b.p_holm)}"),
        "H3a": ("turn-1 confidence predicts holding", "AUROC > .70, bootstrap",
                f"AUROC {a.auroc:.2f} [{a.ci_lo:.2f}, {a.ci_hi:.2f}]"),
        "H3b": ("confidence lower at q4_K_M than fp16", "Wilcoxon, Holm; bootstrap",
                f"{c.mean_diff_q4_minus_fp16:+.4f} [{c.ci_lo:+.4f}, {c.ci_hi:+.4f}], p_Holm {fp(h3b.p_holm)}"),
        "Validity": ("first-token letter = text letter", "Cohen's κ per config.",
                     f"κ ≥ {np.floor(1000 * k.cohens_kappa.min()) / 1000:.3f}; mismatch ≤ {100 * k.mismatch_rate.max():.1f} %"),
    }
    if len(g0) and len(g1):
        short["H3c"] = ("q4_K_M effect shrinks with log c0", "GEE ± log c0",
                        f"coefficient {g0.coef.iloc[0]:.3f} → {g1.coef.iloc[0]:.3f}")
    t["prediction_short"] = t["hypothesis"].map(lambda h: short.get(h, ("", "", ""))[0])
    t["test_short"] = t["hypothesis"].map(lambda h: short.get(h, ("", "", ""))[1])
    t["result_short"] = t["hypothesis"].map(lambda h: short.get(h, ("", "", ""))[2])
    return t


def overview(df: pd.DataFrame, items_path: Path) -> pd.DataFrame:
    items = [json.loads(l) for l in items_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    main = main_data(df)
    rows = {"items": len(items), "items_mmlu_pro": sum(i["source"] == "mmlu_pro" for i in items),
            "items_arc": sum(i["source"] == "arc" for i in items), "control_items": sum(i["control"] for i in items),
            "follow_up_conditions": len(CONDITIONS), "families": main["family"].nunique(),
            "precisions": main["precision"].nunique(), "configurations": main["model"].nunique(),
            "records_main": len(main), "records_controls": int(df["run"].isin(["reversed", "para1", "para2"]).sum()),
            "records_determinism": int(df["run"].isin(["det1", "det2"]).sum())}
    return pd.DataFrame([{"quantity": k, "value": v} for k, v in rows.items()])


def by_source(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, prec, cond, src), g in main.groupby(["family", "precision", "condition", "source"]):
        g = g[(g["correct_0"] == True) & g["harmful"].notna()]  # noqa: E712
        rows.append({"family": fam, "precision": prec, "condition": cond, "source": src,
                     "n_correct0": len(g), "hfr": g["harmful"].astype(bool).mean() if len(g) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def write(t: pd.DataFrame, out_dir: Path, name: str) -> None:
    t.to_csv(out_dir / name, index=False, encoding="utf-8", float_format="%.6g", lineterminator="\n")
    print(f"  {name:<40} {len(t):>4} rows")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis")
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items.jsonl")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    main_df = main_data(df)
    print(f"loaded {len(df)} records; main runs: {main_df['model'].nunique()} configurations")

    desc = descriptives(main_df)
    primary, h1b = primary_and_h1b(main_df)
    gee, p_inter, gee_notes = gee_models(main_df)
    sec, conf = secondary(main_df, p_inter)
    auroc = auroc_table(main_df)
    kappa = validity_kappa(df[df["run"] == "main"])
    calib = calibration(main_df)
    bins, curve = confidence_bins_and_curve(main_df)

    summary = hypothesis_summary(primary, h1b, sec, auroc, conf, gee, kappa)

    print(f"writing to {args.out_dir}")
    for name, t in [("hypothesis_summary.csv", summary), ("overview.csv", overview(df, args.items)),
                    ("descriptives.csv", desc), ("primary_mcnemar.csv", primary), ("h1b_equivalence.csv", h1b),
                    ("gee_coefficients.csv", gee), ("secondary_holm.csv", sec), ("confidence_q4_fp16.csv", conf),
                    ("auroc.csv", auroc), ("validity_kappa.csv", kappa), ("floor_social.csv", floor_social(desc)),
                    ("calibration.csv", calib), ("confidence_bins.csv", bins), ("confidence_curve.csv", curve),
                    ("descriptive_size_comparison.csv", size_comparison(desc)),
                    ("table1.csv", table1(desc, auroc, calib, kappa)),
                    ("robustness_controls.csv", controls(df)), ("robustness_determinism.csv", determinism(df)),
                    ("exploratory_offceiling_primary.csv", offceiling(main_df)),
                    ("exploratory_precision_by_condition.csv", precision_by_condition(main_df)),
                    ("exploratory_by_source.csv", by_source(main_df))]:
        write(t, args.out_dir, name)
    if gee_notes:
        print("GEE warnings (also check gee_coefficients.csv):\n  " + "\n  ".join(sorted(set(gee_notes))[:5]))
    p = primary.iloc[0]
    print(f"\nPRIMARY (pooled, user, q4_K_M vs fp16): n={p.n_pairs} pairs, HFR {p.hfr_a:.3f} vs {p.hfr_b:.3f}, "
          f"q4 only {p.only_a}, fp16 only {p.only_b}, exact McNemar p={p.p_mcnemar_exact:.4g}")
    e = h1b.iloc[0]
    print(f"H1b (q8_0 - fp16): {e.diff_a_minus_b:+.3f}, 90% CI [{e.ci_lo:+.3f}, {e.ci_hi:+.3f}], "
          f"equivalent within ±{TOST_MARGIN:.2f}: {bool(e.equivalent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
