"""Descriptive tables: flip rates, confidence bins, size comparison, overview."""
import json

import numpy as np
import pandas as pd

from cavein.config import CONDITIONS, CONF_BIN_EDGES
from cavein.results import main_data
from cavein.stats import boot_ratio_ci, wilson_ci


def descriptives(main):
    """Harmful flip rate (with 95% CI) and some counts per family, precision and follow-up."""
    rows = []
    for (family, precision, condition), g in main.groupby(["family", "precision", "condition"]):
        valid0 = g[g["a0"].notna()]                            # turn 1 gave a letter
        correct = valid0[valid0["correct_0"] == True]          # ... and it was correct
        correct_valid = correct[correct["harmful"].notna()]    # ... and turn 2 gave a letter too
        harmful = correct_valid["harmful"].astype(bool)
        lo, hi = boot_ratio_ci(harmful.astype(float), np.ones(len(harmful)))

        if len(harmful) > 0:
            hfr = harmful.mean()
        else:
            hfr = np.nan

        rows.append({
            "family": family,
            "precision": precision,
            "condition": condition,
            "n_items": len(g),
            "turn1_valid": len(valid0),
            "n_correct0": len(correct),
            "n_correct0_valid_turn2": len(correct_valid),
            "invalid_turn2": int(valid0["a1"].isna().sum()),
            "harmful": int(harmful.sum()),
            "hfr": hfr,
            "hfr_ci_lo": lo,
            "hfr_ci_hi": hi,
        })
    return pd.DataFrame(rows)


def bin_labels(edges):
    # names of the confidence bins, e.g. '<.5', '.5–.8', ..., '≥.999'
    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        labels.append(f"{lo:g}–{hi:g}".replace("0.", "."))
    labels[0] = f"<{edges[1]:g}".replace("0.", ".")
    labels[-1] = f"≥{edges[-2]:g}".replace("0.", ".")
    return labels


def confidence_bins(main):
    """Harmful flip rate per turn-1 confidence bin and precision (with Wilson CIs). The re-ask follow-up is left out."""
    d = main[(main["correct_0"] == True) & main["harmful"].notna() & main["c0"].notna()
             & (main["condition"] != "reask")].copy()
    d["harmful"] = d["harmful"].astype(int)
    edges = np.array(CONF_BIN_EDGES)
    d["bin"] = pd.cut(d["c0"], edges, right=False, labels=bin_labels(edges))

    bins = d.groupby(["precision", "bin"], observed=True).agg(
        n=("harmful", "size"), flips=("harmful", "sum"), mean_c0=("c0", "mean")).reset_index()
    bins["flip_rate"] = bins["flips"] / bins["n"]

    ci_lo = []
    ci_hi = []
    for i in range(len(bins)):
        flips = int(bins["flips"].iloc[i])
        n = int(bins["n"].iloc[i])
        lo, hi = wilson_ci(flips, n)
        ci_lo.append(lo)
        ci_hi.append(hi)
    bins["ci_lo"] = ci_lo
    bins["ci_hi"] = ci_hi
    bins["bin_index"] = bins["bin"].cat.codes
    bins["bin"] = bins["bin"].astype(str)
    return bins


def size_comparison(desc):
    """A larger model at 4 bits vs a smaller model of the same developer at 16 bits."""
    # (developer, larger family, its precision, smaller family, its precision)
    pairs = [("Meta", "llama3.1-8b", "q4_K_M", "llama3.2-3b", "fp16"),
             ("Microsoft", "phi4-14b", "q4_K_M", "phi4-mini-3.8b", "fp16")]
    rows = []
    for developer, big, big_prec, small, small_prec in pairs:
        for condition in CONDITIONS:
            a = desc[(desc.family == big) & (desc.precision == big_prec) & (desc.condition == condition)]
            b = desc[(desc.family == small) & (desc.precision == small_prec) & (desc.condition == condition)]
            if len(a) > 0 and len(b) > 0:
                rows.append({"developer": developer,
                             "condition": condition,
                             "larger_4bit": f"{big} {big_prec}",
                             "hfr_larger_4bit": a.hfr.iloc[0],
                             "smaller_16bit": f"{small} {small_prec}",
                             "hfr_smaller_16bit": b.hfr.iloc[0]})
    return pd.DataFrame(rows)


def overview(df, items_path):
    """The design numbers shown on the poster (number of items, models, records ...)."""
    items = []
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))

    n_mmlu = 0
    n_arc = 0
    n_control = 0
    for item in items:
        if item["source"] == "mmlu_pro":
            n_mmlu += 1
        if item["source"] == "arc":
            n_arc += 1
        if item["control"]:
            n_control += 1

    main = main_data(df)
    counts = {
        "items": len(items),
        "items_mmlu_pro": n_mmlu,
        "items_arc": n_arc,
        "control_items": n_control,
        "follow_up_conditions": len(CONDITIONS),
        "families": main["family"].nunique(),
        "precisions": main["precision"].nunique(),
        "configurations": main["model"].nunique(),
        "records_main": len(main),
        "records_controls": int(df["run"].isin(["reversed", "para1", "para2"]).sum()),
        "records_determinism": int(df["run"].isin(["det1", "det2"]).sum()),
    }

    rows = []
    for key in counts:
        rows.append({"quantity": key, "value": counts[key]})
    return pd.DataFrame(rows)
