"""Exploratory analyses: NOT pre-registered, labelled as exploratory in the CSV names."""
import numpy as np
import pandas as pd

from cavein.config import CONDITIONS, FAMILIES
from cavein.results import paired
from cavein.stats import compare


def offceiling(main):
    """The primary comparison on items with c0 < 0.9 at both precisions."""
    pairs = paired(main, "user", "q4_K_M", "fp16")
    sub = pairs[(pairs["c0_a"] < 0.9) & (pairs["c0_b"] < 0.9)]
    row = {"scope": "pooled, c0 < 0.9 at both precisions", "role": "exploratory"}
    row.update(compare(sub))
    return pd.DataFrame([row])


def precision_by_condition(main):
    """q4_K_M vs fp16 and q8_0 vs fp16 for every follow-up, pooled and per family (the `user` rows repeat
    the pre-registered tests)."""
    rows = []
    for prec_a, level in [("q4_K_M", 0.95), ("q8_0", 0.90)]:
        for condition in CONDITIONS:
            pairs = paired(main, condition, prec_a, "fp16")
            role = "pre-registered (see primary/h1b)" if condition == "user" else "exploratory"
            row = {"comparison": f"{prec_a} vs fp16", "condition": condition,
                   "scope": "pooled over six families", "role": role}
            row.update(compare(pairs, level))
            rows.append(row)
            for family in FAMILIES:
                row = {"comparison": f"{prec_a} vs fp16", "condition": condition, "scope": family,
                       "role": "per family (descriptive)"}
                row.update(compare(pairs[pairs["family"] == family], level))
                rows.append(row)
    return pd.DataFrame(rows)


def by_source(main):
    """Harmful flip rate separately for MMLU-Pro and ARC items."""
    rows = []
    for (family, precision, condition, source), g in main.groupby(["family", "precision", "condition", "source"]):
        g = g[(g["correct_0"] == True) & g["harmful"].notna()]
        rows.append({"family": family, "precision": precision, "condition": condition, "source": source,
                     "n_correct0": len(g), "hfr": g["harmful"].astype(bool).mean() if len(g) else np.nan})
    return pd.DataFrame(rows)
