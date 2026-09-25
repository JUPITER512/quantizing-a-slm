"""Robustness checks from the design: control variants and the determinism rerun."""
import numpy as np
import pandas as pd

from cavein.stats import boot_ratio_ci


def controls(df):
    """Harmful flip rate on the 100 control items: main wording vs reversed order and the two paraphrases."""
    control_items = set(df.loc[df["control"] == True, "item_id"])
    d = df[df["item_id"].isin(control_items) & df["run"].isin(["main", "reversed", "para1", "para2"])]
    models = set(d.loc[d["run"] != "main", "model"])
    rows = []
    for (model, run, condition), g in d[d["model"].isin(models)].groupby(["model", "run", "condition"]):
        g = g[(g["correct_0"] == True) & g["harmful"].notna()]
        harmful = g["harmful"].astype(bool)
        lo, hi = boot_ratio_ci(harmful.astype(float), np.ones(len(harmful)))
        rows.append({"model": model, "family": g["family"].iloc[0] if len(g) else "",
                     "precision": g["precision"].iloc[0] if len(g) else "", "variant": run, "condition": condition,
                     "n_correct0": len(g), "hfr": harmful.mean() if len(harmful) else np.nan,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def determinism(df):
    """How often two runs of the same model give the same answers (det1 vs det2, det1 vs the main run)."""
    d = df[df["run"].isin(["det1", "det2", "main"])]
    rows = []
    for model, g in d.groupby("model"):
        runs = set(g["run"])
        if "det1" not in runs or "det2" not in runs:
            continue
        for x, y in [("det1", "det2"), ("det1", "main")]:
            m = g[g.run == x].merge(g[g.run == y], on=["item_id", "condition"], suffixes=("_x", "_y"))
            rows.append({"model": model, "compare": f"{x} vs {y}", "n": len(m),
                         "same_a0": (m.a0_x == m.a0_y).mean(), "same_a1": (m.a1_x == m.a1_y).mean(),
                         "same_raw_0": (m.raw_0_x == m.raw_0_y).mean(), "same_raw_1": (m.raw_1_x == m.raw_1_y).mean(),
                         "max_abs_diff_c0": (m.c0_x - m.c0_y).abs().max()})
    return pd.DataFrame(rows)
