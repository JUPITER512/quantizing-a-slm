"""Descriptive tables: flip rates, calibration, confidence bins, size comparison, Table 1, overview."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cavein.config import CONDITIONS, CONF_BIN_EDGES, FAMILIES, PRECISIONS
from cavein.results import main_data, turn1
from cavein.stats import boot_ratio_ci, ece, wilson_ci


def descriptives(main):
    """Harmful flip rate (with 95% CI) and related counts per family, precision and follow-up."""
    rows = []
    for (family, precision, condition), g in main.groupby(["family", "precision", "condition"]):
        valid0 = g[g["a0"].notna()]
        correct = valid0[valid0["correct_0"] == True]
        correct_valid = correct[correct["harmful"].notna()]
        wrong_valid = valid0[(valid0["correct_0"] == False) & valid0["a1"].notna()]
        harmful = correct_valid["harmful"].astype(bool)
        lo, hi = boot_ratio_ci(harmful.astype(float), np.ones(len(harmful)))
        rows.append({
            "family": family, "precision": precision, "condition": condition, "n_items": len(g),
            "turn1_valid": len(valid0), "n_correct0": len(correct), "n_correct0_valid_turn2": len(correct_valid),
            "invalid_turn2": int(valid0["a1"].isna().sum()), "harmful": int(harmful.sum()),
            "hfr": harmful.mean() if len(harmful) else np.nan, "hfr_ci_lo": lo, "hfr_ci_hi": hi,
            "capitulation_rate": correct_valid["went_to_x"].astype(bool).mean() if len(correct_valid) else np.nan,
            "beneficial_rate": wrong_valid["beneficial"].astype(bool).mean() if len(wrong_valid) else np.nan,
            "median_answer_mass_0": valid0["answer_mass_0"].median(),
        })
    return pd.DataFrame(rows)


def floor_social(desc):
    """Speaker-free floor (speaker_free - reask) and the social increment (user/expert - speaker_free)."""
    w = desc.pivot_table(index=["family", "precision"], columns="condition", values="hfr").reset_index()
    w["floor_speaker_free_minus_reask"] = w["speaker_free"] - w["reask"]
    w["social_user_minus_speaker_free"] = w["user"] - w["speaker_free"]
    w["social_expert_minus_speaker_free"] = w["expert"] - w["speaker_free"]
    return w


def calibration(main):
    rows = []
    for (family, precision), g in turn1(main).groupby(["family", "precision"]):
        g = g[g["c0"].notna()]
        rows.append({"family": family, "precision": precision, "n": len(g), "mean_c0": g["c0"].mean(),
                     "accuracy": g["correct_0"].astype(bool).mean(),
                     "ece_15": ece(g["c0"], g["correct_0"].astype(bool))})
    return pd.DataFrame(rows)


def confidence_bins_and_curve(main):
    """Harmful flip rate per turn-1 confidence bin, and a logistic fit per precision."""
    d = main[(main["correct_0"] == True) & main["harmful"].notna() & main["c0"].notna()
             & (main["condition"] != "reask")].copy()
    d["harmful"] = d["harmful"].astype(int)
    edges = np.array(CONF_BIN_EDGES)
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo:g}–{hi:g}".replace("0.", "."))
    labels[0] = f"<{edges[1]:g}".replace("0.", ".")
    labels[-1] = f"≥{edges[-2]:g}".replace("0.", ".")
    d["bin"] = pd.cut(d["c0"], edges, right=False, labels=labels)

    bins = d.groupby(["precision", "bin"], observed=True).agg(
        n=("harmful", "size"), flips=("harmful", "sum"), mean_c0=("c0", "mean")).reset_index()
    bins["flip_rate"] = bins["flips"] / bins["n"]
    ci_lo = []
    ci_hi = []
    for flips, n in zip(bins["flips"], bins["n"]):
        lo, hi = wilson_ci(int(flips), int(n))
        ci_lo.append(lo)
        ci_hi.append(hi)
    bins["ci_lo"] = ci_lo
    bins["ci_hi"] = ci_hi
    bins["bin_index"] = bins["bin"].cat.codes
    bins["bin"] = bins["bin"].astype(str)

    grid = np.linspace(0.3, 0.9999, 60)
    curves = []
    for precision, g in d.groupby("precision"):
        X = sm.add_constant(np.log(g["c0"].to_numpy(float)))
        fit = sm.Logit(g["harmful"].to_numpy(), X).fit(disp=0)
        predicted = fit.predict(sm.add_constant(np.log(grid)))
        for c0, p in zip(grid, predicted):
            curves.append({"precision": precision, "c0": c0, "predicted_flip": p})
    return bins, pd.DataFrame(curves)


def size_comparison(desc):
    """RQ4: a larger model at 4 bits vs a smaller model of the same developer at 16 bits."""
    pairs = [("Meta", "llama3.1-8b", "q4_K_M", "llama3.2-3b", "fp16"),
             ("Microsoft", "phi4-14b", "q4_K_M", "phi4-mini-3.8b", "fp16")]
    rows = []
    for developer, big, big_prec, small, small_prec in pairs:
        for condition in CONDITIONS:
            a = desc[(desc.family == big) & (desc.precision == big_prec) & (desc.condition == condition)]
            b = desc[(desc.family == small) & (desc.precision == small_prec) & (desc.condition == condition)]
            if len(a) and len(b):
                rows.append({"developer": developer, "condition": condition, "larger_4bit": f"{big} {big_prec}",
                             "hfr_larger_4bit": a.hfr.iloc[0], "smaller_16bit": f"{small} {small_prec}",
                             "hfr_smaller_16bit": b.hfr.iloc[0]})
    return pd.DataFrame(rows)


def table1(desc, auroc, calib, kappa):
    hfr = desc.pivot_table(index=["family", "precision"], columns="condition", values="hfr")
    hfr.columns = [f"hfr_{c}" for c in hfr.columns]
    counts = desc[desc.condition == "user"].set_index(["family", "precision"])[["turn1_valid", "n_correct0"]]
    au = auroc[(auroc.scope == "by configuration") & (auroc.condition == "user")]
    au = au.set_index(["family", "precision"])[["auroc"]].rename(columns={"auroc": "auroc_user"})
    ca = calib.set_index(["family", "precision"])[["accuracy", "ece_15"]]
    ca = ca.rename(columns={"accuracy": "turn1_accuracy"})
    ka = kappa[(kappa.run == "main") & (kappa.turn == "turn1")]
    ka = ka.set_index(["family", "precision"])[["mismatch_rate"]].rename(columns={"mismatch_rate": "ft_text_mismatch"})
    out = counts.join([ca, hfr, au, ka]).reset_index()
    out["family"] = pd.Categorical(out["family"], FAMILIES, ordered=True)
    out["precision"] = pd.Categorical(out["precision"], PRECISIONS, ordered=True)
    return out.sort_values(["family", "precision"])


def overview(df, items_path):
    """The design numbers shown on the poster."""
    items = []
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    main = main_data(df)
    counts = {
        "items": len(items),
        "items_mmlu_pro": sum(item["source"] == "mmlu_pro" for item in items),
        "items_arc": sum(item["source"] == "arc" for item in items),
        "control_items": sum(item["control"] for item in items),
        "follow_up_conditions": len(CONDITIONS),
        "families": main["family"].nunique(),
        "precisions": main["precision"].nunique(),
        "configurations": main["model"].nunique(),
        "records_main": len(main),
        "records_controls": int(df["run"].isin(["reversed", "para1", "para2"]).sum()),
        "records_determinism": int(df["run"].isin(["det1", "det2"]).sum()),
    }
    return pd.DataFrame([{"quantity": key, "value": value} for key, value in counts.items()])
