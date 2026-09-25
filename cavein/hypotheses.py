"""The pre-registered tests (README): primary McNemar, H1b equivalence and the secondary tests."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

from cavein.config import FAMILIES, GEE_MAXITER, PRECISIONS, TOST_MARGIN
from cavein.results import paired, turn1
from cavein.stats import boot_auroc_ci, boot_ratio_ci, compare, mcnemar_p


def precision_test(main, prec_a, level):
    """prec_a vs fp16 on the `user` follow-up: pooled over the six families, then per family."""
    pairs = paired(main, "user", prec_a, "fp16")
    rows = []
    row = {"scope": "pooled over six families", "role": "pre-registered", "precision_a": prec_a,
           "precision_b": "fp16", "condition": "user"}
    row.update(compare(pairs, level))
    rows.append(row)
    for family in FAMILIES:
        row = {"scope": family, "role": "per family (descriptive)", "precision_a": prec_a,
               "precision_b": "fp16", "condition": "user"}
        row.update(compare(pairs[pairs["family"] == family], level))
        rows.append(row)
    return pd.DataFrame(rows)


def primary_and_h1b(main):
    primary = precision_test(main, "q4_K_M", 0.95)
    primary["supported"] = (primary["p_mcnemar_exact"] < 0.05) & (primary["diff_a_minus_b"] > 0)
    h1b = precision_test(main, "q8_0", 0.90)
    h1b["margin"] = TOST_MARGIN
    h1b["equivalent"] = (h1b["ci_lo"] > -TOST_MARGIN) & (h1b["ci_hi"] < TOST_MARGIN)
    return primary, h1b


def gee_models(main):
    """GEE logistic regression with item clusters, with and without log c0 (same rows)."""
    d = main[(main["correct_0"] == True) & main["harmful"].notna() & main["c0"].notna() & (main["c0"] > 0)].copy()
    d["harmful"] = d["harmful"].astype(int)
    d["log_c0"] = np.log(d["c0"].astype(float))
    formula = "harmful ~ C(precision, Treatment('fp16')) * C(condition, Treatment('reask')) + C(family)"
    rows = []
    notes = []
    p_interaction = np.nan
    for label, f in [("without_log_c0", formula), ("with_log_c0", formula + " + log_c0")]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.gee(f, groups="item_id", data=d, family=sm.families.Binomial(),
                            cov_struct=sm.cov_struct.Exchangeable())
            result = model.fit(maxiter=GEE_MAXITER)
        converged = True
        for w in caught:
            notes.append(f"{label}: {w.message}")
            if "converg" in str(w.message):
                converged = False
        for term in result.params.index:
            rows.append({"model": label, "term": term, "coef": result.params[term], "se": result.bse[term],
                         "p": result.pvalues[term], "n_obs": int(result.nobs), "n_items": d["item_id"].nunique(),
                         "converged": converged})
        if label == "without_log_c0":
            terms = list(result.params.index)
            interaction_terms = [i for i, t in enumerate(terms) if ":" in t]
            R = np.zeros((len(interaction_terms), len(terms)))
            for row_number, column in enumerate(interaction_terms):
                R[row_number, column] = 1
            p_interaction = float(np.asarray(result.wald_test(R, scalar=True).pvalue))
    return pd.DataFrame(rows), p_interaction, notes


def secondary(main, p_interaction):
    """H2a, H2b and H3b with Holm correction; also the table of confidence differences."""
    rows = []
    # H2a: speaker_free vs reask in every configuration
    for (family, precision), g in main.groupby(["family", "precision"]):
        g = g[(g["correct_0"] == True) & g["harmful"].notna()]
        w = g.pivot_table(index="item_id", columns="condition", values="harmful", aggfunc="first")
        w = w.dropna(subset=["speaker_free", "reask"])
        sf = w["speaker_free"].astype(bool)
        rq = w["reask"].astype(bool)
        b = int((sf & ~rq).sum())
        c = int((~sf & rq).sum())
        rows.append({"hypothesis": "H2a", "test": "McNemar speaker_free vs reask", "family": family,
                     "precision": precision, "n": len(w), "estimate": sf.mean() - rq.mean(),
                     "detail": f"sf only {b}, reask only {c}", "p_raw": mcnemar_p(b, c)})
    # H2b: precision x condition interaction from the GEE
    rows.append({"hypothesis": "H2b", "test": "GEE interaction precision x condition (joint Wald)",
                 "family": "all", "precision": "all", "n": np.nan, "estimate": np.nan, "detail": "",
                 "p_raw": p_interaction})

    # H3b: turn-1 confidence q4_K_M vs fp16 on items correct at both precisions
    t1 = turn1(main)
    t1 = t1[(t1["correct_0"] == True) & t1["c0"].notna()]
    w = t1.pivot_table(index=["family", "item_id"], columns="precision", values="c0", aggfunc="first")
    w = w.dropna(subset=["q4_K_M", "fp16"])
    diff = (w["q4_K_M"] - w["fp16"]).to_numpy()
    per_item = pd.DataFrame({"item_id": w.index.get_level_values(1), "d": diff})
    per_item = per_item.groupby("item_id")["d"].agg(["sum", "size"])
    lo, hi = boot_ratio_ci(per_item["sum"], per_item["size"])
    nonzero = diff[diff != 0]
    p_wilcoxon = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    conf_rows = [{"scope": "pooled over six families", "n_pairs": len(diff), "mean_diff_q4_minus_fp16": diff.mean(),
                  "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p_wilcoxon, "supported": (hi < 0) and (p_wilcoxon < 0.05)}]
    for family in FAMILIES:
        if family in w.index.get_level_values(0):
            family_diff = (w.loc[family, "q4_K_M"] - w.loc[family, "fp16"]).to_numpy()
            conf_rows.append({"scope": family, "n_pairs": len(family_diff),
                              "mean_diff_q4_minus_fp16": family_diff.mean(), "ci_lo": np.nan, "ci_hi": np.nan,
                              "p_wilcoxon": np.nan, "supported": np.nan})
    rows.append({"hypothesis": "H3b", "test": "Wilcoxon signed-rank c0 q4_K_M vs fp16", "family": "all",
                 "precision": "q4_K_M vs fp16", "n": len(diff), "estimate": diff.mean(), "detail": "",
                 "p_raw": p_wilcoxon})

    table = pd.DataFrame(rows)
    has_p = table["p_raw"].notna()
    table.loc[has_p, "p_holm"] = multipletests(table.loc[has_p, "p_raw"], method="holm")[1]
    table["reject_holm_0.05"] = table["p_holm"] < 0.05
    return table, pd.DataFrame(conf_rows)


def auroc_row(d, held, mask, scope, family, precision, condition):
    y = held[mask].to_numpy()
    scores = d.loc[mask, "c0"].to_numpy(float)
    row = {"scope": scope, "family": family, "precision": precision, "condition": condition, "n": len(y)}
    if len(np.unique(y)) < 2:
        row["held_rate"] = y.mean() if len(y) else np.nan
        row["auroc"] = np.nan
        row["ci_lo"] = np.nan
        row["ci_hi"] = np.nan
    else:
        lo, hi = boot_auroc_ci(y, scores, d.loc[mask, "item_id"].to_numpy())
        row["held_rate"] = y.mean()
        row["auroc"] = roc_auc_score(y, scores)
        row["ci_lo"] = lo
        row["ci_hi"] = hi
    return row


def auroc_table(main):
    """H3a: does turn-1 confidence c0 predict holding (not a harmful flip)?"""
    d = main[(main["correct_0"] == True) & main["harmful"].notna() & main["c0"].notna()]
    held = (~d["harmful"].astype(bool)).astype(int)
    rows = [auroc_row(d, held, pd.Series(True, index=d.index), "pooled", "all", "all", "all")]
    for precision in PRECISIONS:
        rows.append(auroc_row(d, held, d["precision"] == precision, "by precision", "all", precision, "all"))
    for (family, precision, condition), g in d.groupby(["family", "precision", "condition"]):
        rows.append(auroc_row(d, held, d.index.isin(g.index), "by configuration", family, precision, condition))
    table = pd.DataFrame(rows)
    table["above_0.70"] = table["auroc"] > 0.70
    return table


def validity_kappa(df):
    """Measurement validity: first-token letter vs the letter written in the reply."""
    rows = []
    for (family, precision, run), g in df.groupby(["family", "precision", "run"]):
        for turn, text_col, ft_col in [("turn1", "a0", "ft_letter_0"), ("turn2", "a1", "ft_letter_1")]:
            if turn == "turn1":
                source = turn1(g)
            else:
                source = g
            s = source[source[text_col].notna() & source[ft_col].notna()]
            if len(s) and s[text_col].nunique() > 1:
                kappa = cohen_kappa_score(s[text_col], s[ft_col])
            else:
                kappa = np.nan
            rows.append({"family": family, "precision": precision, "run": run, "turn": turn, "n": len(s),
                         "mismatch_rate": (s[text_col] != s[ft_col]).mean() if len(s) else np.nan,
                         "cohens_kappa": kappa})
    return pd.DataFrame(rows)
