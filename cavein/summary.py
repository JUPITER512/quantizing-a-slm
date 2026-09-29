"""hypothesis_summary.csv: one row per hypothesis, with its result and whether the pre-registered rule is met."""
import numpy as np
import pandas as pd


def pct(x):
    # 0.597 -> '59.7'
    return f"{100 * x:.1f}"


def format_p(x):
    # 0.0001 -> '< .001', 0.042 -> '= .042'
    if x < 0.001:
        return "< .001"
    return f"= {x:.3f}".replace("0.", ".")


def yes_no(rule_is_met):
    if rule_is_met:
        return "yes"
    return "no"


def h1a_row(primary):
    p = primary.iloc[0]  # the pooled row
    return {
        "hypothesis": "H1a",
        "claim": "harmful flip rate q4_K_M > fp16 (user; pooled, six families)",
        "test": "exact McNemar, two-sided",
        "estimate": f"{pct(p.hfr_a)} % vs {pct(p.hfr_b)} % (diff {100 * p.diff_a_minus_b:+.1f} pp)",
        "interval": f"95% CI {100 * p.ci_lo:+.1f} to {100 * p.ci_hi:+.1f} pp",
        "p": p.p_mcnemar_exact,
        "n": f"{p.n_pairs} pairs ({p.only_a} vs {p.only_b} discordant)",
        "rule": "p < .05 and q4_K_M higher",
        "rule_met": yes_no(bool(p.supported)),
        "prediction_short": "q4_K_M flips more than fp16 (user)",
        "test_short": "exact McNemar",
        "result_short": f"{pct(p.hfr_a)} vs {pct(p.hfr_b)} % ({100 * p.diff_a_minus_b:+.1f} pp), "
                        f"p {format_p(p.p_mcnemar_exact)}",
    }


def h1b_row(h1b):
    e = h1b.iloc[0]  # the pooled row
    return {
        "hypothesis": "H1b",
        "claim": "q8_0 and fp16 equivalent (user; pooled)",
        "test": "paired bootstrap 90% CI (TOST)",
        "estimate": f"diff {100 * e.diff_a_minus_b:+.1f} pp",
        "interval": f"90% CI {100 * e.ci_lo:+.1f} to {100 * e.ci_hi:+.1f} pp",
        "p": np.nan,
        "n": f"{e.n_pairs} pairs",
        "rule": "CI inside ±3 pp",
        "rule_met": yes_no(bool(e.equivalent)),
        "prediction_short": "q8_0 ≈ fp16 within ±3 pp (user)",
        "test_short": "paired bootstrap, 90% CI",
        "result_short": f"{100 * e.diff_a_minus_b:+.1f} pp [{100 * e.ci_lo:+.1f}, {100 * e.ci_hi:+.1f}]",
    }


def h2a_row(sec):
    h2a = sec[sec.hypothesis == "H2a"]
    # count the configurations where the difference is significant after Holm, and its direction
    higher = int(((h2a.p_holm < 0.05) & (h2a.estimate > 0)).sum())
    lower = int(((h2a.p_holm < 0.05) & (h2a.estimate < 0)).sum())
    if higher == len(h2a):
        p_value = h2a.p_holm.max()
    else:
        p_value = np.nan
    return {
        "hypothesis": "H2a",
        "claim": "speaker_free flips more than reask",
        "test": "exact McNemar per configuration, Holm",
        "estimate": f"higher in {higher} of {len(h2a)} configurations, lower in {lower}",
        "interval": "",
        "p": p_value,
        "n": f"{len(h2a)} configurations",
        "rule": "significant and higher in most configurations",
        "rule_met": yes_no(higher > len(h2a) / 2),
        "prediction_short": "speaker-free flips more than re-ask",
        "test_short": "McNemar per config., Holm",
        "result_short": f"higher in {higher}/{len(h2a)}, lower in {lower} (p_Holm < .05)",
    }


def h2b_row(sec):
    h2b = sec[sec.hypothesis == "H2b"].iloc[0]
    return {
        "hypothesis": "H2b",
        "claim": "precision x condition interaction",
        "test": "GEE (item clusters), joint Wald, Holm",
        "estimate": "",
        "interval": "",
        "p": h2b.p_holm,
        "n": "",
        "rule": "interaction significant (Holm)",
        "rule_met": yes_no(h2b.p_holm < 0.05),
        "prediction_short": "precision × follow-up interaction",
        "test_short": "GEE, joint Wald, Holm",
        "result_short": f"p_Holm {format_p(h2b.p_holm)}",
    }


def h3a_row(auroc):
    a = auroc[auroc.scope == "pooled"].iloc[0]
    return {
        "hypothesis": "H3a",
        "claim": "turn-1 confidence predicts holding",
        "test": "AUROC, item bootstrap",
        "estimate": f"AUROC {a.auroc:.2f}",
        "interval": f"95% CI {a.ci_lo:.2f} to {a.ci_hi:.2f}",
        "p": np.nan,
        "n": f"{int(a.n)} turn-2 answers",
        "rule": "AUROC > 0.70 and CI above 0.5",
        "rule_met": yes_no(a.auroc > 0.70 and a.ci_lo > 0.5),
        "prediction_short": "turn-1 confidence predicts holding",
        "test_short": "AUROC > .70, bootstrap",
        "result_short": f"AUROC {a.auroc:.2f} [{a.ci_lo:.2f}, {a.ci_hi:.2f}]",
    }


def h3b_row(sec, conf):
    c = conf.iloc[0]  # the pooled row
    h3b = sec[sec.hypothesis == "H3b"].iloc[0]
    return {
        "hypothesis": "H3b",
        "claim": "turn-1 confidence lower at q4_K_M than fp16",
        "test": "Wilcoxon signed-rank, Holm; bootstrap",
        "estimate": f"mean diff {c.mean_diff_q4_minus_fp16:+.4f}",
        "interval": f"95% CI {c.ci_lo:+.4f} to {c.ci_hi:+.4f}",
        "p": h3b.p_holm,
        "n": f"{int(c.n_pairs)} pairs",
        "rule": "CI below 0 and p < .05",
        "rule_met": yes_no(c.ci_hi < 0 and h3b.p_holm < 0.05),
        "prediction_short": "confidence lower at q4_K_M than fp16",
        "test_short": "Wilcoxon, Holm; bootstrap",
        "result_short": f"{c.mean_diff_q4_minus_fp16:+.4f} [{c.ci_lo:+.4f}, {c.ci_hi:+.4f}], "
                        f"p_Holm {format_p(h3b.p_holm)}",
    }


def validity_row(kappa):
    k = kappa[kappa.run == "main"]
    # rounded down to 3 decimals, so 'kappa >= ...' is always true
    kappa_min = np.floor(1000 * k.cohens_kappa.min()) / 1000
    return {
        "hypothesis": "Validity",
        "claim": "first-token letter agrees with text letter",
        "test": "Cohen's kappa per configuration and turn",
        "estimate": f"kappa {k.cohens_kappa.min():.3f} to {k.cohens_kappa.max():.3f}",
        "interval": f"mismatch {100 * k.mismatch_rate.max():.1f} % at most",
        "p": np.nan,
        "n": f"{len(k)} configuration x turn",
        "rule": "reported, no threshold",
        "rule_met": "n/a",
        "prediction_short": "first-token letter = text letter",
        "test_short": "Cohen's κ per config.",
        "result_short": f"κ ≥ {kappa_min:.3f}; mismatch ≤ {100 * k.mismatch_rate.max():.1f} %",
    }


def hypothesis_summary(primary, h1b, sec, auroc, conf, gee, kappa):
    # `gee` is not needed here (its p-value is already in `sec`); it stays so the call in analyze.py is unchanged
    rows = []
    rows.append(h1a_row(primary))
    rows.append(h1b_row(h1b))
    rows.append(h2a_row(sec))
    rows.append(h2b_row(sec))
    rows.append(h3a_row(auroc))
    rows.append(h3b_row(sec, conf))
    rows.append(validity_row(kappa))
    return pd.DataFrame(rows)
