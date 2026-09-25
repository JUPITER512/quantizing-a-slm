"""Small statistics helpers: bootstrap CIs, exact McNemar, Wilson CI, ECE, Cohen's h."""
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

from cavein.config import ECE_BINS, N_BOOT, SEED


def cohens_h(p1, p2):
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def ece(conf, correct, bins=ECE_BINS):
    """Expected calibration error with equal-width bins."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def bootstrap_indices(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def boot_ratio_ci(num, den, level=0.95, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI of sum(num) / sum(den), resampling items (one value per item)."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    if len(num) == 0 or den.sum() == 0:
        return (np.nan, np.nan)
    idx = bootstrap_indices(len(num), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    alpha = (1 - level) / 2
    return tuple(np.nanpercentile(stats, [100 * alpha, 100 * (1 - alpha)]))


def boot_diff_ci(x_sum, y_sum, n, level=0.95, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap CI of sum(x)/sum(n) - sum(y)/sum(n), resampling items."""
    x_sum = np.asarray(x_sum, float)
    y_sum = np.asarray(y_sum, float)
    n = np.asarray(n, float)
    if len(n) == 0:
        return (np.nan, np.nan)
    idx = bootstrap_indices(len(n), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = (x_sum[idx].sum(axis=1) - y_sum[idx].sum(axis=1)) / n[idx].sum(axis=1)
    alpha = (1 - level) / 2
    return tuple(np.nanpercentile(stats, [100 * alpha, 100 * (1 - alpha)]))


def boot_auroc_ci(y, score, clusters, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    score = np.asarray(score, float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    rows_of = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        sample = rng.choice(unique, len(unique))
        rows = np.concatenate([rows_of[c] for c in sample])
        if len(np.unique(y[rows])) == 2:
            aucs.append(roc_auc_score(y[rows], score[rows]))
    if not aucs:
        return (np.nan, np.nan)
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts."""
    if b + c == 0:
        return 1.0
    return float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)


def wilson_ci(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def compare(pairs, level=0.95):
    """For paired items (harmful_a, harmful_b): McNemar, both rates, Cohen's h and a CI of a - b."""
    a = pairs["harmful_a"].to_numpy()
    b = pairs["harmful_b"].to_numpy()
    n = len(pairs)
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    per_item = pairs.groupby("item_id").agg(x=("harmful_a", "sum"), y=("harmful_b", "sum"), n=("harmful_a", "size"))
    lo, hi = boot_diff_ci(per_item["x"], per_item["y"], per_item["n"], level=level)
    if n:
        rate_a = a.mean()
        rate_b = b.mean()
        h = cohens_h(rate_a, rate_b)
    else:
        rate_a = rate_b = h = np.nan
    return {"n_pairs": n, "n_items": pairs["item_id"].nunique(), "both_harmful": both,
            "only_a": only_a, "only_b": only_b, "neither": n - both - only_a - only_b,
            "hfr_a": rate_a, "hfr_b": rate_b, "diff_a_minus_b": rate_a - rate_b, "ci_lo": lo, "ci_hi": hi,
            "ci_level": level, "cohens_h": h, "p_mcnemar_exact": mcnemar_p(only_a, only_b)}
