"""Small statistics helpers: bootstrap CIs, exact McNemar test, Wilson CI.

Bootstrap = we pick items at random (with replacement) 2000 times, compute the number
each time, and look at the middle 95% (or 90%) of those 2000 numbers.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

from cavein.config import N_BOOT, SEED


def bootstrap_indices(n, n_boot, seed):
    # a table with n_boot rows; every row is one random resample of the n items
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def percentile_interval(values, level):
    # the middle part of the bootstrap values, e.g. 2.5% to 97.5% for a 95% interval
    alpha = (1 - level) / 2
    lower_percent = 100 * alpha
    upper_percent = 100 * (1 - alpha)
    return tuple(np.nanpercentile(values, [lower_percent, upper_percent]))


def boot_ratio_ci(num, den, level=0.95, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI of sum(num) / sum(den). One value per item, and items are resampled."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    if len(num) == 0 or den.sum() == 0:
        return (np.nan, np.nan)

    idx = bootstrap_indices(len(num), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return percentile_interval(ratios, level)


def boot_diff_ci(x_sum, y_sum, n, level=0.95, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap CI of sum(x)/sum(n) - sum(y)/sum(n). Items are resampled."""
    x_sum = np.asarray(x_sum, float)
    y_sum = np.asarray(y_sum, float)
    n = np.asarray(n, float)
    if len(n) == 0:
        return (np.nan, np.nan)

    idx = bootstrap_indices(len(n), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        differences = (x_sum[idx].sum(axis=1) - y_sum[idx].sum(axis=1)) / n[idx].sum(axis=1)
    return percentile_interval(differences, level)


def boot_auroc_ci(y, score, clusters, n_boot=N_BOOT, seed=SEED):
    """95% bootstrap CI of the AUROC. We resample whole items (all rows of one item together)."""
    y = np.asarray(y)
    score = np.asarray(score, float)
    clusters = np.asarray(clusters)

    # for every item: the row numbers that belong to it
    unique_items = np.unique(clusters)
    rows_of_item = {}
    for item in unique_items:
        rows_of_item[item] = np.flatnonzero(clusters == item)

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        sampled_items = rng.choice(unique_items, len(unique_items))
        row_lists = []
        for item in sampled_items:
            row_lists.append(rows_of_item[item])
        rows = np.concatenate(row_lists)
        # the AUROC needs both classes (held and flipped) in the sample
        if len(np.unique(y[rows])) == 2:
            aucs.append(roc_auc_score(y[rows], score[rows]))

    if len(aucs) == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value. b and c are the two 'disagreeing' counts."""
    if b + c == 0:
        return 1.0
    table = [[0, b], [c, 0]]
    return float(mcnemar(table, exact=True).pvalue)


def wilson_ci(k, n, z=1.959964):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def compare(pairs, level=0.95):
    """Paired items (harmful_a, harmful_b): McNemar test, both flip rates and a CI of a - b."""
    a = pairs["harmful_a"].to_numpy()
    b = pairs["harmful_b"].to_numpy()
    n = len(pairs)

    # the four cells of the 2 x 2 table
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    neither = n - both - only_a - only_b

    # the bootstrap works on items, so first count per item
    per_item = pairs.groupby("item_id").agg(x=("harmful_a", "sum"), y=("harmful_b", "sum"),
                                            n=("harmful_a", "size"))
    lo, hi = boot_diff_ci(per_item["x"], per_item["y"], per_item["n"], level=level)

    if n > 0:
        rate_a = a.mean()
        rate_b = b.mean()
    else:
        rate_a = np.nan
        rate_b = np.nan

    result = {}
    result["n_pairs"] = n
    result["n_items"] = pairs["item_id"].nunique()
    result["both_harmful"] = both
    result["only_a"] = only_a
    result["only_b"] = only_b
    result["neither"] = neither
    result["hfr_a"] = rate_a
    result["hfr_b"] = rate_b
    result["diff_a_minus_b"] = rate_a - rate_b
    result["ci_lo"] = lo
    result["ci_hi"] = hi
    result["ci_level"] = level
    result["p_mcnemar_exact"] = mcnemar_p(only_a, only_b)
    return result
