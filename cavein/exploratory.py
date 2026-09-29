"""Exploratory analyses: NOT pre-registered, and labelled 'exploratory' in the CSV name."""
import pandas as pd

from cavein.config import CONDITIONS, FAMILIES
from cavein.results import paired
from cavein.stats import compare

# (precision compared with fp16, CI level): 95% for q4_K_M like H1a, 90% for q8_0 like H1b
COMPARISONS = [("q4_K_M", 0.95), ("q8_0", 0.90)]


def precision_by_condition(main):
    """q4_K_M vs fp16 and q8_0 vs fp16 for every follow-up, pooled and per family.

    The `user` rows repeat the pre-registered tests; the other follow-ups are exploratory.
    """
    rows = []
    for prec_a, level in COMPARISONS:
        for condition in CONDITIONS:
            pairs = paired(main, condition, prec_a, "fp16")
            if condition == "user":
                role = "pre-registered (see primary/h1b)"
            else:
                role = "exploratory"

            # pooled over the six families
            row = {"comparison": f"{prec_a} vs fp16", "condition": condition,
                   "scope": "pooled over six families", "role": role}
            row.update(compare(pairs, level))
            rows.append(row)

            # each family on its own
            for family in FAMILIES:
                family_pairs = pairs[pairs["family"] == family]
                row = {"comparison": f"{prec_a} vs fp16", "condition": condition, "scope": family,
                       "role": "per family (descriptive)"}
                row.update(compare(family_pairs, level))
                rows.append(row)
    return pd.DataFrame(rows)
