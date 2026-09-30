"""Exploratory analyses: NOT pre-registered, and labelled 'exploratory' in the CSV name."""
import pandas as pd

from cavein.config import CONDITIONS, FAMILIES
from cavein.results import paired
from cavein.stats import compare

# (precision compared with fp16, CI level): 95% for q4_K_M like H1a, 90% for q8_0 like H1b
COMPARISONS = [("q4_K_M", 0.95), ("q8_0", 0.90)]
 
# short family names: the row labels of the leave-one-family-out table (and of its figure)
SHORT_NAMES = {"llama3.2-3b": "Llama 3.2", "qwen2.5-3b": "Qwen 3B", "phi4-mini-3.8b": "Phi-4-mini",
               "qwen2.5-7b": "Qwen 7B", "llama3.1-8b": "Llama 3.1", "phi4-14b": "Phi-4 14B"}

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
def leave_one_family_out(main):
    """q4_K_M vs fp16 on the `user` follow-up: all six families, then with each family left out in turn.
 
    Same pairs, same test and same bootstrap CI (95%) as H1a, so the `all six` row is the H1a row.
    Exploratory: it only shows whether one family drives the pooled difference.
    """
    pairs = paired(main, "user", "q4_K_M", "fp16")
    cases = [("all six", pairs)]
    for family in FAMILIES:
        cases.append((f"\u2212 {SHORT_NAMES[family]}", pairs[pairs["family"] != family]))
 
    rows = []
    for label, case_pairs in cases:
        row = {"left_out": label, "role": "exploratory"}
        row.update(compare(case_pairs, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)
 