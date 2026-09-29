"""Statistics for the poster: results/*.jsonl -> analysis/*.csv.

Pre-registered tests, descriptive tables, robustness checks and exploratory tables
(the file names say which is which).

    python scripts/analyze.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so that `import cavein` works

from cavein.config import ANALYSIS_DIR, ITEMS_FILE, RESULTS_DIR, TOST_MARGIN
from cavein.descriptive import confidence_bins, descriptives, overview, size_comparison
from cavein.exploratory import precision_by_condition
from cavein.hypotheses import auroc_table, gee_models, primary_and_h1b, secondary, validity_kappa
from cavein.results import load_results, main_data
from cavein.robustness import controls, determinism
from cavein.summary import hypothesis_summary


def write(table, out_dir, name):
    table.to_csv(out_dir / name, index=False, encoding="utf-8", float_format="%.6g", lineterminator="\n")
    print(f"  {name:<40} {len(table):>4} rows")


def main(argv=None):
    parser = argparse.ArgumentParser(description="results/*.jsonl -> analysis/*.csv")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # step 1: read all result files (df = everything, main_df = only the main runs)
    df = load_results(args.results_dir)
    main_df = main_data(df)
    print(f"loaded {len(df)} records; main runs: {main_df['model'].nunique()} configurations")

    # step 2: the tests and tables
    desc = descriptives(main_df)                                  # flip rates per configuration
    primary, h1b = primary_and_h1b(main_df)                       # H1a and H1b
    gee, p_interaction, gee_notes = gee_models(main_df)           # H2b
    sec, conf = secondary(main_df, p_interaction)                 # H2a, H2b, H3b with Holm
    auroc = auroc_table(main_df)                                  # H3a
    kappa = validity_kappa(df[df["run"] == "main"])               # measurement validity
    bins = confidence_bins(main_df)                               # for Figure 3
    summary = hypothesis_summary(primary, h1b, sec, auroc, conf, gee, kappa)

    # step 3: write every table as a CSV file
    tables = [
        ("hypothesis_summary.csv", summary),
        ("overview.csv", overview(df, args.items)),
        ("descriptives.csv", desc),
        ("primary_mcnemar.csv", primary),
        ("h1b_equivalence.csv", h1b),
        ("gee_coefficients.csv", gee),
        ("secondary_holm.csv", sec),
        ("confidence_q4_fp16.csv", conf),
        ("auroc.csv", auroc),
        ("validity_kappa.csv", kappa),
        ("confidence_bins.csv", bins),
        ("descriptive_size_comparison.csv", size_comparison(desc)),
        ("robustness_controls.csv", controls(df)),
        ("robustness_determinism.csv", determinism(df)),
        ("exploratory_precision_by_condition.csv", precision_by_condition(main_df)),
    ]
    print(f"writing to {args.out_dir}")
    for name, table in tables:
        write(table, args.out_dir, name)

    if gee_notes:
        print("GEE warnings (see gee_coefficients.csv):")
        for note in sorted(set(gee_notes))[:5]:
            print("  " + note)
    p = primary.iloc[0]
    print(f"\nPRIMARY (pooled, user, q4_K_M vs fp16): n={p.n_pairs} pairs, HFR {p.hfr_a:.3f} vs {p.hfr_b:.3f}, "
          f"q4 only {p.only_a}, fp16 only {p.only_b}, exact McNemar p={p.p_mcnemar_exact:.4g}")
    e = h1b.iloc[0]
    print(f"H1b (q8_0 - fp16): {e.diff_a_minus_b:+.3f}, 90% CI [{e.ci_lo:+.3f}, {e.ci_hi:+.3f}], "
          f"equivalent within ±{TOST_MARGIN:.2f}: {bool(e.equivalent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
