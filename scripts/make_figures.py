"""Poster figures: analysis/*.csv -> figures/*.svg.

    python scripts/make_figures.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so that `import cavein` works

import pandas as pd

from cavein.config import ANALYSIS_DIR, FIGURES_DIR
from cavein.figures import fig1_flip_rates, fig2_precision_effect, fig3_confidence,fig_user_bars,fig_leave_one_out_bars,fig_precision_bars


def main(argv=None):
    parser = argparse.ArgumentParser(description="analysis/*.csv -> figures/*.svg")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    folder = args.analysis_dir


    desc = pd.read_csv(folder / "descriptives.csv")
    precision_table = pd.read_csv(folder / "exploratory_precision_by_condition.csv")


    print(f"writing to {args.out_dir}")
    fig1_flip_rates(pd.read_csv(folder / "descriptives.csv"), args.out_dir)
    fig2_precision_effect(pd.read_csv(folder / "exploratory_precision_by_condition.csv"), args.out_dir)
    fig3_confidence(pd.read_csv(folder / "confidence_bins.csv"), pd.read_csv(folder / "auroc.csv"), args.out_dir)
    fig_user_bars(desc, args.out_dir)
    fig_precision_bars(precision_table, args.out_dir)
    loo_file = folder / "exploratory_leave_one_family_out.csv"   # made by scripts/analyze.py
    if loo_file.exists():
        fig_leave_one_out_bars(pd.read_csv(loo_file), args.out_dir)
    else:
        print(f"  skipped fig_leave_one_out_bars: run scripts/analyze.py first ({loo_file.name} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
