"""Poster figures: analysis/*.csv -> figures/*.svg (text smaller than 24 pt is refused).

    python scripts/make_figures.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so that `import cavein` works

import pandas as pd

from cavein.config import ANALYSIS_DIR, FIGURES_DIR
from cavein.figures import fig1_flip_rates, fig2_precision_effect, fig3_confidence


def main(argv=None):
    parser = argparse.ArgumentParser(description="analysis/*.csv -> figures/*.svg")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    folder = args.analysis_dir
    print(f"writing to {args.out_dir}")
    fig1_flip_rates(pd.read_csv(folder / "descriptives.csv"), args.out_dir)
    fig2_precision_effect(pd.read_csv(folder / "exploratory_precision_by_condition.csv"), args.out_dir)
    fig3_confidence(pd.read_csv(folder / "confidence_bins.csv"), pd.read_csv(folder / "auroc.csv"), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
