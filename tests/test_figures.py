"""Tests for cavein/figures.py: every text in the saved SVG files is at least 24 pt."""
import re
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from cavein import figures

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"


def font_sizes(svg_text):
    return [float(size) for size in re.findall(r"font-size:\s*([\d.]+)px", svg_text)]


def test_all_figure_text_is_at_least_24_pt(tmp_path):
    figures.fig1_flip_rates(pd.read_csv(ANALYSIS / "descriptives.csv"), tmp_path)
    figures.fig2_precision_effect(pd.read_csv(ANALYSIS / "exploratory_precision_by_condition.csv"), tmp_path)
    figures.fig3_confidence(pd.read_csv(ANALYSIS / "confidence_bins.csv"), pd.read_csv(ANALYSIS / "auroc.csv"), tmp_path)
    for name in ["fig1_flip_rates.svg", "fig2_precision_effect.svg", "fig3_confidence.svg"]:
        svg = (tmp_path / name).read_text(encoding="utf-8")
        sizes = font_sizes(svg)
        assert sizes and min(sizes) >= 24, name
        assert "<image" not in svg  # vector only


def test_check_fonts_refuses_small_or_math_text():
    fig, ax = plt.subplots()
    ax.set_xlabel("small", fontsize=10)
    with pytest.raises(ValueError):
        figures.check_fonts(fig)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.set_xlabel("confidence $c_0$", fontsize=30)
    with pytest.raises(ValueError):
        figures.check_fonts(fig)
    plt.close(fig)
