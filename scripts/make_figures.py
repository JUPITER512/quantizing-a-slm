"""Poster figures: analysis/*.csv -> figures/*.svg.

Every figure is drawn at its final size on the A1 poster, so the font sizes are the printed
sizes. A figure with text smaller than 24 pt is not saved.

    python scripts/make_figures.py
"""
import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent.parent
MM = 1 / 25.4
MIN_PT = 24

# (width, height) in mm on the poster
SIZE_FIG1 = (539, 150)
SIZE_FIG2 = (263, 142)
SIZE_FIG3 = (263, 142)

PRECISIONS = ["q4_K_M", "q8_0", "fp16"]
COLOURS = {"q4_K_M": "#0072B2", "q8_0": "#007A5E", "fp16": "#8E4A8C"}
MARKERS = {"q4_K_M": "o", "q8_0": "s", "fp16": "^"}
FAMILIES = {"llama3.2-3b": "Llama 3.2 3B", "qwen2.5-3b": "Qwen2.5 3B", "phi4-mini-3.8b": "Phi-4-mini 3.8B",
            "qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "phi4-14b": "Phi-4 14B"}
CONDITIONS = ["reask", "speaker_free", "user", "expert"]
GREY = "#555555"
LIGHT = "#E9E9E9"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 26,
    "axes.titlesize": 28, "axes.labelsize": 26, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "legend.fontsize": 24, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 7, "ytick.major.size": 7, "svg.fonttype": "none", "svg.hashsalt": "cave-in",
})


def check_fonts(fig):
    too_small = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip() and text.get_fontsize() < MIN_PT:
            too_small.append((text.get_text(), text.get_fontsize()))
    if too_small:
        raise ValueError(f"text below {MIN_PT} pt: {too_small[:5]}")


def save(fig, out_dir, name):
    check_fonts(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    plt.close(fig)
    # always LF line endings, so the files are the same on every operating system
    (out_dir / name).write_bytes(buffer.getvalue().replace(b"\r\n", b"\n"))
    print(f"  {name}")


def face_colour(precision):
    # fp16 uses open markers, so the three precisions differ by shape and fill, not only by colour
    if precision == "fp16":
        return "white"
    return COLOURS[precision]


def fig1_flip_rates(desc, out_dir):
    """Harmful flip rate per follow-up and precision, one panel per model family."""
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(SIZE_FIG1[0] * MM, SIZE_FIG1[1] * MM), sharey=True,
                             layout="constrained")
    offsets = {"q4_K_M": -0.24, "q8_0": 0.0, "fp16": 0.24}
    labels = {"reask": "re-ask", "speaker_free": "speaker-free", "user": "user", "expert": "expert"}
    for ax, (family, title) in zip(axes.flat, FAMILIES.items()):
        ax.axhspan(1.5, 2.5, color=LIGHT, zorder=0)
        for precision in PRECISIONS:
            d = desc[(desc.family == family) & (desc.precision == precision)]
            d = d.set_index("condition").reindex(CONDITIONS)
            y = [i + offsets[precision] for i in range(len(CONDITIONS))]
            x = 100 * d["hfr"]
            err = [100 * (d["hfr"] - d["hfr_ci_lo"]), 100 * (d["hfr_ci_hi"] - d["hfr"])]
            ax.errorbar(x, y, xerr=err, fmt=MARKERS[precision], color=COLOURS[precision], ms=11, mew=2,
                        mfc=face_colour(precision), elinewidth=2.2, capsize=4, zorder=3)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_yticks(range(len(CONDITIONS)), [labels[c] for c in CONDITIONS])
        ax.set_ylim(3.55, -0.55)
        ax.set_xlim(-6, 106)
        ax.set_xticks([0, 50, 100])
        ax.grid(axis="x", color=LIGHT, lw=1.2, zorder=0)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("harmful flip rate (%)", fontsize=26)

    handles = []
    for precision in PRECISIONS:
        handles.append(Line2D([], [], marker=MARKERS[precision], color=COLOURS[precision],
                              mfc=face_colour(precision), mew=2, ms=11, lw=0, label=precision))
    handles.append(Patch(color=LIGHT, label="user = pre-registered primary follow-up"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, handletextpad=0.3,
               columnspacing=1.4)
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.04, wspace=0.03)
    save(fig, out_dir, "fig1_flip_rates.svg")


def fig2_precision_effect(table, out_dir):
    """Difference to fp16 per follow-up, pooled over the six families."""
    d = table[table.scope == "pooled over six families"]
    fig, ax = plt.subplots(figsize=(SIZE_FIG2[0] * MM, SIZE_FIG2[1] * MM), layout="constrained")
    ax.axvspan(-3, 3, color=LIGHT, zorder=0)
    ax.axvline(0, color=GREY, lw=1.6, zorder=1)
    labels = {"reask": "re-ask", "speaker_free": "speaker-free", "user": "user\n(pre-registered)", "expert": "expert"}
    comparisons = [("q4_K_M vs fp16", "q4_K_M", -0.16, "95%"), ("q8_0 vs fp16", "q8_0", 0.16, "90%")]
    for comparison, precision, shift, level in comparisons:
        rows = d[d.comparison == comparison].set_index("condition").reindex(CONDITIONS)
        y = [i + shift for i in range(len(CONDITIONS))]
        x = 100 * rows["diff_a_minus_b"]
        err = [100 * (rows["diff_a_minus_b"] - rows["ci_lo"]), 100 * (rows["ci_hi"] - rows["diff_a_minus_b"])]
        ax.errorbar(x, y, xerr=err, fmt=MARKERS[precision], color=COLOURS[precision], ms=13, mew=2, elinewidth=2.4,
                    capsize=5, zorder=3, label=f"{precision} − fp16 ({level} CI)")
    ax.set_yticks(range(len(CONDITIONS)), [labels[c] for c in CONDITIONS])
    ax.invert_yaxis()
    ax.set_xlim(-8.5, 8.5)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xlabel("difference in harmful flip rate (pp)")
    ax.grid(axis="x", color=LIGHT, lw=1.2, zorder=0)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(color=LIGHT, label="±3 pp equivalence margin"))
    fig.legend(handles=handles, loc="outside upper center", ncol=1, frameon=False, handletextpad=0.4)
    save(fig, out_dir, "fig2_precision_effect.svg")


def fig3_confidence(bins, auroc, out_dir):
    """Harmful flip rate per turn-1 confidence bin and precision; AUROC in the legend."""
    fig, ax = plt.subplots(figsize=(SIZE_FIG3[0] * MM, SIZE_FIG3[1] * MM), layout="constrained")
    offsets = {"q4_K_M": -0.14, "q8_0": 0.0, "fp16": 0.14}
    auroc_of = auroc[auroc.scope == "by precision"].set_index("precision")["auroc"]
    order = bins.drop_duplicates("bin_index").sort_values("bin_index")
    for precision in PRECISIONS:
        d = bins[bins.precision == precision].sort_values("bin_index")
        x = d["bin_index"] + offsets[precision]
        err = [100 * (d["flip_rate"] - d["ci_lo"]), 100 * (d["ci_hi"] - d["flip_rate"])]
        ax.errorbar(x, 100 * d["flip_rate"], yerr=err, fmt="-" + MARKERS[precision], color=COLOURS[precision],
                    ms=12, mew=2, mfc=face_colour(precision), lw=2, elinewidth=2, capsize=4,
                    label=f"{precision} (AUROC {auroc_of[precision]:.2f})")
    ax.set_xticks(order["bin_index"], order["bin"])
    ax.set_xlabel("turn-1 confidence $c_0$ in the correct answer")
    ax.set_ylabel("harmful flip rate (%)")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=LIGHT, lw=1.2)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    save(fig, out_dir, "fig3_confidence.svg")


def main(argv=None):
    parser = argparse.ArgumentParser(description="analysis/*.csv -> figures/*.svg")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
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
