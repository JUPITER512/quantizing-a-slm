"""Poster figures: analysis/*.csv -> figures/*.svg (step 4 of the pipeline).

Reads only the CSV files written by analyze.py; no statistics here. Every figure is drawn at
its final size on the DIN A1 poster (sizes in mm below), so font sizes in points are the
printed sizes; the script refuses to save a figure with any text smaller than 24 pt.
Precision colours and markers are fixed; colour is never the only cue (marker shapes differ).

    python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MM = 1 / 25.4
MIN_PT = 24

# final print sizes on the A1 poster, landscape (width, height) in mm
SIZE_FIG1 = (539, 150)
SIZE_FIG2 = (263, 142)
SIZE_FIG3 = (263, 142)

PRECISIONS = ["q4_K_M", "q8_0", "fp16"]
COLOURS = {"q4_K_M": "#0072B2", "q8_0": "#007A5E", "fp16": "#8E4A8C"}
MARKERS = {"q4_K_M": "o", "q8_0": "s", "fp16": "^"}
FAMILIES = {"llama3.2-3b": "Llama 3.2 3B", "qwen2.5-3b": "Qwen2.5 3B", "phi4-mini-3.8b": "Phi-4-mini 3.8B",
            "qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "phi4-14b": "Phi-4 14B"}
CONDITIONS = ["reask", "speaker_free", "user", "expert"]
COND_LABELS = {"reask": "re-ask", "speaker_free": "speaker-\nfree", "user": "user", "expert": "expert"}
GREY, LIGHT = "#555555", "#E9E9E9"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 26,
    "axes.titlesize": 28, "axes.labelsize": 26, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "legend.fontsize": 24, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 7, "ytick.major.size": 7, "svg.fonttype": "none", "svg.hashsalt": "cave-in",
})


def check_fonts(fig) -> None:
    """Raise if any visible text is smaller than MIN_PT at print size."""
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT]
    if small:
        raise ValueError(f"text below {MIN_PT} pt: {small[:5]}")


def save(fig, out_dir: Path, name: str) -> None:
    """Write the SVG with LF line endings so rebuilds are byte-identical on every OS."""
    check_fonts(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    (out_dir / name).write_bytes(buf.getvalue().replace(b"\r\n", b"\n"))
    print(f"  {name}")


def fig1_flip_rates(desc: pd.DataFrame, out_dir: Path) -> None:
    """Harmful flip rate per follow-up and precision, one panel per family in one row (95% CI)."""
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(SIZE_FIG1[0] * MM, SIZE_FIG1[1] * MM), sharey=True,
                             layout="constrained")
    offsets = {"q4_K_M": -0.24, "q8_0": 0.0, "fp16": 0.24}
    ylab = {"reask": "re-ask", "speaker_free": "speaker-free", "user": "user", "expert": "expert"}
    for ax, (fam, label) in zip(axes.flat, FAMILIES.items()):
        ax.axhspan(1.5, 2.5, color=LIGHT, zorder=0)
        for prec in PRECISIONS:
            d = desc[(desc.family == fam) & (desc.precision == prec)].set_index("condition").reindex(CONDITIONS)
            y = [i + offsets[prec] for i in range(len(CONDITIONS))]
            x = 100 * d["hfr"]
            err = [100 * (d["hfr"] - d["hfr_ci_lo"]), 100 * (d["hfr_ci_hi"] - d["hfr"])]
            ax.errorbar(x, y, xerr=err, fmt=MARKERS[prec], color=COLOURS[prec], ms=11, mew=2,
                        mfc=COLOURS[prec] if prec != "fp16" else "white", elinewidth=2.2, capsize=4, zorder=3)
        ax.set_title(label, fontweight="bold", pad=8)
        ax.set_yticks(range(len(CONDITIONS)), [ylab[c] for c in CONDITIONS])
        ax.set_ylim(3.55, -0.55)
        ax.set_xlim(-6, 106)
        ax.set_xticks([0, 50, 100])
        ax.grid(axis="x", color=LIGHT, lw=1.2, zorder=0)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("harmful flip rate (%)", fontsize=26)
    handles = [Line2D([], [], marker=MARKERS[p], color=COLOURS[p], mfc=COLOURS[p] if p != "fp16" else "white",
                      mew=2, ms=11, lw=0, label=p) for p in PRECISIONS]
    handles.append(Patch(color=LIGHT, label="user = pre-registered primary follow-up"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, handletextpad=0.3,
               columnspacing=1.4)
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.04, wspace=0.03)
    save(fig, out_dir, "fig1_flip_rates.svg")


def fig2_precision_effect(prec: pd.DataFrame, out_dir: Path) -> None:
    """Difference to fp16 per follow-up, pooled over the six families (q4: 95% CI, q8: 90% CI)."""
    d = prec[prec.scope == "pooled over six families"]
    fig, ax = plt.subplots(figsize=(SIZE_FIG2[0] * MM, SIZE_FIG2[1] * MM), layout="constrained")
    ax.axvspan(-3, 3, color=LIGHT, zorder=0)
    ax.axvline(0, color=GREY, lw=1.6, zorder=1)
    ylab = {"reask": "re-ask", "speaker_free": "speaker-free", "user": "user\n(pre-registered)", "expert": "expert"}
    for comp, prec_key, dy, level in (("q4_K_M vs fp16", "q4_K_M", -0.16, "95%"), ("q8_0 vs fp16", "q8_0", 0.16, "90%")):
        rows = d[d.comparison == comp].set_index("condition").reindex(CONDITIONS)
        y = [i + dy for i in range(len(CONDITIONS))]
        x = 100 * rows["diff_a_minus_b"]
        err = [100 * (rows["diff_a_minus_b"] - rows["ci_lo"]), 100 * (rows["ci_hi"] - rows["diff_a_minus_b"])]
        ax.errorbar(x, y, xerr=err, fmt=MARKERS[prec_key], color=COLOURS[prec_key], ms=13, mew=2, elinewidth=2.4,
                    capsize=5, zorder=3, label=f"{prec_key} − fp16 ({level} CI)")
    ax.set_yticks(range(len(CONDITIONS)), [ylab[c] for c in CONDITIONS])
    ax.invert_yaxis()
    ax.set_xlim(-8.5, 8.5)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xlabel("difference in harmful flip rate (pp)")
    ax.grid(axis="x", color=LIGHT, lw=1.2, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=LIGHT, label="±3 pp equivalence margin"))
    fig.legend(handles=handles, loc="outside upper center", ncol=1, frameon=False, handletextpad=0.4)
    save(fig, out_dir, "fig2_precision_effect.svg")


def fig3_confidence(bins: pd.DataFrame, auroc: pd.DataFrame, out_dir: Path) -> None:
    """Harmful flip rate by turn-1 confidence bin, per precision (Wilson 95% CI); AUROC in the legend."""
    fig, ax = plt.subplots(figsize=(SIZE_FIG3[0] * MM, SIZE_FIG3[1] * MM), layout="constrained")
    offsets = {"q4_K_M": -0.14, "q8_0": 0.0, "fp16": 0.14}
    au = auroc[auroc.scope == "by precision"].set_index("precision")["auroc"]
    order = bins.drop_duplicates("bin_index").sort_values("bin_index")
    for prec in PRECISIONS:
        d = bins[bins.precision == prec].sort_values("bin_index")
        x = d["bin_index"] + offsets[prec]
        err = [100 * (d["flip_rate"] - d["ci_lo"]), 100 * (d["ci_hi"] - d["flip_rate"])]
        ax.errorbar(x, 100 * d["flip_rate"], yerr=err, fmt="-" + MARKERS[prec], color=COLOURS[prec], ms=12, mew=2,
                    mfc=COLOURS[prec] if prec != "fp16" else "white", lw=2, elinewidth=2, capsize=4,
                    label=f"{prec} (AUROC {au[prec]:.2f})")
    ax.set_xticks(order["bin_index"], order["bin"])
    ax.set_xlabel("turn-1 confidence $c_0$ in the correct answer")
    ax.set_ylabel("harmful flip rate (%)")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=LIGHT, lw=1.2)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    save(fig, out_dir, "fig3_confidence.svg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    a = args.analysis_dir
    print(f"writing to {args.out_dir}")
    fig1_flip_rates(pd.read_csv(a / "descriptives.csv"), args.out_dir)
    fig2_precision_effect(pd.read_csv(a / "exploratory_precision_by_condition.csv"), args.out_dir)
    fig3_confidence(pd.read_csv(a / "confidence_bins.csv"), pd.read_csv(a / "auroc.csv"), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
