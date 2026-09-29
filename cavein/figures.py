"""The three poster figures, drawn at their final size on the A1 poster (font sizes = printed sizes).

This file only draws. All numbers come from the CSV tables made by analyze.py.
"""
import io

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from cavein.config import CONDITIONS, PRECISIONS

MM = 1 / 25.4   # matplotlib wants inches; 1 mm = 1/25.4 inch
MIN_PT = 24     # exam rule: no text smaller than 24 pt

# (width, height) in mm on the poster
SIZE_FIG1 = (539, 150)
SIZE_FIG2 = (263, 142)
SIZE_FIG3 = (263, 142)

# colour AND marker shape per precision, so colour is never the only cue
COLOURS = {"q4_K_M": "#0072B2", "q8_0": "#007A5E", "fp16": "#8E4A8C"}
MARKERS = {"q4_K_M": "o", "q8_0": "s", "fp16": "^"}
FAMILY_NAMES = {"llama3.2-3b": "Llama 3.2 3B", "qwen2.5-3b": "Qwen2.5 3B", "phi4-mini-3.8b": "Phi-4-mini 3.8B",
                "qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B", "phi4-14b": "Phi-4 14B"}
GREY = "#555555"
LIGHT = "#E9E9E9"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 26,
    "axes.titlesize": 28, "axes.labelsize": 26, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "legend.fontsize": 24, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "svg.fonttype": "none",      # keep text as text in the SVG
    "svg.hashsalt": "cave-in",   # fixed ids, so the SVG is the same every time
})


def check_fonts(fig):
    """Stop with an error if any text is smaller than 24 pt."""
    too_small = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        # math text ($...$) is not allowed: its subscripts are drawn smaller than the font size
        if text.get_fontsize() < MIN_PT or "$" in text.get_text():
            too_small.append((text.get_text(), text.get_fontsize()))
    if too_small:
        raise ValueError(f"text below {MIN_PT} pt or math text: {too_small[:5]}")


def save(fig, out_dir, name):
    check_fonts(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})  # no date, so the file does not change
    plt.close(fig)
    # always LF line endings, so the files are the same on every operating system
    svg_bytes = buffer.getvalue().replace(b"\r\n", b"\n")
    (out_dir / name).write_bytes(svg_bytes)
    print(f"  {name}")


def face_colour(precision):
    # fp16 uses open (white) markers, so the three precisions differ by shape and fill, not only by colour
    if precision == "fp16":
        return "white"
    return COLOURS[precision]


def error_bars(values, lower, upper):
    # matplotlib wants the distance from the point to each end of the CI, in %
    return [100 * (values - lower), 100 * (upper - values)]


def fig1_flip_rates(desc, out_dir):
    """Figure 1: harmful flip rate per follow-up and precision, one panel per model family."""
    fig, axes = plt.subplots(1, len(FAMILY_NAMES), figsize=(SIZE_FIG1[0] * MM, SIZE_FIG1[1] * MM), sharey=True,
                             layout="constrained")
    offsets = {"q4_K_M": -0.24, "q8_0": 0.0, "fp16": 0.24}   # the three precisions side by side
    labels = {"reask": "re-ask", "speaker_free": "speaker-free", "user": "user", "expert": "expert"}
    panels = list(axes.flat)
    families = list(FAMILY_NAMES)

    for i in range(len(families)):
        ax = panels[i]
        family = families[i]
        title = FAMILY_NAMES[family]
        ax.axhspan(1.5, 2.5, color=LIGHT, zorder=0)   # grey band behind the `user` row
        for precision in PRECISIONS:
            d = desc[(desc.family == family) & (desc.precision == precision)]
            d = d.set_index("condition").reindex(CONDITIONS)
            y = []
            for row in range(len(CONDITIONS)):
                y.append(row + offsets[precision])
            x = 100 * d["hfr"]
            err = error_bars(d["hfr"], d["hfr_ci_lo"], d["hfr_ci_hi"])
            ax.errorbar(x, y, xerr=err, fmt=MARKERS[precision], color=COLOURS[precision], ms=11, mew=2,
                        mfc=face_colour(precision), elinewidth=2.2, capsize=4, zorder=3)
        ax.set_title(title, fontweight="bold", pad=8)
        tick_labels = []
        for c in CONDITIONS:
            tick_labels.append(labels[c])
        ax.set_yticks(range(len(CONDITIONS)), tick_labels)
        ax.set_ylim(3.55, -0.55)
        ax.set_xlim(-6, 106)
        ax.set_xticks([0, 50, 100])
        ax.grid(axis="x", color=LIGHT, lw=1.2, zorder=0)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("harmful flip rate (%)", fontsize=26)

    # the legend: one entry per precision, plus the grey band
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
    """Figure 2: difference to fp16 per follow-up, pooled over the six families."""
    d = table[table.scope == "pooled over six families"]
    fig, ax = plt.subplots(figsize=(SIZE_FIG2[0] * MM, SIZE_FIG2[1] * MM), layout="constrained")
    ax.axvspan(-3, 3, color=LIGHT, zorder=0)       # the ±3 pp equivalence margin
    ax.axvline(0, color=GREY, lw=1.6, zorder=1)    # no difference
    labels = {"reask": "re-ask", "speaker_free": "speaker-free", "user": "user\n(pre-registered)", "expert": "expert"}
    # (row name in the table, precision, vertical shift, CI level)
    comparisons = [("q4_K_M vs fp16", "q4_K_M", -0.16, "95%"), ("q8_0 vs fp16", "q8_0", 0.16, "90%")]

    for comparison, precision, shift, level in comparisons:
        rows = d[d.comparison == comparison].set_index("condition").reindex(CONDITIONS)
        y = []
        for row in range(len(CONDITIONS)):
            y.append(row + shift)
        x = 100 * rows["diff_a_minus_b"]
        err = error_bars(rows["diff_a_minus_b"], rows["ci_lo"], rows["ci_hi"])
        ax.errorbar(x, y, xerr=err, fmt=MARKERS[precision], color=COLOURS[precision], ms=13, mew=2, elinewidth=2.4,
                    capsize=5, zorder=3, label=f"{precision} − fp16 ({level} CI)")

    tick_labels = []
    for c in CONDITIONS:
        tick_labels.append(labels[c])
    ax.set_yticks(range(len(CONDITIONS)), tick_labels)
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
    """Figure 3: harmful flip rate per turn-1 confidence bin and precision; the AUROC is shown in the legend."""
    fig, ax = plt.subplots(figsize=(SIZE_FIG3[0] * MM, SIZE_FIG3[1] * MM), layout="constrained")
    offsets = {"q4_K_M": -0.14, "q8_0": 0.0, "fp16": 0.14}
    auroc_of = auroc[auroc.scope == "by precision"].set_index("precision")["auroc"]
    order = bins.drop_duplicates("bin_index").sort_values("bin_index")   # the bins from left to right

    for precision in PRECISIONS:
        d = bins[bins.precision == precision].sort_values("bin_index")
        x = d["bin_index"] + offsets[precision]
        err = error_bars(d["flip_rate"], d["ci_lo"], d["ci_hi"])
        ax.errorbar(x, 100 * d["flip_rate"], yerr=err, fmt="-" + MARKERS[precision], color=COLOURS[precision],
                    ms=12, mew=2, mfc=face_colour(precision), lw=2, elinewidth=2, capsize=4,
                    label=f"{precision} (AUROC {auroc_of[precision]:.2f})")

    ax.set_xticks(order["bin_index"], order["bin"])
    ax.set_xlabel("turn-1 confidence c₀ in the correct answer")
    ax.set_ylabel("harmful flip rate (%)")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=LIGHT, lw=1.2)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    save(fig, out_dir, "fig3_confidence.svg")
