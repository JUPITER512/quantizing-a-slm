"""Build the editable DIN A1 poster in PowerPoint (poster.pptx) and export the print PDF from it.

Same content and layout as build_poster.py (the HTML version), drawn as native PowerPoint shapes:
text boxes, a PowerPoint table and the SVG figures (kept as vector graphics).

Inputs:  poster/poster_text.md, analysis/overview.csv, analysis/hypothesis_summary.csv,
         data/followups.json, figures/fig1_flip_rates.svg, figures/fig2_precision_effect.svg
Outputs: poster/poster.pptx, poster/poster.pdf
Needs Microsoft PowerPoint (Windows); pptx_render.ps1 drives it.

    .venv\\Scripts\\python.exe poster\\build_pptx.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from build_poster import COLOURS, H_MM, MIN_PT, POSTER, ROOT, SECTIONS, W_MM, check_pdf, minus, parse_text, read_csv, words

NAVY, GOLD, INK, GREY = "#13233a", "#f2b632", "#17202b", "#3d4652"
TINT, BUBBLE, BUBBLE_LINE = "#f3f6fa", "#e8eef6", "#c3d0e0"
SYMBOL = "Segoe UI Symbol"
LEFT_X, LEFT_W, RIGHT_X, RIGHT_W = 14, 262, 288, 539


def run(t, b=False, i=False, c=None, s=None, f=None):
    return {"t": t, "b": b, "i": i, "c": c, "s": s, "f": f}


def para(runs, align="l", after=0, within=1.2, left=0, first=0, tabs=()):
    return {"runs": runs, "align": align, "after": after, "within": within, "left": left, "first": first,
            "tabs": list(tabs)}


def md_runs(text, **style):
    """**bold** and *italic* in poster_text.md -> runs."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*|(?<!\*)\*(?!\s).+?(?<!\s)\*(?!\*))", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(run(part[2:-2], b=True, **{k: v for k, v in style.items() if k != "b"}))
        elif part.startswith("*"):
            out.append(run(part[1:-1], i=True, **{k: v for k, v in style.items() if k != "i"}))
        else:
            out.append(run(part, **style))
    return out


def md_paras(text, after=2.5 * 72 / 25.4, **style):
    paras = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if lines and all(l.startswith("- ") for l in lines):
            for l in lines:
                paras.append(para(md_runs("•\t" + l[2:], **style), left=8, first=-8, tabs=[8], after=2))
        elif lines:
            paras.append(para(md_runs(" ".join(lines), **style), after=after))
    return paras


def text(name, x, y, w, h, paras, fill=None, margins=(0, 0, 0, 0), anchor="top", **extra):
    return {"kind": "text", "name": name, "x": x, "y": y, "w": w, "h": h, "paras": paras, "fill": fill,
            "margins": list(margins), "anchor": anchor, **extra}


def rect(name, x, y, w, h, fill, shape="rect", line=None, line_mm=0, radius_mm=0, paras=None,
         margins=(0, 0, 0, 0), anchor="middle"):
    return {"kind": "rect", "name": name, "x": x, "y": y, "w": w, "h": h, "fill": fill, "shape": shape,
            "line": line, "line_mm": line_mm, "radius_mm": radius_mm, "paras": paras, "margins": list(margins),
            "anchor": anchor}


def heading(name, x, y, label, w=250):
    return [rect(name + "_bar", x, y + 1, 3, 11, GOLD),
            text(name, x + 7, y, w, 14, [para([run(label, b=True, c=NAVY, s=36)], within=1.0)])]


def number(n):
    return f"{int(n):,}"


def method_items(followups, overview):
    fu, closing = followups["variants"]["main"], followups["closing"]
    lead = {c: fu[c].replace(" " + closing, "").replace("{x}", "C") for c in fu}
    items = []
    # turn 1
    items.append(rect("turn1", LEFT_X, 211, 198, 15, BUBBLE, "round", BUBBLE_LINE, 0.4, 4))
    items.append(rect("turn1_no", LEFT_X + 2, 214, 9, 9, NAVY, "oval",
                      paras=[para([run("1", b=True, c="#ffffff")], align="c", within=1.0)]))
    items.append(text("turn1_text", LEFT_X + 13, 211, 184, 15,
                      [para([run("Question + A–D + “Answer with the letter only.”")], within=1.0)], anchor="middle"))
    items.append(rect("turn1_answer", 216, 211, 24, 15, NAVY, "round", radius_mm=4,
                      paras=[para([run("B ", c="#ffffff"), run("✓", b=True, c="#9be3b3", f=SYMBOL)], align="c",
                                  within=1.0)]))
    # turn 2
    items.append(rect("turn2", LEFT_X, 228, LEFT_W, 77, BUBBLE, "round", BUBBLE_LINE, 0.4, 4))
    items.append(rect("turn2_no", LEFT_X + 2, 230.5, 9, 9, NAVY, "oval",
                      paras=[para([run("2", b=True, c="#ffffff")], align="c", within=1.0)]))
    tag_w = 58
    lines = [para([run("one of four follow-ups, same conversation:", i=True, c="#4b5563")], left=0, first=11,
                  within=1.15, after=1.5)]
    for key, label in (("reask", "re-ask"), ("speaker_free", "speaker-free"), ("user", "user"), ("expert", "expert")):
        lines.append(para([run(label, b=True, c=NAVY), run("\t“" + lead[key] + "”")], left=tag_w, first=-tag_w,
                          tabs=[tag_w], within=1.15, after=1.5))
    lines.append(para([run("each + “" + closing + "”", i=True, c="#4b5563")], within=1.15))
    items.append(text("turn2_text", LEFT_X + 2, 230.5, LEFT_W - 6, 73, lines))
    items.append(rect("outcome", 166, 308, 110, 13, NAVY, "round", radius_mm=3,
                      paras=[para([run("B = held · ", c="#ffffff"), run("C = harmful flip", b=True, c="#ffffff")],
                                  align="c", within=1.0)]))
    # tiles
    tiles = [(overview["items"], "items"), (overview["follow_up_conditions"], "follow-ups"),
             (overview["configurations"], "configurations"), (number(overview["records_main"]), "turn-2 answers")]
    tile_w = (LEFT_W - 3 * 2.5) / 4
    for k, (n, label) in enumerate(tiles):
        items.append(rect(f"tile{k + 1}", LEFT_X + k * (tile_w + 2.5), 324, tile_w, 26, TINT, "round", radius_mm=3,
                          paras=[para([run(str(n), b=True, c=NAVY, s=40)], align="c", within=1.0),
                                 para([run(label)], align="c", within=1.0)]))
    # facts
    facts_a = [para([run("Items ", b=True), run(f'{overview["items_mmlu_pro"]} MMLU-Pro (reduced to 4 options) + '
                                                 f'{overview["items_arc"]} ARC-Challenge; gold letter balanced')],
                    within=1.15, after=1),
               para([run("Models ", b=True), run("Llama 3.2 3B · Qwen2.5 3B · Phi-4-mini 3.8B · Qwen2.5 7B · "
                                                 "Llama 3.1 8B · Phi-4 14B (one Ollama GGUF source each), all at")],
                    within=1.15)]
    items.append(text("facts_models", LEFT_X, 354, LEFT_W, 41, facts_a))
    x = LEFT_X
    for prec, glyph, bits, w in (("q4_K_M", "●", "≈4-bit", 72), ("q8_0", "■", "8-bit", 55), ("fp16", "△", "16-bit", 60)):
        colour = COLOURS[prec]
        items.append(rect(f"chip_{prec}", x, 396, w, 12, "#ffffff", "round", colour, 0.7, 6,
                          paras=[para([run(glyph + " ", c=colour, f=SYMBOL), run(prec + " ", b=True, c=colour),
                                       run(bits, c="#333333")], align="c", within=1.0)]))
        x += w + 3
    facts_b = [para([run("Decoding ", b=True), run("temperature 0 · seed 42 · top-20 log-probs · thinking off")],
                    within=1.15, after=1),
               para([run("Measures ", b=True), run("answer = reply letter · c₀ = its first-token probability")],
                    within=1.15)]
    items.append(text("facts_decoding", LEFT_X, 410, LEFT_W, 22, facts_b))
    return items


def table_item(summary):
    cols = [36, 158, 116, 180, 49]
    head = ["", "Pre-registered prediction", "Test", "Result", "Rule met"]
    rows = [{"cells": [para([run(h, b=True, c=NAVY)], within=1.12) for h in head], "border": NAVY, "border_mm": 0.7}]
    for r in summary:
        if r["hypothesis"] == "H3c":
            continue
        met = r["rule_met"]
        mark = {"yes": [run("✓", b=True, c="#0b6b3a", f=SYMBOL), run(" yes", b=True, c="#0b6b3a")],
                "no": [run("✗", b=True, c="#a3261b", f=SYMBOL), run(" no", b=True, c="#a3261b")]}.get(met,
                                                                                                    [run("–", c="#555555")])
        rows.append({"cells": [para([run(r["hypothesis"], b=True)], within=1.12),
                               para([run(r["prediction_short"])], within=1.12),
                               para([run(r["test_short"])], within=1.12),
                               para([run(minus(r["result_short"]))], within=1.12),
                               para(mark, within=1.12)],
                     "border": "#cfd6df", "border_mm": 0.3})
    return {"kind": "table", "name": "table1", "x": RIGHT_X, "y": 438, "w": sum(cols), "cols": cols, "rows": rows,
            "pad_x": 2.2, "pad_y": 0.8, "row_min": 10}


def last_analysis_commit():
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", "analysis", "figures"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_spec(meta, sec, overview, summary, followups):
    fig = ROOT / "figures"
    repo = meta["repo"]
    repo_short = repo.replace("https://", "")
    items = [rect("header", 0, 0, W_MM, 72, NAVY),
             text("title", 14, 7, 545, 30, [para([run(meta["title"], c="#ffffff", s=78, f="Segoe UI Black")],
                                                 within=1.0)]),
             text("subtitle", 14, 38, 545, 13, [para([run(meta["subtitle"], c="#dfe8f4", s=30)], within=1.0)]),
             text("authors", 14, 53, 545, 11, [para([run(f'{meta["authors"]} · {meta["affiliation"]}',
                                                          c="#b7c7db", s=26)], within=1.0)]),
             rect("repo_bar", 569, 9, 2.2, 54, GOLD),
             text("repo", 578, 8, 250, 58,
                  [para([run("Code, data and every number on this poster:", c="#eef3f9")], within=1.1),
                   para([run(repo_short, b=True, c="#ffffff", s=26)], within=1.1),
                   para([run("Analysis plan pre-registered before the full runs (commit 099408a)", c="#eef3f9")],
                        within=1.1),
                   para([run("† marks AI-generated poster wording; tool details are in the appendix.",
                             c="#eef3f9")], within=1.1)],
                  anchor="middle", link={"text": repo_short, "url": repo})]
    # left column
    items += heading("h_hypothesis", LEFT_X, 79, "Hypothesis")
    items.append(text("hypothesis", LEFT_X, 95, LEFT_W, 97, md_paras(sec["Hypothesis"])))
    items += heading("h_methodology", LEFT_X, 195, "Methodology")
    items += method_items(followups, overview)
    items.append(text("methodology", LEFT_X, 432, LEFT_W, 84, md_paras(sec["Methodology"])))
    items.append(rect("takehome_box", LEFT_X, 519, LEFT_W, 44, NAVY, "round", radius_mm=3))
    items.append(text("takehome", LEFT_X + 5, 521, LEFT_W - 10, 40,
                      [para([run("Take-home", b=True, c=GOLD, s=28)], within=1.1, after=2)]
                      + md_paras(sec["Take-home"], c="#ffffff")))
    # right column
    items += heading("h_results", RIGHT_X, 79, "Results")
    items.append({"kind": "picture", "name": "figure1", "path": str(fig / "fig1_flip_rates.svg"),
                  "x": RIGHT_X, "y": 97, "w": 539, "h": 150})
    items.append(text("caption_fig1", RIGHT_X, 248, RIGHT_W, 11, md_paras(sec["Figure 1 caption"], c=GREY),
                      ))
    items.append({"kind": "picture", "name": "figure2", "path": str(fig / "fig2_precision_effect.svg"),
                  "x": RIGHT_X, "y": 263, "w": 263, "h": 142})
    items.append(text("caption_fig2", RIGHT_X, 406, 263, 23, md_paras(sec["Figure 2 caption"], c=GREY, ),
                      ))
    col2_x, col2_w = RIGHT_X + 263 + 12, 264
    items.append(text("results", col2_x, 263, col2_w, 98, md_paras(sec["Results"])))
    items.append(rect("limitations_box", col2_x, 362, col2_w, 69, TINT, "round", radius_mm=3))
    items.append(text("limitations", col2_x + 5, 363.5, col2_w - 10, 66,
                      [para([run("Limitations", b=True, c=NAVY, s=28)], within=1.1, after=1.5)]
                      + md_paras(sec["Limitations"])))
    items.append(table_item(summary))
    items.append(text("caption_table1", RIGHT_X, 0, RIGHT_W, 11, md_paras(sec["Table 1 caption"], c=GREY),
                      below="table1", gap=1))
    # footer
    items.append({"kind": "line", "name": "footer_rule", "x": 14, "y": 567, "w": W_MM - 28, "h": 0,
                  "line": NAVY, "line_mm": 0.6})
    refs = ("Clark et al. 2018 (ARC) · Wang et al. 2024a (MMLU-Pro) · Laban et al. 2024 · Sharma et al. 2024 · "
            "Hu & Qu 2026 · Hong et al. 2024 · Proskurina et al. 2024 · Fu et al. 2025 · "
            "Wang et al. 2024b (“My Answer is C”)")
    items.append(text("references", 14, 569, 700, 23,
                      [para([run("References", b=True), run(" (full list in the appendix): " + refs)],
                            within=1.12)], ))
    # only print the commit stamp when git can tell us the commit (no "unknown" on the poster)
    commit = last_analysis_commit()
    if commit != "unknown":
        items.append(text("numbers_commit", 716, 569, W_MM - 14 - 716, 11,
                          [para([run(f"numbers: commit {commit}", c=GREY)], align="r", within=1.12)]))
    for item in items:
        for p in item.get("paras") or []:
            for r in p["runs"]:
                if r["t"] and r["s"] is not None and r["s"] < MIN_PT:
                    raise ValueError(f"font below {MIN_PT} pt in {item['name']}")
    return {"width_mm": W_MM, "height_mm": H_MM, "title": meta["title"], "author": meta["authors"], "defaults": {"font": "Segoe UI", "size": 24, "color": INK},
            "items": items, "report": ["hypothesis", "methodology", "results", "limitations", "caption_table1"]}


def main() -> int:
    meta, sec = parse_text((POSTER / "poster_text.md").read_text(encoding="utf-8"))
    placeholders = sum(len(re.findall(r"\[WRITE", v)) for v in sec.values())
    if placeholders or "YOUR-" in meta.get("repo", "") or not meta.get("repo"):
        print("STATUS: DRAFT (text or repo link missing)")
        return 1
    overview = {r["quantity"]: r["value"] for r in read_csv(ROOT / "analysis" / "overview.csv")}
    summary = read_csv(ROOT / "analysis" / "hypothesis_summary.csv")
    followups = json.loads((ROOT / "data" / "followups.json").read_text(encoding="utf-8"))
    spec = build_spec(meta, sec, overview, summary, followups)

    pptx, pdf = POSTER / "poster.pptx", POSTER / "poster.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = Path(tmp) / "spec.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                              str(POSTER / "pptx_render.ps1"), str(spec_path), str(pptx), str(pdf)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if out.returncode or "SAVED" not in out.stdout:
        print(out.stdout, out.stderr)
        raise SystemExit("PowerPoint rendering failed")

    print("Words per box:")
    for s in SECTIONS:
        print(f"  {s:<12} {words(sec.get(s, '')):>4}")
    report = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    problems = [line for line in report if line.startswith("CHECK")]
    for line in report:
        if line.startswith(("BOX", "TABLE")):
            print("  " + line)
        if line.startswith("MINFONT"):
            pt = float(line.split()[1])
            print(f"smallest text: {pt:.1f} pt ({'OK' if pt >= MIN_PT - 0.05 else 'TOO SMALL'}) in {line.split()[2]}")
    for line in problems:
        print(line.replace("CHECK ", ""))
    if not problems:
        print("layout: no box overflows, nothing off the slide")
    print(f"PPTX: {pptx}")
    print(f"PDF: {pdf}")
    for line in check_pdf(pdf):
        print("  " + line)
    print("STATUS: text complete; check the PDF by eye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
