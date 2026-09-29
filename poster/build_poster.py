"""Build the DIN A1 poster (landscape, 841 x 594 mm) and check it against the exam rules.

Inputs:  poster/poster_text.md   all poster prose (written by the student)
         analysis/*.csv          numbers (from scripts/analyze.py)
         figures/*.svg           figures (from scripts/make_figures.py)
         data/followups.json     exact follow-up wording shown in the method graphic
Outputs: poster/poster.html, poster/poster_html.pdf, poster/poster_preview.png
         (layout reference; the submitted poster.pptx/poster.pdf come from build_pptx.py)

Checks printed at the end: one A1 page, fonts embedded, smallest text >= 24 pt, no box
overflowing, no [WRITE ...] placeholder left, repo link filled in, word count per box.

    .venv\\Scripts\\python.exe poster\\build_poster.py
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

POSTER = Path(__file__).resolve().parent
ROOT = POSTER.parent
W_MM, H_MM = 841, 594
MIN_PT = 24
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
SECTIONS = ["Hypothesis", "Methodology", "Results", "Limitations", "Take-home"]
COLOURS = {"q4_K_M": "#0072B2", "q8_0": "#007A5E", "fp16": "#8E4A8C"}


# ---------------------------------------------------------------- text file

def parse_text(text: str) -> tuple[dict, dict]:
    """Front matter (key: value between --- lines) and '## Heading' sections; <!-- --> removed."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.splitlines()
    meta, sections, i = {}, {}, 0
    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            key, _, val = lines[i].partition(":")
            if key.strip():
                meta[key.strip()] = val.strip()
            i += 1
        i += 1
    current, buf = None, []
    for line in lines[i:]:
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return meta, sections


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def render_block(text: str) -> str:
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        lines = [l.rstrip() for l in para.splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].lstrip().startswith("[WRITE"):
            out.append(f'<div class="todo">{inline(" ".join(lines))}</div>')
        elif all(l.lstrip().startswith("- ") for l in lines):
            out.append("<ul>" + "".join(f"<li>{inline(l.lstrip()[2:])}</li>" for l in lines) + "</ul>")
        else:
            out.append(f"<p>{inline(' '.join(l.strip() for l in lines))}</p>")
    return "\n".join(out)


def words(text: str) -> int:
    text = re.sub(r"\[WRITE[^\]]*\]", "", text)
    return len(re.findall(r"[A-Za-z0-9ÄÖÜäöüß][\w'’\-.%]*", text))


# ---------------------------------------------------------------- data

def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def minus(s: str) -> str:
    """Typographic minus for negative numbers."""
    return re.sub(r"(?<![\w])-(?=\d|\.\d)", "−", s)


def method_graphic(followups: dict, overview: dict) -> str:
    fu = followups["variants"]["main"]
    closing = followups["closing"]
    lead = {c: html.escape(fu[c].replace(" " + closing, "").replace("{x}", "C")) for c in fu}
    tiles = [(overview["items"], "items"), (overview["follow_up_conditions"], "follow-ups"),
             (overview["configurations"], "configurations"), (f'{int(overview["records_main"]):,}', "turn-2 answers")]
    tile_html = "".join(f'<div class="tile"><span class="num">{n}</span><span class="lab">{html.escape(l)}</span></div>'
                        for n, l in tiles)
    chips = "".join(f'<span class="chip" style="border-color:{c};color:{c}"><span class="glyph">{g}</span>{p} '
                    f'<span class="bits">{b}</span></span>'
                    for p, c, g, b in (("q4_K_M", COLOURS["q4_K_M"], "●", "≈4-bit"), ("q8_0", COLOURS["q8_0"], "■", "8-bit"),
                                       ("fp16", COLOURS["fp16"], "△", "16-bit")))
    return f"""
<div class="method">
  <div class="chat">
    <div class="turn"><div class="bubble user"><span class="tno">1</span>Question + A–D + “Answer with the letter only.”</div>
      <div class="bubble model">B <span class="ok">✓</span></div></div>
    <div class="turn"><div class="bubble user fu"><span class="tno">2</span><span class="one-of">one of four follow-ups, same conversation:</span>
      <div class="fl"><span class="tag">re-ask</span>“{lead['reask']}”</div>
      <div class="fl"><span class="tag">speaker-free</span>“{lead['speaker_free']}”</div>
      <div class="fl"><span class="tag">user</span>“{lead['user']}”</div>
      <div class="fl"><span class="tag">expert</span>“{lead['expert']}”</div>
      <div class="closing">each + “{html.escape(closing)}”</div></div></div>
    <div class="turn right"><div class="bubble model">B = held · <b>C = harmful flip</b></div></div>
  </div>
  <div class="tiles">{tile_html}</div>
  <div class="facts">
    <div><b>Items</b> {overview["items_mmlu_pro"]} MMLU-Pro (reduced to 4 options) + {overview["items_arc"]} ARC-Challenge;
      gold letter balanced</div>
    <div><b>Models</b> Llama 3.2 3B · Qwen2.5 3B · Phi-4-mini 3.8B · Qwen2.5 7B · Llama 3.1 8B · Phi-4 14B
      (one Ollama GGUF source each), all at</div>
    <div class="chips">{chips}</div>
    <div><b>Decoding</b> temperature 0 · seed 42 · top-20 log-probs · thinking off</div>
    <div><b>Measures</b> answer = reply letter · c₀ = its first-token probability</div>
  </div>
</div>"""

def results_table(rows: list[dict]) -> str:
    body = []
    for r in rows:
        if r["hypothesis"] == "H3c":
            continue
        met = r["rule_met"]
        cls = {"yes": "yes", "no": "no"}.get(met, "na")
        label = {"yes": "✓ yes", "no": "✗ no"}.get(met, "–")
        body.append(f'<tr><td class="h">{html.escape(r["hypothesis"])}</td><td>{html.escape(r["prediction_short"])}</td>'
                    f'<td>{html.escape(r["test_short"])}</td><td class="res">{html.escape(minus(r["result_short"]))}</td>'
                    f'<td class="met {cls}">{label}</td></tr>')
    return ('<table class="results"><thead><tr><th></th><th>Pre-registered prediction</th><th>Test</th>'
            '<th>Result</th><th>Rule met</th></tr></thead><tbody>' + "".join(body) + "</tbody></table>")


# ---------------------------------------------------------------- page

CSS = f"""
@page {{ size: {W_MM}mm {H_MM}mm; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{ width: {W_MM}mm; height: {H_MM}mm; overflow: hidden; background: #ffffff; color: #17202b;
  font-family: "Segoe UI", Arial, sans-serif; font-size: 24pt; line-height: 1.2;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.poster {{ position: relative; width: {W_MM}mm; height: {H_MM}mm; display: grid; grid-template-rows: 72mm 1fr auto; }}
header {{ background: #13233a; color: #ffffff; display: grid; grid-template-columns: 1fr 258mm; gap: 14mm;
  align-items: center; padding: 0 14mm; }}
h1 {{ font-size: 78pt; line-height: 1; margin: 0; font-weight: 800; letter-spacing: -0.5pt; }}
.subtitle {{ font-size: 30pt; margin: 3mm 0 0; color: #dfe8f4; line-height: 1.15; }}
.authors {{ font-size: 26pt; margin: 3mm 0 0; color: #b7c7db; }}
.repo {{ font-size: 24pt; border-left: 2.2mm solid #f2b632; padding: 2mm 0 2mm 7mm; color: #eef3f9; }}
.repo b {{ color: #ffffff; font-size: 26pt; word-break: break-all; }}
.repo .todo-inline {{ color: #ffb3b3; }}
main {{ display: grid; grid-template-columns: 262mm 539mm; gap: 12mm; padding: 7mm 14mm 5mm; min-height: 0; }}
.col {{ min-height: 0; display: flex; flex-direction: column; gap: 5mm; }}
h2 {{ font-size: 36pt; line-height: 1.1; margin: 0 0 2.5mm; color: #13233a; font-weight: 750;
  border-left: 3mm solid #f2b632; padding-left: 4mm; }}
h3 {{ font-size: 28pt; margin: 0 0 1.5mm; color: #13233a; font-weight: 700; }}
p {{ margin: 0 0 2.5mm; }}
ul {{ margin: 0 0 2mm; padding-left: 8mm; }}
li {{ margin: 0 0 1mm; }}
code {{ font-family: Consolas, monospace; font-size: 24pt; }}
.todo {{ border: 1mm dashed #c0392b; background: #fdecea; color: #7b241c; padding: 2.5mm 4mm; margin: 0 0 2.5mm;
  font-size: 24pt; }}
figure {{ margin: 0; }}
figure img {{ display: block; }}
figure.confidence img {{ width: 100%; height: auto; }}
figcaption {{ font-size: 24pt; color: #3d4652; margin-top: 1mm; line-height: 1.15; }}
.row2 {{ display: grid; grid-template-columns: 263mm 264mm; gap: 12mm; align-items: start; }}
.discuss {{ display: flex; flex-direction: column; gap: 3mm; }}
.box-tint {{ background: #f3f6fa; border-radius: 3mm; padding: 2.5mm 5mm; }}
.takehome {{ margin-top: auto; background: #13233a; color: #ffffff; border-radius: 3mm; padding: 3mm 5mm; }}
.takehome h3 {{ color: #f2b632; }}
.takehome .todo {{ background: #3b2a2a; color: #ffd6d1; border-color: #ff8a80; }}
table.results {{ width: 100%; border-collapse: collapse; font-size: 24pt; line-height: 1.12; }}
table.results th {{ text-align: left; font-weight: 700; color: #13233a; border-bottom: 0.7mm solid #13233a; padding: 1mm 2.5mm; }}
table.results td {{ border-bottom: 0.3mm solid #cfd6df; padding: 1mm 2.5mm; vertical-align: top; }}
table.results td.h {{ font-weight: 700; white-space: nowrap; }}
table.results td.res {{ white-space: nowrap; }}
table.results td.met {{ white-space: nowrap; font-weight: 700; }}
td.met.yes {{ color: #0b6b3a; }} td.met.no {{ color: #a3261b; }} td.met.na {{ color: #555; }}
.method {{ display: flex; flex-direction: column; gap: 3mm; font-size: 24pt; }}
.chat {{ display: flex; flex-direction: column; gap: 2mm; }}
.turn {{ display: flex; align-items: center; gap: 3mm; }}
.turn.right {{ justify-content: flex-end; }}
.who {{ font-size: 24pt; color: #5b6572; white-space: nowrap; }}
.bubble {{ border-radius: 4mm; padding: 1.8mm 4mm; line-height: 1.15; }}
.bubble.user {{ background: #e8eef6; border: 0.4mm solid #c3d0e0; }}
.bubble.model {{ background: #13233a; color: #fff; }}
.bubble .ok {{ color: #9be3b3; font-weight: 700; margin-left: 2mm; }}
.bubble.fu div {{ margin: 0.8mm 0; }}
.bubble.fu div.fl {{ display: grid; grid-template-columns: 60mm 1fr; }}
.tno {{ display: inline-block; width: 9mm; height: 9mm; line-height: 9mm; text-align: center; border-radius: 50%;
  background: #13233a; color: #fff; font-weight: 800; margin-right: 2.5mm; }}
.one-of {{ color: #4b5563; font-style: italic; }}
.tag {{ font-weight: 700; color: #13233a; }}
.closing {{ color: #4b5563; font-style: italic; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 2.5mm; }}
.tile {{ background: #f3f6fa; border-radius: 3mm; padding: 1.5mm 2mm; display: flex; flex-direction: column;
  align-items: center; text-align: center; }}
.tile .num {{ font-size: 40pt; font-weight: 800; color: #13233a; line-height: 1.05; }}
.tile .lab {{ font-size: 24pt; line-height: 1.05; }}
.chips {{ display: flex; gap: 3mm; flex-wrap: wrap; margin: 1mm 0; }}
.chip {{ border: 0.7mm solid; border-radius: 10mm; padding: 0.8mm 4mm; font-weight: 700; font-size: 24pt; }}
.chip .glyph {{ margin-right: 2mm; }} .chip .bits {{ font-weight: 400; color: #333; }}
.facts div {{ margin: 0.6mm 0; line-height: 1.15; }}
footer {{ border-top: 0.6mm solid #13233a; margin: 0 14mm; padding: 2.5mm 0 4mm; font-size: 24pt; color: #3d4652;
  line-height: 1.15; display: grid; grid-template-columns: 1fr auto; gap: 10mm; }}
.draft {{ position: absolute; top: 76mm; right: 16mm; background: #c0392b; color: #fff; font-weight: 800;
  font-size: 30pt; padding: 2mm 6mm; border-radius: 2mm; }}
"""

CHECK_JS = """
<script>
window.addEventListener('load', function () {
  var rep = [];
  document.querySelectorAll('[data-check]').forEach(function (el) {
    if (el.scrollHeight > el.clientHeight + 2 || el.scrollWidth > el.clientWidth + 2)
      rep.push('OVERFLOW ' + el.getAttribute('data-check') + ' by ' + ((el.scrollHeight - el.clientHeight) * 25.4 / 96).toFixed(1) + ' mm');
  });
  var min = 1e9, minWhat = '';
  document.querySelectorAll('body *').forEach(function (el) {
    if (el.closest('#layout-report') || el.tagName === 'SCRIPT' || el.tagName === 'STYLE') return;
    var own = Array.prototype.some.call(el.childNodes, function (n) { return n.nodeType === 3 && n.textContent.trim(); });
    if (!own) return;
    var px = parseFloat(getComputedStyle(el).fontSize);
    if (px < min) { min = px; minWhat = el.tagName.toLowerCase() + (el.className ? '.' + el.className : ''); }
  });
  rep.push('MINFONT ' + (min * 0.75).toFixed(1) + ' ' + minWhat);
  var imgs = Array.prototype.filter.call(document.images, function (i) { return !i.complete || i.naturalWidth === 0; });
  if (imgs.length) rep.push('IMAGE_MISSING ' + imgs.map(function (i) { return i.getAttribute('src'); }).join(','));
  document.getElementById('layout-report').textContent = rep.join('\\n');
});
</script>"""


def build_html(meta: dict, sec: dict, overview: dict, summary: list[dict], followups: dict, draft: bool) -> str:
    repo = meta.get("repo", "")
    repo_html = (f'<span class="todo-inline">{html.escape(repo)} (fill in)</span>' if "YOUR-" in repo or not repo
                 else html.escape(repo.replace("https://", "")))
    body = f"""
<div class="poster">
  <header>
    <div>
      <h1>{inline(meta.get('title', ''))}</h1>
      <p class="subtitle">{inline(meta.get('subtitle', ''))}</p>
      <p class="authors">{inline(meta.get('authors', ''))} · {inline(meta.get('affiliation', ''))}</p>
    </div>
    <div class="repo">Code, data and every number on this poster:<br><b>{repo_html}</b><br>
      Analysis plan pre-registered before the full runs (commit 099408a)<br>
      † marks AI-generated poster wording; tool details are in the appendix.</div>
  </header>
  <main>
    <div class="col" data-check="left column">
      <section><h2>Hypothesis</h2>{render_block(sec.get('Hypothesis', ''))}</section>
      <section><h2>Methodology</h2>{method_graphic(followups, overview)}
        <div style="margin-top:3mm">{render_block(sec.get('Methodology', ''))}</div></section>
      <section class="takehome"><h3>Take-home</h3>{render_block(sec.get('Take-home', ''))}</section>
    </div>
    <div class="col" data-check="results column">
      <h2 style="margin-bottom:0">Results</h2>
      <figure><img src="../figures/fig1_flip_rates.svg" alt="Figure 1">
        <figcaption>{inline(sec.get('Figure 1 caption', ''))}</figcaption></figure>
      <div class="row2">
        <figure><img src="../figures/fig2_precision_effect.svg" alt="Figure 2">
          <figcaption>{inline(sec.get('Figure 2 caption', ''))}</figcaption></figure>
        <div class="discuss">{render_block(sec.get('Results', ''))}
          <div class="box-tint"><h3>Limitations</h3>{render_block(sec.get('Limitations', ''))}</div></div>
      </div>
      <div>{results_table(summary)}<figcaption>{inline(sec.get('Table 1 caption', ''))}</figcaption></div>
    </div>
  </main>
  <footer>
    <div><b>References</b> (full list in the appendix): Clark et al. 2018 (ARC) · Wang et al. 2024a (MMLU-Pro) ·
      Laban et al. 2024 · Sharma et al. 2024 · Hu &amp; Qu 2026 · Hong et al. 2024 · Proskurina et al. 2024 ·
      Fu et al. 2025 · Wang et al. 2024b (“My Answer is C”)</div>
    <div style="white-space:nowrap">numbers: commit {git_head()}</div>
  </footer>
  {'<div class="draft">DRAFT – text missing</div>' if draft else ''}
</div>"""
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>{html.escape(meta.get("title", "Poster"))}</title>'
            f"<style>{CSS}</style></head><body>{body}<pre id=\"layout-report\" style=\"display:none\"></pre>{CHECK_JS}</body></html>")


def git_head() -> str:
    try:
        return subprocess.run(["git", "-c", f"safe.directory={ROOT}", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------- browser

def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    raise SystemExit("Edge or Chrome not found")


def run_browser(args: list[str]) -> subprocess.CompletedProcess:
    output_flags = ("--print-to-pdf=", "--screenshot=")
    targets = {}
    for arg in args:
        for flag in output_flags:
            if arg.startswith(flag):
                path = Path(arg[len(flag):])
                targets[path] = path.stat().st_mtime_ns if path.exists() else None
                break
    with tempfile.TemporaryDirectory() as profile:
        cmd = [browser(), "--headless=new", "--disable-gpu", "--disable-software-rasterizer",
               "--disable-gpu-compositing", "--no-sandbox", "--hide-scrollbars", "--no-first-run",
               f"--user-data-dir={profile}", "--virtual-time-budget=15000", *args]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if result.returncode:
        raise RuntimeError(f"Browser failed ({result.returncode}): {result.stderr[-2000:]}")
    if "--dump-dom" in args and not result.stdout.strip():
        raise RuntimeError("Browser returned an empty DOM; layout checks did not run")
    for path, previous_mtime in targets.items():
        if not path.is_file():
            raise RuntimeError(f"Browser did not create the expected output: {path}")
        if previous_mtime is not None and path.stat().st_mtime_ns <= previous_mtime:
            raise RuntimeError(f"Browser did not update the expected output: {path}")
    return result


def check_pdf(pdf: Path) -> list[str]:
    data = pdf.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page(?![s\w])", data))
    boxes = re.findall(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", data)
    out = [f"pages: {pages} (must be 1)"]
    if boxes:
        w, h = (float(boxes[0][2]) - float(boxes[0][0])) * 25.4 / 72, (float(boxes[0][3]) - float(boxes[0][1])) * 25.4 / 72
        ok = abs(w - W_MM) < 1.5 and abs(h - H_MM) < 1.5
        out.append(f"page size: {w:.0f} x {h:.0f} mm ({'A1 landscape OK' if ok else 'NOT A1'})")
    out.append("fonts embedded: " + ("yes" if re.search(rb"/FontFile[23]?", data) else "NO"))
    images = len(re.findall(rb"/Subtype\s*/Image", data))
    out.append(f"raster images: {images}" + (" (vector only, 150 PPI rule cannot fail)" if images == 0
                                              else " (check that each has >= 150 PPI at print size)"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", type=Path, default=POSTER / "poster_text.md")
    ap.add_argument("--no-pdf", action="store_true", help="only write poster.html and run the layout checks")
    ap.add_argument("--fill-test", action="store_true",
                    help="layout test: fill every box with dummy words at its full budget (writes poster_filltest.*)")
    args = ap.parse_args(argv)

    meta, sec = parse_text(args.text.read_text(encoding="utf-8"))
    budget = {"Hypothesis": 90, "Methodology": 60, "Results": 100, "Limitations": 45, "Take-home": 25}
    stem = "poster"
    if args.fill_test:
        dummy = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
                 "et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris").split()
        sec = {**sec, **{s: " ".join(dummy[i % len(dummy)] for i in range(n)) for s, n in budget.items()}}
        meta = {**meta, "repo": "https://github.com/example-user/cave-in-at-4-bits"}
        stem = "poster_filltest"
    overview = {r["quantity"]: r["value"] for r in read_csv(ROOT / "analysis" / "overview.csv")}
    summary = read_csv(ROOT / "analysis" / "hypothesis_summary.csv")
    followups = json.loads((ROOT / "data" / "followups.json").read_text(encoding="utf-8"))
    placeholders = sum(len(re.findall(r"\[WRITE", sec.get(s, ""))) for s in sec)
    repo_missing = "YOUR-" in meta.get("repo", "") or not meta.get("repo")
    draft = placeholders > 0 or repo_missing

    html_path = POSTER / f"{stem}.html"
    html_path.write_text(build_html(meta, sec, overview, summary, followups, draft), encoding="utf-8", newline="\n")
    url = html_path.as_uri()

    dom = run_browser(["--dump-dom", url]).stdout
    m = re.search(r'<pre id="layout-report"[^>]*>(.*?)</pre>', dom, re.S)
    report = html.unescape(m.group(1)).strip().splitlines() if m else ["layout check did not run"]

    print("Words per box (budget):")
    for s in SECTIONS:
        n = words(sec.get(s, ""))
        print(f"  {s:<12} {n:>4}  ({budget[s]})" + ("  <- over budget" if n > budget[s] * 1.15 else ""))
    print(f"placeholders left: {placeholders}" + ("  <- replace every [WRITE ...] with your text" if placeholders else ""))
    print("repo link: " + ("MISSING (set 'repo:' in poster_text.md)" if repo_missing else meta["repo"]))
    for line in report:
        if line.startswith("MINFONT"):
            pt = float(line.split()[1])
            print(f"smallest text: {pt:.1f} pt ({'OK' if pt >= MIN_PT - 0.05 else 'TOO SMALL'}) {line.split(' ', 2)[2]}")
        else:
            print(line)
    if not any(l.startswith("OVERFLOW") for l in report):
        print("layout: no box overflows")

    if not args.no_pdf:
        pdf = POSTER / f"{stem}_html.pdf"
        run_browser(["--no-pdf-header-footer", "--run-all-compositor-stages-before-draw", f"--print-to-pdf={pdf}", url])
        png = POSTER / f"{stem}_preview.png"
        run_browser([f"--screenshot={png}", f"--window-size={round(W_MM / 25.4 * 96)},{round(H_MM / 25.4 * 96)}",
                     "--force-device-scale-factor=0.5", url])
        print(f"PDF: {pdf}" if pdf.exists() else "PDF was not written")
        if pdf.exists():
            for line in check_pdf(pdf):
                print("  " + line)
        print(f"preview: {png}" if png.exists() else "preview not written")
    print("STATUS: " + ("DRAFT (text or repo link missing)" if draft else "text complete; check the PDF by eye"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
