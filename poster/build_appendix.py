"""Build the A4 appendix: data-derived tables, references and declaration scan.

Inputs: analysis/*.csv, poster/appendix_references.md, poster/poster_text.md,
        poster/declaration/*.png|jpg
Outputs: poster/appendix.html, poster/appendix.pdf

    .venv\\Scripts\\python.exe poster\\build_appendix.py
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys

from build_poster import POSTER, ROOT, check_pdf, parse_text, run_browser

DECL_DIR = POSTER / "declaration"

CSS = """
@page { size: A4; margin: 15mm 15mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.3; color: #17202b; }
h1 { font-size: 18pt; margin: 0 0 1mm; }
.sub { color: #4b5563; margin: 0 0 6mm; }
h2 { font-size: 13pt; margin: 4mm 0 2mm; color: #13233a; border-bottom: 0.3mm solid #13233a; }
h3 { font-size: 10.5pt; margin: 3mm 0 1mm; color: #13233a; }
p { margin: 1.5mm 0; }
.table-block { page-break-inside: avoid; }
.table-summary, .robustness, .refs { page-break-before: always; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 8.4pt; line-height: 1.14; }
th { text-align: left; color: #13233a; background: #edf2f7; font-weight: 700; }
th, td { border: 0.25mm solid #cbd3dd; padding: 1.2mm 1.4mm; vertical-align: top; overflow-wrap: anywhere; }
tbody tr:nth-child(even) { background: #f7f9fb; }
.pct, .num, .decision { text-align: center; white-space: nowrap; }
.rates th.num { white-space: normal; }
.yes { color: #12653c; font-weight: 700; }
.no { color: #a3261b; font-weight: 700; }
.table-note { color: #46515f; font-size: 9pt; line-height: 1.25; }
.refs ol { margin: 0; padding-left: 7mm; }
.refs li { margin: 0 0 1.4mm; break-inside: avoid; }
a { color: #0b4f8a; text-decoration: none; overflow-wrap: anywhere; }
.suppfig { page-break-before: always; }
.suppfig img { display: block; width: 100%; height: auto; }
.suppfig figcaption { margin-top: 2mm; color: #46515f; font-size: 9pt; }
.ai-disclosure { page-break-before: always; }
.ai-disclosure table { font-size: 9pt; }
.ai-disclosure th:first-child, .ai-disclosure td:first-child { width: 32%; }
.facts td:first-child { width: 17%; font-weight: 700; color: #13233a; }
.inventory-block { page-break-before: always; }
code { font-family: Consolas, monospace; font-size: 9pt; }
.decl { page-break-before: always; }
.decl img { display: block; width: 100%; max-height: 255mm; object-fit: contain; margin: 0 auto; }
.decl img + img { page-break-before: always; }
.missing { border: 0.6mm dashed #c0392b; background: #fdecea; color: #7b241c; padding: 6mm; margin-top: 10mm; }
"""

FAMILY_LABELS = {
    "llama3.2-3b": "Llama 3.2 3B",
    "qwen2.5-3b": "Qwen2.5 3B",
    "phi4-mini-3.8b": "Phi-4-mini 3.8B",
    "qwen2.5-7b": "Qwen2.5 7B",
    "llama3.1-8b": "Llama 3.1 8B",
    "phi4-14b": "Phi-4 14B",
}
FAMILY_ORDER = {name: i for i, name in enumerate(FAMILY_LABELS)}
PRECISION_ORDER = {"q4_K_M": 0, "q8_0": 1, "fp16": 2}
CONDITIONS = ("reask", "speaker_free", "user", "expert")
CONDITION_LABELS = {
    "reask": "Re-ask",
    "speaker_free": "Speaker-free",
    "user": "User",
    "expert": "Expert",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_refs(text: str) -> list[tuple[str, str, list[str]]]:
    """[(level, heading, entries)] from '# ' / '## ' headings; % lines are comments."""
    blocks, level, head, entries = [], "", "", []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("#"):
            if head or entries:
                blocks.append((level, head, entries))
            level, head, entries = line.split()[0], line.lstrip("#").strip(), []
        else:
            entries.append(line)
    if head or entries:
        blocks.append((level, head, entries))
    return blocks


def linkify(entry: str) -> str:
    esc = html.escape(entry, quote=False)
    return re.sub(r"(https?://[^\s)]+)", r'<a href="\1">\1</a>', esc)


def table_a1() -> str:
    rows = read_csv(ROOT / "analysis" / "descriptives.csv")
    grouped = {}
    for row in rows:
        key = (row["family"], row["precision"])
        grouped.setdefault(key, {})[row["condition"]] = row

    body = []
    for (family, precision), cells in sorted(
        grouped.items(),
        key=lambda item: (FAMILY_ORDER.get(item[0][0], 99), PRECISION_ORDER.get(item[0][1], 99)),
    ):
        if set(cells) != set(CONDITIONS):
            raise ValueError(f"Incomplete condition rows for {family} {precision}")
        denominators = {int(cells[c]["n_correct0"]) for c in CONDITIONS}
        if len(denominators) != 1:
            raise ValueError(f"Turn-1 denominators differ by condition for {family} {precision}")
        values = [
            FAMILY_LABELS.get(family, family),
            precision,
            str(denominators.pop()),
            *[f'{100 * float(cells[c]["hfr"]):.1f}%' for c in CONDITIONS],
        ]
        body.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")

    return (
        '<table class="rates"><colgroup>'
        '<col style="width:22%"><col style="width:13%"><col style="width:8%">'
        '<col style="width:14.25%"><col style="width:15.75%"><col style="width:13.5%"><col style="width:13.5%">'
        "</colgroup><thead><tr><th>Model family</th><th>Precision</th><th class=\"num\">N0</th>"
        "<th>Re-ask</th><th>Speaker-free</th><th>User</th><th>Expert</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def table_a2() -> str:
    wanted = ("H1a", "H1b", "H2a", "H2b", "H3a", "H3b", "Validity")
    summary = {row["hypothesis"]: row for row in read_csv(ROOT / "analysis" / "hypothesis_summary.csv")}
    body = []
    for key in wanted:
        row = summary[key]
        met = row["rule_met"]
        decision = "Yes" if met == "yes" else "No" if met == "no" else "Reported"
        cls = "yes" if met == "yes" else "no" if met == "no" else ""
        fields = (key, row["rule"], row["result_short"], decision)
        body.append(
            f'<tr><td><b>{html.escape(fields[0])}</b></td><td>{html.escape(fields[1])}</td>'
            f'<td>{html.escape(fields[2])}</td><td class="decision {cls}">{html.escape(fields[3])}</td></tr>'
        )
    return (
        "<table><colgroup>"
        '<col style="width:10%"><col style="width:34%"><col style="width:45%"><col style="width:11%">'
        "</colgroup><thead><tr><th>Test</th><th>Registered rule</th><th>Observed result</th>"
        "<th>Rule met</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def table_a3() -> str:
    rows = read_csv(ROOT / "analysis" / "robustness_determinism.csv")
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f'<td>{html.escape(row["compare"])}</td>'
            f'<td class="num">{html.escape(row["n"])}</td>'
            f'<td class="pct">{100 * float(row["same_a1"]):.1f}%</td>'
            f'<td class="pct">{100 * float(row["same_raw_1"]):.1f}%</td>'
            f'<td class="num">{float(row["max_abs_diff_c0"]):.3f}</td>'
            "</tr>"
        )
    return (
        "<table><colgroup>"
        '<col style="width:34%"><col style="width:10%"><col style="width:18%"><col style="width:18%"><col style="width:20%">'
        "</colgroup><thead><tr><th>Comparison</th><th>N</th><th>Same answer letter</th>"
        "<th>Same raw answer</th><th>Maximum |Δc₀|</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def pct(value) -> str:
    return f"{100 * float(value):.1f}%"


def pp(value) -> str:
    return f"{100 * float(value):+.1f}".replace("-", "−")


def table_family() -> str:
    """Paired q4_K_M-fp16 and q8_0-fp16 contrasts per family (user follow-up), descriptive."""
    rows = []
    for name, label, level in (("primary_mcnemar.csv", "q4_K_M − fp16", "95%"),
                               ("h1b_equivalence.csv", "q8_0 − fp16", "90%")):
        for row in read_csv(ROOT / "analysis" / name):
            scope = "Pooled (registered)" if row["scope"].startswith("pooled") else FAMILY_LABELS.get(row["scope"], row["scope"])
            extra = ""
            if "equivalent" in row:
                extra = "inside ±3 pp" if row["equivalent"] == "True" else "not inside ±3 pp"
            else:
                extra = f"p = {float(row['p_mcnemar_exact']):.2g}"
            rows.append((label, scope, row["n_pairs"], f"{pct(row['hfr_a'])} vs {pct(row['hfr_b'])}",
                         f"{pp(row['diff_a_minus_b'])} [{pp(row['ci_lo'])}, {pp(row['ci_hi'])}] ({level})",
                         f"{row['only_a']} / {row['only_b']}", extra))
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in rows)
    return ("<table><colgroup><col style='width:13%'><col style='width:17%'><col style='width:7%'>"
            "<col style='width:17%'><col style='width:22%'><col style='width:9%'><col style='width:15%'></colgroup>"
            "<thead><tr><th>Contrast</th><th>Scope</th><th>Pairs</th><th>HFR a vs b</th><th>Difference, pp [CI]</th>"
            "<th>Only a / only b</th><th>Test / rule</th></tr></thead><tbody>" + body + "</tbody></table>")


def table_controls() -> str:
    rows = read_csv(ROOT / "analysis" / "robustness_controls.csv")
    cells = {}
    for r in rows:
        cells.setdefault((r["family"], r["precision"], r["variant"]), {})[r["condition"]] = r
    variants = ("main", "reversed", "para1", "para2")
    body = []
    for (family, precision, variant), cond in sorted(cells.items(), key=lambda kv: (
            FAMILY_ORDER.get(kv[0][0], 99), PRECISION_ORDER.get(kv[0][1], 99), variants.index(kv[0][2]))):
        n = cond["user"]["n_correct0"]
        values = [FAMILY_LABELS.get(family, family), precision, variant, n] + [pct(cond[c]["hfr"]) for c in CONDITIONS]
        body.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    return ("<table><colgroup><col style='width:20%'><col style='width:11%'><col style='width:11%'><col style='width:7%'>"
            "<col style='width:12.75%'><col style='width:12.75%'><col style='width:12.75%'><col style='width:12.75%'></colgroup>"
            "<thead><tr><th>Model family</th><th>Precision</th><th>Variant</th><th>N0</th><th>Re-ask</th>"
            "<th>Speaker-free</th><th>User</th><th>Expert</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def table_size() -> str:
    rows = read_csv(ROOT / "analysis" / "descriptive_size_comparison.csv")
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in (
            r["developer"], CONDITION_LABELS[r["condition"]], r["larger_4bit"], pct(r["hfr_larger_4bit"]),
            r["smaller_16bit"], pct(r["hfr_smaller_16bit"]))) + "</tr>" for r in rows)
    return ("<table><thead><tr><th>Developer</th><th>Follow-up</th><th>Larger model at 4 bits</th><th>HFR</th>"
            "<th>Smaller model at 16 bits</th><th>HFR</th></tr></thead><tbody>" + body + "</tbody></table>")


def model_inventory() -> list[tuple]:
    """One row per main configuration, from results/*__main.jsonl and env_info.txt (digest, GPU share)."""
    env = (ROOT / "env_info.txt").read_text(encoding="utf-8")
    blocks = {}
    for block in re.split(r"\n(?==== )", env):
        head = re.match(r"=== (\S+)\s+(\S+)\s+->\s+(\S+)", block)
        if not head:
            continue
        info = blocks.setdefault(head.group(3), {})
        for line in block.splitlines():
            if line.startswith("digest "):
                info["digest"] = line.split()[1]
            elif line.startswith("details "):
                details = json.loads(line[8:])
                info["params"], info["quant"] = details.get("parameter_size", ""), details.get("quantization_level", "")
            elif line.startswith("loaded size"):
                info["gpu"] = re.search(r"\((\d+%)\)", line).group(1)
    rows = []
    for path in sorted((ROOT / "results").glob("*__main.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        first, info = records[0], blocks.get(path.name, {})
        dates = sorted({r["timestamp"][:10] for r in records})
        rows.append((FAMILY_ORDER.get(first["family"], 99), PRECISION_ORDER.get(first["precision"], 99),
                     FAMILY_LABELS.get(first["family"], first["family"]), first["model"], info.get("params", ""),
                     info.get("quant", ""), info.get("digest", "")[:12], info.get("gpu", ""), str(len(records)),
                     ", ".join(dates)))
    return [r[2:] for r in sorted(rows)]


def table_inventory() -> str:
    body = "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in r) + "</tr>" for r in model_inventory())
    return ("<table class='inventory'><colgroup><col style='width:14%'><col style='width:24%'><col style='width:7%'>"
            "<col style='width:9%'><col style='width:15%'><col style='width:8%'><col style='width:9%'><col style='width:14%'>"
            "</colgroup><thead><tr><th>Family</th><th>Ollama tag</th><th>Size</th><th>Weights</th><th>Digest (12)</th>"
            "<th>On GPU</th><th>Records</th><th>Run date</th></tr></thead><tbody>" + body + "</tbody></table>")


def main() -> int:
    meta, _ = parse_text((POSTER / "poster_text.md").read_text(encoding="utf-8"))
    ref_blocks = parse_refs((POSTER / "appendix_references.md").read_text(encoding="utf-8"))
    scans = sorted(
        p for p in DECL_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ) if DECL_DIR.exists() else []
    n_refs = sum(len(entries) for _, _, entries in ref_blocks)

    overview = {r["quantity"]: r["value"] for r in read_csv(ROOT / "analysis" / "overview.csv")}
    parts = [
        f"<h1>Appendix · {html.escape(meta.get('title', ''))}</h1>",
        f"<p class='sub'>{html.escape(meta.get('subtitle', ''))}<br>{html.escape(meta.get('authors', ''))} · "
        f"{html.escape(meta.get('affiliation', ''))}<br>Repository: {linkify(meta.get('repo', ''))}</p>",
        '<section><h2>† Contents and study record</h2>'
        '<p>† Contents: study record; Tables A1–A7; Figure A1; references; generative-AI disclosure; declaration of '
        'academic integrity. All tables and the figure are generated by <code>poster/build_appendix.py</code> from '
        '<code>analysis/*.csv</code>, <code>results/*.jsonl</code> and <code>env_info.txt</code>; '
        '<code>python scripts/verify_repro.py</code> rebuilds every analysis table and figure from the frozen results '
        'and checks that they are byte-identical.</p>'
        '<table class="facts"><tbody>'
        f'<tr><td>Items</td><td>† {overview["items"]} English four-option questions: {overview["items_mmlu_pro"]} '
        f'MMLU-Pro (test split, stratified over its 14 categories, reduced to the gold answer plus three '
        f'seed-selected distractors) and {overview["items_arc"]} ARC-Challenge (test split); gold letter balanced '
        '(75 per letter); one fixed wrong target letter X per item; seed 42.</td></tr>'
        f'<tr><td>Models</td><td>† {overview["families"]} families × {overview["precisions"]} precisions = '
        f'{overview["configurations"]} local configurations (Ollama library GGUF tags, one source per family; '
        'Table A7): Llama 3.2 3B, Qwen2.5 3B, Phi-4-mini 3.8B, Qwen2.5 7B, Llama 3.1 8B, Phi-4 14B, each at q4_K_M, '
        'q8_0 and fp16. No API model was run; the optional gemma4:12b and gpt-4o-mini runs listed in the design '
        'were not carried out.</td></tr>'
        '<tr><td>Procedure</td><td>† Turn 1: question and options A–D, “Answer with the letter only.” Turn 2: '
        '[question] [verbatim turn-1 reply] [one follow-up], separately for re-ask, speaker-free, user and expert; '
        'each follow-up ends with the same closing instruction.</td></tr>'
        '<tr><td>Decoding</td><td>† Ollama native /api/chat; temperature 0, seed 42, top-20 log-probabilities, '
        'thinking off, at most 16 generated tokens; identical for every configuration.</td></tr>'
        '<tr><td>Measures</td><td>† Answer = letter parsed from the reply text; c₀ = probability of that letter at '
        'the first generated position (letter variants summed, normalised over A–D; a letter outside the top 20 is '
        'missing, not zero). Harmful flip rate (HFR) = turn-1-correct answers that are wrong after the follow-up, '
        'divided by turn-1-correct answers.</td></tr>'
        f'<tr><td>Records</td><td>† {int(overview["records_main"]):,} main turn-2 answers, '
        f'{int(overview["records_controls"]):,} control answers (reversed option order and two paraphrases, '
        f'{overview["control_items"]} items, six small-model configurations), '
        f'{int(overview["records_determinism"]):,} determinism-rerun answers; no error records.</td></tr>'
        '<tr><td>Pre-registration</td><td>† Analysis plan committed before the full runs (commit 099408a); '
        'results frozen with the git tag <code>data-frozen</code>. Only the hypotheses in Table A2 are '
        'pre-registered; Tables A3–A6 and Figure A1 are descriptive, robustness or exploratory.</td></tr>'
        '</tbody></table></section>',
        '<section class="table-block"><h2>Appendix tables</h2>'
        "<h3>† Table A1. Harmful flip rate by model family, precision and follow-up</h3>",
        table_a1(),
        '<p class="table-note">† Harmful flip rate (HFR) is the share of turn-1-correct answers that become wrong '
        "after a follow-up. N0 is the number of turn-1-correct cases for that family and precision. Rates are "
        "descriptive; only the user follow-up enters the registered precision contrast. Text letters are scored as "
        "the visible answer (Wang et al., 2024b).</p></section>",
        '<section class="table-block table-summary"><h2>Registered tests</h2>'
        "<h3>† Table A2. Registered decision rules and observed outcomes</h3>",
        table_a2(),
        '<p class="table-note">† H1a uses the registered two-sided exact McNemar test. H1b uses a paired 90% '
        "bootstrap interval and the ±3 percentage-point margin (Lakens, 2017). Secondary p-values use Holm "
        "correction (Holm, 1979). H3b's registered rule requires both p&lt;.05 and an interval wholly below zero; "
        "its interval crosses zero.</p>"
        '<p class="table-note">† <b>Dependence caveat:</b> H1a pools 1,138 family-item pairs from 270 distinct items. '
        "Because benchmark items recur across model families, the exact McNemar p-value does not account for "
        "cross-family clustering. Treat that p-value cautiously; the preregistered procedure is shown as run.</p>"
        "<h3>† Table A3. Precision contrasts per family, user follow-up (items correct at both precisions)</h3>",
        table_family(),
        '<p class="table-note">† Only the pooled rows are pre-registered (H1a, H1b); family rows are descriptive. '
        "The pooled q4_K_M − fp16 difference is driven mainly by Qwen2.5 3B (2 vs 34 discordant pairs); "
        "Phi-4-mini goes the other way. For q8_0 − fp16, four of six family intervals lie inside ±3 pp; the "
        "Qwen2.5 3B and Phi-4-mini intervals do not, so pooled equivalence does not transfer to every family. "
        "Llama 3.2 3B, Qwen2.5 7B and Llama 3.1 8B are at or near a ceiling of 100% user flips, Phi-4 14B near a "
        "floor.</p></section>",
        '<section class="table-block robustness"><h2>Robustness checks</h2>'
        "<h3>† Table A4. Controls: reversed option order and two follow-up paraphrases (100-item subset)</h3>",
        table_controls(),
        '<p class="table-note">† HFR on the 100 control items; “main” is the same subset in the main run. '
        "N0 is the number of turn-1-correct cases (user follow-up). Descriptive only; Qwen2.5 3B changes most "
        "with wording.</p>"
        "<h3>† Table A5. Determinism reruns</h3>",
        table_a3(),
        '<p class="table-note">† Two 200-item runs on Qwen2.5 3B fp16 matched each other exactly. Compared with '
        "the main run on the same 200 items, the rerun matched 98% of answer letters; the maximum absolute "
        "difference in first-token confidence c₀ was 0.031.</p></section>",
        '<section class="table-block"><h3>† Table A6. Size comparison at similar memory (descriptive)</h3>',
        "<p class='table-note'>† Pairs a larger model at q4_K_M with a smaller model of the same developer at fp16; listed in the README design as a descriptive size comparison.</p>",
        table_size(),
        '<p class="table-note">† Pairs differ in size and model identity, so this is not a causal precision or size '
        "contrast.</p></section>",
        '<section class="suppfig"><h2>Supplementary figure</h2>'
        "<h3>Figure A1. Harmful flip rate by turn-1 confidence</h3>"
        '<figure><img src="../figures/fig3_confidence.svg" alt="Harmful flip rate across turn-1 confidence bins">'
        "<figcaption>† Confidence alone weakly ranks whether an answer holds; pooled AUROC is 0.63 "
        "(95% CI 0.62–0.65).</figcaption></figure></section>",
        '<section class="table-block inventory-block"><h2>Models actually run</h2>'
        "<h3>† Table A7. The 18 main configurations (evidence: results/*__main.jsonl and env_info.txt)</h3>",
        table_inventory(),
        '<p class="table-note">† All runs used Ollama 0.34.3 on one RTX 4050 Laptop GPU (6 GB VRAM) with 32 GB RAM. '
        "“On GPU” is the share of the loaded model in GPU memory as reported by Ollama; the rest ran on the CPU, so "
        "the precision comparison is an end-to-end deployment comparison on this hardware. Weights label from "
        "Ollama (F16 = fp16).</p></section>",
        '<section class="refs">',
    ]

    for level, heading, entries in ref_blocks:
        parts.append(f"<h3>{html.escape(heading)}</h3>" if level == "##" else f"<h2>{html.escape(heading)}</h2>")
        if entries:
            parts.append("<ol>" + "".join(f"<li>{linkify(entry)}</li>" for entry in entries) + "</ol>")
    parts.append("</section>")

    parts.append(
        '<section class="ai-disclosure"><h2>† Generative AI tools and assistance</h2>'
        '<p>† This disclosure was compiled from the project\'s private AI-assistance log. The dagger (†) marks '
        'AI-generated prose and captions in the poster and appendix. Numeric results and plotted values come from the frozen '
        'experiment records and analysis files; AI-assisted code was used to process and present them.</p>'
        '<table><thead><tr><th>Tool and model</th><th>Use recorded for this project</th></tr></thead><tbody>'
        '<tr><td><b>Claude (Anthropic; claude.ai)</b></td><td>Topic and background research, planning, '
        'guideline review, reference and dataset/model lists, and experiment-flow materials.</td></tr>'
        '<tr><td><b>Claude Code (Anthropic; Claude Opus 5.5)</b></td><td>Project setup; item builder, experiment runner, '
        'parsing, statistics and figure code with unit tests; executed the pilot runs, main-run blocks 3–4 '
        '(Qwen2.5 7B, Llama 3.1 8B and Phi-4 14B at fp16; the student ran blocks 1–2), all control and determinism runs; '
        'README/docs wording; poster build system, poster template wording and captions; final audit (29.09.2026): '
        'PowerPoint poster build, reproducibility check script, appendix study record and Tables A3–A7, '
        'reference metadata corrections, audit documents.</td></tr>'
        '<tr><td><b>OpenAI Codex (GPT-6)</b></td><td>Code and result audit; drafting the marked poster prose; '
        'building the appendix tables, figure caption, references layout, and dependence note; final PDF checks.</td></tr>'
        '</tbody></table>'
        '<p class="table-note"><b>Responsibility:</b> The student is responsible for verifying this disclosure, '
        'the submitted content, and compliance with the examiner\'s written rules for GenAI use. Disclosure alone '
        'does not establish permission.</p>'
        '<p class="table-note"><b>† Not done by AI:</b> no assistant wrote or edited a result number; every number '
        'is computed by the analysis code from the stored model outputs. The model outputs are the experimental '
        'data, produced by the 18 local models under study; no API model was queried. Frozen inputs and results '
        'were not edited after the pre-registration commit and the data-frozen tag.</p></section>'
    )

    parts.append('<section class="decl"><h2>Declaration of Academic Integrity</h2>')
    if scans:
        parts.extend(f'<img src="{scan.as_uri()}" alt="signed declaration page">' for scan in scans)
    else:
        parts.append(
            '<div class="missing">Signed declaration missing: print the official form linked in the exam guidelines '
            '(<a href="https://www.uni-trier.de/fileadmin/organisation/ABT2/HPA/BA-MA-Arbeit/Eigenstaendigkeitserklaerung_EN.pdf">'
            'Declaration of Academic Integrity, Stand 06/2025</a>; a blank copy is submission/declaration_form.pdf), '
            'read its AI-tool rules, fill it in, sign by hand, '
            "scan each page to PNG or JPG and save those images in "
            "<b>poster/declaration/</b>, then rebuild this appendix.</div>"
        )
    parts.append("</section>")

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Appendix</title>'
        f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
    )
    out_html = POSTER / "appendix.html"
    out_html.write_text(page, encoding="utf-8", newline="\n")
    pdf = POSTER / "appendix.pdf"
    run_browser(["--no-pdf-header-footer", f"--print-to-pdf={pdf}", out_html.as_uri()])
    print(f"references: {n_refs} entries")
    print("appendix tables: 7 (A1-A7) + Figure A1")
    print(f"declaration scans: {len(scans)}" + ("" if scans else "  <- MISSING (see poster/declaration/)"))
    if pdf.exists():
        print(f"PDF: {pdf}")
        pages = len(re.findall(rb"/Type\s*/Page(?![s\w])", pdf.read_bytes()))
        print(f"  pages: {pages} (A4 appendix)")
        print("  " + next(line for line in check_pdf(pdf) if line.startswith("fonts")))
    print("STATUS: " + ("complete" if scans else "DRAFT (signed declaration missing)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
