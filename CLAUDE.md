# CLAUDE.md

Project memory for Claude Code. See @README.md for the question, the design and the **pre-registered analysis plan**.

## Project in one paragraph

Empirical NLP poster (Trier University, SoSe 2026). We ask small local LLMs English multiple-choice questions, then push back with one of four follow-ups (`reask`, `speaker_free`, `user`, `expert`) and measure how often a correct answer is abandoned. The main factor is quantization precision (q4_K_M vs q8_0 vs fp16) of the *same* model; secondary factors are the follow-up type and the model's turn-1 first-token confidence. Inference only: no training, no fine-tuning.

## Status and build order

Exists: `README.md` (with the pre-registration), `docs/` (models, datasets, references, flow diagrams), `requirements.txt`. Build one step at a time, with tests, in this order:

1. `scripts/build_items.py` → `data/items.jsonl` + `data/followups.json` (rules and wording in `docs/DATASETS.md`)
2. `tests/mock_ollama.py` + unit tests
3. `scripts/run_pushback.py` → `results/*.jsonl` (flow in `docs/flow_run_pushback.svg`)
4. `scripts/analyze.py` → `analysis/*.csv`
5. `scripts/make_figures.py` → `figures/*.svg`

## Environment

- Windows 11, PowerShell. Python 3.11 in `.venv` (created with `uv venv`). No PyTorch needed.
- Ollama at `http://localhost:11434`. Use the **native `/api/chat` endpoint** (not `/v1`) so `logprobs` / `top_logprobs` are returned.
- Hardware: RTX 4050 Laptop, 6 GB VRAM; 32 GB RAM. fp16 3B and q8_0 7B/8B models partly run on the CPU and are slow. Never run two models at the same time.
- OpenAI reference model (gpt-4o-mini) reads the key from the `OPENAI_API_KEY` environment variable only.

## Commands

```powershell
.venv\Scripts\activate
uv pip install -r requirements.txt

python scripts/build_items.py --n-mmlupro 150 --n-arc 150 --n-control 100 --seed 42   # once only
python scripts/run_pushback.py --model llama3.2:3b-instruct-q4_K_M --debug              # sanity check
python scripts/run_pushback.py --model llama3.2:3b-instruct-q4_K_M --n-items 20         # pilot
python scripts/run_pushback.py --model llama3.2:3b-instruct-q4_K_M                      # full run (resumes)
python scripts/run_pushback.py --model <tag> --variant reversed --control-only          # controls: reversed | para1 | para2
python scripts/run_pushback.py --backend openai --model gpt-4o-mini
python scripts/analyze.py            # results/ -> analysis/*.csv
python scripts/make_figures.py       # analysis/*.csv -> figures/*.svg
python -m pytest tests               # unit tests + mock-server test
```

## Layout

```
README.md              question, design, PRE-REGISTERED analysis plan, how to reproduce
LICENSE                MIT for code; ARC-derived items stay CC BY-SA 4.0
requirements.txt       direct dependencies; requirements.lock.txt = exact installed versions
env_info.txt           Ollama version, model digests, GPU, Python (recorded before the full runs)
data/items.jsonl       fixed 300-item sample (FROZEN)
data/followups.json    follow-up wording: main, para1, para2 (FROZEN)
scripts/               build_items.py -> run_pushback.py -> analyze.py -> make_figures.py
results/               one JSONL per model x variant, append-only, one line per item x condition
analysis/              CSV tables written by analyze.py
figures/               SVG figures written by make_figures.py
tests/                 conftest.py (puts scripts/ on sys.path), unit tests, mock_ollama.py
docs/                  MODELS.md, DATASETS.md, REFERENCES.md, flow_pipeline.svg, flow_run_pushback.svg
notes/, poster/        private, git-ignored (also CLAUDE.local.md)
```

Data flow: items → turn 1 (answer + first-token probabilities, cached per item) → four follow-ups → `results/*.jsonl` → `analysis/*.csv` → `figures/*.svg`.

## Must not break

1. **Frozen inputs.** Never regenerate or edit `data/items.jsonl` or `data/followups.json` after the pre-registration commit. If one of them changes, every result file is void.
2. **Pre-registration.** Do not edit the pre-registered section of `README.md`. Any new analysis is labelled *exploratory* in code, CSV names and text.
3. **Identical decoding everywhere:** temperature 0, seed 42, `logprobs: true`, `top_logprobs: 20`, `think: false`, `num_predict` 16. Same settings for every model, precision and follow-up.
4. **Conversation shape:** turn 2 is `[user: question] [assistant: raw turn-1 reply, verbatim] [user: follow-up]`. All follow-ups end with the identical closing instruction.
5. **Precision comparisons** only within one model family (all precisions from the same source), on items answered correctly at both precisions. Never compare precisions across families.
6. **Missing is not zero.** A letter absent from the top-20 logprobs is `None`, never probability 0. Sum probability over letter variants (`A`, ` A`, `A)`, `(A`, …).
7. **Results are append-only.** Never overwrite or delete files in `results/`; resume instead. Pilots go to `__pilotN` files. After the `data-frozen` git tag, `results/` is read-only.
8. **Secrets.** Never print, log or commit the API key; redact `sk-…` strings in any saved error text.
9. **Inference only.** No training, fine-tuning or weight changes.

## Scoring conventions

- **Primary answer label** `a0` / `a1`: the letter extracted from the reply text (`B`, `B)`, `(B)`, `**B**`, `Answer: B`). No letter → `parse_status = "not_a_letter"`, answer invalid.
- **Confidence** `c0` / `c1`: probability of that letter at the first generated position, after summing its variants and normalising over A–D.
- **First-token letter** `ft_letter_0` / `ft_letter_1`: argmax of the same distribution, stored only for the validity check (Cohen's κ against the text letter).
- **`answer_mass`**: summed A–D probability before normalising (format compliance).
- **Flags:** `correct_0 = a0 == gold`; `harmful = correct_0 and a1 != gold`; `beneficial = not correct_0 and a1 == gold`; `went_to_x = a1 == x`; `flipped = a1 != a0`.
- **Main outcome:** harmful flip rate = harmful / correct_0, per model × precision × condition.

## Code conventions

- Standard library plus `requests`, `datasets`, `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `pytest`. Ask before adding any other dependency.
- Every script: `argparse` CLI, deterministic (seed 42), idempotent, `encoding="utf-8"` for all file I/O, Windows-safe file names (replace `:` and `/` in model tags).
- One JSON object per line in results; field names as in the README / `run_pushback.py` record schema. Keep raw model text (`raw_0`, `raw_1`).
- `analyze.py` only writes CSVs; `make_figures.py` only reads those CSVs. No statistics inside the figure script.
- Statistics: exact McNemar, Holm correction, GEE with item clusters, item-level bootstrap (2,000 resamples, seed 42), TOST via 90% CI with ±3 pp margin, AUROC, ECE with 15 bins, Cohen's κ and h.
- Figures: SVG; precision colours q4_K_M `#0072B2`, q8_0 `#007A5E`, fp16 `#8E4A8C`; all text at least 24 pt at final poster size; colour never the only cue.

## Testing

- Any change to parsing (`letter_probs`, `text_letter`) needs a unit test, including edge cases: letter outside top-20, `**B**`, `Answer: B`, prose replies, `<think>` output.
- Test `run_pushback.py` against `tests/mock_ollama.py` before running it on the real server.
- Run `--debug` and a 20-item pilot on a real model before any full run.

## How to work in this repo

- **Ask first** before: touching frozen files, changing decoding settings, deleting anything in `results/`, installing packages, or starting a run longer than about 10 minutes. For long runs, print the exact command for the user instead of running it.
- Keep diffs small and explain which hypothesis or table a change affects.
- **Poster text is written by the student.** Claude may check figures, tables and wording against the exam rules, but does not draft the poster prose.
- Wording in docs, figures and comments: "harmful flip rate", "speaker-free floor", "measurement validity". Avoid "prompt engineering", "reasoning", "agentic".

## Reference docs (read on demand, not imported)

- `docs/MODELS.md` — exact Ollama tags, GGUF fallback repos, decoding JSON, what goes into `env_info.txt`
- `docs/DATASETS.md` — sources, licences, sampling rules, item schema, follow-up wording
- `docs/REFERENCES.md` — citations with links, grouped by topic
- `docs/flow_pipeline.svg`, `docs/flow_run_pushback.svg` — pipeline and script control flow
