# simplified code — kya hai aur kya badla

Yeh `submission\code` ki **copy** hai. Asal folder ko haath nahi lagaya gaya.
Is copy mein `cavein\` ka code aasaan tareeqe se likha gaya hai, jaise ek beginner likhta.
**Kaam bilkul wahi hai:** same tables, same figures, same result lines. Har cheez byte-by-byte check ki gayi hai (neeche dekhein).

Date: 29.09.2026 · AI-assisted (Claude Code): code simplification (type: code, no change in behaviour)

## Kya badla (sirf likhne ka tareeqa)

| Pehle | Ab |
|---|---|
| `lambda` (e.g. `sorted(..., key=lambda c: c["item_id"])`) | ek chhota naam wala function: `get_item_id(c)` |
| `divmod(n, k)` | `n // k` aur `n % k` alag alag |
| list/dict comprehensions (`[x for x in ...]`) | seedha `for` loop + `append` |
| `a if cond else b` ek line mein | poora `if / else` block |
| `zip(...)` se do lists saath chalana | `for i in range(len(...))` |
| `row.update({...})` ek lambi line | har field alag line pe (`record["a1"] = ...`) |
| `max(sums, key=sums.get)` | loop jo sab se bara letter dhoondta hai (barabri par pehla, pehle jaisa) |
| `nonlocal`, ek element wala loop (GEE) | seedha code |
| lambi functions (`secondary`, `hypothesis_summary`) | chhote hisson mein: `h2a_rows`, `confidence_q4_vs_fp16`, `h1a_row`, `h1b_row`, ... |
| kam comments | har step par aasaan English comment ("step 1: ...") |
| regex bina samjhaye | har regex ke upar ek misaal: `"Answer: B"`, `"(B)"`, ... |

Jin files mein yeh kaam hua: `cavein\` ke 16 modules (`config`, `stats`, `results`, `hypotheses`, `summary`,
`descriptive`, `exploratory`, `robustness`, `parsing`, `records`, `runner`, `backend`, `items`, `prompts`,
`dataset`, `figures`), aur `scripts\analyze.py`, `scripts\run_pushback.py`, `scripts\build_items.py` (sirf comments aur khali lines).

## Kya jaan boojh kar NAHI badla

- **Function ke naam aur arguments**: sab wahi hain, taake scripts aur tests bilkul waise hi chalein.
- **Math ki calls**: `numpy`/`pandas`/`statsmodels` ki har call wahi hai. Masalan `sum()` ko hath se likha hua loop nahi banaya, kyunki float numbers ka jod aakhri digit tak badal sakta hai.
- **Random ka order**: `dataset.py` mein `rng.random()`, `rng.sample()` aur `rng.shuffle()` usi tarteeb mein chalte hain. Warna 300 sawaalon ka sample badal jata.
- **Figures mein matplotlib calls ka order**: SVG ke andar ki ids isi order se banti hain.
- **Result line mein fields ki tarteeb** (`build_record`): result files ka format wahi hai.
- **Regex patterns**: ek character bhi nahi badla, sirf comments add kiye.
- **Frozen data** (`data\`), **results** (`results\`), **decoding settings** (temperature 0, seed 42, top-20, think off, 16 tokens): koi tabdeeli nahi.
- `scripts\verify_numbers.py` aur `scripts\verify_repro.py`: yeh "referee" hain jo code ko check karte hain, isliye inhein nahi badla.
- `poster\` ke build tools: yeh poster banate hain, experiment ka hissa nahi hain.

## Kaise check kiya ke kuch nahi toota

| Check | Natija |
|---|---|
| `python -m pytest tests` (89 unit tests + fake Ollama server) | **89 passed** |
| `analyze.py` + `make_figures.py` naye code se chalaye, output ko **asal** `submission\code\analysis` aur `figures` se milaya | **18 / 18 files byte-identical** (15 CSV + 3 SVG) |
| `scripts\verify_repro.py` | **PASS** (frozen data, 38 result files, rebuild byte-identical) |
| `scripts\verify_numbers.py` (raw replies se har number dobara gina, poster aur appendix samet) | **PASS**, 4,186 comparisons, 0 farq |
| Purana aur naya code ek hi input par: 15,154 replies (saare asli raw replies + 20,000 random mushkil texts), 5,000 log-prob lists, 9 dataset builds (3 seeds), 2,000 result lines, file names, model names, prompts | **dono ka output byte-identical** (3.8 MB) |
| Fake Ollama server par poora run (40 items main + 15 control items reversed, phir resume) purane aur naye code se | **220 result lines aur har request identical**; farq sirf progress line ke ETA timer mein ("0.1 min" vs "0.0 min") |

Note: `tests\` folder GitHub/backup se copy kiya gaya hai. Is mein se 2 tests (Cohen's h aur ECE) hata diye gaye hain,
kyunki woh statistics 29.09.2026 ko project se pehle hi nikaal di gayi thin.

## Chalane ka tareeqa (same as before)

```powershell
uv venv --python 3.11 .venv; .venv\Scripts\activate
uv pip install -r requirements.lock.txt
python -m pytest tests
python scripts/analyze.py          # results/  -> analysis/*.csv
python scripts/make_figures.py     # analysis/ -> figures/*.svg
python scripts/verify_repro.py
```

Poster aur appendix isi code se bante hain: `python poster\make_submission.py --student-id 1801156`
(PowerPoint aur Chrome chahiye). Is se `poster.pdf`, `appendix.pdf` aur `1801156.zip` bante hain.
Ek hi aur tabdeeli: `poster\build_pptx.py` ab git na milne par footer mein "numbers: commit unknown" nahi likhta.
