---
title: Cave In at 4 Bits?
subtitle: Quantization precision, speaker-free pressure, and first-token confidence in answer flipping of local LLMs
authors: Syed Ali Murtaza Bokhari
affiliation: Trier University · Trends in Natural Language Processing · SoSe 2026
repo: https://github.com/JUPITER512/quantizing-a-slm
---

<!--
HOW TO USE THIS FILE
- This is the editable poster copy. Keep claims aligned with the frozen results and the registered analysis.
- Lines inside these comment brackets are notes for you; they never appear on the poster.
- Formatting: blank line = new paragraph; "- " at the start of a line = bullet point;
  **bold**, *italic*. Keep the headings (## ...) exactly as they are.
- Build:  .venv\Scripts\python.exe poster\build_poster.py
  The script prints the word count per box and warns if a box overflows or a placeholder is left.
  While a [WRITE ...] placeholder remains, the PDF carries a red "DRAFT" mark.
- Word budgets below are what fits at 25 pt. Less text is fine; more will overflow.
-->

## Hypothesis

<!--
About 90 words. Points to cover (your own words):
- The problem: people run 4-bit models on laptops/phones and argue with them.
- Prior work: answer flipping under pushback ("Are you sure?", Laban et al. 2024; Sharma et al. 2024);
  most conformity needs no speaker (Hu & Qu 2026); quantization and trust/confidence
  (Hong et al. 2024; Proskurina et al. 2024; Fu et al. 2025).
- The gap ("to our knowledge ...", see README "Question").
- Your research questions RQ1-RQ3 and the pre-registered primary hypothesis (H1a), see README.
- That the plan was pre-registered before the full runs (git commit 099408a).
-->

† Local models often run at 4 bits; users may challenge correct answers. Prior work finds flips after pushback (Laban et al., 2024; Sharma et al., 2024), conformity without a speaker (Hu & Qu, 2026), and quantization effects on trust or confidence (Hong et al., 2024; Proskurina et al., 2024). We test precision effects, speaker attribution beyond an alternative, and whether first-token confidence predicts holding. Pre-run H1a predicted q4_K_M would flip more than fp16 after a user challenge, among items correct at both precisions, pooled across six families.

## Methodology

<!--
About 60 words. The graphic above your text already shows: 300 items (150 MMLU-Pro reduced to
4 options + 150 ARC-Challenge), the four follow-ups with their exact wording, the six model
families x three precisions, and the decoding settings. Use your text to JUSTIFY the choices:
- why these two datasets (difficulty, contamination, enough correct turn-1 answers);
- why the same model at three precisions from one source (only quantization differs);
- why a speaker-free follow-up (separates the floor from social pressure);
- why the text letter is the label and the first-token probability the confidence;
- which tests (exact McNemar on paired items, TOST with ±3 pp, Holm, GEE with item clusters).
-->

† 150 four-option MMLU-Pro and 150 ARC-Challenge items sample hard questions with enough turn-1 correct answers. One Ollama source per family keeps precision ladders comparable. Four follow-ups separate re-asking, an unattributed alternative, user attribution and expert attribution. We score the visible letter; first-token probability is confidence. Tests: paired McNemar, ±3 pp equivalence, Holm and item-clustered GEE.

## Results

<!--
About 100 words. Discuss the findings against your hypotheses. Questions to answer:
- Which pre-registered rules are met (Table 1)? Is the H1a difference in the predicted direction?
- Which family drives the pooled result (Fig. 1)? Which families sit at the ceiling?
- What does the speaker-free vs re-ask comparison say about RQ2 (H2a; Qwen2.5 3B is the exception)?
- What is unexpected in Fig. 2 (re-ask vs expert at q4_K_M)? Say it is exploratory.
- Does turn-1 confidence protect against flipping (H3a, repo figure fig3_confidence.svg)?
Numbers: analysis/hypothesis_summary.csv, analysis/descriptives.csv,
analysis/exploratory_precision_by_condition.csv.
-->

† Against H1a, pooled q4_K_M flipped less often than fp16 after the user claim (59.7% vs 62.6%; −2.9 pp, p<.001), opposite the prediction. Qwen2.5 3B drove this gap; three families had ceiling-level user flips. H1b equivalence held only in the pooled estimate. Speaker-free flips exceeded re-ask in 15/18 configurations (H2a), except Qwen2.5 3B; precision interacted with follow-up (H2b). Confidence missed H3a's AUROC target (0.63). H3b's Wilcoxon p was <.001, but its interval crossed zero, so its rule failed. Exploratory: q4 raised re-ask flips 5.5 pp but lowered expert flips 4.7 pp vs fp16.

## Limitations

<!--
About 45 words. Candidates (pick the important ones): scripted follow-ups, not real users;
one GGUF source (k-quants) per family, no GPTQ/AWQ; four-option multiple choice; possible
contamination; top-20 log-probs only; ceiling in three families; wording sensitivity
(analysis/robustness_controls.csv); determinism: identical within a session, 98 % identical
across sessions (analysis/robustness_determinism.csv); Phi-4 14B answers ~13 % of turn-1
questions in prose; size comparison (RQ4) is descriptive only.
-->

† Scripted four-option prompts may not reflect real dialogue. McNemar pools family-item pairs although benchmark items repeat across families, so its p-value may understate uncertainty. Speaker-free wording still uses a user-role message. FP16 and larger models used partial CPU offload.

## Take-home

<!-- One or two sentences (at most 25 words): your answer to the title question. -->

† Not in this sample: pooled q4 flipped less than fp16; follow-up wording and model family shaped the outcome.

## Figure 1 caption

† Figure 1 · Harmful flip rate = share of correct turn-1 answers that are wrong after the follow-up; bars: 95 % CI.

## Figure 2 caption

† Figure 2 · Difference to fp16 in harmful flip rate, pooled over the six families; only the user row is pre-registered.


## Table 1 caption

† Table 1 · Pre-registered hypotheses and results (paired items correct in turn 1 at both precisions).
