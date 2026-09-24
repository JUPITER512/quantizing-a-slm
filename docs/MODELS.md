# Models

Hardware: RTX 4050 Laptop GPU (6 GB VRAM), 32 GB DDR5, i7-13620H. Everything runs through Ollama's native `/api/chat` endpoint; one reference model runs through the OpenAI API.

**Rule:** all precisions of one family come from the **same source** (Ollama's own library tags), so the only difference between them is quantization. If a tag is missing, switch the **whole family** to one Hugging Face GGUF repository — never mix sources inside a family.

## 1. Precision ladder (main test: RQ1–RQ3)

| Family | Ollama pull tags | Size approx. | Fits 6 GB? | Pages |
|---|---|---|---|---|
| Llama 3.2 3B Instruct | `llama3.2:3b-instruct-q4_K_M` · `llama3.2:3b-instruct-q8_0` · `llama3.2:3b-instruct-fp16` | 2.0 / 3.4 / 6.4 GB | q4, q8 fully; fp16 with small CPU offload | [Ollama](https://ollama.com/library/llama3.2) · [tags](https://ollama.com/library/llama3.2/tags) · [model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Qwen2.5 3B Instruct | `qwen2.5:3b-instruct-q4_K_M` · `qwen2.5:3b-instruct-q8_0` · `qwen2.5:3b-instruct-fp16` | 1.9 / 3.3 / 6.2 GB | same | [Ollama](https://ollama.com/library/qwen2.5) · [tags](https://ollama.com/library/qwen2.5/tags) · [model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |

## 2. Size check (RQ4, supports RQ1)

| Family | Ollama pull tags | Size approx. | Fits 6 GB? | Pages |
|---|---|---|---|---|
| Qwen2.5 7B Instruct | `qwen2.5:7b-instruct-q4_K_M` · `qwen2.5:7b-instruct-q8_0` | 4.7 / 8.1 GB | q4 yes; q8 with heavy offload | [Ollama](https://ollama.com/library/qwen2.5) · [model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Llama 3.1 8B Instruct | `llama3.1:8b-instruct-q4_K_M` · `llama3.1:8b-instruct-q8_0` | 4.9 / 8.5 GB | q4 yes; q8 with heavy offload | [Ollama](https://ollama.com/library/llama3.1) · [tags](https://ollama.com/library/llama3.1/tags) · [model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |

Your existing `qwen2.5:7b` and `llama3.1:8b` are the q4_K_M builds; confirm with `ollama show qwen2.5:7b` and reuse them.

## 3. Optional

| Model | Tag | Size | Note | Page |
|---|---|---|---|---|
| Gemma 4 12B | `gemma4:12b` (q4_K_M default) | 7.6 GB | Partial CPU offload; one precision only; 150 items overnight. **Set `"think": false`** | [Ollama](https://ollama.com/library/gemma4) · [12b tag](https://ollama.com/library/gemma4:12b) |
| gpt-4o-mini | OpenAI API | — | Non-quantized, larger reference point (§6) | [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations) · [pricing](https://openai.com/api/pricing/) |

## 4. Fallback: Hugging Face GGUF (only if an Ollama tag is missing)

Pull syntax: `ollama pull hf.co/<user>/<repo>:<quant>` ([Hugging Face docs](https://huggingface.co/docs/hub/en/ollama)). Check the exact file names on the repo's *Files* tab first; tags are case-insensitive, file names are not.

| Family | Repository (use the same one for all three precisions) |
|---|---|
| Llama 3.2 3B | [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) (Q4_K_M, Q8_0, f16) |
| Qwen2.5 3B | [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) (official) or [bartowski/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF) |
| Qwen2.5 7B | [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) (official) |
| Llama 3.1 8B | [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |

## 5. Technical reports and licences

| Family | Report | Licence (confirm on the model card) |
|---|---|---|
| Llama 3.1 / 3.2 | Abhimanyu Dubey et al. 2024. The Llama 3 Herd of Models. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) | Llama 3.1 / Llama 3.2 Community License; read its attribution clause |
| Qwen2.5 | An Yang et al. 2024. Qwen2.5 Technical Report. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) | 7B: Apache 2.0; 3B: Qwen research licence (check the card) |
| Gemma 4 | Google's Gemma 4 model card (via the Ollama page) | Apache 2.0 |
| GPT-4o mini | OpenAI. 2024. GPT-4o System Card. [arXiv:2410.21276](https://arxiv.org/abs/2410.21276) | API terms of use |
| GGUF k-quants | [llama.cpp](https://github.com/ggml-org/llama.cpp) · [arXiv:2601.14277](https://arxiv.org/abs/2601.14277) **(preprint)** | MIT (llama.cpp) |

## 6. OpenAI API reference run

- **Model:** gpt-4o-mini, or the cheapest current chat model that returns logprobs. Check the deprecations page first.
- **Calls:** 300 items × 5 = 1,500; about 0.45 M input tokens plus tiny outputs.
- **Cost:** well under $1 at gpt-4o-mini's list price ($0.15 per 1M input, $0.60 per 1M output tokens). Check current prices.
- **Settings:** temperature 0, seed 42 (best-effort on the API), `logprobs: true`, `top_logprobs: 20`, `max_completion_tokens: 16`.
- **Key:** `OPENAI_API_KEY` environment variable only; never in code, logs, screenshots or the repo.
- **Role:** reference point, not a replication target, and not a judge. Record the run date.

## 7. Decoding settings (identical for every call)

```json
{
  "stream": false,
  "think": false,
  "options": {"temperature": 0, "seed": 42, "num_predict": 16},
  "logprobs": true,
  "top_logprobs": 20
}
```

`think: false` only matters for models with a thinking mode (Gemma 4); it is harmless elsewhere. Ollama returns logprobs only for the top-k tokens (max 20; needs Ollama ≥ 0.12.11): a letter outside the top-20 is recorded as missing, not as zero.

## 8. Record for every configuration (`env_info.txt`)

- exact pull string and `ollama show <tag>` output (parameters, quantization, context length);
- digest from `ollama list`;
- `ollama --version`;
- CPU/GPU split from `ollama ps` while it runs (fp16 and q8 7B/8B are partly on the CPU — this affects speed, not the outputs you analyse);
- run date and time.

## 9. Disk space

| Group | New download |
|---|---|
| Llama 3.2 3B ladder | ~12 GB |
| Qwen2.5 3B ladder | ~11 GB |
| Qwen2.5 7B q8_0 + Llama 3.1 8B q8_0 | ~17 GB |
| Gemma 4 12B | ~8 GB |
| **Total** | **~48 GB** |

Models live in `C:\Users\<you>\.ollama\models`. To use another drive, set `OLLAMA_MODELS` before pulling and restart Ollama. If space is short, skip the two q8_0 size-check models first.

## 10. What not to use

- fp16 of 7B/8B models (~15–16 GB, almost all on the CPU) — unless 100 items overnight.
- Reasoning models through the API (hidden tokens would eat the $5).
- Different fine-tunes presented as "the same model at another precision".
