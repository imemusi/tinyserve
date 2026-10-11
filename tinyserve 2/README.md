# tinyserve

A small vLLM-style LLM inference server, built from scratch in Python and C++.

Roadmap: HF baseline -> own generate loop (matches HF token for token) -> paged KV cache (C++ block manager) -> continuous batching -> streaming FastAPI server.

## Layout

```
src/common.py               model loading, prompts, HF generate helper
benchmarks/run_baseline.py  Hugging Face "before" numbers
tests/reference_tokens.json greedy HF token IDs (answer key)
```

## Run the baseline (needs a GPU, e.g. a free Colab T4)

```bash
git clone <your-repo-url> && cd tinyserve
pip install -q transformers   # torch is preinstalled on Colab; restart the session if pip changes packages
python -m benchmarks.run_baseline
```

Then download `benchmarks/baseline_hf.json` and `tests/reference_tokens.json` and commit them.

## Baseline results (Hugging Face `generate`, GPT-2 124M)

| Mode | Tokens/sec | p50 latency | p99 latency | Peak GPU mem |
|---|---|---|---|---|
| Batched (16 prompts) | TBD | n/a | n/a | TBD |
| Sequential (1 at a time) | TBD | TBD | TBD | TBD |

GPU: TBD. Fill these in from `benchmarks/baseline_hf.json`.
