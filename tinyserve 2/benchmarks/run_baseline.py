"""Run the Hugging Face baseline and save the "before" numbers.

Usage (from the repo root):
    python -m benchmarks.run_baseline

Writes:
    benchmarks/baseline_hf.json   throughput, latency, peak GPU memory
    tests/reference_tokens.json   greedy token IDs, the answer key for your own generate loop
"""
import json
import statistics
import time
from pathlib import Path

import torch

from src.common import (
    MAX_NEW_TOKENS,
    MODEL_NAME,
    PROMPTS,
    get_device,
    hf_generate,
    load_model_and_tokenizer,
)

ROOT = Path(__file__).resolve().parent.parent


def timed(fn):
    """Time a GPU call correctly: synchronize before and after (CUDA is async)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def percentile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    device = get_device()
    model, tok = load_model_and_tokenizer(device)

    # Warmup: the first call pays for CUDA init and kernel setup.
    hf_generate(model, tok, PROMPTS[:2], device)
    torch.cuda.synchronize()

    # Test A: all 16 prompts in one batch -> throughput and memory.
    torch.cuda.reset_peak_memory_stats()
    out, batch_s = timed(lambda: hf_generate(model, tok, PROMPTS, device))
    batch_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    total_tokens = len(PROMPTS) * MAX_NEW_TOKENS

    # Save the answer key from this same run (greedy, so it is deterministic).
    ref_path = ROOT / "tests" / "reference_tokens.json"
    ref_path.parent.mkdir(exist_ok=True)
    ref_path.write_text(json.dumps({
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tok.eos_token_id,
        "prompts": PROMPTS,
        "token_ids": out.cpu().tolist(),  # left-padded: [pad..., prompt..., new...]
    }))

    # Test B: one request at a time -> per-request latency.
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for p in PROMPTS:
        _, dt = timed(lambda p=p: hf_generate(model, tok, [p], device))
        latencies.append(dt)
    seq_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    results = {
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "num_prompts": len(PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "batched": {
            "total_s": batch_s,
            "tokens_per_s": total_tokens / batch_s,
            "peak_mem_mb": batch_mem_mb,
        },
        "sequential": {
            "tokens_per_s": total_tokens / sum(latencies),
            "p50_latency_s": statistics.median(latencies),
            "p99_latency_s": percentile(latencies, 0.99),
            "peak_mem_mb": seq_mem_mb,
        },
    }

    out_path = ROOT / "benchmarks" / "baseline_hf.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved {out_path.relative_to(ROOT)} and {ref_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
