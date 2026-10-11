"""Shared helpers for the HF baseline: model loading, prompts, and generation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 128

_BASE_PROMPTS = [
    "The future of artificial intelligence is",
    "In a small village by the sea,",
    "The key difference between a process and a thread is",
    "Once upon a time, a robot learned to",
    "To make a great cup of coffee, you should",
    "The history of the Roman Empire begins",
    "A distributed system is",
    "Today I want to talk about",
]
PROMPTS = _BASE_PROMPTS * 2  # 16 prompts


def get_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    return "cuda"


def load_model_and_tokenizer(device: str):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token
    tok.padding_side = "left"      # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    return model, tok


@torch.no_grad()
def hf_generate(model, tok, prompts, device, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy generation with exactly `max_new_tokens` new tokens per prompt.

    Returns the full output tensor: [batch, prompt_len + max_new_tokens].
    """
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    return model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # fixed length so token counts are exact
        do_sample=False,                # greedy = deterministic
        pad_token_id=tok.eos_token_id,
    )
