"""Model loading for small open-weight instruct models (CPU-friendly)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(name: str = DEFAULT_MODEL, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.to(device)
    model.eval()
    return model, tok
