"""Single generation loop shared by all experimental conditions.

Why a manual decode loop instead of model.generate():
  - we need per-step logit access for (a) injecting modulation and (b) recording
    the surprisal of the token actually emitted -- our primary MFDFA signal;
  - keeping ONE loop means conditions differ only in their sampling params and
    their modulator, never in incidental decoding behavior. That is essential
    for a clean comparison.

Surprisal recorded is the *post-modulation, pre-sampling* model belief about
the chosen token under the UNPERTURBED distribution, so the signal reflects the
text's predictability to the base model, not to the perturbed one. (We keep
both; see GenerationResult.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..modulation.logit_injector import LogitModulator


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    max_new_tokens: int = 256
    greedy: bool = False  # if True, argmax (ignores temperature)
    seed: int = 0


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    # surprisal_base: -log p(token) under the UNPERTURBED softmax (text dynamics)
    surprisal_base: np.ndarray
    # surprisal_eff: -log p(token) under the perturbed softmax (what was sampled)
    surprisal_eff: np.ndarray
    c_trace: np.ndarray  # modulation signal actually applied per step
    prompt: str
    condition: str
    meta: dict = field(default_factory=dict)


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.clone()
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]))[0][..., -1, None]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter_remove = remove.scatter(-1, sorted_idx, remove)
        logits[scatter_remove] = float("-inf")
    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    sampling: SamplingParams,
    condition: str,
    modulator: Optional[LogitModulator] = None,
    use_chat_template: bool = True,
    device: str = "cpu",
) -> GenerationResult:
    if modulator is not None:
        modulator.reset()

    if use_chat_template and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # generator must live on the same device as the probs tensor we sample from
    gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(sampling.seed)
    eos_id = tokenizer.eos_token_id

    past = None
    cur = input_ids
    out_ids: list[int] = []
    surp_base: list[float] = []
    surp_eff: list[float] = []
    t0 = time.time()

    for _ in range(sampling.max_new_tokens):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()  # [1, vocab]

        # unperturbed distribution -> defines text "predictability"
        base_logprobs = torch.log_softmax(logits, dim=-1)

        # ---- modulation (no-op for baseline/sampling conditions) ----
        if modulator is not None:
            logits = modulator(logits)

        # ---- temperature / truncation ----
        if sampling.greedy or sampling.temperature == 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            scaled = logits / max(sampling.temperature, 1e-5)
            scaled = _apply_top_k_top_p(scaled, sampling.top_k, sampling.top_p)
            probs = torch.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=gen)[:, 0]

        tid = int(next_id.item())
        eff_logprobs = torch.log_softmax(logits, dim=-1)
        surp_base.append(-base_logprobs[0, tid].item())
        surp_eff.append(-eff_logprobs[0, tid].item())
        out_ids.append(tid)

        if eos_id is not None and tid == eos_id:
            break
        cur = next_id.unsqueeze(0)

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    elapsed = time.time() - t0
    return GenerationResult(
        text=text,
        token_ids=out_ids,
        surprisal_base=np.asarray(surp_base, dtype=np.float64),
        surprisal_eff=np.asarray(surp_eff, dtype=np.float64),
        c_trace=(modulator.c_trace if modulator is not None else np.zeros(len(out_ids))),
        prompt=prompt,
        condition=condition,
        meta={
            "n_tokens": len(out_ids),
            "seconds": round(elapsed, 2),
            "tok_per_s": round(len(out_ids) / elapsed, 2) if elapsed > 0 else 0.0,
        },
    )
