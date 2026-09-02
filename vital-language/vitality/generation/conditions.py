"""Map the experimental conditions to concrete (sampling, modulator, prompt-fn).

The six conditions (the spec's four plus the two controls we agreed matter):

  1. baseline        -- greedy / low temp, no modulation
  2. sampling        -- temperature+top-p, no modulation        (ordinary randomness)
  3. prompt_only     -- sampling + a "write vital/multifractal prose" instruction
  4. logit_chaos     -- sampling + chaotic logit modulation
  5. logit_matched   -- sampling + OU noise matched to the chaotic stream  (KEY CONTROL)
  6. logit_white     -- sampling + IID white-noise logit modulation        (lower control)

5 is the falsifiability control: same autocorrelation as chaos, no determinism.
6 separates "any logit noise" from "correlated logit noise".

All sampling-based conditions share identical SamplingParams so the only
varying factor is the C_t stream (and, for prompt_only, the instruction text).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..modulation.logit_injector import LogitModulator, ModulationConfig
from ..modulation.signals import make_signal, sample_stream, OrnsteinUhlenbeck
from .harness import SamplingParams

MULTIFRACTAL_INSTRUCTION = (
    "Write in a vital, multifractal style: let motifs recur at irregular "
    "intervals, vary your rhythm and sentence length, allow semantic drift "
    "that returns, resist premature closure, and sustain pressure across the "
    "whole passage rather than resolving each idea immediately.\n\n"
)


@dataclass
class Condition:
    name: str
    sampling: SamplingParams
    make_modulator: Callable[[], Optional[LogitModulator]]
    prompt_prefix: str = ""


def build_conditions(
    base_sampling: SamplingParams,
    eps: float = 0.5,
    chaotic_map: str = "lorenz",
    mask_mode: str = "topk_complement",
    mask_topk: int = 50,
    every_k: int = 1,
    warmup: int = 8,
    calib_len: int = 512,
    seed: int = 0,
) -> list[Condition]:
    """Construct the six conditions. The OU control is calibrated to the chosen
    chaotic map's empirical autocorrelation, so conditions 4 and 5 differ only
    in determinism vs. stochasticity at matched second-order statistics."""

    mod_cfg = ModulationConfig(
        eps=eps,
        every_k=every_k,
        warmup=warmup,
        mask_mode=mask_mode,
        mask_topk=mask_topk,
    )

    # Calibrate OU to the chaotic stream once, up front.
    chaos_stream = sample_stream(make_signal(chaotic_map), calib_len)
    ou_template = OrnsteinUhlenbeck.match_to(chaos_stream, seed=seed)

    greedy = SamplingParams(
        greedy=True,
        max_new_tokens=base_sampling.max_new_tokens,
        seed=base_sampling.seed,
    )

    def mod(signal_name: str, **kw):
        return lambda: LogitModulator(
            signal=make_signal(signal_name, **kw), config=mod_cfg
        )

    def mod_ou():
        return LogitModulator(
            signal=OrnsteinUhlenbeck(
                theta=ou_template.theta,
                sigma=ou_template.sigma,
                mu=ou_template.mu,
                seed=seed,
            ),
            config=mod_cfg,
        )

    return [
        Condition("baseline", greedy, lambda: None),
        Condition("sampling", base_sampling, lambda: None),
        Condition(
            "prompt_only",
            base_sampling,
            lambda: None,
            prompt_prefix=MULTIFRACTAL_INSTRUCTION,
        ),
        Condition("logit_chaos", base_sampling, mod(chaotic_map)),
        Condition("logit_matched", base_sampling, mod_ou),
        Condition("logit_white", base_sampling, mod("white", seed=seed)),
    ]
