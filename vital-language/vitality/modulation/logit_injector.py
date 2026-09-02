"""Logit-level modulation: L'_t = L_t + eps * C_t * M_t.

Design decisions that matter:

1. SCALE-RELATIVE EPSILON. Raw logit magnitudes vary by step/model, so a fixed
   additive epsilon is meaningless across contexts. We express the perturbation
   in units of the current step's logit standard deviation. eps=0.5 then means
   "nudge by half the spread of this step's logits" -- comparable everywhere.

2. THE MASK IS WHERE STRUCTURE LIVES. Unmasked perturbation is just noise on
   the vocabulary; it mostly raises temperature. A motif mask biases the signal
   toward a *coherent subset* of tokens (a tracked semantic thread), so the
   chaotic stream reactivates dormant material instead of smearing everything.
   This is the operational version of the spec's H2 / section 8.

3. SCHEDULING. The signal can be gated (e.g. only fire every k tokens, or only
   after the passage has built some context) so modulation is bursty rather
   than constant -- itself a multifractal-friendly choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .signals import Signal, NullSignal


@dataclass
class ModulationConfig:
    eps: float = 0.0  # strength, in units of per-step logit std
    every_k: int = 1  # fire every k tokens (1 = always)
    warmup: int = 0  # no modulation for the first `warmup` tokens
    mask_mode: str = "none"  # 'none' | 'topk_complement' | 'motif'
    mask_topk: int = 50  # for topk_complement: protect the top-k, perturb rest
    clamp_std: float = 4.0  # never move a logit by more than this many stds


class LogitModulator:
    """Stateful per-generation logit perturber.

    A fresh instance should be used per generated passage (call .reset()).
    `motif_token_ids` (optional) is the set of vocabulary ids that constitute
    the currently-active motif thread; the mask concentrates the signal there.
    """

    def __init__(
        self,
        signal: Optional[Signal] = None,
        config: Optional[ModulationConfig] = None,
        motif_token_ids: Optional[torch.Tensor] = None,
    ):
        self.signal = signal if signal is not None else NullSignal()
        self.cfg = config if config is not None else ModulationConfig()
        self.motif_token_ids = motif_token_ids
        self._t = 0
        self._c_trace: list[float] = []  # record C_t actually applied

    def reset(self) -> None:
        self.signal.reset()
        self._t = 0
        self._c_trace = []

    @property
    def c_trace(self) -> np.ndarray:
        return np.asarray(self._c_trace, dtype=np.float64)

    def _build_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Return a [vocab] multiplicative mask in {0,1}-ish (float)."""
        vocab = logits.shape[-1]
        if self.cfg.mask_mode == "none":
            return torch.ones(vocab, device=logits.device, dtype=logits.dtype)

        if self.cfg.mask_mode == "topk_complement":
            # Protect the model's confident top-k; perturb the long tail. This
            # keeps local coherence (we don't fight the top choices) while
            # letting the signal reshuffle plausible alternatives.
            mask = torch.ones(vocab, device=logits.device, dtype=logits.dtype)
            topk = torch.topk(logits, min(self.cfg.mask_topk, vocab)).indices
            mask[topk] = 0.0
            return mask

        if self.cfg.mask_mode == "motif":
            # Concentrate the signal on the active motif thread: positive C_t
            # lifts motif tokens, encouraging delayed recurrence. Falls back to
            # uniform if no motif is currently active.
            mask = torch.zeros(vocab, device=logits.device, dtype=logits.dtype)
            if self.motif_token_ids is not None and len(self.motif_token_ids) > 0:
                mask[self.motif_token_ids.to(logits.device)] = 1.0
            else:
                mask += 1.0
            return mask

        raise ValueError(f"unknown mask_mode {self.cfg.mask_mode!r}")

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Perturb a [vocab] (or [1, vocab]) logit tensor in place-safe fashion."""
        squeeze = False
        if logits.dim() == 2:
            logits = logits[0]
            squeeze = True

        fire = (
            self.cfg.eps != 0.0
            and self._t >= self.cfg.warmup
            and (self._t % self.cfg.every_k == 0)
        )
        c = self.signal.step() if fire else 0.0
        self._c_trace.append(c)
        self._t += 1

        if fire and c != 0.0:
            std = logits.std().item()
            if std > 0:
                mask = self._build_mask(logits)
                delta = self.cfg.eps * c * std * mask
                delta = torch.clamp(
                    delta, -self.cfg.clamp_std * std, self.cfg.clamp_std * std
                )
                logits = logits + delta

        return logits.unsqueeze(0) if squeeze else logits
