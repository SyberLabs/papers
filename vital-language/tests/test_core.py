"""Fast unit tests for the non-model components."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitality.modulation.signals import (
    make_signal, sample_stream, OrnsteinUhlenbeck, LogisticMap,
)
from vitality.modulation.logit_injector import LogitModulator, ModulationConfig
from vitality.metrics.mfdfa import mfdfa


def _autocorr1(x):
    xc = x - x.mean()
    return float((xc[:-1] * xc[1:]).sum() / (xc[:-1] ** 2).sum())


def test_signals_bounded_and_deterministic():
    for name in ["logistic", "henon", "lorenz", "multifreq"]:
        s1 = sample_stream(make_signal(name), 500)
        s2 = sample_stream(make_signal(name), 500)
        assert np.all(np.abs(s1) <= 1.0 + 1e-9), f"{name} out of range"
        assert np.allclose(s1, s2), f"{name} not deterministic"
    print("PASS signals bounded + deterministic")


def test_ou_matches_lorenz_autocorr():
    # Lorenz is the meaningful driver: strong positive autocorr, so the OU
    # control is genuinely "correlated noise matched to chaos", not white noise.
    chaos = sample_stream(make_signal("lorenz"), 4000)
    ou = OrnsteinUhlenbeck.match_to(chaos, seed=1)
    ou_stream = sample_stream(ou, 4000)
    ac_c, ac_o = _autocorr1(chaos), _autocorr1(ou_stream)
    assert ou.representable, "lorenz autocorr should be representable by OU"
    assert abs(ac_c - ac_o) < 0.08, f"autocorr mismatch {ac_c:.3f} vs {ac_o:.3f}"
    assert abs(chaos.var() - ou_stream.var()) < 0.05, "variance mismatch"
    print(f"PASS OU<-lorenz match: ac chaos={ac_c:.3f} ou={ac_o:.3f}")


def test_logistic_is_decorrelated():
    # Documents the design finding: r=4 logistic map is decorrelated, so it is a
    # WEAK chaotic driver for this experiment -- its matched OU is white noise.
    chaos = sample_stream(LogisticMap(), 4000)
    assert abs(_autocorr1(chaos)) < 0.1, "logistic should be ~decorrelated"
    ou = OrnsteinUhlenbeck.match_to(chaos, seed=1)
    assert not ou.representable, "logistic match should be flagged non-representable"
    print(f"PASS logistic decorrelated (ac1={_autocorr1(chaos):.3f}), flagged")


def test_modulator_scale_relative_and_masking():
    torch.manual_seed(0)
    logits = torch.randn(1, 1000) * 3.0
    # null signal -> identity
    nullmod = LogitModulator(config=ModulationConfig(eps=0.0))
    assert torch.allclose(nullmod(logits.clone()), logits)
    # topk_complement promise: the protected top-k LOGIT VALUES are untouched
    # (we never push the model's confident choices around). It does NOT promise
    # the top-k set is frozen -- a tail token can be lifted into it; that's the
    # intended "structured turbulence". We verify both halves of that contract.
    mod = LogitModulator(
        signal=make_signal("lorenz"),
        config=ModulationConfig(eps=0.5, mask_mode="topk_complement",
                                mask_topk=10, warmup=0),
    )
    base = logits.clone()
    topk_idx = torch.topk(base[0], 10).indices
    out = mod(base.clone())
    assert torch.allclose(out[0, topk_idx], base[0, topk_idx]), "top-k values moved"
    # at moderate eps the greedy argmax should be preserved (coherence guard)
    assert torch.argmax(out[0]) == torch.argmax(base[0]), "argmax not preserved"
    print("PASS modulator identity + top-k value protection + argmax preserved")


def test_mfdfa_short_series_refuses():
    r = mfdfa(np.random.randn(30))
    assert not r.ok and np.isnan(r.width), "short series should refuse"
    print("PASS mfdfa refuses short series")


def test_mfdfa_monofractal_vs_multifractal():
    rng = np.random.default_rng(0)
    # white noise: near-monofractal, narrow width, Hurst ~ 0.5
    white = rng.standard_normal(2000)
    rw = mfdfa(white)
    assert rw.ok
    assert abs(rw.hurst - 0.5) < 0.2, f"white Hurst off: {rw.hurst:.3f}"
    # binomial multiplicative cascade: known multifractal, wide spectrum
    def cascade(depth, p=0.3):
        x = np.array([1.0])
        for _ in range(depth):
            x = np.concatenate([x * p, x * (1 - p)])
        return x
    casc = cascade(11)  # 2048 points
    rc = mfdfa(casc)
    assert rc.ok
    assert rc.width > rw.width, f"cascade width {rc.width:.3f} !> white {rw.width:.3f}"
    print(f"PASS mfdfa width: white={rw.width:.3f} < cascade={rc.width:.3f} "
          f"(white Hurst={rw.hurst:.3f})")


if __name__ == "__main__":
    test_signals_bounded_and_deterministic()
    test_ou_matches_lorenz_autocorr()
    test_logistic_is_decorrelated()
    test_modulator_scale_relative_and_masking()
    test_mfdfa_short_series_refuses()
    test_mfdfa_monofractal_vs_multifractal()
    print("\nALL CORE TESTS PASSED")
