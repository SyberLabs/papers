"""Deterministic and stochastic modulation signal sources.

Every source implements the same contract: given a timestep index, return a
scalar in roughly [-1, 1]. The generation loop multiplies this by a strength
epsilon and a mask to perturb logits. Keeping the interface identical across
chaotic and noise sources is what makes the conditions honestly comparable --
the only thing that differs between "chaotic modulation" and "matched noise"
is the *temporal autocorrelation* of the C_t stream, not the plumbing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Signal(Protocol):
    """A per-timestep scalar source, range approximately [-1, 1]."""

    def step(self) -> float:  # advance one timestep, return C_t
        ...

    def reset(self) -> None:
        ...


# --------------------------------------------------------------------------- #
# Chaotic sources
# --------------------------------------------------------------------------- #
@dataclass
class LogisticMap:
    """x_{n+1} = r x_n (1 - x_n), r=4 is fully chaotic on (0,1).

    Output is centered to ~[-1, 1]. Note: at r=4 the invariant density is the
    arcsine distribution (heavy at the edges), so the raw stream is NOT
    zero-mean uniform -- we record its empirical stats so the matched-noise
    control can be calibrated to the *same* mean/variance/autocorrelation.
    """

    r: float = 4.0
    x0: float = 0.5037  # avoid the unstable fixed points / period seeds
    _x: float = field(init=False)

    def __post_init__(self) -> None:
        self._x = self.x0

    def step(self) -> float:
        self._x = self.r * self._x * (1.0 - self._x)
        return 2.0 * self._x - 1.0  # map (0,1) -> (-1,1)

    def reset(self) -> None:
        self._x = self.x0


@dataclass
class HenonMap:
    """Henon attractor; we read out the x-coordinate, normalized."""

    a: float = 1.4
    b: float = 0.3
    x0: float = 0.0
    y0: float = 0.0
    _x: float = field(init=False)
    _y: float = field(init=False)
    # empirical x range for the canonical attractor, used to normalize
    _scale: float = 1.3

    def __post_init__(self) -> None:
        self._x, self._y = self.x0, self.y0

    def step(self) -> float:
        x_new = 1.0 - self.a * self._x * self._x + self._y
        y_new = self.b * self._x
        self._x, self._y = x_new, y_new
        return float(np.clip(self._x / self._scale, -1.0, 1.0))

    def reset(self) -> None:
        self._x, self._y = self.x0, self.y0


@dataclass
class LorenzOscillator:
    """Lorenz system, integrated with RK4; read out x, normalized.

    dt is the integration step taken per generated token. The classic chaotic
    regime (sigma=10, rho=28, beta=8/3) lives on the famous attractor; x
    typically ranges ~[-20, 20], so we normalize by `scale`.
    """

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.05
    x0: float = 1.0
    y0: float = 1.0
    z0: float = 1.0
    scale: float = 20.0
    _s: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._s = np.array([self.x0, self.y0, self.z0], dtype=np.float64)

    def _deriv(self, s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array(
            [self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]
        )

    def step(self) -> float:
        s = self._s
        k1 = self._deriv(s)
        k2 = self._deriv(s + 0.5 * self.dt * k1)
        k3 = self._deriv(s + 0.5 * self.dt * k2)
        k4 = self._deriv(s + self.dt * k3)
        self._s = s + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return float(np.clip(self._s[0] / self.scale, -1.0, 1.0))

    def reset(self) -> None:
        self._s = np.array([self.x0, self.y0, self.z0], dtype=np.float64)


@dataclass
class MultiFrequencyOscillator:
    """Sum of incommensurate sinusoids -- quasiperiodic, not chaotic.

    A useful intermediate control: structured and long-range correlated but
    with zero Lyapunov exponent. If chaotic maps beat this, the *sensitivity to
    initial conditions* / broadband spectrum is doing work, not mere periodicity.
    """

    freqs: tuple[float, ...] = (0.013, 0.0211, 0.0537)  # near-incommensurate
    phases: tuple[float, ...] = (0.0, 1.3, 2.7)
    _t: int = 0

    def step(self) -> float:
        val = sum(
            math.sin(2 * math.pi * f * self._t + p)
            for f, p in zip(self.freqs, self.phases)
        )
        self._t += 1
        return val / len(self.freqs)

    def reset(self) -> None:
        self._t = 0


# --------------------------------------------------------------------------- #
# Stochastic control sources
# --------------------------------------------------------------------------- #
@dataclass
class WhiteNoise:
    """IID uniform[-1,1]. The 'ordinary randomness' baseline for C_t."""

    seed: int = 0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def step(self) -> float:
        return float(self._rng.uniform(-1.0, 1.0))

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)


@dataclass
class OrnsteinUhlenbeck:
    """Mean-reverting AR(1)-style noise: dx = -theta*x*dt + sigma*dW.

    THE critical control. Calibrated (via `match_to`) so its mean, variance,
    and lag-1 autocorrelation match a target chaotic stream. If chaotic
    modulation does not beat this, the result is "temporally correlated noise
    helps," not "chaos helps."
    """

    theta: float = 0.05  # reversion rate -> sets autocorrelation timescale
    sigma: float = 0.3
    mu: float = 0.0
    dt: float = 1.0
    seed: int = 0
    _x: float = field(init=False)
    _rng: np.random.Generator = field(init=False)
    matched_rho1: float = field(default=float("nan"))  # diagnostic, set by match_to
    representable: bool = field(default=True)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._x = self.mu

    def step(self) -> float:
        dx = -self.theta * (self._x - self.mu) * self.dt + self.sigma * math.sqrt(
            self.dt
        ) * self._rng.standard_normal()
        self._x += dx
        return float(np.clip(self._x, -1.0, 1.0))

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._x = self.mu

    @classmethod
    def match_to(
        cls, target_stream: np.ndarray, seed: int = 0
    ) -> "OrnsteinUhlenbeck":
        """Build an OU process matching mean, variance, lag-1 autocorr of a
        reference stream. For an OU process, lag-1 autocorr rho1 = exp(-theta*dt)
        and stationary var = sigma^2 / (2 theta). Solve for theta, sigma."""
        x = np.asarray(target_stream, dtype=np.float64)
        mu = float(x.mean())
        var = float(x.var())
        xc = x - mu
        denom = float((xc[:-1] ** 2).sum())
        rho1 = float((xc[:-1] * xc[1:]).sum() / denom) if denom > 0 else 0.0
        # OU autocorrelation is exp(-theta) in [0,1). A negative or ~0 lag-1
        # autocorr (e.g. the r=4 logistic map, which is decorrelated) cannot be
        # represented by a stationary OU process -- the closest OU is essentially
        # white noise. We clamp into the representable band and flag it.
        # Below ~0.05 an OU process is indistinguishable from white noise over a
        # few-hundred-step generation, so we treat that as "not representable as
        # correlated noise" and flag it for the caller.
        repr_floor = 0.05
        eff_rho = min(max(rho1, 1e-2), 0.999)
        theta = -math.log(eff_rho)  # dt = 1
        # stationary variance of this discretization = sigma^2/(2 theta); invert:
        sigma = math.sqrt(max(var, 1e-9) * 2 * theta)
        ou = cls(theta=theta, sigma=sigma, mu=mu, dt=1.0, seed=seed)
        ou.matched_rho1 = rho1  # diagnostic: the target we tried to hit
        ou.representable = rho1 >= repr_floor
        return ou


@dataclass
class NullSignal:
    """Always zero -- used so baseline/sampling conditions share the loop."""

    def step(self) -> float:
        return 0.0

    def reset(self) -> None:
        pass


_REGISTRY = {
    "logistic": LogisticMap,
    "henon": HenonMap,
    "lorenz": LorenzOscillator,
    "multifreq": MultiFrequencyOscillator,
    "white": WhiteNoise,
    "ou": OrnsteinUhlenbeck,
    "null": NullSignal,
}


def make_signal(name: str, **kwargs) -> Signal:
    if name not in _REGISTRY:
        raise KeyError(f"unknown signal '{name}'; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def sample_stream(sig: Signal, n: int) -> np.ndarray:
    """Run a signal for n steps (then reset) -- for calibration & inspection."""
    sig.reset()
    out = np.fromiter((sig.step() for _ in range(n)), dtype=np.float64, count=n)
    sig.reset()
    return out
