"""Multifractal Detrended Fluctuation Analysis (MFDFA).

Implemented from scratch (Kantelhardt et al. 2002) so we control every choice.

CAVEAT WE TAKE SERIOUSLY: MFDFA needs a reasonably long series. With only a few
hundred tokens the high-q / large-scale fluctuation functions are noisy, and the
multifractal spectrum width can be inflated purely by finite-size effects --
which would be a beautiful way to fool ourselves into "chaos adds
multifractality." So:
  - we require a minimum length and a sane range of scales,
  - we return diagnostics (n, scale range, R^2 of the log-log fits),
  - the width should only be compared ACROSS conditions at MATCHED length.
Treat absolute width values with suspicion; treat between-condition deltas at
equal n as the real evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MFDFAResult:
    q: np.ndarray
    hq: np.ndarray  # generalized Hurst exponent h(q)
    tau: np.ndarray  # mass exponent tau(q) = q*h(q) - 1
    alpha: np.ndarray  # singularity strength (Legendre transform)
    f_alpha: np.ndarray  # singularity spectrum f(alpha)
    width: float  # alpha_max - alpha_min  (the headline multifractality number)
    hurst: float  # h(q=2), classic Hurst exponent
    n: int
    fit_r2: float  # mean R^2 of the log F_q(s) vs log s fits (quality flag)
    ok: bool  # False if series too short / degenerate -> do not trust
    note: str = ""


def _fit_loglog(scales: np.ndarray, fq: np.ndarray) -> tuple[float, float]:
    """Slope and R^2 of log Fq vs log s."""
    x, y = np.log(scales), np.log(fq)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[0]
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), r2


def mfdfa(
    series: np.ndarray,
    q_values: np.ndarray | None = None,
    order: int = 1,
    min_len: int = 120,
    n_scales: int = 12,
) -> MFDFAResult:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if q_values is None:
        q_values = np.linspace(-4, 4, 17)
    q_values = q_values[q_values != 0]  # q=0 handled by log-averaging variant; skip

    if n < min_len:
        return MFDFAResult(
            q_values, np.array([]), np.array([]), np.array([]), np.array([]),
            width=float("nan"), hurst=float("nan"), n=n, fit_r2=0.0, ok=False,
            note=f"series too short for MFDFA (n={n} < {min_len})",
        )

    # 1) profile: cumulative sum of mean-removed series
    y = np.cumsum(x - x.mean())

    # 2) scales: log-spaced between ~order+2 and n//4
    s_min = max(order + 2, 8)
    s_max = max(s_min + 1, n // 4)
    scales = np.unique(
        np.floor(np.logspace(np.log10(s_min), np.log10(s_max), n_scales)).astype(int)
    )
    if len(scales) < 4:
        return MFDFAResult(
            q_values, np.array([]), np.array([]), np.array([]), np.array([]),
            width=float("nan"), hurst=float("nan"), n=n, fit_r2=0.0, ok=False,
            note=f"too few usable scales (n={n})",
        )

    # 3) for each scale, detrend in non-overlapping windows (fwd + bwd) and get
    #    the q-th order fluctuation function F_q(s)
    fq = np.zeros((len(q_values), len(scales)))
    for si, s in enumerate(scales):
        n_seg = n // s
        # forward and backward segmentation (uses the whole series)
        var = []
        for seg_start in list(range(0, n_seg * s, s)) + list(
            range(n - n_seg * s, n, s)
        ):
            seg = y[seg_start : seg_start + s]
            if len(seg) < s:
                continue
            t = np.arange(s)
            coef = np.polyfit(t, seg, order)
            trend = np.polyval(coef, t)
            var.append(np.mean((seg - trend) ** 2))
        var = np.asarray(var)
        var = var[var > 0]
        if len(var) == 0:
            fq[:, si] = np.nan
            continue
        for qi, q in enumerate(q_values):
            fq[qi, si] = np.mean(var ** (q / 2.0)) ** (1.0 / q)

    # drop scales with any nan
    good = ~np.any(np.isnan(fq), axis=0)
    scales, fq = scales[good], fq[:, good]
    if fq.shape[1] < 4:
        return MFDFAResult(
            q_values, np.array([]), np.array([]), np.array([]), np.array([]),
            width=float("nan"), hurst=float("nan"), n=n, fit_r2=0.0, ok=False,
            note="degenerate fluctuation function",
        )

    # 4) h(q) = slope of log F_q(s) vs log s
    hq = np.zeros(len(q_values))
    r2s = np.zeros(len(q_values))
    for qi in range(len(q_values)):
        hq[qi], r2s[qi] = _fit_loglog(scales, fq[qi])

    # 5) Legendre transform -> singularity spectrum
    tau = q_values * hq - 1.0
    # alpha = d tau / d q  (numerical gradient)
    alpha = np.gradient(tau, q_values)
    f_alpha = q_values * alpha - tau
    width = float(np.nanmax(alpha) - np.nanmin(alpha))

    # h(q=2) via interpolation
    hurst = float(np.interp(2.0, q_values, hq))

    return MFDFAResult(
        q=q_values, hq=hq, tau=tau, alpha=alpha, f_alpha=f_alpha,
        width=width, hurst=hurst, n=n, fit_r2=float(np.mean(r2s)), ok=True,
        note="ok",
    )
