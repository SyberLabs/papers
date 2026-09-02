"""P8: M2 genuine component at matched reward_top_frac across NAEB landscapes."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from ..config import Config
from ..utils import get_logger
from .m2_analysis import TOST_DELTA_AUDC, TOST_DELTA_RATIONALE, paired_genuine_component
from .m2_ladder import M2_CONDITIONS, _m2_advantage_summary, _with_degree_random

log = get_logger("validation.m2_matched")


def run_matched_genuine(
    cfg: Config,
    reward_top_frac: float,
    force: bool = False,
    landscape_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run M2 controls at ``reward_top_frac`` and return (genuine, advantage)."""
    from ..simulation.study import run_controls

    cfg = copy.deepcopy(cfg)
    cfg.manifold.reward_top_frac = reward_top_frac
    label = landscape_label or Path(cfg.data.cache_dir).name

    log.info("matched genuine: landscape=%s reward_top_frac=%.2f", label, reward_top_frac)
    per_rep = run_controls(_with_degree_random(cfg), M2_CONDITIONS, force=force)
    genuine = paired_genuine_component(per_rep)
    advantage = _m2_advantage_summary(per_rep)

    for df in (genuine, advantage):
        df["landscape"] = label
        df["reward_top_frac"] = reward_top_frac

    real = advantage[advantage["condition"] == "real"]
    n_useful = None
    if len(per_rep):
        sub = per_rep[(per_rep["condition"] == "real") & (per_rep["strategy"] == "random")]
        if len(sub) and "n_total_useful" in sub.columns:
            n_useful = int(sub["n_total_useful"].iloc[0])
    genuine["n_useful"] = n_useful
    genuine["real_sensory_adv"] = float(
        real.loc[real["strategy"] == "sensory", "advantage_over_random"].iloc[0]
    ) if len(real[real["strategy"] == "sensory"]) else float("nan")

    return genuine, advantage


def run_matched_genuine_batch(
    configs: list[tuple[str, Config]],
    top_fracs: list[float],
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched genuine across landscapes × top_frac grid."""
    g_parts: list[pd.DataFrame] = []
    a_parts: list[pd.DataFrame] = []
    for label, cfg in configs:
        for tf in top_fracs:
            g, a = run_matched_genuine(cfg, tf, force=force, landscape_label=label)
            g_parts.append(g)
            a_parts.append(a)
    return pd.concat(g_parts, ignore_index=True), pd.concat(a_parts, ignore_index=True)


def summarize_matched_genuine(genuine: pd.DataFrame) -> pd.DataFrame:
    """Pivot sensory genuine component for enriched vs full at each top_frac."""
    rows = []
    for tf, grp in genuine.groupby("reward_top_frac"):
        row: dict = {"reward_top_frac": tf}
        for _, r in grp.iterrows():
            key = f"{r['landscape']}_{r['strategy']}"
            row[f"{key}_genuine"] = r["genuine_component_mean"]
            row[f"{key}_tost_p"] = r["tost_p_equivalent"]
            row[f"{key}_equiv"] = r["equivalent_at_delta"]
            if r["strategy"] == "sensory":
                row[f"{r['landscape']}_n_useful"] = r.get("n_useful")
        rows.append(row)
    return pd.DataFrame(rows)
