"""Command-line interface for the Green Hypercube simulator.

Commands
--------
- ``info``  : build the manifold and print its structure.
- ``build`` : (re)materialize the data cache only.
- ``run``   : run a full strategy-comparison experiment from a config.
- ``sweep`` : run a sensitivity sweep over one parameter and plot it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import typer

from .config import Config
from .data import ingest
from .data import schema
from .hypercube import build_manifold
from .simulation.study import run_study, run_controls
from .simulation.metrics import aggregate, summarize
from .validation import NULLS, null_advantage_table, measure_coupling, measure_multivariate_coupling, measure_phylo_community
from .utils.rng import make_rng
from .analysis import (
    plot_discovery_curves,
    plot_summary_bars,
    plot_sensitivity,
    plot_null_controls,
    plot_coupling,
    plot_multivariate_coupling,
    plot_m2_advantage,
)
from .utils import get_logger

app = typer.Typer(add_completion=False, help="Searching the Green Hypercube.")
log = get_logger("cli")


def _load_config(config: str | None) -> Config:
    if config:
        return Config.from_yaml(config)
    log.info("no config given; using built-in defaults")
    return Config()


def _prepare_manifold(cfg: Config, force: bool):
    cache = ingest(cfg.data, cfg.seed, force=force)
    return build_manifold(cache, cfg.manifold)


@app.command()
def build(
    config: str = typer.Option(None, help="Path to a YAML config."),
    force: bool = typer.Option(False, help="Rebuild the cache even if present."),
):
    """Materialize (or refresh) the normalized data cache."""
    cfg = _load_config(config)
    ingest(cfg.data, cfg.seed, force=force)
    typer.echo(f"cache ready at {Path(cfg.data.cache_dir).resolve()}")


@app.command()
def info(
    config: str = typer.Option(None, help="Path to a YAML config."),
    force: bool = typer.Option(False, help="Rebuild the cache."),
):
    """Build the manifold and report its structure."""
    cfg = _load_config(config)
    m = _prepare_manifold(cfg, force)
    typer.echo("Green Hypercube manifold")
    typer.echo(f"  species         : {m.n}")
    typer.echo(f"  features        : {m.X.shape[1]} ({', '.join(m.feature_names[:4])}, ...)")
    typer.echo(f"  useful species  : {m.n_useful} ({100 * m.useful_mask.mean():.1f}%)")
    typer.echo(f"  total reward    : {m.total_reward:.2f}")
    typer.echo(f"  eco edges       : {m.eco.number_of_edges()}")
    typer.echo(f"  bio edges       : {m.bio.number_of_edges()}")
    typer.echo(f"  phylo distance  : median={_median_phylo(m):.2f}")


@app.command()
def run(
    config: str = typer.Option(None, help="Path to a YAML config."),
    force: bool = typer.Option(False, help="Rebuild the cache."),
    output: str = typer.Option(None, help="Override output directory."),
):
    """Run the full strategy-comparison experiment."""
    cfg = _load_config(config)
    if output:
        cfg.output_dir = output
    if not cfg.strategies:
        raise typer.BadParameter("config has no strategies; see configs/ for examples")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = run_study(cfg, force=force)

    summary = summarize(exp["results"])
    agg = aggregate(summary)
    summary.to_csv(out_dir / "summary_replicates.csv", index=False)
    agg.to_csv(out_dir / "summary_aggregate.csv", index=False)

    curve_df = pd.DataFrame(exp["curves"])
    curve_df.index.name = "experiment"
    curve_df.to_csv(out_dir / "discovery_curves.csv")

    plot_discovery_curves(exp["curves"], exp["n_useful"], out_dir / "discovery_curves.png")
    plot_summary_bars(summary, out_dir / "summary_bars.png")
    cfg.to_yaml(out_dir / "config.used.yaml")

    typer.echo("\n=== Aggregate results (sorted by AUDC) ===")
    typer.echo(agg.to_string(index=False))
    typer.echo(f"\nwrote results + figures to {out_dir.resolve()}")


@app.command()
def sweep(
    param: str = typer.Option(..., help="signal_strength | reward_density | reward_top_frac | observation_noise."),
    values: str = typer.Option(..., help="Comma-separated values, e.g. '0.0,0.25,0.5,1.0'."),
    config: str = typer.Option(None, help="Path to a YAML config."),
    output: str = typer.Option(None, help="Override output directory."),
):
    """Sweep a single parameter and plot strategy AUDC against it.

    Key rigor checks:
    - ``signal_strength`` (synthetic): at 0 every strategy must match random; the
      advantage should grow with the planted cue-reward coupling.
    - ``reward_top_frac`` (any source, esp. live): imposes increasing reward
      sparsity without touching the cache, testing whether the structured
      advantage survives a real needle-in-haystack regime.
    """
    base = _load_config(config)
    if output:
        base.output_dir = output
    out_dir = Path(base.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_params = {"signal_strength", "reward_density"}
    manifold_params = {"reward_top_frac"}
    if param not in (data_params | manifold_params | {"observation_noise"}):
        raise typer.BadParameter(
            "param must be signal_strength, reward_density, reward_top_frac, "
            "or observation_noise"
        )

    vals = [float(v) for v in values.split(",")]
    rows = []
    for v in vals:
        cfg = copy.deepcopy(base)
        force = param in data_params  # only data-generation changes rebuild the cache
        if param == "signal_strength":
            cfg.data.signal_strength = v
        elif param == "reward_density":
            cfg.data.reward_density = v
        elif param == "reward_top_frac":
            cfg.manifold.reward_top_frac = v
        elif param == "observation_noise":
            cfg.simulation.observation_noise = v
        log.info("sweep %s=%s", param, v)
        exp = run_study(cfg, force=force)
        s = summarize(exp["results"])
        s[param] = v
        rows.append(s)

    full = pd.concat(rows, ignore_index=True)
    full.to_csv(out_dir / f"sweep_{param}.csv", index=False)
    plot_sensitivity(full, param, out_dir / f"sweep_{param}.png")
    typer.echo(f"wrote sweep results to {out_dir.resolve()}")


@app.command()
def controls(
    config: str = typer.Option(None, help="Path to a YAML config."),
    nulls: str = typer.Option(
        "permute_reward,rewire_graphs,shuffle_phylo",
        help="Comma-separated null models to run.",
    ),
    force: bool = typer.Option(False, help="Rebuild the cache."),
    output: str = typer.Option(None, help="Override output directory."),
):
    """Run the negative-control battery: real landscape vs. structure ablations.

    Healthy output: structured strategies show a large advantage over random
    under 'real' that collapses to ~0 under 'permute_reward' (and under the
    cue-specific nulls for the cue they rely on).
    """
    cfg = _load_config(config)
    if output:
        cfg.output_dir = output
    if not cfg.strategies:
        raise typer.BadParameter("config has no strategies")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in nulls.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in NULLS]
    if unknown:
        raise typer.BadParameter(f"unknown nulls {unknown}; choose from {sorted(NULLS)}")

    per_rep = run_controls(cfg, conditions, force=force)
    adv = null_advantage_table(per_rep)
    per_rep.to_csv(out_dir / "controls_replicates.csv", index=False)
    adv.to_csv(out_dir / "controls_advantage.csv", index=False)
    plot_null_controls(per_rep, out_dir / "controls.png")

    typer.echo("\n=== Advantage over random (AUDC) by condition ===")
    typer.echo(adv.to_string(index=False))
    typer.echo(f"\nwrote control results + figure to {out_dir.resolve()}")


@app.command(name="refresh-reward")
def refresh_reward_cmd(
    config: str = typer.Option(None, help="Path to a YAML config."),
):
    """Recompute only the reward (bioassay) on the cached landscape.

    Use this to complete a run after a transient ChEMBL outage: it reuses the
    cached species + compound bridge and rewrites the bioassay table only, so the
    flora and cues (and thus the landscape) are unchanged.
    """
    from .data.ingest import refresh_reward

    cfg = _load_config(config)
    refresh_reward(cfg.data)
    typer.echo(f"reward refreshed on cache at {Path(cfg.data.cache_dir).resolve()}")


@app.command()
def coupling(
    config: str = typer.Option(None, help="Path to a YAML config."),
    n_perm: int = typer.Option(999, help="Label-permutation null draws."),
    force: bool = typer.Option(False, help="Rebuild the cache."),
    output: str = typer.Option(None, help="Override output directory."),
    control: bool = typer.Option(
        False, help="Also report coverage-controlled coupling (partial on "
        "occurrence/interaction/chemistry counts)."),
    residual_reward: bool = typer.Option(
        False, help="Residualize reward on documentation depth before coupling "
        "(reward* = reward - E[reward | reward_depth])."),
    multivariate: bool = typer.Option(
        False, help="Also test joint coupling across all four channel cues "
        "(rank partial R² with block permutation null)."),
):
    """Measure the cue->reward coupling actually present in the landscape.

    This is the anti-tautology check: the synthetic study *plants* coupling
    (AUDIT), so only the measured magnitude on a real landscape can stand as
    evidence. A channel whose observed Spearman coupling exits its permutation
    null band (p < 0.05) carries exploitable signal; one inside the band does
    not -- which bounds how much advantage any strategy on that cue can be real.

    With ``--control`` it additionally partials out research-coverage covariates,
    so a coupling that is really just "well-studied plants are bioactive" is
    exposed as it collapses toward zero.

    With ``--residual-reward`` it removes the component of reward predictable
    from reward-side documentation depth (NAEB uses/tribes/sources; ChEMBL
    compound-bridge counts) before measuring coupling: the symmetric counterpart
    to ``--control``.

    With ``--multivariate`` it tests whether sensory, phylo, eco, and bio cues
    jointly predict reward (partial R²), addressing whether integrative strategies
    can exploit multivariate structure missed by single-channel tests.
    """
    cfg = _load_config(config)
    if output:
        cfg.output_dir = output
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = _prepare_manifold(cfg, force)

    def _suffix(base: str = "coupling") -> str:
        parts = []
        if multivariate:
            parts.append("multivariate")
        if residual_reward:
            parts.append("residual")
        if control:
            parts.append("controlled")
        return f"{base}_{'_'.join(parts)}" if parts else base

    if residual_reward:
        tbl = measure_coupling(
            m, make_rng(cfg.seed + 5), n_perm=n_perm,
            control=control, residual_reward=True,
        )
        suffix = "_residual"
        if control:
            suffix += "_controlled"
        tbl.to_csv(out_dir / f"coupling{suffix}.csv", index=False)
        plot_coupling(tbl, out_dir / f"coupling{suffix}.png")
        label = "reward-residualized"
        if control:
            label += " + coverage-controlled"
        typer.echo(f"\n=== {label} cue->reward coupling (univariate) ===")
        typer.echo(tbl.to_string(index=False))
    else:
        tbl = measure_coupling(m, make_rng(cfg.seed + 5), n_perm=n_perm)
        tbl.to_csv(out_dir / "coupling.csv", index=False)
        plot_coupling(tbl, out_dir / "coupling.png")

        typer.echo("\n=== Measured cue->reward coupling (Spearman vs. permutation null) ===")
        typer.echo(tbl.to_string(index=False))

        if control:
            ctl = measure_coupling(m, make_rng(cfg.seed + 5), n_perm=n_perm, control=True)
            ctl.to_csv(out_dir / "coupling_controlled.csv", index=False)
            merged = tbl[["channel", "observed", "p_value"]].merge(
                ctl[["channel", "observed", "p_value"]],
                on="channel", suffixes=("_raw", "_controlled"))
            typer.echo("\n=== Coverage-controlled coupling (partial on study effort) ===")
            typer.echo(merged.to_string(index=False))

    if multivariate:
        mv = measure_multivariate_coupling(
            m, make_rng(cfg.seed + 6), n_perm=n_perm,
            control=control, residual_reward=residual_reward,
        )
        mv_path = out_dir / f"{_suffix()}.csv"
        mv.to_csv(mv_path, index=False)
        plot_multivariate_coupling(mv, out_dir / f"{_suffix()}.png")
        mv_label = "Multivariate (partial R²)"
        if residual_reward:
            mv_label += ", reward-residualized"
        if control:
            mv_label += ", coverage-controlled"
        typer.echo(f"\n=== {mv_label} ===")
        typer.echo(mv.to_string(index=False))

    typer.echo(f"\nwrote coupling table + figure to {out_dir.resolve()}")


@app.command(name="m2-ladder")
def m2_ladder_cmd(
    config: str = typer.Option(None, help="Path to a YAML config."),
    force: bool = typer.Option(False, help="Rebuild the cache."),
    output: str = typer.Option(None, help="Override output directory."),
):
    """M2 decomposition: partition structured-search advantage by null type.

    Runs four landscape conditions on cached data:

    1. **real**: observed pool
    2. **permute_reward**: destroys all reward-linked signal (leakage audit)
    3. **permute_reward_within_effort**: preserves reward–effort link (effort-tracking test)
    4. **permute_features**: shuffles observable cues (feature-linked signal test)

    Also adds **degree_random** (topology-weighted baseline, rung 4) and reports
    depth-stratified coupling tables for the ChEMBL identifiability check.
    """
    from .validation.m2_ladder import run_m2_ladder

    cfg = _load_config(config)
    if output:
        cfg.output_dir = output
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_m2_ladder(cfg, force=force)
    for name, df in results.items():
        if name == "high_doc_eco":
            for layer, table in df.items():
                if len(table):
                    table.to_csv(out_dir / f"high_doc_eco_{layer}.csv", index=False)
            continue
        if name in ("tost_delta", "tost_delta_rationale"):
            continue
        if hasattr(df, "to_csv"):
            df.to_csv(out_dir / f"{name}.csv", index=False)

    plot_m2_advantage(results["advantage"], out_dir / "m2_advantage.png")

    typer.echo("\n=== M2 advantage over random (key strategies) ===")
    key = results["advantage"]
    key = key[key["strategy"].isin(["ecological", "cultural", "sensory", "degree_random"])]
    typer.echo(key.to_string(index=False))

    from .validation.m2_analysis import TOST_DELTA_AUDC, TOST_DELTA_RATIONALE

    typer.echo("\n=== Genuine component: paired(real - effort-stratified null) ===")
    typer.echo(f"TOST delta (SESOI): {TOST_DELTA_AUDC} AUDC: {TOST_DELTA_RATIONALE}")
    typer.echo(results["genuine_component"].to_string(index=False))

    typer.echo("\n=== Depth-bin reward diagnostics (vacuity check) ===")
    typer.echo(results["stratified_bin_diagnostics"].to_string(index=False))

    typer.echo("\n=== Stratified coupling FP context (informative cells only) ===")
    typer.echo(results["stratified_fp_context"].to_string(index=False))

    typer.echo("\n=== High-documentation eco (label-noise robustness, bins 3-4) ===")
    for layer in ("raw", "controlled", "controlled_residual", "multivariate_controlled"):
        tbl = results["high_doc_eco"].get(layer)
        if tbl is not None and len(tbl):
            typer.echo(f"\n--- {layer} ---")
            typer.echo(tbl.to_string(index=False))

    if len(results.get("high_doc_eco_decomposition", [])):
        typer.echo("\n=== High-doc eco decomposition (depth confound vs redundancy) ===")
        typer.echo(results["high_doc_eco_decomposition"].to_string(index=False))

    typer.echo(f"\nwrote M2 ladder results to {out_dir.resolve()}")


@app.command(name="matched-genuine")
def matched_genuine_cmd(
    configs: str = typer.Option(
        "configs/live_naeb_nam.yaml,configs/live_naeb_full_flora.yaml",
        help="Comma-separated YAML configs (enriched, full flora, ...).",
    ),
    top_frac: str = typer.Option(
        "0.2,0.35",
        help="Comma-separated reward_top_frac values to match across landscapes.",
    ),
    force: bool = typer.Option(False, help="Rebuild caches."),
    output: str = typer.Option("results/p8_matched_genuine", help="Output directory."),
):
    """P8: M2 genuine component at matched reward_top_frac (sensory TOST re-test)."""
    from .validation.m2_matched import run_matched_genuine_batch, summarize_matched_genuine
    from .validation.m2_analysis import TOST_DELTA_AUDC, TOST_DELTA_RATIONALE

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_pairs: list[tuple[str, Config]] = []
    for path in [p.strip() for p in configs.split(",") if p.strip()]:
        cfg = Config.from_yaml(path)
        label = Path(path).stem.replace("live_", "")
        cfg_pairs.append((label, cfg))

    fracs = [float(v) for v in top_frac.split(",")]
    genuine, advantage = run_matched_genuine_batch(cfg_pairs, fracs, force=force)
    summary = summarize_matched_genuine(genuine)

    genuine.to_csv(out_dir / "matched_genuine_component.csv", index=False)
    advantage.to_csv(out_dir / "matched_advantage.csv", index=False)
    summary.to_csv(out_dir / "matched_genuine_summary.csv", index=False)

    typer.echo("\n=== P8: Genuine component at matched reward_top_frac ===")
    typer.echo(f"TOST delta: {TOST_DELTA_AUDC}: {TOST_DELTA_RATIONALE}")
    typer.echo(genuine.to_string(index=False))
    typer.echo("\n=== Summary (sensory focus) ===")
    typer.echo(summary.to_string(index=False))
    typer.echo(f"\nwrote P8 results to {out_dir.resolve()}")


@app.command(name="phylo-community")
def phylo_community_cmd(
    config: str = typer.Option(None, help="Path to a YAML config."),
    n_perm: int = typer.Option(999, help="Null draws for NRI/NTI and hot-node tests."),
    force: bool = typer.Option(False, help="Rebuild the cache."),
    output: str = typer.Option(None, help="Override output directory."),
    min_clade_tips: int = typer.Option(4, help="Minimum tips for internal clade tests."),
    min_genus_tips: int = typer.Option(3, help="Minimum tips for genus enrichment tests."),
):
    """P7: Saslis-style NRI/NTI and hot-node enrichment on NAEB use labels.

    Runs clade-level community metrics (distinct from pairwise H2.2 coupling):
    raw documented-use labels, documentation-residual labels when reward_depth
    exists, and effort-matched nulls for NRI/NTI.
    """
    cfg = _load_config(config)
    if output:
        cfg.output_dir = output
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = ingest(cfg.data, cfg.seed, force=force)
    m = build_manifold(cache, cfg.manifold)
    newick = cache.read_text(schema.PHYLOGENY_NAME)

    summary, hot, fp_context = measure_phylo_community(
        m, newick, make_rng(cfg.seed + 17), n_perm=n_perm,
        min_clade_tips=min_clade_tips, min_genus_tips=min_genus_tips,
    )
    summary.to_csv(out_dir / "phylo_community_summary.csv", index=False)
    hot.to_csv(out_dir / "phylo_community_hot_nodes.csv", index=False)
    fp_context.to_csv(out_dir / "phylo_community_fp_context.csv", index=False)

    typer.echo("\n=== Phylogenetic community metrics (P7 / Saslis estimand) ===")
    typer.echo("(NRI/NTI p-values: two-sided permutation rank, not normal z→p)")
    typer.echo(summary.to_string(index=False))

    typer.echo("\n=== Hot-node false-positive context (keystone discipline) ===")
    typer.echo(fp_context.to_string(index=False))

    sig_hot = hot[(hot["p_value"] < 0.05) & hot["enriched"]]
    fdr_hot = hot[(hot["q_value"] < 0.05) & hot["enriched"]]
    if len(fdr_hot):
        typer.echo(f"\n=== FDR-significant enriched units (q < 0.05, n={len(fdr_hot)}) ===")
        typer.echo(fdr_hot.head(15).to_string(index=False))
    elif len(sig_hot):
        raw_fp = fp_context[(fp_context["label"] == "raw") & (fp_context["unit"] == "all")]
        if len(raw_fp):
            r = raw_fp.iloc[0]
            typer.echo(
                f"\n(raw: {int(r['n_sig_p05_enriched'])}/{int(r['n_tested'])} uncorrected p<0.05 "
                f"enriched vs E[FP]≈{r['e_fp_p05']:.0f}; 0 survive BH-FDR)"
            )
    else:
        typer.echo("\n(no clades/genera enriched at p < 0.05 under label permutation)")

    typer.echo(f"\nwrote P7 results to {out_dir.resolve()}")


def _median_phylo(m) -> float:
    import numpy as np

    pos = m.D_phylo[m.D_phylo > 0]
    return float(np.median(pos)) if pos.size else 0.0


if __name__ == "__main__":  # pragma: no cover
    app()
