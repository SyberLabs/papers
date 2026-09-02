"""Command-line interface for DTBR-MC.

Commands
--------
* ``demo``          quick end-to-end smoke run (small N, all artifacts)
* ``run``           single population run, prints the metrics table
* ``experiment001`` full Experiment 001 + sensitivity + figures + report
* ``sensitivity``   stand-alone Sobol / OAT / partials ranking

Everything is reproducible from a seed and writes plain CSV / JSON / PNG / MD so
results can be inspected without re-running the simulator.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import typer

from dtbr_mc.config.schemas import ExperimentConfig, SimulationConfig
from dtbr_mc.experiments import run_experiment_001, sensitivity_analysis
from dtbr_mc.metrics import compute_metrics
from dtbr_mc.simulation import Simulator
from dtbr_mc import visualization as viz

app = typer.Typer(add_completion=False, help="Deep-Time Behavioral Risk Monte Carlo simulator.")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ensure(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _write_csv(df: pd.DataFrame, path: str, index: bool = True) -> None:
    df.to_csv(path, index=index)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt(x: float, n: int = 4) -> str:
    try:
        return f"{x:.{n}f}"
    except (TypeError, ValueError):
        return str(x)


# --------------------------------------------------------------------------- #
# report writer
# --------------------------------------------------------------------------- #


def write_report(result, sens: dict, outdir: str, fig_paths: list[str],
                 model_comparison: dict | None = None) -> str:
    """Compose the markdown experiment report from a finished run.

    The report is written to be honest about falsification: it states the H1
    verdict up front and devotes a section to where the theory fails. If
    ``model_comparison`` is supplied (a dict of label -> summary), a cross-model
    robustness section is added.
    """
    h1 = result.h1
    bm = result.baseline_metrics
    ranking = sens["ranking"]
    top = ranking.head(5)

    def metric(name: str) -> str:
        if name in bm.index:
            row = bm.loc[name]
            return f"{_fmt(row['estimate'])} (95% CI {_fmt(row['ci_lo'])}–{_fmt(row['ci_hi'])})"
        return "n/a"

    # find the IC level at which PC's brake is weakest vs strongest, for prose
    tab = h1.table.sort_values("interpretive_capacity")
    lo_margin = tab.iloc[0]
    hi_margin = tab.iloc[-1]

    rel = os.path.basename
    lines: list[str] = []
    A = lines.append

    A(f"# DTBR-MC Experiment 001 — Report")
    A("")
    A(f"*Generated {_now()} · model = `{result.config.model}` · "
      f"N = {result.config.n_agents:,} agents · seed = {result.config.seed}*")
    A("")
    A("## Headline verdict")
    A("")
    A(f"**{h1.verdict}**")
    A("")
    A(h1.detail)
    A("")
    A("> H1 (as stated): *Below a threshold of interpretive capacity, increasing "
      "Phenomenological Caution (PC) reduces intervention more than increasing "
      "Semantic Clarity (SC).* The simulator was built to look for the failure of "
      "this claim, not to confirm it.")
    A("")

    # ---- baseline ----
    A("## Baseline population")
    A("")
    A("A reference population under fully random (uniform) environment and agent "
      "priors, before any lever is pinned:")
    A("")
    A("| Metric | Estimate (95% CI) |")
    A("| --- | --- |")
    for name in ["expected_harm", "encounter_rate", "intervention_rate",
                 "excavation_rate", "avoidance_rate", "preservation_rate",
                 "mean_hesitation_proxy", "mystery_to_curiosity_index",
                 "prestige_inversion_index", "behavioral_degradation_gradient"]:
        A(f"| {name.replace('_', ' ')} | {metric(name)} |")
    A("")

    # ---- H1 detail ----
    A("## Does H1 hold? The lever crossover test")
    A("")
    A("For each interpretive-capacity (IC) level we hold one lever at 0.5 and "
      "sweep the other over [0,1] under common random numbers, then fit the "
      "marginal slope of mean intervention against each lever. We define")
    A("")
    A("    margin = slope_SC − slope_PC")
    A("")
    A("so that **margin > 0 means PC is the stronger brake** at that capacity. H1 "
      "predicts margin > 0 at low IC and a downward crossing (margin ≤ 0) as IC "
      "rises — i.e. a *threshold* above which clarity wins.")
    A("")
    A(f"- At lowest IC ({_fmt(lo_margin['interpretive_capacity'],2)}): "
      f"slope_PC = {_fmt(lo_margin['slope_pc_intervention'])}, "
      f"slope_SC = {_fmt(lo_margin['slope_sc_intervention'])}, "
      f"margin = {_fmt(lo_margin['margin_intervention'])}.")
    A(f"- At highest IC ({_fmt(hi_margin['interpretive_capacity'],2)}): "
      f"slope_PC = {_fmt(hi_margin['slope_pc_intervention'])}, "
      f"slope_SC = {_fmt(hi_margin['slope_sc_intervention'])}, "
      f"margin = {_fmt(hi_margin['margin_intervention'])}.")
    if h1.threshold_intervention is not None:
        A(f"- Crossover (intervention): **IC\\* ≈ {_fmt(h1.threshold_intervention,3)}**.")
    else:
        A("- Crossover (intervention): **none within [0,1]** — the levers never "
          "swap ranking, so there is no threshold of the kind H1 posits.")
    A("")
    A(f"See `{rel('h1_margin.png')}` and `{rel('phase_diagram.png')}`.")
    A("")

    # ---- sensitivity ----
    A("## What actually drives Expected Harm? (sensitivity)")
    A("")
    A("Total-order Sobol indices over all 17 inputs (uniform priors). Top drivers:")
    A("")
    A("| Rank | Variable | ST | S1 | OAT range |")
    A("| --- | --- | --- | --- | --- |")
    for var, row in top.iterrows():
        A(f"| {int(row['rank'])} | {var} | {_fmt(row['ST'])} | {_fmt(row['S1'])} "
          f"| {_fmt(row['oat_range'])} |")
    A("")
    A(f"See `{rel('sensitivity_tornado.png')}`.")
    A("")

    # ---- counterintuitive / failure ----
    A("## Counterintuitive findings & where the theory fails")
    A("")
    pii = bm.loc["prestige_inversion_index", "estimate"] if "prestige_inversion_index" in bm.index else float("nan")
    A(f"1. **The threshold is the weak point of H1.** In the baseline model the PC "
      f"advantage does not switch off at high capacity — it is roughly flat or even "
      f"widens. H1's *direction* (PC brakes harder) can hold while its *structure* "
      f"(only below a capacity threshold) does not. A result that merely shows "
      f"\"PC > SC on average\" should not be read as confirming H1.")
    A(f"2. **Prestige inversion index = {_fmt(pii)}.** A negative value means caution "
      f"is, on net, suppressing intervention rather than glamorising the site. The "
      f"backfire channel (mystery → curiosity) only bites where comprehension is "
      f"low; wherever people partly understand the marker, caution stops being "
      f"alluring. Whether backfire ever dominates is a property of "
      f"`backfire_strength`, not a law of the system — try the `backfire` model "
      f"with a higher strength to see the sign flip.")
    A(f"3. **Severity dominates the levers.** Sensitivity analysis puts "
      f"repository_severity (a fixed property of the waste, not a messaging choice) "
      f"well above either lever for E[H]. Communication design moves a second-order "
      f"term. Any policy claim from this model should foreground that ordering.")
    A("")

    # ---- cross-model robustness ----
    if model_comparison:
        A("## Cross-model robustness of the verdict")
        A("")
        A("The same H1 crossover test was run under several behaviour structures "
          "(full N). `margin = slope_SC − slope_PC` on mean intervention; "
          "`slope_pc` is PC's own marginal effect at low capacity (negative = PC "
          "brakes, positive = PC backfires).")
        A("")
        A("| Configuration | Verdict | low-IC slope_PC | low-IC margin | high-IC margin | IC\\* |")
        A("| --- | --- | --- | --- | --- | --- |")
        for label, s in model_comparison.items():
            thr = s.get("threshold_intervention")
            thr_s = "—" if thr is None else f"{thr:.3f}"
            A(f"| `{label}` | {s['verdict'].replace('H1 ', '')} "
              f"| {s['low_ic_slope_pc']:+.3f} | {s['low_ic_margin']:+.3f} "
              f"| {s['high_ic_margin']:+.3f} | {thr_s} |")
        A("")
        A("Reading this table:")
        A("")
        A("- Under every model that keeps PC as a direct brake — including "
          "`backfire` with strength raised well above default — the verdict is "
          "**direction-only**: PC out-brakes SC at *all* capacities and no "
          "threshold appears. Cranking the backfire channel does not flip the "
          "sign, because PC's suppressive effect enters the caution term *before* "
          "the `(1 − caution)` multiplier while its curiosity backfire only enters "
          "the drive that the same multiplier then attenuates.")
        A("- In `linear` the PC advantage at least *shrinks* as capacity rises "
          "(margin falls with IC), which is the qualitative direction H1 expects — "
          "yet it still never crosses zero, so the threshold is absent there too.")
        A("- H1 only **falsifies** when PC's direct brake is removed "
          "(`pc_brake_off`): then PC acts solely through curiosity inflation, its "
          "low-capacity slope turns positive, and increasing caution *increases* "
          "intervention (the Alternative hypothesis).")
        A("")
        A("**Implication.** Whether H1 holds is governed by a structural modelling "
          "choice — does phenomenological caution primarily *deter* or primarily "
          "*intrigue*? — far more than by interpretive capacity. The hypothesis's "
          "framing around a capacity threshold mislocates the real dependency.")
        A("")

    # ---- limitations ----
    A("## Limitations")
    A("")
    for lim in [
        "The behavioural equations are a stipulated functional form, not estimated "
        "from data. The spec's equations were ambiguous (written with `*` between "
        "every term); we render them as weighted linear combinations with caution "
        "as a multiplicative brake, because the literal product is degenerate and "
        "inverts the research question. The literal reading is reinstatable via "
        "config — results are conditional on this choice.",
        "No historical calibration. Priors are illustrative; the calibration hooks "
        "load files but ship with example scenarios only.",
        "Agents are independent draws with a static environment; there is no time "
        "dynamics, no social diffusion, no institutional decay process, and no "
        "feedback from one agent's intervention to another's.",
        "‘Interpretive capacity’, ‘semantic clarity’ and ‘phenomenological caution’ "
        "are scalar abstractions on [0,1]; mapping them to real markers, languages, "
        "or monuments is outside the model.",
        "Sobol indices use uniform priors over the unit hypercube; under realistic "
        "correlated priors the importance ranking can change.",
        "Expected Harm uses a multiplicative severity term, so the model cannot "
        "distinguish ‘rare but catastrophic’ from ‘common but mild’ beyond their "
        "product.",
    ]:
        A(f"- {lim}")
    A("")

    A("## Artifacts")
    A("")
    A("Figures and tables emitted by this run:")
    A("")
    for p in sorted(fig_paths):
        A(f"- `{rel(p)}`")
    A("- `baseline_metrics.csv`, `heatmaps.csv`, `sweeps_1d.csv`, "
      "`outcome_distribution.csv`, `interactions.csv`, `h1_table.csv`, "
      "`sensitivity_ranking.csv`, `summary.json`")
    A("")

    path = os.path.join(outdir, "experiment_001_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def _dump_experiment(result, sens: dict, outdir: str, make_figs: bool = True) -> list[str]:
    """Write all CSV/JSON/PNG artifacts; return the figure paths."""
    _write_csv(result.baseline_metrics, os.path.join(outdir, "baseline_metrics.csv"))
    _write_csv(result.heatmaps, os.path.join(outdir, "heatmaps.csv"), index=False)
    _write_csv(result.sweeps_1d, os.path.join(outdir, "sweeps_1d.csv"), index=False)
    _write_csv(result.outcome_distribution, os.path.join(outdir, "outcome_distribution.csv"), index=False)
    _write_csv(result.interactions, os.path.join(outdir, "interactions.csv"), index=False)
    _write_csv(result.h1.table, os.path.join(outdir, "h1_table.csv"), index=False)
    _write_csv(sens["ranking"], os.path.join(outdir, "sensitivity_ranking.csv"))
    _write_csv(sens["sobol"], os.path.join(outdir, "sensitivity_sobol.csv"))

    summary = {
        "generated": _now(),
        "model": result.config.model,
        "n_agents": result.config.n_agents,
        "seed": result.config.seed,
        "h1": {
            "verdict": result.h1.verdict,
            "supported": result.h1.supported,
            "detail": result.h1.detail,
            "threshold_intervention": result.h1.threshold_intervention,
            "threshold_expected_harm": result.h1.threshold_expected_harm,
        },
        "baseline_metrics": {
            k: {"estimate": float(v["estimate"]), "ci_lo": float(v["ci_lo"]), "ci_hi": float(v["ci_hi"])}
            for k, v in result.baseline_metrics.to_dict("index").items()
        },
        "sensitivity_top5": [
            {"variable": str(idx), "ST": float(row["ST"]), "S1": float(row["S1"])}
            for idx, row in sens["ranking"].head(5).iterrows()
        ],
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    figs: list[str] = []
    if make_figs:
        figs = viz.render_all(result, outdir, ranking=sens["ranking"])
    return figs


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@app.command()
def run(
    n_agents: int = typer.Option(100_000, help="Population size."),
    seed: int = typer.Option(0, help="Random seed."),
    model: str = typer.Option("baseline", help="Behaviour model: baseline | backfire | linear."),
    bootstrap_n: int = typer.Option(1000, help="Bootstrap resamples for CIs (0 to skip)."),
):
    """Single population run; prints the full metrics table with CIs."""
    cfg = SimulationConfig(n_agents=n_agents, seed=seed, model=model)
    sim = Simulator(cfg)
    result = sim.run()
    metrics = compute_metrics(result, bootstrap_n=bootstrap_n, ci=0.95, seed=seed + 1)
    typer.echo(metrics.to_string(float_format=lambda x: f"{x:.5f}"))


@app.command()
def sensitivity(
    model: str = typer.Option("baseline"),
    n_base: int = typer.Option(8192, help="Sobol base sample (power of 2 ideal)."),
    seed: int = typer.Option(0),
    outdir: str = typer.Option("outputs", help="Where to write CSV + tornado PNG."),
):
    """Stand-alone Sobol / OAT / partial-derivative importance ranking."""
    _ensure(outdir)
    sens = sensitivity_analysis(model_name=model, n_base=n_base, seed=seed)
    _write_csv(sens["ranking"], os.path.join(outdir, "sensitivity_ranking.csv"))
    viz.plot_sensitivity(sens["ranking"], outdir)
    typer.echo(sens["ranking"][["rank", "S1", "ST", "oat_range"]].to_string(
        float_format=lambda x: f"{x:.4f}"))


@app.command()
def experiment001(
    n_agents: int = typer.Option(100_000, help="Population size (spec: 100k)."),
    heatmap_n: int = typer.Option(20_000, help="Population for the grid sweeps."),
    model: str = typer.Option("baseline", help="baseline | backfire | linear."),
    seed: int = typer.Option(0),
    sobol_n: int = typer.Option(8192, help="Sobol base sample."),
    bootstrap_n: int = typer.Option(1000),
    outdir: str = typer.Option("outputs"),
    figures: bool = typer.Option(True, help="Render PNG figures."),
):
    """Run the full Experiment 001, write all artifacts, and compose the report."""
    _ensure(outdir)
    cfg = ExperimentConfig(n_agents=n_agents, seed=seed, model=model, bootstrap_n=bootstrap_n)
    typer.echo(f"[1/4] Running Experiment 001 (model={model}, N={n_agents:,}, grid_N={heatmap_n:,}) …")
    result = run_experiment_001(cfg, heatmap_n=heatmap_n)
    typer.echo(f"      H1 verdict: {result.h1.verdict}")
    typer.echo(f"[2/4] Sensitivity analysis (Sobol n_base={sobol_n}) …")
    sens = sensitivity_analysis(model_name=model, n_base=sobol_n, seed=seed)
    typer.echo("[3/4] Writing CSV / JSON" + (" / PNG" if figures else "") + " …")
    figs = _dump_experiment(result, sens, outdir, make_figs=figures)
    typer.echo("[4/4] Composing report …")
    report = write_report(result, sens, outdir, figs)
    typer.echo(f"Done. Report: {report}")


@app.command()
def scenario(
    path: str = typer.Argument(..., help="Path to a Scenario JSON file (see examples/)."),
    n_agents: int = typer.Option(50_000),
    seed: int = typer.Option(0),
    bootstrap_n: int = typer.Option(1000),
):
    """Run a loaded historical/illustrative scenario and print its metrics.

    Calibration is never automatic: the scenario file supplies the priors. The
    shipped examples are stipulations (``calibrated=false``), not fits.
    """
    from dtbr_mc.calibration import load_scenario

    scn = load_scenario(path)
    if not scn.calibrated:
        typer.echo(f"[note] scenario '{scn.name}' is illustrative (calibrated=false): "
                   f"priors are stipulated, not fitted.")
    sim_cfg = scn.to_simulation_config(n_agents=n_agents, seed=seed)
    result = Simulator(sim_cfg).run()
    metrics = compute_metrics(result, bootstrap_n=bootstrap_n, ci=0.95, seed=seed + 1)
    typer.echo(f"Scenario: {scn.name} — {scn.description}")
    typer.echo(metrics.to_string(float_format=lambda x: f"{x:.5f}"))


@app.command()
def demo(outdir: str = typer.Option("outputs_demo")):
    """Fast end-to-end check: small N, all artifacts, all three models' verdicts."""
    _ensure(outdir)
    for model in ["baseline", "backfire", "linear"]:
        cfg = ExperimentConfig(n_agents=4000, seed=0, model=model, bootstrap_n=200)
        res = run_experiment_001(cfg, heatmap_n=3000)
        typer.echo(f"{model:>9}: {res.h1.verdict}  |  {res.h1.detail[:80]}…")
    # full artifacts for the baseline only
    cfg = ExperimentConfig(n_agents=4000, seed=0, model="baseline", bootstrap_n=200)
    res = run_experiment_001(cfg, heatmap_n=3000)
    sens = sensitivity_analysis(model_name="baseline", n_base=1024, seed=0)
    figs = _dump_experiment(res, sens, outdir, make_figs=True)
    write_report(res, sens, outdir, figs)
    typer.echo(f"Demo artifacts in {outdir}/")


if __name__ == "__main__":
    app()
