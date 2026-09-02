"""Experiments and sensitivity analysis.

Experiment 001 compares two levers for suppressing harmful intervention as
interpretive capacity declines:

    * Semantic Clarity (SC)            -> environment.marker_clarity
    * Phenomenological Caution (PC)    -> environment.phenomenological_caution

The headline test (``h1_test``) estimates, at each interpretive-capacity level,
the marginal slope of mean intervention with respect to each lever (the other
held at 0.5, common random numbers). H1 predicts that *below* some capacity
threshold PC reduces intervention more steeply than SC.

The module also provides a from-scratch Sobol sensitivity analysis (Saltelli
sampling + Jansen/Saltelli estimators), one-at-a-time ranges, and local partial
derivatives of Expected Harm over all 17 input variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import qmc

from dtbr_mc.agents import AGENT_VARIABLES
from dtbr_mc.behavior import get_model
from dtbr_mc.config.schemas import (
    AGENT_VARIABLE_NAMES,
    ENVIRONMENT_VARIABLE_NAMES,
    BehaviorWeights,
    ExperimentConfig,
    OutcomeThresholds,
    SimulationConfig,
)
from dtbr_mc.environment import ENVIRONMENT_VARIABLES
from dtbr_mc.metrics import compute_metrics
from dtbr_mc.simulation import Simulator, map_outcomes

ALL_VARIABLES: tuple[str, ...] = tuple(AGENT_VARIABLE_NAMES) + tuple(ENVIRONMENT_VARIABLE_NAMES)

# Map the two policy levers to underlying variables.
LEVER_SC = "marker_clarity"
LEVER_PC = "phenomenological_caution"
AXIS_IC = "interpretive_capacity"


# --------------------------------------------------------------------------- #
# Low-level evaluation helpers (operate on shared numpy-backed frames)
# --------------------------------------------------------------------------- #


def _expected_harm(agents: pd.DataFrame, env: pd.DataFrame, model_name: str, weights: BehaviorWeights):
    model = get_model(model_name)
    b = model(agents, env, weights)
    eh = b.p_encounter * b.intervention * env["repository_severity"].to_numpy()
    return eh, b


def _eval_cell(
    agents: pd.DataFrame,
    env: pd.DataFrame,
    model_name: str,
    weights: BehaviorWeights,
    *,
    mc: float | None = None,
    pc: float | None = None,
    ic: float | None = None,
) -> dict[str, float]:
    """Evaluate population means with the controlled variables overridden.

    Mutates the three controlled columns in place (everything else is the shared,
    fixed population -- common random numbers).
    """
    if mc is not None:
        env[LEVER_SC] = mc
    if pc is not None:
        env[LEVER_PC] = pc
    if ic is not None:
        agents[AXIS_IC] = ic
    eh, b = _expected_harm(agents, env, model_name, weights)
    outcomes = map_outcomes(b.intervention, OutcomeThresholds())
    return {
        "mean_intervention": float(b.intervention.mean()),
        "expected_harm": float(eh.mean()),
        "disturbance_rate": float(np.isin(outcomes, ["INVESTIGATE", "EXCAVATE"]).mean()),
        "excavation_rate": float((outcomes == "EXCAVATE").mean()),
        "avoidance_rate": float((outcomes == "AVOID").mean()),
        "mean_caution": float(b.caution.mean()),
        "mean_curiosity": float(b.curiosity.mean()),
    }


# --------------------------------------------------------------------------- #
# Experiment 001
# --------------------------------------------------------------------------- #


@dataclass
class Experiment001Result:
    config: ExperimentConfig
    baseline_metrics: pd.DataFrame
    heatmaps: pd.DataFrame              # tidy: ic_level, marker_clarity, phen_caution, metrics
    sweeps_1d: pd.DataFrame             # tidy: variable, value, metrics
    outcome_distribution: pd.DataFrame  # tidy: variable, value, outcome, fraction
    interactions: pd.DataFrame          # tidy: lever, ic_level, value, expected_harm, mean_intervention
    h1: "H1Result"
    extra: dict = field(default_factory=dict)


@dataclass
class H1Result:
    table: pd.DataFrame            # per ic: slope_pc_*, slope_sc_*, margin_*
    threshold_intervention: float | None
    threshold_expected_harm: float | None
    verdict: str
    supported: bool
    detail: str


def _h1_test(
    agents: pd.DataFrame,
    env: pd.DataFrame,
    model_name: str,
    weights: BehaviorWeights,
    ic_grid: np.ndarray,
    lever_points: int = 11,
) -> H1Result:
    """Compare marginal lever slopes across interpretive capacity.

    For each IC level, fit the slope of (mean intervention, E[H]) vs PC (SC=0.5)
    and vs SC (PC=0.5). ``margin = slope_SC - slope_PC``; ``margin > 0`` means PC
    reduces the target more steeply than SC at that capacity level.
    """
    lvals = np.linspace(0.0, 1.0, lever_points)
    rows = []
    for ic in ic_grid:
        # PC sweep (SC fixed at 0.5)
        pc_int, pc_eh = [], []
        for v in lvals:
            r = _eval_cell(agents, env, model_name, weights, mc=0.5, pc=v, ic=ic)
            pc_int.append(r["mean_intervention"])
            pc_eh.append(r["expected_harm"])
        # SC sweep (PC fixed at 0.5)
        sc_int, sc_eh = [], []
        for v in lvals:
            r = _eval_cell(agents, env, model_name, weights, mc=v, pc=0.5, ic=ic)
            sc_int.append(r["mean_intervention"])
            sc_eh.append(r["expected_harm"])
        slope_pc_int = float(np.polyfit(lvals, pc_int, 1)[0])
        slope_sc_int = float(np.polyfit(lvals, sc_int, 1)[0])
        slope_pc_eh = float(np.polyfit(lvals, pc_eh, 1)[0])
        slope_sc_eh = float(np.polyfit(lvals, sc_eh, 1)[0])
        rows.append(
            {
                "interpretive_capacity": float(ic),
                "slope_pc_intervention": slope_pc_int,
                "slope_sc_intervention": slope_sc_int,
                "margin_intervention": slope_sc_int - slope_pc_int,
                "slope_pc_expected_harm": slope_pc_eh,
                "slope_sc_expected_harm": slope_sc_eh,
                "margin_expected_harm": slope_sc_eh - slope_pc_eh,
            }
        )
    table = pd.DataFrame(rows)

    def _threshold(col: str) -> float | None:
        # Largest IC at which margin>0 immediately precedes margin<=0 (a downward crossing).
        m = table[col].to_numpy()
        ic = table["interpretive_capacity"].to_numpy()
        cross = None
        for i in range(len(m) - 1):
            if m[i] > 0 >= m[i + 1]:
                # linear interpolation of the zero crossing
                x0, x1, y0, y1 = ic[i], ic[i + 1], m[i], m[i + 1]
                cross = float(x0 + (x1 - x0) * (y0 / (y0 - y1)))
        return cross

    thr_int = _threshold("margin_intervention")
    thr_eh = _threshold("margin_expected_harm")

    margin = table["margin_intervention"].to_numpy()
    ic_arr = table["interpretive_capacity"].to_numpy()
    low_mask = ic_arr <= 0.3
    high_mask = ic_arr >= 0.7

    # Does PC out-reduce SC at low capacity? (H1's core directional claim.)
    low_ic_pc_dominates = bool((margin[low_mask] > 0).any())
    # Does SC overtake PC at high capacity? (H1's threshold/ordering claim.)
    high_ic_sc_dominates = bool((margin[high_mask] <= 0).all()) if high_mask.any() else False
    # Does PC dominate at *every* capacity? (direction holds, threshold absent.)
    pc_dominates_everywhere = bool((margin > 0).all())
    # Sign of PC's own marginal effect at low IC: <0 = PC brakes, >0 = PC backfires.
    slope_pc_low = float(table.loc[low_mask, "slope_pc_intervention"].mean()) if low_mask.any() else float("nan")
    pc_backfires_low = slope_pc_low > 0

    # A clean crossover = PC dominates low, SC dominates high, with an interior threshold.
    clean_crossover = low_ic_pc_dominates and high_ic_sc_dominates and (thr_int is not None)
    supported = clean_crossover

    if clean_crossover:
        verdict = "H1 SUPPORTED"
        detail = (
            f"Clean crossover: PC reduces intervention more than SC below "
            f"IC*={thr_int:.3f} and SC overtakes PC above it. The threshold "
            f"structure predicted by H1 is present."
        )
    elif pc_dominates_everywhere and not pc_backfires_low:
        # The key baseline finding: direction right, threshold claim wrong.
        verdict = "H1 DIRECTION-ONLY (threshold claim UNSUPPORTED)"
        detail = (
            "PC reduces intervention more steeply than SC at EVERY interpretive-"
            "capacity level, with no crossover. H1's qualitative direction (PC is "
            "the stronger brake) holds, but its central claim — that this advantage "
            "is specific to LOW capacity, below a threshold — is unsupported: there "
            "is no threshold and the advantage does not vanish (or even grows) as "
            "capacity rises."
        )
    elif not low_ic_pc_dominates or pc_backfires_low:
        verdict = "H1 FALSIFIED"
        why = (
            "increasing phenomenological caution INCREASES intervention at low "
            "capacity (prestige/curiosity backfire)"
            if pc_backfires_low
            else "semantic clarity reduces intervention at least as much as PC at low capacity"
        )
        detail = (
            f"At low interpretive capacity, {why}. H1's directional prediction does "
            f"not hold; the null or the curiosity-backfire alternative is the better "
            f"description here."
        )
    else:
        verdict = "H1 PARTIAL / AMBIGUOUS"
        detail = (
            "A low-capacity PC advantage exists but the ordering across capacity is "
            "non-monotone or noisy, so neither the clean threshold pattern nor a "
            "clean falsification is established within this grid."
        )

    return H1Result(table, thr_int, thr_eh, verdict, supported, detail)


def run_experiment_001(cfg: ExperimentConfig, heatmap_n: int | None = None) -> Experiment001Result:
    """Run Experiment 001 end to end (in memory; CSV writing handled by caller)."""
    sim_cfg = SimulationConfig(n_agents=cfg.n_agents, seed=cfg.seed, model=cfg.model)
    if cfg.weights is not None:
        sim_cfg = sim_cfg.model_copy(update={"weights": cfg.weights})
    sim = Simulator(sim_cfg)

    # One reference population for baseline metrics (full random environment).
    ref = sim.run()
    baseline_metrics = compute_metrics(ref, bootstrap_n=cfg.bootstrap_n, ci=cfg.bootstrap_ci, seed=cfg.seed + 1)

    # Shared population for all controlled sweeps (common random numbers).
    n_grid = heatmap_n if heatmap_n is not None else cfg.n_agents
    agents, env = sim.sample(n=n_grid, seed=cfg.seed)
    agents = agents.copy()
    env = env.copy()
    w = sim_cfg.weights

    # -- 2D heatmaps: marker_clarity x phenomenological_caution at IC levels -- #
    res = next(s for s in cfg.sweeps if s.variable == LEVER_SC)
    mc_vals = np.linspace(res.start, res.stop, res.num)
    res = next(s for s in cfg.sweeps if s.variable == LEVER_PC)
    pc_vals = np.linspace(res.start, res.stop, res.num)

    heat_rows = []
    for ic in cfg.ic_levels:
        for mc in mc_vals:
            for pc in pc_vals:
                r = _eval_cell(agents, env, cfg.model, w, mc=mc, pc=pc, ic=ic)
                heat_rows.append({"ic_level": ic, "marker_clarity": mc, "phen_caution": pc, **r})
    heatmaps = pd.DataFrame(heat_rows)

    # -- 1D sweeps for each controlled variable (others at 0.5) -------------- #
    sweep_rows = []
    dist_rows = []
    for spec in cfg.sweeps:
        vals = np.linspace(spec.start, spec.stop, spec.num)
        for v in vals:
            kw = {"mc": 0.5, "pc": 0.5, "ic": 0.5}
            if spec.variable == LEVER_SC:
                kw["mc"] = v
            elif spec.variable == LEVER_PC:
                kw["pc"] = v
            elif spec.variable == AXIS_IC:
                kw["ic"] = v
            r = _eval_cell(agents, env, cfg.model, w, **kw)
            sweep_rows.append({"variable": spec.variable, "value": float(v), **r})
            # outcome distribution at this point
            if spec.variable == LEVER_SC:
                env[LEVER_SC] = v; env[LEVER_PC] = 0.5; agents[AXIS_IC] = 0.5
            elif spec.variable == LEVER_PC:
                env[LEVER_SC] = 0.5; env[LEVER_PC] = v; agents[AXIS_IC] = 0.5
            else:
                env[LEVER_SC] = 0.5; env[LEVER_PC] = 0.5; agents[AXIS_IC] = v
            _, b = _expected_harm(agents, env, cfg.model, w)
            outs = map_outcomes(b.intervention, OutcomeThresholds())
            counts = pd.Series(outs).value_counts(normalize=True)
            for label in OutcomeThresholds().labels:
                dist_rows.append(
                    {"variable": spec.variable, "value": float(v), "outcome": label,
                     "fraction": float(counts.get(label, 0.0))}
                )
    sweeps_1d = pd.DataFrame(sweep_rows)
    outcome_distribution = pd.DataFrame(dist_rows)

    # -- Interaction data: E[H] & intervention vs each lever at IC levels ---- #
    inter_rows = []
    lvals = np.linspace(0, 1, 21)
    for ic in cfg.ic_levels:
        for v in lvals:
            r_pc = _eval_cell(agents, env, cfg.model, w, mc=0.5, pc=v, ic=ic)
            inter_rows.append({"lever": "PC", "ic_level": ic, "value": float(v),
                               "expected_harm": r_pc["expected_harm"],
                               "mean_intervention": r_pc["mean_intervention"]})
            r_sc = _eval_cell(agents, env, cfg.model, w, mc=v, pc=0.5, ic=ic)
            inter_rows.append({"lever": "SC", "ic_level": ic, "value": float(v),
                               "expected_harm": r_sc["expected_harm"],
                               "mean_intervention": r_sc["mean_intervention"]})
    interactions = pd.DataFrame(inter_rows)

    # -- H1 test ------------------------------------------------------------- #
    ic_grid = np.linspace(0.0, 1.0, 21)
    h1 = _h1_test(agents, env, cfg.model, w, ic_grid)

    return Experiment001Result(
        config=cfg,
        baseline_metrics=baseline_metrics,
        heatmaps=heatmaps,
        sweeps_1d=sweeps_1d,
        outcome_distribution=outcome_distribution,
        interactions=interactions,
        h1=h1,
    )


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #


def _frames_from_matrix(M: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_agent = len(AGENT_VARIABLE_NAMES)
    a = pd.DataFrame(M[:, :n_agent], columns=list(AGENT_VARIABLE_NAMES))
    e = pd.DataFrame(M[:, n_agent:], columns=list(ENVIRONMENT_VARIABLE_NAMES))
    return a, e


def sobol_indices(
    model_name: str = "baseline",
    weights: BehaviorWeights | None = None,
    n_base: int = 8192,
    seed: int = 0,
) -> pd.DataFrame:
    """First-order (S1) and total-order (ST) Sobol indices of Expected Harm.

    Saltelli sampling with the Saltelli(2010)/Jansen estimators. Inputs are the
    17 variables on [0,1] under uniform priors (the natural Sobol domain).
    """
    weights = weights or BehaviorWeights()
    d = len(ALL_VARIABLES)
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    pts = sampler.random(n_base)
    A = pts[:, :d]
    B = pts[:, d:]

    def f(M: np.ndarray) -> np.ndarray:
        a, e = _frames_from_matrix(M)
        eh, _ = _expected_harm(a, e, model_name, weights)
        return eh

    fA, fB = f(A), f(B)
    var = np.var(np.concatenate([fA, fB]))
    rows = []
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB = f(AB)
        # Saltelli 2010 first-order; Jansen total-order
        s1 = np.mean(fB * (fAB - fA)) / var if var > 0 else np.nan
        st = 0.5 * np.mean((fA - fAB) ** 2) / var if var > 0 else np.nan
        rows.append({"variable": ALL_VARIABLES[i], "S1": float(s1), "ST": float(st)})
    return pd.DataFrame(rows).set_index("variable")


def oat_ranges(
    model_name: str = "baseline",
    weights: BehaviorWeights | None = None,
    points: int = 21,
    nominal: float = 0.5,
) -> pd.DataFrame:
    """One-at-a-time: sweep each variable over [0,1] with others at ``nominal``."""
    weights = weights or BehaviorWeights()
    d = len(ALL_VARIABLES)
    vals = np.linspace(0, 1, points)
    rows = []
    for i, name in enumerate(ALL_VARIABLES):
        M = np.full((points, d), nominal)
        M[:, i] = vals
        a, e = _frames_from_matrix(M)
        eh, _ = _expected_harm(a, e, model_name, weights)
        rows.append({"variable": name, "eh_min": float(eh.min()), "eh_max": float(eh.max()),
                     "oat_range": float(eh.max() - eh.min())})
    return pd.DataFrame(rows).set_index("variable")


def partial_derivatives(
    model_name: str = "baseline",
    weights: BehaviorWeights | None = None,
    nominal: float = 0.5,
    h: float = 1e-3,
) -> pd.DataFrame:
    """Central-difference partial derivatives of E[H] at the nominal point."""
    weights = weights or BehaviorWeights()
    d = len(ALL_VARIABLES)
    rows = []
    for i, name in enumerate(ALL_VARIABLES):
        Mp = np.full((1, d), nominal); Mp[0, i] = min(1.0, nominal + h)
        Mm = np.full((1, d), nominal); Mm[0, i] = max(0.0, nominal - h)
        ap, ep = _frames_from_matrix(Mp)
        am, em = _frames_from_matrix(Mm)
        fp, _ = _expected_harm(ap, ep, model_name, weights)
        fm, _ = _expected_harm(am, em, model_name, weights)
        deriv = float((fp[0] - fm[0]) / (Mp[0, i] - Mm[0, i]))
        rows.append({"variable": name, "dEH": deriv, "abs_dEH": abs(deriv)})
    return pd.DataFrame(rows).set_index("variable")


def sensitivity_analysis(
    model_name: str = "baseline",
    weights: BehaviorWeights | None = None,
    n_base: int = 8192,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Run Sobol + OAT + partials and produce an importance ranking by ST."""
    sob = sobol_indices(model_name, weights, n_base=n_base, seed=seed)
    oat = oat_ranges(model_name, weights)
    par = partial_derivatives(model_name, weights)
    ranking = sob.join(oat[["oat_range"]]).join(par[["abs_dEH"]])
    ranking = ranking.sort_values("ST", ascending=False)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return {"sobol": sob, "oat": oat, "partials": par, "ranking": ranking}


__all__ = [
    "run_experiment_001",
    "Experiment001Result",
    "H1Result",
    "sobol_indices",
    "oat_ranges",
    "partial_derivatives",
    "sensitivity_analysis",
]
