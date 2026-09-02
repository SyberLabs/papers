"""Minimal public reference pipeline for TOK.

This is not the private TOK engine. It is a compact, deterministic toy example
that demonstrates the public artifact sequence and the safety boundaries that
the full research system is organized around.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path(__file__).resolve().parent / "generated" / "toy_reference_run_v0.json"


def build_toy_reference_run() -> dict[str, Any]:
    """Build a synthetic TOK reference run with explicit non-evidence gates."""

    narrative = (
        "A facilities team reports that boiler outlet temperature is drifting "
        "below target during stable demand. Recent maintenance notes mention a "
        "possible sensor recalibration, but operators also suspect capacity loss."
    )

    mechanism_graph = {
        "artifact_type": "mechanism_graph",
        "artifact_id": "toy_boiler_mechanism_graph_v0",
        "status": "candidate_mechanism_hypothesis",
        "source_boundary": "synthetic_public_toy_example",
        "nodes": [
            {"id": "demand", "kind": "context_signal"},
            {"id": "sensor_calibration", "kind": "candidate_disturbance"},
            {"id": "heat_transfer_capacity", "kind": "candidate_mechanism"},
            {"id": "outlet_temperature", "kind": "observed_signal"},
        ],
        "edges": [
            {"from": "demand", "to": "outlet_temperature", "relation": "conditions"},
            {
                "from": "sensor_calibration",
                "to": "outlet_temperature",
                "relation": "may_bias_observation",
            },
            {
                "from": "heat_transfer_capacity",
                "to": "outlet_temperature",
                "relation": "may_reduce_response",
            },
        ],
        "claim_boundary": "candidate_graph_not_causal_truth",
    }

    dynamics_binding = {
        "artifact_type": "dynamics_template_binding",
        "artifact_id": "toy_capacity_loss_binding_v0",
        "template": "first_order_capacity_loss_or_sensor_bias",
        "status": "candidate_not_evidence",
        "bound_variables": {
            "input_signal": "demand",
            "state_proxy": "outlet_temperature",
            "candidate_parameter": "effective_heat_transfer_capacity",
        },
        "may_update_evidence": False,
    }

    dynamic_divergence_trace = {
        "artifact_type": "dynamic_divergence_trace",
        "artifact_id": "toy_boiler_dyndiv_trace_v0",
        "trace_boundary": "toy_trace_not_empirical_claim",
        "windows": [
            {
                "window_id": "stable_demand_lag_window",
                "candidate_signal": "slower_than_expected_recovery",
                "interpretation": "consistent_with_capacity_loss_or_sensor_bias",
                "evidence_status": "not_evidence",
            }
        ],
    }

    observation_registry = {
        "artifact_type": "observation_registry",
        "artifact_id": "toy_observation_registry_v0",
        "registry_status": "reviewed_seen_data",
        "observations": [
            {
                "observation_id": "toy_temp_profile_review_v0",
                "source": "synthetic_public_toy_profile",
                "summary": "Outlet temperature appears low during stable demand.",
                "evidence_status": "not_evidence",
                "review_state": "human_review_required",
            }
        ],
    }

    evidence_transition_proposal = {
        "artifact_type": "evidence_transition_proposal",
        "artifact_id": "toy_evidence_transition_proposal_v0",
        "requested_transition": "seen_data_to_candidate_evidence",
        "proposal_status": "not_applied",
        "human_gate": "required_not_satisfied",
        "may_update_evidence": False,
        "may_update_wisdom_db": False,
        "production_effect": "none_research_only",
        "update_effect": {"applied": False},
        "rationale": (
            "The toy trace may motivate review, but it does not authorize an "
            "evidence update or an intervention recommendation."
        ),
    }

    causal_status_assessment = {
        "artifact_type": "causal_status_assessment",
        "artifact_id": "toy_causal_status_assessment_v0",
        "status": "candidate_mechanism_unconfirmed",
        "does_support": [
            "A source-bounded candidate mechanism can be represented.",
            "Observation and evidence status can remain separated.",
        ],
        "does_not_support": [
            "The mechanism is true.",
            "The observation is evidence.",
            "A production update or intervention is authorized.",
        ],
        "production_effect": "none_research_only",
    }

    return {
        "run_id": "toy_reference_pipeline_v0",
        "public_scope": "synthetic_public_reference_example",
        "narrative": narrative,
        "artifacts": {
            "mechanism_graph": mechanism_graph,
            "dynamics_binding": dynamics_binding,
            "dynamic_divergence_trace": dynamic_divergence_trace,
            "observation_registry": observation_registry,
            "evidence_transition_proposal": evidence_transition_proposal,
            "causal_status_assessment": causal_status_assessment,
        },
        "safety_boundaries": {
            "candidate_not_evidence": True,
            "observations_are_seen_data": True,
            "may_update_evidence": False,
            "may_update_wisdom_db": False,
            "production_effect": "none_research_only",
        },
    }


def validate_toy_reference_run(run: dict[str, Any]) -> None:
    proposal = run["artifacts"]["evidence_transition_proposal"]
    registry = run["artifacts"]["observation_registry"]
    assessment = run["artifacts"]["causal_status_assessment"]
    boundaries = run["safety_boundaries"]

    assert boundaries["candidate_not_evidence"] is True
    assert boundaries["may_update_evidence"] is False
    assert boundaries["may_update_wisdom_db"] is False
    assert boundaries["production_effect"] == "none_research_only"
    assert proposal["may_update_evidence"] is False
    assert proposal["may_update_wisdom_db"] is False
    assert proposal["update_effect"]["applied"] is False
    assert proposal["production_effect"] == "none_research_only"
    assert registry["observations"][0]["evidence_status"] == "not_evidence"
    assert assessment["status"] == "candidate_mechanism_unconfirmed"


def write_toy_reference_run(path: Path = OUTPUT_PATH) -> Path:
    run = build_toy_reference_run()
    validate_toy_reference_run(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public TOK toy reference pipeline.")
    parser.add_argument("--write", action="store_true", help="Write the generated artifact to disk.")
    args = parser.parse_args()

    run = build_toy_reference_run()
    validate_toy_reference_run(run)

    if args.write:
        output_path = write_toy_reference_run()
        print(f"Wrote {output_path}")
    else:
        print(json.dumps(run, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
