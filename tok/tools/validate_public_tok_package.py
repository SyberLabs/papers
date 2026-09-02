"""Validate the public TOK portfolio package.

This is intentionally narrower than the full internal TOK smoke suite. It
checks that public release boundaries are intact and that the included freeze
summaries preserve the research-only claims.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def assert_no_private_ground_truth() -> None:
    forbidden = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        lowered = [part.lower() for part in path.parts]
        name = path.name.lower()
        if "private_ground_truth" in lowered:
            forbidden.append(path)
        if "answer_key" in name:
            forbidden.append(path)
    assert not forbidden, "private benchmark material present: " + ", ".join(map(str, forbidden[:8]))


def assert_paper1_public_summary() -> None:
    summary = load_json("research/paper1_benchmark_freeze_v0/generated/paper1_public_summary_v0.json")
    assert summary["production_effect"] == "none_research_only"
    assert summary["public_boundaries"]["private_ground_truth_included"] is False
    assert summary["public_boundaries"]["may_update_evidence"] is False
    assert summary["results"]["naturalistic_holdout_v1"]["problem_count"] == 240
    assert summary["results"]["boundary_note_mini_holdout_v0"]["exact_coordinate_correct"] == 69


def assert_paper2_manifest() -> None:
    manifest = load_json("research/paper2_dynamics_freeze_v0/generated/paper2_dynamics_public_summary_v0.json")
    assert manifest["production_effect"] == "none_research_only"
    assert manifest["results"]["dynamics_corpus"]["system_count"] == 12
    assert manifest["results"]["dynamics_corpus"]["regime_count"] == 7
    assert manifest["results"]["admissibility_atlas_v1"]["acceptance_all_passed"] is True
    assert "broad invalidation of LTC or LNN model classes" in manifest["claim_boundary"]["does_not_support"]


def assert_demo_manifest() -> None:
    manifest = load_json("research/demo_freeze_v0/generated/demo_public_summary_v0.json")
    assert manifest["production_effect"] == "none_research_only"
    assert manifest["validation_summary"]["failed_validation_count"] == 0
    assert manifest["key_results"]["registry_evidence_status"] == "not_evidence"
    assert manifest["key_results"]["evidence_update_applied"] is False
    assert manifest["key_results"]["may_update_wisdom_db"] is False


def load_toy_reference_module() -> ModuleType:
    module_path = ROOT / "examples/toy_reference_pipeline/run_toy_reference_pipeline.py"
    spec = importlib.util.spec_from_file_location("tok_toy_reference_pipeline", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_toy_reference_pipeline() -> None:
    module = load_toy_reference_module()
    run = module.build_toy_reference_run()
    module.validate_toy_reference_run(run)

    proposal = run["artifacts"]["evidence_transition_proposal"]
    registry = run["artifacts"]["observation_registry"]
    assert proposal["proposal_status"] == "not_applied"
    assert proposal["human_gate"] == "required_not_satisfied"
    assert proposal["may_update_evidence"] is False
    assert proposal["update_effect"]["applied"] is False
    assert registry["observations"][0]["evidence_status"] == "not_evidence"


def main() -> int:
    assert_no_private_ground_truth()
    assert_paper1_public_summary()
    assert_paper2_manifest()
    assert_demo_manifest()
    assert_toy_reference_pipeline()
    print("TOK public package validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
