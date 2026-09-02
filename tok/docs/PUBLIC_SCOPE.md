# TOK Public Release Scope

This repository is a curated public research artifact package for TOK. It is
designed to make selected freeze packages, summaries, and safety boundaries
inspectable without exposing private benchmark material or claiming that the
full internal research engine is open sourced here.

## What This Package Includes

- Portfolio-facing overview and architecture documentation.
- Public summaries for the Paper 1 / Paper 3 benchmark freeze.
- Public summaries for the Paper 2 dynamics atlas freeze.
- Public Demo Freeze v0 replay artifacts from a research cockpit session.
- A public validator for release boundaries and freeze-summary invariants.
- A minimal toy reference pipeline that demonstrates the artifact sequence and
  safety gates on a synthetic example.

## What This Package Does Not Include

- Private benchmark answer keys.
- Private scoring maps or private scorecards.
- Private ground-truth folders.
- The full internal TOK research workspace.
- Production Clarity Engine update paths.
- Any autonomous evidence update, memory update, or intervention mechanism.

## Claim Boundary

This package supports narrow claims about:

- how TOK represents reasoning as auditable artifacts;
- how selected public freeze summaries preserve research-only boundaries;
- how benchmark hygiene and dynamics-atlas results are recorded in public-safe
  form;
- how the Demo Freeze v0 replay preserves observation/evidence separation;
- how a minimal reference pipeline can keep candidates from becoming evidence
  without human review.

It does not support claims that TOK:

- automatically discovers real-world causal truth;
- solves causality;
- proves that a candidate mechanism is true;
- autonomously updates evidence, production memory, or `wisdom_db`;
- exposes every internal implementation component used in the research
  workspace.

## Reader Guidance

Treat this repository as a public release surface for a larger research system:
freeze summaries, replay artifacts, release-boundary checks, and a toy
reference implementation. The strongest way to read it is not as a product demo
or a complete open-source engine, but as evidence of disciplined research
packaging around mechanism-centered reasoning.
