# Demo Freeze v0

`demo_freeze_v0` builds the canonical TOK research cockpit replay.

It creates a session-shaped artifact tree under:

```text
research/demo_freeze_v0/generated/demo_session/
```

and records hashes, validation status, and claim boundaries in:

```text
research/demo_freeze_v0/generated/demo_freeze_manifest_v0.json
```

Run from the repository root:

```powershell
.venv\Scripts\python.exe -B -m research.demo_freeze_v0.run_demo_freeze
.venv\Scripts\python.exe -B -m research.demo_freeze_v0.test_demo_freeze_v0
```

The replay is research-only. It does not call networked LLMs, expose private
benchmark answer keys, update evidence, update learning, or write to production
memory.

