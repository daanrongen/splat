# splat documentation

This directory holds contributor and reference docs. The root
[README](../README.md) is the canonical user-facing guide.

## Reading order

| Document | Purpose |
|---|---|
| [README](../README.md) | Product model, command taxonomy, common workflows, install, environment, and current backend table. |
| [architecture.md](architecture.md) | Contributor architecture: layers, import boundaries, backend loading, manifests, and transport parity. |
| [pipeline.md](pipeline.md) | Walkthrough: one diffused image through every stage, with the commands, numbers and pictures. |
| [gaps.md](gaps.md) | Active defects and design gaps, mapped to the current GitHub issue backlog. |
| [roadmap.md](roadmap.md) | Future converter stages and candidate backends for anything-to-anything visual synthesis. |

## Documentation principles

- The README describes what is true now, not what might exist later.
- Backend/model tables should come from catalog metadata whenever possible.
- Command option docs should come from Typer/env metadata whenever possible.
- Research notes belong in `roadmap.md`, not in the first-run user path.
- Reproduction-heavy audit notes belong in `pipeline.md` or `gaps.md`.

## Core language

Use these terms consistently:

- **Manifest**: cached, typed pipeline asset with provenance.
- **Manifest DAG**: parent/child graph connecting generated assets.
- **GaussianCloud**: canonical in-memory representation for splat formats and
  splat transforms.
- **Model-backed stage**: command that selects a backend model/runtime.
- **Deterministic tool**: fixed transform with no model catalog.
- **Inspection/admin command**: reads facts or manages local state without
  loading ML runtimes.
- **Backend descriptor**: pure metadata catalog entry.
- **Backend adapter**: concrete runtime implementation imported only when used.
