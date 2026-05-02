# Architecture documentation

## C4 (Structurizr)

- `workspace.dsl` — source model (C1–C3).
- `workspace.json` — exported for tooling; do not edit manually.
- `.structurizr/` — local runtime; gitignored.

## Code diagrams (`code/*.puml`)

Hand-curated PlantUML; raw `pyreverse` output is generated on demand and not versioned. Workflow: `code/README.md`.

Runbook / traceability: [`../RUNBOOK.md`](../RUNBOOK.md), [`../TRACEABILITY.md`](../TRACEABILITY.md).
