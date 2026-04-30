# Architecture Documentation

This folder keeps the source architecture documentation for the public repository.

## C4 Model

- `workspace.dsl` is the source of truth for the Structurizr C1-C3 model.
- `workspace.json` is an exported snapshot generated from the DSL for tooling compatibility. Do not edit it by hand.
- `.structurizr/` contains local Structurizr runtime state, thumbnails, logs, locks, and indexes. It is intentionally ignored and should not be versioned.

## Code Diagrams

Curated PlantUML code diagrams live in `code/*.puml`. Raw `pyreverse` output and rendered SVG files are generated on demand and are not part of the stable source tree.

See `code/README.md` for the C4 level 4 workflow.
