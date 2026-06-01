# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`starshot-benchmark` is an orchestrator for a **text-to-3D scene pipeline**. A user prompt like "A beautiful modern mansion" or "A swamp with islands" is recursively decomposed by LLMs into subzones and objects, each generated as a 3D mesh via **Trellis 2**, then composed into a single `.glb` file.

The broader goal is to **benchmark LLM spatial reasoning** — the dashboard lets you swap the LLM used at every reasoning step in the pipeline and compare outputs.

## Engineering principles

- **Do the minimum that achieves the result.** If three lines work, don't write a helper. If a bug fix is one line, don't refactor the surrounding code.
- **No speculative features.** Don't add framework niceties, extra endpoints, abstractions, validators, or config knobs that aren't required by the current task. YAGNI hard.
- **No scope creep.** Stick to what the pipeline spec below describes. If something seems missing, ask — don't invent.
- **No tests.** This is a prototyping benchmark tool; tests have been explicitly removed.
- **FastAPI default docs/openapi routes stay disabled.**

## Repository layout (planned)

Two top-level parts:

- **`client/`** — Node process running a Three.js sandbox + dashboard. Lets the user pick the LLM used across the pipeline, submits a prompt to the server, then loads and renders the returned `.glb` URL.
- **`server/`** — Orchestrator that runs the full pipeline described below and returns a URL to the final `.glb`.

## Core domain type: `Node`

Everything in the scene tree is a `Node`. Zones (subscenes) and objects are both Nodes; they differ only in what's populated:

- `id: str`
- `prompt: str`
- `bbox: BoundingBox` — axis-aligned, world-space.
- `relationships: list[Relationship]` — how this node is anchored to its parent / siblings / a frame.
- `mesh_url: str | None` — set for concrete nodes (objects, realized frames). `None` for abstract zones.
- `children: list[Node]` — zones populate this; atomic leaves don't.

### Canonical front view

One global convention, used to interpret every bbox corner and relationship:

- `+X` = right
- `+Y` = up
- `+Z` = toward the viewer (front)
- `-Z` = away from the viewer (back)

Right-handed, Y-up, meters. There is no per-scene "front view" — the axes above are it.

### `Relationship`

Relates a node to a target (parent id, sibling id, or a frame id):

- `target: str`
- `kind: RelationshipKind` — `ON`, `BESIDE`, `BELOW`, `ABOVE`, `ATTACHED`. Centrally defined; the vocabulary may grow.
- `reference_point: Corner` — which of the target bbox's 8 corners this relationship anchors against, under the canonical front view.

`Corner` is one of: `TOP_LEFT_FRONT`, `TOP_LEFT_BACK`, `TOP_RIGHT_FRONT`, `TOP_RIGHT_BACK`, `BOTTOM_LEFT_FRONT`, `BOTTOM_LEFT_BACK`, `BOTTOM_RIGHT_FRONT`, `BOTTOM_RIGHT_BACK`.

## Pipeline

The pipeline has two phases. Phase 1 (divider) recursively decomposes the prompt into a tree of Nodes with resolved bboxes. Phase 2 (generation) realizes meshes for atomic leaves.

## Cross-cutting concerns

- **Pluggable LLM selection.** Every LLM call site reads the model from a request-scoped value (currently passed as `model: str` through the pipeline). No hard-coded model IDs at call sites.
- **Deterministic vs. LLM steps.** Frame realization (plane/curve meshing), mesh rescaling, and `.glb` assembly are deterministic. LLM calls are: overall bbox, zone planning, children materialization, bbox resolution, frame decider. Keep determinism free of LLM calls so benchmark results isolate LLM quality.
- **Retry logging (when validators exist).** The current flow is optimistic — the LLM is trusted on bbox placement. If/when deterministic validators are added back, every retry must be logged; retry counts are a core benchmark signal. Don't preemptively add the validators.
- **Recursion termination** is an LLM decision (step 3's `is_atomic`). No hard depth cap.
- **Frames are not Trellis** (except the `generated` escape hatch). Plane and curve frames are baked from their specs.

## Client ↔ server contract

- Client `POST`s `(prompt, model_selection)` to the server.
- Server runs the divider (and eventually phase 2) and responds with a URL to the final `.glb`.
- Client fetches the `.glb` from that URL and renders it in the Three.js sandbox.

Keep this surface small; the dashboard's primary job is model selection and prompt submission.

## Commands

No build/test/lint commands are wired up yet. `enx.toml` exposes `enx up`, `enx down`, `enx start`, `enx test` hooks; populate them as stacks stabilize.
