# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`starshot-benchmark` is an orchestrator for a **text-to-3D scene pipeline**. A user prompt like "A beautiful modern mansion" or "A swamp with islands" is recursively decomposed by LLMs into subzones and objects, each generated as a 3D mesh via **Trellis 2**, then composed into a single `.glb` file.

The broader goal is to **benchmark LLM spatial reasoning** — the dashboard lets you swap the LLM used at every reasoning step in the pipeline and compare outputs.

## Repository layout (planned)

Two top-level parts:

- **`client/`** — Node process running a Three.js sandbox + dashboard. Lets the user pick the LLM used across the pipeline, submits a prompt to the server, then loads and renders the returned `.glb` URL.
- **`server/`** — Orchestrator that runs the full pipeline described below and returns a URL to the final `.glb`.

## Pipeline

The pipeline has two phases. Phase 1 (divider) recursively decomposes the prompt into a tree of Nodes with resolved bboxes. Phase 2 (generation) realizes meshes for atomic leaves.

## Client ↔ server contract

- Client `POST`s `(prompt, model_selection)` to the server.
- Server runs the divider (and eventually phase 2) and responds with a URL to the final `.glb`.
- Client fetches the `.glb` from that URL and renders it in the Three.js sandbox.

## Commands

No build/test/lint commands are wired up yet. `enx.toml` exposes `enx up`, `enx down`, `enx start`, `enx test` hooks; populate them as stacks stabilize.
