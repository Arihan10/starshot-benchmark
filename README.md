# starshot-benchmark

An LLM orchestration pipeline that turns a single text prompt (e.g. *"A modern
house"*, *"A super mario bros type platformer level"*) into a fully parametric,
**object-by-object 3D scene** — recursively decomposed by LLMs into zones and
objects, each realized as a `.glb` mesh and composed into one navigable scene.

Every reasoning step in the pipeline is a **single LLM call**. Because each step
is isolated and cached, the project doubles as a **benchmark for LLM spatial
reasoning**: the dashboard lets you run the same scene with different models and
compare their decompositions, placements, and final scenes side by side.

---

## Table of contents

- [Architecture at a glance](#architecture-at-a-glance)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running it](#running-it)
- [Environment variables](#environment-variables)
- [The pipeline, end to end](#the-pipeline-end-to-end)
- [Runs, artifacts & resumability](#runs-artifacts--resumability)
- [Scripts reference](#scripts-reference)
- [Warnings & gotchas](#warnings--gotchas)

---

## Architecture at a glance

Two top-level parts talk over HTTP:

- **`server/`** — a FastAPI orchestrator (Python ≥ 3.12, managed with [`uv`](https://docs.astral.sh/uv/)).
  Runs the full decomposition + generation pipeline, streams progress over SSE,
  serves the resulting meshes, and stores every run on disk.
- **`client/`** — a tiny Node static server (`server.mjs`) that hosts a Three.js
  dashboard. It renders the scene live as meshes land and drives the benchmark
  grid (which model runs which scene).

The unit of work is a **cell = (slot, model)**:

- A **slot** is one of a fixed set of benchmark prompts (see `server/app/core/slots.py`,
  e.g. `modern-house`, `platformer-level`, `battle-arena`).
- A **model** is one of the aliased LLMs (e.g. `gpt`, `opus`, `gemini-pro`,
  `kimi`, `deepseek`; default `gemini-flash-lite`).

Each cell is an **independent, resumable run**. You drive cells from the
dashboard, watch them stream, and compare cells across models.

```
prompt ──► [ Phase 1: divider ] ──► tree of zones with resolved bounding boxes
                                       │
                                       ▼
             [ Phase 2: generation ] ──► objects per zone ──► meshes ──► composed .glb scene
```

---

## Prerequisites

| Tool | Why | Notes |
|------|-----|-------|
| **[`uv`](https://docs.astral.sh/uv/)** | Python env + dependency manager for `server/` | Provides the `uv run` used everywhere |
| **Python ≥ 3.12** | Server runtime | `uv` will fetch it if missing |
| **Node.js (18+)** | Client viewer + the mesh optimizer | `node` must be on `PATH` |
| **An `OPENROUTER_API_KEY`** | Drives every LLM reasoning step | Required for any real run |
| *(optional)* **`enx`** | Convenience task runner wired up in `enx.toml` | Everything also works with raw `uv`/`node` |

---

## Setup

### 1. Configure secrets

Copy the example env file and fill in your keys:

```bash
cp server/.env.example server/.env
```

At minimum set `OPENROUTER_API_KEY`. See [Environment variables](#environment-variables)
for the full list and what each backend needs. The server auto-loads
`server/.env` on boot (via `python-dotenv`).

### 2. Install the server (Python)

```bash
cd server
uv sync
```

> Or, from the repo root: `enx up` (runs `cd server && uv sync`).

### 3. Install the client (Node + Three.js)

```bash
cd client
npm install
```

The viewer refuses to boot without `client/node_modules/three`.

### 4. *(Only for from-scratch mesh generation)* install the optimizer

The "Generate" path runs every raw mesh through a Node optimizer. Install its deps once:

```bash
cd server/tools/optimize-assets
npm install
```

### 5. *(Only for asset-library mode)* obtain the asset library

The **default** pipeline does **not** generate meshes on the fly — it matches
each object to a pre-built catalog and bakes a placement transform into the
existing `.glb`. The catalog index (`server/app/assets_library/library.json`) is
in the repo, but the actual asset files are **not** (they are gitignored):

```
server/app/assets_library/assets-optimized/   # served library GLBs — NOT in git
server/app/assets_library/assets/             # raw library GLBs      — NOT in git
```

Without these on disk, library matches resolve but the meshes are missing and
objects won't render. Either supply the library out-of-band, or run in
from-scratch mode (`USE_ASSET_LIBRARY=false`) so meshes are generated with Nano
Banana + Trellis instead. See [Warnings & gotchas](#warnings--gotchas).

---

## Running it

### The main flow (full pipeline + dashboard)

```bash
uv run scripts/run_request.py
```

> Equivalent: `enx test`.

This boots the API server (`app.main:app`) and the Node viewer on
auto-picked free ports, then opens the dashboard in your browser. Both processes
stay alive until `Ctrl-C`.

Useful flags:

- `--run <name>` — write this session's artifacts under `runs/<name>/` instead of `runs/`.
- `--promote` — flip bbox-only completed slots in `--run` to resumable (so a
  later boot generates their meshes), then exit without booting.

**Using the dashboard**

1. It shows a grid of benchmark slots (fixed prompts) × models.
2. Pick a cell and start/resume it — the server runs Phase 1 then Phase 2 for
   that `(slot, model)` and streams events over SSE.
3. The Three.js panel renders each mesh as it lands; the tree panel shows the
   zone/object decomposition; the flights/cost panels track LLM calls and spend.
4. The prompt-lab lets you edit the prompt templates, pin a different model per
   step, and "simulate downstream" to A/B reasoning steps.
5. The per-cell **Generate** action builds fresh meshes from scratch (Nano
   Banana → Trellis) into an isolated `generated/<version>/` set, with per-object
   controls (regenerate, symmetrize, reorient, glassify, link/unlink prefabs).

### Alternate launchers

All live in `scripts/` and take the same `--run`/`--promote` flags where relevant.

| Command | What it boots | Use it for |
|---------|---------------|-----------|
| `uv run scripts/run_request.py` | Full pipeline, **depth-first** divider | The normal path |
| `uv run scripts/run_bfs.py` | Full pipeline, **breadth-first** divider (`app.main_bfs`) | A/B the two traversals; dashboard backdrop lightens |
| `uv run scripts/run_noframes.py` | Full pipeline, skips meshing the encapsulating **shells** (walls/moats/fences) | Faster iteration on object content |
| `uv run scripts/run_bboxes_only.py` | Decompose only — **no meshes**, every node is a wireframe bbox (`app.main_nomesh`) | Debugging Phase 1 layout / spatial reasoning cheaply |
| `uv run scripts/run_oneshot.py` | Server + viewer opened on the `/oneshot` bench | Single-call, whole-scene design experiments |
| `uv run scripts/run_studio.py` | The phrase **studio** (`app.studio`) | One object at a time: noun phrase → Banana → Trellis → GLB |
| `uv run scripts/run_playground.py` | The **playground** (`app.playground`, port `8767`) | Prompt → Nano Banana image → Trellis mesh, tuning image/mesh params. Equivalent: `enx playground` |

> On Windows/PowerShell, always invoke via `uv run scripts/<name>.py` (the
> `./scripts/<name>.py` shebang form is POSIX-only).

---

## Environment variables

Set these in `server/.env`. Only `OPENROUTER_API_KEY` is required for a basic
library-mode run.

### Core

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENROUTER_API_KEY` | **Yes** | — | Routes almost every LLM reasoning step |
| `GOOGLE_API_KEY` | For from-scratch meshes | — | Nano Banana image generation (`gemini-*-flash-image`) |
| `USE_ASSET_LIBRARY` | No | `true` | `true` = match a pre-built catalog (no image/3D gen). `false` = generate every mesh from scratch |

### Mesh backends (from-scratch generation)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GENERATE_SCENE_BACKEND` | `trellis` | Whole-scene generate backend: `trellis` \| `hunyuan` \| `hunyuan-tencent` |
| `TRELLIS_BASE_URL` | hosted Modal router | Trellis 2 endpoint; no key needed for the default hosted router |
| `TRELLIS_MODEL` | `microsoft:trellis-2@4b` | Trellis model id (scripts/playground) |
| `HUNYUAN_BASE_URL` | hosted Modal router | Hunyuan (Omni) endpoint |
| `TENCENTCLOUD_SECRET_ID` / `TENCENTCLOUD_SECRET_KEY` / `TENCENTCLOUD_REGION` | — / — / `ap-singapore` | Tencent Hunyuan 3D backend |
| `RUNWARE_API_KEY` | — | Runware Hunyuan backend + playground |
| `FAL_KEY` | — | Legacy; present in the example env, not used by current paths |

### Third-party OpenAI-compatible models

Only needed if you run the corresponding model alias. On HTTP 429, keys in the
`*_ARRAY` pool rotate to the next entry.

| Variable | Model alias |
|----------|-------------|
| `MOONSHOT_API_KEY` (`_ARRAY`) | `kimi-k3` |
| `LONGCAT_API_KEY` (`_ARRAY`) | `longcat` |
| `SILICONFLOW_API_KEY` (`_ARRAY`) | `longcat-sf` |
| `ALIBABA_API_KEY` | `qwen-max-preview` |

### Tuning knobs

| Variable | Default | Purpose |
|----------|---------|---------|
| `STARSHOT_SPEND_CAP_USD` | `200` | Per-cell soft spend tripwire (pauses a cell that crosses it) |
| `STARSHOT_NEXT_OBJECT_CAP` | unset (uncapped) | Cap the anchor completion loop to N rounds; `0` = none |
| `DOWNGRADE_NOUN_PHRASE` | `true` | Run the cheap `image_prompt` distill on `gemini-flash-lite`, off the benchmark model |
| `STARSHOT_CULL_NEXT_OBJECT` | `false` | Trim `{SCENE_CONTEXT}` for huge scenes that overflow the model window |
| `STARSHOT_CULL_EARLY_REGION_FRAC` | `0` | Extra cull lever (fraction of earliest regions to drop) |
| `LIBRARY_ASSETS_SUBDIR` | `assets-optimized` | Which library asset dir to serve (`assets` = raw) |
| `STARSHOT_RUNS_DIR` | `<repo>/runs` | Where per-cell artifacts are written |
| `STARSHOT_OBJECTS_SUBDIR` | `objects` | Per-cell mesh subdir the scene bundle streams from |
| `STARSHOT_VERSIONS_DIR` | `<repo>/versions` | Prompt-template version snapshots |
| `STARSHOT_NODE_BIN` | `node` | Node binary for the mesh optimizer / symmetry |

### Production publish (optional)

Cloudflare R2 + D1 credentials for publishing scenes to the prod site:
`CLOUDFLARE_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`,
`CLOUDFLARE_API_TOKEN`, `D1_DATABASE_ID`.

---

## The pipeline, end to end

A run has **two phases**. Every step below is one LLM call whose prompt comes
from the run's prompt snapshot (see `versions/` and `server/app/core/prompt_store.py`),
and whose output is a structured JSON schema. Each call is content-addressed and
logged, so a step never re-bills on resume.

### Phase 1 — divider (recursive top-down decomposition)

Starting from the root prompt, the divider builds a tree of **zone Nodes** with
resolved world-space bounding boxes. Per zone:

1. **`zone_plan`** *(`zone_plan_root` at the root)* — authors the zone's
   high-level character/intent **and** decides whether it is *atomic* (a leaf).
2. **`overall_bbox`** *(root only)* — sizes the whole scene's canvas to the root plan.
3. **`zone_decompose`** *(non-atomic only)* — emits each child sub-zone in one
   call (id, prompt, proxy shape, sibling relationships).
4. **`child_bbox_batch`** — resolves a bounding box for **every** child in one
   call, placed in the parent's local frame.
5. **encapsulating pass** — generates the zone's physical shell / floor / ground
   (see Phase 2), run *after* its own decomposition so the shell sees what's inside it.
6. If **atomic** → hand off to the Phase 2 **anchor** pass to populate objects.
7. Recurse into each child (plan, then build).
8. **negative-space pass** — fills the interstitial gaps between a zone's named children.

Depth-first by default; `run_bfs.py` swaps in a breadth-first traversal.

### Phase 2 — generation (populate zones with objects)

Runs in three **scenarios**, each: decompose objects (1 call) → resolve every
object's bbox + yaw (1 batch call) → realize meshes.

- **`anchor_decompose`** — an atomic zone's defining objects, then a completion
  loop (**`next_object`**) that keeps proposing objects until the model says the
  zone is done (or `STARSHOT_NEXT_OBJECT_CAP` is hit).
- **`encapsulating_decompose`** — the shell/ground objects for a zone (gated: may
  decide none are needed).
- **`negative_space_decompose`** — ambient/interstitial fill over the whole scene (gated).
- **`object_bbox_batch`** — places all objects in a scenario in one call and
  solves each object's discrete yaw from its semantic orientation text.
- **`image_prompt`** — distills each object's verbose seed into a concise visual
  noun phrase (its canonical "what it is"), used for matching, symmetry, prefab
  grouping, and image generation.

### Realizing a mesh

For each placed object, one of two paths runs:

- **Asset-library mode (default):** the distilled phrase is LLM-matched to the
  closest catalog item, and a placement transform (scale/rotate/translate into
  the object's bbox + yaw) is baked into the pre-built `.glb`. No image or 3D
  generation.
- **From-scratch mode** (`USE_ASSET_LIBRARY=false`, or the dashboard **Generate**
  gate): resolve a **symmetry** cut plane → generate a studio-shot image with
  **Nano Banana** → turn it into a textured mesh with **Trellis 2** (or Hunyuan)
  → symmetrize + rescale into the bbox → run the **optimizer** (decimate + prune +
  KTX2 + Meshopt) into the served twin. Identical objects are de-duplicated via
  **prefab reuse** (built once, rescaled into each slot).

Helper LLM calls (library match, prefab grouping, symmetry) run on a cheap model
and are deliberately kept **off** the benchmark model surface.

---

## Runs, artifacts & resumability

Everything a cell produces lives under `runs/` (or `STARSHOT_RUNS_DIR`):

```
runs/<run-name>/<slot>/<model>/
├── events.jsonl                     # append-only event log — the source of truth
├── objects/                         # library-mode served GLBs (+ ref images)
└── generated/<version>/
    ├── events.generated.jsonl       # per-version resumable log
    ├── objects-generated/           # raw Trellis meshes (intermediate)
    └── objects-generated-optimized/ # served twin (KTX2/Meshopt)
```

- **`events.jsonl` is the run.** Every LLM decision (`cache.llm`), bbox, mesh,
  and gate is an event. Re-running a cell **resumes**: committed decisions replay
  from the log (instant, no re-billing) and only unfinished work runs for real.
- **Content-addressed LLM cache** means editing a prompt or swapping a model
  invalidates only the steps that actually changed.
- `board.db` / `flights.db` are SQLite indexes over the logs (dashboard board +
  the per-call "flights" cost/latency view). They are derived — safe to rebuild.
- **Generated builds are versioned**: a cell can hold any number of independent
  from-scratch versions of the same layout, each fully isolated.

---

## Scripts reference

### Launchers (`scripts/`, run with `uv run scripts/<name>.py`)

Covered in [Running it](#running-it): `run_request`, `run_bfs`, `run_noframes`,
`run_bboxes_only`, `run_oneshot`, `run_studio`, `run_playground`.

Plus two headless tools:

- **`export_scene_usd.py`** — export one cell's world-placed meshes to a single
  `.usd`/`.usdz`, no browser. E.g.
  `uv run scripts/export_scene_usd.py --run <r> --slot <s> --model <m> --version <v> --out scene.usdz`.
- **`scene_edit.py`** — reusable geometry editor that keeps a node's `bbox` event
  and its baked GLB transform in sync.

### Maintenance (`server/scripts/`, run with `uv run scripts/<name>.py` from `server/`)

Grouped by purpose — reach for these occasionally, not per run:

- **Library tooling:** `generate_library.py`, `match_library_assets.py`, `reorient_v5_assets.py`
- **Smoke tests:** `test_trellis.py`, `test_nano_banana.py`, `test_prompt_templates.py`,
  `test_flightlog.py`, `test_key_rotation.py`
- **Backends benchmarking:** `bench_hunyuan_runware.py`, `bench_hunyuan_tencent.py`
- **Backfills / migrations:** `backfill_board_cache.py`, `backfill_external_costs.py`,
  `backfill_scenes.py`, `migrate_event_schema.py`, `migrate_flights_to_sqlite.py`,
  `migrate_banana_events.py`, `migrate_generated_versions.py`, `reindex_events.py`, `rebake_runs.py`
- **Recovery:** `replay_stuck_meshes.py`, `populate_from_bundle.py`

### `enx` tasks (`enx.toml`)

`enx up` (install), `enx test` (= `run_request.py`), `enx playground` (= `run_playground.py`),
`enx open repo`.

---

## Warnings & gotchas

- **Real runs cost money.** Every step is a live LLM call, and from-scratch
  generation adds image + 3D API spend. Each cell has a **$200 soft cap**
  (`STARSHOT_SPEND_CAP_USD`) that pauses it when crossed — raise/lower it
  deliberately. Start on cheap models (`gemini-flash-lite`) and `run_bboxes_only.py`
  to validate layout before spending on meshes.
- **The asset library is not in the repo.** In the default `USE_ASSET_LIBRARY=true`
  mode, the catalog index ships but the GLBs don't. Supply them, or set
  `USE_ASSET_LIBRARY=false` to generate meshes from scratch (needs `GOOGLE_API_KEY`).
- **Very large scenes can overflow the context window.** `{SCENE_CONTEXT}` can
  approach ~1M tokens on the biggest prompts. `STARSHOT_CULL_NEXT_OBJECT=true`
  (and `STARSHOT_CULL_EARLY_REGION_FRAC`) trims it — a deliberate, opt-in
  compromise; full context is always preferred when it fits.
- **Don't hand-edit `events.jsonl` / `board.db`.** The log is the source of truth
  and resume replays it; corrupting it can wedge a cell. The SQLite DBs are
  derived and can be rebuilt from the logs (see the backfill scripts).
- **Node must be on `PATH`.** The client viewer and the mesh optimizer both shell
  out to `node`; from-scratch generation silently degrades if the optimizer's
  `node_modules` isn't installed.
- **Key rotation is reactive.** For pooled providers, a key only rolls forward on
  a 429 — a single key means no rotation.
- **`gen/` and `prod/` are separate embedded repos** (git links with no
  `.gitmodules`) and are empty on a fresh clone; they're companion projects, not
  part of the server/client build.
