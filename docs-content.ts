// ---------------------------------------------------------------------
// Doc tree. Real content sourced from the repo's API.md plus handler
// docstrings in modal_app.py. The slugs and nesting must stay in sync
// with what Docs.tsx renders.
//
// A DocNode is either a folder (groups other nodes) or a page (renders
// markdown). The slug is a stable path-style identifier used for
// selection state and as the default export filename.
// ---------------------------------------------------------------------

export type DocPage = {
  slug: string;
  title: string;
  content: string;
};
export type DocFolder = {
  type: "folder";
  title: string;
  children: DocNode[];
};
export type DocLeaf = { type: "page"; page: DocPage };
export type DocNode = DocFolder | DocLeaf;

// ---------------------------------------------------------------------
// Page bodies. Kept as exported consts so they're easy to grep for and
// easy to swap if we ever generate this file from API.md + docstrings.
// ---------------------------------------------------------------------

const introContent = `# Starshot Assets API

A single HTTP API for image-to-3D generation, hosted on Modal. One **router**
fronts the API and dispatches each request to one of several **generation
services** — currently:

- **TRELLIS.2** — image → textured GLB (\`microsoft/TRELLIS.2-4B\`), with optional
  sparse-voxel structural conditioning.
- **Hunyuan-Omni** — image **+ a control signal** (point / voxel / bbox / pose)
  → textured GLB (\`Hunyuan3D-Omni\` + the Hunyuan3D-2.1 PBR paint stack).

Every service speaks the **same job lifecycle** — you spawn against a service,
get a \`job_id\`, and poll one shared set of endpoints. Only the generation
parameters and the per-service queue/GPU pool differ. See **Services** for each
backend's parameters, and **Reference → Capacity & routing** for the dual-queue
model.

## Base URL

\`\`\`
https://starshot-aitools--starshot-assets-router-fastapi-app.modal.run
\`\`\`

FastAPI auto-generated docs are also live:

- Swagger UI: \`/docs\`
- OpenAPI schema: \`/openapi.json\`

CORS is open (\`allow_origins=["*"]\`) for development.

## Picking a service

The canonical selector is the **\`?model=\`** query parameter on \`/generate\`:

\`\`\`
POST /generate?model=trellis2
POST /generate?model=hunyuan-omni
POST /generate?model=structured-texture
\`\`\`

The path style \`POST /generate/<slug>\` still resolves but is **deprecated** —
prefer the query form. Omitting the selector falls back to the default service
(\`trellis2\`). \`GET /services\` lists the live registry.

## How it works (read this first)

A single \`/generate\` request can take ~1-3 minutes end-to-end depending on the
service, parameters, and current queue depth. Modal's HTTP edge drops idle
connections at ~60 s, so the API uses a **spawn-and-poll** pattern that is
identical across services:

1. \`POST /generate?model=<slug>\` — uploads the image (+ params) and
   **immediately returns a \`job_id\`**.
2. \`GET /jobs/{job_id}\` — non-blocking status probe. Cycles through
   \`queued\` -> \`dispatched\` -> \`running\` -> terminal (\`done\` / \`failed\` /
   \`cancelled\`).
3. \`GET /jobs/{job_id}/result\` — downloads the GLB binary once status is \`done\`.

Recommended polling cadence: 2 s. Hard timeout: 10 min on the client. Job IDs
and the polling API are **shared** across services — the router tracks which
service owns each job.

## Concepts

- **Service** — a generation backend (\`trellis2\`, \`hunyuan-omni\`,
  \`structured-texture\`) selected via \`?model=\`. Each has its own GPU pool, queue,
  analytics, and archive.
- **Job** — a single generation request. Moves through \`queued -> dispatched -> running -> done\`.
- **Phase** — sub-stages of \`running\` (\`preprocess\`, \`sample\`, \`postprocess\`, \`export\`).
- **Archive** — terminal jobs (\`done\`, \`failed\`, \`cancelled\`) persisted to a Modal Volume.

## Architecture (platform)

\`\`\`
client --HTTP--> Router (CPU, FastAPI, @modal.asgi_app)
                     |  resolve ?model= -> service
                     |  admit (per-service cap AND global cap)
            +--------+--------+
            v                 v
     trellis2 queue     hunyuan queue
            |                 |
            v                 v
   TRELLIS.2 GPU worker  Hunyuan-Omni GPU worker
\`\`\`

Production is ONE Modal app — \`starshot-assets\`: the router and BOTH GPU workers
live on it, and the router spawns either worker **in-process**. The router holds
all queueing, scheduling, and archive state and is service-agnostic; each worker
runs one service's model. See **Reference → Architecture & routing** for the
dispatcher, reconciler, and heartbeat detail, and **Reference → Capacity &
routing** for the cap model.

## Quick start

\`\`\`bash
BASE="https://starshot-aitools--starshot-assets-router-fastapi-app.modal.run"

# 1. Spawn (Trellis2)
JOB=$(curl -s -X POST "$BASE/generate?model=trellis2" \\
  -F "image=@input.png" \\
  -F "resolution=1024" \\
  -F "seed=42" | jq -r .job_id)

# 2. Poll
while :; do
  STATUS=$(curl -s "$BASE/jobs/$JOB" | jq -r .status)
  case "$STATUS" in
    done)             break ;;
    failed|cancelled) curl -s "$BASE/jobs/$JOB" | jq .error; exit 1 ;;
  esac
  sleep 2
done

# 3. Download
curl -s "$BASE/jobs/$JOB/result" -o out.glb
\`\`\`

See **Core API** for the shared lifecycle endpoints and **Services** for each
backend's parameters.

---

**Suggested reading:** [Authentication](#auth) · [POST /generate](#endpoints/generate) · [Architecture & Routing](#reference/architecture) · [Capacity & routing](#reference/capacity)

`;

const authContent = `# Authentication

No authentication is in place. Treat the URL as a shared dev endpoint,
not a production secret.

The router currently relies on IP-based rate limiting only. A token-auth
layer is on the roadmap. See **Rate Limits** for the per-IP quotas.

## What's exposed

CORS is open (\`allow_origins=["*"]\`) for development. Anyone with the
base URL can submit jobs, list active jobs, view the archive, and pull
GLB results.

## Recommended posture for production

- Front the router with a reverse proxy that enforces an auth header.
- Restrict CORS \`allow_origins\` to known origins.
- Apply tighter per-IP / per-token rate limits than the defaults in
  **Rate Limits**.

---

**Suggested reading:** [Rate Limits](#rate-limits) · [POST /generate](#endpoints/generate) · [Errors](#reference/errors)

`;

const rateLimitsContent = `# Rate Limiting

Per-IP sliding windows enforced on the router. When you exceed a limit
the server returns \`429 Too Many Requests\` with a \`Retry-After: <seconds>\`
header and a structured body:

\`\`\`json
{
  "detail": {
    "error": "rate_limited",
    "scope": "generate",
    "limit": 10,
    "window_seconds": 60.0,
    "retry_after": 7.4,
    "message": "too many requests on generate: 10/60s exceeded; retry in 7.4s"
  }
}
\`\`\`

## Limits per IP

| Endpoint scope | Limit (per IP) |
|---|---|
| \`POST /generate\` | **10** per 60 s (\`RATE_GENERATE\`) |
| \`GET /jobs*\`, \`/archive\`, \`/queue\`, \`/analytics\`, \`/jobs/{id}/cancel\`, \`/jobs/{id}/priority\`, \`/queue/pause\`, \`/queue/resume\` | **60** per 60 s (~1/s) (\`RATE_JOBS\`) |
| \`GET /health\`, \`GET /worker_status\` | unlimited |

## Client behaviour

**Clients should implement exponential backoff with jitter** on \`429\`
and \`5xx\`. Always respect \`Retry-After\` when present.

> Note: the router scales horizontally (up to 4 containers) and the
> limiter is per-container in-memory. Under burst traffic the effective
> limit can be up to ~4x the per-container limit. Treat the documented
> numbers as a *minimum guarantee*, not a hard cap.

The frontend ships with a \`fetchWithBackoff\` helper you can copy as a
starting point.

---

**Suggested reading:** [Capacity & routing](#reference/capacity) · [GET /queue](#endpoints/queue) · [Errors](#reference/errors)

`;

const generateContent = `# POST /generate

Submit an image and start a GPU job on a chosen service. Returns the \`job_id\`
immediately — **this does not return the GLB.**

**Content-Type:** \`multipart/form-data\`

## Selecting the service

\`\`\`
POST /generate?model=trellis2          # default if ?model= omitted
POST /generate?model=hunyuan-omni
POST /generate?model=structured-texture
\`\`\`

\`?model=<slug>\` is the canonical selector. The path style
\`POST /generate/<slug>\` still works but is **deprecated**.
[\`GET /services\`](#endpoints/services) lists valid slugs. The request is admitted
only if both the target service's cap **and** the global cap have headroom.

## Parameters

\`image\` (file, required) is common to every service — PNG / JPG / WEBP, RGBA with
an alpha-masked subject works best, max 10 MiB. \`priority\` (int, default \`0\`;
higher dispatches earlier while queued) is also shared.

**All other parameters are service-specific.** See each service's page:

- [**TRELLIS.2** — overview & parameters](#services/trellis2) — \`resolution\`,
  sampling steps, decimation, texture size, VLM front-alignment, aspect control,
  sparse-voxel SDEdit.
- [**Hunyuan-Omni** — parameters](#hunyuan/parameters) — \`control_type\` +
  [\`control\` payload](#hunyuan/control), \`steps\`, \`octree_resolution\`,
  \`guidance_scale\`, \`texture\`, … (start at the
  [Hunyuan-Omni overview](#hunyuan/intro)).
- [**Structural Texture** — parameters](#structured-texture/parameters) —
  \`thickness\`, \`orientation\`, \`height\`, \`length\`, \`num_orientations\`, \`align\`,
  … a Hunyuan subclass that tiles a front-aligned texture panel (start at the
  [Structural Texture overview](#structured-texture/intro)).

Unknown fields for a service are ignored.

## Success (200)

\`\`\`json
{ "job_id": "fc-01KSBABCD7XT3GJF2Q8EXMZRY" }
\`\`\`

## Errors

| Status | Error kind | When |
|---|---|---|
| \`400\` | \`bad_request\` | Empty upload, missing \`image\`, invalid service-specific field. |
| \`404\` | \`unknown_service\` | \`?model=\` slug isn't in the registry. |
| \`413\` | \`payload_too_large\` | Image exceeds the 10 MiB stored-input cap. |
| \`429\` | \`capacity\` | Per-service or global queue cap is full — retry later. |
| \`502\` | \`spawn_failed\` | Router couldn't enqueue the job (registry write failure). |

## Example

\`\`\`bash
curl -X POST "$BASE/generate?model=trellis2" \\
  -F "image=@cat.png" \\
  -F "resolution=1024" \\
  -F "seed=42" \\
  -F "object_name=cat" \\
  -F "face=front"
\`\`\`

## Notes (from handler)

Enqueue a job on the resolved service. Returns immediately with our internal
\`job_id\`; the per-service dispatcher spawns it on that service's Modal app when
capacity is available, in priority order. Use \`POST /jobs/{id}/cancel\` or
\`POST /jobs/{id}/priority\` to manage.

Image bytes are stored separately in the service's \`<slug>-job-images\` Modal
Dict from the metadata in \`<slug>-job-inputs\` so listing endpoints stay small.
The job entry is tagged with its owning service so the shared polling endpoints
can route correctly.

---

**Suggested reading:** [GET /jobs/{id}](#endpoints/jobs-id) · [GET /jobs/{id}/result](#endpoints/jobs-id-result) · [Job States](#reference/states) · [TRELLIS.2](#services/trellis2) · [Hunyuan-Omni](#hunyuan/intro)

`;

const trellis2OverviewContent = `# TRELLIS.2

Image → textured GLB using the
[microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) model
family. Optionally conditioned on a coarse **sparse-voxel draft** via SDEdit
(see **Structural Conditioning**).

\`\`\`
POST /generate?model=trellis2      multipart: image + params below
\`\`\`

Returns a \`job_id\`; poll the shared **Core API** endpoints for status + result.

## Parameters

Sent as multipart form fields alongside the \`image\` file.

| Field | Type | Default | Notes |
|---|---|---|---|
| \`seed\` | int | \`0\` | Determinism. Threaded into \`torch.manual_seed\`. |
| \`resolution\` | str | \`"1024"\` | One of \`"512"\`, \`"1024"\`, \`"1536"\`. \`1024\` and \`1536\` use cascade decoding (LR then HR). |
| \`decimation_target\` | int | \`500000\` | Target face count after mesh decimation. Halving this materially reduces postprocess time. |
| \`texture_size\` | int | \`2048\` | 1024 / 2048 / 4096. Texture bake cost scales ~quadratically. |
| \`ss_sampling_steps\` | int | \`30\` | Sparse-structure diffusion steps. |
| \`shape_slat_sampling_steps\` | int | \`30\` | Shape SLat diffusion steps. |
| \`tex_slat_sampling_steps\` | int | \`30\` | Texture SLat diffusion steps. |
| \`object_name\` | str | \`""\` | Optional subject name (e.g. \`chair\`) passed to the VLM front-orientation aligner for better grounding. Empty = generic "object". |
| \`face\` | str | \`"front"\` | Which face to point at +Z (world forward): \`front\` / \`back\` / \`left\` / \`right\` / \`top\` / \`bottom\`. |
| \`num_orientations\` | int | \`4\` | Number of candidate azimuth views rendered + scored by the VLM for alignment. |
| \`aspect_ratio\` | str | \`""\` | Optional \`"x:y:z"\` (or \`"x,y,z"\`) relative aspect ratio (\`x\` = side, \`y\` = up, \`z\` = front). The sparse-structure voxel group is anisotropically rescaled — and 90°-reoriented if its long side is flipped — *before* the SLat stages. Empty / malformed = off. |
| \`aspect_fill\` | float | \`0.9\` | Only used when \`aspect_ratio\` is set. Fraction of the voxel grid the largest target axis spans (leaves a margin). |
| \`aspect_mode\` | str | \`"nearest"\` | Only used when \`aspect_ratio\` is set. Resample kernel: \`nearest\` (hole-safe) or \`trilinear\` (smooth, can erode on stretch). |
| \`draft_voxels\` | str (JSON) | \`""\` | Optional draft sparse voxels \`{"grid","coords"}\` injected as a structural prior via SDEdit at the sparse-structure stage. Empty = off. See **Structural Conditioning**. |
| \`sdedit_t0\` | float | \`0.7\` | Only used with \`draft_voxels\`. SDEdit start time (0,1]: lower = closer to draft (flatter), higher = more surface detail (rugged). Typ. 0.5–0.75. |
| \`metallic_cap\` | float | \`1.0\` | Clamp predicted metalness to this ceiling. \`1.0\` = off; \`0.0\` = fully matte (kills glossy/glass). Color/roughness/geometry untouched. |

## Front-orientation alignment (VLM)

After generation, the worker renders an azimuth orbit, asks a VLM which view
shows the requested \`face\`, and rotates the GLB so that face points +Z. Tune
with \`object_name\`, \`face\`, and \`num_orientations\`. Skipped when \`draft_voxels\`
is supplied (the draft defines the frame).

## Modal app

Runs as the \`starshot-assets\` app (router + Trellis2 GPU worker). Core model:
\`core/trellis2-src/\`. Infra: \`modal-scripts/trellis_modal/\`.

---

**Suggested reading:** [POST /generate](#endpoints/generate) · [Structural Conditioning](#structural/overview) · [Phase Timings](#reference/phase-timings) · [Hunyuan-Omni](#hunyuan/intro)

`;

const jobsListContent = `# GET /jobs

List currently-active jobs (everything not yet in the archive), most
recent first. Use \`/archive\` for terminal jobs.

## Query parameters

| Query | Type | Default | Notes |
|---|---|---|---|
| \`limit\` | int | \`50\` | Capped at \`MAX_JOB_HISTORY=100\`. |
| \`state\` | str | \`null\` | Filter: \`queued\`, \`dispatched\`, \`running\`. |

## Response (200)

\`\`\`json
{
  "count": 2,
  "jobs": [
    {
      "job_id": "fc-01KSB...",
      "state": "running",
      "priority": 0,
      "created_at": 1748050228.10,
      "dispatched_at": 1748050228.45,
      "started_at": 1748050231.42,
      "finished_at": null,
      "elapsed": 8.3,
      "params": { "seed": 0, "resolution": "1024" },
      "image_filename": "input.png",
      "image_mime": "image/png",
      "image_size_bytes": 1248302,
      "size_bytes": null,
      "error": null,
      "modal_call_id": "fc-01KSB...",
      "phase_timings": null
    }
  ]
}
\`\`\`

## Notes (from handler)

Lists ACTIVE jobs (queued / dispatched / running) only. Terminal jobs
(\`done\` / \`failed\` / \`cancelled\`) live in \`/archive\` and are returned
via \`GET /archive\` with pagination.

Optional \`state\` filter narrows to one of \`queued\` / \`dispatched\` /
\`running\`. Other values yield an empty list (use \`/archive\` for terminal
states).

The router overlays an effective \`"running"\` state only if the worker's
heartbeat is fresh — a marker with no recent heartbeat means the
container is dead and the reconciler hasn't flipped it to \`failed\` yet.

---

**Suggested reading:** [GET /jobs/{id}](#endpoints/jobs-id) · [GET /queue](#endpoints/queue) · [GET /archive](#endpoints/archive) · [Job States](#reference/states)

`;

const jobByIdContent = `# GET /jobs/{job_id}

Non-blocking status. Safe to poll every 1–2 seconds.

\`\`\`bash
curl https://.../jobs/fc-01KSBABCD7XT3GJF2Q8EXMZRY
\`\`\`

Every response also includes the stored context (\`params\`, \`created_at\`,
\`image_filename\`, \`image_mime\`, \`modal_call_id\`, and — once the worker
finishes — \`phase_timings\`) when the job's input is still in the active
table or the archive. Omitted if the entry was pruned (>1000 archived).

## Response shapes by state

\`\`\`json
// Queued — no worker has picked it up yet (or all workers are busy)
{
  "status": "queued",
  "queue_position": 3,
  "queue_length": 5,
  "params": { "...": "..." },
  "created_at": 1748050228.10,
  "image_filename": "input.png",
  "image_mime": "image/png",
  "modal_call_id": null,
  "phase_timings": null
}
\`\`\`

\`\`\`json
// Dispatched — Modal has the call but the worker hasn't entered generate() yet
{
  "status": "dispatched",
  "dispatched_at": 1748050228.45,
  "params": { "...": "..." },
  "created_at": 1748050228.10,
  "modal_call_id": "fc-01KSBABCD7XT3GJF2Q8EXMZRY",
  "phase_timings": null
}
\`\`\`

\`\`\`json
// Actively executing on a GPU worker
{
  "status": "running",
  "started_at": 1748050231.42,
  "elapsed": 8.3,
  "dispatched_at": 1748050228.45,
  "params": { "...": "..." },
  "created_at": 1748050228.10,
  "modal_call_id": "fc-01KSBABCD7XT3GJF2Q8EXMZRY",
  "phase_timings": null
}
\`\`\`

\`\`\`json
// Done — go fetch /jobs/{id}/result
{
  "status": "done",
  "size_bytes": 7340032,
  "finished_at": 1748050361.18,
  "params": { "...": "..." },
  "created_at": 1748050228.10,
  "modal_call_id": "fc-01KSBABCD7XT3GJF2Q8EXMZRY",
  "phase_timings": {
    "decode": 0.04,
    "lock_wait": 0.12,
    "preprocess": 4.18,
    "sample": 53.71,
    "postprocess": 70.04,
    "align": 2.18,
    "export": 3.22,
    "total": 131.35,
    "idle_gap": 2.41,
    "size_bytes": 7340032,
    "resolution": "1024"
  }
}
\`\`\`

\`\`\`json
// Worker raised — do not retry blindly; surface the message
{
  "status": "failed",
  "error": {
    "type": "OSError",
    "message": "Cannot access gated repo ..."
  },
  "finished_at": 1748050241.02,
  "modal_call_id": "fc-01KSBABCD7XT3GJF2Q8EXMZRY",
  "phase_timings": null
}
\`\`\`

\`\`\`json
// Cancelled by user
{
  "status": "cancelled",
  "finished_at": 1748050241.02,
  "modal_call_id": "fc-01KSBABCD7XT3GJF2Q8EXMZRY"
}
\`\`\`

Treat \`queued\`, \`dispatched\`, and \`running\` as "keep polling." Only
\`done\`, \`failed\`, and \`cancelled\` are terminal.

## Notes (from handler)

Status probe. \`"running"\` is distinguished from \`"queued"\` by a marker
the worker writes to a shared \`modal.Dict\` when \`generate()\` begins
executing — purely metadata lookup, no GPU call.

NOTE: \`timeout=0\` on Modal's \`FunctionCall.get\` always raises
\`TimeoutError\` (round-trip is nonzero). The handler uses a small
positive timeout so results that are already ready get picked up.

---

**Suggested reading:** [Job States](#reference/states) · [Phase Timings](#reference/phase-timings) · [GET /jobs/{id}/result](#endpoints/jobs-id-result) · [Errors](#reference/errors)

`;

const jobResultContent = `# GET /jobs/{job_id}/result

Downloads the GLB binary. Call this only after \`/jobs/{job_id}\` returns
\`"status":"done"\`.

\`\`\`
Content-Type: model/gltf-binary
Content-Disposition: attachment; filename="trellis2.glb"
<binary GLB bytes>
\`\`\`

## Errors

| Status | Error kind | When |
|---|---|---|
| \`409\` | \`cancelled\` | Job was cancelled. |
| \`425\` | \`still_queued\` | Job hasn't been dispatched yet. |
| \`425\` | \`not_dispatched\` | Job has no underlying Modal call id (transient). |
| \`425\` | \`still_pending\` | Worker is still running — poll \`/jobs/{id}\` first. |
| \`502\` | \`worker_failure\` | The job raised an exception — read \`detail.message\`. |

## Persistence

The GLB lives in Modal's transient \`FunctionCall\` result store, **not**
in a Volume. **Save your own copy as soon as you download it** — once
the call is garbage-collected or the archive entry is pruned, the GLB
is unrecoverable.

## Notes (from handler)

Download the result blob. For active or recently-finished jobs the
router still re-peeks the Modal \`FunctionCall\` (fast path); for older
jobs whose \`FunctionCall\` result has aged out it falls back to the
archive volume.

The MIME map currently understood by the archive fallback:

| Extension | Content-Type |
|---|---|
| \`glb\` | \`model/gltf-binary\` |
| \`ply\` | \`application/octet-stream\` |
| \`png\` | \`image/png\` |
| \`jpg\` / \`jpeg\` | \`image/jpeg\` |

---

**Suggested reading:** [GET /jobs/{id}](#endpoints/jobs-id) · [POST /generate](#endpoints/generate) · [Phase Timings](#reference/phase-timings)

`;

const jobCancelContent = `# POST /jobs/{job_id}/cancel

Cancel a job. If \`queued\`, it's removed from the queue. If \`dispatched\`
or \`running\`, Modal's \`FunctionCall.cancel()\` is invoked on the
underlying call so the GPU container is freed (the container itself is
**not** terminated — it stays warm for the next job).

Terminal jobs return a no-op.

## Response (200)

\`\`\`json
// In-flight: cancelled
{ "job_id": "fc-...", "state": "cancelled" }

// Already terminal: no-op
{ "job_id": "fc-...", "state": "done", "noop": true }
\`\`\`

## Errors

| Status | Error kind | When |
|---|---|---|
| \`404\` | \`not_found\` | Unknown \`job_id\`. |
| \`502\` | \`cancel_failed\` | Modal's \`FunctionCall.cancel()\` raised. |

## Notes (from handler)

If \`queued\`, removes it from the queue. If \`dispatched\` or \`running\`,
calls Modal's \`FunctionCall.cancel()\`. If the job is already terminal
(in the archive), no-op.

After a successful cancel a dispatch pass is triggered to fill the
slot that just opened.

---

**Suggested reading:** [POST /jobs/{id}/priority](#endpoints/jobs-priority) · [Job States](#reference/states) · [GET /jobs/{id}](#endpoints/jobs-id)

`;

const jobPriorityContent = `# POST /jobs/{job_id}/priority

Re-prioritize a **queued** job. Higher numeric priority dispatches
first. Already-dispatched jobs cannot be reordered (Modal owns the
scheduling at that point).

**Body:** form field \`priority\` (int).

## Response (200)

\`\`\`json
{ "job_id": "fc-...", "state": "queued", "priority": 5 }
\`\`\`

## Errors

| Status | Error kind | When |
|---|---|---|
| \`404\` | \`not_found\` | Unknown \`job_id\`. |
| \`409\` | \`not_queued\` | Job has already been dispatched. |

## Notes (from handler)

Re-prioritize a queued job. Higher priority dispatches first. Only
valid while \`state == "queued"\` — once dispatched to Modal the router
can't reorder.

After the update the dispatcher is kicked to re-evaluate ordering.

---

**Suggested reading:** [POST /jobs/{id}/cancel](#endpoints/jobs-cancel) · [GET /queue](#endpoints/queue) · [Capacity & routing](#reference/capacity)

`;

const queueContent = `# GET /queue

Current dispatcher state and queue ordering. Also opportunistically
writes one analytics sample per ~30 s (so dashboards that poll this
endpoint get a free time-series).

## Multi-service

Pass \`?service=<slug>\` for one service's queue (atomic). Omitting it returns the
**aggregate** with a \`by_service\` breakdown and a \`limits\` block (per-service +
global caps):

\`\`\`
GET /queue?service=hunyuan-omni     # this service only
GET /queue                          # aggregate + by_service + limits
\`\`\`

## Response (200)

\`\`\`json
{
  "paused": false,
  "max_in_flight": 7,
  "in_flight": 2,
  "queued": 3,
  "containers": 3,
  "containers_busy": 2,
  "containers_warm": 1,
  "containers_warming": 0,
  "containers_by_phase": { "running": 2, "idle": 1, "booting": 0, "loading": 0 },
  "spawn_ratio": 0.75,
  "target_in_flight": 4,
  "slots_until_next_spawn": 1,
  "queue": [
    { "position": 0, "job_id": "fc-A...", "priority": 5, "created_at": 1748050240, "image_filename": "a.png", "params": {} },
    { "position": 1, "job_id": "fc-B...", "priority": 0, "created_at": 1748050245, "image_filename": "b.png", "params": {} }
  ],
  "in_flight_jobs": [
    { "job_id": "fc-C...", "state": "running", "dispatched_at": 1748050231, "started_at": 1748050233, "elapsed": 12.4, "image_filename": "c.png", "params": {} }
  ]
}
\`\`\`

## Field reference

- \`max_in_flight\` — \`WORKER_MAX_CONTAINERS\` ceiling.
- \`containers*\` fields — live counts from the worker heartbeat channel
  (\`trellis2-container-state\`), summarized by phase.
- \`target_in_flight\` — what the dispatcher *wants* to run given the
  current queue depth, computed as
  \`ceil(queue_depth * SPAWN_RATIO)\` clamped to \`[1, max_in_flight]\`.
- \`slots_until_next_spawn\` — how many more jobs need to be queued
  before another cold container is eligible to spawn.

## Notes (from handler)

Snapshot of the current queue: ordered list of queued jobs plus the
set of dispatched/running jobs. Also writes one analytics sample (at
most one per \`MIN_SAMPLE_INTERVAL\` seconds).

Only counts a job as \`"running"\` if the worker's heartbeat is fresh —
a stale marker means the container is dead and the reconciler will
flip the job to \`failed\` shortly.

---

**Suggested reading:** [GET /analytics](#endpoints/analytics) · [Capacity & routing](#reference/capacity) · [POST /queue/pause](#endpoints/queue-pause)

`;

const queuePauseContent = `# POST /queue/pause

Stop dispatching new jobs. In-flight jobs keep running; queued jobs
stay queued indefinitely.

## Response (200)

\`\`\`json
{ "paused": true }
\`\`\`

## Notes (from handler)

Stop dispatching new jobs. In-flight jobs continue. Queued jobs stay
queued indefinitely until \`/queue/resume\` is called.

The pause flag is persisted to the \`trellis2-queue-config\` Modal Dict
under the \`"config"\` key so it survives router restarts.

---

**Suggested reading:** [POST /queue/resume](#endpoints/queue-resume) · [GET /queue](#endpoints/queue)

`;

const queueResumeContent = `# POST /queue/resume

Re-enable dispatch and immediately attempt to spawn pending jobs (in
priority order) up to capacity.

## Response (200)

\`\`\`json
{ "paused": false }
\`\`\`

## Notes (from handler)

Re-enable dispatch and immediately attempt to spawn pending jobs (in
priority order) up to capacity. A dispatch pass is kicked off in the
background — the response returns before the spawn round-trips
complete.

---

**Suggested reading:** [POST /queue/pause](#endpoints/queue-pause) · [GET /queue](#endpoints/queue)

`;

const analyticsContent = `# GET /analytics

Time-series ring buffer of queue / dispatched / running / container
counts. Samples are captured opportunistically inside \`/queue\`
(rate-limited to one per \`MIN_SAMPLE_INTERVAL=30 s\`). Up to
\`MAX_SAMPLES=2000\` retained (~16 hours at default cadence).

## Query parameters

| Query | Type | Default | Notes |
|---|---|---|---|
| \`service\` | str | \`null\` | One service's samples (atomic). Omit for the aggregate. |
| \`since\` | float | \`null\` | Only return samples with \`t > since\` (epoch seconds). Incremental polling. |
| \`limit\` | int | \`500\` | Cap on number returned. |

## Response (200)

\`\`\`json
{
  "count": 2,
  "sample_interval_seconds": 30.0,
  "max_samples_retained": 2000,
  "samples": [
    {
      "t": 1748050200.0,
      "queued": 3,
      "dispatched": 0,
      "running": 2,
      "in_flight": 2,
      "containers": 3,
      "containers_busy": 2,
      "containers_warm": 1,
      "containers_warming": 0,
      "paused": false
    }
  ]
}
\`\`\`

Newest-last so client-side charts can append incrementally.

## Notes (from handler)

Returns analytics time-series samples (queued / dispatched / running /
container counts over time). Newest last so the client can append to
a chart directly.

Samples are captured opportunistically on \`/queue\` polls, rate-limited
to one per ~30 s. Up to \`MAX_SAMPLES\` are retained in a ring buffer
(~6 hours at default cadence in the handler's comment; \`MAX_SAMPLES=2000\`
yields closer to ~16 h at the documented 30 s cadence).

---

**Suggested reading:** [GET /queue](#endpoints/queue) · [GET /archive](#endpoints/archive) · [Phase Timings](#reference/phase-timings)

`;

const archiveContent = `# GET /archive

Paginated history of terminal jobs (\`done\` / \`failed\` / \`cancelled\`).
Newest-first by \`finished_at\` (falls back to \`created_at\`).

## Query parameters

| Query | Type | Default | Notes |
|---|---|---|---|
| \`service\` | str | \`null\` | One service's archive (atomic). Omit for the aggregate across services. |
| \`limit\` | int | \`20\` | Capped at \`MAX_ARCHIVE_PAGE=100\`. |
| \`offset\` | int | \`0\` | Offset-based pagination. |
| \`state\` | str | \`null\` | Optional filter (\`done\` / \`failed\` / \`cancelled\`). |
| \`since\` | float | \`null\` | Only return entries with \`finished_at > since\`. Use for incremental polling. |

## Response (200)

\`\`\`json
{
  "total": 84,
  "limit": 20,
  "offset": 0,
  "has_more": true,
  "jobs": [
    {
      "job_id": "fc-...",
      "state": "done",
      "priority": 0,
      "created_at": 1748050228.10,
      "dispatched_at": 1748050228.45,
      "started_at": 1748050231.42,
      "finished_at": 1748050361.18,
      "params": { "...": "..." },
      "image_filename": "input.png",
      "image_mime": "image/png",
      "image_size_bytes": 1248302,
      "size_bytes": 7340032,
      "error": null,
      "modal_call_id": "fc-...",
      "phase_timings": { "...": "..." }
    }
  ]
}
\`\`\`

Archive entries are evicted oldest-first when total exceeds
\`MAX_ARCHIVE_HISTORY=1000\`.

## Notes (from handler)

Paginated history of terminal jobs (\`done\` / \`failed\` / \`cancelled\`).
Newest first by \`finished_at\` (falls back to \`created_at\`).

The \`total\` field reflects the state-filtered set BEFORE the \`since\`
filter so a client using incremental polling can still display a
stable "X of Y" counter — otherwise Y would collapse to "entries
newer than my watermark" on every poll.

Internally the handler serves from a cached manifest. A stale cache
triggers one volume walk; subsequent requests within
\`ARCHIVE_CACHE_TTL\` serve from RAM.

---

**Suggested reading:** [GET /jobs](#endpoints/jobs) · [GET /analytics](#endpoints/analytics) · [Job States](#reference/states)

`;

const healthContent = `# GET /health

Liveness check for the router. No params. Service-agnostic — returns 200 from any
healthy router container regardless of GPU worker state.

\`\`\`bash
curl https://.../health
\`\`\`

## Response (200)

\`\`\`json
{
  "status": "ok",
  "services": ["trellis2", "hunyuan-omni", "structured-texture"],
  "default_service": "trellis2"
}
\`\`\`

This endpoint is unrate-limited and does **not** probe any GPU worker pool — use
\`/worker_status\` (optionally \`?service=<slug>\`) for per-service load state, or
\`/services\` for the registry.

---

**Suggested reading:** [GET /worker_status](#endpoints/worker-status) · [GET /services](#endpoints/services) · [Architecture & Routing](#reference/architecture)

`;

const servicesEndpointContent = `# GET /services

Lists the generation services registered on the router — the valid \`?model=\`
slugs for \`POST /generate\`. Service-agnostic, unrate-limited.

\`\`\`bash
curl https://.../services
\`\`\`

## Response (200)

\`\`\`json
{
  "default": "trellis2",
  "services": [
    {
      "slug": "trellis2",
      "app": "starshot-assets",
      "requires_text": false,
      "limits": { "max_gpu": 7, "max_queue": 100 }
    },
    {
      "slug": "hunyuan-omni",
      "app": "starshot-assets",
      "requires_text": false,
      "limits": { "max_gpu": 2, "max_queue": 50 }
    },
    {
      "slug": "structured-texture",
      "app": "starshot-assets",
      "requires_text": false,
      "limits": { "max_gpu": 2, "max_queue": 50 }
    }
  ],
  "global_limits": { "max_gpu": 8, "max_queue": 120 }
}
\`\`\`

Each entry is a \`ServiceSpec\` from \`modal-scripts/_shared/service.py\`; the
\`limits\` reflect the per-service caps and the \`global_limits\` the shared ceiling
on their sum (see **Reference → Capacity & routing**).

---

**Suggested reading:** [POST /generate](#endpoints/generate) · [TRELLIS.2](#services/trellis2) · [Hunyuan-Omni](#hunyuan/intro)

`;

const workerStatusContent = `# GET /worker_status

Cheap probe for the GPU worker. **Backed by a shared key-value store —
does NOT dispatch a call to the GPU worker**, so it returns in
milliseconds and never queues behind a running \`/generate\`. Safe to
poll freely.

The state is written by each GPU worker as it finishes its \`enter()\`
load phase, so it reflects the most recent worker's load result
(last-write wins across the pool).

Pass \`?service=<slug>\` to probe one service's worker pool; omit it for the
default service. Each service publishes into its own \`<slug>-worker-state\` /
\`<slug>-container-state\` Dicts, so the probe never crosses services.

## Response shapes

| Status | Response |
|---|---|
| Worker healthy | \`200\` \`{"loaded": true, "error": null, "updated_at": 1748048192.1}\` |
| Worker booted but load failed (e.g. HF auth) | \`200\` \`{"loaded": false, "error": {"type": "OSError", "message": "...", "traceback": "..."}, "updated_at": ...}\` |
| No worker has ever published state (cold pool) | \`503\` \`{"detail": {"error": "worker_unavailable", "message": "no worker has published state yet — pool may be cold"}}\` |
| Internal error reading shared state | \`502\` \`{"detail": {"error": "state_unavailable", "type": "...", "message": "..."}}\` |

## Notes (from handler)

Fast probe — reads the worker's last published state from a shared
\`modal.Dict\`. Does NOT dispatch a call to the GPU worker, so it never
queues behind an in-flight \`/generate\` and costs no GPU time. Returns
\`503\` only if no worker has ever published state (i.e. cold pool that
has never booted).

Response includes a rich per-container view: every container that has
phoned home recently with its current phase
(\`booting\` / \`loading\` / \`loaded\` / \`running\` / \`idle\` / \`failed_load\` /
\`shutdown\`) plus how long ago we last heard from it. Counts grouped by
phase let the router (and humans) tell at a glance whether the pool
is healthy, warming, or stuck.

Containers whose heartbeat is older than \`WORKER_HEARTBEAT_STALE_AFTER\`
get a \`_stale\` suffix on their phase (e.g. \`running_stale\`) — the
phase is what they LAST said but the router no longer trusts they're
actually doing it.

---

**Suggested reading:** [GET /health](#endpoints/health) · [Architecture & Routing](#reference/architecture) · [Phase Timings](#reference/phase-timings)

`;

const architectureContent = `# Architecture & Routing

Production is ONE Modal app — \`starshot-assets\`: a CPU **router** that fronts
HTTP plus one **GPU worker per service** (Trellis2 + Hunyuan), all on the same
app. The router spawns whichever worker a request selected with \`?model=\`
**in-process**. The router is **service-agnostic**: it holds all queueing,
scheduling, and archive state. Workers are stateless between calls.

## Components

| Component | Where | Concurrency | Responsibility |
|---|---|---|---|
| \`Router\` cls | CPU container, FastAPI under \`@modal.asgi_app\` | \`max_inputs=ROUTER_MAX_INFLIGHT\` (100 per replica), up to \`ROUTER_MAX_CONTAINERS\` (4) replicas | HTTP API, \`?model=\` resolution, job admission (per-service + global caps), per-service dispatcher loops, reconciler, archive write/read. |
| Service GPU worker | GPU container on the SAME \`starshot-assets\` app (own image per worker) | \`max_inputs\` per the service (1–2 per GPU) | Loads that service's model, runs inference, publishes heartbeats + phase timings. \`GPUWorker\` (Trellis2) / \`HunyuanOmni\` both inherit \`BaseGPUWorker\`. |
| \`modal.Dict\`s | Distributed KV, **namespaced per service** | Shared across replicas | \`<slug>-job-inputs\`, \`<slug>-job-images\`, \`<slug>-running-jobs\`, \`<slug>-container-state\`, \`<slug>-worker-state\`, \`<slug>-queue-config\`, \`<slug>-analytics-samples\`. |
| Archive \`modal.Volume\` | Mounted on router | Shared | Per-job JSON metadata + GLB blobs, per-service shard layout. |

The per-service \`modal.Dict\` namespacing is produced by \`ServiceState(slug)\` in
\`modal-scripts/_shared/gpu_worker.py\`, so \`trellis2-*\` and \`hunyuan-*\` keys never
collide and each service scales independently.

## Request flow

A typical \`/generate?model=<slug>\` → result download flow:

1. **\`POST /generate?model=<slug>\`** — router resolves the service, checks the
   per-service + global caps, writes the image to \`<slug>-job-images[jid]\` and
   params to \`<slug>-job-inputs[jid]\` in state \`queued\`, tags the entry with its
   service, and returns the \`job_id\` (no Modal call yet).
2. **Dispatcher tick** — every \`DISPATCH_TICK_SECONDS\` (3 s) the router runs
   \`_reconcile\` then \`_dispatch\` per service. \`_dispatch\` walks that service's
   queued jobs in priority order and, for each one it has capacity for, calls the
   service's spawn adapter (\`spawn_job(service, image, params)\`) and writes the
   resulting \`modal_call_id\` back with state \`dispatched\`.
3. **Worker boot** — Modal cold-starts that service's GPU container if needed.
   \`@modal.enter()\` loads the model once per container lifetime and publishes
   \`booting → loading → loaded\` phases into \`<slug>-container-state\`.
4. **Worker run** — inference writes \`started_at\` into \`<slug>-running-jobs\` and
   emits heartbeats. The next reconciler pass flips \`dispatched → running\`.
5. **Worker return** — the worker returns GLB bytes plus a \`phase_timings\` dict.
   The reconciler peeks the call, archives the job (metadata + GLB → volume), and
   removes it from \`<slug>-job-inputs\`.
6. **\`GET /jobs/{id}/result\`** — fast path peeks the Modal \`FunctionCall\`; if
   the per-call store aged out, falls back to the archive volume blob.

## Service registry + spawn adapters

Each service registers a \`ServiceSpec\` (slug, Modal app name, worker class) and a
**spawn adapter** into \`SERVICE_REGISTRY\` (\`modal-scripts/_shared/service.py\` +
\`_shared/router/dispatch.py\`). In production BOTH workers live on the same
\`starshot-assets\` app as the router and are spawned **in-process** — Trellis2 via
\`GPUWorker().generate.spawn\`, Hunyuan via \`HunyuanOmni().inference.spawn\`. The
dispatcher is uniform — it just calls the registered adapter for the job's
service.

## Dispatcher policy

\`_dispatch\` runs under a per-replica \`asyncio.Lock\` so two concurrent triggers
can't double-spawn the same queued job. Capacity is enforced **twice** per
admit/spawn: against the job's service cap (\`<SLUG>_MAX_GPU\` / \`_MAX_QUEUE\`) and
against the global cap on the sum across services (\`STARSHOT_GLOBAL_MAX_*\`). See
**Reference → Capacity & routing**. Within a service the model is **warm-aware**:

- Counts how many of that service's containers are \`loaded\` / \`idle\` (warm) via
  \`<slug>-container-state\`. Warm slots are spent before cold ones.
- Throttles new cold starts via \`SPAWN_RATIO\` — at most
  \`ceil(SPAWN_RATIO × queued_count)\` cold containers spawned at once.
- Hard ceiling at that service's GPU cap (and the global GPU cap on the sum).

## Reconciler

\`_reconcile\` is the only place the router writes terminal state. On
every dispatcher tick it walks the \`dispatched | running\` jobs and:

- Peeks the Modal call (\`timeout=0.3 s\`). Success → \`done\` + archive.
- Detects \`dispatched → running\` transitions when a fresh heartbeat
  shows up in \`running_jobs\`.
- On \`TimeoutError\` (call still pending): checks heartbeat freshness
  on both channels. \`running\` + both stale ⇒ \`WorkerHeartbeatLost\`
  failure. \`dispatched\` + stuck past \`WATCHDOG_DISPATCHED_TIMEOUT\`
  ⇒ \`WatchdogTimeout\` failure.
- Any other exception during peek ⇒ \`failed\` with the exception kind.

## Heartbeat channels

Two independent channels feed liveness. The router treats a job as
dead only when **both** go stale, so a single Dict-write blip can't
kill a healthy job.

| Channel | Key | Written by | Read by |
|---|---|---|---|
| \`running_jobs\` | \`modal_call_id\` | Worker on each tick of its heartbeat thread | Reconciler, status endpoints |
| \`container_state\` | container id | Worker on each phase change + periodic refresh | Reconciler, \`/worker_status\`, capacity math |

Both stale after \`WORKER_HEARTBEAT_STALE_AFTER\` (120 s) of no writes.

## Multi-replica notes

The router scales to \`ROUTER_MAX_CONTAINERS=4\`. Most shared state
lives in Modal Dicts (atomic per key), so writes from different
replicas don't corrupt. The archive volume's manifest cache is
per-replica — each replica refreshes from disk on
\`ARCHIVE_CACHE_TTL\` (30 s), so newly-archived jobs surface in
\`/archive\` listings within at most one refresh window regardless of
which replica handled the write.

## Failure modes

| Mode | Detected by | Outcome |
|---|---|---|
| Worker raises in \`generate()\` | Reconciler peek raises | \`failed\` with the exception kind/message |
| Container dies during boot (no heartbeat) | Watchdog: \`dispatched_at + WATCHDOG_DISPATCHED_TIMEOUT < now\` | \`WatchdogTimeout\` |
| Container dies mid-run | Both heartbeat channels go stale | \`WorkerHeartbeatLost\` |
| Router restart | Startup wipes \`job_inputs\` of non-terminal jobs (5-min cooldown so a scale-up doesn't nuke a sibling's queue) | Affected jobs vanish; client gets \`not_found\` |
| Modal call expired after archive | \`/jobs/{id}/result\` peek raises | Falls back to GLB blob on the archive volume |

---

**Suggested reading:** [Capacity & routing](#reference/capacity) · [Job States](#reference/states) · [GET /worker_status](#endpoints/worker-status) · [GET /services](#endpoints/services)

`;

const statesContent = `# Job States

A job moves through a small state machine. Every transition has a
single cause; terminal states are reached exactly once.

> \`pending\` in old API responses is an alias for \`queued\` — the
> public status string emitted by \`/jobs/{id}\` for a queued job is
> \`"queued"\`, but \`"pending"\` may still appear in client code.

## State descriptions

| State | Terminal? | Meaning |
|---|---|---|
| \`queued\` | no | Accepted, sitting in \`job_inputs\` waiting for a worker slot. |
| \`dispatched\` | no | \`FunctionCall.spawn.aio()\` returned a \`modal_call_id\`, but no worker has entered \`generate()\` yet (cold start or Modal scheduler queueing). |
| \`running\` | no | Worker is actively executing. Router detects this from a fresh heartbeat in \`running_jobs[call_id]\`. |
| \`done\` | **yes** | Worker returned successfully. Result blob persisted to the archive volume. |
| \`failed\` | **yes** | Worker raised, watchdog fired, or heartbeats went stale. \`error\` field carries details. |
| \`cancelled\` | **yes** | User issued \`POST /jobs/{id}/cancel\` from any non-terminal state. |

## Transitions

| From | To | Trigger |
|---|---|---|
| \`queued\` | \`dispatched\` | Dispatcher picks the job up and \`FunctionCall.spawn\`s it. |
| \`dispatched\` | \`running\` | Worker enters \`generate()\` and publishes a heartbeat into \`running_jobs\`. |
| \`running\` | \`done\` | Worker returns. Router peeks the call, archives the result. |
| \`running\` | \`failed\` | Worker raises inside \`generate()\`, OR both heartbeat channels go stale (\`WorkerHeartbeatLost\`). |
| \`dispatched\` | \`failed\` | Watchdog timeout — dispatched longer than the ceiling without ever publishing a heartbeat (\`WatchdogTimeout\`, container died during boot). |
| any non-terminal | \`cancelled\` | User posts \`/jobs/{id}/cancel\`. Modal \`FunctionCall.cancel()\` is called when the job has already been spawned. |

Once terminal, a job is moved out of \`job_inputs\` and into the archive
volume. The GLB result lives in Modal's transient \`FunctionCall\`
result store until you fetch it (or until the archive copy takes over
once Modal's TTL expires).

## Router-induced \`failed\` transitions

\`failed\` can be raised by two router-side mechanisms even if the
worker never returns:

- **WorkerHeartbeatLost** — both the call-keyed and container-keyed
  heartbeat channels go stale (>120 s).
- **WatchdogTimeout** — \`dispatched\` state held longer than the
  watchdog ceiling without ever publishing a heartbeat (i.e.
  container died during boot).

## Polling guidance

Treat \`queued\`, \`dispatched\`, and \`running\` as "keep polling." Only
\`done\`, \`failed\`, and \`cancelled\` are terminal.

---

**Suggested reading:** [GET /jobs/{id}](#endpoints/jobs-id) · [Phase Timings](#reference/phase-timings) · [Errors](#reference/errors)

`;

const errorsContent = `# Errors

All non-2xx responses use FastAPI's standard envelope:

\`\`\`json
{
  "detail": {
    "error": "worker_failure",
    "type": "OSError",
    "message": "Cannot access gated repo ..."
  }
}
\`\`\`

\`detail.error\` is a stable machine-readable kind. \`detail.message\` is
human-readable. \`detail.type\` is the Python exception class when the
failure came from worker code.

## Known error kinds

| Kind | Meaning |
|---|---|
| \`bad_request\` | Validation error on the request (missing field, bad enum). |
| \`payload_too_large\` | Image exceeds the 10 MiB stored-input cap. |
| \`not_found\` | Unknown \`job_id\` or job pruned from history. |
| \`not_queued\` | Attempted to reprioritize a job that's already dispatched. |
| \`still_queued\` | Tried to fetch \`/result\` before the job was dispatched. |
| \`not_dispatched\` | Job has no underlying Modal call yet (transient). |
| \`still_pending\` | Worker is still running — poll \`/jobs/{id}\` first. |
| \`cancelled\` | Tried to download the result of a cancelled job. |
| \`spawn_failed\` | Router couldn't enqueue the job (registry write failure). |
| \`cancel_failed\` | Modal's \`FunctionCall.cancel()\` raised. |
| \`worker_failure\` | The job raised an exception inside \`generate()\`. |
| \`worker_timeout\` | The worker exceeded its wall-clock budget. |
| \`worker_unavailable\` | No worker has published state yet (cold pool). |
| \`worker_unreachable\` | Worker heartbeat lost while job was in flight. |
| \`registry_unavailable\` | Modal Dict read/write failure. |
| \`state_unavailable\` | Internal error reading shared state. |
| \`rate_limited\` | Per-IP rate quota exceeded — see **Rate Limits**. |

## Worker-raised failures

When a worker raises inside \`generate()\` the failure is surfaced on
\`/jobs/{id}\` with the structured error:

\`\`\`json
{
  "status": "failed",
  "error": {
    "type": "OSError",
    "message": "Cannot access gated repo ..."
  },
  "finished_at": 1748050241.02,
  "modal_call_id": "fc-..."
}
\`\`\`

The router additionally injects two synthetic failure types when the
worker never returns at all:

- \`WatchdogTimeout\` — \`dispatched\` longer than the watchdog ceiling
  with no heartbeat (container died during boot).
- \`WorkerHeartbeatLost\` — both the call-keyed and container-keyed
  heartbeat channels go stale (>120 s).

---

**Suggested reading:** [Job States](#reference/states) · [Rate Limits](#rate-limits) · [POST /generate](#endpoints/generate)

`;

const phaseTimingsContent = `# Phase Timings

Per-call timings emitted by the worker at the end of \`generate()\` and
echoed on \`/jobs/{id}\` (only for terminal \`done\` responses) and inside
each \`/jobs\` / \`/archive\` entry.

## Fields

| Field | Unit | Meaning |
|---|---|---|
| \`decode\` | seconds | PIL decode of uploaded bytes. CPU, lock-free. |
| \`lock_wait\` | seconds | Time blocked acquiring the worker's pipeline lock. >5 s means another in-flight call held the GPU. |
| \`preprocess\` | seconds | \`pipeline.preprocess_image\` (rembg). CPU, lock-held. |
| \`sample\` | seconds | \`pipeline.run\`: sparse structure + shape SLat + tex SLat diffusion. **GPU-bound**, lock-held. |
| \`postprocess\` | seconds | \`o_voxel.postprocess.to_glb\`: mesh extraction, decimation, texture bake. CPU, lock-free. |
| \`align\` | seconds | Front-orientation alignment: render an azimuth orbit + ask the VLM (Trellis2: Gemini; Hunyuan / Structural Texture: Qwen) which view is the canonical front, then rotate the GLB so it points +Z (world forward). Anti-Janus. GPU render + network, lock-free. Omitted on jobs from before alignment shipped. |
| \`export\` | seconds | \`glb.export(...)\` + file readback. CPU, lock-free. |
| \`total\` | seconds | End-to-end wall-clock for this call inside \`generate()\`. |
| \`idle_gap\` | seconds | Time this container sat idle since its previous job finished. \`null\` for the first call on a container. Large values with a non-empty queue point to dispatcher lag, not pipeline slowness. |
| \`size_bytes\` | int | GLB output size. |
| \`resolution\` | str | Echo of the resolution used. |

## Lock structure

\`pipeline.preprocess_image\` and \`pipeline.run\` are held under a
\`threading.Lock\` because the pipeline carries mutable state (global
\`torch.manual_seed\`, \`image_cond_model.image_size\`, \`low_vram\` module
moves). Mesh extraction, decimation, texture bake and GLB serialize
run **outside** that lock — they release the GIL through
numpy / trimesh / o_voxel C extensions, so two concurrent jobs naturally
overlap their CPU postprocess on separate cores.

## Approximate wall-clock (warm A100, no contention)

| Resolution | Sample (GPU) | Postprocess (CPU) | Total wall |
|---|---|---|---|
| \`512\` | ~5 s | ~30 s | ~40 s |
| \`1024\` (cascade) | ~54 s | ~70 s | ~130 s |
| \`1536\` (cascade) | ~120 s | ~140 s | ~270 s |

## Observability

Every completed job emits a single structured log line on the worker:

\`\`\`
[trellis.phase] res=1024 total=131.35s decode=0.04s lock_wait=0.12s preprocess=4.18s sample=53.71s postprocess=70.04s align=2.18s export=3.22s idle_gap=2.41s size_bytes=7340032
\`\`\`

And on the router side, for each dispatch:

\`\`\`
[dispatch.spawn] jid=480d1c7f... mcid=fc-01KSB... dispatched_at=1748050228.45 priority=0
[trellis.start]  call_id=fc-01KSB... container=def456... started_at=1748050231.42
\`\`\`

The wall-clock gap between \`[dispatch.spawn]\` and \`[trellis.start]\` is
Modal's scheduling + cold-start overhead. The \`lock_wait\` field inside
\`[trellis.phase]\` is how long this call waited for the in-container
pipeline lock — a useful tuning signal for whether \`max_inputs\` is
set correctly.

---

**Suggested reading:** [Job States](#reference/states) · [GET /jobs/{id}](#endpoints/jobs-id) · [GET /analytics](#endpoints/analytics)

`;

// ---------------------------------------------------------------------
// Structural Conditioning (Sparse Voxels + SDEdit) — its own section because
// it's a multi-stage feature, not a single request parameter.
// ---------------------------------------------------------------------
const scOverviewContent = `# Structural Conditioning (Sparse Voxels + SDEdit)

A way to give generation a **coarse structural prior** instead of leaving global
shape entirely to the image. You author (or have an LLM author) a low-res sparse
voxel **draft**, and it is injected into the sparse-structure stage via **SDEdit**
so the model *refines* the draft into a detailed object rather than hallucinating
structure from scratch.

This targets the cases plain image-to-3D struggles with: flat walls that warp,
repeating tiles that drift, cylinders/ovals that come out lumpy, and random global
orientation. It is fully **optional** — with no draft, generation is unchanged.

## How it flows

1. **Draft** — a binary voxel grid (≤ 64³), built in the **Sparse Voxel** sub-tab
   (Try It / Experiment) or emitted by an LLM via the build protocol.
2. **Power-of-two embed** — the draft is centered and integer-rescaled (powers of
   two only, so a voxel becomes a clean N×N×N block — no interpolation, no holes)
   into the sparse-structure manifold.
3. **Encode + SDEdit** — encoded to the flow latent, noised to a start time
   \`t0\`, and the sparse-structure sampler runs from \`t0 → 0\`. Lower \`t0\` sticks
   closer to the draft; higher gives the model more freedom.
4. **Downstream unchanged** — the resulting active voxels feed the shape & texture
   SLat stages as usual. **Orientation alignment is skipped** when a draft is
   present (the draft defines the frame).

## Request parameters

These are sent to \`POST /generate\` alongside the usual fields (all optional):

| Field | Type | Default | Meaning |
|---|---|---|---|
| \`draft_voxels\` | str (JSON) | \`""\` | \`{"grid":int,"coords":[[x,y,z],...]}\`. Empty = off. |
| \`sdedit_t0\` | float | \`0.7\` | SDEdit start time (0,1]. Lower = closer to draft (flatter); higher = more model detail (rugged). |
| \`aspect_ratio\` | str | \`""\` | \`"x:y:z"\` (x=side, y=up, z=front). Rescales the SS voxel group before SLat. |
| \`aspect_mode\` | str | \`"nearest"\` | Rescale kernel: \`nearest\` (hole-safe) or \`trilinear\`. |
| \`metallic_cap\` | float | \`1.0\` | Clamp predicted metalness. \`0.0\` = matte (kills glossy/glass); \`1.0\` = off. |

Coordinates are **centered on the origin** (range \`[-grid/2, grid/2)\`), axes
\`x\`=side, \`y\`=up, \`z\`=front.

---

**Suggested reading:** [Voxel Build Protocol](#structural/build-protocol) · [Tuning & Caveats](#structural/tuning) · [TRELLIS.2 parameters](#services/trellis2)

`;

const scProtocolContent = `# Voxel Build Protocol (LLM ⇄ constructor)

A draft is a JSON \`VoxelProgram\`: a grid size plus an ordered list of ops. The same
primitives back the Sparse Voxel sub-tab, so a human and an LLM produce identical
results.

\`\`\`jsonc
{ "grid": 64,            // power of two, <= 64
  "ops": [ /* ... */ ] }
\`\`\`

Coordinates are centered (origin = grid center), axes \`x\`=side, \`y\`=up, \`z\`=front.
Planes \`xy|xz|yz\` are named by their two in-plane axes; the third is the \`layer\`.

| op | params | meaning |
|---|---|---|
| \`box\` | \`min:[x,y,z]\`, \`size:[lx,ly,lz]\`, \`mode?\` | filled/\`shell\` 3D cuboid |
| \`ellipsoid\` | \`center:[x,y,z]\`, \`radii:[rx,ry,rz]\`, \`mode?\` | filled/\`shell\` 3D ellipsoid |
| \`rect2d\` | \`plane\`,\`layer\`,\`min:[a,b]\`,\`size:[la,lb]\`,\`mode?\` | 2D rectangle on a plane |
| \`ellipse2d\` | \`plane\`,\`layer\`,\`center:[a,b]\`,\`radii:[ra,rb]\`,\`mode?\` | 2D ellipse on a plane |
| \`voxels\` | \`cells:[[x,y,z],...]\` | explicit cells |
| \`extend\` | \`axis\`,\`count\`,\`step?\`,\`name?\` | repeat current occupancy along an axis |
| \`repeat\` | \`name\` | replay a named direction |
| \`clear\` | — | empty the grid |

\`mode\` is \`"solid"\` (default) or \`"shell"\`. \`extend\` validates the occupancy is a
clean prism along the axis (so it tiles seamlessly) and can store the direction
under \`name\` for later \`repeat\`.

**Cylinder** — an 8×8 ellipse swept 32 along z, centered:

\`\`\`json
{ "grid": 64, "ops": [
  { "op": "ellipse2d", "plane": "xy", "layer": -16, "center": [0,0], "radii": [8,8] },
  { "op": "extend", "axis": "z", "count": 32, "name": "barrel" }
] }
\`\`\`

**Tiled wall** — a tile repeated in x then y:

\`\`\`json
{ "grid": 64, "ops": [
  { "op": "box", "min": [-24,-24,0], "size": [6,6,1] },
  { "op": "extend", "axis": "x", "count": 8, "name": "row" },
  { "op": "extend", "axis": "y", "count": 8 }
] }
\`\`\`

---

**Suggested reading:** [Structural Conditioning](#structural/overview) · [Tuning & Caveats](#structural/tuning) · [TRELLIS.2 parameters](#services/trellis2)

`;

const scTuningContent = `# Tuning & Caveats

## \`sdedit_t0\` — the master knob

\`t0\` trades draft-adherence against model freedom, and you can't max both:

- **Low (≈0.4–0.5)** — output matches the drafted shape/proportions closely, but
  the model adds *less* of its own surface detail (flatter).
- **High (≈0.7–0.8)** — more rugged surface detail, but the geometry drifts from
  the draft (proportions and even silhouette can wander).

Default is \`0.7\`. For "match my wall exactly" use low; for "use my wall as a loose
hint, give me detail" use high.

## Surface detail lives in the texture/normal map

The sparse structure is coarse (32³), so brick-scale relief was never going to be
*geometry* — it's baked into the normal/albedo maps by the texture stage. A flat
wall is also out-of-distribution for an object generator, so expect it to lean on
the normal map. Don't expect displaced brick geometry from a flat draft.

## Glossy / glass look → \`metallic_cap\`

The texture stage predicts per-voxel metalness from the image; specular cues can
make a wall come out shiny/glass-like. \`metallic_cap\` clamps that prediction to a
ceiling **after** the model decides — \`0.0\` forces fully matte (kills the gloss),
\`1.0\` leaves the model's choice. It's a deterministic post-decode clamp, so color,
roughness and geometry are untouched.

## Don't over-force proportions

Resampling the model's output to a very different aspect ratio tears the mesh
(gaps on stretch, eroded thin shells). Prefer enforcing proportions via a **lower
\`t0\`** (the draft seed), not an aggressive \`aspect_ratio\`. \`aspect_mode\` defaults to
\`nearest\` precisely because it's hole-safe.

## Postprocess / unwrap speed

Complex or disconnected meshes (walls, tiles) make UV unwrap (xatlas) slow because
it makes ≥1 chart per component. If a run is slow in \`postprocess\`, lower
\`decimation_target\` (fewer faces → fewer charts). Watch the \`[to_glb.timing]\`
breakdown in logs.

---

**Suggested reading:** [Structural Conditioning](#structural/overview) · [Voxel Build Protocol](#structural/build-protocol) · [TRELLIS.2 parameters](#services/trellis2)

`;

// ---------------------------------------------------------------------
// Tree. Order matters — flattenPages walks left-to-right and the first
// page becomes the default selection.
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// Multi-service + Hunyuan-Omni docs. The platform now serves two
// generation backends behind one router; these pages document the
// shared contract and the Hunyuan-specific service.
// ---------------------------------------------------------------------

const multiServiceContent = `# Capacity & routing

The Starshot Assets API serves **two** 3D-generation backends behind one
service-agnostic router:

| | TRELLIS.2 | Hunyuan-Omni |
|---|-----------|--------------|
| Conditioning | image (+ optional sparse-voxel SDEdit prior) | image **+ a control signal** (point / voxel / bbox / pose) |
| Endpoint | \`POST /generate?model=trellis2\` | \`POST /generate?model=hunyuan-omni\` |
| Modal app (prod) | \`starshot-assets\` (one app: router + both workers) | \`starshot-assets\` (same app) |
| Queue / limits | \`trellis2-*\`, \`TRELLIS2_MAX_*\` | \`hunyuan-*\`, \`HUNYUAN_MAX_*\` |

## Picking a service

The canonical, query-style selector is \`?model=<slug>\`:

\`\`\`
POST /generate?model=trellis2
POST /generate?model=hunyuan-omni
\`\`\`

The path style \`POST /generate/<slug>\` still resolves but is **deprecated** —
prefer the query form. Omitting the selector falls back to the default service
(\`trellis2\`).

## Shared vs per-service

**Shared** across both services: the job-id format and the entire polling
lifecycle — \`GET /jobs/{id}\`, \`GET /jobs/{id}/result\`,
\`POST /jobs/{id}/cancel\`. You spawn against either service and poll the *same*
endpoints; the router tracks which service owns each job.

**Per-service** (atomic): the queue, analytics, and archive views accept a
\`?service=\` filter, and each service has its **own GPU pool**:

\`\`\`
GET /queue?service=hunyuan-omni       # this service's queue only
GET /analytics?service=trellis2       # this service's stats only
GET /queue                             # aggregate, with a by_service breakdown
\`\`\`

## Capacity model

Two independent queues, each capped individually, **plus a global cap on their
sum** — for both queue length and concurrent GPU containers:

- Per-service: \`TRELLIS2_MAX_GPU\` / \`TRELLIS2_MAX_QUEUE\`,
  \`HUNYUAN_MAX_GPU\` / \`HUNYUAN_MAX_QUEUE\`.
- Global (shared ceiling on the sum): \`STARSHOT_GLOBAL_MAX_GPU\`,
  \`STARSHOT_GLOBAL_MAX_QUEUE\`.

A request is admitted only if both its service cap **and** the global cap have
headroom. See \`modal-scripts/_shared/limits.py\`.

---

**Suggested reading:** [Rate Limits](#rate-limits) · [GET /queue](#endpoints/queue) · [Architecture & Routing](#reference/architecture)

`;

const hunyuanIntroContent = `# Hunyuan3D-Omni

Hunyuan3D-Omni generates a textured GLB from an **image plus one control
signal** — a point cloud, voxels, a bounding box, or a pose. The control signal
steers the shape where a plain image-to-3D model would be ambiguous.

In production it runs on the shared \`starshot-assets\` app (router + both GPU
workers); the router spawns it **in-process** for \`?model=hunyuan-omni\` requests.

## Endpoint

\`\`\`
POST /generate?model=hunyuan-omni      multipart: image + control params
\`\`\`

Returns a \`job_id\` immediately (same spawn-and-poll contract as Trellis2). Poll
\`GET /jobs/{id}\` then \`GET /jobs/{id}/result\` for the GLB.

## Two-stage pipeline

1. **Shape** — \`Hunyuan3DOmniSiTFlowMatchingPipeline\` (SiT flow-matching)
   consumes the image + control signal and runs marching cubes at
   \`octree_resolution\`.
2. **Paint (PBR)** — the Hunyuan3D-2.1 texture stack bakes base color, normal,
   and metallic-roughness maps. Skipped when \`texture=false\`.

Background removal uses **BiRefNet** (the same matte model Trellis2 uses) before
the shape stage, unless the image already has a usable alpha channel.

Optionally, **front-view alignment** (\`align=true\`) inserts a Qwen-VLM step
between shape and paint that rotates the true front to +Z to defeat the Janus
problem — see [Parameters](#hunyuan/parameters).

---

**Suggested reading:** [Parameters](#hunyuan/parameters) · [Control payloads](#hunyuan/control) · [Architecture](#hunyuan/architecture) · [POST /generate](#endpoints/generate)

`;

const hunyuanParamsContent = `# Hunyuan-Omni — Parameters

Sent as multipart form fields alongside the \`image\` file on
\`POST /generate?model=hunyuan-omni\`.

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| \`image\` | file | — | input subject (alpha matte respected, else BiRefNet) |
| \`control_type\` | str | \`point\` | one of \`point\`, \`voxel\`, \`bbox\`, \`pose\` |
| \`control\` | JSON | \`{}\` | control payload (authored client-side; see below) |
| \`steps\` | int | 30 | flow-matching steps |
| \`octree_resolution\` | int | 512 | marching-cubes grid (256 / 384 / 512) |
| \`guidance_scale\` | float | 6.7 | CFG scale |
| \`seed\` | int | 1234 | RNG seed |
| \`texture\` | bool | true | run the PBR paint stack after shape |
| \`remove_background\` | bool | true | BiRefNet matte if no usable alpha |
| \`use_ema\` | bool | false | use EMA shape weights |
| \`text_prompt\` | str | "" | optional caption nudge |
| \`align\` | bool | false | front-view alignment (anti-Janus, see below) |
| \`num_orientations\` | int | 4 | turntable candidate views the VLM scores when \`align\` |
| \`object_name\` | str | "" | hint fed to the VLM prompt (e.g. \`wooden chair\`); falls back to \`text_prompt\` |

Defaults live in \`modal-scripts/hunyuan_modal/config.py\` and are editable
from the orchestrator's Config tab (\`hunyuan-prod\` scope) — nothing is
hardcoded at the call site.

## Front-view alignment (\`align\`)

Optional, **off by default**. When enabled, the worker renders a short turntable
of the shape, asks a **Qwen** VLM (via OpenRouter) which view matches the
reference image — the canonical front — and rotates that face to **+Z before
paint**. Its job is to defeat the **Janus problem**: image-to-3D models often
hallucinate a second, busier-looking face on the *back* of the object, and
without anchoring to the reference a pipeline can ship the model facing backward.
Needs the \`openrouter-secret\`; hard-fails (500) if requested but the render/VLM
is unavailable. This is the same shared aligner the Structural Texture service
uses.

---

**Suggested reading:** [Control payloads](#hunyuan/control) · [Overview](#hunyuan/intro) · [Architecture](#hunyuan/architecture)

`;

const hunyuanControlContent = `# Hunyuan-Omni — Control payloads

The control signal is a small JSON object authored client-side. Its shape
depends on \`control_type\` (mirrors the Trellis voxel editor's
\`{ grid, coords }\` convention):

- **point** — \`{ "points": [[x,y,z], …] }\`
- **voxel** — \`{ "grid": 64, "coords": [[x,y,z], …] }\`
- **bbox**  — \`{ "bbox": [sx, sy, sz] }\` — box **size** along each axis,
  origin-centered (the model's native format). A legacy 6-value
  \`[x_min, y_min, z_min, x_max, y_max, z_max]\` is also accepted and converted to
  size (\`abs(max − min)\`) — only the extents are injected. Becomes a \`[1,1,3]\`
  tensor.
- **pose**  — \`{ "bones": [...] }\`

In the **Try It → Hunyuan** tab, the **Control Signal** sub-tab gives you a JSON
editor seeded with a template per type, with validation before you generate.

## Example (point control)

\`\`\`bash
curl -X 'POST' \\
  'https://…modal.run/generate?model=hunyuan-omni' \\
  -F 'image=@subject.png' \\
  -F 'control_type=point' \\
  -F 'control={"points":[[0,0,0],[0.2,0.1,-0.1]]}' \\
  -F 'steps=30' -F 'octree_resolution=512'
# -> { "job_id": "..." }  then poll /jobs/{id}
\`\`\`

## Direct (SDK / local) use

\`\`\`bash
# Warm all weights once (CPU, no GPU billing):
modal run modal-scripts/modal_app.py::prefetch
\`\`\`

---

**Suggested reading:** [Parameters](#hunyuan/parameters) · [Overview](#hunyuan/intro) · [POST /generate](#endpoints/generate)

`;

const hunyuanArchContent = `# Hunyuan-Omni — Architecture

Two stages run in one A100-80GB container:

1. **Shape** — \`Hunyuan3DOmniSiTFlowMatchingPipeline\` (SiT flow-matching):
   image + control → mesh via marching cubes at \`octree_resolution\`. Code:
   \`core/hunyuan-omni-src/hy3dshape/\`.
2. **Paint (PBR)** — the Hunyuan3D-2.1 texture stack (\`hy3dpaint\`) bakes base
   color / normal / metallic-roughness. Code:
   \`core/hunyuan-omni-src/Hunyuan3D-2.1/\`. Two CUDA extensions compile at
   image-build time and lazy-load on the first request.

## Modal layout

The package mirrors \`trellis_modal/\` — split by concern, not one monolith:

- \`hunyuan_modal/app.py\` — imports the shared \`starshot-assets\` app (from
  \`trellis_modal.app\`) + defines the Hunyuan container images. Mounts the
  vendored core from \`core/hunyuan-omni-src/\`.
- \`hunyuan_modal/config.py\` — constants + inference defaults.
- \`hunyuan_modal/state.py\` — weight / HF / paint-ckpt / output Volumes.
- \`hunyuan_modal/worker.py\` — the \`HunyuanOmni\` GPU worker (+ \`BiRefNet\`,
  \`build_control\`, and the \`prefetch\` entrypoint), decorated on the shared app.
- \`hunyuan_modal/service.py\` — registers the \`hunyuan-omni\` \`ServiceSpec\` +
  the in-process spawn adapter.

The single platform entry point is **\`modal-scripts/modal_app.py\`**: it imports
the router + both GPU workers + the Hunyuan scripts, all on the one
\`starshot-assets\` app, so the whole platform deploys from one command:

\`\`\`
modal deploy modal-scripts/modal_app.py        # App: starshot-assets (router + both workers)
\`\`\`

## Shared infrastructure

\`HunyuanOmni\` inherits \`_shared.gpu_worker.BaseGPUWorker\` (load-error stash,
phase timing, health, per-service \`ServiceState\`) — exactly like Trellis2's
worker. In production it's decorated onto the **same** \`starshot-assets\` app as
the router, which spawns it in-process (\`HunyuanOmni().inference.spawn.aio(...)\`)
— no cross-app lookup. (The Hunyuan *experimental* server is a separate app.)

## Cold-start

- **Weights Volume** (\`hy3domni-weights\` / \`hy3domni-hf\`) — downloaded once via
  \`prefetch\`, reused across cold starts and redeploys.
- **Memory snapshot** — the pipeline loads to CPU under
  \`@modal.enter(snap=True)\` so restore only pays the CPU→GPU transfer.

---

**Suggested reading:** [Overview](#hunyuan/intro) · [Parameters](#hunyuan/parameters) · [Architecture & Routing](#reference/architecture)

`;

const structuralIntroContent = `# Structural Texture

Structural Texture turns a single **texture image** (a road surface, a brick
course, a tiled façade) into a tileable, double-sided, front-aligned GLB panel.
It is a **subclass of Hunyuan3D-Omni** — it reuses the exact same shape + PBR
paint stack — wrapped with a geometry post-process that builds a thin slab,
finds the true front face with a VLM, symmetrizes through the thickness, and
tiles the painted panel to a requested length.

In production it runs on the shared \`starshot-assets\` app; the router spawns it
**in-process** for \`?model=structured-texture\` requests.

## Endpoint

\`\`\`
POST /generate?model=structured-texture    multipart: image + params below
\`\`\`

Returns a \`job_id\` immediately (same spawn-and-poll contract as Trellis2 /
Hunyuan). Poll \`GET /jobs/{id}\` then \`GET /jobs/{id}/result\` for the GLB.

## Pipeline (mesh → paint → align → tile + symmetric)

1. **Shape** — builds a thickness bounding box from the image aspect and runs the
   inherited Hunyuan \`bbox\`-control shape stage.
2. **Paint (PBR)** — the inherited Hunyuan3D-2.1 paint stack textures the **single
   panel** first, so the reference image matches before any geometry ops (trimesh
   carries the full PBR — base color, normal, metallic-roughness — and UVs through
   every slice / transform / concatenate).
3. **Front-view align** — renders \`num_orientations\` turntable candidates and asks
   a **Qwen** VLM which is the true front (anti-Janus), rotating the good side to
   **+Z**. This is **essential**: the generated slab is not inherently symmetric
   and may be rotated.
4. **Symmetrize** — reflects the +Z half through the thickness axis so both faces
   match.
5. **Tile** — stacks the painted panel along **X** to the requested \`length\`
   (fractional tiles preserve the far side face and stretch-to-fill to avoid gaps),
   then applies the final \`orientation\`.

Failures are **hard 500s** — there is no silent best-effort degradation.

---

**Suggested reading:** [Parameters](#structured-texture/parameters) · [Hunyuan Overview](#hunyuan/intro) · [POST /generate](#endpoints/generate)

`;

const structuralParamsContent = `# Structural Texture — Parameters

Sent as multipart form fields alongside the \`image\` file on
\`POST /generate?model=structured-texture\`.

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| \`image\` | file | — | texture face (alpha matte respected, else BiRefNet) |
| \`thickness\` | float | 0.1 | slab depth along the thickness axis (Z) |
| \`orientation\` | str | \`vertical\` | \`vertical\` (aligned +Z frame) or \`horizontal\` (laid flat, R_x(−90°)) |
| \`height\` | float | 0.0 | height override; 0 derives height from image aspect |
| \`length\` | float | 1.0 | tiles the painted panel along X to this length |
| \`steps\` | int | 30 | inherited HY flow-matching steps |
| \`octree_resolution\` | int | 512 | inherited HY marching-cubes grid (256 / 384 / 512) |
| \`guidance_scale\` | float | 6.7 | inherited HY CFG scale |
| \`seed\` | int | 1234 | RNG seed |
| \`num_orientations\` | int | 4 | turntable candidate views for the VLM front-view pick |
| \`texture\` | bool | true | run the inherited PBR paint stack on the panel |
| \`align\` | bool | true | VLM front-view alignment (keep on — the slab is not symmetric) |
| \`remove_background\` | bool | true | BiRefNet matte if no usable alpha |
| \`use_ema\` | bool | false | use EMA shape weights |

Defaults live in \`modal-scripts/structured_texture_modal/config.py\`. The worker
\`GPUWorker_Structural\` inherits \`StructuralTextureWorker\`
(\`FrontViewAlignmentMixin\` + \`HunyuanOmniWorker\`); the experimental
\`structured-texture-experimental\` app reuses the exact same prod worker.

## Example

\`\`\`bash
curl -X 'POST' \\
  'https://…modal.run/generate?model=structured-texture' \\
  -F 'image=@road.png' \\
  -F 'thickness=0.1' -F 'orientation=horizontal' \\
  -F 'length=4.0' -F 'steps=30' -F 'octree_resolution=512'
# -> { "job_id": "..." }  then poll /jobs/{id}
\`\`\`

---

**Suggested reading:** [Overview](#structured-texture/intro) · [Hunyuan Parameters](#hunyuan/parameters) · [POST /generate](#endpoints/generate)

`;

export const DOC_TREE: DocNode[] = [
  {
    type: "folder",
    title: "Getting Started",
    children: [
      {
        type: "page",
        page: { slug: "intro", title: "Introduction", content: introContent },
      },
      {
        type: "page",
        page: { slug: "auth", title: "Authentication", content: authContent },
      },
      {
        type: "page",
        page: {
          slug: "rate-limits",
          title: "Rate Limits",
          content: rateLimitsContent,
        },
      },
    ],
  },
  {
    type: "folder",
    title: "Core API",
    children: [
      {
        type: "page",
        page: {
          slug: "endpoints/generate",
          title: "POST /generate",
          content: generateContent,
        },
      },
      {
        type: "folder",
        title: "Jobs",
        children: [
          {
            type: "page",
            page: {
              slug: "endpoints/jobs",
              title: "GET /jobs",
              content: jobsListContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/jobs-id",
              title: "GET /jobs/{id}",
              content: jobByIdContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/jobs-id-result",
              title: "GET /jobs/{id}/result",
              content: jobResultContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/jobs-cancel",
              title: "POST /jobs/{id}/cancel",
              content: jobCancelContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/jobs-priority",
              title: "POST /jobs/{id}/priority",
              content: jobPriorityContent,
            },
          },
        ],
      },
      {
        type: "folder",
        title: "Queue",
        children: [
          {
            type: "page",
            page: {
              slug: "endpoints/queue",
              title: "GET /queue",
              content: queueContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/queue-pause",
              title: "POST /queue/pause",
              content: queuePauseContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/queue-resume",
              title: "POST /queue/resume",
              content: queueResumeContent,
            },
          },
        ],
      },
      {
        type: "folder",
        title: "Analytics & Archive",
        children: [
          {
            type: "page",
            page: {
              slug: "endpoints/analytics",
              title: "GET /analytics",
              content: analyticsContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/archive",
              title: "GET /archive",
              content: archiveContent,
            },
          },
        ],
      },
      {
        type: "folder",
        title: "Health & Services",
        children: [
          {
            type: "page",
            page: {
              slug: "endpoints/health",
              title: "GET /health",
              content: healthContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/worker-status",
              title: "GET /worker_status",
              content: workerStatusContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "endpoints/services",
              title: "GET /services",
              content: servicesEndpointContent,
            },
          },
        ],
      },
    ],
  },
  {
    type: "folder",
    title: "Services",
    children: [
      {
        type: "folder",
        title: "TRELLIS.2",
        children: [
          {
            type: "page",
            page: {
              slug: "services/trellis2",
              title: "Overview & parameters",
              content: trellis2OverviewContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "structural/overview",
              title: "Structural Conditioning",
              content: scOverviewContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "structural/build-protocol",
              title: "Voxel Build Protocol",
              content: scProtocolContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "structural/tuning",
              title: "Tuning & Caveats",
              content: scTuningContent,
            },
          },
        ],
      },
      {
        type: "folder",
        title: "Hunyuan-Omni",
        children: [
          {
            type: "page",
            page: {
              slug: "hunyuan/intro",
              title: "Overview",
              content: hunyuanIntroContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "hunyuan/parameters",
              title: "Parameters",
              content: hunyuanParamsContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "hunyuan/control",
              title: "Control payloads",
              content: hunyuanControlContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "hunyuan/architecture",
              title: "Architecture",
              content: hunyuanArchContent,
            },
          },
        ],
      },
      {
        type: "folder",
        title: "Structural Texture",
        children: [
          {
            type: "page",
            page: {
              slug: "structured-texture/intro",
              title: "Overview",
              content: structuralIntroContent,
            },
          },
          {
            type: "page",
            page: {
              slug: "structured-texture/parameters",
              title: "Parameters",
              content: structuralParamsContent,
            },
          },
        ],
      },
    ],
  },
  {
    type: "folder",
    title: "Reference",
    children: [
      {
        type: "page",
        page: {
          slug: "reference/architecture",
          title: "Architecture & Routing",
          content: architectureContent,
        },
      },
      {
        type: "page",
        page: {
          slug: "reference/capacity",
          title: "Capacity & routing",
          content: multiServiceContent,
        },
      },
      {
        type: "page",
        page: {
          slug: "reference/states",
          title: "Job States",
          content: statesContent,
        },
      },
      {
        type: "page",
        page: {
          slug: "reference/errors",
          title: "Errors",
          content: errorsContent,
        },
      },
      {
        type: "page",
        page: {
          slug: "reference/phase-timings",
          title: "Phase Timings",
          content: phaseTimingsContent,
        },
      },
    ],
  },
];
