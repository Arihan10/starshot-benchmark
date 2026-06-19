"""Read-model types for a cell's renderable scene + run status.

This is the in-memory projection a cell exposes for rendering and for the
dashboard. It is deliberately distinct from two other things:

  * the pipeline's `Node` graph (`app.core.types`), which carries the rich
    authoring detail (placement prose, relationships, parent kinds, image
    prompts) the divider/generation steps reason over; and
  * the durable event log, which stays the source of truth for resume,
    caching, and the executed-prompt ground truth.

`SceneState` answers only "what is the scene right now, and what is the run's
status" — never HOW it was produced. Generation diagnostics (why a mesh
failed, retries, token spend, the prompt bytes) and the cell's control-plane
state (stepped-mode gate, simulation branches, current pipeline phase) are
intentionally kept out: they have different lifetimes and are served
separately. This object is DERIVED — rebuilt by folding a cell's event log and
updated incrementally as new events land — so it is never authoritative.

Taxonomy note: there is no `kind` field. Zones vs concrete leaves is the
bucket split (`Scene.zones` vs `Scene.objects`); within leaves, the
decomposition pass (`SceneObject.emitted_by`) is the sole classifier and color
key — a "frame" is simply the `ENCAPSULATING` pass's output, not a distinct
kind.

First module of the new `app.types` package.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.core.types import Orientation, ProxyShape, Vec3Tuple


class CellStatus(StrEnum):
    """A cell's run status, mirroring the event-log status derivation:
    `DONE` is terminal and sticky, `ERROR`/`PAUSED` are recognized as the
    latest state, a non-empty log with no terminal marker reads `RUNNING`,
    and an empty log is `IDLE`."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"


class ObjectPhase(StrEnum):
    """Where a concrete leaf is in mesh generation. The viewer shows the
    proxy/bbox wireframe until `DONE` (when `glb` is set); `ERROR` lets the UI
    surface a retry affordance and stop waiting on a spinner. The failure
    MESSAGE is not carried here — it stays on the `mesh.error` event (a
    generation diagnostic, read from the log on demand, not scene content)."""

    PENDING = "pending"        # placed (bbox known), mesh not yet submitted
    GENERATING = "generating"  # mesh submitted, in flight
    DONE = "done"              # mesh landed; `glb` is set
    ERROR = "error"            # mesh generation failed (see the log for why)


class EmittedBy(StrEnum):
    """The decomposition pass that produced a leaf — the sole taxonomy and
    color key for objects/frames (the per-pass "imperial" coloring). A frame
    is not a separate kind: it is simply the `ENCAPSULATING` pass's output.

    Values are the pipeline step/template names, so this maps straight from
    the `template` of the `cache.llm` decompose call that named the node.
    `None` on a leaf means the emitting pass is unknown (e.g. the one-shot
    track, which has no decomposition steps), and the renderer falls back to a
    default color."""

    ANCHOR = "anchor_decompose"                  # the defining objects of an atomic zone
    NEXT = "next_object"                         # objects added by the anchor completion loop
    ENCAPSULATING = "encapsulating_decompose"    # the zone's shell / ground (rendered as "frames")
    NEGATIVE_SPACE = "negative_space_decompose"  # interstitial ambient fill


class SceneZone(BaseModel):
    """An abstract region: an axis-aligned box with no mesh, carrying its
    LLM-authored `plan` (the zone's intent). `parent_id` is the enclosing
    region; geometry is the world-frame `origin` (minimum corner) plus
    `dimensions`."""

    id: str
    parent_id: str | None = None
    prompt: str = ""
    plan: str | None = None
    origin: Vec3Tuple
    dimensions: Vec3Tuple
    proxy_shape: ProxyShape | None = None


class SceneObject(BaseModel):
    """A concrete leaf — an object, or a frame (`emitted_by == ENCAPSULATING`).
    Geometry is the world-frame `origin` (minimum corner) + `dimensions` +
    discrete `orientation` (yaw). `emitted_by` is the taxonomy/color key.
    `glb` is the artifact URL, set once `phase == DONE`."""

    id: str
    parent_id: str | None = None
    emitted_by: EmittedBy | None = None
    prompt: str = ""
    origin: Vec3Tuple
    dimensions: Vec3Tuple
    orientation: Orientation = 0
    proxy_shape: ProxyShape | None = None
    phase: ObjectPhase = ObjectPhase.PENDING
    glb: str | None = None


class Scene(BaseModel):
    """The renderable scene graph: abstract regions and concrete leaves, split
    by bucket — zones carry plans (and have children), objects carry meshes."""

    zones: list[SceneZone] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)


class SceneState(BaseModel):
    """A cell's full read model: run status + the scene graph, plus the
    watermark a client needs to attach the live event tail with no gap.

    Derived from the cell's event log (folded once, then kept current as
    events land). NOT authoritative: resume and caching read the log, not
    this. Control-plane state (gate, branches, token spend, current pipeline
    phase) is deliberately excluded and served separately."""

    status: CellStatus = CellStatus.IDLE
    model: str | None = None
    prompt: str | None = None
    # Highest event-log index folded into this state; the client subscribes to
    # the live SSE tail at `?since=last_index` for a gapless handoff. -1 means
    # nothing has been folded yet.
    last_index: int = -1
    scene: Scene = Field(default_factory=Scene)
