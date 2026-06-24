"""Structured-output schemas for the pipeline's LLM calls.

These are pipeline code, not prompt text: prompt versions (the txt templates
under versions/) control wording, but every version returns JSON in the shapes
below. Moving a field is a code change, not a prompt-version change.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.types import (
    BoundingBox,
    Orientation,
    ParentRelationshipKind,
    ProxyShape,
    Relationship,
)


class RootZonePlanOutput(BaseModel):
    # The root planning step ("OVERALL SCENE PLAN") keeps `plan` as its wire
    # field — the scene-level plan, distinct from a nested region's
    # `description`. Same shape as ZonePlanOutput, but no alias: the wire name
    # equals the `plan` attribute the shared divider/committed code reads.
    plan: str
    is_atomic: bool


class ZonePlanOutput(BaseModel):
    # The non-root region step ("REGION DESCRIPTION"). The LLM-facing field is
    # aliased so the schema/wire name matches the prompt text (`description`),
    # while the Python attribute stays `plan` — the name the shared
    # divider/committed code and the `divider.zone_plan` event use.
    # `populate_by_name` keeps attribute-name construction (committed-event
    # replay) working.
    model_config = ConfigDict(populate_by_name=True)

    plan: str = Field(alias="description")
    is_atomic: bool


class OverallBboxOutput(BaseModel):
    bbox: BoundingBox

    @field_validator("bbox", mode="after")
    @classmethod
    def _canonicalize(cls, v: BoundingBox) -> BoundingBox:
        # Force the root canvas to min-corner origin + non-negative dims so
        # the parent frame every downstream placement step reads is never
        # self-contradictory (a negative dim makes origin != min corner).
        return v.canonical()


class ChildNodeSpec(BaseModel):
    """A single anchored child node (an object / frame) emitted by a
    decomposition LLM call. Objects can physically rest on or attach to other
    nodes, so they carry an explicit structural `parent`. Abstract subregions
    do not — those use `SubregionSpec` and are always contained in their zone.

    Fields:
      * `parent` — the structural anchor id: the supporter (object or
        frame) this node physically rests on, or the containing region
        when there's no physical supporter (floating objects, frames
        themselves, subregions inside a parent region).
      * `parent_relationship_kind` — how this node relates to its `parent`. MUST be
        one of ON / ATTACHED / IN:
          - ON: rests on parent's outward surface (most often top face).
          - ATTACHED: flush against any face of the parent (wall mount,
            ceiling fixture, magnet, hanger).
          - IN: contained inside the parent's volume or footprint
            (subregion inside a region, fish inside a tank, cloud inside a
            sky region, embedded particle).
        BESIDE / ABOVE / BELOW are reserved for sibling/peer hints in
        `relationships` and are NOT valid here.
      * `placement` — prose describing where this node sits, with
        precise positioning (centered, edge-aligned, two-thirds along,
        etc.). The bbox-resolution step uses this verbatim to choose
        coordinates.
      * `relationships` — optional list of spatial relationships to
        other already-placed nodes that assist the downstream spatial
        resolver. Each entry is a Relationship with `target` (the peer's
        id) and `kind` (ON, BESIDE, ABOVE, BELOW, ATTACHED, IN) —
        categorical, no anchor point. Do NOT repeat the parent here.
        Empty list is fine when the placement only references the parent.
    """

    # Two LLM-facing field names are aliased so the schema/wire names match the
    # prompt text, while the Python attributes stay the names the shared
    # divider/generation/topology + Node use: `relationships` -> attr
    # `referenced_ids`, `parent_relationship_kind` -> attr `parent_kind`.
    # `serialize_by_alias` makes every model_dump (the observability/committed
    # logs) emit the alias names, so logs match exactly what the model is shown
    # and emits; `populate_by_name` still accepts the attribute names so older
    # logs dumped before this round-trip back in.
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    prompt: str
    parent: str
    parent_kind: ParentRelationshipKind = Field(alias="parent_relationship_kind")
    placement: str
    referenced_ids: list[Relationship] = Field(
        default_factory=list, alias="relationships"
    )
    proxy_shape: ProxyShape | None = None
    # Semantic orientation the decompose steps author (e.g. "facing the
    # conversation pit") as free text, NOT a yaw angle — object_bbox_batch
    # resolves it to the discrete yaw from this plus the surrounding layout.
    orientation: str = ""

    @field_validator("proxy_shape", mode="before")
    @classmethod
    def _box_means_none(cls, v: object) -> object:
        # BOX is offered as an explicit, selectable proxy_shape value (the
        # rectilinear default), but the rest of the pipeline encodes "box" as
        # None — the AABB is its own proxy. Canonicalize the literal "BOX" (and
        # any null/omission) to None here so a realized Node only ever carries
        # None / SPHERE / CAPSULE / HEMISPHERE.
        if isinstance(v, str) and v.upper() == "BOX":
            return None
        return v


class SubregionSpec(BaseModel):
    """A single subregion emitted by zone decomposition.

    Unlike an object, a subregion is an ABSTRACT spatial subdivision, not
    grounded geometry — it is always contained in (IN) the zone being
    decomposed. So it carries no structural `parent`/`parent_relationship_kind`:
    the pipeline fixes its parent to that zone, and its bbox is resolved in the
    zone's local frame. Spatial relationships to sibling subregions are
    expressed through `relationships` (advisory peer hints), never through
    parenthood — an object can physically rest on another object (apple on a
    desk), but one region does not "rest on" another.

    Fields:
      * `placement` — prose describing where this subregion sits within its
        parent zone, used verbatim by the bbox-resolution step.
      * `relationships` — optional peer hints (`target` + `kind`) to other
        already-placed nodes that assist the spatial resolver.
    """

    # Same alias/serialization contract as ChildNodeSpec so logs round-trip.
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: str
    prompt: str
    placement: str
    referenced_ids: list[Relationship] = Field(
        default_factory=list, alias="relationships"
    )
    proxy_shape: ProxyShape | None = None

    @field_validator("proxy_shape", mode="before")
    @classmethod
    def _box_means_none(cls, v: object) -> object:
        # Mirror ChildNodeSpec: "BOX" (and null/omission) collapse to None.
        if isinstance(v, str) and v.upper() == "BOX":
            return None
        return v


class ZoneDecomposeOutput(BaseModel):
    subregions: list[SubregionSpec]


class BboxAssignment(BaseModel):
    id: str
    bbox: BoundingBox

    @field_validator("bbox", mode="after")
    @classmethod
    def _canonicalize(cls, v: BoundingBox) -> BoundingBox:
        # Normalize each resolved (parent-local) box to min-corner origin +
        # non-negative dims. A negative dim here would extend the child OUT of
        # the min-corner-anchored local frame (behind the parent) once
        # translated to world; canonicalizing keeps every placed box in-frame.
        return v.canonical()


class BboxBatchOutput(BaseModel):
    assignments: list[BboxAssignment] = Field(default_factory=list)


class ObjectBboxAssignment(BboxAssignment):
    # object_bbox_batch RESOLVES the object's discrete yaw from its semantic
    # `orientation` text (carried on the spec) plus the surrounding layout, so
    # its assignment carries the numeric `orientation` on top of the bbox.
    # Subregion placement (child_bbox_batch) stays bbox-only.
    orientation: Orientation = 0


class ObjectBboxBatchOutput(BaseModel):
    assignments: list[ObjectBboxAssignment] = Field(default_factory=list)


class ObjectSpec(ChildNodeSpec):
    """A single object in a region. Identical shape to ChildNodeSpec.
    The structural parent (`parent` field) may be the enclosing region,
    a frame, an earlier-placed peer, or another object listed in the
    same decomp call. Secondary relationships (`relationships`)
    capture additional spatial connections — sibling alignment, the
    wall a painting hangs against, etc."""


class ObjectDecompOutput(BaseModel):
    # NEWLY EMITTED object specs that become Nodes downstream. The anchor pass
    # uses this bare shape: it always produces the region's defining objects, so
    # it carries no gate. Every pass that can instead decide a region needs no
    # objects at all subclasses this and adds the `objects_required` gate below.
    objects: list[ObjectSpec] = Field(default_factory=list)


class GatedObjectDecompOutput(ObjectDecompOutput):
    # Shared by every pass that may decide a region needs no objects at all: the
    # encapsulating shell, the negative-space fill, and the anchor completion
    # loop (`next_object`). When the model sets this False the region needs
    # nothing from this pass and `objects` is ignored even if non-empty — for the
    # completion loop that False is also how it signals "done" (no more objects),
    # so it needs no separate `done` field.
    objects_required: bool = True


class ImagePromptOutput(BaseModel):
    prompt: str
