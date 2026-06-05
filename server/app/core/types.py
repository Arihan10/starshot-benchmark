"""
Domain types for the pipeline.

Coordinates: Y-up, right-handed, meters.
Canonical front view: +X = right, +Y = up, +Z = toward the viewer.
All bounding boxes are expressed under this convention. Spatial
relationships between nodes are carried as prose `placement`
descriptions on each Node (resolved by the bbox-resolution LLM step).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

Vec3Tuple = tuple[float, float, float]


def _coerce_int(v: object) -> object:
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            pass
    return v


Orientation = Annotated[
    Literal[-180, -135, -90, -45, 0, 45, 90, 135, 180],
    BeforeValidator(_coerce_int),
]

Vec3Cm = tuple[float, float, float]

# Frame transforms translate an origin by adding/subtracting a parent corner.
# Those sums reintroduce IEEE-754 dust (0.1 + 0.0 lands on a double whose
# shortest repr is "0.09999999999999998"). Since every coordinate is authored
# on a centimeter grid, snapping the translated origin back to micrometer
# precision strips the dust without touching any real authored value, and —
# crucially — stops the dust from propagating into child placements.
_COORD_DECIMALS = 6


def _round_vec(v: Vec3Tuple) -> Vec3Cm:
    return (
        round(v[0], _COORD_DECIMALS),
        round(v[1], _COORD_DECIMALS),
        round(v[2], _COORD_DECIMALS),
    )


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box, defined by `origin` vertex and signed `dimensions`.
    The sign of each component of `dimensions` chooses the direction of expansion.
    """

    model_config = ConfigDict(frozen=True)

    origin: Vec3Cm
    dimensions: Vec3Cm

    @classmethod
    def from_center_size(cls, center: Vec3Tuple, size: Vec3Tuple) -> BoundingBox:
        half = (size[0] / 2, size[1] / 2, size[2] / 2)
        return cls(
            origin=(center[0] - half[0], center[1] - half[1], center[2] - half[2]),
            dimensions=size,
        )

    @property
    def min_corner(self) -> Vec3Tuple:
        return (
            min(self.origin[0], self.origin[0] + self.dimensions[0]),
            min(self.origin[1], self.origin[1] + self.dimensions[1]),
            min(self.origin[2], self.origin[2] + self.dimensions[2]),
        )

    @property
    def max_corner(self) -> Vec3Tuple:
        return (
            max(self.origin[0], self.origin[0] + self.dimensions[0]),
            max(self.origin[1], self.origin[1] + self.dimensions[1]),
            max(self.origin[2], self.origin[2] + self.dimensions[2]),
        )

    @property
    def size(self) -> Vec3Tuple:
        return (abs(self.dimensions[0]), abs(self.dimensions[1]), abs(self.dimensions[2]))

    @property
    def center(self) -> Vec3Tuple:
        return (
            self.origin[0] + self.dimensions[0] / 2,
            self.origin[1] + self.dimensions[1] / 2,
            self.origin[2] + self.dimensions[2] / 2,
        )

    @property
    def max_dimension(self) -> float:
        return max(self.size)

    def to_local_frame(self, parent: BoundingBox) -> BoundingBox:
        """Translate this world-frame bbox into `parent`'s local frame.

        The local frame's origin sits at `parent.min_corner` and uses
        the same canonical axes (+X right, +Y up, +Z front), so the
        transform is a pure translation — signed dimensions are
        preserved.
        """
        px, py, pz = parent.min_corner
        ox, oy, oz = self.origin
        return BoundingBox(
            origin=_round_vec((ox - px, oy - py, oz - pz)),
            dimensions=self.dimensions,
        )

    def to_world_frame(self, parent: BoundingBox) -> BoundingBox:
        """Inverse of `to_local_frame`: translate a bbox expressed in
        `parent`'s local frame back to world coordinates."""
        px, py, pz = parent.min_corner
        ox, oy, oz = self.origin
        return BoundingBox(
            origin=_round_vec((ox + px, oy + py, oz + pz)),
            dimensions=self.dimensions,
        )


class ProxyShape(StrEnum):
    """Optional collision-proxy primitive describing the mesh's silhouette
    inside its AABB. `None` on a Node means the AABB itself is the proxy
    (a rectangular prism). The proxy is always inscribed axis-aligned in
    the AABB; its parameters are derived from the AABB's dimensions."""

    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    HEMISPHERE = "HEMISPHERE"


class RelationshipKind(StrEnum):
    """Abstract category of a spatial relationship between two nodes.
    The category is structural — precise positioning is carried in the
    node's `placement` prose. Used to compose multi-target constraints
    without needing per-corner anchor points."""

    ON = "ON"          # child rests on / against target's outward surface
    BESIDE = "BESIDE"  # child adjacent to target (X or Z axis), non-stacked
    ABOVE = "ABOVE"    # child higher in Y than target, no contact required
    BELOW = "BELOW"    # child lower in Y than target, no contact required
    ATTACHED = "ATTACHED"  # child flush against target (any face)
    IN = "IN"          # child contained inside target's volume / footprint


class ParentRelationshipKind(StrEnum):
    """Subset of `RelationshipKind` that's valid as a parent-child anchor.

    A parent edge encodes structural support / containment, so only
    contact-bearing kinds make sense:
      * ON      — child rests on the parent's outward (top) surface.
      * ATTACHED — child is flush against any of the parent's faces
                   (wall mounts, ceiling fixtures, magnets, sconces).
      * IN      — child is contained inside the parent's volume or
                  footprint (subzone inside a zone, fish inside a tank,
                  floating drone inside an enclosing dome, particles
                  inside a fog volume).

    `BESIDE` / `ABOVE` / `BELOW` describe peer/sibling arrangements
    without contact, so they're not load-bearing and would not give a
    layout solver a real anchor; they remain in `RelationshipKind` and
    can still appear in `referenced_ids` for secondary peer hints.
    """

    ON = "ON"
    ATTACHED = "ATTACHED"
    IN = "IN"


class Relationship(BaseModel):
    """One secondary structural relationship between a node and a peer.

    The `target` is another node's id; `kind` is the abstract category
    of the relationship. There is NO `reference_point` — precise
    positioning lives in the placement prose. Use `Relationship` for
    secondary anchors only; a node's primary structural parent lives in
    `Node.parent_id` (and is authored as `ChildNodeSpec.parent`).
    """

    model_config = ConfigDict(frozen=True)

    target: str
    kind: RelationshipKind


class Node(BaseModel):
    """Tree node for the scene.

    Zones and objects are both Nodes. Zones are abstract (mesh_url is None)
    and carry a high-level `plan` (zone identity/character). Concrete nodes
    (objects) set mesh_url and have no plan. Each node stores only its
    parent id; the full tree is recoverable via the run-scoped flat
    registry, but the pipeline emits state to clients incrementally via
    SSE events rather than by traversing the Node graph.

    `prompt` is always the bare subject phrase — what the node *is* in plain
    language. `image_prompt` is the Nano-Banana–specific directive (the
    studio-shot wrapper around the subject phrase) used only at the
    image-generation boundary. Keeping them separate prevents the wrapper
    boilerplate from leaking into LLM context lookups like "objects placed
    so far" / "prior subject phrases in this scene".

    `placement` is the prose description of where this node sits in the
    scene, authored at the decomposition step and resolved by the
    bbox-resolution step. `parent_id` is the structural anchor (the
    load-bearing supporter or the containing zone), and `parent_kind`
    classifies that anchor as `ON` (rests on parent's surface), `ATTACHED`
    (flush against any of the parent's faces), or `IN` (contained inside
    the parent's volume / footprint). `referenced_ids` is an optional
    list of *secondary* relationships — each carrying a target id and a
    `RelationshipKind` — used when the node's placement text references
    more than one peer. The parent is NOT repeated in `referenced_ids`;
    that's what `parent_id`+`parent_kind` are for. `placement`, `parent_id`,
    and `parent_kind` are None only for the root.
    """

    id: str
    prompt: str
    bbox: BoundingBox
    proxy_shape: ProxyShape | None = None
    orientation: Orientation = 0
    placement: str | None = None
    referenced_ids: list[Relationship] = Field(default_factory=list)
    mesh_url: str | None = None
    image_prompt: str | None = None
    # Set before Nano Banana from `symmetry.resolve_cut_plane`. Drives the
    # reference-image view (3/4 when not `none`) and the post-Trellis mirror.
    symmetry_cut_plane: Literal["none", "xy", "xz"] = "none"
    parent_id: str | None = None
    # How this node anchors to its parent: ON (rest), ATTACHED (flush
    # mount), or IN (containment). None only for the root, which has no
    # parent.
    parent_kind: ParentRelationshipKind | None = None
    plan: str | None = None
    # True for subzone/region nodes, set at divider decomposition time; False
    # for concrete objects/frames. A zone is flagged the moment it is placed —
    # before its `plan` is authored — so scene-context renderers treat it as a
    # subregion (seed prompt + bbox) instead of misreading a plan-less zone as
    # a concrete object.
    is_zone: bool = False
