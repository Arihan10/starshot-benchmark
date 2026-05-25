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


class ProxyShape(StrEnum):
    """Optional collision-proxy primitive describing the mesh's silhouette
    inside its AABB. `None` on a Node means the AABB itself is the proxy
    (a rectangular prism). The proxy is always inscribed axis-aligned in
    the AABB; its parameters are derived from the AABB's dimensions."""

    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    HEMISPHERE = "HEMISPHERE"


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
    bbox-resolution step. `referenced_ids` lists every node id mentioned
    in `placement` — `referenced_ids[0]` is the structural parent by
    prompting convention (the load-bearing supporter or the containing
    zone). Both are None/empty for the root node only.
    """

    id: str
    prompt: str
    bbox: BoundingBox
    proxy_shape: ProxyShape | None = None
    orientation: Orientation = 0
    placement: str | None = None
    referenced_ids: list[str] = Field(default_factory=list)
    mesh_url: str | None = None
    image_prompt: str | None = None
    parent_id: str | None = None
    plan: str | None = None
