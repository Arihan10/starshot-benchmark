"""
Domain types for the pipeline.

Coordinates: Y-up, right-handed, meters.
Canonical front view: +X = right, +Y = up, +Z = toward the viewer.
All bounding boxes, corners, and relationships are expressed under this
convention.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Vec3Tuple = tuple[float, float, float]

Orientation = Literal[-180, -135, -90, -45, 0, 45, 90, 135, 180]

Vec3Cm = tuple[float, float, float]


class Corner(StrEnum):
    """One of the 8 corners of an AABB under the canonical front view."""

    TOP_LEFT_FRONT = "TOP_LEFT_FRONT"
    TOP_LEFT_BACK = "TOP_LEFT_BACK"
    TOP_RIGHT_FRONT = "TOP_RIGHT_FRONT"
    TOP_RIGHT_BACK = "TOP_RIGHT_BACK"
    BOTTOM_LEFT_FRONT = "BOTTOM_LEFT_FRONT"
    BOTTOM_LEFT_BACK = "BOTTOM_LEFT_BACK"
    BOTTOM_RIGHT_FRONT = "BOTTOM_RIGHT_FRONT"
    BOTTOM_RIGHT_BACK = "BOTTOM_RIGHT_BACK"


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

    def corner_point(self, corner: Corner) -> Vec3Tuple:
        lo, hi = self.min_corner, self.max_corner
        x = hi[0] if "_RIGHT_" in corner.value else lo[0]
        y = hi[1] if corner.value.startswith("TOP") else lo[1]
        z = hi[2] if corner.value.endswith("FRONT") else lo[2]
        return (x, y, z)


class RelationshipKind(StrEnum):
    ON = "ON"
    BESIDE = "BESIDE"
    BELOW = "BELOW"
    ABOVE = "ABOVE"
    ATTACHED = "ATTACHED"


class ProxyShape(StrEnum):
    """Optional collision-proxy primitive describing the mesh's silhouette
    inside its AABB. `None` on a Node means the AABB itself is the proxy
    (a rectangular prism). The proxy is always inscribed axis-aligned in
    the AABB; its parameters are derived from the AABB's dimensions."""

    SPHERE = "SPHERE"
    CAPSULE = "CAPSULE"
    HEMISPHERE = "HEMISPHERE"


class Relationship(BaseModel):
    """Anchors a node's bbox to a target corner of another node's bbox."""

    model_config = ConfigDict(frozen=True)

    target: str
    kind: RelationshipKind
    reference_point: Corner


class Node(BaseModel):
    """Tree node for the scene.

    Zones and objects are both Nodes. Zones are abstract (mesh_url is None)
    and carry a high-level `plan` (zone identity/character). Concrete nodes
    (objects) set mesh_url and have no plan. Each node stores only its
    parent id; the full tree is recoverable via the run-scoped flat
    registry, but the pipeline emits state to clients incrementally via
    SSE events rather than by traversing the Node graph.
    """

    id: str
    prompt: str
    bbox: BoundingBox
    proxy_shape: ProxyShape | None = None
    orientation: Orientation = 0
    relationships: list[Relationship] = Field(default_factory=list)
    mesh_url: str | None = None
    parent_id: str | None = None
    plan: str | None = None
