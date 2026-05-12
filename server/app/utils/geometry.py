"""Mesh post-processing: yaw rotation + rescaling into a target bbox.

Per-axis scaling: each axis is scaled independently so the mesh exactly
fills the bbox on every axis. Proportions are not preserved — the
guarantee is that the mesh is fully contained in the bbox.

Orientation contract:
  Trellis 2 returns a mesh whose intrinsic front face points along +Z
  in mesh frame. The Node's `orientation` is a yaw (integer degrees,
  right-handed about +Y) that rotates the mesh into world pose: 0 leaves
  the front facing world +Z; 90 turns the front to world -X. The
  image-prompt step requests an orthographic head-on front view so
  Trellis's output frame is predictable, leaving this rotation as the
  only orientation knob.

Order of operations: translate to origin → yaw → scale → translate to
bbox center. Yaw is applied to the unscaled mesh so it composes cleanly
with per-axis scaling.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from app.core.types import BoundingBox, Orientation


def rescale_mesh_to_bbox(
    mesh: trimesh.Trimesh | trimesh.Scene,
    bbox: BoundingBox,
    *,
    orientation: Orientation = 0,
) -> trimesh.Trimesh | trimesh.Scene:
    if mesh.is_empty:
        raise ValueError("cannot rescale an empty mesh")
    out = mesh.copy()

    # Rotate first (around origin), THEN re-derive the AABB. Rotating an
    # origin-centered AABB doesn't preserve centering — vertex positions
    # don't have to be symmetric inside their AABB, so a rotation around
    # the AABB centre can land the new AABB off-origin.
    R = trimesh.transformations.rotation_matrix(math.radians(orientation), [0.0, 1.0, 0.0])
    out.apply_transform(R)

    rotated_min, rotated_max = out.bounds
    rotated_extents = np.asarray(rotated_max - rotated_min, dtype=float)
    if np.any(rotated_extents <= 0):
        raise ValueError("degenerate mesh has zero extent on some axis")
    rotated_center = (rotated_min + rotated_max) / 2.0
    target_extents = np.asarray(bbox.size, dtype=float)
    target_center = np.asarray(bbox.center, dtype=float)
    scale_vec = target_extents / rotated_extents

    # Compose the recenter + per-axis scale + final translate into ONE
    # matrix. Applying these as three separate `apply_transform` calls is
    # mathematically equivalent but harder to reason about for a Scene
    # (each call mutates every per-geometry transform in the graph), and
    # any drift between intermediate `bounds` reads compounds. A single
    # composed matrix collapses the whole sequence to one mutation.
    #     M @ p = S @ (p - rotated_center) + target_center
    M = np.eye(4)
    M[:3, :3] = np.diag(scale_vec)
    M[:3, 3] = target_center - np.diag(scale_vec) @ rotated_center
    out.apply_transform(M)

    # Belt-and-suspenders: re-read the actual world AABB after all
    # transforms. If the final bounds drift from the requested bbox by
    # more than centimetre tolerance (multi-geometry scene graphs,
    # nested per-geom transforms, float precision), apply a corrective
    # translate+scale that forces a fit. This guarantees the mesh is
    # inside the bbox regardless of any upstream transform pathology.
    final_min, final_max = out.bounds
    final_extents = final_max - final_min
    final_center = (final_min + final_max) / 2.0
    if (
        not np.allclose(final_extents, target_extents, atol=1e-3)
        or not np.allclose(final_center, target_center, atol=1e-3)
    ):
        if np.any(final_extents <= 0):
            return out
        correction_scale = target_extents / final_extents
        C = np.eye(4)
        C[:3, :3] = np.diag(correction_scale)
        C[:3, 3] = target_center - np.diag(correction_scale) @ final_center
        out.apply_transform(C)
    return out
