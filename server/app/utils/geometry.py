"""Mesh post-processing: yaw rotation + rescaling into a target bbox.

Scaling policy depends on the yaw:
  * Axis-aligned yaw (a multiple of 90°): each axis is scaled independently so
    the mesh exactly fills the bbox on every axis. The yaw only permutes axes,
    so a per-axis world-space scale stays shear-free. Proportions are not
    preserved — the guarantee is an exact fill.
  * Oblique yaw (±45, ±135): the yaw is about +Y, so it rotates only the X and Z
    axes off the world axes. Scaling X and Z by DIFFERENT amounts then shears the
    mesh, so they instead SHARE one scale — the tighter of the two fills, which
    inscribes the footprint and keeps the in-plane proportions (a ±45/±135 yaw
    makes the footprint square, so a square bbox still fills exactly). Y is the
    yaw axis, untouched by the rotation, so it keeps its own per-axis fill — the
    object keeps its full height. Only a non-square footprint under-fills, which
    is the unavoidable cost of staying shear-free.

Orientation contract:
  Trellis 2 returns a mesh whose intrinsic front face points along +Z
  in mesh frame. The Node's `orientation` is a yaw (integer degrees,
  right-handed about +Y) that rotates the mesh into world pose: 0 leaves
  the front facing world +Z; +90 turns the front to world +X (right),
  -90 to world -X (left). The
  image-prompt step requests an orthographic head-on front view so
  Trellis's output frame is predictable, leaving this rotation as the
  only orientation knob.

Order of operations: yaw → scale → translate to bbox center. Yaw is applied to
the unscaled mesh; an axis-aligned yaw composes cleanly with the per-axis scale,
and an oblique yaw's shared X/Z scale is in-plane uniform (a similarity), so
neither path shears.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from trimesh.intersections import slice_faces_plane

from app.core.types import BoundingBox, Orientation


def _fill_scale(
    target_extents: np.ndarray, source_extents: np.ndarray, orientation: int
) -> np.ndarray:
    """Per-axis fill ratios (`target / source`), made shear-free for `orientation`.

    An axis-aligned yaw only permutes axes, so each axis fills independently. An
    oblique yaw (±45/±135) rotates X and Z off the world axes, so they must share
    ONE scale — the tighter of the two fills — or the mesh shears; Y (the yaw
    axis) still fills on its own, so height is unaffected.
    """
    ratios = target_extents / source_extents
    if orientation % 90 != 0:
        h = float(min(ratios[0], ratios[2]))
        ratios = np.array([h, ratios[1], h])
    return ratios


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

    # Per-axis fill, made shear-free for an oblique yaw (X and Z share the tighter
    # fill; Y fills on its own — see `_fill_scale`). `expected_extents` is what we
    # actually aim to occupy: the full bbox when axis-aligned or when an oblique
    # footprint is square, the inscribed extents otherwise.
    scale_vec = _fill_scale(target_extents, rotated_extents, orientation)
    expected_extents = scale_vec * rotated_extents

    # Collapse recenter + scale + final translate into ONE matrix (one mutation,
    # so no drift compounds across intermediate `bounds` reads on a Scene):
    #     M @ p = S @ (p - rotated_center) + target_center
    M = np.eye(4)
    M[:3, :3] = np.diag(scale_vec)
    M[:3, 3] = target_center - np.diag(scale_vec) @ rotated_center
    out.apply_transform(M)

    # Belt-and-suspenders: re-read the world AABB and correct any drift from what
    # we aimed to occupy (`expected_extents`, NOT the raw bbox — an oblique,
    # non-square footprint deliberately under-fills). The correction follows the
    # same policy (`_fill_scale`) so it can never re-introduce shear.
    final_min, final_max = out.bounds
    final_extents = final_max - final_min
    final_center = (final_min + final_max) / 2.0
    if (
        not np.allclose(final_extents, expected_extents, atol=1e-3)
        or not np.allclose(final_center, target_center, atol=1e-3)
    ):
        if np.any(final_extents <= 0):
            return out
        correction_scale = _fill_scale(expected_extents, final_extents, orientation)
        C = np.eye(4)
        C[:3, :3] = np.diag(correction_scale)
        C[:3, 3] = target_center - np.diag(correction_scale) @ final_center
        out.apply_transform(C)
    return out


def rotate_mesh(
    mesh: trimesh.Trimesh | trimesh.Scene,
    *,
    axis: str,
    degrees: float,
) -> trimesh.Trimesh | trimesh.Scene:
    """Rotate `mesh` by `degrees` about a world axis ('x'/'y'/'z'), returning a
    rotated copy (the input is left untouched).

    Used to re-front a raw mesh — change which face points along +Z, the intrinsic
    "front" Trellis bakes in — before it is symmetrized + rescaled into its bbox.
    The rotation is about the mesh-frame origin; the downstream rescale re-derives
    the rotated AABB and re-centers it into the target box, so the pivot is moot
    for placement (and the per-object raw preview just fits to whatever it loads)."""
    vecs = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    vec = vecs.get(axis.lower())
    if vec is None:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(math.radians(degrees), vec))
    return out


def symmetrize_mesh(
    mesh: trimesh.Trimesh | trimesh.Scene,
    *,
    axis: int = 2,
    keep_positive: bool = True,
) -> trimesh.Trimesh:
    """Mirror one half of `mesh` onto the other so it's symmetric across a plane.

    The mesh is cut at its AABB midpoint along `axis` (mesh frame; default 2 = Z,
    Trellis's intrinsic front/back). The `keep_positive` half is kept — the +Z
    front by default, the side that matches the source image — and reflected
    across the plane to replace the other half. Trellis hallucinates the back
    from a single front view, so mirroring the clean front over it yields an
    object that reads well from every side.

    The reflected half reuses the kept half's UVs and material verbatim, so the
    new back samples the front's texture — the image itself is never touched.
    Winding is reversed on the mirror so normals stay outward, and cut-plane
    vertices map onto themselves so the seam welds shut.

    Cuts with `slice_faces_plane`, which is scipy-free, so no optional trimesh
    dependency is pulled in.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis!r}")
    m = mesh.to_mesh() if isinstance(mesh, trimesh.Scene) else mesh
    if m.is_empty:
        raise ValueError("cannot symmetrize an empty mesh")

    plane_origin = m.bounds.mean(axis=0)
    plane = float(plane_origin[axis])
    normal = np.zeros(3)
    normal[axis] = 1.0 if keep_positive else -1.0

    src_uv = getattr(m.visual, "uv", None)
    # A zero-area source triangle straddling the cut makes slice_faces_plane's
    # barycentric UV interpolation divide by zero — emitting numpy warnings and
    # producing NaN/inf UVs at the new cut vertices. Silence the expected
    # warnings here and clamp the result below: a non-finite UV left in is
    # written verbatim into the exported GLB's accessor min/max as a literal
    # `NaN` (invalid JSON that crashes strict glTF readers like the optimizer's
    # @gltf-transform).
    with np.errstate(divide="ignore", invalid="ignore"):
        verts, faces, uv = slice_faces_plane(
            np.asarray(m.vertices, dtype=np.float64),
            np.asarray(m.faces),
            normal,
            plane_origin,
            uv=None if src_uv is None else np.asarray(src_uv, dtype=np.float64),
        )
    if len(faces) == 0:
        raise ValueError("cut plane left an empty half; cannot symmetrize")
    if uv is not None:
        uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)

    mirror = verts.copy()
    mirror[:, axis] = 2.0 * plane - mirror[:, axis]
    all_verts = np.vstack([verts, mirror])
    # Reverse winding on the mirrored faces so the reflection keeps normals
    # outward, and offset their indices into the appended mirror block.
    all_faces = np.vstack([faces, faces[:, ::-1] + len(verts)])

    visual = None
    if uv is not None:
        visual = trimesh.visual.TextureVisuals(
            uv=np.vstack([uv, uv]), material=m.visual.material
        )
    out = trimesh.Trimesh(vertices=all_verts, faces=all_faces, visual=visual, process=False)
    out.merge_vertices()
    return out


def export_glb(mesh: trimesh.Trimesh | trimesh.Scene, path: Path) -> None:
    """Export `mesh` to a GLB at `path`, clamping any non-finite UVs to 0 first.

    trimesh writes each accessor's min/max straight from the array, so a single
    NaN/inf UV (e.g. interpolated from a degenerate triangle during symmetrize)
    emits a literal `NaN` token into the GLB's JSON chunk — invalid JSON that
    crashes strict glTF readers (the optimizer's @gltf-transform NodeIO). This is
    the last line of defence, independent of where a bad UV originated.
    """
    geoms = mesh.geometry.values() if isinstance(mesh, trimesh.Scene) else (mesh,)
    for g in geoms:
        uv = getattr(getattr(g, "visual", None), "uv", None)
        if uv is None:
            continue
        uv = np.asarray(uv, dtype=np.float64)
        if not np.isfinite(uv).all():
            g.visual.uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    mesh.export(path, file_type="glb")
