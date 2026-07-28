import {
	DoubleSide,
	type Intersection,
	type Material,
	Matrix3,
	Mesh,
	MeshBasicMaterial,
	type PerspectiveCamera,
	RingGeometry,
	type Scene,
	Vector3,
} from "three";

// RingGeometry lies in the XY plane facing +Z, so rotating +Z onto a surface
// normal lays the ring flat on that surface.
const _normal = new Vector3();
const _normalMat = new Matrix3();
const _toCam = new Vector3();
const Z_AXIS = new Vector3(0, 0, 1);
const RING_OUTER_PX = 18; // constant on-screen outer radius

// Surface-adhering ring cursor (interior only): a flat ring laid on the point
// under the native cursor, oriented to the hit surface's normal so it sits flush
// on floors / walls / objects (and foreshortens with them). Drawn over everything
// (depthTest off) and kept a constant on-screen size; the OS cursor is never
// hidden — this rides on top of it. The interior raycast that finds the point is
// owned by the engine (it's shared with click auto-aim) and the hit is fed in.
export class SurfaceCursor {
	private readonly ring: Mesh;

	constructor(private readonly scene: Scene) {
		this.ring = new Mesh(
			new RingGeometry(0.72, 1, 48),
			new MeshBasicMaterial({
				color: 0x7fe9ff,
				transparent: true,
				opacity: 0.9,
				side: DoubleSide,
				depthTest: false,
				depthWrite: false,
			}),
		);
		this.ring.renderOrder = 1000;
		this.ring.frustumCulled = false;
		this.ring.visible = false;
		scene.add(this.ring);
	}

	// Lay the ring on `hit` (null hides it), scaled to a constant on-screen radius.
	// The hit point lies on the cursor ray, so its projection IS the pointer — the
	// ring is centered on the cursor. It is deliberately not pushed along the normal:
	// depth testing is off (nothing to z-fight), and that offset only shoved the ring
	// off the pointer at grazing angles.
	update(
		hit: Intersection | null,
		camera: PerspectiveCamera,
		viewportHeight: number,
	) {
		if (!hit) {
			this.ring.visible = false;
			return;
		}
		if (hit.face) {
			_normalMat.getNormalMatrix(hit.object.matrixWorld);
			_normal.copy(hit.face.normal).applyMatrix3(_normalMat).normalize();
		} else {
			_normal.copy(camera.position).sub(hit.point).normalize();
		}
		_toCam.copy(camera.position).sub(hit.point);
		if (_normal.dot(_toCam) < 0) _normal.negate();
		this.ring.position.copy(hit.point);
		this.ring.quaternion.setFromUnitVectors(Z_AXIS, _normal);
		const tan = Math.tan((camera.fov * Math.PI) / 360);
		this.ring.scale.setScalar(
			(RING_OUTER_PX * 2 * hit.distance * tan) / (viewportHeight || 1),
		);
		this.ring.visible = true;
	}

	hide() {
		this.ring.visible = false;
	}

	setColor(color: number) {
		(this.ring.material as MeshBasicMaterial).color.setHex(color);
	}

	dispose() {
		this.scene.remove(this.ring);
		this.ring.geometry.dispose();
		(this.ring.material as Material).dispose();
	}
}
