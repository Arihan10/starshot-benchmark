import {
	BufferAttribute,
	BufferGeometry,
	DoubleSide,
	Group,
	type Intersection,
	MathUtils,
	Matrix3,
	Mesh,
	MeshBasicMaterial,
	type PerspectiveCamera,
	Quaternion,
	RingGeometry,
	type Scene,
	Vector3,
} from "three";

// RingGeometry lies in the XY plane facing +Z, so rotating +Z onto a surface
// normal lays the ring flat on that surface.
const _normal = new Vector3();
const _normalMat = new Matrix3();
const _toCam = new Vector3();
const _travel = new Vector3();
const _tangent = new Vector3();
const _local = new Vector3();
const _inv = new Quaternion();
const Z_AXIS = new Vector3(0, 0, 1);
// The cursor is a thing LYING ON a surface, so it takes its size from the world
// and lets perspective do the rest: near the floor at your feet it is large, across
// the room it is small. It used to hold a constant on-screen radius instead, which
// meant its physical footprint grew with distance and shrank as you approached —
// backwards, and close up it collapsed to a sliver that read as cut into the ground
// rather than resting on it.
const CURSOR_WORLD_R = 0.22; // metres — the ring's outer radius in the scene
// Perspective is only honest over a useful range. Past these the cursor would
// either swallow the frame underfoot or vanish across a large scene, so the
// on-screen radius is held between them and world size follows from that.
const CURSOR_MIN_PX = 8;
const CURSOR_MAX_PX = 64;
const CURSOR_OPACITY = 0.9;

// --- the direction arrow ------------------------------------------------------
//
// The arrow says which way a click will move you, drawn IN THE SURFACE the cursor
// is lying on. That only means something when the move can actually be expressed
// as a direction along that surface — which is a question about geometry, not
// about what the surface "is".
//
// Take the travel direction, strip the component along the surface normal, and see
// what survives. On a floor almost all of it does: you move across the floor, and
// the arrow points the way. On a surface square to your path — a wall, the side of
// a crate, a cliff face, the hull of a ship — the travel direction is nearly
// parallel to the normal, so what survives is a near-zero residue whose direction
// is numerical noise. An arrow there would be worse than none: it would point
// somewhere, confidently, at random. So below a floor the arrow is dropped and the
// bare ring stands. The click still works; it simply moves you toward that surface
// rather than along it, which is the thing an arrow cannot say.
//
// The surviving fraction is |sin θ| between the travel direction and the surface
// normal, so these thresholds read as angles: full arrow once the move is more than
// ~33° off the normal, gone below ~20°, fading between so sweeping the cursor across
// a corner doesn't make it blink.
const ARROW_MIN_TANGENT = 0.35;
const ARROW_FULL_TANGENT = 0.55;

// A flat triangle just outside the ring, pointing along +X in the ring's own plane
// (RingGeometry is built in XY facing +Z, so the arrow shares that frame and one Z
// rotation aims it).
function makeArrowGeometry(): BufferGeometry {
	const g = new BufferGeometry();
	g.setAttribute(
		"position",
		new BufferAttribute(
			new Float32Array([
				1.12, -0.38, 0, 1.12, 0.38, 0, 1.8, 0.0, 0,
			]),
			3,
		),
	);
	return g;
}

// Surface-adhering ring cursor: a flat ring laid on the point
// under the native cursor, oriented to the hit surface's normal so it sits flush
// on floors / walls / objects (and foreshortens with them), plus a direction arrow
// whenever the click's movement can be drawn on that surface (see above). Drawn
// over everything (depthTest off) and sized in WORLD units, so perspective grows it
// as you approach; the OS cursor is never hidden — this rides on top of it. The interior raycast that finds the
// point is owned by the engine (it's shared with click auto-aim) and the hit is fed
// in, along with the travel a click from here would take.
export class SurfaceCursor {
	private readonly group = new Group();
	private readonly ring: Mesh;
	private readonly arrow: Mesh;

	constructor(private readonly scene: Scene) {
		this.ring = new Mesh(
			new RingGeometry(0.72, 1, 48),
			new MeshBasicMaterial({
				color: 0x7fe9ff,
				transparent: true,
				opacity: CURSOR_OPACITY,
				side: DoubleSide,
				depthTest: false,
				depthWrite: false,
			}),
		);
		this.arrow = new Mesh(
			makeArrowGeometry(),
			new MeshBasicMaterial({
				color: 0x7fe9ff,
				transparent: true,
				opacity: CURSOR_OPACITY,
				side: DoubleSide,
				depthTest: false,
				depthWrite: false,
			}),
		);
		this.arrow.visible = false;
		this.group.add(this.ring, this.arrow);
		for (const o of [this.group, this.ring, this.arrow]) {
			o.renderOrder = 1000;
			o.frustumCulled = false;
		}
		this.group.visible = false;
		scene.add(this.group);
	}

	// Lay the cursor on `hit` (null hides it), sized from the world so it grows as
	// the surface comes nearer, and aim its arrow along `travel` — the world vector from the eye to
	// where this click would land (null when there is no destination). The hit point
	// lies on the cursor ray, so its projection IS the pointer — the ring is centered
	// on the cursor. It is deliberately not pushed along the normal: depth testing is
	// off (nothing to z-fight), and that offset only shoved the ring off the pointer
	// at grazing angles.
	update(
		hit: Intersection | null,
		camera: PerspectiveCamera,
		viewportHeight: number,
		travel: Vector3 | null = null,
	) {
		if (!hit) {
			this.group.visible = false;
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
		this.group.position.copy(hit.point);
		this.group.quaternion.setFromUnitVectors(Z_AXIS, _normal);
		// Fixed world size, expressed as the on-screen radius it works out to, so it
		// can be clamped in the units the limits are actually about, then converted
		// back. Inside the clamps this is exactly `scale = CURSOR_WORLD_R`.
		const h = viewportHeight || 1;
		const tan = Math.tan((camera.fov * Math.PI) / 360);
		const perPx = (2 * hit.distance * tan) / h; // world metres per screen pixel
		const px = MathUtils.clamp(
			CURSOR_WORLD_R / perPx,
			CURSOR_MIN_PX,
			CURSOR_MAX_PX,
		);
		this.group.scale.setScalar(px * perPx);
		this.aimArrow(travel);
		this.group.visible = true;
	}

	// Point the arrow along the part of `travel` that lies IN the surface, or hide
	// it when too little of it does. `_normal` is the surface normal `update` just
	// resolved (already flipped to face the camera).
	private aimArrow(travel: Vector3 | null) {
		if (!travel || travel.lengthSq() < 1e-8) {
			this.arrow.visible = false;
			return;
		}
		_travel.copy(travel).normalize();
		_tangent.copy(_travel).addScaledVector(_normal, -_travel.dot(_normal));
		const strength = _tangent.length(); // |sin θ| between travel and the normal
		if (strength < ARROW_MIN_TANGENT) {
			this.arrow.visible = false;
			return;
		}
		_tangent.divideScalar(strength);
		// Into the ring's own plane, where a single Z rotation aims the triangle.
		_local
			.copy(_tangent)
			.applyQuaternion(_inv.copy(this.group.quaternion).invert());
		this.arrow.rotation.z = Math.atan2(_local.y, _local.x);
		(this.arrow.material as MeshBasicMaterial).opacity =
			CURSOR_OPACITY *
			MathUtils.clamp(
				(strength - ARROW_MIN_TANGENT) /
					(ARROW_FULL_TANGENT - ARROW_MIN_TANGENT),
				0,
				1,
			);
		this.arrow.visible = true;
	}

	hide() {
		this.group.visible = false;
	}

	setColor(color: number) {
		(this.ring.material as MeshBasicMaterial).color.setHex(color);
		(this.arrow.material as MeshBasicMaterial).color.setHex(color);
	}

	dispose() {
		this.scene.remove(this.group);
		for (const m of [this.ring, this.arrow]) {
			m.geometry.dispose();
			(m.material as MeshBasicMaterial).dispose();
		}
	}
}
