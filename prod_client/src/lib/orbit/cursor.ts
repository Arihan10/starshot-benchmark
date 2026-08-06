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

const _normal = new Vector3();
const _normalMat = new Matrix3();
const _toCam = new Vector3();
const _travel = new Vector3();
const _tangent = new Vector3();
const _local = new Vector3();
const _inv = new Quaternion();
const Z_AXIS = new Vector3(0, 0, 1);
const CURSOR_WORLD_R = 0.22;
const CURSOR_MIN_PX = 8;
const CURSOR_MAX_PX = 64;
const CURSOR_OPACITY = 0.9;

const ARROW_MIN_TANGENT = 0.35;
const ARROW_FULL_TANGENT = 0.55;

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
		const h = viewportHeight || 1;
		const tan = Math.tan((camera.fov * Math.PI) / 360);
		const perPx = (2 * hit.distance * tan) / h;
		const px = MathUtils.clamp(
			CURSOR_WORLD_R / perPx,
			CURSOR_MIN_PX,
			CURSOR_MAX_PX,
		);
		this.group.scale.setScalar(px * perPx);
		this.aimArrow(travel);
		this.group.visible = true;
	}

	private aimArrow(travel: Vector3 | null) {
		if (!travel || travel.lengthSq() < 1e-8) {
			this.arrow.visible = false;
			return;
		}
		_travel.copy(travel).normalize();
		_tangent.copy(_travel).addScaledVector(_normal, -_travel.dot(_normal));
		const strength = _tangent.length();
		if (strength < ARROW_MIN_TANGENT) {
			this.arrow.visible = false;
			return;
		}
		_tangent.divideScalar(strength);
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
