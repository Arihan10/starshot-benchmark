import {
	MathUtils,
	type PerspectiveCamera,
	type Raycaster,
	Vector2,
	Vector3,
} from "three";

const _ndc = new Vector2();

export function forwardToLonLat(f: [number, number, number]): {
	lon: number;
	lat: number;
} {
	const v = new Vector3().fromArray(f).normalize();
	return {
		lon: Math.atan2(v.z, v.x),
		lat: Math.asin(MathUtils.clamp(v.y, -1, 1)),
	};
}

export function lookTargetFrom(pos: Vector3, lon: number, lat: number): Vector3 {
	return pos
		.clone()
		.add(
			new Vector3(
				Math.cos(lat) * Math.cos(lon),
				Math.sin(lat),
				Math.cos(lat) * Math.sin(lon),
			),
		);
}

export const MAX_PITCH = 1.55;

export function applyLook(
	camera: PerspectiveCamera,
	lon: number,
	lat: number,
): number {
	const clamped = MathUtils.clamp(lat, -MAX_PITCH, MAX_PITCH);
	camera.lookAt(lookTargetFrom(camera.position, lon, clamped));
	return clamped;
}

export function cursorRayDir(
	camera: PerspectiveCamera,
	canvas: HTMLCanvasElement,
	ray: Raycaster,
	clientX: number,
	clientY: number,
): Vector3 {
	const rect = canvas.getBoundingClientRect();
	_ndc.set(
		((clientX - rect.left) / rect.width) * 2 - 1,
		-((clientY - rect.top) / rect.height) * 2 + 1,
	);
	camera.updateMatrixWorld();
	ray.setFromCamera(_ndc, camera);
	return ray.ray.direction;
}

export function pinLook(
	camera: PerspectiveCamera,
	canvas: HTMLCanvasElement,
	clientX: number,
	clientY: number,
	grabDir: Vector3,
): { lon: number; lat: number } {
	const rect = canvas.getBoundingClientRect();
	const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
	const ny = -((clientY - rect.top) / rect.height) * 2 + 1;
	const t = Math.tan((camera.fov * Math.PI) / 360);
	const gx = nx * camera.aspect * t;
	const gy = ny * t;
	const invL = 1 / Math.hypot(gx, gy, 1);
	const cX = gx * invL;
	const cY = gy * invL;
	const cF = invL;
	const d = grabDir;
	const rh = Math.hypot(d.x, d.z);
	const lon =
		Math.atan2(d.z, d.x) - Math.asin(MathUtils.clamp(cX / rh, -1, 1));
	const h = Math.sqrt(Math.max(0, rh * rh - cX * cX));
	const lat = Math.atan2(cF * d.y - h * cY, cF * h + d.y * cY);
	return { lon, lat };
}
