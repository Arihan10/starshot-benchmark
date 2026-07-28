import {
	MathUtils,
	type PerspectiveCamera,
	type Raycaster,
	Vector2,
	Vector3,
} from "three";

// Interior look is a yaw/pitch rig: lon (azimuth) + lat (elevation). These are
// pure helpers; the engine owns the live lon/lat and feeds them back in.
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

// Pitch limit for the yaw/pitch rig — just shy of straight up/down, where a
// lookAt with a +Y up vector degenerates. Exported so a camera flight can aim at
// a pre-clamped pitch and land on exactly the pose the rig will then hold.
export const MAX_PITCH = 1.55;

// Clamp pitch and aim the camera; returns the clamped lat so the caller keeps
// its stored value in range.
export function applyLook(
	camera: PerspectiveCamera,
	lon: number,
	lat: number,
): number {
	const clamped = MathUtils.clamp(lat, -MAX_PITCH, MAX_PITCH);
	camera.lookAt(lookTargetFrom(camera.position, lon, clamped));
	return clamped;
}

// World-space ray direction through a screen pixel (normalized). The passed
// raycaster is reused as scratch; the returned direction is its live ray.
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

// Turn the look so the grabbed world direction reprojects onto the current
// cursor pixel — the point stays welded under the cursor as you drag. The eye
// is fixed during a look-drag, so this is the exact inverse of the perspective
// projection for the yaw/pitch rig: the pixel becomes a camera-space unit ray
// (cX right, cY up, cF forward), then lon/lat are the angles that map grabDir
// onto it. {forward, right, up} is orthonormal, so cF = 1/‖ray‖ falls straight
// out; lon comes from grabDir's azimuth offset by the ray's horizontal angle,
// and lat rotates grabDir's (horizontal, vertical) parts onto (cF, cY).
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
