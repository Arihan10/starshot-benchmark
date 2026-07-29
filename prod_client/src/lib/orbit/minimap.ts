import type { PanoEntry } from "./panoTextures";
import type { MinimapLevel, OrbitMode, OrbitState } from "./types";

export type MinimapSlice = MinimapLevel & { url: string };

// A zone name to print on the map, with the world centre of the zone it names.
export type MapLabel = {
	id: string;
	label: string;
	center: [number, number, number];
};

// Nearest slice to a capture height. Levels are Y-separated, so argmin-|Δy|
// reproduces the grouping the capturer used.
export function levelForY(minimaps: MinimapSlice[], y: number): number {
	let best = -1;
	let bestD = Infinity;
	for (let i = 0; i < minimaps.length; i++) {
		const d = Math.abs(minimaps[i].y - y);
		if (d < bestD) {
			bestD = d;
			best = i;
		}
	}
	return best;
}

// The minimap for whatever level the user is on right now — the matching slice
// plus every same-level anchor placed onto it. Only while walking the interior
// (or peeking); the overview already shows the whole dollhouse. Surfaces every
// floor's slice + anchors so the chrome can browse other levels without moving
// the camera; the live capture lights up on its level.
export function buildMinimapState(args: {
	minimaps: MinimapSlice[];
	panos: PanoEntry[];
	panoLevel: number[];
	currentIndex: number;
	mode: OrbitMode;
	labels: MapLabel[];
}): OrbitState["minimap"] {
	const { minimaps, panos, panoLevel, currentIndex, mode, labels } = args;
	if (minimaps.length === 0 || currentIndex < 0) return null;
	if (mode !== "interior" && mode !== "peek") return null;
	const currentLevel = panoLevel[currentIndex];
	if (currentLevel < 0) return null;
	const pct = (n: number) => Math.max(0, Math.min(100, n * 100));
	const levels = minimaps.map((mm, idx) => {
		const b = mm.bounds;
		const spanX = b.maxX - b.minX;
		const spanZ = b.maxZ - b.minZ;
		// Scope the map to the storey. Every slice is rendered over the whole scene
		// footprint, so an upper floor occupying one wing sat marooned in the middle
		// of ground it has nothing to do with. Its described volume is its real
		// extent, so clip the image to that and express everything below — aspect,
		// anchors, labels — in the clipped frame. No volume (an older tour) falls
		// back to the full slice, unchanged.
		const v = mm.volume;
		let x0 = b.minX;
		let x1 = b.maxX;
		let z0 = b.minZ;
		let z1 = b.maxZ;
		if (v && spanX > 0 && spanZ > 0) {
			const cx0 = Math.max(b.minX, v.origin[0]);
			const cx1 = Math.min(b.maxX, v.origin[0] + v.dimensions[0]);
			const cz0 = Math.max(b.minZ, v.origin[2]);
			const cz1 = Math.min(b.maxZ, v.origin[2] + v.dimensions[2]);
			if (cx1 - cx0 > 0 && cz1 - cz0 > 0) {
				x0 = cx0;
				x1 = cx1;
				z0 = cz0;
				z1 = cz1;
			}
		}
		const w = x1 - x0;
		const d = z1 - z0;
		const crop = {
			u0: spanX > 0 ? (x0 - b.minX) / spanX : 0,
			v0: spanZ > 0 ? (z0 - b.minZ) / spanZ : 0,
			u1: spanX > 0 ? (x1 - b.minX) / spanX : 1,
			v1: spanZ > 0 ? (z1 - b.minZ) / spanZ : 1,
		};
		const points: {
			index: number;
			id: string;
			name?: string;
			leftPct: number;
			topPct: number;
			current: boolean;
		}[] = [];
		for (let i = 0; i < panos.length; i++) {
			if (panoLevel[i] !== idx) continue;
			const p = panos[i].position;
			points.push({
				index: i,
				id: panos[i].id,
				name: panos[i].name,
				// Against the CROP's origin, not the slice's — the anchors, the
				// labels and the image all have to be measured in the same frame or
				// the you-are-here dot drifts off the room it is standing in.
				leftPct: w > 0 ? pct((p[0] - x0) / w) : 50,
				topPct: d > 0 ? pct((p[2] - z0) / d) : 50,
				current: i === currentIndex,
			});
		}
		// Zone names land on whichever storey their centre is nearest, placed by the
		// same world→image mapping as the anchors. Each carries the closest capture
		// to that centre, so clicking the name travels there.
		const placed: Array<{
			id: string;
			label: string;
			leftPct: number;
			topPct: number;
			index: number;
		}> = [];
		for (const lab of labels) {
			if (levelForY(minimaps, lab.center[1]) !== idx) continue;
			let index = -1;
			let best = Infinity;
			for (let i = 0; i < panos.length; i++) {
				if (panoLevel[i] !== idx) continue;
				const p = panos[i].position;
				const d =
					(p[0] - lab.center[0]) ** 2 + (p[2] - lab.center[2]) ** 2;
				if (d < best) {
					best = d;
					index = i;
				}
			}
			if (index < 0) continue;
			placed.push({
				id: lab.id,
				label: lab.label,
				leftPct: w > 0 ? pct((lab.center[0] - x0) / w) : 50,
				topPct: d > 0 ? pct((lab.center[2] - z0) / d) : 50,
				index,
			});
		}
		return {
			level: idx,
			// The describer's name for this storey, so the floor control can say
			// "Living & Pool Terrace" rather than "2" (see anchors.py). Null on tours
			// captured before floors were described.
			name: mm.name ?? null,
			url: mm.url,
			aspect: d > 0 ? w / d : 1,
			crop,
			points,
			labels: placed,
		};
	});
	return { currentLevel, levels };
}
