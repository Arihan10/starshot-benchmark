import type { PanoEntry } from "./panoTextures";
import type { MinimapLevel, OrbitMode, OrbitState } from "./types";

export type MinimapSlice = MinimapLevel & { url: string };

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
}): OrbitState["minimap"] {
	const { minimaps, panos, panoLevel, currentIndex, mode } = args;
	if (minimaps.length === 0 || currentIndex < 0) return null;
	if (mode !== "interior" && mode !== "peek") return null;
	const currentLevel = panoLevel[currentIndex];
	if (currentLevel < 0) return null;
	const pct = (n: number) => Math.max(0, Math.min(100, n * 100));
	const levels = minimaps.map((mm, idx) => {
		const w = mm.bounds.maxX - mm.bounds.minX;
		const d = mm.bounds.maxZ - mm.bounds.minZ;
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
				leftPct: w > 0 ? pct((p[0] - mm.bounds.minX) / w) : 50,
				topPct: d > 0 ? pct((p[2] - mm.bounds.minZ) / d) : 50,
				current: i === currentIndex,
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
			points,
		};
	});
	return { currentLevel, levels };
}
