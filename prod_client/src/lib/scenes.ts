// The scene catalog: the client's view of the D1 `scenes` table. Each row is one
// published (run / slot / model) cell — unversioned — carrying the R2 keys of its
// assets. The client fetches the catalog from `/api/scenes` (a Route Handler that
// holds the D1 credentials server-side) and resolves asset URLs straight from the
// stored keys — it never reconstructs the bucket layout itself.
import type { TourSource } from "./orbit/types";
import { assetUrl, cfImageUrl } from "./r2";

export type Scene = {
	run: string;
	slot: string;
	model: string;
	previewKey: string; // previews/.../scene-lite.glb (the dollhouse, always present)
	tourKey: string | null; // tours/.../tour.json (the walkthrough plan; null = no tour)
	proxyKey: string | null; // proxies/.../proxy.glb (projection proxy; null = none)
	panoPrefix: string | null; // panoramas/.../  (panos hang under here; null = none)
	panoCount: number;
	publishedAt: string;
};

/** Stable unique id for a scene, e.g. "run/slot/model". */
export const sceneId = (s: Scene): string => `${s.run}/${s.slot}/${s.model}`;

/** Human label for the picker, e.g. "modern-house · opus-new". */
export const sceneLabel = (s: Scene): string => `${s.slot} · ${s.model}`;

/** Fetch the catalog (newest first) from the same-origin Route Handler. */
export async function fetchScenes(): Promise<Scene[]> {
	const res = await fetch("/api/scenes", { cache: "no-store" });
	if (!res.ok) {
		const detail = await res.json().catch(() => null);
		throw new Error(detail?.error ?? `failed to load scenes (${res.status})`);
	}
	const data = (await res.json()) as { scenes?: Scene[] };
	return data.scenes ?? [];
}

// --- asset resolution (from the catalog's stored keys) ----------------------

export const previewUrl = (s: Scene): string => assetUrl(s.previewKey);

export const panoUrl = (s: Scene, file: string): string => assetUrl(`${s.panoPrefix ?? ""}${file}`);

export const panoPlaceholderUrl = (s: Scene, file: string): string =>
	cfImageUrl(`${s.panoPrefix ?? ""}${file}`, { width: 640, quality: 50, blur: 16 });

// Panos are captured as a contiguous `anchor-NNN.jpg` run, so the catalog's
// pano_count is enough to list them without fetching the manifest.
export const panoFiles = (s: Scene): string[] =>
	Array.from({ length: s.panoCount }, (_, i) => `anchor-${String(i).padStart(3, "0")}.jpg`);

// Everything the orbit engine needs to load one scene. The engine fetches the
// manifest (the "plan") itself and resolves each pano/proxy filename through
// these closures, so it stays oblivious to the catalog and bucket layout.
export function tourSource(s: Scene): TourSource {
	return {
		dollhouseUrl: previewUrl(s),
		manifestUrl: s.tourKey ? assetUrl(s.tourKey) : null,
		resolvePano: (file) => ({ url: panoUrl(s, file), placeholderUrl: panoPlaceholderUrl(s, file) }),
		resolveProxy: () => assetUrl(s.proxyKey ?? ""),
		// Minimap slices and object-ID masks are published under the same prefix as
		// the panos. Masks deliberately resolve through `panoUrl`, never the image
		// transformation edge — a resized or re-encoded mask is a corrupt one.
		resolveMinimap: (file) => panoUrl(s, file),
		resolveMask: (file) => panoUrl(s, file),
	};
}
