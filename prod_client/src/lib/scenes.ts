import type { TourSource } from "./orbit/types";
import { assetUrl, cfImageUrl } from "./r2";

export type Scene = {
	run: string;
	slot: string;
	model: string;
	previewKey: string;
	tourKey: string | null;
	proxyKey: string | null;
	panoPrefix: string | null;
	splatKey: string | null;
	panoCount: number;
	publishedAt: string;
};

export const sceneId = (s: Scene): string => `${s.run}/${s.slot}/${s.model}`;

export const sceneLabel = (s: Scene): string => `${s.slot} · ${s.model}`;

export async function fetchScenes(): Promise<Scene[]> {
	const res = await fetch("/api/scenes", { cache: "no-store" });
	if (!res.ok) {
		const detail = await res.json().catch(() => null);
		throw new Error(detail?.error ?? `failed to load scenes (${res.status})`);
	}
	const data = (await res.json()) as { scenes?: Scene[] };
	return data.scenes ?? [];
}

export const previewUrl = (s: Scene): string => assetUrl(s.previewKey);

export const splatUrl = (s: Scene): string | null =>
	s.splatKey ? assetUrl(s.splatKey) : null;

export const panoUrl = (s: Scene, file: string): string => assetUrl(`${s.panoPrefix ?? ""}${file}`);

export const panoPlaceholderUrl = (s: Scene, file: string): string =>
	cfImageUrl(`${s.panoPrefix ?? ""}${file}`, { width: 640, quality: 50, blur: 16 });

export const panoFiles = (s: Scene): string[] =>
	Array.from({ length: s.panoCount }, (_, i) => `anchor-${String(i).padStart(3, "0")}.jpg`);

export function tourSource(s: Scene): TourSource {
	return {
		dollhouseUrl: previewUrl(s),
		manifestUrl: s.tourKey ? assetUrl(s.tourKey) : null,
		splatUrl: splatUrl(s),
		resolvePano: (file) => ({ url: panoUrl(s, file), placeholderUrl: panoPlaceholderUrl(s, file) }),
		resolveProxy: () => assetUrl(s.proxyKey ?? ""),
		resolveMinimap: (file) => panoUrl(s, file),
	};
}
