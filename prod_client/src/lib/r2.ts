// Assets live in the Cloudflare R2 bucket `benchmark-assets-prod`, served from
// benchmark.tryflopilot.com. That origin sends no CORS headers, which WebGL
// (GLTFLoader fetch + texture crossOrigin) requires, so by default we go
// through the same-origin `/r2` proxy defined in next.config.ts. Set
// NEXT_PUBLIC_R2_BASE_URL to hit the bucket directly once CORS is enabled.
const RAW_BASE = process.env.NEXT_PUBLIC_R2_BASE_URL ?? "/r2";

export const R2_BASE_URL = RAW_BASE.replace(/\/+$/, "");

export const PANORAMA_COUNT = 22;

export const SCENE_PREVIEW_KEY = "previews/scene-lite.glb";

// The capture-tour manifest (pano world positions + optional proxy) for the
// preview scene. Optional: when absent, orbit shows the dollhouse on its own.
// The manifest names panos/proxy by bare filename; in R2 they live under their
// own prefixes (tours/, panoramas/, proxies/), so resolve via panoFileUrls /
// proxyFileUrl rather than relative to the manifest URL.
export const TOUR_MANIFEST_KEY = "tours/tour.json";

export const PANORAMA_KEYS = Array.from(
	{ length: PANORAMA_COUNT },
	(_, i) => `panoramas/anchor-${String(i).padStart(3, "0")}.jpg`,
);

export function assetUrl(key: string): string {
	return `${R2_BASE_URL}/${key.replace(/^\/+/, "")}`;
}

export function scenePreviewUrl(): string {
	return assetUrl(SCENE_PREVIEW_KEY);
}

export function tourManifestUrl(): string {
	return assetUrl(TOUR_MANIFEST_KEY);
}

// Map a manifest's bare pano filename to its R2 URLs: the full image plus a
// low-res blurred placeholder (Cloudflare Image Transform), so the viewer can
// show the preview first and sharpen in place — same as the panorama page.
export function panoFileUrls(file: string): { url: string; placeholderUrl: string } {
	const key = `panoramas/${file.replace(/^.*\//, "")}`;
	return {
		url: assetUrl(key),
		placeholderUrl: cfImageUrl(key, { width: 640, quality: 50, blur: 16 }),
	};
}

// The proxy GLB sits under proxies/.
export function proxyFileUrl(file: string): string {
	return assetUrl(`proxies/${file.replace(/^.*\//, "")}`);
}

export function panoramaUrl(index: number): string {
	const key = PANORAMA_KEYS[index];
	if (!key) throw new Error(`panorama index out of range: ${index}`);
	return assetUrl(key);
}

type CfImageOptions = {
	width?: number;
	height?: number;
	quality?: number;
	blur?: number;
	fit?: "scale-down" | "contain" | "cover" | "crop" | "pad";
	format?: "auto" | "webp" | "avif";
};

// Cloudflare Image Transformations (/cdn-cgi/image/<options>/<source>) are
// enabled on the zone, so we can derive resized/blurred variants on the fly.
export function cfImageUrl(key: string, options: CfImageOptions): string {
	const params = Object.entries(options)
		.filter(([, value]) => value !== undefined)
		.map(([name, value]) => `${name}=${value}`)
		.join(",");
	return assetUrl(`cdn-cgi/image/${params}/${key.replace(/^\/+/, "")}`);
}

export function panoramaPlaceholderUrl(index: number): string {
	const key = PANORAMA_KEYS[index];
	if (!key) throw new Error(`panorama index out of range: ${index}`);
	return cfImageUrl(key, { width: 640, quality: 50, blur: 16 });
}
