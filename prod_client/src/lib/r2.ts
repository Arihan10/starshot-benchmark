// Low-level R2 URL helpers. Assets live in the Cloudflare R2 bucket
// `benchmark-assets-prod`, served from benchmark.tryflopilot.com. That origin
// sends no CORS headers (which WebGL's GLTFLoader fetch + texture crossOrigin
// require), so by default we go through the same-origin `/r2` proxy defined in
// next.config.ts. Set NEXT_PUBLIC_R2_BASE_URL to hit the bucket directly once
// CORS is enabled.
//
// These take a bare object key; the per-scene keys themselves come from the D1
// catalog (see lib/scenes.ts) — this module never hard-codes a layout.
const RAW_BASE = process.env.NEXT_PUBLIC_R2_BASE_URL ?? "/r2";

export const R2_BASE_URL = RAW_BASE.replace(/\/+$/, "");

const stripLeadingSlashes = (key: string) => key.replace(/^\/+/, "");

/** The same-origin URL for an R2 object key. */
export function assetUrl(key: string): string {
	return `${R2_BASE_URL}/${stripLeadingSlashes(key)}`;
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
// enabled on the zone, so we can derive resized/blurred variants on the fly —
// used for the low-res panorama placeholders.
export function cfImageUrl(key: string, options: CfImageOptions): string {
	const params = Object.entries(options)
		.filter(([, value]) => value !== undefined)
		.map(([name, value]) => `${name}=${value}`)
		.join(",");
	return assetUrl(`cdn-cgi/image/${params}/${stripLeadingSlashes(key)}`);
}
