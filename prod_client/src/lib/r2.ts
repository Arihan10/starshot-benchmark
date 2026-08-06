const RAW_BASE = process.env.NEXT_PUBLIC_R2_BASE_URL ?? "/r2";

export const R2_BASE_URL = RAW_BASE.replace(/\/+$/, "");

const stripLeadingSlashes = (key: string) => key.replace(/^\/+/, "");

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

const LOCAL = !!process.env.NEXT_PUBLIC_LOCAL_API;

export function cfImageUrl(key: string, options: CfImageOptions): string {
	if (LOCAL) return assetUrl(key);
	const params = Object.entries(options)
		.filter(([, value]) => value !== undefined)
		.map(([name, value]) => `${name}=${value}`)
		.join(",");
	return assetUrl(`cdn-cgi/image/${params}/${stripLeadingSlashes(key)}`);
}
