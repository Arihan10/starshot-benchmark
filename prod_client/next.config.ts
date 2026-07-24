import type { NextConfig } from "next";

const R2_ORIGIN = "https://benchmark.tryflopilot.com";

// LOCAL (no-Cloudflare) testing: when NEXT_PUBLIC_LOCAL_API points at the local
// orchestrator, /r2/* proxies to its /artifacts/* (the same-origin proxy pattern
// keeps WebGL happy without CORS) and /api/scenes reads the on-disk catalog
// (src/app/api/scenes/route.ts). Unset → the normal R2 bucket.
const LOCAL_API = process.env.NEXT_PUBLIC_LOCAL_API?.replace(/\/+$/, "");

const nextConfig: NextConfig = {
	// Proxy asset requests through our own origin so WebGL can load them without
	// the source needing CORS headers. In local mode this targets the
	// orchestrator's /artifacts route; otherwise the R2 bucket.
	async rewrites() {
		return [
			{
				source: "/r2/:path*",
				destination: LOCAL_API
					? `${LOCAL_API}/artifacts/:path*`
					: `${R2_ORIGIN}/:path*`,
			},
		];
	},
};

export default nextConfig;
