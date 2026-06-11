import type { NextConfig } from "next";

const R2_ORIGIN = "https://benchmark.tryflopilot.com";

const nextConfig: NextConfig = {
	// Proxy R2 assets through our own origin so WebGL can load them without the
	// bucket needing CORS headers (it currently sends none).
	async rewrites() {
		return [
			{
				source: "/r2/:path*",
				destination: `${R2_ORIGIN}/:path*`,
			},
		];
	},
};

export default nextConfig;
