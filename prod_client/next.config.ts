import type { NextConfig } from "next";

const R2_ORIGIN = "https://benchmark.tryflopilot.com";

const LOCAL_API = process.env.NEXT_PUBLIC_LOCAL_API?.replace(/\/+$/, "");

const nextConfig: NextConfig = {
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
