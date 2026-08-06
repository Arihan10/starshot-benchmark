import { d1Query } from "@/lib/d1";
import type { Scene } from "@/lib/scenes";

export const dynamic = "force-dynamic";

const LOCAL_API = process.env.NEXT_PUBLIC_LOCAL_API?.replace(/\/+$/, "");

const str = (v: unknown): string => String(v ?? "");
const nullable = (v: unknown): string | null => (v == null ? null : String(v));

function toScene(r: Record<string, unknown>): Scene {
	return {
		run: str(r.run),
		slot: str(r.slot),
		model: str(r.model),
		previewKey: str(r.preview_key),
		tourKey: nullable(r.tour_key),
		proxyKey: nullable(r.proxy_key),
		panoPrefix: nullable(r.pano_prefix),
		splatKey: nullable(r.splat_key),
		panoCount: Number(r.pano_count ?? 0),
		publishedAt: str(r.published_at),
	};
}

async function loadRows(): Promise<Record<string, unknown>[]> {
	if (LOCAL_API) {
		const res = await fetch(`${LOCAL_API}/tour/scenes`, { cache: "no-store" });
		if (!res.ok) throw new Error(`local catalog failed (${res.status})`);
		const data = (await res.json()) as { scenes?: Record<string, unknown>[] };
		return data.scenes ?? [];
	}
	return d1Query(
		`SELECT run, slot, model, preview_key, tour_key, proxy_key,
		        pano_prefix, pano_count, published_at
		   FROM scenes
		  ORDER BY published_at DESC`,
	);
}

export async function GET() {
	try {
		const scenes: Scene[] = (await loadRows()).map(toScene);
		return Response.json({ scenes });
	} catch (err) {
		const message = err instanceof Error ? err.message : "failed to load scenes";
		return Response.json({ error: message }, { status: 500 });
	}
}
