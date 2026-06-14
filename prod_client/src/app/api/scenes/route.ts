// GET /api/scenes — the published-scene catalog from D1, newest first.
//
// Runs server-side so the D1 token stays off the client (lib/d1.ts). Route
// Handlers aren't cached by default; force-dynamic makes that explicit so the
// list always reflects the current D1 state (scenes appear as soon as the
// backend auto-publishes them).
import { d1Query } from "@/lib/d1";
import type { Scene } from "@/lib/scenes";

export const dynamic = "force-dynamic";

const str = (v: unknown): string => String(v ?? "");
const nullable = (v: unknown): string | null => (v == null ? null : String(v));

export async function GET() {
	try {
		const rows = await d1Query(
			`SELECT run, slot, model, version, preview_key, tour_key, proxy_key,
			        pano_prefix, pano_count, published_at
			   FROM scenes
			  ORDER BY published_at DESC`,
		);
		const scenes: Scene[] = rows.map((r) => ({
			run: str(r.run),
			slot: str(r.slot),
			model: str(r.model),
			version: str(r.version),
			previewKey: str(r.preview_key),
			tourKey: nullable(r.tour_key),
			proxyKey: nullable(r.proxy_key),
			panoPrefix: nullable(r.pano_prefix),
			panoCount: Number(r.pano_count ?? 0),
			publishedAt: str(r.published_at),
		}));
		return Response.json({ scenes });
	} catch (err) {
		const message = err instanceof Error ? err.message : "failed to load scenes";
		return Response.json({ error: message }, { status: 500 });
	}
}
