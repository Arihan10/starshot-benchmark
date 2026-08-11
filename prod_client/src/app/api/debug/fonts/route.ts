import { NextResponse } from "next/server";
import { DEBUG_ENABLED } from "@/lib/flags";

export const dynamic = "force-dynamic";

type GoogleFamily = {
	family: string;
	category: string;
};

/**
 * Slim Google Fonts catalog for the debug picker — family + category only.
 * Sourced from fonts.google.com metadata (no API key).
 */
export async function GET() {
	if (!DEBUG_ENABLED) {
		return new NextResponse(null, { status: 404 });
	}

	const res = await fetch("https://fonts.google.com/metadata/fonts", {
		next: { revalidate: 86_400 },
		headers: { Accept: "application/json" },
	});
	if (!res.ok) {
		return NextResponse.json(
			{ error: `Google Fonts metadata failed (${res.status})` },
			{ status: 502 },
		);
	}

	let text = await res.text();
	if (text.startsWith(")]}'")) text = text.slice(4);

	const data = JSON.parse(text) as { familyMetadataList?: GoogleFamily[] };
	const fonts = (data.familyMetadataList ?? []).map((f) => ({
		family: f.family,
		category: f.category,
	}));

	return NextResponse.json({ fonts });
}
