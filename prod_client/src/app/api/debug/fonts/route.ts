export const dynamic = "force-dynamic";

const METADATA = "https://fonts.google.com/metadata/fonts";

export type FontFamily = {
	name: string;
	category: string;
	weights: number[];
	italics: number[];
};

type Metadata = {
	familyMetadataList: {
		family: string;
		category: string;
		popularity: number;
		fonts: Record<string, unknown>;
	}[];
};

let catalogue: Promise<FontFamily[]> | null = null;

function variants(fonts: Record<string, unknown>, italic: boolean): number[] {
	return Object.keys(fonts)
		.filter((key) => key.endsWith("i") === italic)
		.map((key) => Number.parseInt(key, 10))
		.filter((weight) => Number.isFinite(weight))
		.sort((a, b) => a - b);
}

async function load(): Promise<FontFamily[]> {
	const res = await fetch(METADATA, { cache: "no-store" });
	if (!res.ok) throw new Error(`google fonts metadata failed (${res.status})`);

	const { familyMetadataList } = (await res.json()) as Metadata;

	return familyMetadataList
		.sort((a, b) => a.popularity - b.popularity)
		.map((f) => ({
			name: f.family,
			category: f.category,
			weights: variants(f.fonts, false),
			italics: variants(f.fonts, true),
		}));
}

export async function GET() {
	if (process.env.NODE_ENV !== "development") {
		return new Response(null, { status: 404 });
	}

	try {
		catalogue ??= load();
		return Response.json({ families: await catalogue });
	} catch (err) {
		catalogue = null;
		const message = err instanceof Error ? err.message : "failed to load fonts";
		return Response.json({ error: message }, { status: 500 });
	}
}
