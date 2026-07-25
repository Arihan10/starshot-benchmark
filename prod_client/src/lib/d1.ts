// Server-only Cloudflare D1 client. Talks to the D1 HTTP query endpoint with the
// secret API token, so this module must never be imported into client code —
// only the `/api/scenes` Route Handler uses it. The token stays server-side
// because it is NOT prefixed with NEXT_PUBLIC_.
const API_BASE = "https://api.cloudflare.com/client/v4";

type D1Row = Record<string, unknown>;

export async function d1Query(sql: string, params: unknown[] = []): Promise<D1Row[]> {
	const account = process.env.CLOUDFLARE_ACCOUNT_ID;
	const token = process.env.CLOUDFLARE_API_TOKEN;
	const databaseId = process.env.D1_DATABASE_ID;
	if (!account || !token || !databaseId) {
		throw new Error(
			"D1 access needs CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN and D1_DATABASE_ID in the environment",
		);
	}

	const res = await fetch(`${API_BASE}/accounts/${account}/d1/database/${databaseId}/query`, {
		method: "POST",
		headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
		body: JSON.stringify({ sql, params }),
		cache: "no-store",
	});

	const data = await res.json().catch(() => null);
	if (!res.ok || !data?.success) {
		throw new Error(`D1 query failed (${res.status}): ${JSON.stringify(data?.errors ?? data)}`);
	}
	return data.result?.[0]?.results ?? [];
}
