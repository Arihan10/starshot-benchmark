// The Logs view — styled after OpenRouter's Logs page: light theme, "Logs"
// header with tabs, a green activity histogram, a Date/Model/Provider table,
// and a "Generation details" side panel with stat cards, Overview/Request
// sections, provider-response latency bars, and collapsible Prompt/Completion.
//
// Data: the first-party SQLite flight ledger (app/utils/flightlog.py), scoped
// to one run (unified server-side via ATTACH). The list is keyset-paginated
// 100 at a time with server-side facet filters; prompt bytes are fetched only
// when a row's detail panel is opened.

import { api } from "./api.js";
import { state as appState } from "./state.js";
import { el } from "./ui.js";

const PAGE = 100;

const FACETS = [
	{ key: "transport", label: "transport" },
	{ key: "status", label: "status" },
	{ key: "kind", label: "kind" },
	{ key: "model", label: "model" },
	{ key: "provider", label: "provider" },
	{ key: "slot", label: "scene", display: (v) => shortSlotLabel(v) },
	{ key: "step", label: "step" },
	{ key: "key", label: "key" },
];

const state = {
	open: false,
	es: null,
	run: null,
	rows: [], // loaded rows, newest first
	seen: new Set(),
	cursor: null,
	hasMore: false,
	loading: false,
	filters: {}, // facet key -> Set(values)
	facetData: {},
	total: 0,
	histo: null,
	histoAt: 0,
	selectedKey: null,
};

// --- formatting ------------------------------------------------------------------

// "Jul 18, 05:38 PM" — OpenRouter's date column format.
function fmtDate(ts) {
	if (!ts) return "—";
	const d = new Date(ts * 1000);
	const day = d.toLocaleString("en-US", { month: "short", day: "2-digit" });
	const time = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: true });
	return `${day}, ${time}`;
}

const fmtStamp = (ts) => {
	if (!ts) return "—";
	const d = new Date(ts * 1000);
	return `${d.toLocaleTimeString([], { hour12: false })}.${String(d.getMilliseconds()).padStart(3, "0")}`;
};

function fmtFlight(ms) {
	if (ms == null) return "—";
	if (ms < 1000) return `${ms} ms`;
	if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
	const m = Math.floor(ms / 60_000);
	return `${m}m ${String(Math.round((ms % 60_000) / 1000)).padStart(2, "0")}s`;
}

const fmtTok = (n) => (n == null ? "?" : n.toLocaleString("en-US"));

function throughput(r) {
	if (!r.tokens_out || !r.flight_ms) return null;
	return `${(r.tokens_out / (r.flight_ms / 1000)).toFixed(1)} tok/s`;
}

// --- friendly names + avatar chips -------------------------------------------------

const ACRONYMS = new Set(["gpt", "glm", "hy3", "ai"]);

// "moonshotai/kimi-k3" -> "Kimi K3"; "google/gemini-3.1-flash-lite" ->
// "Gemini 3.1 Flash Lite" — the OpenRouter-style display name.
function friendlyModel(id) {
	const tail = String(id ?? "?").split("/").pop() || "?";
	return tail
		.split(/[-_ ]+/)
		.map((w) => (ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase()
			: /^[a-z]/.test(w) ? w[0].toUpperCase() + w.slice(1) : w))
		.join(" ");
}

const PROVIDER_NAMES = {
	"openrouter": "OpenRouter",
	"api.moonshot.ai": "Moonshot AI",
	"api.moonshot.cn": "Moonshot AI",
	"api.longcat.chat": "LongCat",
	"api.siliconflow.com": "SiliconFlow",
	"api.inceptionlabs.ai": "Inception",
};

function friendlyProvider(r) {
	if (r.transport === "openrouter") return "OpenRouter";
	const host = r.provider || "?";
	if (PROVIDER_NAMES[host]) return PROVIDER_NAMES[host];
	const core = host.replace(/^api\./, "").split(".")[0] || "?";
	return core[0].toUpperCase() + core.slice(1);
}

function hueOf(s) {
	let h = 0;
	for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
	return h % 360;
}

const avatar = (name) =>
	el("span", {
		class: "fl-ava",
		style: `background:hsl(${hueOf(name)} 60% 46%)`,
		text: (String(name)[0] || "?").toUpperCase(),
	});

function shortSlotLabel(slot) {
	if (!slot) return "Unknown";
	const p = String(slot).split("/");
	if (p.length >= 3 && p[1] === "_branches") return `branch:${p.slice(2).join("/")}`;
	return p.length >= 3 ? p[1] : String(slot);
}

function statusValue(r) {
	if (r.status != null) return String(r.status);
	return r.exc_type || "error";
}

function bucket(r) {
	if (r.ok) return "ok";
	if (r.status === 429) return "429";
	return "err";
}

function statusChip(r) {
	if (r.ok) return el("span", { class: "fl-chip ok", text: String(r.status ?? 200) });
	if (r.status === 429) return el("span", { class: "fl-chip warn", text: "429" });
	if (r.status != null) return el("span", { class: "fl-chip err", text: String(r.status) });
	return el("span", { class: "fl-chip err", text: r.exc_type || "error", title: r.error || "" });
}

const rowKey = (r) => `${r.slot}\u0000${r.id}`;

function facetValue(key, r) {
	if (key === "status") return statusValue(r);
	return r[key] ?? null;
}

// --- view ------------------------------------------------------------------------

export function initFlights() {
	const view = document.getElementById("flights");
	const btn = document.getElementById("btn-flights");
	if (!view || !btn) return;

	const runSel = el("select", { id: "fl-run", title: "which run's logs to show" });
	runSel.addEventListener("change", () => setRun(runSel.value));
	const liveDot = el("span", { class: "fl-live", title: "live tail" });
	const statsEl = el("span", { class: "fl-stats" });

	// --- facet popovers ---------------------------------------------------------

	let openPopover = null;
	function closePopover() {
		if (openPopover) { openPopover.node.remove(); openPopover = null; }
	}

	function openFacetPopover(def, wrap, syncBtn) {
		closePopover();
		const sel = filterSet(def.key);
		const search = el("input", { class: "fl-pop-search", type: "text", placeholder: `filter ${def.label}…` });
		const list = el("div", { class: "fl-pop-list" });
		function renderList() {
			const values = state.facetData[def.key] || [];
			const q = search.value.trim().toLowerCase();
			const label = (v) => (def.display ? def.display(v) : String(v));
			list.textContent = "";
			const shown = q ? values.filter((x) => label(x.value).toLowerCase().includes(q)) : values;
			if (!shown.length) {
				list.append(el("div", { class: "fl-pop-empty", text: "no values" }));
				return;
			}
			for (const { value, count } of shown) {
				const on = sel.has(value);
				const cb = el("input", { type: "checkbox", ...(on ? { checked: "" } : {}) });
				const item = el("label", { class: `fl-pop-item${on ? " on" : ""}` },
					cb,
					el("span", { class: "fl-pop-val", text: label(value), title: String(value) }),
					el("span", { class: "fl-pop-count", text: String(count) }),
				);
				cb.addEventListener("change", () => {
					if (cb.checked) sel.add(value); else sel.delete(value);
					item.classList.toggle("on", cb.checked);
					syncBtn();
					applyFilters();
				});
				list.append(item);
			}
		}
		search.addEventListener("input", renderList);
		const clear = el("button", { class: "fl-pop-clear", text: "clear" });
		clear.addEventListener("click", () => { sel.clear(); syncBtn(); renderList(); applyFilters(); });
		const node = el("div", { class: "fl-pop" },
			search,
			list,
			el("div", { class: "fl-pop-foot" }, el("span", { class: "fl-pop-lab", text: def.label }), clear),
		);
		node._wrap = wrap;
		node.addEventListener("click", (ev) => ev.stopPropagation());
		wrap.append(node);
		openPopover = { node, _wrap: wrap, renderList };
		renderList();
		search.focus();
	}

	function buildFacet(def) {
		const badge = el("span", { class: "fl-facet-badge" });
		const btnf = el("button", { class: "fl-facet" },
			el("span", { text: def.label }),
			badge,
			el("span", { class: "fl-facet-caret", text: "▾" }),
		);
		const wrap = el("div", { class: "fl-facet-wrap" }, btnf);
		function sync() {
			const n = filterSet(def.key).size;
			badge.textContent = n ? String(n) : "";
			wrap.classList.toggle("active", n > 0);
		}
		btnf.addEventListener("click", (ev) => {
			ev.stopPropagation();
			if (openPopover && openPopover._wrap === wrap) closePopover();
			else openFacetPopover(def, wrap, sync);
		});
		sync();
		return { wrap, sync };
	}

	const facetControls = FACETS.map(buildFacet);
	const clearAllBtn = el("button", { class: "fl-clear-all", title: "reset every filter" });
	clearAllBtn.addEventListener("click", () => {
		state.filters = {};
		for (const fc of facetControls) fc.sync();
		closePopover();
		syncClearAll();
		applyFilters();
	});
	function syncClearAll() {
		const active = FACETS.filter((f) => filterSet(f.key).size > 0).length;
		clearAllBtn.style.display = active ? "" : "none";
		clearAllBtn.textContent = `clear filters (${active})`;
	}

	// --- header: Logs title, tabs, toolbar ---------------------------------------

	const head = el("div", { id: "flights-head" },
		el("button", { id: "flights-close", text: "← board" }),
		el("div", { class: "fl-title-block" },
			el("div", { class: "fl-title", text: "Logs" }),
			el("div", { class: "fl-subtitle", text: "View your request logs and history." }),
		),
		el("span", { style: "margin-left:auto" }),
		runSel,
		liveDot,
	);

	const tabs = el("div", { class: "or-tabs" },
		el("button", { class: "or-tab active", text: "Generations" }),
		el("button", { class: "or-tab", disabled: true, text: "Upstream Requests", title: "not tracked in this benchmark" }),
		el("button", { class: "or-tab", disabled: true, text: "Sessions", title: "not tracked in this benchmark" }),
		el("button", { class: "or-tab", disabled: true, text: "Videos", title: "not tracked in this benchmark" }),
	);

	const filterBar = el("div", { id: "flights-filters" },
		...facetControls.map((fc) => fc.wrap), clearAllBtn,
		el("span", { style: "margin-left:auto" }),
		statsEl,
	);

	const chartEl = el("div", { class: "fl-chart" });

	const cols = el("div", { class: "fl-row fl-cols" },
		el("span", { text: "Date" }),
		el("span", { text: "Model" }),
		el("span", { text: "Provider" }),
		el("span", { text: "App" }),
		el("span", { text: "Tokens" }),
		el("span", { text: "Status" }),
	);
	const rowsEl = el("div", { id: "flights-rows" });
	const detailEl = el("div", { id: "flights-detail" });
	const main = el("div", { id: "flights-main" },
		el("div", { id: "flights-table" }, chartEl, cols, rowsEl),
		detailEl,
	);
	view.append(head, tabs, filterBar, main);

	rowsEl.addEventListener("scroll", () => {
		if (state.hasMore && !state.loading &&
			rowsEl.scrollTop + rowsEl.clientHeight >= rowsEl.scrollHeight - 240) {
			loadOlder();
		}
	});

	head.querySelector("#flights-close").addEventListener("click", close);
	document.addEventListener("keydown", (ev) => {
		if (ev.key !== "Escape" || !state.open || document.getElementById("modal-root").firstChild) return;
		if (openPopover) closePopover();
		else if (state.selectedKey != null) deselect();
		else close();
	});
	document.addEventListener("click", (ev) => {
		if (openPopover && !openPopover.node.contains(ev.target) && !openPopover._wrap.contains(ev.target)) {
			closePopover();
		}
	});

	// --- filters + data -----------------------------------------------------------

	function filterSet(key) {
		return (state.filters[key] ??= new Set());
	}
	function filtersParam() {
		const out = {};
		for (const [k, s] of Object.entries(state.filters)) if (s.size) out[k] = [...s];
		return out;
	}
	function matches(r) {
		return Object.entries(state.filters).every(([k, s]) => !s.size || s.has(facetValue(k, r)));
	}

	let applyTimer = null;
	function applyFilters() {
		syncClearAll();
		clearTimeout(applyTimer);
		applyTimer = setTimeout(() => { reload(); refreshFacets(); refreshHistogram(); }, 160);
	}

	async function reload() {
		if (!state.run) return;
		state.loading = true;
		state.rows = [];
		state.seen.clear();
		state.cursor = null;
		state.hasMore = false;
		render();
		try {
			const res = await api.flightsPage(state.run, { filters: filtersParam(), limit: PAGE });
			ingest(res, true);
		} catch (e) {
			state.loading = false;
			rowsEl.textContent = "";
			rowsEl.append(el("div", { class: "fl-empty", text: `failed to load: ${e.message}` }));
			return;
		}
		state.loading = false;
		render();
	}

	async function loadOlder() {
		if (!state.run || state.loading || !state.hasMore) return;
		state.loading = true;
		renderFooter();
		try {
			const res = await api.flightsPage(state.run, { cursor: state.cursor, filters: filtersParam(), limit: PAGE });
			ingest(res, false);
		} catch { /* transient — retried by scrolling */ }
		state.loading = false;
		render();
	}

	function ingest(res, reset) {
		if (reset) { state.rows = []; state.seen.clear(); }
		for (const r of res.rows || []) {
			const k = rowKey(r);
			if (state.seen.has(k)) continue;
			state.seen.add(k);
			state.rows.push(r);
		}
		state.cursor = res.cursor;
		state.hasMore = !!res.has_more;
		sortRows();
	}

	function sortRows() {
		state.rows.sort((a, b) =>
			(b.t_response || 0) - (a.t_response || 0) ||
			(a.slot < b.slot ? 1 : a.slot > b.slot ? -1 : 0) ||
			(b.id || 0) - (a.id || 0),
		);
	}

	async function refreshFacets() {
		if (!state.run) return;
		try {
			const res = await api.flightFacets(state.run, filtersParam());
			state.facetData = res.facets || {};
			state.total = res.total || 0;
		} catch { return; }
		renderStats();
		if (openPopover) openPopover.renderList();
	}

	async function refreshHistogram() {
		if (!state.run) return;
		state.histoAt = Date.now();
		try {
			state.histo = await api.flightHistogram(state.run, filtersParam());
		} catch { state.histo = null; }
		renderChart();
	}

	// --- chart ----------------------------------------------------------------

	function renderChart() {
		chartEl.textContent = "";
		const h = state.histo;
		if (!h || !h.buckets || !h.buckets.length) return;
		const max = Math.max(...h.buckets, 1);
		for (let i = 0; i < h.buckets.length; i++) {
			const c = h.buckets[i];
			const bar = el("span", { class: `fl-bar${c ? "" : " zero"}` });
			bar.style.height = c ? `${Math.max(9, (c / max) * 100)}%` : "2px";
			bar.title = `${c} request${c === 1 ? "" : "s"} · ${fmtDate(h.t0 + i * h.bucket_s)}`;
			chartEl.append(bar);
		}
	}

	// --- table ------------------------------------------------------------------

	function renderStats() {
		const sc = Object.fromEntries((state.facetData.status || []).map((x) => [String(x.value), x.count]));
		let ok = 0, limited = 0, err = 0;
		for (const [v, c] of Object.entries(sc)) {
			if (/^2\d\d$/.test(v)) ok += c;
			else if (v === "429") limited += c;
			else err += c;
		}
		statsEl.textContent = state.total
			? `${state.total.toLocaleString("en-US")} requests · ${ok.toLocaleString("en-US")} ok · ${limited} × 429 · ${err} err`
			: (state.run ? "no requests match" : "");
	}

	function rowEl(r) {
		const k = rowKey(r);
		const model = friendlyModel(r.model);
		const provider = friendlyProvider(r);
		return el("div",
			{
				class: `fl-row${k === state.selectedKey ? " selected" : ""}`,
				dataset: { k },
				onclick: () => select(r),
			},
			el("span", { class: "fl-date", text: fmtDate(r.t_response), title: fmtStamp(r.t_response) }),
			el("span", { class: "fl-cell", title: r.model ?? "" }, avatar(model), el("span", { class: "fl-cell-txt", text: model })),
			el("span", { class: "fl-cell", title: r.base_url || "" }, avatar(provider), el("span", { class: "fl-cell-txt", text: provider })),
			el("span", {
				class: `fl-app${r.slot ? "" : " unknown"}`,
				text: shortSlotLabel(r.slot),
				title: [r.slot, r.step, r.node].filter(Boolean).join(" · "),
			}),
			el("span", { class: "fl-tokens", text: r.ok ? `${fmtTok(r.tokens_in)} → ${fmtTok(r.tokens_out)}` : "" }),
			statusChip(r),
		);
	}

	const footEl = el("div", { class: "fl-more" });
	function renderFooter() {
		footEl.textContent = "";
		if (state.loading) {
			footEl.append(el("div", { class: "fl-more-msg", text: "loading…" }));
		} else if (state.hasMore) {
			footEl.append(el("button", { class: "fl-more-btn", text: `Load ${PAGE} older`, onclick: loadOlder }));
		} else if (state.rows.length) {
			footEl.append(el("div", { class: "fl-more-msg", text: `end · ${state.rows.length} shown` }));
		}
	}

	let renderQueued = false;
	function scheduleRender() {
		if (renderQueued) return;
		renderQueued = true;
		setTimeout(() => { renderQueued = false; if (state.open) render(); }, 120);
	}

	function render() {
		renderStats();
		const atTop = rowsEl.scrollTop <= 4;
		const prevH = rowsEl.scrollHeight;
		const prevTop = rowsEl.scrollTop;
		rowsEl.textContent = "";
		if (!state.rows.length) {
			rowsEl.append(el("div", { class: "fl-empty", text:
				state.loading ? "loading…"
					: Object.values(state.filters).some((s) => s.size) ? "No requests match the current filters."
					: "No requests recorded for this run yet." }));
			return;
		}
		for (const r of state.rows) rowsEl.append(rowEl(r));
		renderFooter();
		rowsEl.append(footEl);
		if (!atTop) rowsEl.scrollTop = prevTop + (rowsEl.scrollHeight - prevH);
	}

	// --- Generation details panel -------------------------------------------------

	let detailToken = 0;

	function card(label, value, { title = "" } = {}) {
		return el("div", { class: "fl-card" },
			el("div", { class: "fl-card-lab", text: label }),
			el("div", { class: "fl-card-val", text: value ?? "--", title }),
		);
	}

	function kv(key, value, { mono = true, dim = false } = {}) {
		return el("div", { class: "fl-kv" },
			el("span", { class: "fl-kv-k", text: key }),
			el("span", { class: `fl-kv-v${mono ? " mono" : ""}${dim ? " dim" : ""}`,
				text: value ?? "—", title: value == null ? "" : String(value) }),
		);
	}

	function section(title, ...children) {
		return el("div", { class: "fl-d-sec" },
			el("div", { class: "fl-d-sec-title", text: title }),
			...children,
		);
	}

	function collapsible(title, meta, content, { open = false, mono = true } = {}) {
		const caret = el("span", { class: "fl-sec-caret", text: "›" });
		const sec = el("div", { class: `fl-sec${open ? " open" : ""}` });
		sec.append(
			el("div", { class: "fl-sec-head",
				onclick: () => sec.classList.toggle("open") },
				caret,
				el("span", { class: "fl-sec-title", text: title }),
				meta ? el("span", { class: "fl-sec-meta", text: meta }) : null,
			),
			el("pre", { class: `fl-pre${mono ? "" : " sans"}`, text: content || "—" }),
		);
		return sec;
	}

	// Latency bars for every loaded attempt of the same logical call — the
	// "Provider Responses" module (a 429 sweep shows one bar per key tried).
	function responseBars(r) {
		let attempts = [r];
		if (r.call != null) {
			attempts = state.rows
				.filter((x) => x.slot === r.slot && x.call === r.call)
				.sort((a, b) => (a.attempt || 0) - (b.attempt || 0));
			if (!attempts.length) attempts = [r];
		}
		const total = attempts.reduce((s, a) => s + (a.flight_ms || 0), 0) || 1;
		const wrap = el("div", { class: "fl-bars" });
		for (const a of attempts) {
			const frac = Math.max(0.04, (a.flight_ms || 0) / total);
			const cls = a.ok ? "ok" : bucket(a) === "429" ? "warn" : "err";
			wrap.append(el("div", { class: "fl-bar-row" },
				el("span", { class: "fl-bar-lab", text: friendlyProvider(a), title: a.base_url || "" }),
				statusChip(a),
				el("span", { class: "fl-bar-track" },
					el("span", { class: `fl-bar-fill ${cls}`, style: `width:${(frac * 100).toFixed(1)}%`,
						text: a.ok && a.tokens_out ? `${fmtTok(a.tokens_out)} tokens` : "" }),
				),
				el("span", { class: "fl-bar-time", text: fmtFlight(a.flight_ms) }),
			));
		}
		if (attempts.length > 1) {
			wrap.append(el("div", { class: "fl-bar-total", text: `Total: ${fmtFlight(total)}` }));
		}
		return wrap;
	}

	async function loadPrompts(r, container) {
		const token = ++detailToken;
		container.textContent = "";
		container.append(el("div", { class: "fl-d-note", text: "loading prompt + completion…" }));
		let data = null;
		try { data = await api.flightDetail(r.slot, r.id); } catch { data = null; }
		if (token !== detailToken) return;
		container.textContent = "";
		if (!data || (!data.system && !data.user && !data.output)) {
			container.append(el("div", { class: "fl-d-note", text: r.ok
				? "No prompt captured for this request (pre-SQLite history, or served from cache)."
				: "This attempt failed — the prompt + completion live on the retry that succeeded." }));
			return;
		}
		container.append(
			collapsible("Prompt", r.tokens_in != null ? `${fmtTok(r.tokens_in)} tokens` : `${(data.user || "").length} chars`, data.user ?? ""),
			collapsible("System", `${(data.system || "").length} chars`, data.system ?? ""),
			collapsible("Completion", r.tokens_out != null ? `${fmtTok(r.tokens_out)} tokens` : null, data.output ?? "", { open: true }),
			data.reasoning ? collapsible("Reasoning", `${data.reasoning.length} chars`, data.reasoning) : null,
			data.schema ? el("div", { class: "fl-d-schema", text: `schema · ${data.schema}` }) : null,
		);
	}

	function renderDetail(r) {
		const model = friendlyModel(r.model);
		const provider = friendlyProvider(r);
		const parts = (r.slot || "").split("/");
		const isBranch = parts.length >= 3 && parts[1] === "_branches";
		detailEl.textContent = "";
		const promptsEl = el("div", { class: "fl-d-prompts" });
		detailEl.append(
			el("div", { class: "fl-d-head" },
				el("div", { class: "fl-d-head-top" },
					el("span", { class: "fl-d-title", text: "Generation details" }),
					el("button", { class: "fl-d-close", text: "✕", title: "close (Esc)", onclick: deselect }),
				),
				el("div", { class: "fl-d-chips" },
					el("span", { class: "fl-mchip" }, avatar(model), el("span", { text: model })),
					el("span", { class: "fl-mchip" }, avatar(provider), el("span", { text: provider })),
				),
			),
			el("div", { class: "fl-d-body" },
				el("div", { class: "fl-cards" },
					card("Provider latency", fmtFlight(r.flight_ms)),
					card("Throughput", throughput(r) ?? "--"),
					card("Cost", "--", { title: "settled costs live on the run's spend tracker" }),
					card("Tokens", r.tokens_in != null || r.tokens_out != null
						? `${fmtTok(r.tokens_in)} → ${fmtTok(r.tokens_out)}` : "--"),
					card("Attempt", r.attempt != null ? `#${r.attempt}${r.call != null ? ` of call ${r.call}` : ""}` : "--"),
					card("API key", r.key ?? "--"),
				),
				section("Overview",
					kv("Model ID", r.model),
					kv("Kind", r.kind, { mono: false }),
					kv("Transport", r.transport, { mono: false }),
					kv("Scene", r.slot ?? "Unknown"),
					isBranch ? kv("Branch", parts.slice(2).join("/")) : null,
					r.step ? kv("Step", r.step) : null,
					r.node ? kv("Node", r.node) : null,
				),
				section("Request",
					kv("Request at", fmtStamp(r.t_request), { mono: false }),
					kv("Response at", fmtStamp(r.t_response), { mono: false }),
					kv("Endpoint", r.base_url || (r.transport === "openrouter" ? "openrouter.ai/api/v1" : "—")),
					r.generation_id ? kv("Generation ID", r.generation_id) : null,
					kv("Status", statusValue(r), { mono: false }),
				),
				r.error ? el("pre", { class: "fl-d-err", text: r.error }) : null,
				section("Provider Responses", responseBars(r)),
				promptsEl,
			),
			el("div", { class: "fl-d-nav" },
				el("button", { class: "fl-nav-btn", text: "←", title: "previous request", onclick: () => step(1) }),
				el("button", { class: "fl-nav-btn", text: "→", title: "next request", onclick: () => step(-1) }),
			),
		);
		loadPrompts(r, promptsEl);
	}

	// Move the selection up/down the visible (newest-first) list.
	function step(dir) {
		if (state.selectedKey == null) return;
		const idx = state.rows.findIndex((x) => rowKey(x) === state.selectedKey);
		const next = state.rows[idx + dir];
		if (next) select(next);
	}

	function markSelected() {
		for (const node of rowsEl.children) {
			node.classList?.toggle("selected", node.dataset?.k === state.selectedKey);
		}
	}

	function select(r) {
		const k = rowKey(r);
		if (state.selectedKey === k) { deselect(); return; }
		state.selectedKey = k;
		markSelected();
		detailEl.classList.add("open");
		renderDetail(r);
	}

	function deselect() {
		state.selectedKey = null;
		detailEl.classList.remove("open");
		detailEl.textContent = "";
		markSelected();
	}

	// --- live tail -----------------------------------------------------------

	function feed(row) {
		if (row.id == null) return;
		const k = rowKey(row);
		if (state.seen.has(k) || !matches(row)) return;
		state.seen.add(k);
		state.rows.push(row);
		state.total += 1;
		sortRows();
		scheduleRender();
		if (Date.now() - state.histoAt > 15_000) refreshHistogram();
	}

	function openStream() {
		state.es?.close();
		const es = new EventSource(api.flightsStreamUrl(state.run));
		state.es = es;
		es.onopen = () => liveDot.classList.add("on");
		es.onerror = () => liveDot.classList.remove("on");
		es.onmessage = (ev) => { try { feed(JSON.parse(ev.data)); } catch { /* torn frame */ } };
	}

	// --- run selection + lifecycle ---------------------------------------------

	async function populateRuns() {
		let runs = [];
		try { runs = (await api.runs()).runs || []; } catch { /* keep current */ }
		runSel.textContent = "";
		for (const r of runs) runSel.append(el("option", { value: r.name, text: r.name }));
		if (state.run) runSel.value = state.run;
	}

	function setRun(run) {
		if (!run || run === state.run) return;
		deselect();
		closePopover();
		state.run = run;
		runSel.value = run;
		reload();
		refreshFacets();
		refreshHistogram();
		openStream();
	}

	async function open() {
		state.open = true;
		view.classList.add("open");
		await populateRuns();
		const run = state.run || appState.run || runSel.value;
		if (run && run !== state.run) {
			state.run = run;
			runSel.value = run;
		}
		if (state.run) {
			reload();
			refreshFacets();
			refreshHistogram();
			openStream();
		} else {
			render();
		}
	}

	function close() {
		state.open = false;
		view.classList.remove("open");
		closePopover();
		state.es?.close();
		state.es = null;
		liveDot.classList.remove("on");
	}

	btn.addEventListener("click", () => (state.open ? close() : open()));
}
