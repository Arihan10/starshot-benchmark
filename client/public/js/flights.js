// The api log — a first-party request-log page styled after OpenRouter's Logs
// view (server: app/utils/flightlog.py, per-scene SQLite DBs). Left: the
// request table with an activity histogram, scoped to one run and paginated
// 100 at a time. Right: "Generation details" for the selected row — stat cards,
// overview/request key-values, a provider-response latency bar, and the exact
// system/user prompt + output (fetched lazily on selection).

import { api } from "./api.js";
import { state as appState, emit, on } from "./state.js";
import { el } from "./ui.js";

const PAGE = 100;
const MON = [
	"Jan",
	"Feb",
	"Mar",
	"Apr",
	"May",
	"Jun",
	"Jul",
	"Aug",
	"Sep",
	"Oct",
	"Nov",
	"Dec",
];

// First column time mode — request start, response end, or both — persisted so
// the choice sticks across sessions.
const TIMECOL_KEY = "starshot.flog.timeCol";
function loadTimeCol() {
	try {
		const v = localStorage.getItem(TIMECOL_KEY);
		if (v === "start" || v === "end" || v === "both") return v;
	} catch {
		/* private mode */
	}
	return "end";
}

const state = {
	open: false,
	es: null,
	run: null,
	rows: [],
	seen: new Set(),
	selectedKey: null,
	pageIndex: 0,
	cursors: [null], // cursors[i] fetches page i
	hasMore: false,
	loading: false,
	locating: false, // a scene→log jump is resolving a specific row
	sceneScope: null, // locked when the log was entered from one specific scene
	filters: {}, // user-controlled facet key -> Set(selected values)
	facetData: {}, // facet key -> [{value, count}] from the server
	histo: null, // { buckets, t0, t1, bucket_s } activity histogram
	histoAt: 0, // last histogram fetch (throttles live refreshes)
	timeCol: loadTimeCol(), // first column: "start" | "end" | "both"
	detailCollapsed: false,
};

// Attributes the log can be filtered on individually — server-side facets.
const FACETS = [
	{ key: "transport", label: "Transport" },
	{ key: "status", label: "Status" },
	{ key: "kind", label: "Kind" },
	{ key: "model", label: "Model" },
	{ key: "provider", label: "Provider" },
	{ key: "slot", label: "Scene", display: (v) => sceneName(v) },
	{ key: "zone_id", label: "Zone" },
	{ key: "step", label: "Step" },
	{ key: "key", label: "Key" },
];

function statusValue(r) {
	if (r.status != null) return String(r.status);
	return r.exc_type || "error";
}

// A row's value for a facet — used to test live rows against active filters.
function facetValue(key, r) {
	if (key === "status") return statusValue(r);
	return r[key] ?? null;
}

// --- formatting ------------------------------------------------------------------

function fmtDate(ts) {
	if (!ts) return "—";
	const d = new Date(ts * 1000);
	let h = d.getHours();
	const ap = h >= 12 ? "PM" : "AM";
	h = h % 12 || 12;
	return `${MON[d.getMonth()]} ${d.getDate()}, ${String(h).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")} ${ap}`;
}

// Precise wall-clock for the detail view — the exact start (request) / end
// (response) of a call, with seconds + milliseconds, e.g. "Jul 18, 05:38:12.481 PM".
function fmtTime(ts) {
	if (!ts) return "—";
	const d = new Date(ts * 1000);
	let h = d.getHours();
	const ap = h >= 12 ? "PM" : "AM";
	h = h % 12 || 12;
	const p = (n, w = 2) => String(n).padStart(w, "0");
	return `${MON[d.getMonth()]} ${d.getDate()}, ${p(h)}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)} ${ap}`;
}

// Local clock with seconds, no date — "11:19:02 PM".
function fmtClock(ts) {
	if (!ts) return "—";
	const d = new Date(ts * 1000);
	let h = d.getHours();
	const ap = h >= 12 ? "PM" : "AM";
	h = h % 12 || 12;
	const p = (n) => String(n).padStart(2, "0");
	return `${p(h)}:${p(d.getMinutes())}:${p(d.getSeconds())} ${ap}`;
}

// Date + clock with seconds — "Jul 18, 11:19:02 PM". Used for the both-times
// column, where second precision keeps sub-minute start/end pairs distinct.
function fmtStamp(ts) {
	if (!ts) return "—";
	const d = new Date(ts * 1000);
	return `${MON[d.getMonth()]} ${d.getDate()}, ${fmtClock(ts)}`;
}

const sameLocalDay = (a, b) => {
	const x = new Date(a * 1000);
	const y = new Date(b * 1000);
	return (
		x.getFullYear() === y.getFullYear() &&
		x.getMonth() === y.getMonth() &&
		x.getDate() === y.getDate()
	);
};

const fmtNum = (n) => (n == null ? "—" : Number(n).toLocaleString());

function fmtDur(ms) {
	if (ms == null) return "--";
	if (ms < 1000) return `${ms} ms`;
	if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
	const m = Math.floor(ms / 60_000);
	return `${m}m ${Math.round((ms % 60_000) / 1000)}s`;
}

function throughput(r) {
	if (!r.tokens_out || !r.flight_ms) return "--";
	return `${(r.tokens_out / (r.flight_ms / 1000)).toFixed(1)} tok/s`;
}

// Per-call USD cost. Stored on the flight row for the token-priced compat
// backends (Moonshot, Alibaba, …); null for OpenRouter calls, whose settled
// cost lands in the event log (`llm.cost`) and shows in the cost tracker, not
// the flight ledger — those read "--". Fine precision for the sub-cent norm.
function fmtCost(v) {
	if (v == null) return "--";
	return v >= 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(4)}`;
}

function prettyModel(id) {
	if (!id) return "?";
	const s = (id.includes("/") ? id.split("/").pop() : id)
		.replace(/[-_]/g, " ")
		.trim();
	return s
		.split(/\s+/)
		.map((w) => (/^[a-z]/.test(w) ? w[0].toUpperCase() + w.slice(1) : w))
		.join(" ");
}

function prettyProvider(r) {
	if (r.transport === "openrouter") return "OpenRouter";
	const host = (r.provider || "")
		.replace(/^api\./, "")
		.replace(/\.(ai|com|chat|run|io|net)$/, "");
	if (!host) return r.transport || "?";
	return host
		.split(/[.\-]/)
		.map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
		.join(" ");
}

function iconColor(s) {
	let h = 0;
	for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
	return `hsl(${h % 360} 52% 46%)`;
}

function icon(label) {
	const ch = (String(label).match(/[a-z0-9]/i) || ["?"])[0].toUpperCase();
	return el("span", {
		class: "flog-ico",
		style: `background:${iconColor(label)}`,
		text: ch,
	});
}

const rowKey = (r) => `${r.slot}\u0000${r.id}`;
const isErr = (r) => !r.ok && r.status !== 429;

// The first column, honoring the chosen mode: request start, response end, or
// both stacked (start over end). "Both" uses second precision — and drops the
// end's date when it shares the start's day — so the span reads clearly.
function firstColCell(r) {
	if (state.timeCol === "both") {
		const sameDay =
			r.t_request && r.t_response && sameLocalDay(r.t_request, r.t_response);
		const endTxt = sameDay ? fmtClock(r.t_response) : fmtStamp(r.t_response);
		return el(
			"div",
			{ class: "flog-date flog-date-both" },
			el("span", {
				class: "flog-ts-a",
				text: fmtStamp(r.t_request),
				title: `Started ${fmtTime(r.t_request)}`,
			}),
			el("span", {
				class: "flog-ts-b",
				text: `→ ${endTxt}`,
				title: `Responded ${fmtTime(r.t_response)}`,
			}),
		);
	}
	const ts = state.timeCol === "start" ? r.t_request : r.t_response;
	return el("div", { class: "flog-date", text: fmtDate(ts), title: fmtTime(ts) });
}

// --- view ------------------------------------------------------------------------

export function initFlights() {
	const view = document.getElementById("flights");
	const btn = document.getElementById("btn-flights");
	if (!view || !btn) return;

	// --- left: list ------------------------------------------------------------
	const runSel = el("select", {
		class: "flog-run",
		title: "which run's requests to show",
	});
	runSel.addEventListener("change", () => setRun(runSel.value));
	const liveDot = el("span", { class: "flog-live", title: "live tail" });
	const backBtn = el("button", {
		class: "flog-back",
		text: "← board",
		onclick: close,
	});

	const lhead = el(
		"div",
		{ class: "flog-lhead" },
		el(
			"div",
			{ class: "flog-lhead-top" },
			el(
				"div",
				{},
				el("div", { class: "flog-title", text: "Logs" }),
				el("div", {
					class: "flog-sub",
					text: "View your request logs and history.",
				}),
			),
			el("div", { class: "flog-ctl" }, liveDot, runSel, backBtn),
		),
		el(
			"div",
			{ class: "flog-tabs" },
			el("div", { class: "flog-tab active", text: "Generations" }),
		),
	);

	// --- filter bar (facet dropdowns) -----------------------------------------
	let openPopover = null;
	function closePopover() {
		if (openPopover) {
			openPopover.node.remove();
			openPopover = null;
		}
	}

	function openFacetPopover(def, wrap, syncBtn) {
		closePopover();
		const sel = filterSet(def.key);
		const search = el("input", {
			class: "flog-pop-search",
			type: "text",
			placeholder: `Filter ${def.label.toLowerCase()}…`,
		});
		const list = el("div", { class: "flog-pop-list" });
		function renderList() {
			const values = state.facetData[def.key] || [];
			const q = search.value.trim().toLowerCase();
			const label = (v) => (def.display ? def.display(v) : String(v));
			list.textContent = "";
			const shown = q
				? values.filter((x) => label(x.value).toLowerCase().includes(q))
				: values;
			if (!shown.length) {
				list.append(el("div", { class: "flog-pop-empty", text: "no values" }));
				return;
			}
			for (const { value, count } of shown) {
				const on = sel.has(value);
				const cb = el("input", { type: "checkbox", ...(on ? { checked: "" } : {}) });
				const item = el(
					"label",
					{ class: `flog-pop-item${on ? " on" : ""}` },
					cb,
					el("span", { class: "flog-pop-val", text: label(value), title: String(value) }),
					el("span", { class: "flog-pop-count", text: String(count) }),
				);
				cb.addEventListener("change", () => {
					if (cb.checked) sel.add(value);
					else sel.delete(value);
					item.classList.toggle("on", cb.checked);
					syncBtn();
					applyFilters();
				});
				list.append(item);
			}
		}
		search.addEventListener("input", renderList);
		const clear = el("button", { class: "flog-pop-clear", text: "Clear" });
		clear.addEventListener("click", () => {
			sel.clear();
			syncBtn();
			renderList();
			applyFilters();
		});
		const node = el(
			"div",
			{ class: "flog-pop" },
			search,
			list,
			el("div", { class: "flog-pop-foot" }, el("span", { class: "flog-pop-lab", text: def.label }), clear),
		);
		node._wrap = wrap;
		node.addEventListener("click", (ev) => ev.stopPropagation());
		wrap.append(node);
		openPopover = { node, _wrap: wrap, renderList };
		renderList();
		search.focus();
	}

	function buildFacet(def) {
		const badge = el("span", { class: "flog-facet-badge" });
		const btnf = el(
			"button",
			{ class: "flog-facet" },
			el("span", { text: def.label }),
			badge,
			el("span", { class: "flog-facet-caret", text: "▾" }),
		);
		const wrap = el("div", { class: "flog-facet-wrap" }, btnf);
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
		return { key: def.key, wrap, sync };
	}

	const facetControls = FACETS.map(buildFacet);
	const clearAllBtn = el("button", { class: "flog-clear-all", title: "Reset every filter" });
	clearAllBtn.addEventListener("click", () => {
		state.filters = {};
		for (const fc of facetControls) fc.sync();
		closePopover();
		applyFilters();
	});
	function syncClearAll() {
		const active = FACETS.filter((f) => filterSet(f.key).size > 0).length;
		clearAllBtn.style.display = active ? "" : "none";
		clearAllBtn.textContent = `Clear (${active})`;
	}
	syncClearAll();
	const filtersBar = el(
		"div",
		{ class: "flog-filters" },
		...facetControls.map((fc) => fc.wrap),
		clearAllBtn,
	);

	const chartEl = el("div", { class: "flog-chart" });
	const timeColSel = el("select", {
		class: "flog-timecol",
		title: "first column: request start, response end, or both",
	});
	for (const [val, label] of [
		["start", "Start"],
		["end", "End"],
		["both", "Start + End"],
	])
		timeColSel.append(el("option", { value: val, text: label }));
	timeColSel.value = state.timeCol;
	timeColSel.addEventListener("change", () => setTimeCol(timeColSel.value));
	const thead = el(
		"div",
		{ class: "flog-thead" },
		timeColSel,
		el("div", { text: "Model" }),
		el("div", { text: "Provider" }),
		el("div", { text: "Step" }),
		el("div", { class: "flog-scene-col", text: "Scene" }),
		el("div", { text: "Zone" }),
		el("div", { class: "flog-time", text: "Time" }),
	);
	const rowsEl = el("div", { class: "flog-rows" });

	const pageLab = el("span", { class: "flog-page-lab" });
	const prevBtn = el("button", {
		class: "flog-arrow",
		text: "←",
		title: "newer",
		onclick: () => nav(-1),
	});
	const nextBtn = el("button", {
		class: "flog-arrow",
		text: "→",
		title: "older",
		onclick: () => nav(1),
	});
	const pager = el("div", { class: "flog-pager" }, pageLab, prevBtn, nextBtn);

	const listEl = el(
		"div",
		{ class: "flog-list" },
		lhead,
		filtersBar,
		chartEl,
		thead,
		rowsEl,
		pager,
	);
	const detailCollapseBtn = el("button", {
		class: "flog-detail-collapse",
		onclick: toggleDetail,
	});
	const detailBody = el("div", { class: "flog-detail-body" });
	const detailEl = el("aside", { class: "flog-detail" }, detailCollapseBtn, detailBody);
	view.append(listEl, detailEl);
	listEl.classList.toggle("tc-both", state.timeCol === "both");
	syncDetailCollapse();

	document.addEventListener("keydown", (ev) => {
		if (
			ev.key !== "Escape" ||
			!state.open ||
			document.getElementById("modal-root").firstChild
		)
			return;
		if (openPopover) closePopover();
		else close();
	});
	document.addEventListener("click", (ev) => {
		if (
			openPopover &&
			!openPopover.node.contains(ev.target) &&
			!openPopover._wrap.contains(ev.target)
		)
			closePopover();
	});

	// --- data ------------------------------------------------------------------

	async function loadPage() {
		if (!state.run) return;
		state.loading = true;
		renderPager();
		let res;
		try {
			res = await api.flightsPage(state.run, {
				cursor: state.cursors[state.pageIndex],
				limit: PAGE,
				filters: filtersParam(),
			});
		} catch (e) {
			state.loading = false;
			rowsEl.textContent = "";
			rowsEl.append(
				el("div", {
					class: "flog-empty",
					text: `failed to load: ${e.message}`,
				}),
			);
			return;
		}
		state.loading = false;
		state.rows = res.rows || [];
		state.seen = new Set(state.rows.map(rowKey));
		state.hasMore = !!res.has_more;
		state.cursors[state.pageIndex + 1] = res.cursor;
		renderRows();
		renderPager();
		autoSelect();
	}

	// --- filters --------------------------------------------------------------

	function filterSet(key) {
		return (state.filters[key] ??= new Set());
	}
	function filtersParam() {
		const out = {};
		for (const [k, s] of Object.entries(state.filters)) if (s.size) out[k] = [...s];
		if (state.sceneScope) out.slot = [state.sceneScope];
		return out;
	}
	function matches(r) {
		return (
			(!state.sceneScope || r.slot === state.sceneScope) &&
			Object.entries(state.filters).every(
				([k, s]) => !s.size || s.has(facetValue(k, r)),
			)
		);
	}
	let filterTimer = null;
	function applyFilters() {
		syncClearAll();
		clearTimeout(filterTimer);
		filterTimer = setTimeout(() => {
			state.pageIndex = 0;
			state.cursors = [null];
			loadPage();
			refreshFacets();
			refreshHistogram();
		}, 160);
	}
	async function refreshFacets() {
		if (!state.run) return;
		let res;
		try {
			res = await api.flightFacets(state.run, filtersParam());
		} catch {
			return;
		}
		state.facetData = res.facets || {};
		if (openPopover) openPopover.renderList();
	}
	async function refreshHistogram() {
		if (!state.run) return;
		state.histoAt = Date.now();
		try {
			state.histo = await api.flightHistogram(state.run, filtersParam());
		} catch {
			state.histo = null;
		}
		renderChart();
	}

	function nav(dir) {
		const target = state.pageIndex + dir;
		if (target < 0 || (dir > 0 && !state.hasMore) || state.loading) return;
		state.pageIndex = target;
		loadPage();
	}

	function autoSelect() {
		if (state.locating) return; // a scene→log jump will pick the exact row
		if (
			state.selectedKey &&
			state.rows.some((r) => rowKey(r) === state.selectedKey)
		) {
			markSelected();
			return;
		}
		if (state.rows.length) select(state.rows[0]);
		else deselect();
	}

	// --- scene ↔ log wiring ---------------------------------------------------

	// "Open in scene" from a flight's detail: hide the log (the scene overlay
	// sits above it) and ask the app to open that request's cell, focusing the
	// node the call produced. `r.slot` IS the cell/branch path.
	function openScene(r, { focus = null } = {}) {
		const scene = (r.slot || "").split("::generated::", 1)[0];
		const parts = scene.split("/");
		const run = parts[0];
		if (!run) return;
		const payload =
			parts[1] === "_branches"
				? { run, branch: parts.slice(2).join("/"), focus }
				: { run, slot: parts[1], model: parts.slice(2).join("/"), branch: false, focus };
		close();
		emit("open-cell-focus", payload);
	}

	// Scene→log: open the log to one exact call, resolving its row on the server
	// (by generation_id / t_request) so we can select it even if it's not on the
	// current page.
	function syncSceneScope() {
		view.classList.toggle("scene-scoped", !!state.sceneScope);
		const sceneFacet = facetControls.find((fc) => fc.key === "slot");
		if (sceneFacet) sceneFacet.wrap.style.display = state.sceneScope ? "none" : "";
	}

	async function openToFlight(payload) {
		if (!payload?.scene) return;
		state.sceneScope = payload.scene;
		syncSceneScope();
		closePopover();
		if (!state.open) {
			state.open = true;
			view.classList.add("open");
			await populateRuns();
		}
		syncBackLabel();
		state.locating = true;
		if (payload.run && payload.run !== state.run) {
			state.run = payload.run;
			runSel.value = payload.run;
		} else if (!state.run && payload.run) {
			state.run = payload.run;
			runSel.value = payload.run;
		}
		// The scene scope is locked separately from visible filters. Legacy logs
		// may not resolve the exact call, but still stay on the right scene.
		state.filters = {};
		for (const fc of facetControls) fc.sync();
		syncClearAll();
		refreshFacets();
		refreshHistogram();
		openStream();
		state.pageIndex = 0;
		state.cursors = [null];
		await loadPage();
		let row = null;
		try {
			row = await api.flightLocate(payload.scene, {
				generation_id: payload.generation_id ?? undefined,
				t_request: payload.t_request ?? undefined,
			});
		} catch {
			row = null;
		}
		state.locating = false;
		if (!row) {
			autoSelect();
			return;
		}
		if (!state.rows.some((x) => rowKey(x) === rowKey(row))) {
			state.rows.unshift(row);
			state.seen.add(rowKey(row));
			renderRows();
		}
		select(row);
		rowsEl
			.querySelector(`[data-k="${CSS.escape(rowKey(row))}"]`)
			?.scrollIntoView({ block: "center" });
	}
	on("open-flight", openToFlight);

	// --- render: list ----------------------------------------------------------

	// Whole-run activity from the server histogram (filter-aware), not just the
	// current page — same bars, broader picture.
	function renderChart() {
		chartEl.textContent = "";
		const buckets = state.histo?.buckets || [];
		if (!buckets.length) return;
		const mx = Math.max(1, ...buckets);
		for (const c of buckets) {
			chartEl.append(
				el("div", {
					class: "flog-bar",
					style: `height:${c ? Math.max(5, Math.round((c / mx) * 100)) : 2}%`,
					title: c ? `${c} request${c === 1 ? "" : "s"}` : "",
				}),
			);
		}
	}

	function rowEl(r) {
		const mLabel = prettyModel(r.model);
		const pLabel = prettyProvider(r);
		return el(
			"div",
			{
				class: `flog-row${isErr(r) ? " err" : ""}${rowKey(r) === state.selectedKey ? " selected" : ""}`,
				dataset: { k: rowKey(r) },
				onclick: () => select(r),
			},
			firstColCell(r),
			el(
				"div",
				{ class: "flog-mv" },
				icon(mLabel),
				el("span", { text: mLabel, title: r.model || "" }),
			),
			el(
				"div",
				{ class: "flog-mv" },
				icon(pLabel),
				el("span", { text: pLabel, title: r.base_url || "" }),
			),
			el("div", {
				class: "flog-step",
				text: r.step || "—",
				title: r.step || "",
			}),
			el("div", {
				class: "flog-app flog-scene-col",
				text: sceneName(r.slot),
				title: r.slot || "",
			}),
			el("div", {
				class: "flog-zone",
				text: r.zone_id || "—",
				title: r.zone_id || "",
			}),
			el("div", {
				class: "flog-time",
				text: fmtDur(r.flight_ms),
				title: r.flight_ms != null ? `${r.flight_ms} ms` : "",
			}),
		);
	}

	function renderRows() {
		rowsEl.textContent = "";
		if (!state.rows.length) {
			rowsEl.append(
				el("div", {
					class: "flog-empty",
					text: state.loading
						? "loading…"
						: "no requests recorded for this run yet",
				}),
			);
			return;
		}
		for (const r of state.rows) rowsEl.append(rowEl(r));
	}

	function setTimeCol(v) {
		state.timeCol = ["start", "end", "both"].includes(v) ? v : "end";
		try {
			localStorage.setItem(TIMECOL_KEY, state.timeCol);
		} catch {
			/* private mode */
		}
		listEl.classList.toggle("tc-both", state.timeCol === "both");
		renderRows();
	}

	function renderPager() {
		prevBtn.disabled = state.pageIndex === 0 || state.loading;
		nextBtn.disabled = !state.hasMore || state.loading;
		pageLab.textContent = state.loading
			? "loading…"
			: state.run
				? `page ${state.pageIndex + 1}`
				: "";
	}

	function markSelected() {
		for (const node of rowsEl.children) {
			node.classList?.toggle(
				"selected",
				node.dataset?.k === state.selectedKey,
			);
		}
	}

	// --- render: detail --------------------------------------------------------

	let detailToken = 0;

	function card(labelText, valueText, ico) {
		return el(
			"div",
			{ class: "flog-card" },
			el(
				"div",
				{ class: "flog-card-lab" },
				ico ? el("span", { text: ico }) : null,
				el("span", { text: labelText }),
			),
			el("div", { class: "flog-card-val", text: valueText }),
		);
	}
	function kv(k, v, { mono = false, link = false } = {}) {
		return el(
			"div",
			{ class: "flog-kv" },
			el("span", { class: "flog-kv-k", text: k }),
			el("span", {
				class: `flog-kv-v${mono ? " flog-mono" : ""}${link ? " flog-mono" : ""}`,
				text: v ?? "—",
				title: v == null ? "" : String(v),
			}),
		);
	}

	function fold(title, subtitle, { open = false } = {}) {
		const caret = el("span", { text: open ? "▾" : "▸" });
		const body = el("div", { class: "flog-fold-body" });
		const node = el(
			"div",
			{ class: `flog-fold${open ? " open" : ""}` },
			el(
				"div",
				{
					class: "flog-fold-h",
					onclick: () => {
						node.classList.toggle("open");
						caret.textContent = node.classList.contains("open")
							? "▾"
							: "▸";
					},
				},
				caret,
				el("span", { text: title }),
				subtitle
					? el("span", { class: "flog-fold-ct", text: subtitle })
					: null,
			),
			body,
		);
		return { node, body };
	}

	function select(r) {
		state.selectedKey = rowKey(r);
		markSelected();
		renderDetail(r);
	}

	function syncDetailCollapse() {
		detailEl.classList.toggle("collapsed", state.detailCollapsed);
		detailCollapseBtn.textContent = state.detailCollapsed ? "‹" : "›";
		const label = state.detailCollapsed
			? "Expand Generation details"
			: "Collapse Generation details";
		detailCollapseBtn.title = label;
		detailCollapseBtn.setAttribute("aria-label", label);
		detailCollapseBtn.setAttribute("aria-expanded", String(!state.detailCollapsed));
	}

	function toggleDetail() {
		state.detailCollapsed = !state.detailCollapsed;
		syncDetailCollapse();
	}

	function deselect() {
		state.selectedKey = null;
		markSelected();
		detailBody.textContent = "";
		detailBody.append(
			el("div", {
				class: "flog-ph",
				text: "Select a request to see its details.",
			}),
		);
	}

	function renderDetail(r) {
		const mLabel = prettyModel(r.model);
		const pLabel = prettyProvider(r);
		detailBody.textContent = "";
		for (const _k of [
			el(
				"div",
				{ class: "flog-d-head" },
				el("span", {
					class: "flog-d-title",
					text: "Generation details",
				}),
				el("button", {
					class: "flog-x",
					text: "✕",
					title: "close (Esc)",
					onclick: deselect,
				}),
			),
			el(
				"div",
				{ class: "flog-pills" },
				el(
					"span",
					{ class: "flog-pill" },
					icon(mLabel),
					el("span", { text: mLabel }),
				),
				el(
					"span",
					{ class: "flog-pill" },
					icon(pLabel),
					el("span", { text: pLabel }),
				),
			),
			el(
				"div",
				{ class: "flog-d-actions" },
				el("button", {
					class: "flog-goto",
					text: "Open in scene ↗",
					title: "open this request's cell in the 3D scene viewer",
					onclick: () => openScene(r),
				}),
				r.node
					? el("button", {
							class: "flog-goto",
							text: `Focus ${r.node} ↗`,
							title: "open the scene viewer and focus the node this call produced",
							onclick: () => openScene(r, { focus: r.node }),
						})
					: null,
			),
			el(
				"div",
				{ class: "flog-cards" },
				card("Provider latency", fmtDur(r.flight_ms), "◷"),
				card("Throughput", throughput(r), "⚡"),
				card("Cost", fmtCost(r.cost), "$"),
				card(
					"Tokens",
					r.tokens_in != null || r.tokens_out != null
						? `${fmtNum(r.tokens_in)} → ${fmtNum(r.tokens_out)}`
						: "--",
				),
				card("Fallbacks", "--", "⟳"),
				card("Fallback latency", "--"),
			),
			r.error
				? el("div", { class: "flog-err-box", text: r.error })
				: null,
			el("div", { class: "flog-sech", text: "Overview" }),
			kv("Model ID", r.model, { mono: true }),
			kv("Transport", r.transport),
			kv(
				"Endpoint",
				r.base_url ||
					(r.transport === "openrouter"
						? "openrouter.ai/api/v1"
						: "—"),
				{ mono: true },
			),
			kv("Kind", r.kind),
			el("div", { class: "flog-sech", text: "Request" }),
			kv("Requested", fmtTime(r.t_request)),
			kv("Responded", fmtTime(r.t_response)),
			kv("Duration", fmtDur(r.flight_ms)),
			kv("API Key", r.key || "—", { mono: true }),
			kv("Generation ID", r.generation_id || "—", { mono: true }),
			kv("Scene", r.slot, { mono: true }),
			r.zone_id ? kv("Zone", r.zone_id, { mono: true }) : null,
			r.step ? kv("Step", r.step) : null,
			r.node ? kv("Node", r.node) : null,
			kv(
				"Attempt",
				r.call != null
					? `call #${r.call} · try ${r.attempt}`
					: `try ${r.attempt ?? 1}`,
			),
			kv("Streaming", "false"),
			el("div", { class: "flog-sech", text: "Provider Responses" }),
			providerResponses(r),
		].filter(Boolean))
			detailBody.append(_k);

		// Collapsible prompt/output/reasoning (lazy) + raw JSON (immediate).
		const prompt = fold(
			"Prompt",
			r.tokens_in != null ? `${fmtNum(r.tokens_in)} tokens` : "",
		);
		const completion = fold(
			"Completion",
			r.tokens_out != null ? `${fmtNum(r.tokens_out)} tokens` : "",
		);
		const reasoning = fold("Reasoning", "");
		const raw = fold("Generation Data", "Raw JSON");
		raw.body.append(
			el("pre", { class: "flog-pre", text: JSON.stringify(r, null, 2) }),
		);
		for (const f of [prompt, completion, reasoning])
			f.body.append(el("div", { class: "flog-note", text: "loading…" }));
		detailBody.append(prompt.node, completion.node, reasoning.node, raw.node);

		loadPrompts(r, { prompt, completion, reasoning });
	}

	function providerResponses(r) {
		const wrap = el("div", {});
		const ok = r.ok;
		const badge = el("span", {
			class: `flog-badge ${ok ? "ok" : "err"}`,
			text: r.status != null ? String(r.status) : r.exc_type || "err",
		});
		wrap.append(
			el(
				"div",
				{ class: "flog-resp-row" },
				el(
					"div",
					{ class: "flog-resp-lab" },
					icon(prettyProvider(r)),
					el("span", { text: prettyProvider(r) }),
					badge,
				),
				el(
					"div",
					{ class: "flog-track" },
					el("div", {
						class: "flog-fill green",
						style: "width:100%",
					}),
				),
				el("div", { class: "flog-resp-t", text: fmtDur(r.flight_ms) }),
			),
		);
		if (r.tokens_out) {
			wrap.append(
				el(
					"div",
					{ class: "flog-resp-row" },
					el(
						"div",
						{ class: "flog-resp-lab" },
						el("span", { text: "Generation" }),
					),
					el(
						"div",
						{ class: "flog-track" },
						el("div", {
							class: "flog-fill blue",
							style: "width:100%",
							text: `${fmtNum(r.tokens_out)} tokens · ${throughput(r)}`,
						}),
					),
					el("div", {
						class: "flog-resp-t",
						text: fmtDur(r.flight_ms),
					}),
				),
			);
		}
		wrap.append(
			el("div", {
				class: "flog-total",
				text: `Total: ${fmtDur(r.flight_ms)}`,
			}),
		);
		return wrap;
	}

	async function loadPrompts(r, folds) {
		const token = ++detailToken;
		let data = null;
		try {
			data = await api.flightDetail(r.slot, r.id);
		} catch {
			data = null;
		}
		if (token !== detailToken) return; // superseded
		const fill = (f, parts, emptyMsg) => {
			f.body.textContent = "";
			const has = parts.some(([, v]) => v);
			if (!has) {
				f.body.append(
					el("div", { class: "flog-note", text: emptyMsg }),
				);
				return;
			}
			for (const [lab, v] of parts) {
				if (!v) continue;
				if (lab)
					f.body.append(
						el("div", { class: "flog-pre-lab", text: lab }),
					);
				f.body.append(el("pre", { class: "flog-pre", text: v }));
			}
		};
		const note = r.ok
			? "no prompt captured (pre-SQLite history, or served from cache)."
			: "this attempt failed — the prompt + output are on the retry that succeeded.";
		fill(
			folds.prompt,
			[
				["system", data?.system],
				["user", data?.user],
			],
			note,
		);
		fill(folds.completion, [["", data?.output]], note);
		if (data?.reasoning) {
			fill(folds.reasoning, [["", data.reasoning]], "");
		} else {
			folds.reasoning.node.remove();
		}
	}

	// --- live tail -------------------------------------------------------------

	function feed(row) {
		if (row.id == null || state.pageIndex !== 0 || !matches(row)) return;
		const k = rowKey(row);
		if (state.seen.has(k)) return;
		state.seen.add(k);
		state.rows.unshift(row);
		if (state.rows.length > PAGE) state.rows.pop();
		renderRows();
		markSelected();
		if (Date.now() - state.histoAt > 15_000) refreshHistogram();
	}

	function openStream() {
		state.es?.close();
		const es = new EventSource(api.flightsStreamUrl(state.run));
		state.es = es;
		es.onopen = () => liveDot.classList.add("on");
		es.onerror = () => liveDot.classList.remove("on");
		es.onmessage = (ev) => {
			try {
				feed(JSON.parse(ev.data));
			} catch {
				/* torn frame */
			}
		};
	}

	// --- run selection + lifecycle ---------------------------------------------

	async function populateRuns() {
		let runs = [];
		try {
			runs = (await api.runs()).runs || [];
		} catch {
			/* keep */
		}
		runSel.textContent = "";
		for (const r of runs)
			runSel.append(el("option", { value: r.name, text: r.name }));
		if (state.run) runSel.value = state.run;
	}

	function setRun(run) {
		if (!run || run === state.run) return;
		state.sceneScope = null;
		syncSceneScope();
		state.run = run;
		runSel.value = run;
		state.pageIndex = 0;
		state.cursors = [null];
		state.selectedKey = null;
		// Facet values are per-run, so a stale filter would over-restrict.
		state.filters = {};
		for (const fc of facetControls) fc.sync();
		syncClearAll();
		loadPage();
		refreshFacets();
		refreshHistogram();
		openStream();
	}

	async function open() {
		state.sceneScope = null;
		syncSceneScope();
		state.open = true;
		view.classList.add("open");
		syncBackLabel();
		await populateRuns();
		const run = state.run || appState.run || runSel.value;
		if (run) {
			state.run = run;
			runSel.value = run;
			state.pageIndex = 0;
			state.cursors = [null];
			loadPage();
			refreshFacets();
			refreshHistogram();
			openStream();
		} else {
			deselect();
			renderRows();
		}
	}

	function close() {
		state.open = false;
		view.classList.remove("open");
		state.es?.close();
		state.es = null;
		liveDot.classList.remove("on");
	}

	// The back button returns to whatever is stacked beneath the log: a scene
	// (kept loaded behind us) when one is open, otherwise the board.
	function syncBackLabel() {
		const hasScene = !!appState.view;
		backBtn.textContent = hasScene ? "← scene" : "← board";
		backBtn.title = hasScene
			? "return to the scene (stays loaded)"
			: "back to the board";
	}

	// Re-show the log without reloading it, so the scene's "← logs" return
	// lands back on the exact page and selection the user left behind.
	function reopen() {
		if (!state.run) {
			open();
			return;
		}
		state.open = true;
		view.classList.add("open");
		syncBackLabel();
		openStream();
	}
	on("open-flights", reopen);

	btn.addEventListener("click", () => (state.open ? close() : open()));
}

// The cell/branch name out of a "<run>/<slot>/<model>" path (the row's scene).
function sceneName(slot) {
	if (!slot) return "—";
	const raw = String(slot);
	const source = raw.split("::generated::", 1)[0];
	const p = source.split("/");
	const base = p.length >= 3 && p[1] === "_branches"
		? `branch:${p.slice(2).join("/")}`
		: p.length >= 3 ? p[1] : source;
	const version = raw.match(/::generated::v(.+)$/)?.[1];
	return version ? `${base} · generated v${version}` : base;
}
