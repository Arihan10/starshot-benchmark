// The api log — a first-party request-log page styled after OpenRouter's Logs
// view (server: app/utils/flightlog.py, per-scene SQLite DBs). Left: the
// request table with an activity histogram, scoped to one run and paginated
// 100 at a time. Right: "Generation details" for the selected row — stat cards,
// overview/request key-values, a provider-response latency bar, and the exact
// system/user prompt + output (fetched lazily on selection).

import { api } from "./api.js";
import { state as appState } from "./state.js";
import { el } from "./ui.js";

const PAGE = 100;
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

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
};

// --- formatting ------------------------------------------------------------------

function fmtDate(ts) {
	if (!ts) return "—";
	const d = new Date(ts * 1000);
	let h = d.getHours();
	const ap = h >= 12 ? "PM" : "AM";
	h = h % 12 || 12;
	return `${MON[d.getMonth()]} ${d.getDate()}, ${String(h).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")} ${ap}`;
}

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

function prettyModel(id) {
	if (!id) return "?";
	const s = (id.includes("/") ? id.split("/").pop() : id).replace(/[-_]/g, " ").trim();
	return s.split(/\s+/).map((w) => (/^[a-z]/.test(w) ? w[0].toUpperCase() + w.slice(1) : w)).join(" ");
}

function prettyProvider(r) {
	if (r.transport === "openrouter") return "OpenRouter";
	const host = (r.provider || "").replace(/^api\./, "").replace(/\.(ai|com|chat|run|io|net)$/, "");
	if (!host) return r.transport || "?";
	return host.split(/[.\-]/).map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w)).join(" ");
}

function iconColor(s) {
	let h = 0;
	for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
	return `hsl(${h % 360} 52% 46%)`;
}

function icon(label) {
	const ch = (String(label).match(/[a-z0-9]/i) || ["?"])[0].toUpperCase();
	return el("span", { class: "flog-ico", style: `background:${iconColor(label)}`, text: ch });
}

const rowKey = (r) => `${r.slot}\u0000${r.id}`;
const isErr = (r) => !r.ok && r.status !== 429;

// --- view ------------------------------------------------------------------------

export function initFlights() {
	const view = document.getElementById("flights");
	const btn = document.getElementById("btn-flights");
	if (!view || !btn) return;

	// --- left: list ------------------------------------------------------------
	const runSel = el("select", { class: "flog-run", title: "which run's requests to show" });
	runSel.addEventListener("change", () => setRun(runSel.value));
	const liveDot = el("span", { class: "flog-live", title: "live tail" });
	const backBtn = el("button", { class: "flog-back", text: "← board", onclick: close });

	const lhead = el("div", { class: "flog-lhead" },
		el("div", { class: "flog-lhead-top" },
			el("div", {},
				el("div", { class: "flog-title", text: "Logs" }),
				el("div", { class: "flog-sub", text: "View your request logs and history." }),
			),
			el("div", { class: "flog-ctl" }, liveDot, runSel, backBtn),
		),
		el("div", { class: "flog-tabs" }, el("div", { class: "flog-tab active", text: "Generations" })),
	);

	const chartEl = el("div", { class: "flog-chart" });
	const thead = el("div", { class: "flog-thead" },
		el("div", { text: "Date" }), el("div", { text: "Model" }),
		el("div", { text: "Provider" }), el("div", { text: "Scene" }),
	);
	const rowsEl = el("div", { class: "flog-rows" });

	const pageLab = el("span", { class: "flog-page-lab" });
	const prevBtn = el("button", { class: "flog-arrow", text: "←", title: "newer", onclick: () => nav(-1) });
	const nextBtn = el("button", { class: "flog-arrow", text: "→", title: "older", onclick: () => nav(1) });
	const pager = el("div", { class: "flog-pager" }, pageLab, prevBtn, nextBtn);

	const listEl = el("div", { class: "flog-list" }, lhead, chartEl, thead, rowsEl, pager);
	const detailEl = el("div", { class: "flog-detail" });
	view.append(listEl, detailEl);

	document.addEventListener("keydown", (ev) => {
		if (ev.key === "Escape" && state.open && !document.getElementById("modal-root").firstChild) close();
	});

	// --- data ------------------------------------------------------------------

	async function loadPage() {
		if (!state.run) return;
		state.loading = true;
		renderPager();
		let res;
		try {
			res = await api.flightsPage(state.run, { cursor: state.cursors[state.pageIndex], limit: PAGE });
		} catch (e) {
			state.loading = false;
			rowsEl.textContent = "";
			rowsEl.append(el("div", { class: "flog-empty", text: `failed to load: ${e.message}` }));
			return;
		}
		state.loading = false;
		state.rows = res.rows || [];
		state.seen = new Set(state.rows.map(rowKey));
		state.hasMore = !!res.has_more;
		state.cursors[state.pageIndex + 1] = res.cursor;
		renderRows();
		renderChart();
		renderPager();
		autoSelect();
	}

	function nav(dir) {
		const target = state.pageIndex + dir;
		if (target < 0 || (dir > 0 && !state.hasMore) || state.loading) return;
		state.pageIndex = target;
		loadPage();
	}

	function autoSelect() {
		if (state.selectedKey && state.rows.some((r) => rowKey(r) === state.selectedKey)) {
			markSelected();
			return;
		}
		if (state.rows.length) select(state.rows[0]);
		else deselect();
	}

	// --- render: list ----------------------------------------------------------

	function renderChart() {
		chartEl.textContent = "";
		const ts = state.rows.map((r) => r.t_response || 0).filter(Boolean);
		if (ts.length < 2) return;
		const min = Math.min(...ts), max = Math.max(...ts);
		const span = max - min || 1;
		const bins = 32;
		const counts = new Array(bins).fill(0);
		for (const t of ts) counts[Math.min(bins - 1, Math.floor(((t - min) / span) * bins))] += 1;
		const mx = Math.max(1, ...counts);
		for (const c of counts) {
			chartEl.append(el("div", { class: "flog-bar", style: `height:${Math.max(5, Math.round((c / mx) * 100))}%` }));
		}
	}

	function rowEl(r) {
		const mLabel = prettyModel(r.model);
		const pLabel = prettyProvider(r);
		return el("div", {
			class: `flog-row${isErr(r) ? " err" : ""}${rowKey(r) === state.selectedKey ? " selected" : ""}`,
			dataset: { k: rowKey(r) },
			onclick: () => select(r),
		},
			el("div", { class: "flog-date", text: fmtDate(r.t_response), title: fmtDate(r.t_response) }),
			el("div", { class: "flog-mv" }, icon(mLabel), el("span", { text: mLabel, title: r.model || "" })),
			el("div", { class: "flog-mv" }, icon(pLabel), el("span", { text: pLabel, title: r.base_url || "" })),
			el("div", { class: "flog-app", text: sceneName(r.slot), title: r.slot || "" }),
		);
	}

	function renderRows() {
		rowsEl.textContent = "";
		if (!state.rows.length) {
			rowsEl.append(el("div", { class: "flog-empty", text:
				state.loading ? "loading…" : "no requests recorded for this run yet" }));
			return;
		}
		for (const r of state.rows) rowsEl.append(rowEl(r));
	}

	function renderPager() {
		prevBtn.disabled = state.pageIndex === 0 || state.loading;
		nextBtn.disabled = !state.hasMore || state.loading;
		pageLab.textContent = state.loading ? "loading…" : state.run ? `page ${state.pageIndex + 1}` : "";
	}

	function markSelected() {
		for (const node of rowsEl.children) {
			node.classList?.toggle("selected", node.dataset?.k === state.selectedKey);
		}
	}

	// --- render: detail --------------------------------------------------------

	let detailToken = 0;

	function card(labelText, valueText, ico) {
		return el("div", { class: "flog-card" },
			el("div", { class: "flog-card-lab" }, ico ? el("span", { text: ico }) : null, el("span", { text: labelText })),
			el("div", { class: "flog-card-val", text: valueText }),
		);
	}
	function kv(k, v, { mono = false, link = false } = {}) {
		return el("div", { class: "flog-kv" },
			el("span", { class: "flog-kv-k", text: k }),
			el("span", { class: `flog-kv-v${mono ? " flog-mono" : ""}${link ? " flog-mono" : ""}`, text: v ?? "—", title: v == null ? "" : String(v) }),
		);
	}

	function fold(title, subtitle, { open = false } = {}) {
		const caret = el("span", { text: open ? "▾" : "▸" });
		const body = el("div", { class: "flog-fold-body" });
		const node = el("div", { class: `flog-fold${open ? " open" : ""}` },
			el("div", { class: "flog-fold-h",
				onclick: () => { node.classList.toggle("open"); caret.textContent = node.classList.contains("open") ? "▾" : "▸"; } },
				caret,
				el("span", { text: title }),
				subtitle ? el("span", { class: "flog-fold-ct", text: subtitle }) : null,
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

	function deselect() {
		state.selectedKey = null;
		markSelected();
		detailEl.textContent = "";
		detailEl.append(el("div", { class: "flog-ph", text: "Select a request to see its details." }));
	}

	function renderDetail(r) {
		const mLabel = prettyModel(r.model);
		const pLabel = prettyProvider(r);
		detailEl.textContent = "";
		for (const _k of [
			el("div", { class: "flog-d-head" },
				el("span", { class: "flog-d-title", text: "Generation details" }),
				el("button", { class: "flog-x", text: "✕", title: "close (Esc)", onclick: deselect }),
			),
			el("div", { class: "flog-pills" },
				el("span", { class: "flog-pill" }, icon(mLabel), el("span", { text: mLabel })),
				el("span", { class: "flog-pill" }, icon(pLabel), el("span", { text: pLabel })),
			),
			el("div", { class: "flog-cards" },
				card("Provider latency", fmtDur(r.flight_ms), "◷"),
				card("Throughput", throughput(r), "⚡"),
				card("Cost", "--", "$"),
				card("Tokens", r.tokens_in != null || r.tokens_out != null ? `${fmtNum(r.tokens_in)} → ${fmtNum(r.tokens_out)}` : "--"),
				card("Fallbacks", "--", "⟳"),
				card("Fallback latency", "--"),
			),
			r.error ? el("div", { class: "flog-err-box", text: r.error }) : null,
			el("div", { class: "flog-sech", text: "Overview" }),
			kv("Model ID", r.model, { mono: true }),
			kv("Transport", r.transport),
			kv("Endpoint", r.base_url || (r.transport === "openrouter" ? "openrouter.ai/api/v1" : "—"), { mono: true }),
			kv("Kind", r.kind),
			el("div", { class: "flog-sech", text: "Request" }),
			kv("API Key", r.key || "—", { mono: true }),
			kv("Generation ID", r.generation_id || "—", { mono: true }),
			kv("Scene", r.slot, { mono: true }),
			r.step ? kv("Step", r.step) : null,
			r.node ? kv("Node", r.node) : null,
			kv("Attempt", r.call != null ? `call #${r.call} · try ${r.attempt}` : `try ${r.attempt ?? 1}`),
			kv("Streaming", "false"),
			el("div", { class: "flog-sech", text: "Provider Responses" }),
			providerResponses(r),
		].filter(Boolean)) detailEl.append(_k);

		// Collapsible prompt/output/reasoning (lazy) + raw JSON (immediate).
		const prompt = fold("Prompt", r.tokens_in != null ? `${fmtNum(r.tokens_in)} tokens` : "");
		const completion = fold("Completion", r.tokens_out != null ? `${fmtNum(r.tokens_out)} tokens` : "");
		const reasoning = fold("Reasoning", "");
		const raw = fold("Generation Data", "Raw JSON");
		raw.body.append(el("pre", { class: "flog-pre", text: JSON.stringify(r, null, 2) }));
		for (const f of [prompt, completion, reasoning]) f.body.append(el("div", { class: "flog-note", text: "loading…" }));
		detailEl.append(prompt.node, completion.node, reasoning.node, raw.node);

		loadPrompts(r, { prompt, completion, reasoning });
	}

	function providerResponses(r) {
		const wrap = el("div", {});
		const ok = r.ok;
		const badge = el("span", { class: `flog-badge ${ok ? "ok" : "err"}`, text: r.status != null ? String(r.status) : (r.exc_type || "err") });
		wrap.append(el("div", { class: "flog-resp-row" },
			el("div", { class: "flog-resp-lab" }, icon(prettyProvider(r)), el("span", { text: prettyProvider(r) }), badge),
			el("div", { class: "flog-track" }, el("div", { class: "flog-fill green", style: "width:100%" })),
			el("div", { class: "flog-resp-t", text: fmtDur(r.flight_ms) }),
		));
		if (r.tokens_out) {
			wrap.append(el("div", { class: "flog-resp-row" },
				el("div", { class: "flog-resp-lab" }, el("span", { text: "Generation" })),
				el("div", { class: "flog-track" },
					el("div", { class: "flog-fill blue", style: "width:100%", text: `${fmtNum(r.tokens_out)} tokens · ${throughput(r)}` })),
				el("div", { class: "flog-resp-t", text: fmtDur(r.flight_ms) }),
			));
		}
		wrap.append(el("div", { class: "flog-total", text: `Total: ${fmtDur(r.flight_ms)}` }));
		return wrap;
	}

	async function loadPrompts(r, folds) {
		const token = ++detailToken;
		let data = null;
		try { data = await api.flightDetail(r.slot, r.id); } catch { data = null; }
		if (token !== detailToken) return; // superseded
		const fill = (f, parts, emptyMsg) => {
			f.body.textContent = "";
			const has = parts.some(([, v]) => v);
			if (!has) { f.body.append(el("div", { class: "flog-note", text: emptyMsg })); return; }
			for (const [lab, v] of parts) {
				if (!v) continue;
				if (lab) f.body.append(el("div", { class: "flog-pre-lab", text: lab }));
				f.body.append(el("pre", { class: "flog-pre", text: v }));
			}
		};
		const note = r.ok
			? "no prompt captured (pre-SQLite history, or served from cache)."
			: "this attempt failed — the prompt + output are on the retry that succeeded.";
		fill(folds.prompt, [["system", data?.system], ["user", data?.user]], note);
		fill(folds.completion, [["", data?.output]], note);
		if (data?.reasoning) {
			fill(folds.reasoning, [["", data.reasoning]], "");
		} else {
			folds.reasoning.node.remove();
		}
	}

	// --- live tail -------------------------------------------------------------

	function feed(row) {
		if (row.id == null || state.pageIndex !== 0) return;
		const k = rowKey(row);
		if (state.seen.has(k)) return;
		state.seen.add(k);
		state.rows.unshift(row);
		if (state.rows.length > PAGE) state.rows.pop();
		renderRows();
		renderChart();
		markSelected();
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
		try { runs = (await api.runs()).runs || []; } catch { /* keep */ }
		runSel.textContent = "";
		for (const r of runs) runSel.append(el("option", { value: r.name, text: r.name }));
		if (state.run) runSel.value = state.run;
	}

	function setRun(run) {
		if (!run || run === state.run) return;
		state.run = run;
		runSel.value = run;
		state.pageIndex = 0;
		state.cursors = [null];
		state.selectedKey = null;
		loadPage();
		openStream();
	}

	async function open() {
		state.open = true;
		view.classList.add("open");
		await populateRuns();
		const run = state.run || appState.run || runSel.value;
		if (run) {
			state.run = run;
			runSel.value = run;
			state.pageIndex = 0;
			state.cursors = [null];
			loadPage();
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

	btn.addEventListener("click", () => (state.open ? close() : open()));
}

// The cell/branch name out of a "<run>/<slot>/<model>" path (the row's scene).
function sceneName(slot) {
	if (!slot) return "—";
	const p = String(slot).split("/");
	if (p.length >= 3 && p[1] === "_branches") return `branch:${p.slice(2).join("/")}`;
	return p.length >= 3 ? p[1] : String(slot);
}
