// Global mesh-generation queue panel — a live view of every in-flight + waiting
// generation across the concurrency pools (Trellis + Hunyuan on Modal, and
// Hunyuan 3.1 on Tencent), polled from /trellis/queue. The queue is
// process-global (it spans every cell + branch), not one cell's data: a
// from-scratch generate fans a whole scene onto it, and per-asset regenerations
// surface as waiting → processing rows.
//
// It floats at the bottom ONLY while the main scene canvas (the cell overlay) is
// up; the board and the prompt lab hide it (CSS, keyed off `#overlay.open`),
// where a mesh queue is just noise. It keeps polling while hidden, so the count
// is already current the moment the overlay opens. Horizontally it's pinned just
// left of the observability dock (see syncPosition below) so it never overlaps
// the obs tree / investigator. Tuck it off-screen via the ▼ control (or the
// bottom pull tab when tucked) — preference persists in localStorage.

import { api } from "./api.js";
import { el } from "./ui.js";

const OPEN_KEY = "starshot.queueOpen";
const DOCKED_KEY = "starshot.queueDocked";
const POLL_BUSY_MS = 1500; // while anything is in flight
const POLL_IDLE_MS = 5000; // empty queue — a slow background heartbeat

export function initQueuePanel() {
	let open = false;
	let docked = true;
	try {
		open = localStorage.getItem(OPEN_KEY) === "1";
		const dockedRaw = localStorage.getItem(DOCKED_KEY);
		docked = dockedRaw == null ? !open : dockedRaw === "1";
	} catch {
		/* private mode */
	}

	const countEl = el("span", { class: "q-count" });
	const tabCountEl = el("span", { class: "q-count" });
	const caretEl = el("span", { class: "q-caret", text: open ? "▾" : "▸" });
	const dockBtn = el("span", {
		class: "q-dock",
		text: "▼",
		title: "Tuck the queue off-screen",
	});
	const head = el(
		"div",
		{ id: "queue-head", title: "every mesh generation currently in flight or waiting" },
		caretEl,
		el("span", { class: "q-title", text: "queue" }),
		countEl,
		dockBtn,
	);
	const body = el("div", { id: "queue-body" });
	const inner = el("div", { id: "queue-inner" }, head, body);
	const tab = el(
		"div",
		{
			id: "queue-tab",
			title: "Show mesh generation queue",
		},
		el("span", { class: "q-tab-caret", text: "▲" }),
		el("span", { class: "q-title", text: "queue" }),
		tabCountEl,
	);
	const panel = el("div", { id: "queue-panel" }, inner, tab);
	panel.classList.toggle("open", open);
	panel.classList.toggle("docked-away", docked);
	document.body.appendChild(panel);

	const setDocked = (next) => {
		docked = next;
		panel.classList.toggle("docked-away", docked);
		try {
			localStorage.setItem(DOCKED_KEY, docked ? "1" : "0");
		} catch {
			/* private mode */
		}
	};
	tab.addEventListener("click", () => setDocked(false));
	dockBtn.addEventListener("click", (ev) => {
		ev.stopPropagation();
		setDocked(true);
	});

	// Pin the panel just LEFT of the observability dock (and the investigator
	// column when it's open) so it never overlaps the obs tree or anything else
	// docked on the right. The dock is user-resizable and the investigator
	// toggles, so recompute `right` from the live right-column widths, summed via
	// offsetWidth — which, unlike getBoundingClientRect on the slide-transformed
	// overlay, is independent of the overlay's open/close transform. #canvas-host
	// is the flex:1 filler, so it reflows on every dock resize / investigator
	// toggle / window resize, which drives the recompute.
	const RIGHT_GAP = 12;
	const canvasHost = document.getElementById("canvas-host");
	const rightCols = [
		document.getElementById("obsdock-resizer"),
		document.getElementById("obsdock"),
		document.getElementById("investigator-resizer"),
		document.getElementById("investigator"),
	];
	const syncPosition = () => {
		const rightWidth = rightCols.reduce((w, n) => w + (n ? n.offsetWidth : 0), 0);
		panel.style.right = `${rightWidth + RIGHT_GAP}px`;
	};
	syncPosition();
	if (canvasHost && "ResizeObserver" in window) {
		new ResizeObserver(syncPosition).observe(canvasHost);
	}
	window.addEventListener("resize", syncPosition);

	head.addEventListener("click", () => {
		open = !open;
		panel.classList.toggle("open", open);
		caretEl.textContent = open ? "▾" : "▸";
		try {
			localStorage.setItem(OPEN_KEY, open ? "1" : "0");
		} catch {
			/* private mode */
		}
	});

	// Total time a row has been in flight, from the server's first-seen epoch
	// (`enqueued_at`). Ticked locally each second so it advances between the
	// ~1.5s polls; server + browser share the host clock, so no skew handling.
	function fmtElapsed(sinceSec) {
		if (!sinceSec) return "";
		const s = Math.max(0, Math.floor(Date.now() / 1000 - sinceSec));
		if (s < 60) return `${s}s`;
		return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
	}

	function updateElapsed() {
		for (const span of body.querySelectorAll(".q-elapsed[data-since]")) {
			span.textContent = fmtElapsed(Number(span.dataset.since));
		}
	}

	function rowEl(e, { child = false } = {}) {
		return el(
			"div",
			{ class: `q-row${child ? " q-child" : ""}` },
			el("span", { class: `q-dot ${e.state}` }),
			el("span", { class: "q-job", text: e.job_id, title: e.slot_id ?? "" }),
			e.backend ? el("span", { class: "q-backend", text: e.backend }) : null,
			el("span", { class: "q-state", text: e.state }),
			el("span", { class: "q-elapsed", dataset: { since: String(e.enqueued_at ?? "") } }),
		);
	}

	function render(pools, entries) {
		body.textContent = "";
		if (!entries.length) {
			body.appendChild(el("div", { class: "q-empty", text: "nothing in flight" }));
			return;
		}
		// Nest prefab reuses under the canonical they share a mesh with: a child
		// row names its canonical's job id (same slot); everything else is
		// top-level. An orphan (canonical not currently queued) stays top-level.
		const keyOf = (slotId, jobId) => `${slotId}\u0000${jobId}`;
		const byKey = new Map(entries.map((e) => [keyOf(e.slot_id, e.job_id), e]));
		const childrenOf = new Map();
		const top = [];
		for (const e of entries) {
			const parentKey =
				e.canonical && e.canonical !== e.job_id ? keyOf(e.slot_id, e.canonical) : null;
			if (parentKey && byKey.has(parentKey)) {
				(childrenOf.get(parentKey) ?? childrenOf.set(parentKey, []).get(parentKey)).push(e);
			} else {
				top.push(e);
			}
		}
		// Group top-level rows by pool in the server's order; unknown pools last.
		const byPool = new Map(pools.map((p) => [p.id, []]));
		for (const e of top) {
			if (!byPool.has(e.pool)) byPool.set(e.pool, []);
			byPool.get(e.pool).push(e);
		}
		const meta = new Map(pools.map((p) => [p.id, p]));
		for (const [poolId, rows] of byPool) {
			if (!rows.length) continue;
			const pool = meta.get(poolId);
			// Cap counts TOP-LEVEL processing rows only — reuses derive locally and
			// don't consume the pool's (Trellis/Hunyuan) concurrency budget.
			const processing = rows.filter((r) => r.state === "processing").length;
			const section = el(
				"div",
				{ class: "q-section" },
				el(
					"div",
					{ class: "q-sec-head" },
					el("span", { class: "q-sec-lab", text: pool?.label ?? poolId }),
					el("span", {
						class: "q-sec-cap",
						text: pool?.cap ? `${processing}/${pool.cap}` : String(rows.length),
					}),
				),
			);
			for (const e of rows) {
				const kids = childrenOf.get(keyOf(e.slot_id, e.job_id)) ?? [];
				if (!kids.length) {
					section.appendChild(rowEl(e));
					continue;
				}
				// A canonical with reuses: an expandable group (default open). The
				// caret toggles the nested child rows; a "+N" badge shows the count.
				const kidsWrap = el("div", { class: "q-children" }, ...kids.map((k) => rowEl(k, { child: true })));
				const caret = el("span", { class: "q-gcaret", text: "▾" });
				const row = rowEl(e);
				row.classList.add("q-parent");
				row.prepend(caret);
				row.appendChild(el("span", { class: "q-gcount", text: `+${kids.length}` }));
				row.addEventListener("click", () => {
					const collapsed = kidsWrap.classList.toggle("collapsed");
					caret.textContent = collapsed ? "▸" : "▾";
				});
				section.appendChild(row);
				section.appendChild(kidsWrap);
			}
			body.appendChild(section);
		}
		updateElapsed();
	}

	async function tick() {
		let data = null;
		try {
			data = await api.trellisQueue();
		} catch {
			/* transient — retry on the next tick */
		}
		const entries = data?.entries ?? [];
		const pools = data?.pools ?? [];
		render(pools, entries);
		const busy = entries.length > 0;
		panel.classList.toggle("busy", busy);
		const countText = busy ? String(entries.length) : "";
		countEl.textContent = countText;
		tabCountEl.textContent = countText;
		setTimeout(tick, busy ? POLL_BUSY_MS : POLL_IDLE_MS);
	}

	// Advance the in-flight timers every second, independent of the poll cadence.
	setInterval(updateElapsed, 1000);
	tick();
}
