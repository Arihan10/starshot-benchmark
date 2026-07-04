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
// the obs tree / investigator.

import { api } from "./api.js";
import { el } from "./ui.js";

const OPEN_KEY = "starshot.queueOpen";
const POLL_BUSY_MS = 1500; // while anything is in flight
const POLL_IDLE_MS = 5000; // empty queue — a slow background heartbeat

export function initQueuePanel() {
	let open = false;
	try {
		open = localStorage.getItem(OPEN_KEY) === "1";
	} catch {
		/* private mode */
	}

	const countEl = el("span", { class: "q-count" });
	const caretEl = el("span", { class: "q-caret", text: open ? "▾" : "▸" });
	const head = el(
		"div",
		{ id: "queue-head", title: "every mesh generation currently in flight or waiting" },
		caretEl,
		el("span", { class: "q-title", text: "queue" }),
		countEl,
	);
	const body = el("div", { id: "queue-body" });
	const panel = el("div", { id: "queue-panel" }, head, body);
	panel.classList.toggle("open", open);
	document.body.appendChild(panel);

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

	function render(pools, entries) {
		body.textContent = "";
		if (!entries.length) {
			body.appendChild(el("div", { class: "q-empty", text: "nothing in flight" }));
			return;
		}
		// Group by pool in the server's pool order; any unknown pool falls to the end.
		const byPool = new Map(pools.map((p) => [p.id, []]));
		for (const e of entries) {
			if (!byPool.has(e.pool)) byPool.set(e.pool, []);
			byPool.get(e.pool).push(e);
		}
		const meta = new Map(pools.map((p) => [p.id, p]));
		for (const [poolId, rows] of byPool) {
			if (!rows.length) continue;
			const pool = meta.get(poolId);
			const processing = rows.filter((r) => r.state === "processing").length;
			body.appendChild(
				el(
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
					...rows.map((e) =>
						el(
							"div",
							{ class: "q-row" },
							el("span", { class: `q-dot ${e.state}` }),
							el("span", { class: "q-job", text: e.job_id, title: e.slot_id ?? "" }),
							e.backend ? el("span", { class: "q-backend", text: e.backend }) : null,
							el("span", { class: "q-state", text: e.state }),
						),
					),
				),
			);
		}
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
		countEl.textContent = busy ? String(entries.length) : "";
		setTimeout(tick, busy ? POLL_BUSY_MS : POLL_IDLE_MS);
	}

	tick();
}
