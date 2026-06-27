// Global mesh-generation queue panel — a live view of every in-flight + waiting
// generation across the concurrency pools (Trellis + Hunyuan on Modal, and
// Hunyuan 3.1 on Tencent), polled from /trellis/queue. The queue is
// process-global (it spans every cell + branch), so this floats in the corner
// rather than living in one cell's overlay: a from-scratch generate fans a whole
// scene onto it, and per-asset regenerations surface as waiting → processing rows.

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
