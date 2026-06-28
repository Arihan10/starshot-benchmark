// Lighting panel for the main 3D viewer — sliders for the engine in scene3d
// (exposure, key / fill / environment, shadow strength, sun azimuth/elevation,
// shadows on/off). Each input writes straight through `viewer.setLighting` so
// the scene updates live; values persist in localStorage. A toggle button on the
// overlay's viewer-toggles bar opens/closes the floating panel.

import { el } from "./ui.js";

const STORE_KEY = "starshot.lighting.v1";

// field -> slider range + value formatter (the label is shown beside it).
const FIELDS = [
	{ key: "exposure", label: "exposure", min: 0.1, max: 3, step: 0.05, fmt: (v) => v.toFixed(2) },
	{ key: "key", label: "key light", min: 0, max: 6, step: 0.1, fmt: (v) => v.toFixed(1) },
	{ key: "fill", label: "fill", min: 0, max: 2, step: 0.05, fmt: (v) => v.toFixed(2) },
	{ key: "env", label: "environment", min: 0, max: 3, step: 0.05, fmt: (v) => v.toFixed(2) },
	{ key: "shadow", label: "shadow", min: 0, max: 1, step: 0.02, fmt: (v) => v.toFixed(2) },
	{ key: "azimuth", label: "sun azimuth", min: 0, max: 360, step: 1, fmt: (v) => `${Math.round(v)}°` },
	{ key: "elevation", label: "sun elevation", min: 5, max: 89, step: 1, fmt: (v) => `${Math.round(v)}°` },
];

export function initLighting(viewer) {
	const defaults = viewer.lightingDefaults;
	if (!defaults) return; // a viewer without the lighting engine — nothing to drive

	const state = (() => {
		let saved = null;
		try {
			saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
		} catch {
			/* private mode */
		}
		return { ...defaults, ...(saved && typeof saved === "object" ? saved : {}) };
	})();
	const persist = () => {
		try {
			localStorage.setItem(STORE_KEY, JSON.stringify(state));
		} catch {
			/* private mode */
		}
	};

	// Each control re-syncs from `state` on reset; the input writes to `state` +
	// the live scene + storage as it's dragged.
	const sync = [];
	const rows = FIELDS.map((f) => {
		const input = el("input", {
			type: "range",
			class: "lp-slider",
			min: String(f.min),
			max: String(f.max),
			step: String(f.step),
		});
		const val = el("span", { class: "lp-val" });
		input.value = state[f.key];
		val.textContent = f.fmt(state[f.key]);
		input.addEventListener("input", () => {
			state[f.key] = parseFloat(input.value);
			val.textContent = f.fmt(state[f.key]);
			viewer.setLighting({ [f.key]: state[f.key] });
			persist();
		});
		sync.push(() => {
			input.value = state[f.key];
			val.textContent = f.fmt(state[f.key]);
		});
		return el("label", { class: "lp-row" }, el("span", { class: "lp-lab", text: f.label }), input, val);
	});

	const shadows = el("input", { type: "checkbox" });
	shadows.checked = state.shadows;
	shadows.addEventListener("change", () => {
		state.shadows = shadows.checked;
		viewer.setLighting({ shadows: state.shadows });
		persist();
	});
	sync.push(() => {
		shadows.checked = state.shadows;
	});

	const reset = el("button", {
		class: "lp-reset",
		text: "reset",
		onclick: () => {
			Object.assign(state, defaults);
			viewer.setLighting(state);
			for (const s of sync) s();
			persist();
		},
	});

	const panel = el(
		"div",
		{ id: "lighting-panel" },
		el("div", { class: "lp-head" }, el("span", { class: "lp-title", text: "lighting" }), reset),
		...rows,
		el("label", { class: "lp-row lp-check" }, shadows, el("span", { text: "cast shadows" })),
	);
	document.getElementById("canvas-host").appendChild(panel);

	const toggle = el("button", {
		id: "btn-lighting",
		title: "lighting controls (exposure, key / fill / environment, shadows, sun angle)",
		text: "☀ light",
		onclick: () => {
			const open = panel.classList.toggle("open");
			toggle.classList.toggle("on", open);
		},
	});
	document.getElementById("viewer-toggles").appendChild(toggle);

	// Push the persisted/default values onto the live scene now.
	viewer.setLighting(state);
}
