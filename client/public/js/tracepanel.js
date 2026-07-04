// The emittance-trace inspector: a left panel in the single-cell overlay that
// appears when an object/zone is picked in 3D. Two stacked parts:
//
//   1. an INFO block for the focused node — a live mini 3D preview (its mesh +
//      bbox, with the real baked orientation) plus its debug fields: prompt, a
//      zone's plan, orientation (semantic text + resolved degrees), dimensions,
//      world origin, proxy shape, structural parent, placement prose, and relationships.
//   2. the EMITTANCE trace below it — the chain of REGIONS that generated the
//      node (root → … → node, climbing each node's emitting region, NOT its
//      structural parent), each with the LLM calls that brought it into being,
//      each call's output truncated to just that node's slice.
//
// It reads the folded obs model + provenance (events.js). The info block is
// rebuilt only when the focus changes (or its data streams in), so the mini
// viewer's WebGL context survives the body's frequent rebuilds.

import { el, foldedPre, fmtJson, shortBytes } from "./ui.js";
import { emittanceLineage, extractRelevantOutput } from "./events.js";
import { createViewer } from "./scene3d.js";
import { api } from "./api.js";

const AXES_KEY = "starshot.traceAxes"; // mini-canvas axes toggle preference
const TRACE_W_KEY = "starshot.traceWidth"; // resizable sidebar width

// A node's OWN characterization steps — ones that ran ON the node itself rather
// than emitting/placing it from its parent: a zone's plan (zone_plan, root or
// nested) and the root's overall bbox. Shown in the trace alongside the
// provenance (emitted_by / placed_by). A zone's decompose / bbox-batch steps are
// deliberately NOT here — those generate its CHILDREN, so they live in the
// children's traces, not the zone's.
const OWN_STEPS = new Set(["zone_plan", "overall_bbox"]);

function statusDot(n) {
	if (!n) return "idle";
	if (n.error) return "error";
	if (n.phase === "done") return "done";
	return n.calls.length ? "running" : "idle";
}

const fmtVec = (a) => `(${a.map((x) => (+x).toFixed(2)).join(", ")})`;
const fmtDims = (a) => a.map((x) => (+x).toFixed(2)).join(" × ");

export function createTracePanel(
	hostEl,
	{
		onNavigate = () => {},
		onClose = () => {},
		onInquire = null,
		actions = null,
		meshUrlFor = (_id, node) => node.meshUrl ?? null,
		rawMeshUrlFor = (_id, node) =>
			node.meshUrl
				? node.meshUrl.replace(/\.glb(\?|$)/, ".raw.glb$1")
				: null,
	} = {},
) {
	const nodeLabelEl = el("span", { class: "tp-node" });

	// Mini preview + its overlaid canonical-axes toggle. The axes are world-
	// aligned (one global front view), so the gizmo is drawn AT the focused node
	// to show how its baked orientation sits against +X/+Y/+Z. Toggle persists.
	const previewHost = el("div", { class: "tp-preview" });
	let axesOn = false;
	try {
		axesOn = localStorage.getItem(AXES_KEY) === "1";
	} catch {
		/* private mode */
	}
	let lastGeom = null; // {center, size} of the focused node, for the axes gizmo
	const axesBtn = el("button", {
		class: `tp-axes-btn${axesOn ? " on" : ""}`,
		text: "axes",
		title: "show the canonical X/Y/Z axes at this node (X red · Y green · Z blue)",
		onclick: () => {
			axesOn = !axesOn;
			try {
				localStorage.setItem(AXES_KEY, axesOn ? "1" : "0");
			} catch {
				/* private mode */
			}
			applyAxes();
		},
	});
	const axesLegend = el(
		"div",
		{ class: "tp-axes-legend", style: axesOn ? "" : "display:none" },
		el("span", { class: "ax ax-x", text: "X" }),
		el("span", { class: "ax ax-y", text: "Y" }),
		el("span", { class: "ax ax-z", text: "Z" }),
	);
	previewHost.append(axesBtn, axesLegend);

	// Reference image (the Nano-Banana / library photo the mesh was generated
	// from), shown beneath the 3D preview. Hidden when absent or it fails to load.
	const previewImg = el("img", {
		class: "tp-preview-img",
		alt: "reference image",
	});
	const imgWrap = el(
		"div",
		{ class: "tp-img-wrap" },
		el("div", { class: "tp-field-lab", text: "reference image" }),
		previewImg,
	);
	imgWrap.style.display = "none";
	previewImg.onerror = () => {
		imgWrap.style.display = "none";
	};

	// Per-object texture maps: the raw mesh's PBR maps (base color, roughness,
	// metallic, occlusion, normal, emissive), each rendered to its own thumbnail
	// so every map can be inspected separately. Filled by `renderMaps` once the
	// raw GLB loads; hidden when the focused node exposes no readable maps.
	const mapsGrid = el("div", { class: "tp-maps-grid" });
	const mapsWrap = el(
		"div",
		{ class: "tp-maps" },
		el("div", { class: "tp-field-lab", text: "texture maps" }),
		mapsGrid,
	);
	mapsWrap.style.display = "none";

	const fieldsEl = el("div", { class: "tp-fields" });
	const infoEl = el(
		"div",
		{ class: "tp-info" },
		previewHost,
		mapsWrap,
		imgWrap,
		fieldsEl,
	);
	const bodyEl = el("div", { class: "tp-body" });
	const scrollEl = el("div", { id: "trace-panel-scroll" }, infoEl, bodyEl);
	const resizer = el("div", {
		id: "trace-panel-resizer",
		title: "drag to resize the panel",
	});
	hostEl.replaceChildren(
		el(
			"div",
			{ id: "trace-panel-inner" },
			el(
				"div",
				{ id: "trace-panel-head" },
				el(
					"div",
					{ class: "tp-head-top" },
					el("span", { class: "title", text: "emittance trace" }),
					el("button", {
						class: "tp-close",
						text: "close ✕",
						title: "close the trace (Esc / click empty space)",
						onclick: () => onClose(),
					}),
				),
				nodeLabelEl,
			),
			scrollEl,
		),
		resizer,
	);
	initResizer(resizer);

	// Sync the axes gizmo to the toggle + the focused node's geometry. Safe before
	// the mini viewer exists (created lazily) — renderInfo re-applies once it does.
	function applyAxes() {
		axesBtn.classList.toggle("on", axesOn);
		axesLegend.style.display = axesOn ? "" : "none";
		miniViewer?.setAxes(axesOn && !!lastGeom, lastGeom ?? {});
	}

	// Drag the right edge to resize the sidebar. Width lives in the `--trace-w`
	// CSS var so the panel AND the canvas-overlay shift (the :has() rule) track
	// together; clamped to the scene canvas and persisted across reloads.
	function initResizer(handle) {
		try {
			const saved = Number(localStorage.getItem(TRACE_W_KEY));
			if (saved >= 300) {
				// Cap a width saved on a wider screen so it can't swallow a narrow one.
				const w = Math.min(
					saved,
					Math.max(300, window.innerWidth - 200),
				);
				document.documentElement.style.setProperty(
					"--trace-w",
					`${w}px`,
				);
			}
		} catch {
			/* private mode */
		}
		let dragging = false;
		handle.addEventListener("pointerdown", (ev) => {
			dragging = true;
			handle.setPointerCapture(ev.pointerId);
			document.body.classList.add("trace-resizing");
			ev.preventDefault();
		});
		handle.addEventListener("pointermove", (ev) => {
			if (!dragging) return;
			const host = hostEl.parentElement; // #canvas-host
			const left = hostEl.getBoundingClientRect().left;
			const max = Math.max(360, (host?.clientWidth ?? 1200) - 120);
			const w = Math.max(300, Math.min(ev.clientX - left, max));
			document.documentElement.style.setProperty(
				"--trace-w",
				`${Math.round(w)}px`,
			);
		});
		const end = (ev) => {
			if (!dragging) return;
			dragging = false;
			document.body.classList.remove("trace-resizing");
			try {
				handle.releasePointerCapture(ev.pointerId);
			} catch {
				/* already released */
			}
			try {
				const w = parseInt(
					getComputedStyle(document.documentElement).getPropertyValue(
						"--trace-w",
					),
					10,
				);
				if (w) localStorage.setItem(TRACE_W_KEY, String(w));
			} catch {
				/* private mode */
			}
		};
		handle.addEventListener("pointerup", end);
		handle.addEventListener("pointercancel", end);
	}

	let model = null;
	let focusId = null;
	const expandedNodes = new Set(); // lineage node ids whose call list is open
	const expandedCalls = new Set(); // call indices whose detail is open
	let lastSig = null;
	let lastInfoSig = null;
	let lastPreviewKey = null; // focused mesh currently in the mini viewer — skips needless reloads
	// One reusable mini viewer for the whole panel lifetime (1 WebGL context);
	// created lazily, never disposed — just cleared + reloaded per focus, and
	// paused (setActive false) whenever the panel is hidden.
	let miniViewer = null;
	function ensureMini() {
		if (!miniViewer)
			miniViewer = createViewer(previewHost, { keyboard: false });
		return miniViewer;
	}

	// ── focused-node info (rebuilt only on focus change / live data fill-in) ──

	// The object/subregion spec the emitting decompose call named this node with —
	// the source of its semantic orientation text, placement prose, structural
	// parent, and relationships (none of which live on the obs node itself).
	function focusedSpec(id) {
		// Prefer the committed authored spec (final ids + rebound parent /
		// relationships, matching the scene); fall back to slicing the emitting
		// call's raw output for legacy logs that didn't commit full specs.
		const committed = model.specs?.get(id);
		if (committed && typeof committed === "object") return committed;
		const emitted = (model.provenance?.get(id) ?? []).find(
			(p) => p.relation === "emitted_by",
		);
		if (!emitted) return null;
		const { value, truncated } = extractRelevantOutput(emitted.call.output, id);
		return truncated && value && typeof value === "object" ? value : null;
	}

	function fieldGroup(label, valueNode) {
		return el(
			"div",
			{ class: "tp-field-group" },
			el("div", { class: "tp-field-lab", text: label }),
			typeof valueNode === "string"
				? el("div", { class: "tp-field-val", text: valueNode })
				: valueNode,
		);
	}

	function prop(label, value) {
		return el(
			"div",
			{ class: "tp-prop" },
			el("span", { class: "tp-prop-lab", text: label }),
			typeof value === "string"
				? el("span", { class: "tp-prop-val", text: value })
				: value,
		);
	}

	// Per-object generated-asset controls — shown in the info block only while the
	// overlay is viewing the from-scratch generated build (actions.available()).
	// regenerate rebuilds the mesh fresh on the chosen backend (Trellis / Hunyuan
	// on Modal / Hunyuan 3.1 on Tencent); symmetrize / unsymmetrize mirror or
	// reveal the raw mesh with no backend call. A plain regenerate propagates across
	// the prefab group; the prefab section below adds unlink (split this object
	// into a standalone asset) and link (join another group).
	function buildActions(id) {
		const busy = !!actions.isBusy?.(id);
		const backendSel = el(
			"select",
			{
				class: "tp-act-sel",
				title: "mesh backend for regenerate",
				disabled: busy,
			},
			...["trellis", "hunyuan", "hunyuan-tencent"].map((b) =>
				el("option", { value: b, text: b }),
			),
		);
		const reuse = el("input", { type: "checkbox", disabled: busy });
		// Regenerate the noun phrase (re-distill from the object's seed) alongside the
		// image + mesh — works even for objects with no noun phrase yet, and re-logs
		// the image event so later regenerations read the new phrase. Mutually
		// exclusive with "reuse image" (a fresh phrase needs a fresh image).
		const nounChk = el("input", { type: "checkbox", disabled: busy });
		reuse.addEventListener("change", () => {
			if (reuse.checked) nounChk.checked = false;
		});
		nounChk.addEventListener("change", () => {
			if (nounChk.checked) reuse.checked = false;
		});
		const sym = actions.symmetryOf?.(id);
		const mirrored = !!sym && (sym.plane === "xy" || sym.plane === "xz");

		let symRow;
		if (mirrored) {
			symRow = el(
				"div",
				{ class: "tp-act-row" },
				el("button", {
					class: "tp-act-btn",
					text: "unsymmetrize",
					disabled: busy,
					title: busy
						? "this asset is currently generating"
						: `reveal the full, un-mirrored mesh (currently mirrored across ${sym.plane})`,
					onclick: () => actions.onUnsymmetrize(id),
				}),
			);
		} else {
			const planeSel = el(
				"select",
				{ class: "tp-act-sel", title: "mirror plane", disabled: busy },
				el("option", { value: "xy", text: "xy · front/back" }),
				el("option", { value: "xz", text: "xz · top/bottom" }),
			);
			const keepSel = el("select", {
				class: "tp-act-sel",
				title: "which half to keep, then mirror onto the other",
				disabled: busy,
			});
			const KEEP = {
				xy: [
					["true", "keep front"],
					["false", "keep back"],
				],
				xz: [
					["true", "keep top"],
					["false", "keep bottom"],
				],
			};
			const syncKeep = () =>
				keepSel.replaceChildren(
					...KEEP[planeSel.value].map(([v, t]) =>
						el("option", { value: v, text: t }),
					),
				);
			syncKeep();
			planeSel.addEventListener("change", syncKeep);
			symRow = el(
				"div",
				{ class: "tp-act-row" },
				el("button", {
					class: "tp-act-btn",
					text: "symmetrize",
					disabled: busy,
					title: busy
						? "this asset is currently generating"
						: "mirror this asset across the chosen plane, keeping the chosen half",
					onclick: () =>
						actions.onSymmetrize(id, {
							plane: planeSel.value,
							keepPositive: keepSel.value === "true",
						}),
				}),
				planeSel,
				keepSel,
			);
		}
		const symText = mirrored
			? `mirrored · ${sym.plane}`
			: sym && sym.was
				? `un-symmetrized (was ${sym.was})`
				: "";
		const imgPrompt = actions.imagePromptOf?.(id) ?? null;
		// Torn "ghost": renders from a stale optimized twin but its raw/unoptimized
		// mesh is gone (a regen that didn't finish), so it can't be re-derived or
		// shown in the raw view. Don't warn while it's actively rebuilding (busy).
		const incomplete = !!actions.incompleteOf?.(id) && !busy;
		return el(
			"div",
			{ class: "tp-actions" },
			el("div", {
				class: "tp-field-lab",
				text: busy
					? "generated asset · generating…"
					: "generated asset",
			}),
			incomplete
				? el("div", {
						class: "tp-act-sym",
						style: "color:#f0c9c9",
						text: "⚠ raw mesh missing — renders from a stale optimized twin; regenerate to rebuild it",
					})
				: null,
			imgPrompt
				? el(
						"div",
						{ class: "tp-field-group" },
						el("div", {
							class: "tp-field-lab",
							text: "image prompt",
						}),
						el("div", {
							class: "tp-field-val tp-img-prompt",
							text: imgPrompt,
						}),
					)
				: null,
			el(
				"div",
				{ class: "tp-act-row" },
				el("button", {
					class: "tp-act-btn",
					text: "regenerate",
					disabled: busy,
					title: busy
						? "this asset is currently generating"
						: "rebuild this asset's mesh fresh on the chosen backend (propagates to its prefab group)",
					onclick: () =>
						actions.onRegenerate(id, {
							backend: backendSel.value,
							reuseImage: reuse.checked,
							regenNounPhrase: nounChk.checked,
						}),
				}),
				backendSel,
				el(
					"label",
					{
						class: "tp-act-check",
						title: "reuse the existing reference image (skip Nano-Banana)",
					},
					reuse,
					"reuse image",
				),
				el(
					"label",
					{
						class: "tp-act-check",
						title: "regenerate the noun phrase too — re-distill it from the object's seed (even if it has none yet) and re-log the image event so later regenerations use the new phrase",
					},
					nounChk,
					"new noun phrase",
				),
			),
			buildRenderToggle(id),
			symRow,
			symText ? el("div", { class: "tp-act-sym", text: symText }) : null,
			buildFrontView(id, busy),
			buildGlassify(id, busy),
			buildPrefabSection(id, busy),
		);
	}

	// Permanent deletion of the focused object. Unlike the generated-only actions
	// above it wipes BOTH builds' logs + every build's files, so it's gated on
	// `actions.deletable()` (any source cell, either asset view) rather than
	// `actions.available()` (generated view only). The confirm prompt + API call
	// live in the overlay; this only surfaces the button (disabled while the
	// asset is mid-build, to match the other controls).
	function buildDangerZone(id) {
		const busy = !!actions.isBusy?.(id);
		return el(
			"div",
			{ class: "tp-actions" },
			el("div", { class: "tp-field-lab", text: "danger zone" }),
			el(
				"div",
				{ class: "tp-act-row" },
				el("button", {
					class: "tp-act-btn danger",
					text: "delete object",
					disabled: busy,
					title: busy
						? "this asset is currently generating"
						: "permanently remove this object from the cell — both event logs + every build's mesh & image files (irreversible)",
					onclick: () => actions.onDelete(id),
				}),
			),
		);
	}

	// Per-object render toggle: flip THIS object between its optimized (KTX2 /
	// Meshopt) and unoptimized ("raw") mesh in the main scene — a pure view swap
	// (both already on disk), no rebuild. Hidden unless both variants exist
	// (actions.optimizedOf returns the effective mode, or null).
	function buildRenderToggle(id) {
		const mode = actions.optimizedOf?.(id);
		if (!mode || !actions.onSetOptimized) return null;
		return el(
			"div",
			{ class: "tp-act-row" },
			el("span", { class: "tp-act-sym", text: "render" }),
			el("button", {
				class: "tp-act-btn",
				text: mode === "optimized" ? "optimized ✓" : "raw",
				title: "swap this object between the optimized (KTX2/Meshopt) and unoptimized raw mesh in the scene — no rebuild",
				onclick: () =>
					actions.onSetOptimized(
						id,
						mode === "optimized" ? "raw" : "optimized",
					),
			}),
		);
	}

	// Front-view controls: rotate the object's RAW mesh 90° about an axis so a
	// different face becomes the front (the +Z-facing side of the raw, pre-transform
	// mesh). Each rotation re-fronts the WHOLE prefab group — the server bakes it
	// into the canonical's raw and re-derives every reuse + the optimized twin.
	function buildFrontView(id, busy) {
		if (!actions.onReorient) return null;
		const rot = (label, axis, degrees, tip) =>
			el("button", {
				class: "tp-act-btn",
				text: label,
				disabled: busy,
				title: busy ? "this asset is currently generating" : tip,
				onclick: () => actions.onReorient(id, { axis, degrees }),
			});
		return el(
			"div",
			{ class: "tp-frontview" },
			el("div", { class: "tp-field-lab", text: "front view" }),
			el(
				"div",
				{ class: "tp-act-row" },
				rot(
					"pitch ↓",
					"x",
					90,
					"tip the raw mesh 90° about X — brings the +Y face to the +Z front (re-fronts the whole prefab group)",
				),
				rot(
					"pitch ↑",
					"x",
					-90,
					"tip the raw mesh -90° about X — brings the -Y face to the +Z front (re-fronts the whole prefab group)",
				),
				rot(
					"yaw →",
					"y",
					90,
					"turn the raw mesh 90° about Y — brings the -X face to the +Z front (re-fronts the whole prefab group)",
				),
				rot(
					"yaw ←",
					"y",
					-90,
					"turn the raw mesh -90° about Y — brings the +X face to the +Z front (re-fronts the whole prefab group)",
				),
				rot(
					"roll ↺",
					"z",
					90,
					"roll the raw mesh 90° about Z — rotates the front face in-plane (re-fronts the whole prefab group)",
				),
				rot(
					"roll ↻",
					"z",
					-90,
					"roll the raw mesh -90° about Z — rotates the front face in-plane (re-fronts the whole prefab group)",
				),
			),
		);
	}

	// Forced glass controls. "make transparent" bakes this object's white /
	// near-white texels to near-transparent regardless of the pipeline's
	// window/glass + symmetry gates, across its whole prefab group. "reset from
	// raw" rebuilds the group's served meshes from the pristine raw — the undo for
	// glassify — keeping their current symmetry.
	function buildGlassify(id, busy) {
		if (!actions.onGlassify && !actions.onReset) return null;
		const row = el("div", { class: "tp-act-row" });
		if (actions.onGlassify) {
			row.appendChild(
				el("button", {
					class: "tp-act-btn",
					text: "make transparent",
					disabled: busy,
					title: busy
						? "this asset is currently generating"
						: "force the white / near-white areas of this object's texture to near-transparent — bypasses the window/glass + symmetry gates and applies to the whole prefab group; dropped by reset / regenerate",
					onclick: () => actions.onGlassify(id),
				}),
			);
		}
		if (actions.onReset) {
			row.appendChild(
				el("button", {
					class: "tp-act-btn",
					text: "reset from raw",
					disabled: busy,
					title: busy
						? "this asset is currently generating"
						: "rebuild this object's served mesh (and its prefab group) from the pristine raw — drops a forced glass transparency while keeping the current symmetry",
					onclick: () => actions.onReset(id),
				}),
			);
		}
		return el(
			"div",
			{ class: "tp-frontview" },
			el("div", { class: "tp-field-lab", text: "glass" }),
			row,
		);
	}

	// The prefab-group controls within the generated-asset block: a status line
	// (canonical / reuse / standalone + group size), an "unlink" that splits this
	// object out of its group into a standalone asset with its own copy of the
	// shared mesh (so it stops sharing and can then be regenerated alone), and a
	// "link" that joins it into another group so it shares that group's mesh. Hidden
	// until the object has a generated mesh.
	function buildPrefabSection(id, busy) {
		const prefab = actions.prefabOf?.(id) ?? null;
		if (!prefab) return null;
		const targets = actions.linkTargets?.(id) ?? [];
		const rows = [
			el("div", { class: "tp-field-lab", text: "prefab group" }),
		];

		const status = prefab.isReuse
			? `reuse of ${prefab.canonical} · ${prefab.groupSize} in group`
			: prefab.groupSize > 1
				? `canonical · ${prefab.groupSize} in group`
				: "standalone";
		rows.push(el("div", { class: "tp-act-sym", text: status }));

		if (prefab.groupSize > 1) {
			rows.push(
				el(
					"div",
					{ class: "tp-act-row" },
					el("button", {
						class: "tp-act-btn",
						text: "unlink",
						disabled: busy,
						title: busy
							? "this asset is currently generating"
							: "split this object out of its prefab group into a standalone asset with its own copy of the shared mesh — it stops sharing, so you can then regenerate it on its own",
						onclick: () => actions.onUnlink(id),
					}),
				),
			);
		}

		if (targets.length && actions.onLink) {
			const linkSel = el(
				"select",
				{
					class: "tp-act-sel",
					title: "the object to link this asset to — it joins that object's prefab group and shares its mesh",
					disabled: busy,
				},
				...targets.map((t) =>
					el("option", {
						value: t.id,
						text: t.size > 1 ? `${t.id} · ${t.size} in group` : t.id,
						title: t.prompt || t.id,
					}),
				),
			);
			const linkGroup = el("input", { type: "checkbox", disabled: busy });
			const showGroupOpt = prefab.groupSize > 1;
			rows.push(
				el(
					"div",
					{ class: "tp-act-row" },
					el("button", {
						class: "tp-act-btn",
						text: "link →",
						disabled: busy,
						title: busy
							? "this asset is currently generating"
							: "link this object to the chosen object, so it shares that object's prefab group + mesh",
						onclick: () =>
							actions.onLink(id, linkSel.value, {
								group: showGroupOpt && linkGroup.checked,
							}),
					}),
					linkSel,
					showGroupOpt
						? el(
								"label",
								{
									class: "tp-act-check",
									title: "move the entire current prefab group into the destination (canonical + every sibling)",
								},
								linkGroup,
								"entire group",
							)
						: null,
				),
			);
		}

		return el("div", { class: "tp-prefab" }, ...rows);
	}

	function infoSig(id) {
		const n = model.nodes.get(id);
		if (!n) return null;
		const emitted = (model.provenance?.get(id) ?? []).find(
			(p) => p.relation === "emitted_by",
		);
		return [
			id,
			n.prompt,
			n.plan,
			n.kind,
			n.phase,
			n.meshUrl,
			n.proxyShape,
			n.orientation,
			Array.isArray(n.origin) ? n.origin.join(",") : "",
			Array.isArray(n.dimensions) ? n.dimensions.join(",") : "",
			emitted?.call?.index ?? "",
		].join("|");
	}

	// ── per-object texture maps (read off the loaded raw GLB) ──

	const MAP_CANVAS_CAP = 512; // downscale big (2k) textures for cheap channel splits

	// A drawable bitmap for a THREE.Texture, or null. Compressed (KTX2) textures
	// carry no readable image, so the optimized twin yields nothing here — which is
	// exactly why the per-object view loads the raw (uncompressed) mesh.
	function texImage(tex) {
		const img = tex && tex.image;
		if (!img) return null;
		const w = img.width || img.videoWidth || 0;
		const h = img.height || img.videoHeight || 0;
		return w && h ? { img, w, h } : null;
	}

	// Render one texture (optionally a single channel, splatted to grayscale) to a
	// capped 2D canvas. channel ∈ {null, "r", "g", "b"} — glTF packs occlusion in
	// R, roughness in G, and metalness in B of the metallic-roughness texture.
	function drawMapCanvas(src, channel) {
		const scale = Math.min(1, MAP_CANVAS_CAP / Math.max(src.w, src.h));
		const cw = Math.max(1, Math.round(src.w * scale));
		const ch = Math.max(1, Math.round(src.h * scale));
		const canvas = el("canvas", { class: "tp-map-canvas" });
		canvas.width = cw;
		canvas.height = ch;
		const ctx = canvas.getContext("2d");
		try {
			ctx.drawImage(src.img, 0, 0, cw, ch);
		} catch {
			return null;
		}
		if (channel) {
			let data;
			try {
				data = ctx.getImageData(0, 0, cw, ch);
			} catch {
				return null;
			}
			const px = data.data;
			const off = channel === "r" ? 0 : channel === "g" ? 1 : 2;
			for (let i = 0; i < px.length; i += 4) {
				const v = px[i + off];
				px[i] = px[i + 1] = px[i + 2] = v;
				px[i + 3] = 255;
			}
			ctx.putImageData(data, 0, 0);
		}
		return canvas;
	}

	function mapTile(label, src, channel) {
		const canvas = drawMapCanvas(src, channel);
		if (!canvas) return null;
		const tile = el(
			"div",
			{
				class: "tp-map",
				title: "click to enlarge",
				onclick: () => tile.classList.toggle("big"),
			},
			canvas,
			el("div", {
				class: "tp-map-lab",
				text: `${label} · ${src.w}×${src.h}`,
			}),
		);
		return tile;
	}

	function clearMaps() {
		mapsGrid.replaceChildren();
		mapsWrap.style.display = "none";
	}

	// Pull the standard PBR maps off the loaded GLB's materials and render each
	// (and each packed channel) to its own thumbnail. Raw GLBs carry uncompressed
	// textures, so their pixels are readable here.
	function renderMaps(sceneObj) {
		clearMaps();
		if (!sceneObj) return;
		const found = {};
		sceneObj.traverse((o) => {
			if (!o.isMesh || !o.material) return;
			const mats = Array.isArray(o.material) ? o.material : [o.material];
			for (const m of mats) {
				if (!found.base && m.map) found.base = m.map;
				if (!found.mr && (m.roughnessMap || m.metalnessMap))
					found.mr = m.roughnessMap || m.metalnessMap;
				if (!found.normal && m.normalMap) found.normal = m.normalMap;
				if (!found.ao && m.aoMap) found.ao = m.aoMap;
				if (!found.emissive && m.emissiveMap)
					found.emissive = m.emissiveMap;
			}
		});
		const tiles = [];
		const add = (tex, channel, label) => {
			const src = texImage(tex);
			if (!src) return;
			const tile = mapTile(label, src, channel);
			if (tile) tiles.push(tile);
		};
		add(found.base, null, "base color");
		add(found.mr, "g", "roughness");
		add(found.mr, "b", "metallic");
		add(found.ao, "r", "occlusion");
		add(found.normal, null, "normal");
		add(found.emissive, null, "emissive");
		if (!tiles.length) return;
		mapsGrid.replaceChildren(...tiles);
		mapsWrap.style.display = "";
	}

	// Load an object's RAW mesh into the mini viewer, framed on its own bounds (it
	// isn't placed in the world bbox — that context lives in the main scene view).
	// Reads its texture maps on load; falls back to the transformed mesh if the
	// raw one is absent (e.g. an optimized-library cell keeps no raw).
	function loadObjectPreview(mv, id, rawUrl, fallbackUrl) {
		const onLoaded = (sceneObj, bounds) => {
			lastGeom = bounds
				? { center: bounds.center, size: bounds.size * 0.75 || 1 }
				: null;
			applyAxes();
			renderMaps(sceneObj);
		};
		const primary = rawUrl ?? fallbackUrl;
		if (!primary) return;
		mv.loadModel({ id, url: primary }, api.absUrl(primary), {
			onLoaded,
			onError: () => {
				if (rawUrl && fallbackUrl && fallbackUrl !== rawUrl) {
					mv.loadModel(
						{ id, url: fallbackUrl },
						api.absUrl(fallbackUrl),
						{ onLoaded },
					);
				}
			},
		});
	}

	function renderInfo(id) {
		fieldsEl.textContent = "";
		const n = model.nodes.get(id);
		if (!n) {
			previewHost.style.display = "none";
			miniViewer?.setActive(false);
			lastInfoSig = null;
			return;
		}
		const spec = focusedSpec(id);
		const isObject = n.kind === "object" || n.kind === "frame";
		const hasGeom = Array.isArray(n.origin) && Array.isArray(n.dimensions);

		// Mini 3D preview. Objects/frames show the RAW generation-API mesh on its own
		// (its native, un-placed frame), with the canonical axes at the mesh center.
		// Zones (and not-yet-meshed objects) show their bbox volume instead, with the
		// axes at the bbox center sized to its largest span.
		const bboxGeom = hasGeom
			? {
					center: [
						n.origin[0] + n.dimensions[0] / 2,
						n.origin[1] + n.dimensions[1] / 2,
						n.origin[2] + n.dimensions[2] / 2,
					],
					size:
						Math.max(
							Math.abs(n.dimensions[0]),
							Math.abs(n.dimensions[1]),
							Math.abs(n.dimensions[2]),
						) * 0.75 || 1,
				}
			: null;
		// The transformed/optimized mesh resolves the reference image + download
		// link; the per-object 3D view itself shows the RAW generation-API output.
		const meshUrl = meshUrlFor(id, n);
		const rawUrl = rawMeshUrlFor(id, n);
		// The reference image sits next to the transformed mesh on disk as `<id>.png`
		// (library and generated alike), so it follows the transformed url.
		const imageUrl = meshUrl
			? meshUrl.replace(/\.glb(\?|$)/, ".png$1")
			: null;
		if (imageUrl) {
			previewImg.src = api.absUrl(imageUrl);
			imgWrap.style.display = "";
		} else {
			imgWrap.style.display = "none";
		}
		// Preview + maps (re)load only when the focused mesh actually changes. A
		// streamed re-render of the SAME node — a generate-status poll, an infoSig
		// tick — leaves the loaded raw mesh + its map thumbnails in place; reloading
		// a large raw GLB on every tick would jank the panel.
		const previewUrl = isObject ? (rawUrl ?? meshUrl) : null;
		const previewKey = previewUrl
			? `mesh|${id}|${previewUrl}`
			: hasGeom
				? `bbox|${id}`
				: "";
		if (previewKey !== lastPreviewKey) {
			lastPreviewKey = previewKey;
			clearMaps();
			if (previewUrl || hasGeom) {
				previewHost.style.display = "";
				const mv = ensureMini();
				mv.clear();
				if (previewUrl) {
					// Object/frame: the raw mesh on its own (no world bbox — the raw output
					// isn't placed in it). lastGeom/axes + the per-map thumbnails are set
					// from the loaded geometry in loadObjectPreview's onLoaded.
					lastGeom = null;
					loadObjectPreview(mv, id, rawUrl, meshUrl);
				} else {
					// Zones (and objects with no mesh yet): the bbox volume.
					mv.loadBbox({
						id,
						origin: n.origin,
						dimensions: n.dimensions,
						node_kind: n.kind,
						proxy_shape: n.proxyShape ?? null,
					});
					lastGeom = bboxGeom;
				}
				mv.setActive(true);
				applyAxes();
			} else {
				previewHost.style.display = "none";
				miniViewer?.setActive(false);
			}
		}

		if (n.prompt) fieldsEl.appendChild(fieldGroup("prompt", n.prompt));
		// A zone's plan (from zone_plan) characterizes the zone itself, so surface it
		// right under the prompt — a selected zone reads as prompt → plan, not just
		// its prompt. Only zones carry a plan.
		if (n.kind === "zone" && n.plan)
			fieldsEl.appendChild(fieldGroup("plan", n.plan));

		const props = el("div", { class: "tp-props" });
		const addProp = (label, value) => {
			if (value !== null && value !== undefined && value !== "")
				props.appendChild(prop(label, value));
		};
		addProp("kind", n.kind + (n.phase ? ` · ${n.phase}` : ""));
		if (isObject) {
			const text =
				typeof spec?.orientation === "string" ? spec.orientation : null;
			const deg =
				typeof n.orientation === "number" ? `${n.orientation}°` : null;
			const parts = [text ? `“${text}”` : null, deg].filter(Boolean);
			if (parts.length) addProp("orientation", parts.join(" · "));
		}
		if (Array.isArray(n.dimensions))
			addProp("dimensions", fmtDims(n.dimensions));
		if (Array.isArray(n.origin))
			addProp("origin (world)", fmtVec(n.origin));
		if (n.proxyShape) addProp("proxy", n.proxyShape);
		if (spec?.parent)
			addProp(
				"structural parent",
				`${spec.parent}${spec.parent_relationship_kind ? ` · ${spec.parent_relationship_kind}` : ""}`,
			);
		if (props.childElementCount) fieldsEl.appendChild(props);

		if (spec?.placement)
			fieldsEl.appendChild(fieldGroup("placement", spec.placement));

		const rels = Array.isArray(spec?.relationships)
			? spec.relationships
			: [];
		if (rels.length) {
			const list = el(
				"div",
				{ class: "tp-rels" },
				rels.map((r) => {
					const linked =
						typeof r.target === "string" &&
						model.nodes.has(r.target);
					return el(
						"div",
						{ class: "tp-rel" },
						el("span", {
							class: "tp-rel-kind",
							text: r.kind ?? "?",
						}),
						el("span", { class: "tp-rel-arrow", text: "→" }),
						el("span", {
							class: `tp-rel-target${linked ? " link" : ""}`,
							text: r.target ?? "?",
							onclick: linked ? () => onNavigate(r.target) : null,
						}),
					);
				}),
			);
			fieldsEl.appendChild(
				fieldGroup(`relationships (${rels.length})`, list),
			);
		}

		if (rawUrl || meshUrl) {
			const links = el("span", {});
			if (rawUrl)
				links.append(
					el("a", {
						class: "tp-link",
						href: api.absUrl(rawUrl),
						target: "_blank",
						text: "raw ↗",
					}),
				);
			if (rawUrl && meshUrl) links.append(document.createTextNode(" · "));
			if (meshUrl)
				links.append(
					el("a", {
						class: "tp-link",
						href: api.absUrl(meshUrl),
						target: "_blank",
						text: "transformed ↗",
					}),
				);
			fieldsEl.appendChild(prop("mesh", links));
		}

		if (actions && actions.available() && isObject) {
			fieldsEl.appendChild(buildActions(id));
		}
		// Delete sits below the generated actions (or directly below the fields in
		// the library view) — shown for any object/frame on a source cell.
		if (actions && actions.onDelete && isObject && actions.deletable?.()) {
			fieldsEl.appendChild(buildDangerZone(id));
		}

		lastInfoSig = infoSig(id);
	}

	// ── emittance trace body (rebuilt on node/call toggle + streamed updates) ──

	// The calls that GENERATED + characterized a node: its provenance — emitted_by
	// (the decompose that named it) + placed_by (the bbox step that placed it) —
	// PLUS its own steps (a zone's plan; the root's overall bbox), which run on the
	// node itself so they're attributed, not provenance. Same shape for the focused
	// node and every ancestor zone. Falls back to anything attributed if a node has
	// none of the above, so a row is never bare. Ordered by execution.
	function nodeCalls(id) {
		const byIndex = new Map();
		for (const p of model.provenance?.get(id) ?? []) {
			if (p.call?.index != null)
				byIndex.set(p.call.index, {
					call: p.call,
					relation: p.relation,
				});
		}
		for (const c of model.nodes.get(id)?.calls ?? []) {
			if (
				c.index != null &&
				OWN_STEPS.has(c.step) &&
				!byIndex.has(c.index)
			) {
				byIndex.set(c.index, { call: c, relation: null });
			}
		}
		let entries = [...byIndex.values()];
		if (!entries.length) {
			entries = (model.nodes.get(id)?.calls ?? []).map((c) => ({
				call: c,
				relation: null,
			}));
		}
		return entries.sort(
			(a, b) => (a.call.index ?? 0) - (b.call.index ?? 0),
		);
	}

	function collapsedSection(label, text, { variables = null } = {}) {
		const body = variables
			? foldedPre(text, variables)
			: el("pre", { text });
		body.style.display = "none";
		const toggle = el("button", {
			class: "tp-toggle",
			text: "expand",
			onclick: (ev) => {
				ev.stopPropagation();
				const hidden = body.style.display === "none";
				body.style.display = hidden ? "" : "none";
				toggle.textContent = hidden ? "collapse" : "expand";
			},
		});
		return el(
			"div",
			{ class: "obsm-sec" },
			el(
				"div",
				{ class: "tp-sec-head" },
				el("span", {
					class: "tp-sec-lab",
					text: `${label} · ${shortBytes(text)}`,
				}),
				toggle,
			),
			body,
		);
	}

	function outputSection(call, nodeId) {
		const { value, truncated } = extractRelevantOutput(call.output, nodeId);
		const relText = fmtJson(value);
		const fullText = fmtJson(call.output);
		const body = el("pre", { text: relText });
		let full = false;
		const toggle = truncated
			? el("button", {
					class: "tp-toggle",
					text: "show full",
					onclick: (ev) => {
						ev.stopPropagation();
						full = !full;
						body.textContent = full ? fullText : relText;
						toggle.textContent = full
							? "show this node"
							: "show full";
					},
				})
			: null;
		return el(
			"div",
			{ class: "obsm-sec" },
			el(
				"div",
				{ class: "tp-sec-head" },
				el("span", {
					class: "tp-sec-lab",
					text: `output${truncated ? ` · ${nodeId}` : ""} · ${shortBytes(fullText)}`,
				}),
				toggle,
			),
			body,
		);
	}

	function callDetail(call, nodeId) {
		const detail = el(
			"div",
			{ class: "obsm-detail tp-call-detail" },
			el("div", {
				class: "muted",
				style: "margin-bottom:6px",
				text: [
					call.model ?? "",
					`${call.tokens_in ?? "?"} in / ${call.tokens_out ?? "?"} out tok`,
				]
					.filter(Boolean)
					.join(" · "),
			}),
			outputSection(call, nodeId),
			collapsedSection("input (user)", call.user ?? "", {
				variables: call.variables,
			}),
			collapsedSection("system", call.system ?? "", {
				variables: call.variables,
			}),
			call.reasoning
				? collapsedSection("reasoning", call.reasoning)
				: null,
		);
		detail.addEventListener("click", (ev) => ev.stopPropagation());
		return detail;
	}

	function callRow({ call, relation }, nodeId) {
		const wrap = el("div", { class: "tp-call-wrap" });
		const caret = el("span", { class: "caret" });
		let detail = null;
		const sync = () => {
			const open = expandedCalls.has(call.index);
			caret.textContent = open ? "▾" : "▸";
			row.classList.toggle("open", open);
		};
		const row = el(
			"div",
			{
				class: `obsm-call tp-call${relation === "emitted_by" ? " emits" : ""}`,
				onclick: () => {
					if (expandedCalls.has(call.index)) {
						expandedCalls.delete(call.index);
						detail?.remove();
						detail = null;
					} else {
						expandedCalls.add(call.index);
						detail = callDetail(call, nodeId);
						wrap.appendChild(detail);
					}
					sync();
				},
			},
			caret,
			el("span", {
				class: "step-badge",
				text: call.template ?? call.step ?? "?",
			}),
			relation === "emitted_by"
				? el("span", { class: "emit-badge", text: "emitted here" })
				: null,
			relation === "placed_by"
				? el("span", { class: "tp-place-badge", text: "placed here" })
				: null,
			el("span", {
				class: "muted",
				text: `#${call.index ?? "?"} · ${call.tokens_out ?? "?"} tok`,
			}),
			onInquire
				? el("button", {
						class: "call-ask",
						text: "why?",
						style: "margin-left:auto",
						title: "continue this step's conversation with the model that made it — ask it anything",
						onclick: (ev) => {
							ev.stopPropagation();
							onInquire(call);
						},
					})
				: null,
		);
		wrap.appendChild(row);
		if (expandedCalls.has(call.index)) {
			detail = callDetail(call, nodeId);
			wrap.appendChild(detail);
		}
		sync();
		return wrap;
	}

	function nodeBlock(id, focused) {
		const n = model.nodes.get(id);
		const open = expandedNodes.has(id);
		const calls = nodeCalls(id);
		const emittedBy = (model.provenance?.get(id) ?? []).find(
			(p) => p.relation === "emitted_by",
		);
		const via = emittedBy
			? (emittedBy.call.template ?? emittedBy.call.step ?? "?")
			: null;
		const caret = el("span", { class: "caret", text: open ? "▾" : "▸" });
		const row = el(
			"div",
			{
				class: "obsm-row clickable tp-node-row",
				title: n?.prompt ?? "",
				onclick: () => {
					if (expandedNodes.has(id)) expandedNodes.delete(id);
					else expandedNodes.add(id);
					paint();
				},
			},
			caret,
			el("span", { class: `dot ${statusDot(n)}` }),
			el("span", { class: "obsm-id", text: id }),
			n ? el("span", { class: "tp-kind", text: n.kind }) : null,
			via
				? el("span", {
						class: "obsm-via",
						text: `via ${via}`,
						title: `generated by ${via} on ${emittedBy.call.node ?? "?"}`,
					})
				: null,
			el("span", {
				class: "tp-count",
				text: `${calls.length} call${calls.length === 1 ? "" : "s"}`,
			}),
		);
		const block = el(
			"div",
			{ class: `obsm-trace-node tp-node${focused ? " focus" : ""}` },
			row,
		);
		if (open) {
			block.appendChild(
				calls.length
					? el(
							"div",
							{ class: "obsm-calls" },
							calls.map((c) => callRow(c, id)),
						)
					: el("div", {
							class: "muted",
							style: "margin:2px 0 6px 18px",
							text: "no LLM calls recorded for this node",
						}),
			);
		}
		return block;
	}

	function paint() {
		const prevScroll = scrollEl.scrollTop;
		bodyEl.textContent = "";
		const chain = emittanceLineage(model, focusId);
		const crumbs = el("div", { class: "obsm-crumbs" });
		chain.forEach((id, i) => {
			if (i)
				crumbs.appendChild(
					el("span", { class: "obsm-crumb-sep", text: "›" }),
				);
			crumbs.appendChild(
				el("span", {
					class: `obsm-crumb${id === focusId ? " cur" : ""}`,
					text: id,
					onclick: id === focusId ? null : () => onNavigate(id),
				}),
			);
		});
		bodyEl.appendChild(
			el(
				"div",
				{ class: "obsm-trace-head" },
				el("div", { class: "tp-trace-title", text: "generated by" }),
				crumbs,
			),
		);
		for (const id of chain)
			bodyEl.appendChild(nodeBlock(id, id === focusId));
		scrollEl.scrollTop = prevScroll;
		lastSig = signature(chain);
	}

	function signature(chain) {
		if (!model || focusId === null || !model.nodes.has(focusId))
			return null;
		const nodeSig = (id) => {
			const n = model.nodes.get(id);
			const prov = (model.provenance?.get(id) ?? [])
				.map((p) => `${p.relation}#${p.call.index}`)
				.join(",");
			return `${id}:${n?.calls.length ?? 0}:${prov}:${expandedNodes.has(id) ? 1 : 0}`;
		};
		const exp = [...expandedCalls].sort((a, b) => a - b).join(",");
		return `${focusId}|exp:${exp}|${chain.map(nodeSig).join(";")}`;
	}

	// ── lifecycle ──

	function show(m, id) {
		model = m;
		if (!m || !m.nodes.has(id)) {
			hide();
			return;
		}
		const newFocus = id !== focusId;
		if (newFocus) {
			focusId = id;
			expandedNodes.clear();
			expandedNodes.add(id);
			expandedCalls.clear();
		}
		hostEl.classList.add("open");
		if (newFocus) renderInfo(id);
		lastSig = null;
		paint();
		if (newFocus) scrollEl.scrollTop = 0;
	}

	function refresh(m, { streamed = false } = {}) {
		model = m;
		if (focusId === null || !hostEl.classList.contains("open")) return;
		if (!m || !m.nodes.has(focusId)) {
			hide();
			return;
		}
		const engaged = expandedCalls.size > 0 || scrollEl.scrollTop > 0;
		if (streamed && engaged) return;
		if (infoSig(focusId) !== lastInfoSig) renderInfo(focusId);
		if (signature(emittanceLineage(m, focusId)) !== lastSig) paint();
	}

	function hide() {
		hostEl.classList.remove("open");
		miniViewer?.setActive(false);
	}

	function reset() {
		focusId = null;
		expandedNodes.clear();
		expandedCalls.clear();
		model = null;
		lastSig = null;
		lastInfoSig = null;
		lastPreviewKey = null;
		fieldsEl.textContent = "";
		bodyEl.textContent = "";
		clearMaps();
		previewHost.style.display = "none";
		lastGeom = null;
		miniViewer?.setActive(false);
		miniViewer?.setAxes(false);
		miniViewer?.clear();
		hide();
	}

	// Re-render the focused node's info block (e.g. when the asset mode toggles,
	// so the generated-asset actions appear/disappear without a focus change).
	function rerenderInfo() {
		if (focusId !== null && model && model.nodes.has(focusId))
			renderInfo(focusId);
	}

	return {
		show,
		refresh,
		hide,
		reset,
		rerenderInfo,
		isOpen: () => hostEl.classList.contains("open"),
		focusId: () => focusId,
	};
}
