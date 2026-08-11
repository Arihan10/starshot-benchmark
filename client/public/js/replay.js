// Step a cell through its pipeline one STRUCTURAL LLM CALL at a time.
//
// A normal open paints the finished scene at once. Replay instead walks a cut
// through the event log, and each "next" advances that cut past exactly one
// structural call — zone_plan on a zone, then encapsulating_decompose on it,
// then the next zone's plan, and so on.
//
// Two things make the cut mean what it should:
//
//   * A step's cut lands just BEFORE the following call, not right after its own.
//     So everything a call CAUSED — the bboxes it authored, the meshes that
//     landed, the secondary image_prompt / library_match traffic it kicked off —
//     is inside that step rather than orphaned at the head of the next one.
//   * The bbox batches are folded into the decompose they resolve.
//     `child_bbox_batch` and `object_bbox_batch` carry no decision of their own;
//     they position what the preceding decompose named. Splitting them out would
//     double the step count and produce a "next" that only moves boxes.
//
// Replay also keeps its OWN observability model, fed only the events inside the
// cut. That is what makes the rest of the UI honest at each step: hovering a zone
// whose plan has not been authored yet shows no plan, and no calls that have not
// happened. Without it the panels would read from the finished log and describe a
// future the scene on screen hasn't reached.

import { createObsModel, dispatchSceneEvent } from "./events.js";
import { el } from "./ui.js";

// Calls that represent a decision, and so earn their own step.
const STRUCTURAL_STEPS = new Set([
	"overall_bbox",
	"zone_plan",
	"zone_decompose",
	"anchor_decompose",
	"encapsulating_decompose",
	"negative_space_decompose",
	"next_object",
]);

// Calls that only PLACE what a decompose already named — folded into it.
const BATCH_STEPS = new Set(["child_bbox_batch", "object_bbox_batch"]);

const SPEEDS = [
	{ label: "slow", ms: 1200 },
	{ label: "1x", ms: 500 },
	{ label: "4x", ms: 140 },
	{ label: "max", ms: 30 },
];
const DEFAULT_MS = 500;

// Highlight colours for the annotation overlay. The zone reads as "where we are",
// the placements as "what this step just did" — two questions, so two colours.
//
// The node-kind palette already spends red on zones, green on anchor objects,
// light blue on FRAMES, purple on next_object, tan on negative space and teal on
// selection, so magenta — the annotation overlay's own established colour — is
// what's left for the region. Pure yellow for the placements sits close to the
// hover highlight (0xffe14a), but hover is transient and only ever on the one
// node under the cursor, so the two don't compete in practice.
const ZONE_COLOR = 0xff3df5; // magenta — the region the current call ran on
const PLACED_COLOR = 0xffff00; // yellow — boxes this step just resolved

// `child_bbox_batch` tiles a parent exactly, so a zone's highlight and the
// highlights of the children it just placed share faces to the millimetre. Depth
// testing is off in the overlay pass, so coincident outlines don't blend — the
// one drawn last simply hides the other, and the children's borders disappear
// along every shared face. Nudging the ZONE box out by a hair separates them.
//
// Scaled to the box so it holds from a 2m closet to a 40m site, and clamped at
// both ends so it stays a hairline rather than misrepresenting the bounds.
const ZONE_NUDGE_FRAC = 0.004; // of the longest side
const ZONE_NUDGE_MIN = 0.02; // metres
const ZONE_NUDGE_MAX = 0.1;

function zoneNudge(box) {
    const span = Math.max(...box.dimensions.map((d) => Math.abs(d)));
    return Math.min(
        ZONE_NUDGE_MAX,
        Math.max(ZONE_NUDGE_MIN, span * ZONE_NUDGE_FRAC),
    );
}

// Grow a box by `e` on every side. Works off min/max rather than origin +
// dimensions because an authored dimension can be negative, in which case adding
// to it would SHRINK the box.
function inflate(box, e) {
    const origin = [];
    const dimensions = [];
    for (let i = 0; i < 3; i++) {
        const a = box.origin[i];
        const b = a + box.dimensions[i];
        origin.push(Math.min(a, b) - e);
        dimensions.push(Math.abs(b - a) + 2 * e);
    }
    return { origin, dimensions };
}

// Partition the log at structural-call boundaries. Every event stays in exactly
// one step, so advancing through all of them applies the whole log.
function buildSteps(history) {
	const steps = [];
	history.forEach((e, i) => {
		if (e.kind !== "cache.llm") return;
		const step = e.step;
		if (BATCH_STEPS.has(step)) {
			steps[steps.length - 1]?.batches.push(step);
			return;
		}
		if (!STRUCTURAL_STEPS.has(step)) return; // secondary — swept into its step
		steps.push({
			at: i,
			step: e.template ?? step ?? "?",
			node: e.node ?? "",
			index: e.index,
			batches: [],
		});
	});
	for (let k = 0; k < steps.length; k++) {
		steps[k].cut = k + 1 < steps.length ? steps[k + 1].at : history.length;
	}
	return steps;
}

function labelOf(s) {
	if (!s) return "";
	const n = s.batches.length;
	const batch = n ? ` + ${s.batches[0]}${n > 1 ? ` ×${n}` : ""}` : "";
	return `${s.step}${batch}${s.node ? ` · ${s.node}` : ""}`;
}

// --- engine -------------------------------------------------------------------

function createReplay({ viewer, onChange, onCut }) {
	let history = [];
	let steps = [];
	let pos = 0; // structural steps applied
	let applied = 0; // events applied (== steps[pos-1].cut)
	let obs = createObsModel(); // prefix-only model the UI reads while active
	let timer = null;
	let stepMs = DEFAULT_MS;
	let active = false;

	// What the step at `target` acted on: the region its call ran on, and every
	// node it gave a box to. Derived from the step's OWN event range rather than
	// from whatever range we happened to traverse, so a scrub lands on the same
	// highlight as walking there one step at a time.
	function highlightFor(target) {
		if (!target) return { zone: null, placed: [] };
		const s = steps[target - 1];
		const placed = [];
		for (let i = s.at; i < s.cut; i++) {
			const e = history[i];
			if (e.kind === "bbox" && typeof e.id === "string")
				placed.push(e.id);
		}
		return { zone: s.node || null, placed };
	}

	function paintHighlight() {
		const { zone, placed } = highlightFor(pos);
		const boxOf = (id) => {
			const n = obs.model.nodes.get(id);
			return n && Array.isArray(n.origin) && Array.isArray(n.dimensions)
				? { origin: n.origin, dimensions: n.dimensions }
				: null;
		};
		const boxes = [];
		for (const id of placed) {
			const b = boxOf(id);
			if (b) boxes.push({ ...b, color: PLACED_COLOR });
		}
		// The zone goes last so its outline wins where the two coincide — a zone
		// resolved by its parent's bbox batch is in both sets.
		const zb = zone ? boxOf(zone) : null;
		if (zb) boxes.push({ ...inflate(zb, zoneNudge(zb)), color: ZONE_COLOR });
		viewer.setOverlayBoxes(boxes);
	}

	function snapshot() {
		const cur = steps[pos - 1];
		const { placed } = highlightFor(pos);
		return {
			active,
			playing: timer !== null,
			pos,
			total: steps.length,
			stepMs,
			label: labelOf(cur),
			placed: placed.length,
			logIndex: cur?.index ?? null,
		};
	}

	const notify = () => onChange?.(snapshot());

	// Internal position only — deliberately does NOT touch the viewer. `load` runs
	// immediately after `openCell` has painted the full scene, with replay still
	// inactive, so clearing here would wipe the normal view and leave nothing to
	// repaint it. Only `resetScene` takes the scene, and only once we're active.
	function resetState() {
		obs = createObsModel();
		applied = 0;
		pos = 0;
	}

	function resetScene() {
		viewer.clear({ keepCamera: true });
		viewer.clearOverlayBoxes();
		resetState();
	}

	// Feed the events between the current cut and the target one. Mesh loads are
	// fire-and-forget inside `dispatchSceneEvent`, so this stays synchronous.
	function advanceTo(target) {
		const cut = target === 0 ? 0 : steps[target - 1].cut;
		for (let i = applied; i < cut; i++) {
			const e = history[i];
			dispatchSceneEvent(viewer, e);
			obs.feed(e);
		}
		applied = cut;
		pos = target;
	}

	function seek(target) {
		const n = Math.max(0, Math.min(steps.length, Math.round(target)));
		// No per-event undo, so backwards means rebuilding from empty. Cheap after
		// the first pass — the obs fold is in-memory and the meshes are cached.
		if (n < pos) resetScene();
		advanceTo(n);
		paintHighlight();
		onCut?.();
		notify();
	}

	function pause() {
		if (!timer) return;
		clearInterval(timer);
		timer = null;
		notify();
	}

	// Taking ownership of the scene: empty it, drop any previous prefix model, and
	// tell the caller to re-render the panels against the new (empty) one.
	function enter() {
		if (active) return;
		active = true;
		// Nothing else in the app draws on the annotation overlay, so it is safe
		// to force it on rather than trying to save and restore the old state.
		viewer.setOverlayVisible(true);
		resetScene();
		onCut?.();
		notify();
	}

	function play() {
		if (timer || !steps.length) return;
		if (!active) enter();
		else if (pos >= steps.length) resetScene();
		timer = setInterval(() => {
			if (pos >= steps.length) {
				pause();
				return;
			}
			advanceTo(pos + 1);
			paintHighlight();
			onCut?.();
			notify();
		}, stepMs);
		notify();
	}

	return {
		load(events) {
			pause();
			active = false;
			history = events ?? [];
			steps = buildSteps(history);
			resetState();
			notify();
		},
		enter,
		play,
		pause,
		toggle() {
			if (timer) pause();
			else play();
		},
		seek,
		stepBy(delta) {
			pause();
			if (!active) enter();
			seek(pos + delta);
		},
		setSpeed(ms) {
			stepMs = ms;
			if (!timer) {
				notify();
				return;
			}
			pause();
			play();
		},
		exit() {
			pause();
			active = false;
			viewer.clearOverlayBoxes();
			notify();
		},
		isActive: () => active,
		// The prefix-only model — what the viewer, trace panel and obs tree read
		// while replay owns the scene.
		model: () => obs.model,
		state: snapshot,
	};
}

// --- control bar --------------------------------------------------------------

// Builds the bar into `host` and returns the controller the overlay drives.
// `onExit` fires when the user leaves replay, so the caller can repaint normally;
// `onCut` fires whenever the cut moves, so it can re-render the panels.
export function initReplay({ viewer, host, onExit, onCut }) {
	const engine = createReplay({ viewer, onChange: sync, onCut });

	const btnRestart = el("button", {
		class: "rp-btn",
		text: "⏮",
		title: "back to before the first call",
		onclick: () => engine.seek(0),
	});
	const btnBack = el("button", {
		class: "rp-btn",
		text: "◀ prev",
		title: "back one structural call",
		onclick: () => engine.stepBy(-1),
	});
	const btnPlay = el("button", {
		class: "rp-btn rp-play",
		text: "▶",
		title: "auto-advance / pause",
		onclick: () => engine.toggle(),
	});
	const btnNext = el("button", {
		class: "rp-btn rp-next",
		text: "next ▶",
		title: "forward one structural call",
		onclick: () => engine.stepBy(1),
	});
	const speed = el(
		"select",
		{
			class: "rp-speed",
			title: "auto-advance speed",
			onchange: (e) => engine.setSpeed(Number(e.target.value)),
		},
		...SPEEDS.map((s) =>
			el(
				"option",
				{ value: String(s.ms), selected: s.ms === DEFAULT_MS },
				s.label,
			),
		),
	);
	const slider = el("input", {
		type: "range",
		class: "rp-slider",
		min: "0",
		max: "0",
		value: "0",
		title: "scrub through the structural calls",
		oninput: (e) => engine.seek(Number(e.target.value)),
	});
	const counter = el("span", { class: "rp-count" });
	// Swatch colours come from the same constants the overlay draws with, so the
	// key can't drift from what is on screen.
	const key = (color, text) =>
		el(
			"span",
			{ class: "rp-key" },
			el("i", {
				class: "rp-sw",
				style: `background:#${color.toString(16).padStart(6, "0")}`,
			}),
			text,
		);
	const legend = el(
		"span",
		{ class: "rp-legend" },
		key(ZONE_COLOR, "zone"),
		key(PLACED_COLOR, "placed"),
	);
	const label = el("span", { class: "rp-label" });
	const btnExit = el("button", {
		class: "rp-btn",
		text: "✕",
		title: "leave replay and restore the full scene",
		onclick: () => close(),
	});

	// `label` is full-width, so it wraps to its own line — everything that should
	// stay on the control row has to be ordered before it.
	const bar = el(
		"div",
		{ id: "replay-bar" },
		btnRestart,
		btnBack,
		btnPlay,
		btnNext,
		speed,
		slider,
		counter,
		legend,
		btnExit,
		label,
	);
	bar.style.display = "none";
	host.appendChild(bar);

	function sync(s) {
		btnPlay.textContent = s.playing ? "⏸" : "▶";
		slider.max = String(s.total);
		if (document.activeElement !== slider) slider.value = String(s.pos);
		counter.textContent = `${s.pos} / ${s.total}`;
		const placed = s.placed ? ` · placed ${s.placed}` : "";
		label.textContent = s.label ? `${s.label}${placed}` : "";
		label.title =
			s.logIndex != null
				? `${s.label}${placed}  ·  event #${s.logIndex}`
				: s.label;
	}

	function close() {
		engine.exit();
		bar.style.display = "none";
		onExit?.();
	}

	return {
		// Hand replay the history the overlay just folded; resets any previous
		// cell's position so a swap can't leave a stale cut behind.
		load(events) {
			engine.load(events);
		},
		// While replay owns the scene, live events must not paint into it and the
		// panels must read the prefix model, not the finished log.
		isActive: engine.isActive,
		model: engine.model,
		// The `replay` button. Opening starts at an empty scene rather than
		// auto-playing: the point is stepping, so the first "next" is the user's.
		toggle() {
			if (bar.style.display === "none") {
				bar.style.display = "";
				engine.enter();
				return true;
			}
			close();
			return false;
		},
		stop() {
			engine.exit();
			bar.style.display = "none";
		},
	};
}
