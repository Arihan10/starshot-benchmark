/**
 * Does this brand mark still read once its row turns to paper?
 *
 * A hovered standings row lays cream under everything in it, and the row's own type
 * follows because it is written in `ink` and `mark`. A brand mark cannot: it is
 * somebody else's artwork in somebody else's colours, and some of those colours are
 * white — a logo drawn for a dark site. On the cream that is a mark that vanishes at
 * the exact moment the row is being pointed at.
 *
 * THIS MEASURES RATHER THAN LISTS. A hand-kept list of "the white ones" is a list
 * that is right on the day it is written: the icons come from a dependency, the
 * board grows a new lab whenever one ships a model, and nothing about either event
 * makes anyone revisit the list. The failure is silent and invisible — a mark that
 * is simply not there on hover. So the question is asked of the pixels, once, at
 * runtime, and the answer follows the artwork wherever it goes.
 *
 * IT IS ASKED IN THE HOVER PALETTE, not the resting one, which is the only reading
 * that means anything. Marks painted in `currentColor` are near-white at rest and
 * near-black on a hovered row — measure them at rest and every one of them looks
 * like a white logo that needs rescuing, and inverting those would produce the very
 * bug this exists to prevent. The probe below therefore re-points the same two
 * tokens the row does, so what gets rasterised is the mark as the viewer will
 * actually meet it.
 */

// The paper the row turns to — `--paper-rgb` in globals.css. A literal here on
// purpose: this module runs before it has anything to read the token off, and the
// number it needs is the one constant in the question being asked. KEEP IT IN STEP
// with the token; it was the old warm cream for a while after the light system was
// unified, which is a silent 0.5% error in every contrast reading taken here.
const PAPER: [number, number, number] = [237, 237, 237];

// Big enough that a thin monogram survives rasterising, small enough that reading
// it back is free. The measurement is a weighted average over the whole mark, so
// resolution buys nothing past the point where strokes stop dropping out.
const RASTER = 48;

// Below alpha this, a pixel is the antialiased edge of the mark rather than the
// mark, and edge pixels are half background — they drag every average toward the
// paper and make a white logo look mid-grey.
const OPAQUE = 0.35;

/**
 * THE TWO THRESHOLDS, AND BOTH SIT IN GAPS IN THE MEASURED DATA — which is the only
 * honest way to pick a cutoff. Rasterising the current board gives:
 *
 *   OpenAI, Grok (as they were, frozen white)  1.03:1   chroma 0.00
 *   Anthropic                                  2.58:1   chroma 0.51
 *   Gemini                                     2.67:1   chroma 0.60
 *   Meta                                       3.54:1   chroma 0.93
 *   DeepSeek                                   3.57:1   chroma 0.69
 *   Qwen                                       4.30:1   chroma 0.63
 *
 * Nothing lands between 1.03 and 2.58, and nothing between 0.00 and 0.51. Both
 * cutoffs are placed in the middle of empty space, so neither is a close call for
 * any mark on the board and a new lab has to be genuinely unreadable to trip it.
 */
const MIN_CONTRAST = 2;

/**
 * AND IT MUST BE NEARLY NEUTRAL, because inverting is a HUE OPERATION as much as a
 * lightness one: `invert(1)` takes a pale yellow mark to dark blue. For a white or
 * grey logo that is exactly right — white goes to black and the brand is unharmed,
 * since a monochrome mark has no colour to damage. For a coloured one it is worse
 * than the problem, because a brand rendered in the wrong hue is a mistake a reader
 * can see, while a brand rendered a little light is only faint.
 *
 * So a light, saturated mark is deliberately LEFT ALONE here. It would want a
 * different remedy — a plate behind it, or a darker variant of the artwork — and
 * silently inverting it would be this function exceeding what it can know.
 */
const MAX_CHROMA = 0.18;

const srgbToLinear = (u: number) => {
	const c = u / 255;
	return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};

const luminance = (r: number, g: number, b: number) =>
	0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b);

const PAPER_L = luminance(...PAPER);

/** Answers already paid for. The board repeats a lab across rows and re-renders on
 *  every sort and keystroke; the artwork does not change between any of those. */
const cache = new Map<string, boolean>();
const inFlight = new Map<string, Promise<boolean>>();

let probe: HTMLElement | null = null;

/**
 * A detached corner of the document carrying the ROW'S HOVER PALETTE.
 *
 * It has to be in the document and it has to be laid out: `getComputedStyle` on a
 * node in a fragment resolves nothing, and `currentColor` is the whole point. Held
 * off-screen rather than hidden — `display: none` would give the same nothing.
 */
function getProbe(): HTMLElement {
	if (probe) return probe;
	probe = document.createElement("div");
	probe.setAttribute("aria-hidden", "true");
	probe.style.cssText =
		"position:fixed;left:-9999px;top:0;width:0;height:0;overflow:hidden;pointer-events:none";
	// The two the row re-points — keep in step with StandingsTable's hover.
	probe.style.setProperty("--ink-rgb", "var(--paper-ink-rgb)");
	probe.style.setProperty("--mark-rgb", "var(--paper-ink-rgb)");
	document.body.appendChild(probe);
	return probe;
}

// The paint properties, copied from the live node onto the clone so the serialised
// copy carries what the browser RESOLVED rather than what the file asked for. This
// is what turns `currentColor` and `var(...)` into numbers; without it the clone is
// rasterised against an empty stylesheet and every token collapses to black.
const PAINT = [
	"fill",
	"stroke",
	"stop-color",
	"stop-opacity",
	"fill-opacity",
	"stroke-opacity",
	"stroke-width",
	"opacity",
	"color",
];

function inlinePaint(live: Element, clone: Element) {
	const cs = getComputedStyle(live);
	for (const prop of PAINT) {
		const value = cs.getPropertyValue(prop);
		if (value) clone.setAttribute(prop, value);
	}
	const liveKids = live.children;
	const cloneKids = clone.children;
	for (let i = 0; i < liveKids.length; i++) {
		const kid = cloneKids[i];
		if (kid) inlinePaint(liveKids[i], kid);
	}
}

/**
 * Rasterise `svg` in the hover palette and report whether it needs inverting to
 * hold on the paper. Resolves `false` for anything it cannot measure — a mark that
 * merely fails to rasterise is not evidence that it is white, and guessing would
 * invert a perfectly good logo.
 */
export function needsInvertOnPaper(lab: string, svg: SVGSVGElement): Promise<boolean> {
	const known = cache.get(lab);
	if (known !== undefined) return Promise.resolve(known);
	const running = inFlight.get(lab);
	if (running) return running;

	const job = measure(svg)
		.catch(() => false)
		.then((verdict) => {
			cache.set(lab, verdict);
			inFlight.delete(lab);
			return verdict;
		});
	inFlight.set(lab, job);
	return job;
}

async function measure(svg: SVGSVGElement): Promise<boolean> {
	if (typeof window === "undefined" || !svg.isConnected) return false;

	const host = getProbe();
	// The live copy goes in the probe so its paint resolves against the paper
	// tokens; the second copy is what gets serialised, with those resolved values
	// written onto it. One clone cannot do both — reading and freezing the same
	// node would freeze whatever palette it happens to be sitting in.
	const lit = svg.cloneNode(true) as SVGSVGElement;
	host.replaceChildren(lit);

	const flat = lit.cloneNode(true) as SVGSVGElement;
	inlinePaint(lit, flat);
	host.replaceChildren();

	flat.setAttribute("xmlns", "http://www.w3.org/2000/svg");
	flat.setAttribute("width", String(RASTER));
	flat.setAttribute("height", String(RASTER));

	const markup = new XMLSerializer().serializeToString(flat);
	const image = new Image();
	image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;
	await image.decode();

	const canvas = document.createElement("canvas");
	canvas.width = RASTER;
	canvas.height = RASTER;
	const ctx = canvas.getContext("2d");
	if (!ctx) return false;
	ctx.drawImage(image, 0, 0, RASTER, RASTER);
	const { data } = ctx.getImageData(0, 0, RASTER, RASTER);

	// WEIGHTED BY COVERAGE, which is the difference between "what colours are in
	// this logo" and "what does this logo look like". A mark that is 95% white with
	// one dark dot averages to mid-grey unweighted and reads as white to a person.
	let weight = 0;
	let lum = 0;
	let chroma = 0;
	for (let i = 0; i < data.length; i += 4) {
		const alpha = data[i + 3] / 255;
		if (alpha < OPAQUE) continue;
		const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
		weight += alpha;
		lum += luminance(r, g, b) * alpha;
		chroma += ((Math.max(r, g, b) - Math.min(r, g, b)) / 255) * alpha;
	}
	if (!weight) return false;

	const L = lum / weight;
	const contrast =
		(Math.max(L, PAPER_L) + 0.05) / (Math.min(L, PAPER_L) + 0.05);

	return contrast < MIN_CONTRAST && chroma / weight < MAX_CHROMA;
}
