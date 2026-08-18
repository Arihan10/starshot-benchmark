"use client";

import { useCallback, useEffect, useMemo, useRef, type RefObject } from "react";

type Slot = "prompt" | "eyebrow" | "brandName" | "brandSub" | "navEnd" | "keysStart";

type Slots = Partial<Record<Slot, HTMLElement>>;

export type ArenaMetrics = {
	register: (slot: Slot) => (el: HTMLElement | null) => void;
	remeasure: () => void;
};

const edgeInset = () => Math.max(20, Math.min(44, window.innerWidth * 0.024));

let scratch: CanvasRenderingContext2D | null = null;
const textContext = () => (scratch ??= document.createElement("canvas").getContext("2d"));

/** One rect per rendered line of an element's text. */
function lineRects(el: HTMLElement): DOMRect[] {
	const range = document.createRange();
	range.selectNodeContents(el);
	const rows: DOMRect[] = [];
	for (const rect of range.getClientRects()) {
		if (rect.width <= 1 || rect.height <= 1) continue;
		if (rows.some((row) => Math.abs(row.top - rect.top) < 1)) continue;
		rows.push(rect);
	}
	return rows;
}

/** Width of the glyph run itself, not of the box the run sits in. */
function runWidth(el: HTMLElement, trailing: number): number {
	const range = document.createRange();
	range.selectNodeContents(el);
	const rects = [...range.getClientRects()];
	if (!rects.length) return 0;
	return rects[rects.length - 1].right - rects[0].left - trailing;
}

/**
 * A text box carries dead space above the cap line and below the baseline, and
 * how much differs per family. Measuring it from the live font lets the layout
 * trim boxes down to their ink, so the header's gaps read equal optically.
 */
function inkOf(el: HTMLElement, text: string) {
	const box = el.getBoundingClientRect();
	const context = textContext();
	if (!box.height || !context) return null;

	const style = getComputedStyle(el);
	context.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
	const metrics = context.measureText(text);
	const ascent = metrics.fontBoundingBoxAscent;
	const descent = metrics.fontBoundingBoxDescent;
	if (!ascent && !descent) return null;

	const lineBox = box.height / (lineRects(el).length || 1);
	const baseline = (lineBox - (ascent + descent)) / 2 + ascent;
	const top = Math.max(0, baseline - metrics.actualBoundingBoxAscent);
	const bottom = Math.max(0, lineBox - (baseline + metrics.actualBoundingBoxDescent));
	return { top, bottom, height: box.height, ink: box.height - top - bottom };
}

/** Long prompts shrink to fit the space left over, rather than ellipsizing. */
function fitPrompt(root: HTMLElement, els: Slots) {
	const el = els.prompt;
	const slot = el?.parentElement;
	if (!el || !slot) return;

	const slotStyle = getComputedStyle(slot);
	let avail =
		slot.clientWidth -
		(parseFloat(slotStyle.paddingLeft) || 0) -
		(parseFloat(slotStyle.paddingRight) || 0);

	// The two nav groups are flex boxes that meet near the centre, so clear
	// their visible content edges rather than their boxes.
	const { navEnd, keysStart } = els;
	if (navEnd && keysStart) {
		const inset = edgeInset();
		const box = root.getBoundingClientRect();
		const centre = (box.left + box.right) / 2;
		const half = Math.min(
			centre - (navEnd.getBoundingClientRect().right + inset),
			keysStart.getBoundingClientRect().left - inset - centre,
		);
		if (half > 40) avail = Math.min(avail, half * 2);
	}

	const style = root.style;
	style.setProperty("--prompt-max", `${avail.toFixed(2)}px`);
	if (avail <= 0) return;

	style.removeProperty("--prompt-size");
	style.removeProperty("--prompt-wrap");
	const base = parseFloat(getComputedStyle(el).fontSize);
	const natural = el.scrollWidth;
	if (!base || !natural || natural <= avail) return;

	const oneLine = base * (avail / natural);
	if (oneLine >= base * 0.62) {
		style.setProperty("--prompt-size", `${oneLine.toFixed(2)}px`);
		return;
	}

	// Too small to hold one line: wrap, and take the largest size that fits a
	// line budget. The budget grows rather than the prompt ever truncating.
	style.setProperty("--prompt-wrap", "normal");
	const apply = (px: number) => style.setProperty("--prompt-size", `${px.toFixed(2)}px`);
	for (const budget of [2, 3, 4]) {
		const fits = (px: number) => {
			apply(px);
			return lineRects(el).length <= budget;
		};
		let low = base * 0.24;
		let high = base;
		if (!fits(low)) continue;
		for (let i = 0; i < 9; i++) {
			const mid = (low + high) / 2;
			if (fits(mid)) low = mid;
			else high = mid;
		}
		apply(low);
		return;
	}
	apply(base * 0.24);
}

/** Track SCENEBENCH out until its right edge lands on BY STARSHOT LABS'. */
function trackBrand(root: HTMLElement, els: Slots) {
	const { brandName, brandSub } = els;
	if (!brandName || !brandSub) return;

	const chars = Math.max(1, (brandName.textContent ?? "").trim().length);
	const target = runWidth(
		brandSub,
		parseFloat(getComputedStyle(brandSub).letterSpacing) || 0,
	);
	root.style.removeProperty("--brand-track");
	const natural = runWidth(brandName, 0);
	if (!natural || !target) return;

	let track = (target - natural) / chars;
	root.style.setProperty("--brand-track", `${track.toFixed(3)}px`);

	const drift =
		target - runWidth(brandName, parseFloat(getComputedStyle(brandName).letterSpacing) || 0);
	if (Math.abs(drift) > 0.1) {
		track += drift / chars;
		root.style.setProperty("--brand-track", `${track.toFixed(3)}px`);
	}
}

function measure(root: HTMLElement, els: Slots) {
	fitPrompt(root, els); // the trims below depend on the fitted size

	const prompt = els.prompt && inkOf(els.prompt, (els.prompt.textContent ?? "").toUpperCase());
	if (prompt) {
		root.style.setProperty("--prompt-trim-top", `${prompt.top.toFixed(2)}px`);
		root.style.setProperty("--prompt-trim-bot", `${prompt.bottom.toFixed(2)}px`);
		root.style.setProperty("--prompt-ink", `${prompt.ink.toFixed(2)}px`);
	}

	// The eyebrow is centred in the nav row, so its own line box is what puts
	// its baseline where it is.
	const eyebrow = els.eyebrow && inkOf(els.eyebrow, els.eyebrow.textContent ?? "");
	if (eyebrow) {
		root.style.setProperty("--eyebrow-trim-bot", `${eyebrow.bottom.toFixed(2)}px`);
		root.style.setProperty("--eyebrow-half", `${(eyebrow.height / 2).toFixed(2)}px`);
	}

	trackBrand(root, els);
}

export function useArenaMetrics(root: RefObject<HTMLElement | null>): ArenaMetrics {
	const els = useRef<Slots>({});
	const frame = useRef(0);
	const promptResize = useRef<ResizeObserver | null>(null);
	const attach = useRef<Partial<Record<Slot, (el: HTMLElement | null) => void>>>({});

	const remeasure = useCallback(() => {
		cancelAnimationFrame(frame.current);
		frame.current = requestAnimationFrame(() => {
			if (root.current) measure(root.current, els.current);
		});
	}, [root]);

	const register = useCallback(
		(slot: Slot) => {
			const attached =
				attach.current[slot] ??
				((el: HTMLElement | null) => {
					if (el) els.current[slot] = el;
					else delete els.current[slot];

					// The prompt drives every trim and the canvas top, so follow
					// its box rather than waiting for a resize.
					if (slot === "prompt") {
						promptResize.current ??= new ResizeObserver(remeasure);
						promptResize.current.disconnect();
						if (el) promptResize.current.observe(el);
					}
					remeasure();
				});
			attach.current[slot] = attached;
			return attached;
		},
		[remeasure],
	);

	useEffect(() => {
		remeasure();
		window.addEventListener("resize", remeasure);
		void document.fonts.ready.then(() => remeasure());
		document.fonts.addEventListener("loadingdone", remeasure);
		const observer = promptResize;
		return () => {
			cancelAnimationFrame(frame.current);
			window.removeEventListener("resize", remeasure);
			document.fonts.removeEventListener("loadingdone", remeasure);
			observer.current?.disconnect();
		};
	}, [remeasure]);

	return useMemo(() => ({ register, remeasure }), [register, remeasure]);
}
