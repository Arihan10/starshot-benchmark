"use client";

import {
	useEffect,
	useRef,
	type CSSProperties,
	type ReactNode,
	type Ref,
} from "react";

const HIDE_NATIVE =
	"[scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden";

/**
 * A scrollport that paints its own bar.
 *
 * WHY THIS EXISTS: Firefox (and macOS overlay scrollbars in general) will not
 * draw a hard-edged rectangle through `scrollbar-color` / `scrollbar-width` —
 * those properties recolour the OS widget, they do not replace its geometry.
 * Chromium's `::-webkit-scrollbar` can, Firefox has no equivalent. So the bar
 * is a sibling of the viewport, not a pseudo-element of it: a hard rectangle
 * with square corners, grey at rest and mark-white while held or hovered.
 *
 * THE NATIVE BAR IS HIDDEN on the viewport. Leaving it up would stack two thumbs
 * in Chromium and keep the OS pill in Firefox. Wheel, touch, and keyboard scroll
 * still hit the viewport; the rail is the pointer affordance for dragging.
 */
export default function ScrollBox({
	children,
	className = "",
	viewportClassName = "",
	viewportStyle,
	viewportRef,
}: {
	children: ReactNode;
	className?: string;
	viewportClassName?: string;
	viewportStyle?: CSSProperties;
	viewportRef?: Ref<HTMLDivElement | null>;
}) {
	const port = useRef<HTMLDivElement>(null);
	const track = useRef<HTMLDivElement>(null);
	const thumb = useRef<HTMLDivElement>(null);
	const drag = useRef<{ pointer: number; top: number } | null>(null);

	useEffect(() => {
		const box = port.current;
		const rail = track.current;
		const bar = thumb.current;
		if (!box || !rail || !bar) return;

		const sync = () => {
			const { scrollTop, scrollHeight, clientHeight } = box;
			const span = scrollHeight - clientHeight;
			const overflow = span > 1;
			rail.style.opacity = overflow ? "1" : "0";
			rail.style.pointerEvents = overflow ? "auto" : "none";
			if (!overflow) return;

			const trackH = rail.clientHeight;
			const height = Math.max(28, (clientHeight / scrollHeight) * trackH);
			const top = (scrollTop / span) * (trackH - height);
			bar.style.height = `${height}px`;
			bar.style.top = `${top}px`;
		};

		sync();
		box.addEventListener("scroll", sync, { passive: true });
		const ro = new ResizeObserver(sync);
		ro.observe(box);
		ro.observe(rail);
		const content = box.firstElementChild;
		if (content) ro.observe(content);

		return () => {
			box.removeEventListener("scroll", sync);
			ro.disconnect();
		};
	}, []);

	useEffect(() => {
		const box = port.current;
		const rail = track.current;
		const bar = thumb.current;
		if (!box || !rail || !bar) return;

		const scrollToThumb = (top: number) => {
			const span = box.scrollHeight - box.clientHeight;
			const trackH = rail.clientHeight;
			const height = bar.offsetHeight;
			const max = Math.max(0, trackH - height);
			const clamped = Math.min(max, Math.max(0, top));
			box.scrollTop = max > 0 ? (clamped / max) * span : 0;
		};

		const onMove = (ev: PointerEvent) => {
			const d = drag.current;
			if (!d) return;
			scrollToThumb(d.top + ev.clientY - d.pointer);
		};

		const onUp = (ev: PointerEvent) => {
			if (!drag.current) return;
			drag.current = null;
			bar.removeAttribute("data-active");
			if (bar.hasPointerCapture(ev.pointerId)) {
				bar.releasePointerCapture(ev.pointerId);
			}
		};

		const begin = (ev: PointerEvent, top: number) => {
			ev.preventDefault();
			drag.current = { pointer: ev.clientY, top };
			bar.dataset.active = "";
			bar.setPointerCapture(ev.pointerId);
		};

		const onThumbDown = (ev: PointerEvent) => {
			ev.stopPropagation();
			begin(ev, bar.offsetTop);
		};

		// Click the track to jump — the thumb's share of the rail is the viewport's
		// share of the content, so landing the thumb's centre on the click keeps the
		// mapping honest.
		const onTrackDown = (ev: PointerEvent) => {
			if (ev.target !== rail) return;
			const top =
				ev.clientY -
				rail.getBoundingClientRect().top -
				bar.offsetHeight / 2;
			scrollToThumb(top);
			begin(ev, bar.offsetTop);
		};

		bar.addEventListener("pointerdown", onThumbDown);
		rail.addEventListener("pointerdown", onTrackDown);
		bar.addEventListener("pointermove", onMove);
		bar.addEventListener("pointerup", onUp);
		bar.addEventListener("pointercancel", onUp);

		return () => {
			bar.removeEventListener("pointerdown", onThumbDown);
			rail.removeEventListener("pointerdown", onTrackDown);
			bar.removeEventListener("pointermove", onMove);
			bar.removeEventListener("pointerup", onUp);
			bar.removeEventListener("pointercancel", onUp);
		};
	}, []);

	const setPort = (el: HTMLDivElement | null) => {
		port.current = el;
		if (typeof viewportRef === "function") viewportRef(el);
		else if (viewportRef) viewportRef.current = el;
	};

	return (
		<div className={`relative min-h-0 ${className}`}>
			<div
				ref={setPort}
				className={`h-full min-h-0 overflow-y-auto ${HIDE_NATIVE} ${viewportClassName}`}
				style={viewportStyle}
			>
				{children}
			</div>

			{/* THE RAIL IS OUTSIDE THE VIEWPORT so a content mask (the standings
			    foot fade) cannot paint over it — that was what forced the OS bar
			    back onto the leaderboard. Square corners are a style here, not a
			    hope that the platform will honour `border-radius: 0`. */}
			<div
				ref={track}
				aria-hidden
				className="absolute inset-y-0 right-0 z-20 w-(--scroll-size) opacity-0"
			>
				<div
					ref={thumb}
					className="absolute top-0 right-0 w-full cursor-pointer bg-mark-40 transition-colors duration-quick hover:bg-mark data-active:bg-mark"
				/>
			</div>
		</div>
	);
}
