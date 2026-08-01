"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { RefObject } from "react";

// The document IS the store. Fullscreen can be entered and left by routes we
// never see — the Esc key, the browser's own chrome, another element on the page
// taking over — so a locally-held boolean drifts out of sync with all of them.
// Subscribing means the icon always describes what is actually on screen.
const subscribeFullscreen = (onChange: () => void) => {
	document.addEventListener("fullscreenchange", onChange);
	return () => document.removeEventListener("fullscreenchange", onChange);
};

// Support is fixed for the life of the document, so there is nothing to listen to.
const subscribeNever = () => () => {};
const getSupported = () => typeof document.exitFullscreen === "function";

// Rendered on the server there is no document and nothing is fullscreen. Both
// stores agree on that, so the first client render matches the markup.
const serverSnapshot = () => false;

/**
 * Fullscreen for one element, read back from the browser rather than tracked by
 * us.
 *
 * `supported` is reported rather than assumed so a caller can omit the control
 * entirely; iPhone Safari has no element fullscreen at all, and a button that
 * cannot work is worse than no button.
 */
export function useFullscreen(ref: RefObject<HTMLElement | null>) {
	const getIsFullscreen = useCallback(
		() => document.fullscreenElement === ref.current,
		[ref],
	);
	const isFullscreen = useSyncExternalStore(
		subscribeFullscreen,
		getIsFullscreen,
		serverSnapshot,
	);
	const supported = useSyncExternalStore(subscribeNever, getSupported, serverSnapshot);

	const toggle = useCallback(() => {
		const el = ref.current;
		if (!el) return;
		// Both calls reject rather than throw (a denied permission policy, a gesture
		// the browser did not count). Nothing here can recover from that, and an
		// unhandled rejection in the console is worse than a button that did
		// nothing, so both are swallowed deliberately.
		if (document.fullscreenElement === el) void document.exitFullscreen().catch(() => {});
		else void el.requestFullscreen().catch(() => {});
	}, [ref]);

	return { isFullscreen, supported, toggle };
}
