"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { RefObject } from "react";

const subscribeFullscreen = (onChange: () => void) => {
	document.addEventListener("fullscreenchange", onChange);
	return () => document.removeEventListener("fullscreenchange", onChange);
};

const subscribeNever = () => () => {};
const getSupported = () => typeof document.exitFullscreen === "function";

const serverSnapshot = () => false;

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

	const enter = useCallback(() => {
		const el = ref.current;
		if (!el || document.fullscreenElement === el) return Promise.resolve();
		return el.requestFullscreen().catch(() => {});
	}, [ref]);

	const exit = useCallback(() => {
		if (!ref.current || document.fullscreenElement !== ref.current)
			return Promise.resolve();
		return document.exitFullscreen().catch(() => {});
	}, [ref]);

	const toggle = useCallback(() => {
		void (document.fullscreenElement === ref.current ? exit() : enter());
	}, [ref, enter, exit]);

	return { isFullscreen, supported, enter, exit, toggle };
}
