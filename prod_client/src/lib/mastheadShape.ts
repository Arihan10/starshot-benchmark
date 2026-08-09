"use client";

import { useSyncExternalStore } from "react";
import { DEBUG_ENABLED } from "./flags";

/**
 * Which figure the masthead is cut to.
 *
 * The bar's controls run STRAIGHT across and always will — a row of type bent to
 * follow the limb reads as a fault rather than as a flourish. So the question is
 * only ever what the paper does around them, and these are the five answers worth
 * comparing: moon from the top corners, float clear of the edges, give the arc to
 * the arena instead of the nav, feather the arc away, or stand it up.
 */
export type MastheadShape = "limb" | "island" | "split" | "veil" | "flat";

export const SHAPES: { key: MastheadShape; label: string; note: string }[] = [
	{ key: "limb", label: "Limb", note: "Moon from the top corners" },
	{ key: "island", label: "Island", note: "Slab floats, prompt on ground" },
	{ key: "split", label: "Split", note: "Flat bar, arena on a tongue" },
	{ key: "veil", label: "Veil", note: "Arc feathered into the page" },
	{ key: "flat", label: "Flat", note: "Arc stood up, edge shaded" },
];

export const STOCK: MastheadShape = "limb";

const STORE = "scenebench:masthead-shape";

const readers = new Set<() => void>();

function subscribe(onChange: () => void) {
	readers.add(onChange);
	return () => void readers.delete(onChange);
}

const known = (value: string | null): MastheadShape =>
	SHAPES.some((shape) => shape.key === value)
		? (value as MastheadShape)
		: STOCK;

const stored = () => known(window.localStorage.getItem(STORE));

export function setMastheadShape(shape: MastheadShape) {
	window.localStorage.setItem(STORE, shape);
	for (const onChange of readers) onChange();
}

/**
 * The shape in force, live.
 *
 * Served as STOCK on the server and for the first paint of a hydrating client, so
 * the markup both sides produce agrees; React re-renders with the stored pick the
 * moment hydration is done. A production build never reads the store at all — the
 * flag is compiled out, so the experiment cannot escape the dev build.
 */
export function useMastheadShape(): MastheadShape {
	const picked = useSyncExternalStore(subscribe, stored, () => STOCK);
	return DEBUG_ENABLED ? picked : STOCK;
}
