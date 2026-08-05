"use client";

import { useEffect, useRef, type ReactNode, type RefObject } from "react";
import type { OrbitEngine } from "@/lib/orbit/engine";
import type { OrbitState } from "@/lib/orbit/types";

/**
 * The right-click menu for one addressed object.
 *
 * It owns its own dismissal. The outside-press and Escape listeners have to know
 * what "inside" means, and the only thing that knows that is the element itself —
 * hoisting the ref to the parent just to run the same two listeners there made
 * the shell carry a detail that belongs here. Mounting IS the open state, so the
 * listeners are bound exactly while the menu is up.
 */
export default function ObjectMenu({
	menu,
	engine,
}: {
	menu: NonNullable<OrbitState["contextMenu"]>;
	engine: RefObject<OrbitEngine | null>;
}) {
	const ref = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const onPointerDown = (e: PointerEvent) => {
			if (!ref.current?.contains(e.target as Node)) engine.current?.closeMenu();
		};
		const onKeyDown = (e: KeyboardEvent) => {
			if (e.key === "Escape") engine.current?.closeMenu();
		};
		// Capture phase: the canvas below stops propagation on its own pointer
		// handlers, so a bubbling listener would never see the press that should
		// dismiss this.
		document.addEventListener("pointerdown", onPointerDown, true);
		window.addEventListener("keydown", onKeyDown);
		return () => {
			document.removeEventListener("pointerdown", onPointerDown, true);
			window.removeEventListener("keydown", onKeyDown);
		};
	}, [engine]);

	const hasExtras = menu.hiddenCount > 0 || menu.outlinedCount > 0;
	const left = Math.min(menu.x, window.innerWidth - 200);
	const top = Math.min(menu.y, window.innerHeight - 180);

	return (
		<div
			ref={ref}
			style={{ left, top }}
			onContextMenu={(e) => e.preventDefault()}
			className='fixed z-50 min-w-42.5 select-none rounded-md border border-mark-8 bg-surface/95 p-1 text-xs shadow-2xl backdrop-blur'
		>
			{menu.label && (
				<>
					<div className='mb-1 max-w-60 truncate border-b border-mark-8 px-2 py-1.5 font-semibold text-cyan-200'>
						{menu.label}
					</div>
					<MenuButton onClick={() => engine.current?.toggleMenuTargetHidden()}>
						{menu.hidden ? "show" : "hide"}
					</MenuButton>
					<MenuButton onClick={() => engine.current?.toggleMenuTargetOutline()}>
						{menu.outlined ? "remove outline" : "highlight outline"}
					</MenuButton>
					{hasExtras && <div className='my-1 h-px bg-mark-8' />}
				</>
			)}
			{menu.hiddenCount > 0 && (
				<MenuButton onClick={() => engine.current?.showAllHidden()}>
					show all ({menu.hiddenCount})
				</MenuButton>
			)}
			{menu.outlinedCount > 0 && (
				<MenuButton onClick={() => engine.current?.clearOutlines()}>
					clear outlines
				</MenuButton>
			)}
		</div>
	);
}

function MenuButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
	return (
		<button
			type='button'
			onClick={onClick}
			className='block w-full rounded px-2 py-1.5 text-left text-ink transition hover:bg-cyan-500/20 hover:text-ink'
		>
			{children}
		</button>
	);
}
