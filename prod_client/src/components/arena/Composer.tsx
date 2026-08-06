"use client";

import { useEffect, useRef, useState } from "react";
import Button from "@/components/ui/Button";
import { buildStep } from "./buildSequence";

// #TODO: UI only — onSubmit is a stub; wire up to the build queue.

const SUGGESTIONS = [
	"a cliffside villa at dusk",
	"a sky-island temple world",
	"a neon noodle alley",
];

const COLLAPSED = {
	width: "290px",
	fontSize: "15.5px",
	rule: "rgb(var(--mark-rgb) / 0.26)",
	placeholder: "want to generate a scene yourself?",
};
const OPEN = {
	width: "min(620px, 58vw)",
	fontSize: "31px",
	rule: "rgb(var(--mark-rgb) / 0.92)",
	placeholder: "describe any scene…",
};

const SPRING = "cubic-bezier(0.28, 1.35, 0.42, 1)";

const TYPE_MS = 42;

export default function Composer({
	onOpenChange,
	built = true,
}: {
	built?: boolean;
	onOpenChange?: (open: boolean) => void;
}) {
	const [open, setOpen] = useState(false);
	const setOpenAndTell = (next: boolean) => {
		setOpen(next);
		onOpenChange?.(next);
	};
	const [text, setText] = useState("");
	const inputRef = useRef<HTMLInputElement>(null);

	const [typed, setTyped] = useState(0);
	const full = COLLAPSED.placeholder;

	const [wasOpen, setWasOpen] = useState(open);
	if (open !== wasOpen) {
		setWasOpen(open);
		setTyped(0);
	}

	useEffect(() => {
		if (open || typed >= full.length) return;
		const timer = window.setTimeout(() => setTyped((n) => n + 1), TYPE_MS);
		return () => window.clearTimeout(timer);
	}, [open, typed, full.length]);

	const s = open ? OPEN : COLLAPSED;
	const showTyper = !open && text === "";

	return (
		<div className="pointer-events-auto flex flex-col items-center">
			<div
				className="mb-0 flex items-center gap-md transition-all duration-settle ease-out"
				style={{
					opacity: open ? 1 : 0,
					translate: open ? "0 0" : "0 14px",
					pointerEvents: open ? "auto" : "none",
				}}
			>
				{SUGGESTIONS.map((suggestion, i) => (
					<button
						key={suggestion}
						type="button"
						tabIndex={-1}
						onMouseDown={(e) => {
							e.preventDefault();
							setText(suggestion);
							inputRef.current?.focus();
						}}
						className="cursor-pointer font-label text-2xs text-ink-40 transition-[color,translate] duration-quick hover:-translate-y-px hover:text-ink"
						style={{ transitionDelay: open ? `${i * 60}ms` : "0ms" }}
					>
						{suggestion}
					</button>
				))}
			</div>

			<div className="relative flex items-center">
				{showTyper && (
					<span
						aria-hidden
						className="pointer-events-none absolute inset-0 flex items-center justify-center font-serif text-ink-40 italic"
						style={{ fontSize: s.fontSize, paddingBottom: "5px" }}
					>
						{full.slice(0, typed)}
						<span
							className="ml-[2px] inline-block w-[2px] bg-accent"
							style={{
								height: "1.05em",
								boxShadow: "0 0 8px -1px var(--color-accent)",
								animation: "caret-blink 1.1s steps(1) infinite",
							}}
						/>
					</span>
				)}
				<input
					ref={inputRef}
					value={text}
					onChange={(e) => setText(e.target.value)}
					onFocus={() => setOpenAndTell(true)}
					onBlur={() => {
						if (!text.trim()) setOpenAndTell(false);
					}}
					placeholder={showTyper ? "" : s.placeholder}
					spellCheck={false}
					aria-label="Describe a scene to generate"
					className="block bg-transparent text-center font-serif text-ink italic caret-accent outline-none placeholder:text-ink-40"
					style={{
						width: s.width,
						fontSize: s.fontSize,
						borderBottom: "1.5px solid transparent",
						padding: "4px 10px 9px",
						transition: `width 500ms ${SPRING}, font-size 500ms ${SPRING}`,
					}}
				/>

				<span
					aria-hidden
					className="pointer-events-none absolute inset-x-0 bottom-0 flex"
					style={{ height: "1.5px" }}
				>
					<span
						className="h-full flex-1 origin-left transition-[scale,background-color]"
						style={{
							backgroundColor: s.rule,
							scale: built ? "1 1" : "0 1",
							...buildStep("rule", built),
						}}
					/>
					<span
						className="h-full flex-1 origin-right transition-[scale,background-color]"
						style={{
							backgroundColor: s.rule,
							scale: built ? "1 1" : "0 1",
							...buildStep("rule", built),
						}}
					/>
				</span>

				<div
					className="absolute top-1/2 left-full ml-md"
					style={{
						opacity: open ? 1 : 0,
						translate: "0 -50%",
						scale: open ? "1" : "0.6",
						pointerEvents: open ? "auto" : "none",
						transition: `opacity 300ms ease, scale 400ms cubic-bezier(0.3, 1.5, 0.4, 1)`,
					}}
				>
					<Button
						variant="solid"
						shape="standalone"
						sweep
						tabIndex={-1}
						// #TODO: no action yet — should queue a build on both models.
						onClick={() => {}}
					>
						Generate
					</Button>
				</div>
			</div>
		</div>
	);
}
