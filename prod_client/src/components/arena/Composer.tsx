"use client";

import { useEffect, useRef, useState } from "react";
import Button from "@/components/ui/Button";
import { buildStep } from "./buildSequence";

// #TODO: UI ONLY. Nothing here submits — `onSubmit` is a stub, the suggestions
// only fill the field, and no build is queued. The whole point of this pass is the
// shape and the choreography; the wiring comes when the server can take a prompt.

// Three, and they are PROMPTS rather than categories. "Architecture" would be a
// filter; "a cliffside villa at dusk" is someone showing you the kind of sentence
// this field wants, which is the only thing a blank input actually needs to teach.
const SUGGESTIONS = [
	"a cliffside villa at dusk",
	"a sky-island temple world",
	"a neon noodle alley",
];

// The field has two sizes and everything about it interpolates between them. Held
// here rather than inline so the collapsed and open states are legible as a PAIR —
// the whole component is the difference between these two columns.
const COLLAPSED = {
	width: "290px",
	fontSize: "15.5px",
	rule: "rgb(var(--mark-rgb) / 0.26)",
	placeholder: "want to generate your own?",
};
const OPEN = {
	width: "min(620px, 58vw)",
	fontSize: "31px",
	rule: "rgb(var(--mark-rgb) / 0.92)",
	placeholder: "describe any scene…",
};

// Slightly past 1 on the way out, so the field OVERSHOOTS its target width and
// settles back. A linear-ish ease makes an input that grows look like a box being
// resized; the overshoot is what makes it read as something opening.
const SPRING = "cubic-bezier(0.28, 1.35, 0.42, 1)";

// Milliseconds per character while the collapsed prompt types itself out. Slow
// enough to read as typing rather than as text arriving in chunks, fast enough that
// the whole line is down in about a second — any longer and a viewer who looked
// away comes back to a half-finished sentence and wonders what broke.
const TYPE_MS = 42;

/**
 * "Want to generate your own?" — the invitation under the vote bar.
 *
 * IT IS A LINE OF TEXT UNTIL YOU TOUCH IT. Collapsed, it is 290px of italic serif
 * over a dim rule: quiet enough to sit under the vote bar without competing with
 * it, and shaped like the prompt on the moon rather than like a form, because what
 * it wants from you is a sentence and it should look like the place sentences go.
 *
 * Focused, it becomes the widest thing on the page — twice the width, twice the
 * type — the rule lights up, three example prompts rise into view and the GENERATE
 * button springs in beside it. Nothing is revealed that was not implied; the field
 * just stops being modest.
 *
 * It sits directly under the seam between the two panels, which fades to nothing by
 * that point in its own gradient — so the composer arrives in a part of the page
 * that is already empty and dark, and needs no ground of its own to be legible on.
 */
export default function Composer({
	onOpenChange,
	built = true,
}: {
	/** Whether the build sequence has reached the composer's rule. */
	built?: boolean;
	/**
	 * Fires when the field opens and again when it closes.
	 *
	 * The composer does not own what happens to the rest of the page — it reports
	 * that it has been reached for, and the page decides that the vote should step
	 * aside. Keeping that decision out here means the field can be dropped anywhere
	 * without dragging an opinion about the vote bar along with it.
	 */
	onOpenChange?: (open: boolean) => void;
}) {
	const [open, setOpen] = useState(false);
	const setOpenAndTell = (next: boolean) => {
		setOpen(next);
		onOpenChange?.(next);
	};
	const [text, setText] = useState("");
	const inputRef = useRef<HTMLInputElement>(null);

	// HOW MUCH OF THE INVITATION HAS BEEN TYPED. The collapsed placeholder writes
	// itself out one character at a time behind a blinking caret, which is the
	// cheapest possible way to say "this is somewhere you type" — a rule with a
	// sentence over it is otherwise indistinguishable from a caption, and a viewer
	// who does not know it is a field will never click it.
	//
	// It runs ONCE, when the field is closed and empty. Looping would turn a hint
	// into an animation playing on a loop in the corner of the eye for as long as
	// the page is open, which is the sort of thing people close tabs over.
	const [typed, setTyped] = useState(0);
	const full = COLLAPSED.placeholder;

	// Rewound during RENDER rather than in an effect. Setting state from an effect
	// body schedules a second render every time the field opens or closes, and the
	// first of the two paints the old count against the new state — here that is one
	// frame of a fully-typed line the instant the field collapses.
	const [wasOpen, setWasOpen] = useState(open);
	if (open !== wasOpen) {
		setWasOpen(open);
		setTyped(0);
	}

	// ONE TIMEOUT PER CHARACTER, keyed on the count itself, rather than a single
	// interval running the whole line.
	//
	// The interval version was the bug: its callback closed over the state setter
	// but the EFFECT only ever ran once, so anything that interrupted it — a
	// re-render clearing the timer on cleanup, a fast refresh, a StrictMode double
	// mount cancelling the first pass — left the count wherever it had got to with
	// nothing scheduled to move it again. Stuck at zero, that is a caret blinking on
	// an empty line, which is exactly what it looked like.
	//
	// Scheduling from `typed` makes each character its own effect. If one is
	// cancelled, the next render schedules it again from wherever the count
	// actually is, so the line cannot stall part-written.
	useEffect(() => {
		if (open || typed >= full.length) return;
		const timer = window.setTimeout(() => setTyped((n) => n + 1), TYPE_MS);
		return () => window.clearTimeout(timer);
	}, [open, typed, full.length]);

	const s = open ? OPEN : COLLAPSED;
	// The native placeholder is suppressed while the typewriter is on screen, or
	// the two would print over each other. The field keeps its `aria-label`, so
	// nothing is lost to a screen reader by the visible text being decorative.
	const showTyper = !open && text === "";

	return (
		<div className="pointer-events-auto flex flex-col items-center">
			{/* THE SUGGESTIONS SIT ABOVE THE FIELD, and they are always mounted —
			    laid out, taking their height, and merely invisible when closed. Adding
			    them on focus would grow the column upward at the same moment the field
			    grows sideways, and the two movements fighting is what makes an
			    expanding control feel like it is unpacking rather than opening.

			    They stagger, each 60ms behind the last, so the row arrives as a
			    sequence rather than a block. `pointer-events` follows visibility: a
			    chip you cannot see must not be a chip you can press. */}
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
						// `tabIndex={-1}` so tabbing out of the field leaves the composer
						// rather than walking three chips the viewer cannot see yet.
						tabIndex={-1}
						onMouseDown={(e) => {
							// `mousedown`, not `click`: the field's blur fires first and
							// would collapse the composer out from under the pointer before
							// the click ever landed.
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
				{/* THE TYPEWRITER, laid over the field rather than inside it. A native
				    placeholder cannot be animated a character at a time, and swapping the
				    attribute on every tick would have the browser re-measuring the input
				    twenty-four times a second. This is a plain span pinned over the same
				    box, matched to the field's own type, and it cannot be clicked — the
				    input underneath takes every pointer event, so the illusion never
				    costs the field its hit area. */}
				{showTyper && (
					<span
						aria-hidden
						className="pointer-events-none absolute inset-0 flex items-center justify-center font-serif text-ink-40 italic"
						style={{ fontSize: s.fontSize, paddingBottom: "5px" }}
					>
						{full.slice(0, typed)}
						{/* SIZED TO THE TYPE, not to the box. `self-stretch` made it as
						    tall as the input — a full-height rule standing next to the
						    words, which reads as a divider rather than as a cursor. An
						    em-and-a-bit is what a caret actually is.

						    IN THE ACCENT, AND BRIGHTER THAN THE WORDS. It was `ink-40`,
						    the same value as the placeholder it follows — which is right
						    for text and wrong for a cursor, because a caret that matches
						    its line just reads as the last character. It has to be the
						    one thing on that line still moving after the typing stops,
						    and at 40% white blinking on and off it was too faint to
						    notice at all. The glow is what carries it at 2px wide. */}
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
				{/* A BARE INPUT ON A RULE. No box, no ground, no radius — the only
				    chrome is the line under it, which is what a field looks like when
				    the page it is on is a night sky.

				    Set in the serif, italic, centred: the same voice the prompt on the
				    moon is in, because it is the same kind of sentence and will end up
				    in the same place. */}
				<input
					ref={inputRef}
					value={text}
					onChange={(e) => setText(e.target.value)}
					onFocus={() => setOpenAndTell(true)}
					// Only collapses if nothing was typed. Someone who wrote half a
					// prompt and clicked away has not changed their mind, and shrinking
					// their sentence into a 290px slot to prove a point would be the
					// interface tidying up at the user's expense.
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

				{/* THE RULE, IN TWO HALVES THAT CLOSE INWARD.
				
				    It was the input's own `border-bottom`, which can only fade. The beam
				    that draws this page arrives from the ends and converges on the
				    centre — which is exactly where the seam picks up and carries on down
				    — so the line has to be able to travel, and a border cannot.
				
				    Each half is pinned at the OUTER end and grows toward the middle:
				    `origin-left` on the left, `origin-right` on the right. They meet at
				    the centre, and that meeting point is the seam's head. */}
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

				{/* SPRUNG IN FROM THE SIDE, and absolutely positioned so it takes no
				    part in the layout — the field is centred on the page, and a button
				    in flow beside it would push it off-centre by half its own width
				    every time it appeared. */}
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
						// #TODO: no action yet — this should queue a build on both models
						// and take the viewer to it.
						onClick={() => {}}
					>
						Generate
					</Button>
				</div>
			</div>
		</div>
	);
}
