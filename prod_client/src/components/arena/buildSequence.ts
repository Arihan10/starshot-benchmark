/**
 * The masthead's ray building the controls, and taking them apart again.
 *
 * ONE TIMELINE, AND THE LINE IS ALWAYS THE THING MOVING. A beam runs out of the moon
 * down the seam; at the vote bar it turns and opens to both corners; that rule then
 * travels down the bar and the buttons appear in its wake; the composer arrives; its
 * own rule closes inward from both ends; and where the two halves meet, the seam
 * picks up and finishes to the floor.
 *
 * WHAT THIS REPLACED, and why it was wrong: the bar used to fade up from below while
 * the beam merely arrived nearby. That is a cue firing an unrelated animation — the
 * beam was decoration and the reveal was a dissolve wearing its costume. Here every
 * stage is the same stroke continuing: it turns, it descends, it converges. Nothing
 * fades in from a direction the light never came from.
 *
 * EVERY START IS DERIVED, NOT TYPED. This was a table of hand-written `at` values,
 * and the bug that table makes inevitable duly appeared: the ray ran 0–440ms while
 * the horizontal opened at 400, so the line began splitting out of a point the beam
 * had not reached yet. Two independent numbers cannot express "after" — they can
 * only happen to agree, until one is edited. Below, a stage's start IS the previous
 * stage's end, so the handover is a property of the structure and there is no
 * arrangement of these durations that can put an effect before its cause.
 *
 * THE REVERSE IS MIRRORED, NOT REPLAYED. Running the same delays backwards would
 * have the ray recede first and the composer dissolve into a line that had already
 * gone, so the collapse chains through the SAME stages in the opposite order — last
 * built, first removed — which is what makes it read as the ray pulling the
 * structure back with it. It is quicker than the build, because arriving is an
 * announcement and leaving is not.
 */
export type BuildPart =
	| "ray"
	| "beam"
	| "wipe"
	| "composer"
	| "rule"
	| "tail";

// 1. THE BEAM leaves the moon and runs down the seam to the vote bar's top edge.
//    The longest leg: it crosses most of the window, and speed is what makes it read
//    as light rather than as a line being switched on.
const RAY = 400;

// 2. IT TURNS, AND THEN IT STAYS. A rule opens out of the point of contact to both
//    corners, along the bar's top edge, and holds there.
//
//    THE LINE DOES NOT TRAVEL. It used to descend the bar's height with the buttons
//    uncovered in its wake, which sounds like drawing and never read as it — a line
//    crossing a shape only ever looks like a shutter sliding off something already
//    finished. It now marks the edge the bar grows FROM, and is cut once the bar has
//    a top edge of its own to take over: see `vote-rule-hide`, stepped rather than
//    eased so it is never a dissolve.
const OPEN = 260;

// 3. THE BAR GROWS DOWN from that line. Its own movement, on its own curve, rather
//    than a wake tied frame-for-frame to something crossing it.
const REVEAL = 360;

// 5. The composer arrives under it.
const COMPOSER = 240;

// 6. ITS RULE CLOSES INWARD, one half from each end, converging on the centre — which
//    is where the seam picks up again. The convergence point IS the seam's head, so
//    the two read as one stroke handing over.
const CRULE = 300;

// 7. And carries on down to the floor.
const TAIL = 220;

/** How long the beam's composite animation runs. */
export const BEAM_MS = OPEN + REVEAL;

/**
 * Where the beam's beats fall inside that animation, as percentages — the keyframes
 * in globals.css are written against these, so retiming a beat here retimes the
 * motion rather than silently desynchronising it from its own stops.
 */
export const BEAM_OPEN_PCT = (OPEN / BEAM_MS) * 100;

/** The curve the bar grows on. Decelerating, so it arrives and settles. */
export const REVEAL_EASING = "cubic-bezier(0.22, 0.75, 0.3, 1)";

/** How much quicker the collapse runs than the build. */
const OUT = 0.55;

// Forward: each stage begins where the last one ended.
const AT_RAY = 0;
const AT_BEAM = AT_RAY + RAY;
const AT_WIPE = AT_BEAM + OPEN;
const AT_COMPOSER = AT_WIPE + REVEAL;
const AT_RULE = AT_COMPOSER + COMPOSER;
// THE ONE STAGE THAT DOES NOT WAIT FOR THE ONE BEFORE IT, and deliberately.
//
// The tail is the seam picking up again below the composer, and the thing directly
// above it is the composer's RULE — so it starts when that rule starts, not when it
// has finished converging. Chained (`AT_RULE + CRULE`) it landed at 1560 ms, by
// which point the composer's typewriter had been running for half a second and the
// vertical read as an afterthought arriving late rather than as the same stroke
// continuing. Together they read as the horizontal handing over to the vertical,
// which is what they are.
const AT_TAIL = AT_RULE;

// Backward. NOT the forward chain walked in reverse, and that distinction is the
// whole of it.
//
// Reversing the order outright gives "last built, first removed", which sounds right
// and puts the vote bar fifth in a queue. But entering the composer does not take the
// composer apart — `composerBuilt` ignores `composing` on purpose, so the field you
// just clicked into survives its own focus — and it does not take its rule apart
// either. The bar was therefore waiting out three stages that never ran, and the
// click did nothing for the better part of half a second.
//
// So the bar leads, immediately, and the tail goes with it rather than ahead of it:
// nothing the user is not looking at gets to hold up the thing they just touched.
// The beam holds full width while the bar closes up into it and then un-opens, which
// is the build's own two beats in reverse.
const BACK_WIPE = 0;
const BACK_BEAM = 0;
const BACK_TAIL = 0;
const BACK_RAY = BACK_BEAM + BEAM_MS * OUT;

// The composer's own teardown is a different event — it runs when the round ends,
// not when the field is entered — so it keeps its own small chain: the rule closes,
// then the composer follows it out.
const BACK_RULE = 0;
const BACK_COMPOSER = BACK_RULE + CRULE * OUT;

const SEQUENCE: Record<BuildPart, { at: number; back: number; dur: number }> = {
	ray: { at: AT_RAY, back: BACK_RAY, dur: RAY },
	beam: { at: AT_BEAM, back: BACK_BEAM, dur: BEAM_MS },
	// Starts where the line finishes opening, so the bar grows from an edge that is
	// already drawn rather than from nothing.
	wipe: { at: AT_WIPE, back: BACK_WIPE, dur: REVEAL },
	composer: { at: AT_COMPOSER, back: BACK_COMPOSER, dur: COMPOSER },
	rule: { at: AT_RULE, back: BACK_RULE, dur: CRULE },
	tail: { at: AT_TAIL, back: BACK_TAIL, dur: TAIL },
};

/**
 * The `transition` and `transition-delay` a part should carry right now.
 *
 * `building` is the direction. Everything else about a part — what property it
 * animates, what its two end states are — stays at the call site, because that is
 * where it can be read next to the thing it moves.
 */
export function buildStep(part: BuildPart, building: boolean) {
	const { at, back, dur } = SEQUENCE[part];
	return {
		transitionDuration: `${Math.round(building ? dur : dur * OUT)}ms`,
		transitionDelay: `${Math.round(building ? at : back)}ms`,
		// Decelerating on the way out so each part arrives and settles; a touch of
		// acceleration on the way back so the collapse feels drawn rather than
		// merely undone.
		// The bar's growth has its own curve — see REVEAL_EASING.
		transitionTimingFunction:
			part === "wipe"
				? REVEAL_EASING
				: building
					? "cubic-bezier(0.16, 0.84, 0.28, 1)"
					: "cubic-bezier(0.5, 0, 0.75, 0.35)",
	} as const;
}

/** How long the whole build runs, for anything that needs to wait it out. */
export const BUILD_MS = AT_TAIL + TAIL;
