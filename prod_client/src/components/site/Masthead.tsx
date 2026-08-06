import type { ReactNode } from "react";
import Fade from "./Fade";
import MoonLimb from "./MoonLimb";
import Navbar, { ON_PAPER } from "./Navbar";

// HOW FAR THE PAPER HANGS BELOW THE BAR — the room the limb has to fall into,
// and the comp's own expression for it rather than a rounded-off equivalent.
//
// WRITTEN AGAINST THE SAME TOKEN THE LOCKUP IS SIZED FROM, which is what makes it
// hold. The mark is `--spacing-xl * 1.45` and this is `* 0.36`, so bar and belly
// are two readings of one number and the masthead comes out at 1.81 of it plus
// the two paddings — 112px at 1440. Pinned to a `vw` of its own, as it was, the
// belly drifted against the lockup at every width and was 38px where the comp
// wanted 22.
export const BELLY =
	"calc(var(--spacing-xl) * 0.36 + var(--spacing-2xs) - 2px)";

// WHERE THE LABEL SITS, measured from the top of the bar. It rides in the gap the
// two nav groups leave in the middle of the header — not below them — which is
// why this is a small fixed offset and not a fraction of the masthead's height.
const LABEL_TOP = "calc(var(--spacing-2xs) + 12px)";

// THE TWO CAPTIONS ARE PINNED TO OPPOSITE ENDS, not spaced within a column.
//
// They were a flex column with equal flexers above and below the prompt, which
// kept them equidistant from each other — a relationship neither of them actually
// has. The label belongs to the BAR and the prompt belongs to the LIMB, and each
// is anchored to the thing it belongs to: the label a fixed drop from the top
// edge, the prompt a hair off the bottom, where the paper is deepest. Tie them to
// each other instead and the prompt climbs whenever the label moves.
const PROMPT_BOTTOM = "var(--spacing-2xs)";

// ---------------------------------------------------------------------------
// THE PROMPT IS SET TO THE COMP'S OWN MEASUREMENTS, converted rather than copied.
//
// The comp draws this caption as SVG text — a 1600 × 68 viewBox laid out at
// `min(96vw, 1560px)` wide, type at 48 of those units, sitting on an arc whose
// midpoint puts the BASELINE at y = 44, which is 24 units up from the foot of
// that box. We are not drawing an SVG, so what has to be carried across is not
// the drawing but the two numbers a reader can actually see: how big the type is
// and where its baseline lands.
//
// Both convert at one rate — the scale the comp's box is drawn at — so that rate
// is named once and everything else is a multiple of it. This is why the prompt
// was sitting on the limb: the block is pinned at `--spacing-2xs` in both, but
// the comp's SVG carries 24 units of EMPTY BOX under its baseline and our bare
// span carries only the font's descent, so our type landed the best part of half
// its own height lower. It was also a quarter too large, `--text-xl` being a
// type-scale step rather than this drawing's own size.
// ---------------------------------------------------------------------------

// One unit of the comp's viewBox, in CSS pixels: 96vw / 1600, capped where its
// width caps.
const UNIT = "min(0.06vw, 0.975px)";

// 48 units. Comes out at 41.5px against `--text-xl`'s 51.8 at 1440.
//
// BOTH VOICES RIDE IT, which is why it is named for the slot and not for the
// prompt. The leaderboard's name was on `--text-xl` — a step of the type scale,
// picked because it was the nearest one — and came out a quarter larger than the
// caption doing the same job one route away. Two titles in the same slot, at the
// same moment in the same masthead, disagreeing by 10px reads as two pages built
// by different hands. The comp states a size for this slot; that is the size.
const TITLE_SIZE = `calc(48 * ${UNIT})`;

// HOW FAR THE TYPE RIDES UP so its baseline lands where the comp puts it, in `em`
// of the prompt's own size — so it holds at every width without a second copy of
// the scale.
//
// 24 units of clear box under the baseline is half of the 48-unit type, and a
// `leading-none` line box already puts its own baseline 0.1706em above the foot
// of the box. The lift is the difference: 0.5 − 0.1706.
//
// THE 0.1706 IS A PROPERTY OF THE FACE — half of Instrument Serif's descent minus
// ascent — and is MEASURED, not derived: put a zero-height inline-block in the
// line and read where its bottom lands. Re-measure it if the serif is ever
// changed, exactly as the wordmark's ratio is re-measured when its face moves.
const PROMPT_LIFT = "0.3294em";

// THE SAME RULE FOR THE NAME, in the name's own face.
//
// The leaderboard's title sits in the same slot and was getting none of this: its
// baseline landed 0.132em above the foot of the block against the prompt's half an
// em, so it rode 15px lower and finished up on the limb — the block's foot clears
// the paper by 4px, and a `leading-none` box hangs the type most of the way down
// that. The comp has no leaderboard to copy, so what carries across is the
// PROPORTION the comp does state — baseline at half the type's own size above the
// foot — which is a rule about a title in this slot rather than about a caption on
// an arc.
//
// 0.5 − 0.132, and the 0.132 is MANROPE'S, measured on the live element the way
// PROMPT_LIFT's was: drop a zero-height inline-block in the line and read where its
// bottom lands. It is not the serif's number and could not be — this is the whole
// reason the correction is per-voice. Re-measure if the sans is ever changed.
const NAME_LIFT = "0.368em";

// The comp fits the caption to 0.86 of its arc, which is 1308 units of the 1520
// between the arc's ends. Wider than the 64vw that was here, and it is the comp's
// number rather than a guess at one.
const PROMPT_ROOM = `calc(1308 * ${UNIT})`;

/**
 * WHERE THE PROMPT PIVOTS: a point far below the page, so the caption ROLLS.
 *
 * Hand this to whatever carries `prompt-roll-out` / `prompt-settle`, which rotate
 * rather than translate. The angle is meaningless without it — about its own
 * centre, 1.7deg is just a caption tilting; about a point 9,600px down, the same
 * 1.7deg is 285px of travel along an almost flat arc, which is the prompt sliding
 * across the face of the moon.
 *
 * A CONSTANT, AND DELIBERATELY NOT THE LIVE RADIUS — which is the comp's own value
 * and worth defending, because the obvious "correction" here is a real regression.
 * The disc is sized off the window, so its radius is not fixed: 7,672px at 1280 and
 * 28,072px at 2560. Pivot on the true centre and the SAME 1.7deg sweeps an arc of
 * whatever length that radius makes it — 228px on a laptop against 833px on a wide
 * monitor, the prompt hurling itself off the side of the screen on the machines
 * most likely to be showing it.
 *
 * Rotation is a fixed angle; arc length is angle TIMES radius. Only one of the two
 * can be held still across a shape that resizes, and the one that has to be held is
 * the one you can see. 9,600 is the radius at about 1440 — so the roll is exact at
 * the width it was drawn for and stays believable either side of it, which is the
 * right trade for a curve this shallow: at 2560 the prompt travels on an arc a
 * little tighter than the limb, and there is no angle at which that is visible.
 */
export const ROLL_ORIGIN = "50% -9600px";

export default function Masthead({
	label,
	placement = "overlay",
	children,
}: {
	label: string;
	placement?: "overlay" | "flow";
	children: ReactNode;
}) {
	return (
		<div
			className={`pointer-events-none [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
				placement === "overlay"
					? "absolute inset-x-0 top-0"
					: "relative flex-none"
			}`}
			style={{ paddingBottom: BELLY }}
		>
			<MoonLimb />

			<Navbar />

			{/* In the belly, which is paper — so it takes the bar's inverted palette
			    even though it sits outside the bar.

			    ON_PAPER is handed to an element of its own rather than merged into a
			    style literal alongside other properties: spread into another literal
			    it does not survive to the DOM, and the captions silently lose their
			    ink against the paper. */}
			<div className="absolute inset-0" style={ON_PAPER}>
				<Fade enter={700} delay={180} leave={220} className="absolute inset-0">
					<div
						className="absolute inset-x-0 flex justify-center px-lg"
						style={{ top: LABEL_TOP }}
					>
						<span className="font-mono text-2xs tracking-[0.24em] whitespace-nowrap uppercase text-ink-40">
							{label}
						</span>
					</div>

					{/* CENTRED ON THE WINDOW, not laid across it. A full-width flex row
					    would be centred too, but it would also stretch under the whole
					    bar and take the pointer with it; this is only as wide as the
					    prompt it holds. */}
					<div
						className="absolute left-1/2 flex min-w-0 max-w-[64vw] -translate-x-1/2 justify-center"
						style={{ bottom: PROMPT_BOTTOM }}
					>
						{children}
					</div>
				</Fade>
			</div>
		</div>
	);
}

const VOICE = {
	// ITALIC NEEDS A BEARING, upright does not — which is why this is per-voice and
	// not on Title itself. The block below `truncate`s, and `overflow: hidden` cuts
	// at the box edge; an italic face draws its glyphs LEANING PAST their advance
	// width, so the box the browser measures is narrower than the ink it contains
	// and the last letter loses its tail. On the arena that last letter is a closing
	// curly quote — nearly all overhang — so it was the visible casualty.
	//
	// SYMMETRIC, so the centring is untouched. The clip is only ever on the right,
	// but padding one side would walk the prompt half a bearing off centre under a
	// `justify-center` parent, and a title that sits slightly left of the moon is a
	// worse fault than the one being fixed. In `em`, because the overhang scales
	// with the type.
	// TRACKING IS THE COMP'S, not the face's default. It sets this caption at
	// 0.01em, which is small enough to look like nothing and measures ~10px across
	// a short prompt — the whole of the width our version was coming up short by
	// once the size and the baseline agreed.
	prompt: "font-serif italic tracking-[0.01em] px-[0.14em]",
	// NO SIZE HERE — it comes off TITLE_SIZE with the prompt's, below. A `text-*`
	// class alongside that inline `fontSize` would be dead weight at best and a
	// second opinion about the answer at worst.
	name: "font-sans font-black tracking-[-0.015em] uppercase",
} as const;

// SIZE AND LIFT TRAVEL WITH THE VOICE, not with the block. The block is the
// masthead's title slot and both voices sit in it, and both now land their baseline
// half their own size above its foot — but they cannot get there by the same
// number. The correction is the gap between that target and where the FACE already
// puts its baseline in a `leading-none` box, and that second term is a property of
// the typeface: 0.180em for the serif against 0.132em for the sans. Hoisted onto
// the block, one of the two would be riding a measurement of a face it is not set
// in, which is a quarter of a pixel per em of being quietly wrong.
//
// SIZE IS SHARED, and it is the one thing here that is. Both voices take
// TITLE_SIZE, so the two titles are the same height whichever page you arrive on;
// only the lift differs, and only because the faces do. That is the split worth
// holding — one number for what the slot IS, and a per-face correction for what
// each typeface does inside it.
const VOICE_STYLE: Partial<Record<keyof typeof VOICE, React.CSSProperties>> = {
	prompt: {
		fontSize: TITLE_SIZE,
		transform: `translateY(-${PROMPT_LIFT})`,
	},
	name: {
		fontSize: TITLE_SIZE,
		transform: `translateY(-${NAME_LIFT})`,
	},
};

export function Title({
	voice = "prompt",
	className = "",
	children,
}: {
	voice?: keyof typeof VOICE;
	className?: string;
	children: ReactNode;
}) {
	return (
		<span
			style={VOICE_STYLE[voice]}
			className={`block max-w-full truncate leading-none text-ink ${VOICE[voice]} ${className}`}
		>
			{children}
		</span>
	);
}
