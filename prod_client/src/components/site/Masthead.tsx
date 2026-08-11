"use client";

import type { ReactNode } from "react";
import { useLayoutEffect, useRef, useState } from "react";
import Fade from "./Fade";
import MoonLimb from "./MoonLimb";
import Navbar, { ON_PAPER } from "./Navbar";

// ---------------------------------------------------------------------------
// THE PROMPT IS SET TO THE COMP'S OWN SIZE: 48 units of a 1600-wide viewBox laid
// out at `min(96vw, 1560px)`. Vertical place is not a copied number — the title
// sits in the well below the label (see the caption well below).
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

// THE PROTRUDING MOON is struck through three points: the arc STARTS and ENDS at
// the two ends of the span below, and its BOTTOM is the dip under the caption.
// The circle through those is the lip, and the type rides that same circle,
// which is what makes the caption read as sitting ON the moon rather than merely
// near it.
//
// THE DIP IS THE CONSTANT AND THE RADIUS IS WHAT GIVES. How far the lip reaches
// into the page is the masthead's silhouette, so it is held to a token: a wide
// chord over that fixed dip comes out a flatter circle, a narrow one tighter.
const MOON_SAG = "calc(var(--spacing-xl) * 0.62)";

// THE TWO ENDS ARE THE CAPTION'S, PLUS THIS MARGIN — and the margin is a share
// of the TYPE, not a length. The words either side of it scale with the window,
// so air measured in pixels is air that grows relative to everything around it
// as the window narrows.
//
// WHICH IS WHAT THE OLD LIP DID. It was a floor, `2√(s(2r − s))` on a hardcoded
// r = 487 and s = 33.9, and comes out at 357.04px with no `vw` anywhere in it —
// one length at every width. Holding that against a caption that shrinks means
// the air has to open up to fill the difference: 28.8px a side at 1625 and up,
// 45.8 at 1440, 60.6 at 1280, and still climbing until the lip hits the nav
// clusters and is cut back all at once. The moon read as growing while the
// window shrank.
//
// 0.6148em IS THAT SAME AIR at the size the type settles to: the scale clamps at
// 46.8px from 1625 up, where the old lip's own arithmetic leaves 28.77px a side.
// So nothing moves at those widths — the masthead is untouched — and below them
// the margin now comes in with the type instead of opening out against it.
const MOON_END_MARGIN_EMS = 0.6148;

// THE SHORTEST THE LIP IS DRAWN, in title-size ems. The dip is a constant and
// the chord is the caption's, so the fewer the letters the smaller the circle
// they sit on: FAQ is three of them and came out on an 84px radius against the
// arena's 386px, wrapped hard enough to read as a badge rather than a masthead.
//
// ABOUT IS THE FLOOR. It is the shortest title that still reads right, so a
// title under it borrows its chord instead of striking a tighter one — same
// curve, same full dip, just more air around fewer letters. 4.7449em is that
// chord: "ABOUT" runs 3.5153em in the name voice, plus a margin at either end.
//
// In ems for the same reason the margin is: the lip has to come in with the
// type rather than hold a pixel width as the window narrows.
const MOON_MIN_CHORD_EMS = 4.7449;

// AND THE TYPE RIDES THAT SAME CIRCLE. Held to one radius instead, the words
// bend by the amount the homepage's arc bends — right on the homepage and wrong
// everywhere else, because the lip's radius falls out of whatever chord the
// title asks for: 130px on FAQ against 613px on the leaderboard. FAQ's three
// letters came out visibly straight over a rounded moon. The caption is on the
// arc that is drawn beneath it, so it curves ALONG the lip by construction.

/** Air between a chord end and the nearest nav control. */
const MOON_EDGE_GAP = 12;

// WHERE THE LABEL SITS — the origin's own drop from the top of the masthead.
// The prompt lives in the well BELOW this; moving the prompt never retunes it.
const LABEL_TOP = "calc(var(--spacing-2xs) + 12px)";

// Only the roll pivot reads this, and only until the limb has been measured.
const ROLL_RADIUS_FALLBACK = 9600;

/** Arc length of the prompt roll, in CSS pixels — held still across widths. */
const ROLL_ARC_PX = 285;

export type MastheadPrompt = {
    /** The lip's own circle, px — the caption is set on it, so the words curve along the moon. */
    promptRadius: number;
    /** Transform origin that puts the pivot on that circle's centre. */
    rollOrigin: string;
    /** Angle whose arc length is ROLL_ARC_PX on that radius. */
    rollDeg: number;
};

/**
 * Roll angle (degrees) for a constant arc length on the caption's radius.
 * Holding the ANGLE still made the prompt hurl off-screen on wide monitors;
 * holding the DISTANCE keeps the motion the same size you can see.
 */
export function rollDegrees(radius: number): number {
    const r = Math.max(radius, 1);
    return (ROLL_ARC_PX / r) * (180 / Math.PI);
}

/** @deprecated Prefer MastheadPrompt.rollOrigin from the render-prop context. */
export const ROLL_ORIGIN = `50% ${-ROLL_RADIUS_FALLBACK}px`;

function Caption({ children }: { children: ReactNode }) {
    return (
        <span className="font-mono text-2xs tracking-[0.24em] whitespace-nowrap uppercase text-ink-40">
            {children}
        </span>
    );
}

export default function Masthead({
    label,
    placement = "overlay",
    captionWidth,
    children,
}: {
    label: string;
    placement?: "overlay" | "flow";
    /**
     * The caption's own measured width in px, the words alone. The lip is struck
     * to it plus MOON_END_MARGIN_DIPS at either end, capped by the berth between
     * the nav clusters.
     */
    captionWidth?: number;
    children: ReactNode | ((ctx: MastheadPrompt) => ReactNode);
}) {
    const shellRef = useRef<HTMLDivElement>(null);
    const leftClusterRef = useRef<HTMLDivElement>(null);
    const rightClusterRef = useRef<HTMLDivElement>(null);
    const limbRef = useRef<HTMLDivElement>(null);
    const typeRef = useRef<HTMLSpanElement>(null);
    const [moon, setMoon] = useState({
        left: 0,
        width: 0,
        sag: 0,
        radius: ROLL_RADIUS_FALLBACK,
    });

    // Chord centred on the screen. The BERTH is its ceiling — the tighter of
    // (mid → reading cluster end, mid → offer start) less the edge gap, read
    // straight from the clusters rather than as a fraction of one side — so a
    // caption long enough to reach the controls stops there and shrinks its own
    // type to the arc rather than running underneath them.
    useLayoutEffect(() => {
        const shell = shellRef.current;
        const leftCluster = leftClusterRef.current;
        const rightCluster = rightClusterRef.current;
        if (!shell || !leftCluster || !rightCluster) return;

        const sync = () => {
            const s = shell.getBoundingClientRect();
            const mid = s.left + s.width / 2;

            const nav = leftCluster.querySelector("nav");
            const last = nav?.lastElementChild ?? leftCluster;
            const edge = last.getBoundingClientRect();
            const padR =
                Number.parseFloat(getComputedStyle(last).paddingRight) || 0;
            const leftEnd = edge.right - padR;
            const rightStart = rightCluster.getBoundingClientRect().left;
            const berth =
                2 *
                Math.max(
                    0,
                    Math.min(mid - leftEnd, rightStart - mid) - MOON_EDGE_GAP,
                );

            // Both tokens resolved off the one probe: MOON_SAG and TITLE_SIZE are
            // clamps on `vw`, and this is the only place either is evaluated.
            const probe = typeRef.current;
            const typePx = probe
                ? Number.parseFloat(getComputedStyle(probe).fontSize)
                : 0;
            const fullSag = probe?.getBoundingClientRect().height ?? 0;

            const endMargin = MOON_END_MARGIN_EMS * typePx;
            // THE TWO ENDS: the caption's own, one margin further out at each,
            // and never closer together than About's — see MOON_MIN_CHORD_EMS.
            const width = Math.min(
                berth,
                Math.max(
                    MOON_MIN_CHORD_EMS * typePx,
                    (captionWidth ?? 0) + 2 * endMargin,
                ),
            );
            const half = width / 2;
            const left = mid - half - s.left;
            const sag = fullSag;
            // Circle through chord + sagitta — the same construction MoonLimb
            // uses, so the caption is set on the arc it actually sits over.
            const radius =
                sag > 0
                    ? (half * half + sag * sag) / (2 * sag)
                    : ROLL_RADIUS_FALLBACK;
            setMoon((prev) =>
                prev.left === left &&
                prev.width === width &&
                Math.abs(prev.sag - sag) < 0.25 &&
                Math.abs(prev.radius - radius) < 0.5
                    ? prev
                    : { left, width, sag, radius },
            );
        };

        sync();
        document.fonts?.ready.then(sync).catch(() => {});
        const observer = new ResizeObserver(sync);
        observer.observe(shell);
        observer.observe(leftCluster);
        observer.observe(rightCluster);
        const limb = limbRef.current;
        if (limb) observer.observe(limb);
        return () => observer.disconnect();
        // Re-bind when the limb host mounts (it only exists once width > 0).
    }, [moon.width, captionWidth]);

    const prompt: MastheadPrompt = {
        promptRadius: moon.radius,
        rollOrigin: `50% ${-moon.radius}px`,
        rollDeg: rollDegrees(moon.radius),
    };
    const body = typeof children === "function" ? children(prompt) : children;

    const frame = `pointer-events-none [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
        placement === "overlay"
            ? "absolute inset-x-0 top-0"
            : "relative flex-none"
    }`;

    // STRAIGHT BAR + CENTRED MOON LIP. Fill is `mark` — the same ground the vote
    // slabs use (`Button` solid). Paper aliases mark at :root, so ON_PAPER and
    // the bar stay one light.
    //
    // THE LIP HANGS; IT DOES NOT PUSH. In flow it added the dip to the masthead's
    // height and the arena started below that — so left and right of the disc the
    // page ground showed as a black bar between the white navbar and the scene.
    // Absolute under the bar, the limb paints over the arena and the empty berth
    // stays clear.
    return (
        <div data-masthead className={`${frame} z-20`}>
            <div ref={shellRef} className="relative">
				{/* TITLE_SIZE and MOON_SAG, resolved. Both are clamps on `vw` and
				    both are arithmetic below — the margin is a share of the type,
				    the circle is struck through the dip — so the numbers come back
				    from CSS rather than being worked out a second time in JS. */}
				<span
					ref={typeRef}
					aria-hidden
					className="pointer-events-none invisible absolute w-0"
					style={{ fontSize: TITLE_SIZE, height: MOON_SAG }}
				/>
                <div className="relative z-10 bg-mark">
                    <Navbar
                        leftClusterRef={leftClusterRef}
                        rightClusterRef={rightClusterRef}
                    />
                </div>

                {moon.width > 0 && moon.sag > 0 && (
                    <div
                        ref={limbRef}
                        className="absolute top-full -mt-px bg-transparent"
                        style={{
                            width: moon.width,
                            height: moon.sag,
                            left: moon.left,
                        }}
                    >
                        {/* The measured dip, not the host's box: `offsetHeight`
                            is an integer and would round the limb onto a
                            slightly different circle from the type's. */}
                        <MoonLimb
                            sag={moon.sag}
                            chord={(width) => width}
                            shade={false}
                        />
                    </div>
                )}

                {/* Extends into the belly so the prompt can sit on the face of
				    the moon, even though the lip no longer stretches the shell. */}
                <div
                    className="absolute inset-x-0 top-0 z-20"
                    style={{
                        ...ON_PAPER,
                        bottom: `calc(-1 * ${MOON_SAG})`,
                    }}
                >
                    <Fade
                        enter={700}
                        delay={180}
                        leave={220}
                        className="absolute inset-0"
                    >
                        <div
                            className="absolute inset-x-0 bottom-0 flex flex-col px-lg"
                            style={{ top: LABEL_TOP }}
                        >
                            <div className="flex flex-none justify-center">
                                <Caption>{label}</Caption>
                            </div>
                            <div className="flex min-h-0 min-w-0 flex-1 items-center justify-center">
                                {/* Cap to the moon chord — a vw ceiling let the prompt outgrow
								    the face, so the arc math broke and the caption jumped. */}
                                <div
                                    className="flex w-full min-w-0 justify-center"
                                    style={{
                                        maxWidth:
                                            moon.width > 0
                                                ? moon.width
                                                : undefined,
                                    }}
                                >
                                    {body}
                                </div>
                            </div>
                        </div>
                    </Fade>
                </div>
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

// SIZE IS SHARED across voices. Both titles take TITLE_SIZE so the slot is the
// same height whichever page you arrive on. The old per-face baseline lift is
// gone: the prompt is centred between the label and the moon's foot by its box
// edges, and a translateY would steal from the gap above and give it to the gap
// below.
const VOICE_STYLE: Partial<Record<keyof typeof VOICE, React.CSSProperties>> = {
    prompt: { fontSize: TITLE_SIZE },
    name: { fontSize: TITLE_SIZE },
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
