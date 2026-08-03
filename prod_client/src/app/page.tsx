"use client";

import { useCallback, useState, type ReactNode } from "react";
import ScenePanel, {
    SOLO_EASING,
    SOLO_TRANSITION_MS,
} from "@/components/arena/ScenePanel";
import VoteBar from "@/components/arena/VoteBar";
import CurvedPrompt from "@/components/CurvedPrompt";
import Moon from "@/components/Moon";
import { LEFT_VOTE_SHARE, LOCAL_CELLS } from "@/lib/localScenes";

// SceneBench's comparison canvas: the same prompt-shaped task built by two
// different LLMs, side by side, each one orbitable on its own.
//
// The two cells are fixed for now (lib/localScenes) — this is the experiment we
// are looking at, not a picker. Scene selection, the A/B voting UI and the rest
// of the real site get built onto this.
//
// The builds carry NO attribution on screen. Which model made which is the
// question being asked, so naming them next to the render answers it for the
// viewer and poisons the comparison; the cells still know (LocalCell.model), and
// the reveal belongs after a vote rather than before one.

// #TODO: hard-coded. The prompt belongs to the run that produced these cells —
// the pipeline already stores it per slot (runs/<run>/prompts) — so it should
// travel WITH the cell pair rather than being restated here, or the header and
// the scenes under it can drift apart silently. Held in natural case, not caps:
// the shouting is styling, and a real prompt arrives as a sentence.
const PROMPT = "A super mario style platformer level";

// How far the moon's lower limb reaches past the navbar — and, because the disc is
// clipped to this box, the whole of how much moon there is to see.
//
// IT IS ALSO THE PROMPT'S ROOM. The prompt's baseline is a circle inset inside the
// limb, so its lowest point sits exactly that inset above the disc's lowest point:
// a shallow band leaves the line pinned against the label above it with nowhere to
// go. Reaching further down buys the cap height the prompt needs, and buys it
// where the moon is widest.
const MOON_PROTRUDE = "clamp(40px,6.5vh,84px)";

// THE PROMPT TAKES NO GEOMETRY OF ITS OWN. It is given the moon's diameter and
// anchored exactly as the disc is, and its baseline is then struck from the same
// centre — see CurvedPrompt. Curvature that "matches" by having a similar radius
// is not the same thing and does not survive: the arc used to hang from the top
// of the masthead while the disc hung from the bottom, so the two were circles
// about different points, and the line drifted off the limb as it crossed the
// middle however carefully the radius was tuned.

// Capped in BOTH directions. The vw term keeps the arc in proportion on a wide
// display, and the ceiling stops it flattening into a straight edge on an
// ultrawide one — past a certain size a circle cropped to a 100px band has no
// visible curvature left, and the moon just looks like a light grey bar.
//
// The UPPER bound is also what keeps the disc clear of the mark and the nav. A
// wider moon puts its limb straight through a nav label, and a word that is half
// dark-on-moon and half light-on-black reads as broken rather than as blended —
// the prompt in the middle is what the moon is here to sit behind, so the disc is
// sized to hold that and nothing else.
const MOON_DIAMETER = "min(54vw, 900px)";

type Side = "a" | "b";

// What the winner gains and the loser drops. A fixed swing rather than a real Elo
// K-factor calculation: the ratings here are placeholders, so computing an exact
// expected score off them would be arithmetic performed on fiction.
// #TODO: the server owns this — it holds the ratings, and it is the only place
// that can apply the result of a vote to them.
const ELO_SWING = 12;

export default function Page() {
    // The whole round is one piece of state. Everything downstream — who glows,
    // who shatters, which numbers count up — is derived from it, so there is no
    // way for the panels to disagree about what happened.
    const [vote, setVote] = useState<Side | null>(null);

    // Which side has been stepped into, if either. A walkthrough is a place you
    // are standing, not a thumbnail: half a row is the wrong amount of frame for
    // it, and the scene beside it is now a distraction from the one you chose. So
    // entering takes the whole row and leaving gives it straight back — the panels
    // are never unmounted, so the pair returns exactly as it was left.
    //
    // Reported BY THE ENGINE rather than latched on the click that entered: it is
    // the camera leaving the orbit that means "inside", and it leaves on more than
    // a click (and comes back on Escape, on the map, on a fly-out). One source of
    // truth, so the layout cannot get stuck open.
    const [toured, setToured] = useState<Side | null>(null);
    // Per side and stable, because the viewer re-runs its report whenever this
    // identity changes; a closure rebuilt each render would re-fire it every render.
    const onTourA = useCallback(
        (inside: boolean) =>
            setToured((cur) => (inside ? "a" : cur === "a" ? null : cur)),
        [],
    );
    const onTourB = useCallback(
        (inside: boolean) =>
            setToured((cur) => (inside ? "b" : cur === "b" ? null : cur)),
        [],
    );

    // #TODO: there is one fixed pairing, so this replays the round rather than
    // fetching another. It becomes "ask the server for the next pair" once
    // pairings are served.
    //
    // STABLE, and it has to be: the countdown between the two results schedules
    // itself against this, so a new function on every render would cancel and
    // restart the clock each time the page re-rendered — and it would never reach
    // the end.
    const nextPair = useCallback(() => setVote(null), []);

    return (
        <main className="relative h-dvh overflow-hidden bg-black">
            {/* THE SCENES ARE THE PAGE. They fill the frame edge to edge and
                everything else floats over them — no band of chrome holds a strip
                of the viewport back, and the builds are the whole picture rather
                than two thumbnails under a masthead. Everything below this layer is
                an overlay, and each one hands back the pointer everywhere it is not
                actually offering a control, so the scenes stay draggable to their
                own edges. */}
            {/* THE ROW SLIDES; THE PANELS ONLY RESIZE. Entering the first panel is
                enough on its own — it grows to the full row and shoulders the second
                one off the far edge. Entering the SECOND cannot work that way: it
                would grow to the right, off the screen, with the first still sitting
                in front of it. So the row itself travels one panel's width, which
                carries the first out of frame the way it left and brings the second
                up to the near edge. One transform, and the two directions stay
                symmetrical.

                Along the MAIN AXIS, whichever it is: the panels stack below `md`, so
                there the same move is upwards. Both are a half of the row's own size,
                which is what a percentage translate is measured in. */}
            <div
                className={`absolute inset-0 flex flex-col md:flex-row ${
                    toured === "b"
                        ? "-translate-y-1/2 md:translate-y-0 md:-translate-x-1/2"
                        : ""
                }`}
                // `translate`, NOT `transform`. Tailwind's translate utilities write
                // the standalone `translate` property (`translate: var(--tw-translate-x)
                // …`), so a transition naming `transform` covers nothing they do and
                // the row jumps to its new position in a single frame while the panel
                // beside it takes the full second to resize. Same clock and curve as
                // that resize — the slide, the growth and the camera's flight are one
                // movement.
                style={{
                    transitionProperty: "translate",
                    transitionDuration: `${SOLO_TRANSITION_MS}ms`,
                    transitionTimingFunction: SOLO_EASING,
                }}
            >
                {LOCAL_CELLS.map((cell, i) => {
                    const side: Side = i === 0 ? "a" : "b";
                    return (
                        <ScenePanel
                            key={cell.id}
                            cell={cell}
                            outcome={vote === null ? null : vote === side ? "won" : "lost"}
                            share={side === "a" ? LEFT_VOTE_SHARE : 100 - LEFT_VOTE_SHARE}
                            align={side === "a" ? "left" : "right"}
                            dividerRight={i === 0}
                            role={
                                toured === null
                                    ? "paired"
                                    : toured === side
                                        ? "expanded"
                                        : "pushed"
                            }
                            onFocusedChange={side === "a" ? onTourA : onTourB}
                        />
                    );
                })}
            </div>

            {/* THE MASTHEAD, hanging over the scenes. The moon is clipped to exactly
                this box and anchored to its bottom edge, so the padding below the
                navbar is precisely how far the lower limb protrudes past it — and
                because the box is now an overlay, the limb and its glow fall onto
                the scenes instead of stopping at the top of them.

                `isolate` is load-bearing, not tidiness: the chrome inside blends in
                `difference` mode, and blending only reaches a backdrop inside the
                same stacking context. Isolated, a label crossing the moon inverts
                against it, while the same label over open scene has no backdrop to
                fight and simply stays light. Without it, every one of them would
                blend against whatever the two renderers happened to be drawing. */}
            <div
                className="pointer-events-none absolute inset-x-0 top-0 isolate z-30"
                style={{ paddingBottom: MOON_PROTRUDE }}
            >
                {/* THE NAVBAR SETS THE BAND'S HEIGHT, and the disc is clipped to
                    that box — so MOON_PROTRUDE is measured from the bottom of the
                    navbar and means exactly what it says. It used to be the PROMPT
                    that sized the band, which made the limb chase the prompt down
                    the page: a longer prompt curved deeper, the band grew, and the
                    moon swung far below the chrome it was supposed to sit behind.

                    The clip lives INSIDE Moon rather than here, because only part
                    of the moon wants clipping: the disc does, its glow does not
                    (see Moon.tsx). This box still decides the crop — Moon anchors
                    to it — it just no longer imposes it on the light as well. */}
                <Moon diameter={MOON_DIAMETER} />
                {/* The header itself takes NO pointer events — only the two clusters
                    do. A full-width bar that swallowed them would put a dead strip
                    across the top of both scenes, in the exact place you reach to
                    drag the camera up. */}
                <header className="pointer-events-none relative flex items-start justify-between gap-[clamp(8px,1.4vw,24px)] px-[clamp(12px,1.5vw,26px)] pb-[clamp(10px,1.5vh,18px)] pt-[clamp(9px,1.4vh,16px)]">
                    <div className="pointer-events-auto flex min-w-0 items-center gap-[clamp(8px,1.1vw,17px)]">
                        {/* THE MARK IS THE MOON. A plain filled disc — the same body
                            that rises behind the prompt, at the size of a full stop.
                            Nothing else on this page is a circle, so it needs no
                            drawing to be recognised, and it survives being 20px on a
                            phone in a way a glyph would not. */}
                        <div
                            aria-hidden
                            className="size-[clamp(28px,2.9vw,42px)] flex-none rounded-full bg-foreground"
                        />
                        <div className="flex min-w-0 flex-col justify-center gap-[clamp(2px,0.35vh,5px)]">
                            {/* The one place Anton appears. `mix-blend-difference` so
                                the wordmark inverts if the moon ever reaches it. */}
                            <span className="font-display text-[clamp(15px,1.5vw,23px)] leading-none tracking-[0.08em] whitespace-nowrap mix-blend-difference">
                                SCENEBENCH
                            </span>
                            <span className="font-mono text-[clamp(7px,0.68vw,10px)] font-bold leading-none tracking-[0.18em] whitespace-nowrap text-[#6f6f6f]">
                                BY STARSHOT LABS
                            </span>
                        </div>
                        {/* BESIDE THE MARK, not over with the CTA. These two are
                            about the project rather than about the round in front of
                            you, so they belong with the name that owns them — and it
                            leaves the right-hand side holding nothing but the two
                            things the page actually wants you to do. */}
                        <div className="flex items-center gap-[clamp(10px,1.5vw,22px)] pl-[clamp(4px,0.9vw,14px)]">
                            <SubLink onClick={() => {}}>ABOUT</SubLink>
                            <SubLink onClick={() => {}}>FAQ</SubLink>
                        </div>
                    </div>

                    {/* Two items, one row, and nothing else on this side. Both are
                        about the round in front of you — where this one ranks, and
                        how to start your own — which is why About and FAQ went to
                        live beside the wordmark instead. */}
                    <nav className="pointer-events-auto flex flex-none items-center gap-[clamp(10px,1.9vw,28px)]">
                        {/* #TODO: no destinations yet. These become <Link>s the moment
                            /leaderboard, /about and /faq exist; they are buttons only
                            because there is nowhere to point them. */}
                        {/* Underlined on hover by an inset shadow rather than a border:
                            a border would take its space in the layout whether it was
                            being shown or not, and nudge the row every time the
                            pointer crossed it. */}
                        <button
                            type="button"
                            onClick={() => {}}
                            className="cursor-pointer px-0.5 py-2.5 font-sans text-[clamp(11px,0.95vw,14px)] font-black tracking-[0.05em] whitespace-nowrap text-foreground shadow-[inset_0_-3px_0_rgba(255,255,255,0)] transition-shadow duration-200 hover:shadow-[inset_0_-3px_0_var(--color-foreground)] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40"
                        >
                            LEADERBOARD
                        </button>
                        {/* The page inverted: the site's foreground becomes the
                            button's ground and its background the text. The hover is
                            the one colour on the site that is not black, white or a
                            grey between them — the whole page is monochrome, so the
                            single warm accent lands on the single thing it wants you
                            to press. */}
                        <button
                            type="button"
                            // #TODO: no action yet. This should take a prompt from the
                            // user and queue a build on both models.
                            onClick={() => {}}
                            className="cursor-pointer bg-foreground px-[clamp(13px,1.6vw,24px)] py-[clamp(10px,1.15vw,16px)] font-sans text-[clamp(11px,0.98vw,15px)] font-black tracking-[0.04em] whitespace-nowrap text-background transition-[background-color,color,transform] duration-200 hover:-translate-y-0.5 hover:bg-[#e5342b] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:ring-offset-2 focus-visible:ring-offset-black active:translate-y-0"
                        >
                            GENERATE YOUR OWN
                        </button>
                    </nav>
                </header>

                {/* THE PROMPT IS THE BAND'S ONLY FLOW CONTENT, so it — not the
                    navbar — decides how tall the masthead is, and it centres on the
                    page rather than on whatever gap the chrome leaves. The top
                    padding is what holds it clear of the floating navbar above.

                    PAINTED, NOT BLENDED. This used to blend in `difference` so it
                    would survive crossing the moon's limb, but that mode operates
                    on the antialiased edge pixels too: every letter's outline
                    resolved to muddy mid-greys and the type read as soft. Ink on
                    the disc is worth more than the edge case, so the colour is
                    chosen outright — which is only safe because the arc is sized to
                    keep the text inside the disc. */}
                <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center px-[clamp(12px,1.5vw,26px)] pt-[clamp(4px,0.7vh,9px)] text-center">
                    {/* Clearly SECONDARY to the prompt — roughly half its height — or
                        the masthead has two headlines and neither leads. */}
                    <span className="font-mono text-[clamp(8px,0.72vw,13.5px)] tracking-[0.24em] whitespace-nowrap text-[#0b0d12]/55">
                        WHO BUILT IT BETTER?
                    </span>
                </div>

                {/* ANCHORED EXACTLY AS THE DISC IS — bottom edge on the bottom of
                    this box, horizontally centred — and given the disc's own
                    diameter. That is the whole of the alignment: two square boxes
                    pinned to the same corner have the same centre, so the baseline
                    CurvedPrompt strikes about its viewBox centre is struck about the
                    moon's centre, and the two curves are one curve.

                    GREY, NOT BLACK. Pure ink on a lit disc reads as a hole punched
                    through it — the moon has no value darker than its own
                    terminator, so true black belongs to the page behind, not to
                    anything sitting on the surface. Two-thirds strength keeps the
                    type part of the moon while still clearing AA contrast against
                    it, and the drop shadow (a filter, since text-shadow does not
                    reach SVG glyphs) seats it on the surface rather than floating it
                    above. */}
                <h1 className="pointer-events-none absolute left-1/2 top-full -translate-x-1/2 -translate-y-full text-[#0b0d12]/68 filter-[drop-shadow(0_2px_9px_rgba(9,11,16,0.18))]">
                    <CurvedPrompt text={`"${PROMPT}"`} diameter={MOON_DIAMETER} />
                </h1>
            </div>

            {/* THE VOTE, floating clear of the bottom edge rather than sitting in a
                strip of its own. Centred and only as wide as it needs to be, so the
                scenes run underneath it and the two builds stay the full picture.
                The row hands the pointer back everywhere except the bar itself. */}
            <div className="pointer-events-none absolute inset-x-0 bottom-[clamp(18px,3.4vh,52px)] z-30 flex justify-center px-[clamp(12px,1.5vw,26px)]">
                <div className="pointer-events-auto">
                    <VoteBar
                        cells={LOCAL_CELLS}
                        vote={vote}
                        swing={ELO_SWING}
                        onVote={setVote}
                        onNext={nextPair}
                        // Standing inside one of the scenes stops the clock: see
                        // NextTimer for why the round must not turn over under you.
                        paused={toured !== null}
                    />
                </div>
            </div>
        </main>
    );
}

// The quiet rank of header item, set in the machine voice beside the wordmark:
// small tracked-out mono capitals, grey until you reach for them. About and FAQ
// are things a reader goes looking for rather than things the page offers, so
// they are findable without ever being the first thing found.
function SubLink({
    children,
    onClick,
}: {
    children: ReactNode;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className="cursor-pointer py-1 font-mono text-[clamp(8px,0.75vw,11px)] tracking-[0.18em] whitespace-nowrap text-[#7a7a7a] transition-colors duration-150 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40"
        >
            {children}
        </button>
    );
}
