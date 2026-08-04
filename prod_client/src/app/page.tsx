"use client";

import {
    useCallback,
    useEffect,
    useMemo,
    useState,
    type ReactNode,
} from "react";
import ScenePanel, {
    SOLO_EASING,
    SOLO_TRANSITION_MS,
} from "@/components/arena/ScenePanel";
import { usePreloadRound } from "@/components/arena/usePreloadRound";
import VoteBar from "@/components/arena/VoteBar";
import CurvedPrompt from "@/components/CurvedPrompt";
import LogoMark from "@/components/LogoMark";
import Moon from "@/components/Moon";
import { LOCAL_ROUNDS } from "@/lib/localScenes";

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


// How far the moon's lower limb reaches past the navbar. This is the whole effect
// — too little and the disc reads as a gradient behind the header, too much and it
// stops being a horizon and becomes a shape on the page.
//
// IT IS NOT THE PROMPT'S SIZING KNOB. Reaching further down does buy the prompt
// room, but it buys it by making the moon a bigger object on the page, which is a
// change to the masthead to solve a problem belonging to one line of text. The
// prompt fits itself instead: it is set to the arc it is given and scales with its
// own LENGTH, so a long prompt sets smaller rather than asking the moon to grow.
const MOON_PROTRUDE = "clamp(8px,1.2vh,16px)";

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

// THE ONLY COLOUR ON THE SITE, and it is not invented — these are the mark's own
// hues (public/logo.png), in the order the artwork runs them: the pale ice of the
// lit face, through the periwinkle and violet of the glass, into the warm rose
// where the light comes out the other side. Anything wearing it reads as the same
// material as the mark.
//
// PITCHED HIGH, near white. Sampling the artwork's mid-tones straight off gave a
// gradient that sat DARKER than the cream it replaces, so the button dimmed on
// hover — backwards for the one control the page wants pressed, which should feel
// lit rather than shaded. These are the same five hues held at the top of their
// range: the journey survives, the value does not drop, and black type stays hard
// against all of it.
const MARK_GRADIENT =
    "linear-gradient(104deg, #ffffff 0%, #e6f0ff 24%, #d8dcf5 46%, #e6d5e6 68%, #fbdcd2 88%, #ffffff 100%)";

// ONE DECLARATION, TWO COPIES. The glow is the wordmark drawn a second time
// directly over the first, so the two have to set identically — same face, size,
// tracking and leading. Sharing the string is what guarantees it: two lists that
// merely looked alike would drift the moment either was touched, and a glow a
// pixel off its letters reads as a printing error. The BLEND is not in here,
// because the two copies want opposite ones — see where each is used.
const WORDMARK_TYPE =
    "font-display text-[clamp(15px,1.5vw,23px)] leading-none tracking-[0.08em] whitespace-nowrap";

// The travelling window, in mask terms: opaque at its centre, feathered to nothing
// well before either end. Everything inside it is lit, so the softness of these
// edges IS the softness of the glow — a hard-edged window would switch letters on
// and off as it passed.
const GLOW_WINDOW =
    "linear-gradient(100deg, transparent 26%, rgba(0,0,0,0.55) 40%, #000 50%, rgba(0,0,0,0.55) 60%, transparent 74%)";

type Side = "a" | "b";
/** A round ends one of three ways, and declining to choose is one of them. */
type Vote = Side | "skip";

// What the winner gains and the loser drops. A fixed swing rather than a real Elo
// K-factor calculation: the ratings here are placeholders, so computing an exact
// expected score off them would be arithmetic performed on fiction.
// #TODO: the server owns this — it holds the ratings, and it is the only place
// that can apply the result of a vote to them.
const ELO_SWING = 12;

// How long the outgoing prompt is given to be carried off the visible cap before
// the round underneath it is swapped. It is the FIRST half of one 80° turn — see
// `prompt-roll-out` and `moon-cycle`, which have to agree with this number or the
// swap lands while the old words are still legible.
const ROLL_OUT_MS = 420;

export default function Page() {
    // The whole round is one piece of state. Everything downstream — who glows,
    // who shatters, which numbers count up — is derived from it, so there is no
    // way for the panels to disagree about what happened.
    const [vote, setVote] = useState<Vote | null>(null);

    // WHERE THE ROUND IS, as ONE value.
    //
    // `shown` is what is on screen; `target` is what has been asked for. They are
    // the same number except during the moon's turn, when the page holds the old
    // pair up for as long as the old prompt takes to be carried off the cap.
    //
    // Kept in a single piece of state deliberately. Two counters can be advanced
    // independently, and every switching bug in this page has been the two of them
    // disagreeing for a few frames — a result belonging to a pair that is no longer
    // underneath it. As one atom the invariant is enforced where it is decided: an
    // update sees both halves and can refuse.
    //
    // Counted, not indexed: they only go forward, the modulo is taken at the point
    // of use, and the count doubles as "how many rounds has this visitor seen" for
    // whatever wants to know later.
    const [{ shown, target }, setRound] = useState({ shown: 0, target: 0 });
    const round = LOCAL_ROUNDS[shown % LOCAL_ROUNDS.length];
    const turning = shown !== target;

    // ONE MOMENT, EVERYTHING AT ONCE. The scenes, the prompt and the result all
    // change on this timer and nowhere else.
    //
    // Clearing the vote up front — at the click, where it started life — was the
    // switching bug: the ratings and the crowd's percentages vanished the instant
    // "next" was asked for, while the scenes they described stayed up for another
    // 420 ms and the moon was still carrying the old prompt away. Anything that
    // interrupted the page in that window (a scene finishing its upload, a shatter
    // still playing) left the two halves visibly out of step — one side still
    // showing a result, the other already on the next round. A result belongs to
    // the pair underneath it, so it leaves exactly when they do.
    useEffect(() => {
        if (!turning) return;
        const timer = window.setTimeout(() => {
            setRound((r) => ({ shown: r.target, target: r.target }));
            setVote(null);
        }, ROLL_OUT_MS);
        return () => window.clearTimeout(timer);
    }, [turning]);

    // The next pair's mesh and splat, fetched while the countdown between rounds
    // is running — see usePreloadRound for why that window and not another.
    const nextUp = LOCAL_ROUNDS[(target + 1) % LOCAL_ROUNDS.length];
    usePreloadRound(nextUp, vote !== null);

    // WHICH ROUNDS HAVE ENGINES STANDING, which is at most two.
    //
    // Fetching the next pair's bytes early only saves the transfer; the wait a
    // viewer actually sees is the mesh being parsed and the splat decoded and
    // uploaded, and nothing does that work until an engine exists to ask for it. So
    // the next round is mounted off screen a whole countdown early and builds
    // itself there.
    //
    // The target is kept alive THROUGH the turn as well: for the 420 ms the moon
    // spends carrying the old prompt away, `round` is still the outgoing one — and
    // dropping the incoming row for those few frames would throw away the very
    // engines that were warmed, to rebuild them cold on the other side.
    const incoming = LOCAL_ROUNDS[target % LOCAL_ROUNDS.length];
    const liveRounds = useMemo(() => {
        if (incoming.id !== round.id) return [round, incoming];
        return vote !== null && nextUp.id !== round.id ? [round, nextUp] : [round];
    }, [round, incoming, nextUp, vote]);

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

    // #TODO: the rounds are a checked-in list, so this cycles them. It becomes
    // "ask the server for the next pair" once pairings are served.
    //
    // STABLE, and it has to be: the countdown between the two results schedules
    // itself against this, so a new function on every render would cancel and
    // restart the clock each time the page re-rendered — and it would never reach
    // the end.
    //
    // ONE ADVANCE AT A TIME. A round can only be left from a standing start: if the
    // moon is already mid-turn, asking again is dropped rather than queued. Two
    // advances in flight move the count two rounds on, which with a short list
    // wraps back to the pair you were just looking at — the swap appears to do
    // nothing while the result underneath it clears, which is the strangest thing
    // the page can do. Compared against what is SHOWN rather than a flag, so the
    // gate closes for exactly as long as the change takes.
    const nextPair = useCallback(
        () =>
            setRound((r) =>
                r.shown === r.target ? { ...r, target: r.target + 1 } : r,
            ),
        [],
    );

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
            {/* ONE ROW PER LIVE ROUND, and only one of them on screen.
                A WARM ROUND IS A BUILT ROUND, not a downloaded one. Having the next
                pair's bytes in the HTTP cache (usePreloadRound) saves the transfer
                and nothing else — what the "loading scene" overlay is actually
                waiting on is the mesh being parsed, the splat being decoded and both
                being uploaded to the GPU, and none of that can happen until
                something has mounted an engine and asked for it. So the next round
                is MOUNTED early, off screen, and does that work while the reader is
                looking at the result of this one.

                It then stays mounted THROUGH the swap: the row that was warming
                simply becomes the row that is shown. Keying by round id is what
                makes that possible — React keeps the subtree, so the engines,
                contexts and uploaded scenes survive being revealed, and there is
                nothing left to load at the moment of the change.

                HIDDEN BY OPACITY, and it has to be opacity.

                `visibility: hidden` was the obvious choice and it is the wrong one:
                visibility is INHERITED, and a descendant is free to set itself back
                to `visible`. The splat layer does exactly that — `setActive` writes
                `visibility: visible` straight onto its own canvas (splatLayer.ts) —
                so the moment a warm scene finished loading, its splat overrode the
                hidden row and painted, on top, over the round being voted on. That
                was the switching bug: the next pair appearing early and taking the
                percentages with it, while the countdown was still running. It only
                bit when the warm scenes finished inside the countdown, which is why
                it came and went.

                Opacity cannot be opted out of by a child — it applies to the subtree
                as one group — and, unlike `display: none`, it keeps the row LAID OUT,
                so both renderers still see a real size and can build against it. A
                zero-sized row is a graphics device handed a zero-sized backbuffer and
                a scene that never finishes framing. The z-index is belt and braces:
                whatever any descendant does, the shown row is in front. */}
            {liveRounds.map((live) => {
                const shown = live.id === round.id;
                return (
                    <div
                        key={live.id}
                        aria-hidden={!shown}
                        className={`absolute inset-0 flex flex-col md:flex-row ${
                            shown ? "z-[1]" : "z-0 opacity-0 pointer-events-none"
                        } ${
                            shown && toured === "b"
                                ? "-translate-y-1/2 md:translate-y-0 md:-translate-x-1/2"
                                : ""
                        }`}
                        // `translate`, NOT `transform`. Tailwind's translate utilities
                        // write the standalone `translate` property (`translate:
                        // var(--tw-translate-x) …`), so a transition naming `transform`
                        // covers nothing they do and the row jumps to its new position
                        // in a single frame while the panel beside it takes the full
                        // second to resize. Same clock and curve as that resize — the
                        // slide, the growth and the camera's flight are one movement.
                        style={{
                            transitionProperty: "translate",
                            transitionDuration: `${SOLO_TRANSITION_MS}ms`,
                            transitionTimingFunction: SOLO_EASING,
                        }}
                    >
                        {live.cells.map((cell, i) => {
                            const side: Side = i === 0 ? "a" : "b";
                            return (
                                <ScenePanel
                                    key={cell.id}
                                    cell={cell}
                                    // A round nobody can see has no result and cannot be
                                    // toured: the treatments and the reporting belong to
                                    // the row on screen.
                                    outcome={
                                        !shown || vote === null
                                            ? null
                                            : vote === "skip"
                                                ? "skipped"
                                                : vote === side
                                                    ? "won"
                                                    : "lost"
                                    }
                                    share={
                                        side === "a"
                                            ? live.leftShare
                                            : 100 - live.leftShare
                                    }
                                    align={side === "a" ? "left" : "right"}
                                    dividerRight={i === 0}
                                    role={
                                        !shown || toured === null
                                            ? "paired"
                                            : toured === side
                                                ? "expanded"
                                                : "pushed"
                                    }
                                    onFocusedChange={
                                        shown
                                            ? side === "a"
                                                ? onTourA
                                                : onTourB
                                            : undefined
                                    }
                                />
                            );
                        })}
                    </div>
                );
            })}

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
                <Moon diameter={MOON_DIAMETER} cycle={target} />
                {/* The header itself takes NO pointer events — only the two clusters
                    do. A full-width bar that swallowed them would put a dead strip
                    across the top of both scenes, in the exact place you reach to
                    drag the camera up. */}
                <header className="pointer-events-none relative flex items-start justify-between gap-[clamp(8px,1.4vw,24px)] px-[clamp(12px,1.5vw,26px)] pb-[clamp(10px,1.5vh,18px)] pt-[clamp(9px,1.4vh,16px)]">
                    <div className="pointer-events-auto flex min-w-0 items-center gap-[clamp(8px,1.1vw,17px)]">
                        {/* THE MARK AND THE NAME ARE ONE TARGET — the same statement,
                            so reaching for either lights the glass. `group/mark` is
                            what LogoMark's whole effect hangs off; About and FAQ are
                            deliberately outside it, being a different subject. */}
                        {/* TIGHT, and tighter than it looks. The artwork carries its
                            own transparent margin — the disc fills about seven tenths
                            of the square it is drawn in — so the box has to run right
                            up against the wordmark for the two to read as one lockup,
                            and the box has to be bigger than the type it stands
                            beside to match its weight. */}
                        {/* A CONTROL, not a decoration that happens to react. The
                            glass lighting up under the pointer already promises
                            something will happen if you press — so it is a real
                            button, it takes the pointer cursor that promise implies,
                            and it can be reached and fired from the keyboard like
                            everything else in this bar. */}
                        {/* #TODO: no destination yet. This becomes the way back to a
                            fresh round (or home, once there is more than one page) —
                            the same #TODO as its neighbours. */}
                        <button
                            type="button"
                            onClick={() => {}}
                            aria-label="SceneBench by Starshot Labs"
                            className="group/mark flex min-w-0 cursor-pointer items-center gap-[clamp(1px,0.25vw,5px)] text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40"
                        >
                            <LogoMark className="size-[clamp(40px,4.2vw,60px)] flex-none" />
                            <div className="flex min-w-0 flex-col justify-center gap-[clamp(2px,0.35vh,5px)]">
                                {/* The one place Anton appears. `mix-blend-difference`
                                    so the wordmark inverts if the moon reaches it. */}
                                <span className="relative inline-block">
                                    <span className={`${WORDMARK_TYPE} mix-blend-difference`}>
                                        SCENEBENCH
                                    </span>
                                    {/* THE GLOW. The same word again, in white, with a
                                        bloom around the letters — and a soft window
                                        travelling along it, so only the stretch of the
                                        name inside the window is lit. Nothing is laid
                                        over the type: the letters ARE the light, and
                                        the word stays exactly as readable during the
                                        pass as before it.

                                        `plus-lighter` ADDS the light to what is
                                        already there rather than replacing it, which
                                        is the difference between a letter glowing and
                                        a letter being repainted. And no
                                        `mix-blend-difference` here — the copy beneath
                                        has already inverted itself against whatever it
                                        sits on; inverting the glow as well would
                                        subtract exactly the light being added. */}
                                    <span
                                        aria-hidden
                                        className={`pointer-events-none absolute inset-0 text-white opacity-0 mix-blend-plus-lighter group-hover/mark:animate-[wordmark-glow_820ms_cubic-bezier(0.4,0,0.55,1)_60ms] ${WORDMARK_TYPE}`}
                                        style={{
                                            textShadow:
                                                "0 0 5px rgba(255,255,255,0.9), 0 0 13px rgba(255,255,255,0.5), 0 0 26px rgba(255,255,255,0.28)",
                                            maskImage: GLOW_WINDOW,
                                            WebkitMaskImage: GLOW_WINDOW,
                                            // TALLER THAN THE TEXT, on purpose. The bloom
                                            // spills well above and below the letters, and
                                            // a mask only the height of the box cut it off
                                            // flat at both — which drew a lit RECTANGLE
                                            // around the word instead of a halo on it.
                                            // Centred vertically so the overspill is
                                            // covered on both sides.
                                            maskSize: "260% 420%",
                                            WebkitMaskSize: "260% 420%",
                                            maskRepeat: "no-repeat",
                                            WebkitMaskRepeat: "no-repeat",
                                            maskPosition: "100% 50%",
                                            WebkitMaskPosition: "100% 50%",
                                        }}
                                    >
                                        SCENEBENCH
                                    </span>
                                    {/* THE SPARK, where the gleam runs out: above the
                                        last letter, outside the word rather than over
                                        it, timed to land as the light leaves. */}
                                    <span
                                        aria-hidden
                                        className="pointer-events-none absolute -top-[0.42em] -right-[0.3em] size-[clamp(8px,0.78vw,12px)] text-foreground opacity-0 group-hover/mark:animate-[star-twinkle_620ms_cubic-bezier(0.3,1.4,0.4,1)_600ms]"
                                    >
                                        <svg
                                            viewBox="0 0 24 24"
                                            fill="currentColor"
                                            className="size-full"
                                        >
                                            {/* A four-point spark with concave sides —
                                                the shape a point of light makes, not the
                                                five-point badge on a sheriff. */}
                                            <path d="M12 0c0 6.6 5.4 12 12 12-6.6 0-12 5.4-12 12 0-6.6-5.4-12-12-12 6.6 0 12-5.4 12-12z" />
                                        </svg>
                                    </span>
                                </span>
                                <span className="font-mono text-[clamp(7px,0.68vw,10px)] font-bold leading-none tracking-[0.18em] whitespace-nowrap text-[#6f6f6f]">
                                    BY STARSHOT LABS
                                </span>
                            </div>
                        </button>
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
                            button's ground and its background the text.

                            THE HOVER IS THE MARK'S OWN GLASS. The whole page is
                            black, white and the greys between, and the single place
                            any colour lives is the logo — so the one colour the site
                            has is what lights up on the one thing it wants you to
                            press, and pressing it feels like the same object as the
                            name in the corner. Sampled from the artwork rather than
                            invented (see MARK_GRADIENT). It is a light gradient, so
                            the label stays black on it; white text on those pales
                            would fail its contrast where the ice blue is. */}
                        <button
                            type="button"
                            // #TODO: no action yet. This should take a prompt from the
                            // user and queue a build on both models.
                            onClick={() => {}}
                            className="group/cta relative cursor-pointer bg-foreground px-[clamp(13px,1.6vw,24px)] py-[clamp(10px,1.15vw,16px)] font-sans text-[clamp(11px,0.98vw,15px)] font-black tracking-[0.04em] whitespace-nowrap text-background transition-transform duration-200 hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:ring-offset-2 focus-visible:ring-offset-black active:translate-y-0"
                        >
                            {/* A LAYER, not a background swap: `background-image` does
                                not interpolate, so a gradient set on hover would snap
                                in while the lift eased. Fading a copy over the cream
                                is the only way the two arrive together. */}
                            <span
                                aria-hidden
                                className="absolute inset-0 opacity-0 transition-opacity duration-200 group-hover/cta:opacity-100"
                                style={{ backgroundImage: MARK_GRADIENT }}
                            />
                            <span className="relative">GENERATE YOUR OWN</span>
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

                    SOLID BLACK. It was set two-thirds strength for a while, on the
                    argument that ink darker than the moon's own terminator reads as
                    a hole punched through the disc rather than as a mark on it — but
                    that treats the prompt as texture, and it is the headline. At
                    this size, on a body this bright, the greyed version simply read
                    as washed out. The drop shadow stays (a filter, since text-shadow
                    does not reach SVG glyphs); it is what seats the type on the
                    surface now that the value alone no longer does. */}
                {/* THE PROMPT TURNS WITH THE MOON. Because the SVG is the disc's own
                    square, rotating it about its centre carries the text ALONG THE
                    LIMB — the line does not slide sideways off the moon, it travels
                    round it, which is the one motion a word written on a sphere can
                    make. That is the whole reason the prompt is anchored this way;
                    the alignment was the first payoff, this is the second.

                    KEYED BY THE ROUND, so a new prompt is a new element and its
                    entrance plays from the start. It comes in already turning and
                    slows into place — see `prompt-settle`. */}
                <h1
                    // Keyed by what is SHOWN, so the settle plays once per arrival.
                    // While the wheel is carrying the old prompt away the key does
                    // not change — the same element simply picks up the other
                    // animation and rides out on it.
                    key={round.id}
                    className={`pointer-events-none absolute left-1/2 top-full origin-center -translate-x-1/2 -translate-y-full text-black filter-[drop-shadow(0_2px_9px_rgba(9,11,16,0.18))] ${
                        turning
                            ? "animate-[prompt-roll-out_420ms_cubic-bezier(0.5,0,0.85,0.4)_both]"
                            : "animate-[prompt-settle_1000ms_cubic-bezier(0.12,0.78,0.18,1)_both]"
                    }`}
                >
                    <CurvedPrompt text={`"${round.prompt}"`} diameter={MOON_DIAMETER} />
                </h1>
            </div>

            {/* THE VOTE, floating clear of the bottom edge rather than sitting in a
                strip of its own. Centred and only as wide as it needs to be, so the
                scenes run underneath it and the two builds stay the full picture.
                The row hands the pointer back everywhere except the bar itself. */}
            <div className="pointer-events-none absolute inset-x-0 bottom-[clamp(18px,3.4vh,52px)] z-30 flex justify-center px-[clamp(12px,1.5vw,26px)]">
                <div className="pointer-events-auto">
                    <VoteBar
                        cells={round.cells}
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
