"use client";

import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import ScenePanel, {
    SOLO_EASING,
    SOLO_TRANSITION_MS,
} from "@/components/arena/ScenePanel";
import { PairGate } from "@/components/arena/pairGate";
import { buildStep } from "@/components/arena/buildSequence";
import Composer from "@/components/arena/Composer";
import CurvedPrompt from "@/components/CurvedPrompt";
import VoteBar, { REVEAL_SETTLE_MS } from "@/components/arena/VoteBar";
import { RAKE_PX } from "@/components/ui/Button";
import Masthead, { MOON_DIAMETER, MoonArc } from "@/components/site/Masthead";
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

type Side = "a" | "b";
/** A round ends one of three ways, and declining to choose is one of them. */
type Vote = Side | "skip";

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

    // THE NEXT PAIR, SOLVED DURING THE COUNTDOWN — not just downloaded.
    //
    // This used to be `usePreloadRound`, which spent the wait on the BYTES and left
    // the parse, the splat decode and the sightline solve on the swap. Measured,
    // those were the whole cost: a 216 ms frame about 800 ms after the click,
    // landing underneath the vote bar's own transition, which is where the lag was
    // being seen. The engines can now solve a scene without disturbing the one they
    // are showing (OrbitEngine.warmTour), so the countdown absorbs all of it and
    // "next" is left holding nothing but a swap.
    //
    // The old hook is gone rather than kept alongside: it fetched exactly the two
    // files the warm fetches, at the same moment, so running both had the pair
    // downloaded TWICE over one connection. That contention alone stretched the
    // warm to ~8 s — long enough that it was still unfinished when "next" was
    // pressed, which looked like the warm not working at all.
    //
    // KEYED TO `shown`, NOT `target`. Pressing "next" moves `target` at once while
    // the pair underneath stays up for the prompt's roll-out, so reading `target`
    // flipped this to the round AFTER the incoming one at the exact moment the
    // incoming one was wanted — clobbering the warm it had just built and
    // destroying its staged splat. `shown` does not move until the swap, so the
    // warm stays pointed at the pair that is actually next.
    //
    // AND NOT UNTIL THE REVEAL HAS STOPPED MOVING. Starting this on the vote put
    // the whole warm — two GLB parses, two splat decodes, two sightline solves —
    // underneath the cards expanding and the ratings counting up. Measured, a burst
    // of 42/146/62 ms frames in the first 600 ms after the click, and identical for
    // a SKIP, which shatters nothing: the reveal was the only thing on screen, so
    // the reveal was what stuttered. That is the lag as the NEXT button arrives.
    //
    // The countdown is ~8 s and the warm needs about one, so it can afford to let
    // the reveal have its second. `REVEAL_SETTLE_MS` comes from the reveal's own
    // two stages, so retiming those moves this with them.
    // ONE SIDE AT A TIME. The two panels are independent engines with independent
    // work, and starting them together had both splat decodes and both scene
    // solves resolving in the SAME frames — which is how two pieces that are each
    // comfortably short still add up to a dropped one. Offsetting the second means
    // each engine gets the main thread to itself.
    //
    // Long enough to clear the first side's solve, which measured ~500 ms of
    // spread-out work; the countdown has seconds to spare either way.
    // SHORTENED WITH THE ROUND. The reveal now settles at ~1.28 s and the countdown
    // runs 3 s, so a 700 ms gap put the second panel's solve at 1.98 s with under a
    // second left to finish in. At 420 it starts at ~1.7 s, which still gives each
    // engine the main thread to itself and still lands ahead of the swap.
    const WARM_STAGGER_MS = 420;
    const nextUp = LOCAL_ROUNDS[(shown + 1) % LOCAL_ROUNDS.length];
    // 0 = not yet, 1 = the left panel may warm, 2 = both may.
    const [warmStep, setWarmStep] = useState(0);
    // Cleared during render, armed from timers — same reason as the pair gate: a
    // reset that waits for an effect hands the panels a stale "still warming" for
    // one commit, and one commit is the whole event.
    const [warmFor, setWarmFor] = useState(vote);
    if (warmFor !== vote) {
        setWarmFor(vote);
        if (warmStep !== 0) setWarmStep(0);
    }
    useEffect(() => {
        if (vote === null) return;
        const first = window.setTimeout(() => setWarmStep(1), REVEAL_SETTLE_MS);
        const second = window.setTimeout(
            () => setWarmStep(2),
            REVEAL_SETTLE_MS + WARM_STAGGER_MS,
        );
        return () => {
            window.clearTimeout(first);
            window.clearTimeout(second);
        };
    }, [vote]);
    const warmA = warmStep >= 1 ? nextUp.cells[0].source : null;
    const warmB = warmStep >= 2 ? nextUp.cells[1].source : null;

    // BOTH SIDES CHANGE TOGETHER OR NEITHER DOES. The two panels finish loading
    // whenever their own assets allow — measured ~550 ms apart, one cell carrying a
    // 12 MB splat and the other none — so left alone the row shows one build from
    // the new round beside one from the old. A gate per round holds the first
    // swap until the second is ready and then runs both in one frame.
    //
    // Keyed to the round being SHOWN: a new gate is what makes the next change a
    // fresh pair rather than one already spent. The old one is cancelled, since a
    // swap still held for a round nobody is on any more must not fire.
    //
    // Swapped DURING RENDER rather than in an effect — the same pattern as the
    // viewer's `wasInside`. An effect would hand the panels the spent gate for one
    // commit first, and one commit is the entire event.
    const [gate, setGate] = useState(() => new PairGate(2));
    const [gateFor, setGateFor] = useState(shown);
    if (gateFor !== shown) {
        setGateFor(shown);
        gate.cancel();
        setGate(new PairGate(2));
    }
    const commitA = useCallback((c: () => void) => gate.arrive("a", c), [gate]);
    const commitB = useCallback((c: () => void) => gate.arrive("b", c), [gate]);

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
    // Someone is writing a prompt of their own. The vote steps aside while they
    // are: the two are alternative things to do with this page, and leaving the
    // buttons up under an open composer asks the viewer to hold both questions at
    // once — which of these two builds is better, and what would you build.
    const [composing, setComposing] = useState(false);

    // THE PAGE ASSEMBLES ITSELF ON ARRIVAL. `built` is false for exactly one frame
    // after mount, so the ray leaves the moon and draws the controls out on load
    // rather than the whole apparatus simply being there. Flipped from a rAF
    // callback rather than the effect body — the browser has to paint the
    // unbuilt state once, or the transition has nothing to travel from.
    // WHERE THE RAY STOPS, and that is now the only thing here that has to be
    // measured. The beam opens along the bar's top edge and stays, so its travel and
    // its rake — which the descent needed — are gone with the descent.
    //
    // MEASURED, AND MEASURED WHEN IT MOVES. The stack is anchored to the bottom of
    // the window, so when the composer grows, everything above it is lifted: the
    // bar's top rises while its width and height do not change by a pixel. A
    // ResizeObserver watching the BAR therefore never fired, and the ray went on
    // ending where the controls used to start. The container is observed instead, so
    // the growth itself is the trigger.
    const barRef = useRef<HTMLDivElement>(null);
    const rowRef = useRef<HTMLDivElement>(null);
    const stackRef = useRef<HTMLDivElement>(null);
    const [seamBreak, setSeamBreak] = useState(0);
    const [seamUnder, setSeamUnder] = useState(0);
    useEffect(() => {
        const bar = barRef.current;
        const row = rowRef.current;
        if (!bar || !row) return;
        const measure = () => {
            const b = bar.getBoundingClientRect();
            const r = row.getBoundingClientRect();
            if (!b.height) return;
            const next = r.bottom - b.top;
            setSeamBreak((prev) => (Math.abs(prev - next) < 0.5 ? prev : next));
            // Where the line ends once the round is answered: the bar's BOTTOM, so
            // it runs the full height of the controls and is covered by them.
            const under = r.bottom - b.bottom;
            setSeamUnder((prev) => (Math.abs(prev - under) < 0.5 ? prev : under));
        };
        measure();
        const observer = new ResizeObserver(measure);
        observer.observe(bar);
        observer.observe(row);
        const stack = stackRef.current;
        if (stack) observer.observe(stack);
        return () => observer.disconnect();
    }, []);

    const [mounted, setMounted] = useState(false);
    useEffect(() => {
        const frame = requestAnimationFrame(() => setMounted(true));
        return () => cancelAnimationFrame(frame);
    }, []);
    // TWO AUDIENCES FOR THE BUILD, and they are not the same.
    //
    // `built` drives the RAY and the VOTE — the things the beam draws and takes
    // away. Reaching for the composer collapses those, which is the point: the page
    // is clearing its throat so you can write.
    //
    // `composerBuilt` drives the composer, and it deliberately ignores `composing`.
    // It was wired to `built` as well, so touching the field ran the collapse over
    // the field itself — the thing you had just clicked into dissolved under the
    // cursor. It assembles once on load and then stays, because it is the one
    // control that has to survive its own focus.
    // NOT UNTIL IT HAS BEEN MEASURED. The beam's travel and its narrowing both come
    // from the bar's own box, and building before that measurement lands means
    // descending zero pixels and then jumping when the real number arrives.
    // The round the page opened on. Captured once, so "is this the first round" stays
    // true to the session rather than to whatever is currently on screen.
    // COUNTED, NOT COMPARED. This was `round.id !== firstId`, which asks "is this a
    // different round from the one we opened on" — and with a short, cycling list the
    // opening round COMES BACK. Every second turn matched the first id and the seam
    // was told not to replay, so the ray drew itself on some turns and simply
    // appeared on others. That is the nondeterminism, and it is also why it looked
    // like a property of the NEXT BUTTON: the timer and the button both advance the
    // same counter, but only every other turn showed the animation, and a viewer
    // clicking early sees a different half of the alternation than one who waits.
    //
    // `shown` only ever goes up, so "have we turned at least once" is exactly what it
    // says, and every turn after the first replays regardless of which pair it lands
    // on.
    const ready = mounted && seamBreak > 0;
    const built = ready && !composing;
    // Read ONCE and shared by both beam spans and both of their tracks, so the
    // motion and the cut cannot be handed drifting values.
    const { transitionDuration: d, transitionDelay: t } = buildStep("beam", built);
    const composerBuilt = ready && vote === null;

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
        // A REAL PAGE, in bands, top to bottom: navbar, prompt, the two builds,
        // the vote, the footer. It replaces a full-bleed arena that had the scenes
        // filling the window edge to edge with every other element floating over
        // them — which reads as a demo rather than as a site, because nothing has
        // a place of its own and everything is a layer on top of the picture.
        //
        // `h-dvh` and a flex column: the page is exactly one screen and never
        // scrolls, so the bands divide a KNOWN height between them. The four
        // fixed bands take what they need and the scene row takes the rest, which
        // is what lets the builds be as large as the furniture allows without
        // anyone having to pick a number.
        <main
            className="relative flex h-dvh flex-col overflow-hidden bg-ground"
            // WHERE THE SEAM ENDS: level with the composer's underline, which sits at
            // the very bottom of the control stack. Published here rather than
            // measured, because it is the same offset the stack itself is placed at —
            // one number, used twice, so the vertical and the horizontal cannot drift
            // out of line with each other.
            style={{
                // WHERE THE SEAM STOPS AND STARTS AGAIN. The controls sit in a break
                // in the line rather than on top of it: the upper run ends at the top
                // of the SKIP button, the vote and the composer occupy the gap, and
                // the lower run picks up from the composer's own rule and carries on
                // to the floor.
                //
                // `--seam-stop` is the composer's underline, which is exactly where
                // the control stack is anchored — one number used twice, so the
                // vertical and the horizontal cannot drift apart.
                //
                // `--seam-break` IS MEASURED, and it has to be exact. It used to
                // overshoot on purpose — running the seam a few pixels into the
                // opaque bar so the beam could never visibly fall short. That bought
                // contact and cost the thing contact was for: the ray's tip ended up
                // BELOW the bar's top edge, so the horizontal line formed partway
                // along the beam instead of at its point, and on the way back the
                // stub sat there with nothing to retract into. A tip that lands on
                // the edge is not a tuning problem, it is a measurement.
                ["--seam-stop" as string]: "var(--spacing-md)",
                ["--seam-break" as string]: `${seamBreak}px`,
                ["--seam-under" as string]: `${seamUnder}px`,
            }}
        >
            {/* The arena writes the ROUND on the moon. Everything else about the
                masthead — the disc, the navbar, the label — is the shared
                component; only what is written on the arc differs by page. */}
            {/* The masthead and the shadow it casts leave together — half of that
                shadow IS the moon's, so keeping it while the disc goes would be a
                shadow with nothing above it. */}
            <div
                className={`transition-opacity duration-500 ${
                    toured !== null ? "pointer-events-none opacity-0" : "opacity-100"
                }`}
            >
            <Masthead label="Who built it better?">
                <MoonArc>
                    <h1
                        key={round.id}
                        className={`${
                            turning
                                ? "animate-[prompt-roll-out_420ms_cubic-bezier(0.5,0,0.85,0.4)_both]"
                                : "animate-[prompt-settle_1000ms_cubic-bezier(0.12,0.78,0.18,1)_both]"
                        }`}
                    >
                        <CurvedPrompt
                            text={'\u201c' + round.prompt + '\u201d'}
                            diameter={MOON_DIAMETER}
                        />
                    </h1>
                </MoonArc>
            </Masthead>
            </div>

            {/* --- the builds ---------------------------------------------------
                `min-h-0` is load-bearing. A flex child's default `min-height:auto`
                refuses to shrink below its content, and the content here is two
                canvases that size themselves to their parent — so without it the
                row wins every argument about height, pushes the vote bar and the
                footer off the bottom of the screen, and the page silently stops
                being one screen tall. */}
            {/* THE SCENES ARE THE PAGE. They fill the frame top to bottom and the
                masthead floats over them, so no band of chrome holds a strip of the
                window back from the builds — which are the whole point of it. */}
            <div ref={rowRef} className="relative min-h-0 flex-1">

            {/* ONE ROW, FOR THE LIFE OF THE PAGE.
                
                This used to render a row PER LIVE ROUND, keyed by round id and each
                panel keyed by cell id — so answering a round mounted two more
                ScenePanels, two more OrbitViewers and four more canvases. Measured,
                that is two fresh WebGL contexts and two engine constructions in the
                frame the vote lands: ~165ms, and identical for a skip, which
                shatters nothing. None of the things visibly moving were responsible.
                
                The engine has always been able to swap scenes in place —
                useOrbitEngine builds it once on mount and calls `loadTour` when the
                source changes. Nothing needed a new engine; it was the KEYS that
                were throwing them away. Keyed by side instead, the two panels
                outlive every round and a new pair is a load rather than a rebuild.
                Canvas count stays at four for the session. */}
                <div
                    className={`absolute inset-0 flex flex-col md:flex-row ${
                        toured === "b"
                            ? "-translate-y-1/2 md:translate-y-0 md:-translate-x-1/2"
                            : ""
                    }`}
                    // `translate`, NOT `transform`. Tailwind's translate utilities
                    // write the standalone `translate` property, so a transition
                    // naming `transform` covers nothing they do and the row jumps in
                    // one frame while the panel beside it takes the full second.
                    style={{
                        transitionProperty: "translate",
                        transitionDuration: `${SOLO_TRANSITION_MS}ms`,
                        transitionTimingFunction: SOLO_EASING,
                    }}
                >
                    {round.cells.map((cell, i) => {
                            const side: Side = i === 0 ? "a" : "b";
                            return (
                                <ScenePanel
                                    key={side}
                                    cell={cell}
                                    // A round nobody can see has no result and cannot be
                                    // toured: the treatments and the reporting belong to
                                    // the row on screen.
                                    outcome={
                                        vote === null
                                            ? null
                                            : vote === "skip"
                                                ? "skipped"
                                                : vote === side
                                                    ? "won"
                                                    : "lost"
                                    }
                                    share={
                                        side === "a"
                                            ? round.leftShare
                                            : 100 - round.leftShare
                                    }
                                    align={side === "a" ? "left" : "right"}
                                    dividerRight={i === 0}
                                    built={built}
                                    // Every round but the first arrives with the seam
                                    // tucked behind the controls by the round before
                                    // it, so it has to come back out. The first one
                                    // does not: on a cold page the beam descends from
                                    // the moon, which is a different entrance and the
                                    // only one that should play on load.
                                    untuck={shown > 0}
                                    // Keyed to the TURN, so a cycle back to a pair
                                    // already seen is still a fresh element with a
                                    // fresh animation to play.
                                    roundKey={String(shown)}
                                    warm={side === "a" ? warmA : warmB}
                                    commitVia={side === "a" ? commitA : commitB}
                                    role={
                                        toured === null
                                            ? "paired"
                                            : toured === side
                                                ? "expanded"
                                                : "pushed"
                                    }
                                    onFocusedChange={
                                        true
                                            ? side === "a"
                                                ? onTourA
                                                : onTourB
                                            : undefined
                                    }
                                />
                            );
                        })}
                </div>
            </div>

            {/* --- the vote, over the scenes -----------------------------------
                Overlaid rather than banded. A row of its own was tidier and it
                cost the builds a strip of the window they could have been filling;
                floating gives that back, and the scenes are what the page is for.

                The engine knows: SAFE_BOTTOM reserves the strip this occupies so
                no build is ever framed with its near corner under a button. Those
                two numbers have to agree — see engine.ts. */}
            <div
                ref={stackRef}
                // OUT OF THE WAY WHILE A BUILD IS BEING WALKED THROUGH. Stepping
                // inside used to take the browser's screen, which hid the site by
                // hiding everything; it does not any more (see OrbitViewer), so the
                // site has to clear its own furniture. Faded rather than unmounted:
                // the bar keeps its measured height, so the seam and the ray that
                // are pinned to it do not jump the moment someone steps in, and it
                // is all still there when they come back out.
                className={`pointer-events-none absolute inset-x-0 bottom-md z-30 flex flex-col items-center gap-0 px-lg transition-opacity duration-500 ${
                    toured !== null ? "opacity-0" : "opacity-100"
                }`}
            >
                {/* THE VOTE KEEPS ITS SPACE WHILE IT LEAVES. Fading and sinking
                    rather than unmounting, so the composer beneath it does not jump
                    up the screen the instant the field is touched — the point of
                    the movement is that the page makes room quietly, and a layout
                    shift is the loudest thing it could do. Sinking as it goes, so
                    it reads as stepping back rather than switching off. */}
                {/* THE BEAM TURNS, THEN DESCENDS.
                    
                    `sweepRule` is the white line itself: it opens out of the seam's
                    landing point to both corners, then travels down the bar's height.
                    The bar is wiped in behind it — `inset(0 0 100% 0)` retreating to
                    zero — so the buttons are uncovered in the rule's wake and the line
                    is visibly the thing drawing them, rather than a cue that fires a
                    fade somewhere else.
                    
                    The two share one step (`wipe`) deliberately: a rule on one clock
                    and a reveal on another drift apart within a few frames, and the
                    illusion depends entirely on the edge of the wipe sitting exactly
                    under the line at every instant. */}
                <div className="pointer-events-auto relative">
                    {/* One step read once, so the two tracks and both spans
                        cannot be given drifting values. */}
                    <span
                        aria-hidden
                        // REMOUNTED ON EVERY DIRECTION CHANGE, and this is the whole
                        // reason the sequence held together on paper and not on
                        // screen.
                        //
                        // A CSS animation starts its clock when it is APPLIED, and
                        // changing `animation-delay` or `animation-direction` does not
                        // restart it. Every other stage here is a transition, and a
                        // transition starts when its value changes. So the two kinds
                        // of stage were being timed from two different origins:
                        //
                        //  - ON LOAD the animation began ticking at first paint while
                        //    the ray waited for `mounted`, which lands only after two
                        //    WebGL contexts have initialised. The beam spent its delay
                        //    during that gap and opened while the ray was still on its
                        //    way down — the horizontal appearing before the line that
                        //    is supposed to create it.
                        //  - ON RE-ENTRY its currentTime was already past the end, so
                        //    flipping direction only re-resolved a finished animation.
                        //    It snapped to its last frame, which is why the opening and
                        //    the descent arrived as one movement instead of two.
                        //
                        // A key change tears the element down and builds it again in
                        // the SAME commit that changes the transitions, so the beam and
                        // the ray start from one origin by construction. Nothing here
                        // holds state or focus, so remounting it costs nothing.
                        key={built ? "build" : "collapse"}
                        className="absolute inset-x-0 top-0 z-20 h-[3px] origin-center bg-mark"
                        style={{
                            // The resting state, for the frame before the first build:
                            // closed to nothing at the bar's top edge. The animation
                            // fills over this the moment there is one.
                            scale: "0 1",
                            // `scale` AND `translate`, AND NOTHING ELSE. This drew its
                            // narrowing with `left`/`right` first, which is correct
                            // geometry animated the wrong way: insets are layout, so
                            // the browser relaid the line every frame and snapped its
                            // ends to whole pixels while the descent ran smoothly on
                            // the compositor. The ends stepped, the body glided, and
                            // the two came apart — the jagged edge.
                            //
                            // Scaled about the centre, the ends track the rake exactly
                            // at every frame, not just at the two ends of the travel:
                            // at depth t the width is W(1 + (sx-1)t), so each end sits
                            // W(1-sx)t/2 = RAKE*t inside — which is the slant.
                            // ONE PART, BOTH DIRECTIONS. The delay used to be read
                            // from `sweep` when building and `wipe` when collapsing —
                            // two stages for one element, which is how the opening
                            // came to start before the ray had landed. `beam` is the
                            // whole stroke, and its start is derived from the ray's
                            // end, so it cannot begin early.
                            // Not run at all until the page is ready to build. Without
                            // this the first render would mount a COLLAPSE animation and
                            // play it, drawing a line in order to take it away again.
// TWO ANIMATIONS, ONE CLOCK: the motion, and the cut.
                            //
                            // Visibility is a separate track run on `steps(1, end)`
                            // so it holds at full strength for the whole beat and
                            // changes exactly once, at the end. Put on the motion
                            // track it would have to be a fade, and a beam that
                            // dissolves after landing reads as an effect finishing
                            // rather than as a line becoming the edge it drew.
                            animationName: ready
                                ? "vote-rule, vote-rule-hide"
                                : "none",
                            animationDuration: `${d}, ${d}`,
                            animationDelay: `${t}, ${t}`,
                            animationDirection: built
                                ? "normal, normal"
                                : "reverse, reverse",
                            animationFillMode: "both, both",
                            // Per-BEAT easing lives in the keyframes; one curve here
                            // would be re-applied to every segment and make the line
                            // hesitate at each of its own stops.
                            animationTimingFunction: "linear, steps(1, end)",
                        }}
                    />
                    {/* THE LINE CROSSES SKIP AGAIN, and the mask that stopped it is
                        gone.

                        There used to be a black span here, exactly SKIP's width and
                        the rule's height, sitting at z-20 on the bar's top edge. Its
                        argument was that SKIP has no edge of its own — it is the black
                        cut between two white slabs — so a white rule laid across its
                        top would be drawing a border it does not have.

                        SKIP HAS ONE NOW (see VoteBar, where it carries hairlines on
                        its horizontals), so the mask was hiding the very thing it is
                        supposed to have. Worse, it was doing it silently: the border
                        computed correctly at 1px solid ink and simply never appeared,
                        because a z-20 span was painted over it. Measured down through
                        SKIP's centre, the BOTTOM rule showed at full strength and the
                        top read pure black — an asymmetry with no cause anywhere in
                        the button, which is what gave the mask away.

                        With it removed the beam runs the full width of the bar and
                        lands on SKIP's own top rule, so the three segments share one
                        continuous edge instead of two white slabs and a gap. */}

                    <div
                        ref={barRef}
                        // THE WHOLE BUTTON IS DRAWN BY THE LINE, text included. The
                        // clip travels with the beam — measured, the uncovered
                        // fraction equals the beam's depth to the decimal — so
                        // everything inside is revealed exactly as the edge passes it.
                        //
                        // The labels were briefly held back and faded in after the
                        // sweep landed. That is a defensible effect and it was the
                        // wrong one: the text is the part of a button you actually
                        // read, so holding it made the control look like it appeared
                        // late rather than like it was being drawn. Nothing overrides
                        // the clip now.
                        style={{
                            // OPACITY IS NOT PART OF THE REVEAL — the clip is.
                            //
                            // These shared one duration, so while the line uncovered
                            // the bar the bar was ALSO fading up from zero, reaching
                            // full strength only as the sweep landed. Captured
                            // mid-descent the buttons were dark grey, then lighter,
                            // then finally the palette white — revealed on time and
                            // drawn late, which is exactly the gap between the line
                            // and the control it is supposed to be drawing.
                            //
                            // Opacity now switches at the start of the beat and the
                            // clip does all the drawing, so whatever the line has
                            // passed is already at full strength behind it.
                            // THE CLIP IS THE WHOLE MOVEMENT, in both directions.
                            //
                            // Opacity used to ride alongside it. Sharing the clip's
                            // duration it made the bar fade up WHILE being uncovered,
                            // so the buttons ghosted in behind the line; switched
                            // instantly instead, it fixed the build and broke the
                            // collapse — the bar blinked out before the clip could
                            // close, so entering the composer removed the controls
                            // rather than un-drawing them.
                            //
                            // It is gone. `inset(0 0 100% 0)` already hides the bar
                            // completely, so opacity was never carrying anything the
                            // clip was not, and one property animating in one
                            // direction is reversible by construction: the bar grows
                            // down from the line, and closes back up into it.
                            transitionProperty: "clip-path",
                            transitionDuration: buildStep("wipe", built)
                                .transitionDuration,
                            transitionDelay: buildStep("wipe", built).transitionDelay,
                            transitionTimingFunction: buildStep("wipe", built)
                                .transitionTimingFunction,
                            clipPath: built ? "inset(0 0 0 0)" : "inset(0 0 100% 0)",
                            pointerEvents: composing ? "none" : "auto",
                        }}
                    >
                    <VoteBar
                        cells={round.cells}
                        vote={vote}
                        onVote={setVote}
                        onNext={nextPair}
                        // Standing inside one of the scenes stops the clock: see
                        // NextTimer for why the round must not turn over under you.
                        paused={toured !== null}
                        />
                    </div>
                </div>

                {/* THE INVITATION, under the vote. It is the last thing in the
                    column deliberately: a viewer's job on this page is to answer
                    the question above it, and asking them to write a prompt of
                    their own before they have done that is the page interrupting
                    itself. Below the vote it reads as what comes next. */}
                {/* GONE WHILE THE ROUND FINISHES. Between the vote and the next pair
                    the page is telling you something — who made what, how the crowd
                    split, how long until the next one — and an invitation to go and
                    do something else is the one thing that should not be on screen
                    for it. It comes back when the next pair does.

                    Faded and sunk rather than unmounted, so the field keeps its
                    typed text and its scroll position through the round change and
                    does not replay its typewriter every few seconds. */}
                <div
                    className="transition-[opacity,translate]"
                    style={{
                        ...buildStep("composer", composerBuilt),
                        opacity: composerBuilt ? 1 : 0,
                        translate: vote !== null ? "0 var(--spacing-lg)" : "0 0",
                        pointerEvents: composerBuilt ? "auto" : "none",
                    }}
                >
                    <Composer onOpenChange={setComposing} built={composerBuilt} />
                </div>
            </div>

        </main>
    );
}
