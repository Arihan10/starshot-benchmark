"use client";

import {
    useCallback,
    useEffect,
    useLayoutEffect,
    useRef,
    useState,
} from "react";
import ArenaSeam from "@/components/arena/ArenaSeam";
import ScenePanel, {
    SOLO_EASING,
    SOLO_TRANSITION_MS,
} from "@/components/arena/ScenePanel";
import { PairGate } from "@/components/arena/pairGate";
import { buildStep } from "@/components/arena/buildSequence";
import Composer from "@/components/arena/Composer";
import VoteBar, { REVEAL_SETTLE_MS } from "@/components/arena/VoteBar";
import CurvedPrompt, { useCaptionWidth } from "@/components/CurvedPrompt";
import Masthead from "@/components/site/Masthead";
import { LOCAL_ROUNDS } from "@/lib/localScenes";

type Side = "a" | "b";
type Vote = Side | "skip";

const ROLL_OUT_MS = 420;

export default function Page() {
    const [vote, setVote] = useState<Vote | null>(null);

    const [{ shown, target }, setRound] = useState({ shown: 0, target: 0 });
    const round = LOCAL_ROUNDS[shown % LOCAL_ROUNDS.length];
    const turning = shown !== target;
    const promptText = "\u201c" + round.prompt + "\u201d";
    // Masthead lays its margin either side of this to strike the moon.
    const captionWidth = useCaptionWidth(promptText);

    useEffect(() => {
        if (!turning) return;
        const timer = window.setTimeout(() => {
            setRound((r) => ({ shown: r.target, target: r.target }));
            setVote(null);
        }, ROLL_OUT_MS);
        return () => window.clearTimeout(timer);
    }, [turning]);

    const WARM_STAGGER_MS = 420;
    const nextUp = LOCAL_ROUNDS[(shown + 1) % LOCAL_ROUNDS.length];
    const [warmStep, setWarmStep] = useState(0);
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
    // A one-pair queue is already on screen; re-preparing it strands the staging.
    const repeats = nextUp === round;
    const warmA = warmStep >= 1 && !repeats ? nextUp.cells[0].source : null;
    const warmB = warmStep >= 2 && !repeats ? nextUp.cells[1].source : null;

    const [gate, setGate] = useState(() => new PairGate(2));
    const [gateFor, setGateFor] = useState(shown);
    if (gateFor !== shown) {
        setGateFor(shown);
        gate.cancel();
        setGate(new PairGate(2));
    }
    const commitA = useCallback((c: () => void) => gate.arrive("a", c), [gate]);
    const commitB = useCallback((c: () => void) => gate.arrive("b", c), [gate]);

    const [composing, setComposing] = useState(false);

    const shellRef = useRef<HTMLElement>(null);
    const barRef = useRef<HTMLDivElement>(null);
    const stackRef = useRef<HTMLDivElement>(null);
    const headRef = useRef<HTMLDivElement>(null);
    const [seam, setSeam] = useState({ half: 0, notch: 0, drop: 0, head: 0 });
    const resting = useRef(true);
    useLayoutEffect(() => {
        resting.current = vote === null;
    }, [vote]);
    useEffect(() => {
        const bar = barRef.current;
        const shell = shellRef.current;
        if (!bar || !shell) return;
        const measure = () => {
            const b = bar.getBoundingClientRect();
            if (!b.height || !resting.current) return;
            const next = {
                half: b.height / 2,
                notch: b.width / 2,
                drop: shell.getBoundingClientRect().bottom - (b.top + b.height / 2),
                head: headRef.current?.getBoundingClientRect().height ?? 0,
            };
            setSeam((prev) =>
                (Object.keys(next) as (keyof typeof next)[]).every(
                    (k) => Math.abs(prev[k] - next[k]) < 0.5,
                )
                    ? prev
                    : next,
            );
        };
        measure();
        const observer = new ResizeObserver(measure);
        observer.observe(bar);
        observer.observe(shell);
        const stack = stackRef.current;
        if (stack) observer.observe(stack);
        const head = headRef.current;
        if (head) observer.observe(head);
        return () => observer.disconnect();
    }, []);

    const [mounted, setMounted] = useState(false);
    useEffect(() => {
        const frame = requestAnimationFrame(() => setMounted(true));
        return () => cancelAnimationFrame(frame);
    }, []);
    const ready = mounted && seam.half > 0;
    const built = ready && !composing;
    const { transitionDuration: d, transitionDelay: t } = buildStep("beam", built);
    const composerBuilt = ready && vote === null;

    const [toured, setToured] = useState<Side | null>(null);
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

    // #TODO: cycles a checked-in list; become a server-fed queue later.
    const nextPair = useCallback(
        () =>
            setRound((r) =>
                r.shown === r.target ? { ...r, target: r.target + 1 } : r,
            ),
        [],
    );

    return (
        <main
            ref={shellRef}
            className="relative flex h-dvh flex-col overflow-hidden bg-ground"
            style={{
                ["--seam-break" as string]: `${seam.half}px`,
                ["--seam-notch" as string]: `${seam.notch}px`,
                ["--seam-drop" as string]: `${seam.drop}px`,
            }}
        >
            <div
                ref={headRef}
                {...(toured !== null ? { "data-moon-idle": true } : {})}
                className={`flex-none transition-opacity duration-500 ${
                    toured !== null ? "pointer-events-none opacity-0" : "opacity-100"
                }`}
            >
            <Masthead
                label="Who built it better?"
                placement="flow"
                captionWidth={captionWidth}
            >
                {({ promptRadius, rollOrigin, rollDeg }) => (
                    // THE PIVOT HAS TO BE ON THIS ELEMENT, not on anything around
                    // it: `transform-origin` applies to the box being transformed
                    // and is not inherited. Origin is the centre of the circle the
                    // caption is set on, so it rolls along its own arc.
                    <h1
                        key={round.id}
                        style={{
                            transformOrigin: rollOrigin,
                            ["--prompt-roll-deg" as string]: rollDeg,
                        }}
                        className={`m-0 w-full min-w-0 ${
                            turning
                                ? "animate-[prompt-roll-out_420ms_cubic-bezier(0.5,0,0.85,0.4)_both]"
                                : "animate-[prompt-settle_1000ms_cubic-bezier(0.12,0.78,0.18,1)_both]"
                        }`}
                    >
                        <CurvedPrompt text={promptText} radius={promptRadius} />
                    </h1>
                )}
            </Masthead>
            </div>

            <div
                className="relative min-h-0 flex-1"
                style={{
                    marginTop: toured !== null ? `${-seam.head}px` : 0,
                    transitionProperty: "margin",
                    transitionDuration: `${SOLO_TRANSITION_MS}ms`,
                    transitionTimingFunction: SOLO_EASING,
                }}
            >

                <div
                    className={`absolute inset-0 flex flex-col md:flex-row ${
                        toured === "b"
                            ? "-translate-y-1/2 md:translate-y-0 md:-translate-x-1/2"
                            : ""
                    }`}
                    style={{
                        transitionProperty: "translate",
                        transitionDuration: `${SOLO_TRANSITION_MS}ms`,
                        transitionTimingFunction: SOLO_EASING,
                    }}
                >
                    <ScenePanel
                        cell={round.cells[0]}
                        outcome={
                            vote === null
                                ? null
                                : vote === "skip"
                                    ? "skipped"
                                    : vote === "a"
                                        ? "won"
                                        : "lost"
                        }
                        share={round.leftShare}
                        align="left"
                        warm={warmA}
                        commitVia={commitA}
                        role={
                            toured === null
                                ? "paired"
                                : toured === "a"
                                    ? "expanded"
                                    : "pushed"
                        }
                        onFocusedChange={onTourA}
                    />
                    <ScenePanel
                        cell={round.cells[1]}
                        outcome={
                            vote === null
                                ? null
                                : vote === "skip"
                                    ? "skipped"
                                    : vote === "b"
                                        ? "won"
                                        : "lost"
                        }
                        share={100 - round.leftShare}
                        align="right"
                        warm={warmB}
                        commitVia={commitB}
                        role={
                            toured === null
                                ? "paired"
                                : toured === "b"
                                    ? "expanded"
                                    : "pushed"
                        }
                        onFocusedChange={onTourB}
                    />
                    {/* After both panels so it paints above their canvases. Absolute
                        top/bottom — not a stretched empty flex column, which had no
                        height and was why the join stayed missing. */}
                    <ArenaSeam
                        built={built}
                        untuck={shown > 0}
                        roundKey={String(shown)}
                        shown={
                            toured === null &&
                            (vote === null || vote === "skip")
                        }
                    />
                </div>

            </div>

            <div
                ref={stackRef}
                className={`pointer-events-none absolute inset-x-0 bottom-md z-30 flex flex-col items-center gap-0 px-lg transition-opacity duration-500 ${
                    toured !== null ? "opacity-0" : "opacity-100"
                }`}
            >
                <div className="pointer-events-auto relative">
                    <span
                        aria-hidden
                        key={built ? "build" : "collapse"}
                        className="absolute inset-x-0 top-0 z-20 h-(--seam-width) origin-center bg-mark"
                        style={{
                            scale: "0 1",
                            animationName: ready
                                ? "vote-rule, vote-rule-hide"
                                : "none",
                            animationDuration: `${d}, ${d}`,
                            animationDelay: `${t}, ${t}`,
                            animationDirection: built
                                ? "normal, normal"
                                : "reverse, reverse",
                            animationFillMode: "both, both",
                            animationTimingFunction: "linear, steps(1, end)",
                        }}
                    />

                    <div
                        ref={barRef}
                        style={{
                            transitionProperty: "clip-path",
                            transitionDuration: buildStep("wipe", built)
                                .transitionDuration,
                            transitionDelay: buildStep("wipe", built).transitionDelay,
                            transitionTimingFunction: buildStep("wipe", built)
                                .transitionTimingFunction,
                            // THE WIPE ONLY EVER MOVES THE BOTTOM EDGE, so the other
                            // three are held OUTSIDE the box. On `inset(0)` the rect's
                            // top and sides land exactly on the slabs' outer tips —
                            // those vertices sit on the box corners — and a rectangular
                            // clip shears the antialiasing off them, which is what
                            // turned each point into a short vertical stub. Held out at
                            // -8px the reveal is identical and the tips are interior.
                            clipPath: built
                                ? "inset(-8px -8px -8px -8px)"
                                : "inset(-8px -8px 100% -8px)",
                            pointerEvents: composing ? "none" : "auto",
                        }}
                    >
                    <VoteBar
                        cells={round.cells}
                        vote={vote}
                        onVote={setVote}
                        onNext={nextPair}
                        paused={toured !== null}
                        />
                    </div>
                </div>

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
