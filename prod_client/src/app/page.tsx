"use client";

import {
    useCallback,
    useEffect,
    useLayoutEffect,
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
import VoteBar, { REVEAL_SETTLE_MS } from "@/components/arena/VoteBar";
import Masthead, { ROLL_ORIGIN, Title } from "@/components/site/Masthead";
import { useLeaving } from "@/components/site/PageTransition";
import { LOCAL_ROUNDS } from "@/lib/localScenes";

type Side = "a" | "b";
type Vote = Side | "skip";

const ROLL_OUT_MS = 420;

// SHORTER THAN THE NAVIGATION HOLD, so the arena has finished leaving by the time
// the route swaps. Anything longer is cut off mid-fade, which reads as a dropped
// frame rather than an exit. The masthead runs its own on the way out.
const LEAVE_MS = 200;

export default function Page() {
    const [vote, setVote] = useState<Vote | null>(null);
    const leaving = useLeaving();

    const [{ shown, target }, setRound] = useState({ shown: 0, target: 0 });
    const round = LOCAL_ROUNDS[shown % LOCAL_ROUNDS.length];
    const turning = shown !== target;

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
            >
                {/* THE PIVOT HAS TO BE ON THIS ELEMENT, not on anything around it:
                    `transform-origin` applies to the box being transformed and is
                    not inherited, so it belongs wherever the animation does. */}
                <h1
                    key={round.id}
                    style={{ transformOrigin: ROLL_ORIGIN }}
                    className={`${
                        turning
                            ? "animate-[prompt-roll-out_420ms_cubic-bezier(0.5,0,0.85,0.4)_both]"
                            : "animate-[prompt-settle_1000ms_cubic-bezier(0.12,0.78,0.18,1)_both]"
                    }`}
                >
                    <Title>{'\u201c' + round.prompt + '\u201d'}</Title>
                </h1>
            </Masthead>
            </div>

            <div
                className="relative min-h-0 flex-1"
                style={{
                    marginTop: toured !== null ? `${-seam.head}px` : 0,
                    opacity: leaving ? 0 : 1,
                    transitionProperty: "margin, opacity",
                    transitionDuration: `${SOLO_TRANSITION_MS}ms, ${LEAVE_MS}ms`,
                    transitionTimingFunction: `${SOLO_EASING}, ease-out`,
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
                    {round.cells.map((cell, i) => {
                            const side: Side = i === 0 ? "a" : "b";
                            return (
                                <ScenePanel
                                    key={side}
                                    cell={cell}
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
                                    untuck={shown > 0}
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

            <div
                ref={stackRef}
                className={`pointer-events-none absolute inset-x-0 bottom-md z-30 flex flex-col items-center gap-0 px-lg transition-opacity duration-500 ${
                    toured !== null ? "opacity-0" : "opacity-100"
                }`}
                style={leaving ? { opacity: 0, transitionDuration: `${LEAVE_MS}ms` } : undefined}
            >
                <div className="pointer-events-auto relative">
                    <span
                        aria-hidden
                        key={built ? "build" : "collapse"}
                        className="absolute inset-x-0 top-0 z-20 h-[3px] origin-center bg-mark"
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
                            clipPath: built ? "inset(0 0 0 0)" : "inset(0 0 100% 0)",
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
