"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { LOCAL_ROUNDS } from "@/lib/localScenes";
import ArenaHeader from "./ArenaHeader";
import Ballot from "./Ballot";
import ManifestBand from "./ManifestBand";
import { PairGate } from "./pairGate";
import ScenePanel, { type PanelRole } from "./ScenePanel";
import StarField from "./StarField";
import { useArenaMetrics } from "./useArenaMetrics";
import { outcomeFor, type Side, type Vote } from "./vote";
import "./arena.css";

/** How long the result stands before the next round is dealt. */
const HOLD_MS = 3400;

/** How long the outgoing prompt has to leave before the next round lands. */
const TURN_MS = 420;

/** The next pair is prepared one scene at a time, so neither load starves. */
const WARM_FIRST_MS = 600;
const WARM_SECOND_MS = 1020;

export default function Arena() {
	const rootRef = useRef<HTMLElement>(null);
	const composerRef = useRef<HTMLInputElement>(null);
	const turnTimer = useRef(0);
	const turningNow = useRef(false);
	const metrics = useArenaMetrics(rootRef);

	const [shown, setShown] = useState(0);
	const [turning, setTurning] = useState(false);
	const [vote, setVote] = useState<Vote | null>(null);
	const [walked, setWalked] = useState<Side | null>(null);

	const round = LOCAL_ROUNDS[shown % LOCAL_ROUNDS.length];
	const upNext = LOCAL_ROUNDS[(shown + 1) % LOCAL_ROUNDS.length];

	const turn = useCallback(() => {
		if (turningNow.current) return;
		turningNow.current = true;
		setTurning(true);
		turnTimer.current = window.setTimeout(() => {
			turningNow.current = false;
			setShown((n) => n + 1);
			setVote(null);
			setTurning(false);
		}, TURN_MS);
	}, []);

	useEffect(() => () => window.clearTimeout(turnTimer.current), []);

	useEffect(() => {
		if (vote === null) return;
		const held = window.setTimeout(turn, HOLD_MS);
		return () => window.clearTimeout(held);
	}, [vote, turn]);

	const castVote = useCallback((choice: Vote) => {
		setVote((current) => current ?? choice);
	}, []);

	useEffect(() => {
		metrics.remeasure();
	}, [metrics, shown]);

	// Both scenes commit together, so the pair never appears one at a time.
	const [gate, setGate] = useState(() => new PairGate(2));
	const [gateFor, setGateFor] = useState(shown);
	if (gateFor !== shown) {
		setGateFor(shown);
		gate.cancel();
		setGate(new PairGate(2));
	}
	const commitA = useCallback((commit: () => void) => gate.arrive("a", commit), [gate]);
	const commitB = useCallback((commit: () => void) => gate.arrive("b", commit), [gate]);

	const [warmStep, setWarmStep] = useState(0);
	const [warmFor, setWarmFor] = useState(vote);
	if (warmFor !== vote) {
		setWarmFor(vote);
		if (warmStep !== 0) setWarmStep(0);
	}
	useEffect(() => {
		if (vote === null) return;
		const first = window.setTimeout(() => setWarmStep(1), WARM_FIRST_MS);
		const second = window.setTimeout(() => setWarmStep(2), WARM_SECOND_MS);
		return () => {
			window.clearTimeout(first);
			window.clearTimeout(second);
		};
	}, [vote]);
	// A pair already on screen must not be re-prepared; that strands the staging.
	const repeats = upNext === round;
	const warmA = warmStep >= 1 && !repeats ? upNext.cells[0].source : null;
	const warmB = warmStep >= 2 && !repeats ? upNext.cells[1].source : null;

	const walkA = useCallback(
		(inside: boolean) =>
			setWalked((current) => (inside ? "a" : current === "a" ? null : current)),
		[],
	);
	const walkB = useCallback(
		(inside: boolean) =>
			setWalked((current) => (inside ? "b" : current === "b" ? null : current)),
		[],
	);
	const roleFor = (side: Side): PanelRole =>
		walked === null ? "paired" : walked === side ? "expanded" : "pushed";

	const focusComposer = useCallback(() => composerRef.current?.focus(), []);
	const shareOf = (side: Side) =>
		vote === null ? null : side === "a" ? round.leftShare : 100 - round.leftShare;

	return (
		<main ref={rootRef} className="arena" data-solo={walked !== null}>
			<StarField />

			<ArenaHeader metrics={metrics} onGenerate={focusComposer} />

			<div className="arena-title-slot arena-chrome">
				<h1
					key={shown}
					ref={metrics.register("prompt")}
					className="arena-title"
					data-leaving={turning}
				>
					{round.prompt}
				</h1>
			</div>

			<div className="arena-region">
				<ScenePanel
					cell={round.cells[0]}
					outcome={outcomeFor(vote, "a")}
					role={roleFor("a")}
					align="left"
					share={shareOf("a")}
					warm={warmA}
					commitVia={commitA}
					onWalkChange={walkA}
				/>
				<ScenePanel
					cell={round.cells[1]}
					outcome={outcomeFor(vote, "b")}
					role={roleFor("b")}
					align="right"
					share={shareOf("b")}
					warm={warmB}
					commitVia={commitB}
					onWalkChange={walkB}
				/>
			</div>

			<ManifestBand inputRef={composerRef} />

			<Ballot cells={round.cells} vote={vote} onVote={castVote} onNext={turn} />
		</main>
	);
}
