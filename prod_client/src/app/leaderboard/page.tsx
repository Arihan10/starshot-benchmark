import type { Metadata } from "next";
import VoxelSky from "@/components/site/VoxelSky";
import ExitBar from "@/components/leaderboard/ExitBar";
import LeaderboardStage from "@/components/leaderboard/LeaderboardStage";
import Masthead, { Title } from "@/components/site/Masthead";
import { STANDINGS } from "@/lib/leaderboard";

export const metadata: Metadata = {
	title: "Leaderboard — SceneBench",
	description:
		"Blind pairwise Elo across every prompt. Which language models build best in 3D.",
};

// #TODO: STANDINGS is placeholder data imported directly.

// HOW MUCH ROOM THE EXIT BAR TAKES at the foot of the page. Both columns are
// told the same number rather than each guessing: the bar is laid over them
// both, and a table that cleared it while the city did not would put a row of
// rooftops behind the one control that leaves the page.
const FOOT = 76;

/**
 * THE BOARD AND THE CITY STAND SIDE BY SIDE, and the split is what makes both of
 * them work.
 *
 * Stacked, each was the wrong shape for what it held. The standings ran the full
 * measure of the page for five columns of short numbers — a row of figures with
 * a hand's width of nothing between them — while the city was squeezed into a
 * strip a few hundred pixels deep, which is not enough height to tell three
 * towers apart by looking. Neither was short of room; they were short of the
 * RIGHT room, and they wanted it in different directions.
 *
 * Turned on its side the page gives each what it was missing. The board loses
 * width it was not using and gains rows, which is the axis a list actually grows
 * along. The city loses width it does not need — a tighter plan reads more like
 * a city than a sprawl does — and gains the full height of the page, so the
 * three pillars can stand far enough apart in height that the ranking is legible
 * as a SHAPE before a single label is read.
 *
 * SLIGHTLY WIDER ON THE LEFT. The two are close to even, but the board carries
 * type and the city carries mass — and type is the one that stops working when
 * it runs out of room. The city simply gets denser.
 */
export default function LeaderboardPage() {
	const top = STANDINGS.slice(0, 3);

	return (
		<div className="relative flex h-dvh flex-col overflow-hidden bg-ground">
			<VoxelSky voxels={false} />

			<Masthead label="Current standings" placement="flow">
				<Title voice="name">Leaderboard</Title>
			</Masthead>

				<LeaderboardStage rows={STANDINGS} top={top} foot={FOOT} />

			{/* Arrives last, and from under the foot of the page, so the one control
			    that leaves the route is the final thing to settle. */}
			<div
				className="lift-in absolute inset-x-0 bottom-0 z-40"
				style={{
					["--lift-delay" as string]: "420ms",
					["--lift-from" as string]: "100%",
				}}
			>
				<ExitBar />
			</div>
		</div>
	);
}
