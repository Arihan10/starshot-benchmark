import type { Metadata } from "next";
import CurvedPrompt from "@/components/CurvedPrompt";
import VoxelSky from "@/components/site/VoxelSky";
import ExitBar from "@/components/leaderboard/ExitBar";
import Fade from "@/components/site/Fade";
import Podium from "@/components/leaderboard/Podium";
import StandingsTable from "@/components/leaderboard/StandingsTable";
import Masthead, { MOON_DIAMETER, MoonArc } from "@/components/site/Masthead";
import { STANDINGS } from "@/lib/leaderboard";

export const metadata: Metadata = {
	title: "Leaderboard — SceneBench",
	description:
		"Blind pairwise Elo across every prompt. Which language models build best in 3D.",
};

// #TODO: STANDINGS is placeholder data imported directly.

const PODIUM_BAND = "clamp(190px, 32vh, 340px)";

export default function LeaderboardPage() {
	const top = STANDINGS.slice(0, 3);

	return (
		<div className="relative flex h-dvh flex-col overflow-hidden bg-ground">
			<VoxelSky city />

			<Masthead label="Current standings" placement="flow">
				<MoonArc>
					<CurvedPrompt
						text="Leaderboard"
						diameter={MOON_DIAMETER}
						voice="name"
					/>
				</MoonArc>
			</Masthead>

			<div className="relative min-h-0 flex-1">
				<Fade enter={640} delay={120} className="absolute inset-0 z-10">
					<Podium rows={top} band={PODIUM_BAND} />
				</Fade>

				<div className="pointer-events-none absolute inset-0 z-30 flex flex-col">
					<div className="flex-none" style={{ height: PODIUM_BAND }} />
					<Fade
						enter={640}
						delay={200}
						className="pointer-events-auto mx-auto flex min-h-0 w-full max-w-[1180px] flex-1 flex-col px-lg pb-[76px]"
					>
						<StandingsTable rows={STANDINGS} />
					</Fade>
				</div>
			</div>

			<div className="absolute inset-x-0 bottom-0 z-40">
				<ExitBar />
			</div>
		</div>
	);
}
