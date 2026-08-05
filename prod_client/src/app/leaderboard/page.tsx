import type { Metadata } from "next";
import CurvedPrompt from "@/components/CurvedPrompt";
import VoxelSky from "@/components/site/VoxelSky";
import ExitBar from "@/components/leaderboard/ExitBar";
import Fade from "@/components/site/Fade";
import MastheadFade from "@/components/leaderboard/MastheadFade";
import Podium from "@/components/leaderboard/Podium";
import SnapScroller from "@/components/leaderboard/SnapScroller";
import StandingsTable from "@/components/leaderboard/StandingsTable";
import Masthead, { MOON_DIAMETER, MoonArc } from "@/components/site/Masthead";
import { STANDINGS } from "@/lib/leaderboard";

export const metadata: Metadata = {
	title: "Leaderboard — SceneBench",
	description:
		"Blind pairwise Elo across every prompt. Which language models build best in 3D.",
};

/**
 * The standings.
 *
 * TWO SECTIONS, ONE SCREEN EACH, and the page snaps between them. The first is the
 * podium: the top three built out of isometric blocks, under the moon that names
 * the champion. The second is the board itself.
 *
 * NO PAGE SHELL HERE ANY MORE. PageShell frames a document that scrolls normally
 * inside a measure — right for About, wrong for a screen whose whole behaviour is
 * that it does NOT scroll normally. The two pieces this page still wants from it,
 * a full-bleed exit bar and a max width, are three lines; the snapping is not
 * something that could be bolted onto the shell without every other page paying
 * for it.
 *
 * A SERVER COMPONENT. The podium and the scroller are client islands because they
 * need a canvas and a scroll listener; the moon, the champion's name and the exit
 * bar are still rendered once, on the server.
 *
 * #TODO: `STANDINGS` is placeholder data imported directly, and the only things
 * reading it are the champion's name on the arc and the three names on the podium.
 */
export default function LeaderboardPage() {
	const champion = STANDINGS[0];
	const top = STANDINGS.slice(0, 3);

	return (
		<SnapScroller
			// A BAR, NOT A BUTTON IN A ROW. It is the last thing on the page and the
			// only thing left to do, so it takes the full width and reads as the page's
			// own bottom edge. WHAT it offers depends on which section the reader is
			// on, which is why it is its own client island — see ExitBar.
			footer={<ExitBar />}
		>
			{/* --- the podium ------------------------------------------------------
			    A FLEX COLUMN, so the stage is "whatever the masthead leaves". The moon
			    band sizes itself and the canvas takes the rest — no viewport maths
			    anywhere, and the podium reframes itself when the band's own type
			    scales. */}
			<section className="relative flex h-dvh snap-start flex-col">
				<MastheadFade on="hero">
					<Masthead label="Current champion" placement="flow">
						<MoonArc>
							{/* The champion is a NAME, not a sentence — so it takes the
							    interface face in bold rather than the arena's italic serif,
							    which is reserved for the one line a person wrote. */}
							<CurvedPrompt
								text={champion.name}
								diameter={MOON_DIAMETER}
								voice="name"
							/>
						</MoonArc>
					</Masthead>
				</MastheadFade>

				{/* `min-h-0` is load-bearing: a flex child defaults to `min-height:auto`,
				    which refuses to shrink below its content and would push the canvas
				    past the fold on a short window. */}
				{/* THE CITY ARRIVES AND LEAVES WITH THE PAGE. Its own assembly is a
				    separate, much longer performance — this only covers the moment of
				    the navigation itself, so the canvas is not simply cut in or out
				    from under the moon, which stays put across the swap. */}
				<Fade enter={640} delay={120} className="relative min-h-0 flex-1">
					<Podium rows={top} />
				</Fade>
			</section>

			{/* --- the board -------------------------------------------------------
			    EXACTLY A SCREEN, and the table scrolls inside it. Letting the section
			    grow to the table's own height would put a stretch of page between two
			    snap points with no snap point of its own, which under a mandatory
			    snap is the one state a reader must never be able to rest in — they
			    would be dragged back to whichever end was nearer. */}
			<section
				className="relative flex h-dvh snap-start flex-col"
				aria-label="Standings"
			>
				<VoxelSky />
				{/* THE SAME MASTHEAD, SAYING WHERE THE READER NOW IS. The moon carries
				    the section's name rather than the champion's, because the champion
				    is one row of the table underneath it now and the moon's job on any
				    page is to say what the page is about. */}
				<MastheadFade on="board">
					<Masthead label="Current standings" placement="flow">
						<MoonArc>
							<CurvedPrompt
								text="Leaderboard"
								diameter={MOON_DIAMETER}
								voice="name"
							/>
						</MoonArc>
					</Masthead>
				</MastheadFade>

				{/* THE EXIT BAR IS PINNED OVER THE BOTTOM OF THE PAGE, so the column
				    stops short of it. Padding here rather than a spacer inside the
				    table, because it has to come out of the height the table measures
				    itself against — a spacer inside would let the board size itself to
				    a space partly covered by the bar, and the last row and the SHOW
				    MORE button would both end up underneath it. */}
				{/* `relative z-30` is not decoration. It clears two things at once: the
				    sky behind it, which is absolutely positioned and would otherwise
				    paint above these static rows, and the moon, which is fixed at z-10
				    in the root layout. */}
				<Fade
					enter={640}
					delay={160}
					className="relative z-30 mx-auto flex min-h-0 w-full max-w-[1180px] flex-1 flex-col px-lg pb-[76px]"
				>
					<StandingsTable rows={STANDINGS} />
				</Fade>
			</section>
		</SnapScroller>
	);
}
