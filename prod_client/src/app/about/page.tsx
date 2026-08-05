import type { Metadata } from "next";
import Link from "next/link";
import AboutStage from "@/components/about/AboutStage";
import Button from "@/components/ui/Button";

export const metadata: Metadata = {
	title: "About — SceneBench",
	description:
		"How well can a language model build in 3D? SceneBench puts them head to head and lets the public decide.",
};

/** The eyebrow over each screen: the machine voice, tracked wide. */
const EYEBROW =
	"font-mono text-[10.5px] font-bold tracking-[0.26em] uppercase text-ink-40";

/**
 * The serif, italic, and the only place on the site it appears besides the arena's
 * prompt. Both are the same kind of line — a human sentence rather than a readout —
 * which is the whole reason the face is in the stack at all.
 */
const HEADLINE =
	"font-serif italic font-normal leading-[1.04] tracking-[-0.01em] text-ink text-pretty";

/** Body copy, set below the ink so the headline above stays the loudest thing on
 *  the screen. */
const BODY =
	"font-sans text-[clamp(13px,1.1vw,16.5px)] font-medium leading-[1.55] text-ink-64 text-pretty";

/** The column each screen's copy is held in, and the one behaviour it needs: on a
 *  short window the copy is taller than the screen it is snapped to, and the
 *  alternative to an inner scroll is a section that GROWS — which under a mandatory
 *  snap leaves a stretch of page with no snap point of its own. */
const COLUMN =
	"flex max-h-full min-h-0 w-[min(620px,54vw)] flex-col gap-[clamp(10px,2.6vh,30px)] overflow-y-auto max-md:w-auto";

/**
 * The three beats of a round, in the order they happen. Numbered because the order
 * IS the content — a reader who takes these as three unrelated features has missed
 * that step two is blind and step three is a stranger.
 */
const STEPS = [
	{
		n: "01",
		lead: "Prompt.",
		body: "Both models are given the same prompt, selected across a variety of different scene kinds, from buildings to game levels.",
	},
	{
		n: "02",
		lead: "Build.",
		body: "Two models generate the scene independently. Neither sees the other's work, and both utilize Starshot's in-house 3D generation model.",
	},
	{
		n: "03",
		lead: "Vote.",
		body: "You pick the better build. Ratings update after every vote and feed the public leaderboard.",
	},
];

/**
 * About SceneBench.
 *
 * THREE SCREENS THAT SNAP, and the moon travels between them — see AboutStage,
 * which owns the scroller and everything fixed behind it. The copy ALTERNATES
 * SIDES so the moon always has the other half of the screen to stand in: right on
 * the first, left on the second, and on the third the copy takes the middle and the
 * moon rises underneath it as a horizon.
 *
 * A SERVER COMPONENT. Only the stage needs the browser — for a scroll position and
 * a transform — so every word on this page is rendered once and shipped as HTML.
 */
export default function AboutPage() {
	return (
		<AboutStage>
			{/* ================= what it is ===================================== */}
			<section className="relative flex h-dvh snap-start snap-always items-center justify-end overflow-hidden px-[clamp(26px,3vw,58px)] pt-[68px] pb-[40px]">
				<div className={COLUMN}>
					{/* STAGGERED IN, a beat apart, top to bottom. They arrive in the order
					    they are read in, so the screen assembles itself down the column
					    rather than appearing all at once. */}
					<p
						className={`${EYEBROW} animate-[about-rise_0.6s_cubic-bezier(0.16,1,0.3,1)_0.15s_both]`}
					>
						About SceneBench
					</p>

					<h1
						className={`${HEADLINE} animate-[about-rise_0.6s_cubic-bezier(0.16,1,0.3,1)_0.24s_both] text-[clamp(27px,3.6vw,56px)]`}
					>
						How well can a language model build in 3D?
					</h1>

					<div
						className={`${BODY} max-w-[540px] animate-[about-rise_0.6s_cubic-bezier(0.16,1,0.3,1)_0.32s_both]`}
					>
						<p>
							LLMs inherently think in terms of text — a one-dimensional ordered
							sequence of words. How will they fare when asked to construct
							omnidirectional 3D scenes, a fundamentally different modality?
						</p>
						<p className="mt-md">
							SceneBench places different LLMs in a harness we engineered
							in-house here at Starshot Labs, and asks them to build the same
							scene from the same prompt, while the actual 3D generation is taken
							care of by our foundational model. Due to the subjective and
							creative nature of 3D generation, we want{" "}
							<em className="italic">your</em> votes to decide which LLMs reign
							at the top, and which ones feed at the bottom.
						</p>
						<p className="mt-md">
							You might be thinking: “of course the smarter, larger models will
							do better!” But some of the results might surprise you.
						</p>
					</div>

					<ol className="flex max-w-[540px] animate-[about-rise_0.6s_cubic-bezier(0.16,1,0.3,1)_0.4s_both] flex-col gap-[clamp(7px,1.4vh,16px)]">
						{STEPS.map((step) => (
							// A RULE ABOVE EACH, not a box around it. The three are a
							// sequence, and a rule is the mark that says "next" without
							// closing anything off.
							<li
								key={step.n}
								className="grid grid-cols-[34px_minmax(0,1fr)] items-baseline gap-x-md border-t border-mark-8 pt-[clamp(7px,1.3vh,14px)]"
							>
								<span className="font-mono text-[10.5px] font-bold tracking-[0.16em] text-ink-40">
									{step.n}
								</span>
								<p className="font-sans text-[clamp(13px,1vw,15px)] font-medium leading-[1.5] text-ink-64">
									<span className="font-black tracking-[0.02em] uppercase text-ink">
										{step.lead}
									</span>{" "}
									{step.body}
								</p>
							</li>
						))}
					</ol>
				</div>

				{/* THE HINT SITS IN THE MOON'S HALF, bottom left — the one corner of
				    this screen with nothing else in it. */}
				<div className="pointer-events-none absolute bottom-[22px] left-[clamp(26px,3vw,58px)] flex animate-[about-rise_0.6s_cubic-bezier(0.16,1,0.3,1)_0.8s_both] flex-col items-center gap-xs">
					<span className="font-mono text-[9.5px] font-bold tracking-[0.26em] uppercase text-ink-40">
						Scroll
					</span>
					<span
						aria-hidden
						className="animate-[about-nudge_2.4s_ease-in-out_infinite] font-mono text-[11px] text-ink-40"
					>
						↓
					</span>
				</div>
			</section>

			{/* ================= who we are ===================================== */}
			<section className="relative flex h-dvh snap-start snap-always items-center justify-start overflow-hidden px-[clamp(26px,3vw,58px)] pt-[68px] pb-[40px]">
				<div className={COLUMN}>
					<p className={EYEBROW}>Who we are</p>

					<h2 className={`${HEADLINE} text-[clamp(32px,4vw,56px)]`}>
						Starshot Labs builds foundation models for 3D.
					</h2>

					<div className={`${BODY} max-w-[540px] leading-[1.62]`}>
						<p>
							We are a research team working on generative 3D: models that turn
							language into geometry, layout, and the relationships between
							objects in a scene.
						</p>
						<p className="mt-md">
							SceneBench grew out of our own evaluation work. We needed a way to
							compare how language models reason about space, and head-to-head
							voting gave us a clearer signal than any rubric we wrote ourselves.
							We made it public because the same question is open for everyone
							building in this space.
						</p>
					</div>

					<div className="flex flex-wrap items-center gap-[clamp(16px,2vw,28px)] pt-[clamp(2px,1vh,8px)]">
						{/* #TODO: no destination yet — Starshot's own site once there is
						    one to point at. */}
						<Button variant="solid" sweep shape="standalone" href="/">
							Starshot Labs
							<span aria-hidden className="ml-xs text-[13px]">
								↗
							</span>
						</Button>
						<Link
							href="/leaderboard"
							className="font-mono text-[10.5px] font-medium tracking-[0.22em] uppercase text-ink-64 shadow-[inset_0_-1px_0_var(--color-mark-16)] transition-[color,box-shadow] duration-quick hover:text-ink hover:shadow-[inset_0_-1px_0_var(--color-mark)]"
						>
							View the leaderboard
						</Link>
					</div>
				</div>
			</section>

			{/* ================= two ways in ==================================== */}
			<section className="relative flex h-dvh snap-start snap-always flex-col items-center overflow-hidden px-[clamp(26px,3vw,58px)] pt-[clamp(120px,18vh,200px)]">
				<div className="flex w-[min(880px,100%)] flex-col items-center gap-[clamp(18px,3vh,34px)] text-center">
					<p className={EYEBROW}>Get started</p>

					<h2 className={`${HEADLINE} text-[clamp(34px,4.4vw,60px)]`}>
						Two ways in.
					</h2>

					{/* TWO CARDS, ONE FILLED AND ONE OUTLINED, which is the whole of the
					    hierarchy: both are ways in, but voting costs a reader nothing and
					    generating asks them for a prompt — so one is offered and the
					    other is merely available. */}
					<div className="grid w-full grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-[clamp(12px,1.4vw,20px)] pt-[clamp(6px,1.5vh,18px)]">
						<Link
							href="/"
							className="flex flex-col items-start gap-[9px] bg-mark px-[clamp(22px,2.4vw,32px)] py-[clamp(22px,3vh,32px)] text-left transition-transform duration-settle ease-out-soft hover:-translate-y-[3px]"
						>
							<span className="font-sans text-[clamp(18px,1.8vw,24px)] font-black tracking-[0.03em] uppercase text-ground">
								Vote on scenes
							</span>
							<span className="font-sans text-sm font-medium leading-[1.5] text-ground/60">
								Compare two builds of the same prompt and pick the one that gets
								the space right.
							</span>
						</Link>

						<Link
							href="/"
							className="flex flex-col items-start gap-[9px] border border-mark-16 px-[clamp(22px,2.4vw,32px)] py-[clamp(22px,3vh,32px)] text-left transition-[transform,border-color] duration-settle ease-out-soft hover:-translate-y-[3px] hover:border-mark"
						>
							<span className="font-sans text-[clamp(18px,1.8vw,24px)] font-black tracking-[0.03em] uppercase text-ink">
								Generate your own
							</span>
							<span className="font-sans text-sm font-medium leading-[1.5] text-ink-64">
								Describe any scene and watch two models build it head to head.
							</span>
						</Link>
					</div>
				</div>
			</section>
		</AboutStage>
	);
}
