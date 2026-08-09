import type { Metadata } from "next";
import Link from "next/link";
import AboutStage from "@/components/about/AboutStage";
import ScrollBox from "@/components/site/ScrollBox";
import Button from "@/components/ui/Button";

export const metadata: Metadata = {
	title: "About — SceneBench",
	description:
		"How well can a language model build in 3D? SceneBench puts them head to head and lets the public decide.",
};

const EYEBROW =
	"font-mono text-[10.5px] font-bold tracking-[0.26em] uppercase text-ink-40";

const HEADLINE =
	"font-serif italic font-normal leading-[1.04] tracking-[-0.01em] text-ink text-pretty";

const BODY =
	"font-sans text-[clamp(13px,1.1vw,16.5px)] font-medium leading-[1.55] text-ink-64 text-pretty";

const COLUMN =
	"max-h-full min-h-0 w-[min(620px,54vw)] max-md:w-auto";

const COLUMN_PORT =
	"flex flex-col gap-[clamp(10px,2.6vh,30px)]";

// The first screen rises on mount; the rest cannot, because they are a scroll
// away and would have played to nobody. AboutStage marks a screen the first time
// it arrives and this waits for that mark. See data-arrived there.
const ARRIVE = "[[data-arrived]_&]:lift-in";

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

export default function AboutPage() {
	return (
		<AboutStage>
			<section className="relative flex h-dvh snap-start snap-always items-center justify-end overflow-hidden px-[clamp(26px,3vw,58px)] pt-[68px] pb-[40px]">
				<ScrollBox className={COLUMN} viewportClassName={COLUMN_PORT}>
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
				</ScrollBox>

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

			<section className="relative flex h-dvh snap-start snap-always items-center justify-start overflow-hidden px-[clamp(26px,3vw,58px)] pt-[68px] pb-[40px]">
				<ScrollBox className={COLUMN} viewportClassName={COLUMN_PORT}>
					<p className={`${EYEBROW} ${ARRIVE}`}>Who we are</p>

					<h2
						className={`${HEADLINE} ${ARRIVE} text-[clamp(32px,4vw,56px)]`}
						style={{ ["--lift-delay" as string]: "80ms" }}
					>
						Starshot Labs builds foundation models for 3D.
					</h2>

					<div
						className={`${BODY} ${ARRIVE} max-w-[540px] leading-[1.62]`}
						style={{ ["--lift-delay" as string]: "160ms" }}
					>
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

					<div
						className={`${ARRIVE} flex flex-wrap items-center gap-[clamp(16px,2vw,28px)] pt-[clamp(2px,1vh,8px)]`}
						style={{ ["--lift-delay" as string]: "240ms" }}
					>
						{/* #TODO: no destination yet — link to Starshot's site when ready. */}
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
				</ScrollBox>
			</section>

			<section className="relative flex h-dvh snap-start snap-always flex-col items-center overflow-hidden px-[clamp(26px,3vw,58px)] pt-[clamp(120px,18vh,200px)]">
				<div className="flex w-[min(880px,100%)] flex-col items-center gap-[clamp(18px,3vh,34px)] text-center">
					<p className={`${EYEBROW} ${ARRIVE}`}>Get started</p>

					<h2
						className={`${HEADLINE} ${ARRIVE} text-[clamp(34px,4.4vw,60px)]`}
						style={{ ["--lift-delay" as string]: "80ms" }}
					>
						Two ways in.
					</h2>

					{/* The reveal sits on the grid, not on the two cards: they lean up on
					    hover, and a filled animation would hold their transform for good. */}
					<div
						className={`${ARRIVE} grid w-full grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-[clamp(12px,1.4vw,20px)] pt-[clamp(6px,1.5vh,18px)]`}
						style={{ ["--lift-delay" as string]: "170ms" }}
					>
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
