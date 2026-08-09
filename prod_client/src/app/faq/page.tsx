import type { Metadata } from "next";
import Fade from "@/components/site/Fade";
import Masthead, { Title } from "@/components/site/Masthead";
import PageShell from "@/components/site/PageShell";
import VoxelSky from "@/components/site/VoxelSky";
import Button from "@/components/ui/Button";

export const metadata: Metadata = {
	title: "FAQ — SceneBench",
	description:
		"How SceneBench pairs models, how the ratings move, and what the renders you are voting on actually are.",
};

// #TODO: placeholder copy — rewrite once pairings and ratings are real.
const FAQS: { q: string; a: string[] }[] = [
	{
		q: "What am I actually looking at?",
		a: [
			"Two 3D scenes built from the same prompt by two different language models. The models do the spatial reasoning — what goes where, how big, facing which way — and a single in-house generation model turns each object into geometry, so the only variable between the two sides is the model that planned the scene.",
			"Both panels are live 3D. Drag to orbit, scroll to zoom, click to step inside and walk around.",
		],
	},
	{
		q: "Why aren't the models named before I vote?",
		a: [
			"Because knowing which is which is the fastest way to stop looking. Naming a model next to its render answers the question the page is asking, so the labels arrive after your vote and not before it.",
		],
	},
	{
		q: "How are the two sides paired?",
		a: [
			"At random from the models that have a build for that prompt. Neither side is favoured by position — which model lands on the left is a coin flip on every round.",
		],
	},
	{
		q: "How do the ratings move?",
		a: [
			"Every vote is a head-to-head result, and the winner takes rating from the loser. A model that beats one rated well above it gains more than it would for beating a peer, and skipping a round moves nothing.",
			"Ratings only mean something in bulk. A model near the top after a handful of votes is noise; the board is worth reading once a pairing has been seen a few hundred times.",
		],
	},
	{
		q: "What does SKIP do?",
		a: [
			"Ends the round without a result. Use it when neither build answers the prompt, or when you genuinely cannot separate them — a coin-flip vote is worse than no vote, because it is indistinguishable from a real judgement.",
		],
	},
	{
		q: "Can I submit my own prompt?",
		a: [
			"Yes — the field under the vote takes one, and it goes into the pool that future rounds are drawn from. Prompts that describe a place tend to produce a better comparison than prompts that describe an object.",
		],
	},
	{
		q: "Why does a build look broken?",
		a: [
			"Sometimes it is: a model can put a staircase through a wall or float a roof off its walls entirely, and that is exactly the failure the benchmark exists to catch. Vote on what you see. A scene that is wrong in an interesting way is still a result.",
		],
	},
	{
		q: "Who made this?",
		a: [
			"Starshot Labs. The generation model, the harness the models are scored in, and this site are all ours.",
		],
	},
];

export default function FaqPage() {
	return (
		<div className="relative">
			<VoxelSky />

			<PageShell
				masthead={
					<Masthead label="Frequently asked" placement="flow">
						<Title voice="name">FAQ</Title>
					</Masthead>
				}
				footer={
					<div
						className="lift-in flex items-center justify-center gap-md border-t border-mark-8 bg-ground px-lg py-md"
						style={{ ["--lift-delay" as string]: "320ms" }}
					>
						<span className="font-label text-2xs text-ink-40">
							Still wondering?
						</span>
						<Button href="/" shape="standalone" variant="cta" sweep>
							Go to the arena
						</Button>
					</div>
				}
			>
				{/* Leave-only: the questions arrive one at a time on their own, and a
				    veil over the whole list would just fade the cascade in behind it. */}
				<Fade enter={null}>
					<dl className="relative z-10 flex flex-col">
						{FAQS.map((item, i) => (
							<div
								key={item.q}
								className="lift-in border-t border-mark-8 py-lg first:border-t-0 first:pt-0"
								style={{ ["--lift-delay" as string]: `${140 + i * 55}ms` }}
							>
								<dt className="font-sans text-base font-extrabold tracking-[-0.01em] text-ink text-balance">
									{item.q}
								</dt>
								{item.a.map((para) => (
									<dd
										key={para}
										className="mt-sm max-w-[62ch] font-sans text-sm leading-[1.65] text-ink-64 text-pretty"
									>
										{para}
									</dd>
								))}
							</div>
						))}
					</dl>
				</Fade>
			</PageShell>
		</div>
	);
}
