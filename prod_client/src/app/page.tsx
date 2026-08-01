"use client";

import Image from "next/image";
import { useState, type ReactNode } from "react";
import ScenePanel from "@/components/arena/ScenePanel";
import VoteBar from "@/components/arena/VoteBar";
import CurvedPrompt from "@/components/CurvedPrompt";
import Moon from "@/components/Moon";
import { LEFT_VOTE_SHARE, LOCAL_CELLS } from "@/lib/localScenes";

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

// #TODO: hard-coded. The prompt belongs to the run that produced these cells —
// the pipeline already stores it per slot (runs/<run>/prompts) — so it should
// travel WITH the cell pair rather than being restated here, or the header and
// the scenes under it can drift apart silently. Held in natural case, not caps:
// the shouting is styling, and a real prompt arrives as a sentence.
const PROMPT = "A super mario style platformer level";

// How far the moon's lower limb reaches past the navbar. This is the whole effect
// — too little and the disc reads as a gradient behind the header, too much and it
// stops being a horizon and becomes a shape on the page.
const MOON_PROTRUDE = "clamp(8px,1.2vh,16px)";

// The arc the prompt is set on, in CurvedPrompt's viewBox units.
//
// DERIVED FROM THE DISC, not chosen by eye. The moon's radius is half
// MOON_DIAMETER; the text's baseline sits a little above the disc's lowest point,
// so its own radius is that minus the gap between the two — about 24px once the
// protrusion above is accounted for. Converting to viewBox units divides by the
// SVG's scale (PROMPT_WIDTH / 1000):
//
//   (450 − 24) / (480 / 1000)  ≈  885
//
// Both terms are anchored to the same vw, so the ratio holds as the viewport
// changes and the arc stays concentric rather than drifting flat or tight.
//
// The WIDTH is now bounded by height, not by taste. The prompt is overlaid on a
// band the navbar sizes, so its whole arc has to fit inside that band — and the
// bow grows with the square of the text's half-width, so widening it is what
// pushed the disc down the page before. Narrower text buys a shallower curve.
const PROMPT_RADIUS = 885;
const PROMPT_WIDTH = "clamp(260px,30vw,480px)";

// Capped in BOTH directions. The vw term keeps the arc in proportion on a wide
// display, and the ceiling stops it flattening into a straight edge on an
// ultrawide one — past a certain size a circle cropped to a 100px band has no
// visible curvature left, and the moon just looks like a light grey bar.
//
// The UPPER bound is also what keeps the disc clear of the mark and the nav. A
// wider moon puts its limb straight through a nav label, and a word that is half
// dark-on-moon and half light-on-black reads as broken rather than as blended —
// the prompt in the middle is what the moon is here to sit behind, so the disc is
// sized to hold that and nothing else.
const MOON_DIAMETER = "min(54vw, 900px)";

type Side = "a" | "b";

// What the winner gains and the loser drops. A fixed swing rather than a real Elo
// K-factor calculation: the ratings here are placeholders, so computing an exact
// expected score off them would be arithmetic performed on fiction.
// #TODO: the server owns this — it holds the ratings, and it is the only place
// that can apply the result of a vote to them.
const ELO_SWING = 12;

export default function Page() {
    // The whole round is one piece of state. Everything downstream — who glows,
    // who shatters, which numbers count up — is derived from it, so there is no
    // way for the panels to disagree about what happened.
    const [vote, setVote] = useState<Side | null>(null);

    return (
        <main className="flex h-dvh flex-col overflow-hidden bg-black">
            {/* THE MASTHEAD BAND. The moon is clipped to exactly this box and
                anchored to its bottom edge, so the padding below the navbar is
                precisely how far the lower limb protrudes past it.

                `isolate` is load-bearing, not tidiness: the chrome below blends in
                `difference` mode, and blending only reaches a backdrop inside the
                same stacking context. Without it the text would blend against the
                page instead of against the moon. For the same reason the navbar
                takes NO z-index — DOM order already paints it above the disc, and
                a z-index would cut it out of the moon's stacking context and break
                the effect it depends on. */}
            {/* The margin is what the removed caption used to provide: without it
                the canvases butt straight against the moon's limb and the disc
                reads as sitting on them rather than behind everything. */}
            <div
                className="relative isolate flex-none mb-[clamp(10px,1.9vh,26px)]"
                style={{ paddingBottom: MOON_PROTRUDE }}
            >
                {/* THE NAVBAR SETS THE BAND'S HEIGHT, and the disc is clipped to
                    that box — so MOON_PROTRUDE is measured from the bottom of the
                    navbar and means exactly what it says. It used to be the PROMPT
                    that sized the band, which made the limb chase the prompt down
                    the page: a longer prompt curved deeper, the band grew, and the
                    moon swung far below the chrome it was supposed to sit behind. */}
                <div className="pointer-events-none absolute inset-0 overflow-hidden">
                    <Moon diameter={MOON_DIAMETER} />
                </div>
                <header className="relative flex items-center justify-between gap-4 px-[clamp(12px,1.5vw,26px)] pb-[clamp(10px,1.5vh,18px)] pt-[clamp(9px,1.4vh,16px)]">
                {/* THE LEFT CLUSTER. About sits over here rather than in the right
                    stack: four items on one side and a bare mark on the other left
                    the masthead visibly heavier to the right, and the prompt between
                    them read as off-centre even though it is not. Splitting the
                    secondary links two-and-two balances the weight without moving
                    the CTA, which has to stay at the end. */}
                <div className="flex min-w-0 items-center gap-[clamp(6px,1vw,18px)]">
                    {/* The mark: glyph beside a two-line wordmark, per the Arena
                        design. `alt` is empty on purpose — the wordmark right next
                        to it already says "SceneBench by Starshot", so describing
                        the glyph too would just make a screen reader say it twice. */}
                    <div className="flex min-w-0 items-center gap-[clamp(4px,0.55vw,9px)]">
                        <div className="relative h-[clamp(50px,5vw,86px)] w-[clamp(50px,5vw,86px)] flex-none">
                            <Image
                                src="/logo.png"
                                alt=""
                                fill
                                sizes="86px"
                                priority
                                className="object-contain"
                            />
                        </div>
                        <div className="flex min-w-0 flex-col gap-1">
                            {/* Same reason as the title: on a narrow viewport the
                                moon's limb reaches the wordmark. */}
                            <span className="font-display text-[clamp(19px,2vw,31px)] font-bold leading-none tracking-[0.13em] whitespace-nowrap mix-blend-difference">
                                SCENEBENCH
                            </span>
                            <span className="font-sans text-[clamp(9px,0.95vw,15px)] font-medium leading-none tracking-[0.17em] whitespace-nowrap text-foreground/70 mix-blend-difference">
                                BY STARSHOT
                            </span>
                        </div>
                    </div>
                    <NavItem onClick={() => {}}>About</NavItem>
                </div>
                {/* ONE loud item, the rest quiet. The secondary links carry no
                    ground and only 60% of the foreground, so they read as
                    available without competing with the CTA — the page should
                    have exactly one thing that looks like the next step. */}
                <nav className="flex items-center justify-end">
                    {/* #TODO: no destinations yet. These become <Link>s the moment
                        /leaderboard, /about and /faq exist; they are buttons only
                        because there is nowhere to point them. */}
                    <NavItem onClick={() => {}}>Leaderboard</NavItem>
                    <NavItem onClick={() => {}}>FAQ</NavItem>
                    {/* The page inverted: the site's foreground (#ededed) becomes
                        the button's ground and its background (#000000) the text.
                        Taken from the theme tokens rather than restated as hexes so
                        the one CTA on the page cannot drift from the palette. */}
                    <button
                        type="button"
                        // #TODO: no action yet. This should take the prompt the user
                        // types and queue a build on both models — see the footer
                        // input in the Arena design; the header button is the same
                        // submit by another route.
                        onClick={() => {}}
                        className="ml-[clamp(4px,1vw,16px)] cursor-pointer rounded-xs bg-foreground px-[clamp(26px,3.1vw,48px)] py-[clamp(14px,1.75vw,24px)] font-sans text-[clamp(13px,1.12vw,18px)] font-semibold uppercase tracking-[0.11em] whitespace-nowrap text-background transition-colors duration-200 hover:bg-[#d8dae0] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:ring-offset-2 focus-visible:ring-offset-black"
                    >
                        Generate
                    </button>
                </nav>
                </header>

                {/* THE PROMPT IS THE BAND'S ONLY FLOW CONTENT, so it — not the
                    navbar — decides how tall the masthead is, and it centres on the
                    page rather than on whatever gap the chrome leaves. The top
                    padding is what holds it clear of the floating navbar above.

                    PAINTED, NOT BLENDED. This used to blend in `difference` so it
                    would survive crossing the moon's limb, but that mode operates
                    on the antialiased edge pixels too: every letter's outline
                    resolved to muddy mid-greys and the type read as soft. Ink on
                    the disc is worth more than the edge case, so the colour is
                    chosen outright — which is only safe because the arc is sized to
                    keep the text inside the disc. */}
                <div className="pointer-events-none absolute inset-x-0 top-0 flex flex-col items-center gap-[clamp(2px,0.4vh,6px)] px-[clamp(12px,1.5vw,26px)] pt-[clamp(4px,0.7vh,9px)] text-center">
                    <span className="font-sans text-[clamp(7px,0.58vw,11px)] font-medium uppercase tracking-[0.24em] whitespace-nowrap text-[#0b0d12]/45">
                        Who built it better?
                    </span>
                    {/* GREY, NOT BLACK. Pure ink on a lit disc reads as a hole
                        punched through it — the moon has no value darker than its
                        own terminator, so true black belongs to the page behind,
                        not to anything sitting on the surface. Two-thirds strength
                        keeps the type part of the moon while still clearing AA
                        contrast against it, and the drop shadow (a filter, since
                        text-shadow does not reach SVG glyphs) seats it on the
                        surface rather than floating it above. */}
                    <h1 className="text-[#0b0d12]/68 filter-[drop-shadow(0_2px_9px_rgba(9,11,16,0.18))]">
                        <CurvedPrompt
                            text={`"${PROMPT}"`}
                            width={PROMPT_WIDTH}
                            radius={PROMPT_RADIUS}
                        />
                    </h1>
                </div>
            </div>

            <div className="flex min-h-0 flex-1 flex-col md:flex-row">
                {LOCAL_CELLS.map((cell, i) => {
                    const side: Side = i === 0 ? "a" : "b";
                    return (
                        <ScenePanel
                            key={cell.id}
                            cell={cell}
                            outcome={vote === null ? null : vote === side ? "won" : "lost"}
                            share={side === "a" ? LEFT_VOTE_SHARE : 100 - LEFT_VOTE_SHARE}
                            align={side === "a" ? "left" : "right"}
                            dividerRight={i === 0}
                        />
                    );
                })}
            </div>

            <VoteBar
                cells={LOCAL_CELLS}
                vote={vote}
                swing={ELO_SWING}
                onVote={setVote}
                // #TODO: there is one fixed pairing, so this replays the round
                // rather than fetching another. It becomes "ask the server for the
                // next pair" once pairings are served.
                onNext={() => setVote(null)}
            />
        </main>
    );
}

// A secondary header item. Same type and rhythm as the CTA beside it, but no
// ground and a dimmed foreground — the difference is weight of attention, not a
// different kind of control, so the two sit on one baseline and differ only in
// how loudly they ask.
function NavItem({
    children,
    onClick,
}: {
    children: ReactNode;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className="cursor-pointer rounded-xs px-[clamp(7px,0.95vw,14px)] py-[clamp(12px,1.45vw,19px)] font-sans text-[clamp(12px,0.98vw,15.5px)] font-medium uppercase tracking-[0.11em] whitespace-nowrap text-foreground/60 mix-blend-difference transition-colors duration-200 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40"
        >
            {children}
        </button>
    );
}
