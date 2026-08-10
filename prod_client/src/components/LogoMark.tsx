"use client";

import Image from "next/image";

/**
 * The mark: a glass disc, cut on the diagonal — white until you look at it.
 *
 * AT REST IT IS A SILHOUETTE. `brightness(0) invert(1)` crushes every pixel to
 * black and then flips it, which leaves the artwork's alpha intact and its colour
 * gone: one flat white shape, the same weight as the wordmark beside it. The
 * colour is held back so that arriving at the mark is an event.
 *
 * ON HOVER IT TURNS. Not literally — rotating a flat picture only reveals that it
 * is flat, and a 3D asset squashed on its Y axis reads as a closing door. What
 * turns is the LIGHT: a terminator sweeps across the face and the colour is
 * uncovered behind it, with a specular glint riding just ahead of the edge. Three
 * things sell it as a rotation rather than a wipe:
 *
 *   1. THE EDGE IS CURVED. The mask is an ellipse anchored off to one side, so
 *      the boundary between white and colour is a bowed line — which is what the
 *      terminator on a sphere is. A straight edge would read as a blind opening.
 *   2. THE TIMING IS ANGULAR. A point on a spinning sphere crosses the middle of
 *      the face fastest and the limbs slowest, so the sweep is eased in and out
 *      rather than run at a constant rate.
 *   3. THE HIGHLIGHT MOVES AGAINST IT. The glint crosses in the same direction but
 *      on its own schedule, so the surface reads as curved and lit from a fixed
 *      point rather than as an image being slid into view.
 *
 * The artwork is drawn TWICE, which costs nothing — it is one cached file, and
 * the second copy is the same decoded image.
 */
export default function LogoMark({ className }: { className?: string }) {
    return (
        // `aria-hidden`: the wordmark beside it already says SceneBench, and a
        // screen reader does not need to be told twice.
        <span aria-hidden className={`relative block ${className ?? ""}`}>
            {/* THE FLAT STATE, AND IT IS A FILL RATHER THAN A FILTER.

			    This was the artwork under `brightness(0) invert(1)` — a filter chain
			    whose only possible output is white, so the mark was the last thing on
			    the page that could not follow the palette. Masked to the same silhouette
			    instead (the glint below already works this way), it is a `bg-mark`
			    rectangle showing through the logo's own alpha, so it takes whatever the
			    mark currently is and needs no second artwork. */}
            <span className="absolute inset-0 [mask-image:url(/logo.png)] [mask-position:center] [mask-repeat:no-repeat] [mask-size:contain]">
                <span className="absolute inset-0 bg-mark transition-[mask-position] duration-[820ms] ease-[cubic-bezier(0.4,0,0.6,1)] [mask-image:linear-gradient(#000_0_0),radial-gradient(72%_150%_at_10%_50%,#000_0_58%,transparent_80%)] [mask-position:center,130%_50%] [mask-repeat:no-repeat] [mask-size:100%_100%,215%_100%] [mask-composite:subtract] group-hover/mark:[mask-position:center,0%_50%]" />
            </span>

            {/* THE COLOUR, uncovered by the terminator. The mask is twice the width of
			    the mark and slides from one end to the other, so what crosses the face
			    is the ellipse's soft edge — and because the mask keeps travelling past
			    the far limb, the colour is left fully lit rather than half-covered.

			    AT FULL STRENGTH, which is the point of the reveal — the payoff is the
			    artwork as it was rendered, and anything short of that is a worse
			    picture of it.

			    It was briefly held at 45% over the black silhouette, on the reasoning
			    that a pale glass disc drawn for a black page must be too light to hold
			    against the cream. The reasoning was fine and the fix was not: scaling
			    a light-lit render toward black does not relight it as dark glass, it
			    greys it. Highlight and shadow come down by the same factor, so nothing
			    is re-lit and the whole disc loses chroma together — the result reads as
			    a dimmed photograph, not a darker object. If the mark ever does need
			    more separation from the paper, it needs artwork lit for a light ground
			    or something to sit against, not a multiplier on this one.

			    IT STARTS CLEAR OF THE MARK, and at 100% it did not. The mask is 215%
			    wide with its ellipse centred at 10% of that and opaque to 58% of a 72%
			    radius, fading out at 80% — so from 100% the fade reached 30% of the way
			    across the face and the leading edge of the colour sat at roughly 89%
			    alpha before the pointer arrived. Solved rather than nudged: the
			    transparent boundary clears the left edge once the position passes
			    126.4%, and 130% takes that with a little room. Same trap as the
			    wordmark's gleam, which is solved the same way in Navbar. */}
            <Image
                src="/logo.png"
                alt=""
                fill
                sizes="86px"
                priority
                className="object-contain transition-[mask-position] duration-[820ms] ease-[cubic-bezier(0.4,0,0.6,1)] [mask-image:radial-gradient(72%_150%_at_10%_50%,#000_0_58%,transparent_80%)] [mask-position:130%_50%] [mask-repeat:no-repeat] [mask-size:215%_100%] group-hover/mark:[mask-position:0%_50%]"
            />

            {/* THE GLINT. Masked to the artwork's own silhouette so the highlight stays
			    on the glass instead of sweeping a rectangle across the header, and
			    positioned so that at both ends of its travel the bright band sits off
			    the mark entirely — it exists only in passing, which is what a specular
			    is.

			    THE BAND IS PAPER, NOT INK, and that is the whole of the fix. It was
			    written as `--ink-rgb`, which meant "the light" only for as long as the
			    page was black — inverted for the bar, ink resolves to near-black and
			    the highlight became a dark band raking across a dark disc. A specular
			    that DARKENS is not a specular; it is a smear, and it was landing on the
			    one moment the mark is meant to look lit.

			    `--paper-rgb` is the right name for it because it is a literal at :root
			    and never re-pointed, so it stays light in every subtree — which is what
			    a highlight has to be. */}
            <span className="pointer-events-none absolute inset-0 transition-[background-position] duration-[820ms] ease-[cubic-bezier(0.4,0,0.6,1)] [background-image:linear-gradient(104deg,transparent_34%,rgb(var(--paper-rgb)_/_0.92)_50%,transparent_66%)] [background-position:118%_50%] [background-repeat:no-repeat] [background-size:300%_100%] [mask-image:url(/logo.png)] [mask-position:center] [mask-repeat:no-repeat] [mask-size:contain] group-hover/mark:[background-position:-18%_50%]" />
        </span>
    );
}
