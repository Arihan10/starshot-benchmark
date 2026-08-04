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
			<Image
				src="/logo.png"
				alt=""
				fill
				sizes="86px"
				priority
				className="object-contain [filter:brightness(0)_invert(1)]"
			/>

			{/* THE COLOUR, uncovered by the terminator. The mask is twice the width of
			    the mark and slides from one end to the other, so what crosses the face
			    is the ellipse's soft edge — and because the mask keeps travelling past
			    the far limb, the colour is left fully lit rather than half-covered. */}
			<Image
				src="/logo.png"
				alt=""
				fill
				sizes="86px"
				priority
				className="object-contain transition-[mask-position] duration-[820ms] ease-[cubic-bezier(0.4,0,0.6,1)] [mask-image:radial-gradient(72%_150%_at_10%_50%,#000_0_58%,transparent_80%)] [mask-position:100%_50%] [mask-repeat:no-repeat] [mask-size:215%_100%] group-hover/mark:[mask-position:0%_50%]"
			/>

			{/* THE GLINT. Masked to the artwork's own silhouette so the highlight stays
			    on the glass instead of sweeping a rectangle across the header, and
			    positioned so that at both ends of its travel the bright band sits off
			    the mark entirely — it exists only in passing, which is what a specular
			    is. */}
			<span
				className="pointer-events-none absolute inset-0 transition-[background-position] duration-[820ms] ease-[cubic-bezier(0.4,0,0.6,1)] [background-image:linear-gradient(104deg,transparent_34%,rgba(255,255,255,0.92)_50%,transparent_66%)] [background-position:118%_50%] [background-repeat:no-repeat] [background-size:300%_100%] [mask-image:url(/logo.png)] [mask-position:center] [mask-repeat:no-repeat] [mask-size:contain] group-hover/mark:[background-position:-18%_50%]"
			/>
		</span>
	);
}
