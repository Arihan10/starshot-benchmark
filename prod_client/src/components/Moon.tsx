// The moon behind the masthead.
//
// It is ENORMOUS and almost entirely off-screen — only the bottom cap of the disc
// is ever visible, which is what makes it read as a horizon rising behind the page
// rather than as a ball sitting on it. The parent clips it; this file only decides
// what the disc looks like and how it turns.
//
// A shallow cap is also why the disc has to be so large: the visible arc's
// curvature is what sells it, and a small circle cropped to a 20px band reads as a
// dome, not a moon.

// Degrees per full turn is fixed; the DURATION is what makes it read as celestial
// rather than as a loading spinner. At four-plus minutes the movement is below the
// threshold you can watch directly — you only notice the craters have moved if you
// look away and come back.
const SPIN_SECONDS = 260;

// Lit from just above centre, so the cap we actually see is the shaded lower limb
// and the disc has somewhere to fall off to.
const SURFACE =
	"radial-gradient(circle at 50% 44%, #EDEDED 0%, #E8E8EA 68%, #D7D8DC 88%, #BDBFC5 96.5%, #A4A7AE 100%)";

// Craters as polar coordinates rather than hand-written percentages: they are
// placed on a RING, all the way round, so the rotation always has fresh ones to
// bring through the visible cap. (Authoring them as x/y percentages — the obvious
// way — tempts you to place them only where they currently show, and then the
// moon spins to a blank face a minute later.)
//
// `deg` is measured clockwise from the bottom of the disc, `dist` is percent of
// the radius from centre, `size` is the crater radius in px on the disc.
const CRATERS: { deg: number; dist: number; size: number; alpha: number }[] = [
	{ deg: 0, dist: 46, size: 46, alpha: 0.115 },
	{ deg: 22, dist: 43, size: 30, alpha: 0.125 },
	{ deg: 48, dist: 47, size: 38, alpha: 0.1 },
	{ deg: 75, dist: 41, size: 54, alpha: 0.085 },
	{ deg: 104, dist: 45, size: 26, alpha: 0.09 },
	{ deg: 133, dist: 44, size: 60, alpha: 0.075 },
	{ deg: 162, dist: 47, size: 34, alpha: 0.105 },
	{ deg: 195, dist: 42, size: 44, alpha: 0.095 },
	{ deg: 228, dist: 46, size: 28, alpha: 0.115 },
	{ deg: 261, dist: 43, size: 50, alpha: 0.08 },
	{ deg: 295, dist: 47, size: 32, alpha: 0.1 },
	{ deg: 330, dist: 44, size: 40, alpha: 0.09 },
];

// A few craters get a rim instead of a soft dish — the mix is what stops the
// surface reading as evenly-spaced blur. Kept FAINT on purpose: a ring is a much
// louder shape than a dish, and at full strength on a disc this size they stop
// reading as craters and start reading as soap bubbles sitting on the glass.
const RIMMED: { deg: number; dist: number; size: number; alpha: number }[] = [
	{ deg: 12, dist: 45, size: 20, alpha: 0.09 },
	{ deg: 88, dist: 46, size: 11.5, alpha: 0.08 },
	{ deg: 210, dist: 44, size: 8, alpha: 0.075 },
	{ deg: 315, dist: 45, size: 14, alpha: 0.085 },
];

/** Ring position → the `at x% y%` a radial-gradient wants. */
function place(deg: number, dist: number): string {
	// +90° so 0 lands at the bottom of the disc, which is the part on screen.
	const rad = ((deg + 90) * Math.PI) / 180;
	const x = 50 + dist * Math.cos(rad);
	const y = 50 + dist * Math.sin(rad);
	return `${x.toFixed(2)}% ${y.toFixed(2)}%`;
}

const CRATER_LAYERS = [
	...CRATERS.map(
		(c) =>
			`radial-gradient(circle ${c.size}px at ${place(c.deg, c.dist)}, rgba(9,11,16,${c.alpha}) 0%, rgba(9,11,16,${(c.alpha * 0.5).toFixed(3)}) 60%, transparent 100%)`,
	),
	...RIMMED.map(
		(c) =>
			`radial-gradient(circle ${c.size + 6}px at ${place(c.deg, c.dist)}, transparent 0 ${c.size}px, rgba(9,11,16,${c.alpha}) ${c.size}px ${c.size + 2}px, rgba(9,11,16,${(c.alpha * 0.35).toFixed(3)}) ${c.size + 2}px ${c.size + 5}px, transparent ${c.size + 5}px)`,
	),
	// A faint directional grain over everything, so the large smooth areas between
	// craters are not perfectly flat.
	"repeating-linear-gradient(58deg, rgba(9,11,16,0.03) 0 1px, transparent 1px 5px)",
].join(", ");

/**
 * Renders the disc only. Sizing and clipping belong to the caller, which is the
 * thing that knows how much of the moon should show — this just fills whatever
 * box it is given, anchored so its BOTTOM edge meets the bottom of that box.
 */
export default function Moon({ diameter }: { diameter: string }) {
	return (
		<div
			aria-hidden
			className="pointer-events-none absolute left-1/2 top-full -translate-x-1/2 -translate-y-full rounded-full"
			style={{ width: diameter, height: diameter, background: SURFACE }}
		>
			{/* The surface turns; the lighting does not. Separating them is the whole
			    trick — rotating the lit disc as well would swing the highlight around
			    and read as the light source orbiting, not the moon turning. */}
			<div
				className="absolute inset-0 rounded-full"
				style={{
					backgroundImage: CRATER_LAYERS,
					animation: `moon-spin ${SPIN_SECONDS}s linear infinite`,
					willChange: "transform",
				}}
			/>
		</div>
	);
}
