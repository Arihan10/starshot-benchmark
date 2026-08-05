// The moon behind the masthead.
//
// A BODY AGAIN, not weather. It has been a solid disc, then a wash, then a
// crescent; this is the solid disc, and the reason to come back is that the
// masthead now has three things to hold together — the navbar's centre, the
// question, and the prompt — and a wash cannot hold anything. A lit surface can:
// everything sitting on it reads as one object because it shares a ground.
//
// ONLY THE LOWER CAP IS EVER ON SCREEN. The disc hangs off the top of the window,
// which is what makes it read as a horizon rising behind the page rather than a
// ball sitting on it, and is why it has to be large — a small circle cropped to a
// shallow band reads as a dome, not a moon.

// Degrees per turn is fixed; the DURATION is what makes it read as celestial rather
// than as a loading spinner. At four-plus minutes the movement is below the
// threshold you can watch directly — you only notice the craters moved if you look
// away and come back.
const SPIN_SECONDS = 260;

// Lit from just above centre, so the cap we see is the shaded lower limb and the
// disc has somewhere to fall off to.
// THE DISC'S FILL LIVES IN globals.css, not here. It was five literals — #EDEDED
// at the centre falling to #A4A7AE at the limb — which made the largest shape on
// the site the one that did not follow the palette: recolouring the page left the
// moon behind, because a hand-copy of a token is not the token. See `--moon-rgb`.
const SURFACE = "var(--moon-surface)";

// Craters as polar coordinates rather than hand-written percentages: they sit on a
// RING, all the way round, so the rotation always has fresh ones to bring through
// the visible cap. `deg` is clockwise from the bottom of the disc, `dist` is percent
// of the radius from centre, `size` is the crater radius in px.
//
// DENSER THAN IT NEEDS TO BE TO LOOK LIKE A MOON, because it also has to look like
// a moon that TURNS. A sparse surface is a fine still image and a poor wheel.
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

// THE BRIGHT ONES. A young crater throws pale ejecta around itself, and on a grey
// body that is the highest-contrast thing there is — which makes these the features
// you actually track as the surface turns. Kept few: they are landmarks, and a sky
// full of landmarks has none.
const BRIGHT: { deg: number; dist: number; size: number; alpha: number }[] = [
	{ deg: 40, dist: 44, size: 26, alpha: 0.5 },
	{ deg: 127, dist: 47, size: 17, alpha: 0.42 },
	{ deg: 205, dist: 41, size: 30, alpha: 0.38 },
	{ deg: 288, dist: 45, size: 20, alpha: 0.46 },
];

// MARIA — the broad dark plains. Much larger than any crater, much fainter, and
// elliptical at varying proportions so no two read as the same mark. Placed on the
// same ring but deliberately NOT aligned with the craters: on the real moon the
// plains are what the craters sit in, so they have to pass under a crater field
// rather than politely fill the gaps between them.
const MARIA: { deg: number; dist: number; rx: number; ry: number; alpha: number }[] = [
	{ deg: 34, dist: 40, rx: 210, ry: 140, alpha: 0.055 },
	{ deg: 118, dist: 46, rx: 150, ry: 190, alpha: 0.045 },
	{ deg: 178, dist: 38, rx: 240, ry: 130, alpha: 0.05 },
	{ deg: 246, dist: 44, rx: 170, ry: 160, alpha: 0.04 },
	{ deg: 308, dist: 41, rx: 200, ry: 120, alpha: 0.05 },
];

/** Ring position -> the `at x% y%` a radial-gradient wants. */
function place(deg: number, dist: number): string {
	// +90 degrees so 0 lands at the bottom of the disc, which is the part on screen.
	const rad = ((deg + 90) * Math.PI) / 180;
	return `${(50 + dist * Math.cos(rad)).toFixed(2)}% ${(50 + dist * Math.sin(rad)).toFixed(2)}%`;
}

// Earlier layers paint on top, so the order is smallest-first: the dishes sit ON
// the plains rather than the plains washing over them.
const CRATER_LAYERS = [
	...BRIGHT.map(
		(b) =>
			`radial-gradient(circle ${b.size}px at ${place(b.deg, b.dist)}, rgb(var(--moon-rgb) / ${b.alpha}) 0%, rgb(var(--moon-rgb) / ${(b.alpha * 0.4).toFixed(3)}) 42%, transparent 100%)`,
	),
	...CRATERS.map(
		(c) =>
			`radial-gradient(circle ${c.size}px at ${place(c.deg, c.dist)}, rgba(9,11,16,${c.alpha}) 0%, rgba(9,11,16,${(c.alpha * 0.5).toFixed(3)}) 60%, transparent 100%)`,
	),
	...MARIA.map(
		(m) =>
			`radial-gradient(ellipse ${m.rx}px ${m.ry}px at ${place(m.deg, m.dist)}, rgba(11,14,20,${m.alpha}) 0%, rgba(11,14,20,${(m.alpha * 0.55).toFixed(3)}) 52%, transparent 100%)`,
	),
].join(", ");

/**
 * The moon, sized to `diameter` and anchored so its BOTTOM edge meets the bottom of
 * the nearest positioned ancestor — which is the masthead band, and which therefore
 * decides how much of the disc shows.
 *
 * LIFTED by `--moon-lift`, set on the band. The disc used to sit hard on the band's
 * bottom edge, which put its tip in contact with the canvases below — two objects
 * touching with no gap read as one badly-joined object rather than as a moon above
 * a comparison. The prompt is lifted by the SAME variable, because the two have to
 * stay concentric; moving one without the other is how the arc leaves the limb.
 *
 * ANCHORED FROM THE BOTTOM, and that is not arbitrary: CurvedPrompt anchors its
 * square viewBox exactly the same way and is handed the same `diameter`, so the two
 * boxes share a centre and the arc the prompt is struck on IS a circle about the
 * moon's own centre. Matching radii is not enough — an arc hung from the top of the
 * band while the disc hung from the bottom drifts off the limb as it crosses the
 * middle, however carefully the radius is tuned.
 */
export default function Moon({ diameter }: { diameter: string }) {
	return (
		<div
			aria-hidden
			className="pointer-events-none absolute bottom-[var(--moon-lift,0px)] left-1/2 -translate-x-1/2 rounded-full"
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
