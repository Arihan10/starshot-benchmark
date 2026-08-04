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
// DENSER THAN IT NEEDS TO BE TO LOOK LIKE A MOON, because it also has to look
// like a moon that TURNS. A sparse surface is a fine still image and a poor
// wheel: with only a dozen soft dishes on it, an 80° rotation moves almost
// nothing the eye can hold onto, and the disc reads as sitting still while its
// caption changes. Landmarks at mixed sizes and mixed spacings are what let you
// see that the same face is not in front of you any more.
const CRATERS: { deg: number; dist: number; size: number; alpha: number }[] = [
	{ deg: 0, dist: 46, size: 46, alpha: 0.115 },
	{ deg: 12, dist: 38, size: 18, alpha: 0.1 },
	{ deg: 22, dist: 43, size: 30, alpha: 0.125 },
	{ deg: 35, dist: 49, size: 22, alpha: 0.09 },
	{ deg: 48, dist: 47, size: 38, alpha: 0.1 },
	{ deg: 61, dist: 40, size: 16, alpha: 0.12 },
	{ deg: 75, dist: 41, size: 54, alpha: 0.085 },
	{ deg: 92, dist: 48, size: 20, alpha: 0.105 },
	{ deg: 104, dist: 45, size: 26, alpha: 0.09 },
	{ deg: 118, dist: 39, size: 33, alpha: 0.08 },
	{ deg: 133, dist: 44, size: 60, alpha: 0.075 },
	{ deg: 148, dist: 49, size: 19, alpha: 0.11 },
	{ deg: 162, dist: 47, size: 34, alpha: 0.105 },
	{ deg: 176, dist: 40, size: 24, alpha: 0.095 },
	{ deg: 195, dist: 42, size: 44, alpha: 0.095 },
	{ deg: 210, dist: 48, size: 17, alpha: 0.115 },
	{ deg: 228, dist: 46, size: 28, alpha: 0.115 },
	{ deg: 243, dist: 39, size: 36, alpha: 0.08 },
	{ deg: 261, dist: 43, size: 50, alpha: 0.08 },
	{ deg: 278, dist: 48, size: 21, alpha: 0.1 },
	{ deg: 295, dist: 47, size: 32, alpha: 0.1 },
	{ deg: 312, dist: 41, size: 15, alpha: 0.12 },
	{ deg: 330, dist: 44, size: 40, alpha: 0.09 },
	{ deg: 346, dist: 49, size: 25, alpha: 0.085 },
];

// THE BRIGHT ONES. A young crater throws pale ejecta out around itself, and on a
// grey body that is the highest-contrast thing there is — which makes these the
// features you actually track as the surface turns. Kept few: they are landmarks,
// and a sky full of landmarks has none.
const BRIGHT: { deg: number; dist: number; size: number; alpha: number }[] = [
	{ deg: 40, dist: 44, size: 26, alpha: 0.5 },
	{ deg: 127, dist: 47, size: 17, alpha: 0.42 },
	{ deg: 205, dist: 41, size: 30, alpha: 0.38 },
	{ deg: 288, dist: 45, size: 20, alpha: 0.46 },
	{ deg: 342, dist: 38, size: 14, alpha: 0.44 },
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

// MARIA — the broad dark plains, and the honest way to do what the old directional
// grain was reaching for: break up the large flat areas between craters without
// imposing a direction on a body that has none. Much larger than any crater, much
// fainter, and elliptical at varying proportions, so no two read as the same mark.
//
// Placed on the same ring, deliberately NOT aligned with the craters: on the real
// moon the plains are what the craters sit in, so they have to be able to pass
// under a crater field rather than politely fill the gaps between them.
const MARIA: {
	deg: number;
	dist: number;
	rx: number;
	ry: number;
	alpha: number;
}[] = [
	{ deg: 34, dist: 40, rx: 210, ry: 140, alpha: 0.055 },
	{ deg: 118, dist: 46, rx: 150, ry: 190, alpha: 0.045 },
	{ deg: 178, dist: 38, rx: 240, ry: 130, alpha: 0.05 },
	{ deg: 246, dist: 44, rx: 170, ry: 160, alpha: 0.04 },
	{ deg: 308, dist: 41, rx: 200, ry: 120, alpha: 0.05 },
];

/** Ring position → the `at x% y%` a radial-gradient wants. */
function place(deg: number, dist: number): string {
	// +90° so 0 lands at the bottom of the disc, which is the part on screen.
	const rad = ((deg + 90) * Math.PI) / 180;
	const x = 50 + dist * Math.cos(rad);
	const y = 50 + dist * Math.sin(rad);
	return `${x.toFixed(2)}% ${y.toFixed(2)}%`;
}

// Earlier layers paint on top, so the order here is smallest-first: the rims and
// dishes sit ON the plains rather than the plains washing over them.
const CRATER_LAYERS = [
	...BRIGHT.map(
		(b) =>
			`radial-gradient(circle ${b.size}px at ${place(b.deg, b.dist)}, rgba(255,255,255,${b.alpha}) 0%, rgba(255,255,255,${(b.alpha * 0.4).toFixed(3)}) 42%, transparent 100%)`,
	),
	...RIMMED.map(
		(c) =>
			`radial-gradient(circle ${c.size + 6}px at ${place(c.deg, c.dist)}, transparent 0 ${c.size}px, rgba(9,11,16,${c.alpha}) ${c.size}px ${c.size + 2}px, rgba(9,11,16,${(c.alpha * 0.35).toFixed(3)}) ${c.size + 2}px ${c.size + 5}px, transparent ${c.size + 5}px)`,
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

// REGOLITH GRAIN, replacing a `repeating-linear-gradient` that struck 1px lines
// across the whole disc every 5px at 58°. That is not what an absence of flatness
// looks like: repetition at a fixed angle is the one thing the eye is best at
// finding, so at this size it read as brushed metal — machined, and machined in a
// direction.
//
// Fractal noise has no angle and no period to find. It stitches, so one small tile
// covers the disc, and its alpha lives inside the SVG because CSS has no
// per-background-layer opacity.
//
// That alpha is set by what it is FOR: it has to take the sheen off the large
// smooth areas without ever being legible as noise. Anything past ~0.2 is film
// grain sitting on top of the moon rather than dust lying on it.
const GRAIN =
	"url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='moon-grain'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='240' height='240' filter='url(%23moon-grain)' opacity='0.14'/%3E%3C/svg%3E\")";

// The disc hangs off the box it is given by its BOTTOM edge — which is the same
// anchoring CurvedPrompt uses for its viewBox, and that identity is what makes the
// two curves one curve. Changing it here without changing it there silently
// unpins the prompt from the limb.
const ANCHOR =
	"pointer-events-none absolute left-1/2 top-full -translate-x-1/2 -translate-y-full rounded-full";

/**
 * The moon, sized to `diameter` and anchored so its BOTTOM edge meets the bottom
 * of the nearest positioned ancestor — which is the masthead band, and which
 * therefore decides how much of the disc shows.
 *
 * CLIPPED TO THE BAND, so only its lower cap is ever on screen. The disc carried a
 * halo for a while, spilling past the clip onto the page below; it is gone now —
 * the moon reads as a body against black without a light source having to be
 * implied around it, and the prompt sitting on the limb is easier to read without
 * a gradient behind the letters at the edge.
 */
export default function Moon({
	diameter,
	cycle = 0,
}: {
	diameter: string;
	/**
	 * Bumped once per round. Each change turns the SURFACE through 80° — the same
	 * arc the prompt riding on it travels — so a new matchup arrives on a moon that
	 * visibly rotated to bring it round, rather than on one that swapped its
	 * caption. Keyed rather than transitioned: the turn is a one-shot event with a
	 * beginning, and a transition would only interpolate between two resting
	 * angles.
	 */
	cycle?: number;
}) {
	const sized = { width: diameter, height: diameter } as React.CSSProperties;

	return (
		<>
			<div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
				<div className={ANCHOR} style={{ ...sized, background: SURFACE }}>
					{/* THE ROUND'S TURN, on its own layer. Two rotations cannot share one
					    element — the second would replace the first, not add to it — so
					    the 80° cycle sits OUTSIDE the drift and the browser composes the
					    two. `key` restarts it: the drift's phase restarts with it, which
					    is a jump of no consequence, happening as it does underneath the
					    very rotation that is meant to be moving everything. */}
					<div
						key={cycle}
						className="absolute inset-0 rounded-full"
						style={
							cycle
								? {
										animation:
											"moon-cycle 1420ms cubic-bezier(0.42,0,0.14,1) both",
										willChange: "transform",
									}
								: undefined
						}
					>
					{/* The surface turns; the lighting does not. Separating them is the
					    whole trick — rotating the lit disc as well would swing the
					    highlight around and read as the light source orbiting, not the
					    moon turning. */}
					<div
						className="absolute inset-0 rounded-full"
						style={{
							backgroundImage: CRATER_LAYERS,
							animation: `moon-spin ${SPIN_SECONDS}s linear infinite`,
							willChange: "transform",
						}}
					>
						{/* Grain rides the same rotation — it is part of the surface —
						    but has to be its own layer because it needs a background-size
						    to tile at, which a shorthand list of gradients cannot carry
						    per-layer. */}
						<div
							className="absolute inset-0 rounded-full"
							style={{ backgroundImage: GRAIN, backgroundSize: "240px 240px" }}
						/>
					</div>
					</div>
				</div>
			</div>
		</>
	);
}
