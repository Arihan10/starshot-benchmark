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

// THE HALO, and it is always on. Without it the disc meets the page at a hard
// vector edge — a shape cut out of the black rather than a lit body sitting in it.
//
// Three stops, not one: a single blur of any radius decays too evenly and reads as
// a ring hovering off the limb. Stacking a tight bright one, a mid, and a very wide
// faint one approximates how light actually falls away, so there is no radius at
// which the eye can find where the glow "ends".
//
// Radii are FRACTIONS OF THE DIAMETER (via --moon-d) so the falloff stays in
// proportion as the viewport resizes the disc — a fixed px blur is a halo on a
// small screen and a hairline on a large one. Cool-tinted rather than white: this
// is light coming off a grey body into a black sky, and a neutral glow beside the
// faintly warm surface reads as a lens artifact instead.
const GLOW = [
	"0 0 calc(var(--moon-d) * 0.045) rgba(226,233,247,0.30)",
	"0 0 calc(var(--moon-d) * 0.13) rgba(198,212,240,0.15)",
	"0 0 calc(var(--moon-d) * 0.30) rgba(170,190,228,0.07)",
].join(", ");

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

// Both layers hang off the same box and anchor the same way — the disc's bottom
// edge on the bottom of whatever box it is given. Shared so the halo cannot drift
// out of register with the body casting it.
const ANCHOR =
	"pointer-events-none absolute left-1/2 top-full -translate-x-1/2 -translate-y-full rounded-full";

/**
 * The moon, sized to `diameter` and anchored so its BOTTOM edge meets the bottom
 * of the nearest positioned ancestor — which is the masthead band, and which
 * therefore decides how much of the disc shows.
 *
 * TWO ELEMENTS, and the split is the point: the DISC is clipped to that band (only
 * its lower cap should ever be on screen), but the HALO is not. Clipping both — the
 * obvious arrangement, one `overflow-hidden` around the lot — sliced the glow off
 * along the band's bottom edge, which is exactly where the disc runs tangent to it:
 * a hard horizontal line of light ending in mid-air, under a limb that nothing was
 * occluding. Light has to be free to fall past the body it comes off, so the glow
 * spills onto the page below and stops where it runs out.
 */
export default function Moon({ diameter }: { diameter: string }) {
	// Published as a custom property so the halo can be written as a fraction of
	// the disc instead of as a magic number.
	const sized = { "--moon-d": diameter, width: diameter, height: diameter } as React.CSSProperties;

	return (
		<>
			{/* The halo alone: no background, so it is nothing but its own box-shadow
			    — the lit disc is painted after this and covers the circle itself. */}
			<div aria-hidden className={ANCHOR} style={{ ...sized, boxShadow: GLOW }} />

			<div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
				<div className={ANCHOR} style={{ ...sized, background: SURFACE }}>
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
		</>
	);
}
