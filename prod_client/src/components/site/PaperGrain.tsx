// ---------------------------------------------------------------------------
// THE SURFACE, AND WHY IT IS BACK.
//
// An engraved surface was here once and was taken off for reading as blotches on
// the paper rather than as relief on a body. That diagnosis was right and the
// cause was not the marks being too strong — it was that they were FLAT. A disc of
// shadow with no lit side is a stain; the thing that makes the same disc read as a
// crater is that one wall of it catches the light and the opposite wall does not.
//
// So the marks are modelled now, in pairs, and BECAUSE they are modelled they can
// be much fainter than the ones that were removed and still be legible as a
// surface. A crater is about 14 levels of shadow against 9 of highlight on 237
// paper — a dimple you have to be looking for, rather than a mark on the page.
//
// LIT FROM THE UPPER LEFT, which is where `--moon-surface` puts its highlight and
// so where the light on this site comes from. A bowl lit from that corner is dark
// on its upper-left inner wall — the one facing away — and bright on its
// lower-right one. Reverse the pair and every crater turns into a dome.
//
// DRAWN ON A BOX THE SIZE OF THE MASTHEAD, which is why the limb hands this a
// measured box rather than dropping it on the disc. The disc is tens of thousands
// of pixels across, so nothing can be placed against IT — a percentage down its
// face lands a mile above the window. The vertical coordinate is a percentage of
// whatever box this is given: `up` from its foot.
//
// THE TILE IS A WINDOW WIDE, THOUGH, AND CENTRED ON THE WINDOW — not on the piece
// of paper it is drawn on. Craters are marks of a fixed size on one body, so a
// tile that shrank to fit a narrower piece would enlarge every mark on it: the
// tongue is half the window and its craters came out twice the size of the ones in
// the bar directly above, which is two different surfaces rather than one. Pinned
// to the viewport, a narrow piece shows the MIDDLE of the same field the full-width
// pieces show, and the marks line up across the seam.
//
// THE TEXTURE DRIFTS LEFT TO RIGHT while the lighting remains fixed. Two identical
// tiles trade places in a marquee, so the box stays covered through every frame
// and the loop is invisible. The drift is on an inner element so that the centring
// above is not something a transform has to keep re-establishing.
// ---------------------------------------------------------------------------

// The offset of each wall from the crater's centre, as a fraction of its radius.
const TILT = 0.2;

const HIGHLIGHT = 0.48;

const CRATERS: { x: number; up: number; r: number; a: number }[] = [
	{ x: 6, up: 70, r: 24, a: 0.059 },
	{ x: 15, up: 49, r: 14, a: 0.07 },
	{ x: 26, up: 79, r: 32, a: 0.048 },
	{ x: 37, up: 54, r: 18, a: 0.064 },
	{ x: 57, up: 57, r: 27, a: 0.05 },
	{ x: 68, up: 80, r: 16, a: 0.067 },
	{ x: 79, up: 52, r: 22, a: 0.056 },
	{ x: 96, up: 55, r: 19, a: 0.06 },
];

// The broad low ground between the craters. Far too faint to find on their own —
// they are what stops the paper reading as evenly white between the marks.
const MARIA: { x: number; up: number; rx: number; ry: number; a: number }[] = [
	{ x: 20, up: 64, rx: 250, ry: 76, a: 0.027 },
	{ x: 62, up: 77, rx: 310, ry: 64, a: 0.022 },
	{ x: 90, up: 61, rx: 210, ry: 72, a: 0.025 },
];

// The walls are offset in PIXELS off a percentage. A crater's two faces are a
// fixed distance apart — that is its radius, which does not change with the bar —
// so the offset cannot be a share of a box that does.
const wall = (c: (typeof CRATERS)[number]) =>
	`radial-gradient(circle ${c.r}px at calc(${c.x}% - ${(c.r * TILT).toFixed(1)}px) calc(${100 - c.up}% - ${(c.r * TILT).toFixed(1)}px), rgba(9,11,16,${c.a}) 0%, rgba(9,11,16,${(c.a * 0.45).toFixed(3)}) 58%, transparent 100%)`;

const lit = (c: (typeof CRATERS)[number]) =>
	`radial-gradient(circle ${Math.round(c.r * 0.86)}px at calc(${c.x}% + ${(c.r * TILT).toFixed(1)}px) calc(${100 - c.up}% + ${(c.r * TILT).toFixed(1)}px), rgba(255,255,255,${HIGHLIGHT}) 0%, transparent 100%)`;

const SURFACE = [
	...CRATERS.flatMap((c) => [lit(c), wall(c)]),
	...MARIA.map(
		(m) =>
			`radial-gradient(ellipse ${m.rx}px ${m.ry}px at ${m.x}% ${100 - m.up}%, rgba(11,14,20,${m.a}) 0%, rgba(11,14,20,${(m.a * 0.5).toFixed(3)}) 52%, transparent 100%)`,
	),
].join(", ");

export default function PaperGrain() {
	return (
		<div aria-hidden className="absolute inset-0 overflow-hidden">
			<div className="absolute inset-y-0 left-1/2 w-[200vw] -translate-x-1/2">
				<div className="absolute inset-0 flex motion-safe:animate-[moon-texture-marquee_260s_linear_infinite]">
					{[0, 1].map((tile) => (
						<div
							key={tile}
							className="h-full w-1/2 flex-none"
							style={{ backgroundImage: SURFACE }}
						/>
					))}
				</div>
			</div>
		</div>
	);
}
