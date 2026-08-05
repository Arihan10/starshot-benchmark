/**
 * The site's background: stars crossing behind the page, voxels adrift either side.
 *
 * It began on the leaderboard as `BoardSky` and lives under `site/` now because it
 * is the aesthetic rather than that page's furniture — About and the FAQ carry it
 * too. It takes no props deliberately: a sky that could be configured per page is
 * a sky that will drift apart between them.
 *
 * IT CAN BE PUSHED, THOUGH, and that is a different thing from configuring it. A
 * page may set `--voxel-left-o` / `--voxel-right-o` (how strongly each side shows)
 * and `--voxel-left-x` / `--voxel-right-x` (how far that side is pushed toward its
 * own edge). Both default to leaving the field alone, so the leaderboard and the
 * FAQ say nothing and get the sky as composed. About drives them off the scroll,
 * so the blocks are always in the half the copy is not in — see AboutStage.
 *
 * A SERVER COMPONENT, and no canvas. The podium's city needs WebGL because it is a
 * real projection of real geometry that has to be lit and animated per frame; this
 * is ornament, and a second WebGL context on the same page — running whenever the
 * reader is reading a table — would cost more than everything on this section put
 * together. A streak is a gradient in a box, and a block is nine points of SVG.
 *
 * NOTHING HERE IS INTERACTIVE. The whole layer takes no pointer events, so it
 * cannot be clicked, grabbed or tabbed to, and it carries no meaning that a reader
 * who never sees it would miss.
 *
 * LAID OUT FROM A TABLE OF CONSTANTS rather than generated. `Math.random()` in a
 * component that renders on the server and again in the browser is a hydration
 * mismatch waiting to happen, and a hand-placed sky can be composed — these are
 * spread so that no two cross at once and none of them sits behind a column.
 */

// --- the cube -------------------------------------------------------------
//
// THE SAME PROJECTION THE PODIUM IS BUILT IN, so a block out here is the same solid
// as a block in the city rather than a drawing of a box: world X lands at
// (cos45, sin45·sin35.26) on screen, world Z at its mirror, and world Y straight up
// at cos35.26.
const EX = [0.7071, 0.4082];
const EZ = [-0.7071, 0.4082];
const EY = -0.8165;

const at = (x: number, y: number, z: number) =>
	[x * EX[0] + z * EZ[0], x * EX[1] + y * EY + z * EZ[1]]
		.map((n) => n.toFixed(3))
		.join(",");

const poly = (corners: [number, number, number][]) =>
	corners.map((c) => at(...c)).join(" ");

/**
 * ONE CUBE, THREE FACES — the top catching the key and two sides falling away from
 * it at different angles, which are the podium's own values.
 *
 * The hexagon is fixed, so the box around it is too: a unit cube projects to
 * ±cos45 across and ±cos35.26 up, and nothing about that changes per block. Only
 * the size, the angle and the opacity do.
 */
const CUBE = {
	box: "-0.72 -0.83 1.44 1.66",
	faces: [
		{
			fill: "#e8e8e8",
			points: poly([
				[0, 1, 0],
				[1, 1, 0],
				[1, 1, 1],
				[0, 1, 1],
			]),
		},
		{
			fill: "#b9b9b9",
			points: poly([
				[0, 0, 1],
				[0, 1, 1],
				[1, 1, 1],
				[1, 0, 1],
			]),
		},
		{
			fill: "#8c8c8c",
			points: poly([
				[1, 0, 0],
				[1, 1, 0],
				[1, 1, 1],
				[1, 0, 1],
			]),
		},
	],
};

// --- the stars ------------------------------------------------------------
//
// THEY COME IN OFF THE EDGES, always, and that is the difference between a meteor
// and a spark. A streak that starts somewhere in the middle of the picture has to
// begin by EXISTING, which nothing in a sky does; entering from beyond the frame,
// it was always there and the frame simply caught it.
//
// So each is anchored past its own side of the window by more than its own length —
// tail included — and the crossing carries it in. The angle is shallow, because a
// steep one reads as falling rather than as travelling.
const PITCH = 22;
const DROP = Math.tan((PITCH * Math.PI) / 180);

const STARS: {
	/** Which edge it enters from; it travels away from that edge and downward. */
	side: "left" | "right";
	top: string;
	tail: number;
	/** How far it crosses, as a share of the window's width. */
	reach: number;
	seconds: number;
	delay: number;
}[] = [
	{ side: "right", top: "4%", tail: 150, reach: 88, seconds: 7, delay: 0 },
	{ side: "left", top: "16%", tail: 108, reach: 64, seconds: 9, delay: 2.2 },
	{ side: "right", top: "-2%", tail: 190, reach: 104, seconds: 11, delay: 4.6 },
	{ side: "left", top: "38%", tail: 86, reach: 52, seconds: 8, delay: 6.9 },
	{ side: "right", top: "26%", tail: 128, reach: 76, seconds: 12, delay: 9.4 },
	{ side: "left", top: "-6%", tail: 164, reach: 92, seconds: 10, delay: 12.1 },
];

// --- the blocks -----------------------------------------------------------
//
// CUBES ONLY, AT THEIR OWN ANGLES. Held to the projection's own alignment they read
// as a diagram — a row of tidy isometric cubes, all agreeing with each other and
// with the city on the section above. Turned, the same cube reads as an object that
// happens to be floating there, and the variety comes from the angle rather than
// from inventing shapes that the city itself does not contain.
const BLOCKS: {
	edge: string;
	top: string;
	side: "left" | "right";
	size: number;
	opacity: number;
	/** Where it drifts to, and the two angles it turns between. */
	x: number;
	y: number;
	from: number;
	to: number;
	seconds: number;
	turnSeconds: number;
	delay: number;
}[] = [
	{ edge: "2%", top: "12%", side: "left", size: 58, opacity: 0.5, x: 22, y: -62, from: -40, to: -12, seconds: 23, turnSeconds: 19, delay: 0 },
	{ edge: "6.5%", top: "34%", side: "left", size: 30, opacity: 0.32, x: -26, y: -43, from: 15, to: 48, seconds: 31, turnSeconds: 26, delay: 3 },
	{ edge: "1%", top: "52%", side: "left", size: 44, opacity: 0.44, x: 31, y: 50, from: 123, to: 90, seconds: 27, turnSeconds: 22, delay: 7 },
	{ edge: "5.5%", top: "72%", side: "left", size: 52, opacity: 0.4, x: -17, y: -36, from: -65, to: -94, seconds: 35, turnSeconds: 30, delay: 11 },
	{ edge: "2.5%", top: "90%", side: "left", size: 34, opacity: 0.26, x: 19, y: -29, from: 14, to: -15, seconds: 29, turnSeconds: 24, delay: 5 },
	{ edge: "3%", top: "8%", side: "right", size: 42, opacity: 0.38, x: -22, y: 53, from: 160, to: 131, seconds: 29, turnSeconds: 25, delay: 2 },
	{ edge: "6.5%", top: "30%", side: "right", size: 60, opacity: 0.48, x: 24, y: -58, from: -18, to: 12, seconds: 25, turnSeconds: 21, delay: 6 },
	{ edge: "1.5%", top: "50%", side: "right", size: 28, opacity: 0.3, x: -19, y: -38, from: 57, to: 85, seconds: 33, turnSeconds: 28, delay: 9 },
	{ edge: "5%", top: "70%", side: "right", size: 50, opacity: 0.42, x: 17, y: 43, from: -122, to: -151, seconds: 21, turnSeconds: 18, delay: 13 },
	{ edge: "2%", top: "88%", side: "right", size: 36, opacity: 0.28, x: -14, y: -48, from: 47, to: 18, seconds: 37, turnSeconds: 31, delay: 4 },
];

/**
 * THE VOXELS ON THEIR OWN.
 *
 * Separated because About already owns a star layer of its own — a different
 * placement on a different clock — and dropping the full sky on top of it would run
 * two sets of streaks at once. This is not a configuration flag: the two exports are
 * two different things a page can want, and neither takes props, so no page can
 * quietly tune the sky away from the others.
 */
export function VoxelDrift() {
	return (
		<div
			data-voxels
			aria-hidden
			className="pointer-events-none absolute inset-0 overflow-hidden"
		>
			{BLOCKS.map((block) => (
				// TWO ELEMENTS AGAIN, for the same reason the stars need two: the drift
				// owns `transform`, so the turn has to be the standalone `rotate`
				// property on a child — as transforms, one of the two would simply win.
				<span
					key={`${block.side}-${block.top}`}
					data-side={block.side}
					className="absolute hidden xl:block"
					style={
						{
							top: block.top,
							[block.side]: block.edge,
							animation: `voxel-drift ${block.seconds}s ease-in-out ${block.delay}s infinite backwards`,
							"--drift-x": `${block.x}px`,
							"--drift-y": `${block.y}px`,
							// `translate`, NOT a transform: the drift above owns `transform`
							// for its whole cycle, and a second transform on the same element
							// would simply replace it. The standalone property composes.
							translate: `var(${
								block.side === "left" ? "--voxel-left-x" : "--voxel-right-x"
							}, 0px) var(${
								block.side === "left" ? "--voxel-left-y" : "--voxel-right-y"
							}, 0px)`,
							transition: "translate 900ms cubic-bezier(0.22,0.75,0.3,1)",
						} as React.CSSProperties
					}
				>
					<span
						className="block"
						style={
							{
								animation: `voxel-turn ${block.turnSeconds}s ease-in-out ${block.delay}s infinite alternate backwards`,
								"--turn-from": `${block.from}deg`,
								"--turn-to": `${block.to}deg`,
							} as React.CSSProperties
						}
					>
						<svg
							width={block.size}
							height={block.size * 1.155}
							viewBox={CUBE.box}
							aria-hidden
							style={{
								// The block's own weight, times whatever the page has left this
								// side. Pages that set nothing get 1 and the field is unchanged.
								opacity: `calc(${block.opacity} * var(${
									block.side === "left" ? "--voxel-left-o" : "--voxel-right-o"
								}, 1))`,
								transition: "opacity 700ms ease",
								display: "block",
							}}
						>
							{CUBE.faces.map((face) => (
								<polygon key={face.fill} points={face.points} fill={face.fill} />
							))}
						</svg>
					</span>
				</span>
			))}
		</div>
	);
}

/**
 * The full sky: streaks crossing behind the page, voxels adrift either side.
 * What the leaderboard and the FAQ use.
 */
export default function VoxelSky() {
	return (
		<>
			<div
				data-sky
				aria-hidden
				className="pointer-events-none absolute inset-0 overflow-hidden"
			>
			{STARS.map((star) => {
				// Away from its own edge, and down. The streak is drawn as a bar running
				// right from its anchor, so turning it by the travel angle points it —
				// and its bright end, which is the far end — the way it is going.
				const away = star.side === "right" ? -1 : 1;
				const angle = star.side === "right" ? 180 - PITCH : PITCH;
				return (
					<span
						key={`${star.side}-${star.top}`}
						className="absolute"
						style={
							{
								top: star.top,
								// PAST THE EDGE BY ITS WHOLE LENGTH, so even the head starts
								// outside the frame and the streak arrives already moving.
								[star.side]: `${-(star.tail + 24)}px`,
								transformOrigin: "0 50%",
								animation: `shooting-star ${star.seconds}s linear ${star.delay}s infinite backwards`,
								"--shoot-x": `${away * star.reach}vw`,
								"--shoot-y": `${(star.reach * DROP).toFixed(1)}vw`,
							} as React.CSSProperties
						}
					>
						<span
							className="block"
							style={{
								width: star.tail,
								height: 1.6,
								transformOrigin: "0 50%",
								transform: `rotate(${angle}deg)`,
								// The head is a point of light and the tail is what is left of
								// it: opaque at the leading tip, gone well before the anchor.
								background:
									"linear-gradient(90deg, rgb(var(--mark-rgb) / 0) 0%, rgb(var(--mark-rgb) / 0.35) 55%, rgb(var(--mark-rgb) / 1) 100%)",
							}}
						/>
					</span>
				);
			})}

			{/* THE BLOCKS SIT IN THE MARGINS, which is the only place on this section
			    there is room: the standings hold the middle at a fixed measure, so
			    anything decorative belongs outside it or not at all. Below a wide
			    window those margins close up, and the blocks go rather than crowd the
			    table. */}
			</div>
			<VoxelDrift />
		</>
	);
}

