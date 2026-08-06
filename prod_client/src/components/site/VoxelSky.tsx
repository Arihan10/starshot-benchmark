
const EX = [0.7071, 0.4082];
const EZ = [-0.7071, 0.4082];
const EY = -0.8165;

const at = (x: number, y: number, z: number) =>
	[x * EX[0] + z * EZ[0], x * EX[1] + y * EY + z * EZ[1]]
		.map((n) => n.toFixed(3))
		.join(",");

const poly = (corners: [number, number, number][]) =>
	corners.map((c) => at(...c)).join(" ");

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

const PITCH = 22;
const DROP = Math.tan((PITCH * Math.PI) / 180);

const STARS: {
	side: "left" | "right";
	top: string;
	tail: number;
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

const BLOCKS: {
	edge: string;
	top: string;
	side: "left" | "right";
	size: number;
	opacity: number;
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

export function Voxel({
	size,
	opacity,
	drift,
	turn,
	seconds,
	turnSeconds,
	delay,
}: {
	size: number;
	opacity: string;
	drift: [number, number];
	turn: [number, number];
	seconds: number;
	turnSeconds: number;
	delay: number;
}) {
	return (
		<span
			className="block"
			style={
				{
					animation: `voxel-drift ${seconds}s ease-in-out ${delay}s infinite backwards`,
					"--drift-x": `${drift[0]}px`,
					"--drift-y": `${drift[1]}px`,
				} as React.CSSProperties
			}
		>
			<span
				className="block"
				style={
					{
						animation: `voxel-turn ${turnSeconds}s ease-in-out ${delay}s infinite alternate backwards`,
						"--turn-from": `${turn[0]}deg`,
						"--turn-to": `${turn[1]}deg`,
					} as React.CSSProperties
				}
			>
				<svg
					width={size}
					height={size * 1.155}
					viewBox={CUBE.box}
					aria-hidden
					style={{ opacity, transition: "opacity 700ms ease", display: "block" }}
				>
					{CUBE.faces.map((face) => (
						<polygon key={face.fill} points={face.points} fill={face.fill} />
					))}
				</svg>
			</span>
		</span>
	);
}

const BAR_REACH = "calc(var(--text-xs) * 5 + var(--spacing-sm) * 2)";

const BELOW_BAR = `linear-gradient(to bottom, transparent 0, transparent ${BAR_REACH}, #000 calc(${BAR_REACH} + var(--spacing-lg)))`;

export function VoxelDrift() {
	return (
		<div
			data-voxels
			aria-hidden
			className="pointer-events-none absolute inset-0 overflow-hidden"
			style={{ maskImage: BELOW_BAR, WebkitMaskImage: BELOW_BAR }}
		>
			{BLOCKS.map((block) => (
				<span
					key={`${block.side}-${block.top}`}
					data-side={block.side}
					className="absolute hidden xl:block"
					style={
						{
							top: block.top,
							[block.side]: block.edge,
							translate: `var(${
								block.side === "left" ? "--voxel-left-x" : "--voxel-right-x"
							}, 0px) var(${
								block.side === "left" ? "--voxel-left-y" : "--voxel-right-y"
							}, 0px)`,
							transition: "translate 900ms cubic-bezier(0.22,0.75,0.3,1)",
						} as React.CSSProperties
					}
				>
					<Voxel
						size={block.size}
						opacity={`calc(${block.opacity} * var(${
							block.side === "left" ? "--voxel-left-o" : "--voxel-right-o"
						}, 1))`}
						drift={[block.x, block.y]}
						turn={[block.from, block.to]}
						seconds={block.seconds}
						turnSeconds={block.turnSeconds}
						delay={block.delay}
					/>
				</span>
			))}
		</div>
	);
}

export default function VoxelSky() {
	return (
		<>
			<div
				data-sky
				aria-hidden
				className="pointer-events-none absolute inset-0 overflow-hidden"
			>
			{STARS.map((star) => {
				const away = star.side === "right" ? -1 : 1;
				const angle = star.side === "right" ? 180 - PITCH : PITCH;
				return (
					<span
						key={`${star.side}-${star.top}`}
						className="absolute"
						style={
							{
								top: star.top,
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
								background:
									"linear-gradient(90deg, rgb(var(--mark-rgb) / 0) 0%, rgb(var(--mark-rgb) / 0.35) 55%, rgb(var(--mark-rgb) / 1) 100%)",
							}}
						/>
					</span>
				);
			})}

			</div>
			<VoxelDrift />
		</>
	);
}
