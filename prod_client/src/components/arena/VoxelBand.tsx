import { Voxel } from "@/components/site/VoxelSky";

const INSET = "var(--spacing-xl)";

const LIMIT = "(50% - var(--seam-notch, 0px) - var(--seam-gap, 0px))";

const along = (t: number, size: number) =>
	`calc(${INSET} + ${t} * (${LIMIT} - ${INSET} - ${size}px))`;

const centred = (size: number, lift: number) =>
	`calc(50% - ${((size * 1.155) / 2).toFixed(1)}px + ${lift}px)`;

const SIDES = ["left", "right"] as const;

const BAND: {
	t: number;
	size: number;
	opacity: number;
	lift: number;
	drift: [number, number];
	turn: [number, number];
	seconds: number;
	turnSeconds: number;
	delay: number;
}[] = [
	{ t: 0.1, size: 30, opacity: 0.38, lift: 6, drift: [24, -7], turn: [-38, -12], seconds: 38, turnSeconds: 29, delay: 0 },
	{ t: 0.46, size: 20, opacity: 0.26, lift: -8, drift: [-19, 8], turn: [24, 52], seconds: 46, turnSeconds: 35, delay: 4.2 },
	{ t: 0.84, size: 14, opacity: 0.18, lift: 4, drift: [21, -6], turn: [118, 92], seconds: 33, turnSeconds: 27, delay: 8.1 },
];

export default function VoxelBand({ shown = true }: { shown?: boolean }) {
	return (
		<div
			aria-hidden
			className={`pointer-events-none absolute inset-x-0 bottom-0 h-(--seam-drop) overflow-hidden transition-opacity duration-700 ease-out ${
				shown ? "opacity-100" : "opacity-0"
			}`}
		>
			{SIDES.map((side) =>
				BAND.map((block) => (
					<span
						key={`${side}-${block.t}`}
						className="absolute hidden xl:block"
						style={{
							[side]: along(block.t, block.size),
							bottom: centred(block.size, block.lift),
						}}
					>
						<Voxel
							size={block.size}
							opacity={String(block.opacity)}
							drift={
								side === "left"
									? block.drift
									: [-block.drift[0], block.drift[1]]
							}
							turn={
								side === "left"
									? block.turn
									: [-block.turn[0], -block.turn[1]]
							}
							seconds={block.seconds}
							turnSeconds={block.turnSeconds}
							delay={side === "left" ? block.delay : block.delay + 3.7}
						/>
					</span>
				)),
			)}
		</div>
	);
}
