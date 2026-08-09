
const SPIN_SECONDS = 260;

const SURFACE = "var(--moon-surface)";

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

const BRIGHT: { deg: number; dist: number; size: number; alpha: number }[] = [
	{ deg: 40, dist: 44, size: 26, alpha: 0.5 },
	{ deg: 127, dist: 47, size: 17, alpha: 0.42 },
	{ deg: 205, dist: 41, size: 30, alpha: 0.38 },
	{ deg: 288, dist: 45, size: 20, alpha: 0.46 },
];

const MARIA: { deg: number; dist: number; rx: number; ry: number; alpha: number }[] = [
	{ deg: 34, dist: 40, rx: 210, ry: 140, alpha: 0.055 },
	{ deg: 118, dist: 46, rx: 150, ry: 190, alpha: 0.045 },
	{ deg: 178, dist: 38, rx: 240, ry: 130, alpha: 0.05 },
	{ deg: 246, dist: 44, rx: 170, ry: 160, alpha: 0.04 },
	{ deg: 308, dist: 41, rx: 200, ry: 120, alpha: 0.05 },
];

function place(deg: number, dist: number): string {
	const rad = ((deg + 90) * Math.PI) / 180;
	return `${(50 + dist * Math.cos(rad)).toFixed(2)}% ${(50 + dist * Math.sin(rad)).toFixed(2)}%`;
}

const CRATER_LAYERS = [
	...BRIGHT.map(
		(b) =>
			`radial-gradient(circle ${b.size}px at ${place(b.deg, b.dist)}, rgb(var(--moon-rgb) / ${b.alpha}) 0%, rgb(var(--moon-rgb) / ${(b.alpha * 0.4).toFixed(3)}) 42%, transparent 100%)`,
	),
	...CRATERS.map(
		(c) =>
			`radial-gradient(circle ${c.size}px at ${place(c.deg, c.dist)}, rgb(var(--ground-rgb) / ${c.alpha}) 0%, rgb(var(--ground-rgb) / ${(c.alpha * 0.5).toFixed(3)}) 60%, transparent 100%)`,
	),
	...MARIA.map(
		(m) =>
			`radial-gradient(ellipse ${m.rx}px ${m.ry}px at ${place(m.deg, m.dist)}, rgb(var(--ground-rgb) / ${m.alpha}) 0%, rgb(var(--ground-rgb) / ${(m.alpha * 0.55).toFixed(3)}) 52%, transparent 100%)`,
	),
].join(", ");

export default function Moon({ diameter }: { diameter: string }) {
	return (
		<div
			aria-hidden
			className="pointer-events-none absolute bottom-0 left-1/2 -translate-x-1/2 rounded-full"
			style={{ width: diameter, height: diameter, background: SURFACE }}
		>
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
