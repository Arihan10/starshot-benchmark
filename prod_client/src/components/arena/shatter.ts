
const SWEEP = 380;
const COLS = 4;
const ROWS = 3;
const CLEANUP_MS = 1600;

function addSweep(fx: HTMLElement, w: number, h: number) {
	const angle = (Math.atan2(h, w) * 180) / Math.PI;
	const diag = Math.hypot(w, h);
	const rotated = (x: string) => `rotate(${angle}deg) translateX(${x})`;

	const star = document.createElement("div");
	Object.assign(star.style, {
		position: "absolute",
		left: "0",
		top: "0",
		width: `${diag}px`,
		height: "2px",
		transformOrigin: "0 50%",
		transform: rotated(`-${diag}px`),
		background:
			"linear-gradient(90deg, rgb(var(--mark-rgb) / 0) 0%, rgb(var(--mark-rgb) / 0.25) 62%, rgb(var(--mark-rgb) / 0.95) 92%, rgb(var(--mark-rgb)) 100%)",
		filter: "drop-shadow(0 0 10px rgb(var(--mark-rgb) / 0.85))",
		opacity: "0",
	});
	fx.appendChild(star);
	star.animate(
		[
			{ transform: rotated(`-${diag}px`), opacity: 0 },
			{ opacity: 1, offset: 0.12 },
			{ opacity: 1, offset: 0.86 },
			{ transform: rotated("0px"), opacity: 0 },
		],
		{ duration: SWEEP, easing: "linear", fill: "forwards" },
	);

	const seam = document.createElement("div");
	Object.assign(seam.style, {
		position: "absolute",
		left: "0",
		top: "0",
		width: `${diag}px`,
		height: "1px",
		transformOrigin: "0 50%",
		transform: `rotate(${angle}deg) scaleX(0)`,
		background: "rgb(var(--mark-rgb) / 0.75)",
		filter: "drop-shadow(0 0 6px rgb(var(--mark-rgb) / 0.6))",
	});
	fx.appendChild(seam);
	seam.animate(
		[
			{ transform: `rotate(${angle}deg) scaleX(0)` },
			{ transform: `rotate(${angle}deg) scaleX(1)` },
		],
		{ duration: SWEEP, easing: "linear", fill: "forwards" },
	);
	seam.animate([{ opacity: 1 }, { opacity: 0 }], {
		duration: 240,
		delay: SWEEP - 60,
		easing: "ease-out",
		fill: "forwards",
	});
}

function triangulate(): [number, number][][] {
	const jitter = () => (Math.random() - 0.5) * 9;
	const pts: [number, number][][] = [];
	for (let r = 0; r <= ROWS; r++) {
		pts[r] = [];
		for (let c = 0; c <= COLS; c++) {
			const onEdge = r === 0 || c === 0 || r === ROWS || c === COLS;
			pts[r][c] = [
				(c / COLS) * 100 + (onEdge ? 0 : jitter()),
				(r / ROWS) * 100 + (onEdge ? 0 : jitter()),
			];
		}
	}
	const tris: [number, number][][] = [];
	for (let r = 0; r < ROWS; r++) {
		for (let c = 0; c < COLS; c++) {
			const a = pts[r][c];
			const b = pts[r][c + 1];
			const d = pts[r + 1][c + 1];
			const e = pts[r + 1][c];
			const pair =
				Math.random() < 0.5
					? [
							[a, b, d],
							[a, d, e],
						]
					: [
							[a, b, e],
							[b, d, e],
						];
			for (const t of pair) tris.push(t);
		}
	}
	return tris;
}

function launch(shard: HTMLElement, cx: number, cy: number, h: number) {
	const side = cx > cy ? 1 : -1;
	const nx = side * 0.7071;
	const ny = -side * 0.7071;
	const kick = 16 + Math.random() * 30;
	const apex = 26 + Math.random() * 54;
	const yEnd = h * 1.35 + 170;
	const drift = side * (26 + Math.random() * 96);
	const spin = side * (26 + Math.random() * 128) * (Math.random() < 0.22 ? -0.5 : 1);
	const V = 2 * apex + 2 * Math.sqrt(apex * apex + apex * yEnd);
	const G = yEnd + V;

	const frames: Keyframe[] = [];
	for (let i = 0; i <= 22; i++) {
		const u = i / 22;
		const ease = 1 - (1 - u) ** 3;
		const y = -V * u + G * u * u + ny * kick * ease;
		const x = nx * kick * ease + drift * u;
		frames.push({
			transform: `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) rotate(${(spin * u).toFixed(1)}deg) scale(${(1.03 - 0.07 * u).toFixed(3)})`,
			opacity: u < 0.72 ? 1 : 1 - ((u - 0.72) / 0.28) * 0.94,
		});
	}
	shard.animate(frames, {
		duration: 780 + Math.random() * 320,
		delay: ((cx + cy) / 200) * SWEEP,
		easing: "linear",
		fill: "forwards",
	});
}

export function shatter(
	fx: HTMLElement,
	snapshot: HTMLCanvasElement | null,
	width: number,
	height: number,
): () => void {
	if (!width || !height) return () => {};
	fx.replaceChildren();
	addSweep(fx, width, height);

	let cancelled = false;
	let url: string | null = null;
	let timer: ReturnType<typeof setTimeout> | null = null;

	const build = (src: string | null) => {
		if (cancelled) return;
		for (const tri of triangulate()) {
			const cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3;
			const cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3;

			const xs = [tri[0][0], tri[1][0], tri[2][0]];
			const ys = [tri[0][1], tri[1][1], tri[2][1]];
			const x0 = Math.min(...xs);
			const y0 = Math.min(...ys);
			const bw = Math.max(...xs) - x0;
			const bh = Math.max(...ys) - y0;

			const shard = document.createElement("div");
			Object.assign(shard.style, {
				position: "absolute",
				left: `${x0.toFixed(3)}%`,
				top: `${y0.toFixed(3)}%`,
				width: `${bw.toFixed(3)}%`,
				height: `${bh.toFixed(3)}%`,
				backgroundImage: src ? `url("${src}")` : "none",
				backgroundSize: `${width}px ${height}px`,
				backgroundPosition: `${(-(x0 / 100) * width).toFixed(2)}px ${(-(y0 / 100) * height).toFixed(2)}px`,
				backgroundColor: src ? "transparent" : "rgba(150,155,170,0.22)",
				clipPath: `polygon(${tri
					.map(
						(p) =>
							`${(((p[0] - x0) / bw) * 100).toFixed(2)}% ${(((p[1] - y0) / bh) * 100).toFixed(2)}%`,
					)
					.join(", ")})`,
				willChange: "transform, opacity",
			});
			fx.appendChild(shard);
			launch(shard, cx, cy, height);
		}
		timer = setTimeout(() => fx.replaceChildren(), CLEANUP_MS);
	};

	if (snapshot) {
		snapshot.toBlob(
			(blob) => {
				if (cancelled) return;
				if (!blob) return build(null);
				url = URL.createObjectURL(blob);
				build(url);
			},
			"image/webp",
			0.9,
		);
	} else {
		build(null);
	}

	return () => {
		cancelled = true;
		if (timer) clearTimeout(timer);
		if (url) URL.revokeObjectURL(url);
		fx.replaceChildren();
	};
}

export const SHATTER_SWEEP_MS = SWEEP;
