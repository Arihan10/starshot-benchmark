// Breaking a losing panel into shards.
//
// Imperative and outside React on purpose. This is a few dozen short-lived nodes
// with independent trajectories that exist for about a second and a half; routing
// them through state would re-render the panel — and the live 3D canvas inside it
// — on every frame of an animation that React has nothing to say about. The Web
// Animations API runs the whole thing off the main thread instead.

const SWEEP = 380; // ms for the light to cross the panel
const COLS = 4;
const ROWS = 3;
const CLEANUP_MS = 1600;

/** Corner-to-corner sweep: a bright line, then the seam it leaves behind. */
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
			"linear-gradient(90deg, rgba(237,237,237,0) 0%, rgba(237,237,237,0.25) 62%, rgba(237,237,237,0.95) 92%, #fff 100%)",
		filter: "drop-shadow(0 0 10px rgba(237,237,237,0.85))",
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
		background: "rgba(237,237,237,0.75)",
		filter: "drop-shadow(0 0 6px rgba(237,237,237,0.6))",
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

/** A jittered triangulation of the panel — interior vertices wander, edges hold. */
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
			// Split each quad on a random diagonal so the break does not read as a grid.
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

/**
 * Throw one shard. The path is real projectile motion rather than an easing
 * curve — shards leave along the sweep's normal, arc up to an apex, then fall
 * past the bottom of the panel. `V` and `G` are solved from the apex height and
 * the distance to fall so the parabola passes through both.
 */
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
		const ease = 1 - (1 - u) ** 3; // the sideways kick spends itself early
		const y = -V * u + G * u * u + ny * kick * ease;
		const x = nx * kick * ease + drift * u;
		frames.push({
			transform: `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) rotate(${(spin * u).toFixed(1)}deg) scale(${(1.03 - 0.07 * u).toFixed(3)})`,
			opacity: u < 0.72 ? 1 : 1 - ((u - 0.72) / 0.28) * 0.94,
		});
	}
	shard.animate(frames, {
		duration: 780 + Math.random() * 320,
		// Shards leave as the light reaches them, so the break travels with the sweep.
		delay: ((cx + cy) / 200) * SWEEP,
		easing: "linear",
		fill: "forwards",
	});
}

/**
 * Shatter `fx` (an empty overlay covering the panel).
 *
 * `snapshot` is the panel's own pixels; each shard draws the whole image and
 * clips itself to its triangle, so the break looks like the scene coming apart
 * rather than tiles being dealt. Without one the shards fall back to flat grey —
 * the animation still reads, it just is not made of the thing it broke.
 *
 * Returns a canceller, so a component unmounting mid-flight leaves nothing behind.
 */
export function shatter(
	fx: HTMLElement,
	snapshot: HTMLCanvasElement | null,
	width: number,
	height: number,
): () => void {
	if (!width || !height) return () => {};
	fx.replaceChildren();
	addSweep(fx, width, height);

	for (const tri of triangulate()) {
		const cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3;
		const cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3;

		const shard = document.createElement(snapshot ? "canvas" : "div");
		if (snapshot && shard instanceof HTMLCanvasElement) {
			shard.width = snapshot.width;
			shard.height = snapshot.height;
			shard.getContext("2d")?.drawImage(snapshot, 0, 0);
		}
		Object.assign(shard.style, {
			position: "absolute",
			inset: "0",
			width: "100%",
			height: "100%",
			backgroundColor: snapshot ? "transparent" : "rgba(150,155,170,0.22)",
			clipPath: `polygon(${tri.map((p) => `${p[0].toFixed(2)}% ${p[1].toFixed(2)}%`).join(", ")})`,
			willChange: "transform, opacity",
		});
		fx.appendChild(shard);
		launch(shard, cx, cy, height);
	}

	const timer = setTimeout(() => fx.replaceChildren(), CLEANUP_MS);
	return () => {
		clearTimeout(timer);
		fx.replaceChildren();
	};
}

export const SHATTER_SWEEP_MS = SWEEP;
