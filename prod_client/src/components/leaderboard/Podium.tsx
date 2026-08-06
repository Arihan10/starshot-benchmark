"use client";

import { Html } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
	useEffect,
	useMemo,
	useRef,
	useState,
	useSyncExternalStore,
} from "react";
import * as THREE from "three";
import { inkColor } from "@/lib/ink";
import BrandMark, { brandTone } from "./BrandMark";
import { type Loose, MAX_THROW, step } from "./loose";
import { paintSweep, PLINTH_ATTRIBUTE, sweepStops } from "./sweep";
import {
	BLOCKS,
	type BlockSpec,
	CUBE,
	DECK,
	DEPTH,
	FOOTINGS,
	GROUP_Y,
	LABELS,
	PILLARS,
	PLATE,
	PLATE_H,
	RISE,
	VIEW,
} from "./podiumLayout";

const INTRO = 3.2;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const easeOutCubic = (u: number) => 1 - (1 - u) ** 3;
const easeOutBack = (u: number) => {
	const c = 1.34;
	return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2;
};

// Re-read on theme changes; the palette is editable at runtime and a colour
// captured once at mount would keep the city on whatever ink it booted with.
function useInk(): string {
	return useSyncExternalStore(
		(onChange) => {
			const mo = new MutationObserver(onChange);
			mo.observe(document.documentElement, {
				attributes: true,
				attributeFilter: ["style", "class"],
			});
			return () => mo.disconnect();
		},
		inkColor,
		inkColor,
	);
}

function useReducedMotion() {
	const [reduced, setReduced] = useState(
		() =>
			typeof window !== "undefined" &&
			window.matchMedia("(prefers-reduced-motion: reduce)").matches,
	);
	useEffect(() => {
		const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
		const sync = () => setReduced(mq.matches);
		mq.addEventListener("change", sync);
		return () => mq.removeEventListener("change", sync);
	}, []);
	return reduced;
}

const MARGIN = 1.08;

const SKIRT = 0.2;

const ZOOM = 50;

const LEFT_FACE = "matrix(0.7071, 0.4082, 0, 0.8165, 0, 0)";
const TOP_FACE = "matrix(0.7071, -0.4082, 0.7071, 0.4082, 0, 0)";

const ETCH = "rgb(0 0 0 / 0.42)";

const ETCH_HEIGHT = 0.95;

const SWELL = 1.075;
const LIFT = 1.03;

const scratch = new THREE.Object3D();
const spot = new THREE.Vector3();
const tumble = new THREE.Euler();

function sample(s: BlockSpec, t: number): number {
	const u = clamp01((t - s.start) / (s.end - s.start));
	if (u <= 0) return 0;
	const e = easeOutCubic(u);
	spot.set(
		s.from[0] + (s.rest[0] - s.from[0]) * e,
		s.from[1] + (s.rest[1] - s.from[1]) * e,
		s.from[2] + (s.rest[2] - s.from[2]) * e,
	);
	const left = 1 - e;
	tumble.set(s.spin[0] * left, s.spin[1] * left, s.spin[2] * left);
	return easeOutCubic(clamp01(u / 0.5));
}

function fly(
	mesh: THREE.InstancedMesh | null,
	specs: BlockSpec[],
	t: number,
): void {
	if (!mesh) return;
	for (let i = 0; i < specs.length; i++) {
		const scale = sample(specs[i], t);
		if (scale <= 0) {
			scratch.position.set(0, 0, 0);
			scratch.rotation.set(0, 0, 0);
			scratch.scale.setScalar(0);
		} else {
			scratch.position.copy(spot);
			scratch.rotation.copy(tumble);
			scratch.scale.setScalar(scale);
		}
		scratch.updateMatrix();
		mesh.setMatrixAt(i, scratch.matrix);
	}
	mesh.instanceMatrix.needsUpdate = true;
}

function hide(mesh: THREE.InstancedMesh | null, gone: Set<number>): void {
	if (!mesh || gone.size === 0) return;
	scratch.position.set(0, 0, 0);
	scratch.rotation.set(0, 0, 0);
	scratch.scale.setScalar(0);
	scratch.updateMatrix();
	for (const i of gone) mesh.setMatrixAt(i, scratch.matrix);
	mesh.instanceMatrix.needsUpdate = true;
}

function paint(mesh: THREE.InstancedMesh | null, list: Loose[]): void {
	if (!mesh) return;
	for (const b of list) {
		scratch.position.copy(b.pos);
		scratch.quaternion.copy(b.quat);
		scratch.scale.setScalar(1);
		scratch.updateMatrix();
		mesh.setMatrixAt(b.i, scratch.matrix);
	}
	mesh.instanceMatrix.needsUpdate = true;
}

type Row = { rank: number; name: string; lab: string; elo: number };

function Stage({
	rows,
	reduced,
	band,
}: {
	rows: Row[];
	reduced: boolean;
	band: number;
}) {
	const ink = useInk();
	const viewport = useThree((s) => s.viewport);
	const camera = useThree((s) => s.camera);
	const gl = useThree((s) => s.gl);

	const canvas = useRef<HTMLCanvasElement | null>(null);
	useEffect(() => {
		canvas.current = gl.domElement;
	}, [gl]);

	const caster = useMemo(() => new THREE.Raycaster(), []);

	// The canvas runs to the foot of the page so thrown blocks stay drawn on
	// their way out, but the model belongs in the band at the top of it.
	const size = useThree((s) => s.size);
	const share = band > 0 && size.height > 0 ? band / size.height : 1;
	const bandWorld = viewport.height * share;

	// The city sits in the TOP of the band and keeps a skirt clear beneath it,
	// so its deck never runs into the standings below.
	const skirt = bandWorld * SKIRT;
	const seat = bandWorld - skirt;
	const lift = (viewport.height - bandWorld) / 2 + skirt / 2;
	const fit = Math.min(
		viewport.width / (VIEW.w * MARGIN),
		seat / (VIEW.h * MARGIN),
	);
	const floor = -viewport.height / 2 - lift;

	const plates = useRef<THREE.InstancedMesh>(null);
	const blocks = useRef<THREE.InstancedMesh>(null);
	const pillars = useRef<(THREE.Mesh | null)[]>([]);
	const over = useRef(PILLARS.map(() => false));
	const swell = useRef(PILLARS.map(() => 0));
	const labels = useRef<(HTMLDivElement | null)[]>([]);
	const stones = useRef<(THREE.Group | null)[]>([]);
	const model = useRef<THREE.Group>(null);

	const loose = useRef<Loose[]>([]);
	const dropped = useRef(new Set<number>());
	const dragging = useRef<Loose | null>(null);
	const dragPlane = useRef(new THREE.Plane());
	const dragLast = useRef(new THREE.Vector3());
	const dragTime = useRef(0);

	const drawn = useRef(-1);
	const settled = useRef(false);

	const intro = useRef(reduced ? 1 : 0);

	const plate = useMemo(
		() => new THREE.BoxGeometry(PLATE, PLATE_H, PLATE),
		[],
	);
	const cube = useMemo(() => new THREE.BoxGeometry(CUBE, CUBE, CUBE), []);
	const post = useMemo(() => {
		const g = new THREE.BoxGeometry(1, 1, 1);
		g.translate(0, 0.5, 0);
		return g;
	}, []);
	const skin = useMemo(
		() =>
			new THREE.MeshStandardMaterial({
				color: ink,
				roughness: 0.74,
				metalness: 0,
			}),
		[ink],
	);

	const stone = useMemo(() => {
		const m = new THREE.MeshStandardMaterial({
			color: ink,
			roughness: 0.74,
			metalness: 0,
		});
		return { material: m, ...paintSweep(m) };
	}, [ink]);

	useEffect(() => {
		const flags = new Float32Array(BLOCKS.length);
		for (const foot of FOOTINGS) flags.fill(1, foot.from, foot.from + foot.count);
		cube.setAttribute(
			PLINTH_ATTRIBUTE,
			new THREE.InstancedBufferAttribute(flags, 1),
		);
	}, [cube]);

	useEffect(() => {
		const read = sweepStops();
		for (let i = 0; i < stone.stops.value.length; i++) {
			stone.stops.value[i].copy(read[i]);
		}
	}, [stone]);

	const tones = useMemo(
		() =>
			PILLARS.map((p) => {
				const row = rows.find((r) => r.rank === p.rank);
				return new THREE.MeshStandardMaterial({
					color: row ? brandTone(row.lab) : "#ededed",
					roughness: 0.68,
					metalness: 0,
				});
			}),
		[rows],
	);

	useEffect(
		() => () => {
			plate.dispose();
			cube.dispose();
			post.dispose();
			skin.dispose();
			stone.material.dispose();
			for (const m of tones) m.dispose();
		},
		[plate, cube, post, skin, stone, tones],
	);

	useEffect(() => {
		const mesh = blocks.current;
		if (!mesh) return;
		let far = 0;
		for (const b of BLOCKS) {
			far = Math.max(
				far,
				Math.hypot(b.rest[0], b.rest[1], b.rest[2]),
				Math.hypot(b.from[0], b.from[1], b.from[2]),
			);
		}
		mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), far + CUBE);
	}, []);

	useEffect(() => {
		const el = gl.domElement;
		const ndc = new THREE.Vector2();
		const at = new THREE.Vector3();
		const flick = new THREE.Vector3();

		const move = (ev: PointerEvent) => {
			const b = dragging.current;
			const group = model.current;
			if (!b || !group) return;
			const rect = el.getBoundingClientRect();
			ndc.set(
				((ev.clientX - rect.left) / rect.width) * 2 - 1,
				-((ev.clientY - rect.top) / rect.height) * 2 + 1,
			);
			caster.setFromCamera(ndc, camera);
			if (!caster.ray.intersectPlane(dragPlane.current, at)) return;
			group.worldToLocal(at);

			const gap = Math.max(4, ev.timeStamp - dragTime.current) / 1000;
			dragTime.current = ev.timeStamp;
			flick.subVectors(at, dragLast.current).divideScalar(gap);
			dragLast.current.copy(at);
			b.vel.lerp(flick, 0.65);

			b.pos.copy(at);
			b.asleep = false;
		};

		const release = () => {
			const b = dragging.current;
			dragging.current = null;
			el.style.cursor = "";
			if (!b) return;
			b.held = false;
			b.vel.clampLength(0, MAX_THROW);
			const speed = b.vel.length();
			b.spin
				.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)
				.multiplyScalar(speed * 0.5 + 1);
		};

		el.addEventListener("pointermove", move);
		el.addEventListener("pointerup", release);
		el.addEventListener("pointercancel", release);
		return () => {
			el.removeEventListener("pointermove", move);
			el.removeEventListener("pointerup", release);
			el.removeEventListener("pointercancel", release);
		};
	}, [gl, camera, caster]);

	useFrame((_, delta) => {
		const dt = Math.min(delta, 1 / 30);

		if (intro.current < 1) intro.current = Math.min(1, intro.current + dt / INTRO);

		const t = intro.current;
		settled.current = t > 0.999;

		if (Math.abs(t - drawn.current) >= 0.0002) {
			drawn.current = t;
			fly(plates.current, DECK, t);
			fly(blocks.current, BLOCKS, t);
			hide(blocks.current, dropped.current);
		}

		for (let i = 0; i < PILLARS.length; i++) {
			const mesh = pillars.current[i];
			if (!mesh) continue;
			const p = PILLARS[i];
			const u = clamp01((t - p.start) / (p.end - p.start));
			if (u <= 0) {
				mesh.visible = false;
				continue;
			}
			mesh.visible = true;
			swell.current[i] = THREE.MathUtils.damp(
				swell.current[i],
				over.current[i] ? 1 : 0,
				9,
				dt,
			);
			const s = swell.current[i];
			const wide = 1 + (SWELL - 1) * s;
			const tall = 1 + (LIFT - 1) * s;
			const rise = Math.max(1e-3, easeOutBack(u) * tall);
			mesh.scale.set(p.width * wide, p.height * rise, p.depth * wide);

			stones.current[i]?.scale.set(wide, rise, wide);
			for (let k = 0; k < 2; k++) {
				const el = labels.current[i * 2 + k];
				if (el) el.style.scale = String(wide);
			}
		}

		const named = clamp01((t - LABELS[0]) / (LABELS[1] - LABELS[0]));
		for (const el of labels.current) {
			if (el) el.style.opacity = String(named);
		}

		const list = loose.current;
		if (list.length === 0) return;

		// Over the bottom edge is one way. Nothing brings these back short of a
		// remount, so they leave `loose` and stay zeroed in the instanced mesh.
		let lost = false;
		for (const b of list) {
			const below =
				(b.pos.y * RISE - (b.pos.x + b.pos.z) * DEPTH + GROUP_Y * RISE) * fit;
			if (below < floor - CUBE * fit) {
				if (dragging.current === b) dragging.current = null;
				dropped.current.add(b.i);
				lost = true;
			}
		}

		if (lost) {
			loose.current = list.filter((b) => !dropped.current.has(b.i));
			hide(blocks.current, dropped.current);
		}

		step(loose.current, dt, dropped.current);
		paint(blocks.current, loose.current);
	});

	return (
		<group position={[0, lift, 0]}>
		<group scale={fit}>
			<group ref={model} position={[0, GROUP_Y, 0]}>
				<instancedMesh
					ref={plates}
					args={[plate, skin, DECK.length]}
					frustumCulled={false}
				/>

				<instancedMesh
					ref={blocks}
					args={[cube, stone.material, BLOCKS.length]}
					frustumCulled={false}
					onPointerOver={() => {
						if (settled.current && canvas.current) canvas.current.style.cursor = "grab";
					}}
					onPointerOut={() => {
						if (!dragging.current && canvas.current) canvas.current.style.cursor = "";
					}}
					onPointerDown={(e) => {
						const group = model.current;
						const id = e.instanceId;
						if (id == null || !group) return;
						if (e.nativeEvent.pointerType === "touch") return;
						if (!settled.current) return;
						e.stopPropagation();

						let b = loose.current.find((l) => l.i === id);
						if (!b) {
							b = {
								i: id,
								pos: new THREE.Vector3(...BLOCKS[id].rest),
								vel: new THREE.Vector3(),
								quat: new THREE.Quaternion(),
								spin: new THREE.Vector3(),
								held: true,
								asleep: false,
							};
							loose.current.push(b);
						}
						b.held = true;
						b.asleep = false;
						b.vel.set(0, 0, 0);

						camera.getWorldDirection(spot);
						dragPlane.current.setFromNormalAndCoplanarPoint(spot, e.point);
						dragging.current = b;
						dragLast.current.copy(b.pos);
						dragTime.current = e.nativeEvent.timeStamp;
						canvas.current?.style.setProperty("cursor", "grabbing");
						canvas.current?.setPointerCapture(e.nativeEvent.pointerId);
					}}
				/>

				{PILLARS.map((p, i) => (
					<mesh
						key={p.rank}
						ref={(m) => {
							pillars.current[i] = m;
						}}
						geometry={post}
						material={tones[i]}
						position={[p.x, p.base, p.z]}
						visible={false}
						onPointerOver={(e) => {
							e.stopPropagation();
							over.current[i] = true;
						}}
						onPointerOut={() => {
							over.current[i] = false;
						}}
						onPointerDown={(e) => {
							e.stopPropagation();
						}}
					/>
				))}

				{PILLARS.map((p, i) => {
					const row = rows.find((r) => r.rank === p.rank);
					if (!row) return null;
					const px = fit * ZOOM;
					return (
						<group
							key={p.rank}
							ref={(g) => {
								stones.current[i] = g;
							}}
							position={[p.x, p.base, p.z]}
						>
							<Html
								position={[0, ETCH_HEIGHT, p.depth / 2]}
								center
								zIndexRange={[7, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * 2 + 0] = el;
									}}
									className="font-sans font-black tabular-nums"
									style={{
										opacity: 0,
										transform: LEFT_FACE,
										fontSize: `${0.82 * px}px`,
										lineHeight: 1,
										letterSpacing: "-0.03em",
										color: ETCH,
									}}
								>
									{String(row.rank).padStart(2, "0")}
								</div>
							</Html>

							<Html
								position={[0, p.height, 0]}
								center
								zIndexRange={[7, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * 2 + 1] = el;
									}}
									style={{
										opacity: 0,
										transform: TOP_FACE,
										filter: "brightness(0)",
									}}
								>
									<BrandMark lab={row.lab} size={Math.round(1.05 * px)} />
								</div>
							</Html>
						</group>
					);
				})}
			</group>
		</group>
		</group>
	);
}

export default function Podium({ rows, band }: { rows: Row[]; band: string }) {
	const host = useRef<HTMLDivElement>(null);
	const cap = useRef<HTMLDivElement>(null);
	const [live, setLive] = useState(true);
	const [bandPx, setBandPx] = useState(0);
	const reduced = useReducedMotion();

	useEffect(() => {
		const el = cap.current;
		if (!el) return;
		const ro = new ResizeObserver(([entry]) =>
			setBandPx(entry.contentRect.height),
		);
		ro.observe(el);
		return () => ro.disconnect();
	}, []);

	useEffect(() => {
		const el = host.current;
		if (!el) return;
		const io = new IntersectionObserver(
			([entry]) => setLive(entry.isIntersecting),
			{ threshold: 0 },
		);
		io.observe(el);
		return () => io.disconnect();
	}, []);

	return (
		<div ref={host} className="absolute inset-0">
			<div
				ref={cap}
				aria-hidden
				className="pointer-events-none absolute inset-x-0 top-0"
				style={{ height: band }}
			/>
			<Canvas
				flat
				dpr={[1, 2]}
				frameloop={live ? "always" : "never"}
				orthographic
				camera={{ position: [18, 18, 18], zoom: ZOOM, near: 0.1, far: 200 }}
				gl={{ antialias: true }}
			>
				<ambientLight intensity={0.45} />
				<directionalLight position={[9, 16, 7]} intensity={1.9} />
				<directionalLight position={[-11, 5, -9]} intensity={0.42} />
					<Stage rows={rows} reduced={reduced} band={bandPx} />
			</Canvas>
		</div>
	);
}
