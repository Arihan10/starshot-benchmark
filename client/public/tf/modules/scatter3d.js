// Interactive 3D scatter for the ablation "attention vs spatial relevance" graph,
// built on the VENDORED three.js (same engine as the scene viewer) — no new
// dependency, works offline. Floor = distance rank (X) × visibility rank (Z);
// HEIGHT (Y) = attention, optionally log. Orbit to rotate, scroll to zoom, hover
// a point for its label.
//
// A SINGLE instance is reused across report re-renders: its canvas lives in a
// detached wrapper that gets moved into whatever mount the latest render produced,
// so the camera + zoom survive toggles (color-by, log, poll-driven rebuilds). The
// render loop parks itself whenever the wrapper is detached and restarts on mount,
// so a hidden viewport costs nothing.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SPAN = 10;   // floor extent (world units), centered on origin
const HEIGHT = 6;  // max attention height (world units)
// BufferGeometry renders many points cheaply, but per-frame hover raycasting is
// O(n) and a spread-based max over a huge array crashes — so cap the rendered
// cloud (deterministic stride; the change-detection signature still uses the
// full set). Higher than the 2D SVG cap since the GPU does the drawing.
const MAX_3D_POINTS = 8000;

let inst = null;

// Move the (singleton) 3D viewport into `container`, (re)load its data, and make
// sure it's rendering. Returns the instance.
export function mountSpatial3D(container, points, opts = {}) {
	if (!inst) inst = createInstance();
	if (inst.wrapper.parentNode !== container) container.appendChild(inst.wrapper);
	inst.setData(points, opts);
	inst.resize();
	inst.ensureLoop();
	return inst;
}

export function resetSpatial3DView() { if (inst) inst.resetView(); }

function clearGroup(g) {
	for (const c of [...g.children]) {
		g.remove(c);
		if (c.geometry) c.geometry.dispose();
		if (c.material) { const m = c.material; if (m.map) m.map.dispose(); m.dispose(); }
	}
}

// A text label as a camera-facing sprite (canvas texture). depthTest off so labels
// never hide behind geometry.
function makeLabel(text, { color = "#cfe0f5", px = 48, worldH = 0.5 } = {}) {
	const font = `600 ${px}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
	const pad = 10;
	const c = document.createElement("canvas");
	const ctx = c.getContext("2d");
	ctx.font = font;
	const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
	const h = px + pad * 2;
	c.width = w; c.height = h;
	ctx.font = font;
	ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.textAlign = "center";
	ctx.fillText(text, w / 2, h / 2);
	const tex = new THREE.CanvasTexture(c);
	tex.minFilter = THREE.LinearFilter;
	const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false }));
	const s = worldH / h;
	sp.scale.set(w * s, h * s, 1);
	sp.renderOrder = 10;
	return sp;
}

function createInstance() {
	const wrapper = document.createElement("div");
	wrapper.style.cssText = "position:relative;width:100%;height:100%";

	let renderer;
	try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
	catch { renderer = new THREE.WebGLRenderer({ alpha: true }); }
	renderer.setPixelRatio(window.devicePixelRatio);
	renderer.setClearColor(0x000000, 0);
	renderer.domElement.addEventListener("webglcontextlost", (e) => e.preventDefault(), false);
	renderer.domElement.style.cssText = "display:block;width:100%;height:100%;cursor:grab";
	wrapper.appendChild(renderer.domElement);

	const tooltip = document.createElement("div");
	tooltip.style.cssText = "position:absolute;pointer-events:none;opacity:0;transition:opacity .08s ease;z-index:6;" +
		"background:rgba(13,15,20,.96);border:1px solid rgba(255,255,255,.16);border-radius:6px;padding:5px 8px;" +
		"font:11px ui-monospace,Menlo,monospace;color:#dce6f5;white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.5)";
	wrapper.appendChild(tooltip);

	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
	const controls = new OrbitControls(camera, renderer.domElement);
	controls.enableDamping = true;
	controls.dampingFactor = 0.09;

	const staticRoot = new THREE.Group(); // floor grid + axes + axis titles
	const tickRoot = new THREE.Group();   // attention tick labels (rebuilt per setData)
	const dataRoot = new THREE.Group();   // points + stems (rebuilt per setData)
	scene.add(staticRoot, tickRoot, dataRoot);

	// static scaffold
	const grid = new THREE.GridHelper(SPAN, 10, 0x3a3f4c, 0x23262e);
	grid.material.transparent = true; grid.material.opacity = 0.6;
	staticRoot.add(grid);
	const axGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-SPAN / 2, 0, -SPAN / 2), new THREE.Vector3(-SPAN / 2, HEIGHT, -SPAN / 2)]);
	staticRoot.add(new THREE.Line(axGeo, new THREE.LineBasicMaterial({ color: 0x55606f })));
	const tX = makeLabel("distance rank \u2192", { worldH: 0.55 }); tX.position.set(SPAN / 2 + 0.2, 0.15, -SPAN / 2); staticRoot.add(tX);
	const tZ = makeLabel("visibility rank \u2192", { worldH: 0.55 }); tZ.position.set(-SPAN / 2, 0.15, SPAN / 2 + 0.2); staticRoot.add(tZ);
	const tClose = makeLabel("close + visible", { color: "#8fe0a0", px: 38, worldH: 0.4 }); tClose.position.set(-SPAN / 2, -0.35, -SPAN / 2); staticRoot.add(tClose);
	const tY = makeLabel("attention \u2191", { worldH: 0.55 }); tY.position.set(-SPAN / 2, HEIGHT + 0.55, -SPAN / 2); staticRoot.add(tY);

	let pointsObj = null, curPoints = [], lastSig = "";

	const raycaster = new THREE.Raycaster();
	raycaster.params.Points.threshold = 0.22;
	const pointer = new THREE.Vector2();
	let pointerInside = false;

	renderer.domElement.addEventListener("pointermove", (e) => {
		const r = renderer.domElement.getBoundingClientRect();
		pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
		pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
		pointerInside = true;
		tooltip.style.left = `${e.clientX - r.left + 12}px`;
		tooltip.style.top = `${e.clientY - r.top + 12}px`;
	});
	renderer.domElement.addEventListener("pointerleave", () => { pointerInside = false; tooltip.style.opacity = "0"; });

	function setData(pts, o = {}) {
		pts = pts || [];
		// Skip a rebuild when nothing that affects the geometry changed — the report
		// workspace re-renders often (polls, toggles elsewhere), and a no-op rebuild
		// would churn buffers for no reason. Camera + geometry are kept as-is.
		const sig = `${pts.length}|${!!o.logZ}|${!!o.com}|${(o.cats || []).map((c) => `${c.key}:${c.color}`).join(",")}|` +
			pts.reduce((s, p) => s + (p.attn || 0) + p.x * 1e-3 + p.y * 1e-6, 0).toFixed(3);
		if (sig === lastSig && pointsObj) return;
		lastSig = sig;
		// Cap the rendered cloud (uniform stride) so geometry + hover raycasting stay
		// bounded no matter how many variants × objects × replicates are in view.
		curPoints = pts.length > MAX_3D_POINTS
			? pts.filter((_, i) => i % Math.ceil(pts.length / MAX_3D_POINTS) === 0)
			: pts;
		clearGroup(dataRoot); clearGroup(tickRoot);
		pointsObj = null;
		if (!curPoints.length) return;
		const cats = (o.cats && o.cats.length) ? o.cats : [{ key: "", color: "#7aa2f7" }];
		// r163 has color management on by default → new THREE.Color(hex) already holds
		// LINEAR values, exactly what the vertex-color pipeline wants (no manual convert).
		const colorOf = new Map(cats.map((c) => [c.key, new THREE.Color(c.color)]));
		const labelOf = new Map(cats.map((c) => [c.key, c.label || String(c.key)]));
		const fallback = colorOf.values().next().value || new THREE.Color("#7aa2f7");
		// reduce, not Math.max(...spread) — belt-and-suspenders against a large array.
		let maxRank = 1, maxA = 1e-9;
		for (const p of curPoints) { if (p.x > maxRank) maxRank = p.x; if (p.y > maxRank) maxRank = p.y; const a = p.attn || 0; if (a > maxA) maxA = a; }
		const logZ = !!o.logZ;
		const zn = (a) => { a = Math.max(0, a || 0); return logZ ? Math.log1p(a) / Math.log1p(maxA) : a / maxA; };
		const N = curPoints.length;
		const pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
		const sPos = new Float32Array(N * 6), sCol = new Float32Array(N * 6);
		const com = new Map(); // cat -> attention-weighted centroid accumulator (world coords)
		curPoints.forEach((p, i) => {
			const wx = (p.x / maxRank) * SPAN - SPAN / 2;
			const wz = (p.y / maxRank) * SPAN - SPAN / 2;
			const wy = zn(p.attn) * HEIGHT;
			const c = colorOf.get(p.cat) || fallback;
			pos[i * 3] = wx; pos[i * 3 + 1] = wy; pos[i * 3 + 2] = wz;
			col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
			sPos[i * 6] = wx; sPos[i * 6 + 1] = 0; sPos[i * 6 + 2] = wz;
			sPos[i * 6 + 3] = wx; sPos[i * 6 + 4] = wy; sPos[i * 6 + 5] = wz;
			for (const k of [0, 1]) { sCol[i * 6 + k * 3] = c.r; sCol[i * 6 + k * 3 + 1] = c.g; sCol[i * 6 + k * 3 + 2] = c.b; }
			// accumulate the CoM (mass = attention) per category
			const w = Math.max(0, p.attn || 0);
			const a = com.get(p.cat) || com.set(p.cat, { w: 0, x: 0, y: 0, z: 0, n: 0, color: c }).get(p.cat);
			a.w += w; a.x += w * wx; a.y += w * wy; a.z += w * wz; a.n += 1;
		});
		const g = new THREE.BufferGeometry();
		g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
		g.setAttribute("color", new THREE.BufferAttribute(col, 3));
		pointsObj = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.3, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.95 }));
		dataRoot.add(pointsObj);
		const sg = new THREE.BufferGeometry();
		sg.setAttribute("position", new THREE.BufferAttribute(sPos, 3));
		sg.setAttribute("color", new THREE.BufferAttribute(sCol, 3));
		dataRoot.add(new THREE.LineSegments(sg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.3 })));
		// Per-category CENTER OF MASS (attention-weighted): a big haloed sphere at the
		// centroid, a drop-line to the floor, and a label — so each category's typical
		// (distance, visibility, attention) reads at a glance vs the point cloud.
		if (o.com !== false) {
			for (const [cat, a] of com) {
				if (a.w <= 0) continue;
				const cx = a.x / a.w, cy = a.y / a.w, cz = a.z / a.w;
				const sph = new THREE.Mesh(new THREE.SphereGeometry(0.34, 20, 14), new THREE.MeshBasicMaterial({ color: a.color }));
				sph.position.set(cx, cy, cz); sph.renderOrder = 3; dataRoot.add(sph);
				const halo = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.28 }));
				halo.position.set(cx, cy, cz); dataRoot.add(halo);
				const dropG = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(cx, 0, cz), new THREE.Vector3(cx, cy, cz)]);
				dataRoot.add(new THREE.Line(dropG, new THREE.LineBasicMaterial({ color: a.color, transparent: true, opacity: 0.7 })));
				const foot = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8), new THREE.MeshBasicMaterial({ color: a.color, transparent: true, opacity: 0.7 }));
				foot.position.set(cx, 0, cz); dataRoot.add(foot);
				const lab = makeLabel(`◎ ${labelOf.get(cat) || cat}`, { color: "#ffffff", px: 42, worldH: 0.46 });
				lab.position.set(cx, cy + 0.7, cz); dataRoot.add(lab);
			}
		}
		// attention tick labels up the vertical axis
		for (let i = 0; i <= 4; i++) {
			const f = i / 4;
			const av = logZ ? Math.expm1(f * Math.log1p(maxA)) : f * maxA;
			const lab = makeLabel(av >= 100 ? av.toFixed(0) : av >= 1 ? av.toFixed(1) : av.toFixed(2), { color: "#9fb0c8", px: 34, worldH: 0.34 });
			lab.position.set(-SPAN / 2 - 0.55, f * HEIGHT, -SPAN / 2);
			tickRoot.add(lab);
		}
	}

	function resize() {
		const w = wrapper.clientWidth || 1, h = wrapper.clientHeight || 1;
		renderer.setSize(w, h, false);
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
	}
	new ResizeObserver(resize).observe(wrapper);

	function resetView() {
		camera.position.set(SPAN * 0.95, HEIGHT * 1.75, SPAN * 1.2);
		controls.target.set(0, HEIGHT * 0.4, 0);
		controls.update();
	}
	resetView();

	let running = false;
	function loop() {
		if (!wrapper.isConnected) { running = false; return; } // detached (view swapped) → park
		controls.update();
		if (pointerInside && pointsObj) {
			raycaster.setFromCamera(pointer, camera);
			const hit = raycaster.intersectObject(pointsObj, false);
			if (hit.length) {
				const p = curPoints[hit[0].index];
				if (p) { tooltip.textContent = p.label || ""; tooltip.style.opacity = "1"; } else tooltip.style.opacity = "0";
			} else tooltip.style.opacity = "0";
		}
		renderer.render(scene, camera);
		requestAnimationFrame(loop);
	}
	function ensureLoop() { if (!running) { running = true; requestAnimationFrame(loop); } }

	return { wrapper, setData, resize, resetView, ensureLoop };
}
