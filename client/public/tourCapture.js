// =============================================================================
// tourCapture.js — the scene capture / auto-tour pipeline, in one place.
// =============================================================================
//
// This module is the entire client side of the "auto tour" export: turning the
// live three.js scene into the artifacts the prod walkthrough consumes —
//
//   • 360° panoramas   one equirectangular JPEG per capture point ("anchor")
//   • projection proxy  a merged, world-space, material-free GLB the server
//                        decimates; the walkthrough projects the panos onto it
//                        to parallax/interpolate between discrete capture points
//   • bird's-eye minimaps  one top-down slice per storey (Y level)
//   • tour.json manifest    positions + filenames tying the above together
//
// It is deliberately split out of viewer.js (10k+ lines) so the whole capture
// flow can be read top-to-bottom, and so it can be ported to Unity (for high-
// quality environments) against a single, dependency-injected reference.
//
// ── Two entry flows ─────────────────────────────────────────────────────────
//   captureManual + saveTour : the operator frames each shot, then downloads
//                              tour.json + proxy.glb + one JPEG per pano.
//   runAutoTour              : an LLM proposes anchors (POST /anchors); we drive
//                              the camera to each, capture, and PUT/POST the
//                              whole tour back to the server so it persists under
//                              /artifacts/<cell>/tour/ for /pano to load by URL.
//
// ── Server endpoints (relative to serverUrl) ────────────────────────────────
//   POST  /proxy                                  stateless decimation (manual)
//   POST  <cell>/anchors                          LLM anchor plan + names
//   POST  <cell>/tour/reset                       wipe the cell's tour/ dir
//   PUT   <cell>/tour/pano/<id>     (image/jpeg)   one captured pano
//   PUT   <cell>/tour/minimap/<id>  (image/png)    one storey slice
//   POST  <cell>/tour/proxy   (model/gltf-binary)  merged scene → proxy.glb
//   POST  <cell>/tour/manifest     (application/json)  write tour.json → tour_url
//   where  <cell> = /slots/<slotId>/<model>  and every request carries ?run=<run>
//
// ── Porting to Unity (parity notes) ─────────────────────────────────────────
//   • 360 capture: render a cubemap (or 6 RenderTextures, 90° FOV) at the anchor
//     and convert to equirect with the SAME convention used here and by the
//     consumer shaders: u = atan2(z,x)/2π + 0.5, v = asin(y)/π + 0.5.
//   • proxy: export the visible meshes merged into ONE world-space, material-free
//     mesh, preserving a node per source object NAMED with its object id (object
//     identity is what lets the walkthrough highlight individual objects).
//   • minimap: an orthographic top-down camera with a horizontal clip slab
//     (cut above head height, and just below the storey's lowest camera).
//   • everything else is HTTP — the endpoints above are the contract.
//
// ── Injected dependencies (the only coupling to the host viewer) ─────────────
//   renderer, scene, sceneRoot, bboxRoot, camera, controls, modelsById
//                              live three.js runtime objects (consts in viewer)
//   serverUrl                  API origin (string)
//   getCell()                  → { run, slotId, model } | null  (the active cell)
//   getCameraUserMoved()       read the shared "user moved the camera" flag
//   setCameraUserMoved(v)      write it back (auto-tour locks it while driving)
//   onEvent(event)             append to the viewer's event log
//   setStatus(text, cls?)      set the viewer's status line
//   onChange()                 notify the host that busy/count changed (re-render
//                              the toolbar); never called during construction.
// =============================================================================

import * as THREE from "three";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// --- capture tunables --------------------------------------------------------

const PANO_FACE_TARGET = 1280; // device px per cube face, capped by canvas size
const PANO_WIDTH_CAP = 4096; // equirect output width cap (height = width / 2)

// forward/up per face; right = cross(forward, up) — matches what lookAt builds,
// so the analytic projection in the stitch agrees with the render exactly.
// Order: +X, -X, +Y, -Y, +Z, -Z (indexed as axis*2 + (negative ? 1 : 0)).
const PANO_FACES = [
	{ f: [1, 0, 0], up: [0, 1, 0] },
	{ f: [-1, 0, 0], up: [0, 1, 0] },
	{ f: [0, 1, 0], up: [0, 0, 1] },
	{ f: [0, -1, 0], up: [0, 0, -1] },
	{ f: [0, 0, 1], up: [0, 1, 0] },
	{ f: [0, 0, -1], up: [0, 1, 0] },
];
const PANO_FACE_BASIS = PANO_FACES.map(({ f, up }) => ({
	f,
	up,
	right: [
		f[1] * up[2] - f[2] * up[1],
		f[2] * up[0] - f[0] * up[2],
		f[0] * up[1] - f[1] * up[0],
	],
}));

const MINIMAP_LEVEL_EPS = 1.5; // metres; anchors within this Y gap share a level
const MINIMAP_RES = 1024; // longest output side in device px (capped by the canvas)
const MINIMAP_PAD_FRAC = 0.04; // breathing room around the scene footprint
const MINIMAP_SLICE_BELOW = 2; // metres below the level's lowest anchor for the floor cut

// =============================================================================
// createTourCapture(deps) → the capture API bound to one viewer's runtime.
// =============================================================================
export function createTourCapture(deps) {
	const {
		renderer,
		scene,
		sceneRoot,
		bboxRoot,
		camera,
		controls,
		modelsById,
		serverUrl,
		getCell,
		getCameraUserMoved,
		setCameraUserMoved,
		onEvent = () => {},
		setStatus = () => {},
		onChange = () => {},
	} = deps;

	// Manual-tour accumulator: { id, position, forward, blob } per capture.
	const panoTour = [];
	// Shared across every op (manual + auto) so the toolbar can lock out re-entry.
	let busy = false;

	// --- 360° panorama capture ------------------------------------------------
	//
	// Render the live scene six times (cube faces, 90° fov) into a square
	// scissored viewport on the main canvas — full parity with what's on screen
	// (ACES, sRGB, shadows, IBL, AA; preserveDrawingBuffer is already on for the
	// gif exporter, so the canvas is readable after render) — then CPU-stitch
	// into an equirect panorama using three's EquirectangularReflectionMapping
	// direction→uv convention, so the JPEGs drop into any three.js scene.

	// Render the six cube faces and read each back as ImageData. Synchronous;
	// the animate loop repaints the viewport next frame.
	function renderPanoFaces() {
		const canvas = renderer.domElement;
		const dpr = renderer.getPixelRatio();
		const prevSize = renderer.getSize(new THREE.Vector2());
		const faceSize = Math.min(PANO_FACE_TARGET, canvas.width, canvas.height);
		const faceCss = faceSize / dpr; // setViewport/Scissor multiply by dpr

		const faceCam = new THREE.PerspectiveCamera(90, 1, camera.near, camera.far);
		faceCam.position.copy(camera.position);

		const crop = document.createElement("canvas");
		crop.width = faceSize;
		crop.height = faceSize;
		const cropCtx = crop.getContext("2d", { willReadFrequently: true });

		// Debug wireframes don't belong in a "realistic" pano.
		const bboxWasVisible = bboxRoot.visible;
		bboxRoot.visible = false;

		const faces = [];
		try {
			renderer.setScissorTest(true);
			for (const { f, up } of PANO_FACES) {
				faceCam.up.set(up[0], up[1], up[2]);
				faceCam.lookAt(
					camera.position.x + f[0],
					camera.position.y + f[1],
					camera.position.z + f[2],
				);
				renderer.setViewport(0, 0, faceCss, faceCss);
				renderer.setScissor(0, 0, faceCss, faceCss);
				renderer.render(scene, faceCam);
				// Viewport (0,0) is the canvas' bottom-left; drawImage's source rect
				// is top-left-origin device pixels.
				cropCtx.drawImage(
					canvas,
					0,
					canvas.height - faceSize,
					faceSize,
					faceSize,
					0,
					0,
					faceSize,
					faceSize,
				);
				faces.push(cropCtx.getImageData(0, 0, faceSize, faceSize));
			}
		} finally {
			renderer.setScissorTest(false);
			renderer.setViewport(0, 0, prevSize.x, prevSize.y);
			renderer.setScissor(0, 0, prevSize.x, prevSize.y);
			bboxRoot.visible = bboxWasVisible;
		}
		return { faces, faceSize };
	}

	// Stitch six face ImageDatas into one equirect ImageData (bilinear sampling).
	// Chunked by rows so the tab stays responsive on 4096×2048 outputs.
	async function stitchPanoEquirect(faces, faceSize, onProgress) {
		const W = Math.min(PANO_WIDTH_CAP, faceSize * 4);
		const H = W / 2;
		const out = new ImageData(W, H);
		const o = out.data;
		const S = faceSize;
		const maxIdx = S - 1;

		for (let row = 0; row < H; row++) {
			const v = 1 - (row + 0.5) / H;
			const phi = (v - 0.5) * Math.PI;
			const dy = Math.sin(phi);
			const cosPhi = Math.cos(phi);
			let oi = row * W * 4;
			for (let col = 0; col < W; col++, oi += 4) {
				const az = ((col + 0.5) / W - 0.5) * 2 * Math.PI;
				const dx = cosPhi * Math.cos(az);
				const dz = cosPhi * Math.sin(az);

				const ax = Math.abs(dx);
				const ay = Math.abs(dy);
				const az2 = Math.abs(dz);
				let faceIdx;
				if (ax >= ay && ax >= az2) faceIdx = dx > 0 ? 0 : 1;
				else if (ay >= az2) faceIdx = dy > 0 ? 2 : 3;
				else faceIdx = dz > 0 ? 4 : 5;

				const { f, up, right } = PANO_FACE_BASIS[faceIdx];
				const t = dx * f[0] + dy * f[1] + dz * f[2];
				const u2 = (dx * right[0] + dy * right[1] + dz * right[2]) / t;
				const v2 = (dx * up[0] + dy * up[1] + dz * up[2]) / t;

				// Face pixel coords (image y down) + bilinear weights.
				const px = (u2 * 0.5 + 0.5) * S - 0.5;
				const py = (0.5 - v2 * 0.5) * S - 0.5;
				let x0 = Math.floor(px);
				let y0 = Math.floor(py);
				const fx = px - x0;
				const fy = py - y0;
				x0 = x0 < 0 ? 0 : x0 > maxIdx ? maxIdx : x0;
				y0 = y0 < 0 ? 0 : y0 > maxIdx ? maxIdx : y0;
				const x1 = x0 < maxIdx ? x0 + 1 : maxIdx;
				const y1 = y0 < maxIdx ? y0 + 1 : maxIdx;

				const d = faces[faceIdx].data;
				const i00 = (y0 * S + x0) * 4;
				const i10 = (y0 * S + x1) * 4;
				const i01 = (y1 * S + x0) * 4;
				const i11 = (y1 * S + x1) * 4;
				const w00 = (1 - fx) * (1 - fy);
				const w10 = fx * (1 - fy);
				const w01 = (1 - fx) * fy;
				const w11 = fx * fy;

				o[oi] = d[i00] * w00 + d[i10] * w10 + d[i01] * w01 + d[i11] * w11;
				o[oi + 1] =
					d[i00 + 1] * w00 +
					d[i10 + 1] * w10 +
					d[i01 + 1] * w01 +
					d[i11 + 1] * w11;
				o[oi + 2] =
					d[i00 + 2] * w00 +
					d[i10 + 2] * w10 +
					d[i01 + 2] * w01 +
					d[i11 + 2] * w11;
				o[oi + 3] = 255;
			}
			if (row % 128 === 127) {
				onProgress?.(row / H);
				await sleep(0);
			}
		}

		const outCanvas = document.createElement("canvas");
		outCanvas.width = W;
		outCanvas.height = H;
		outCanvas.getContext("2d").putImageData(out, 0, 0);
		return new Promise((resolve, reject) =>
			outCanvas.toBlob(
				(blob) =>
					blob ? resolve(blob) : reject(new Error("JPEG encode failed")),
				"image/jpeg",
				0.92,
			),
		);
	}

	// Render + stitch one 360° equirectangular pano from the CURRENT camera
	// position (orientation-independent — renderPanoFaces builds its own
	// axis-aligned face cameras). Returns the JPEG blob.
	async function capturePanoBlob(onProgress) {
		const { faces, faceSize } = renderPanoFaces();
		return stitchPanoEquirect(faces, faceSize, onProgress);
	}

	// --- projection proxy bake ------------------------------------------------

	// Bake one placed mesh's geometry into world-space, float32, position-only
	// geometry. Reading every vertex through `fromBufferAttribute` DENORMALIZES
	// quantized attributes, and writing into a FRESH Float32 array sidesteps the
	// trap that broke the proxy: placed library GLBs are Meshopt /
	// KHR_mesh_quantization, so three keeps POSITION as an INTEGER buffer with
	// the dequantization folded into the node matrix. `geometry.clone()
	// .applyMatrix4(matrixWorld)` would write world-space floats back into that
	// integer buffer — truncating every vertex onto the integer grid.
	function bakeWorldGeometry(mesh) {
		const src = mesh.geometry.getAttribute("position");
		if (!src) return null;
		const count = src.count;
		const positions = new Float32Array(count * 3);
		const v = new THREE.Vector3();
		const m = mesh.matrixWorld;
		for (let i = 0; i < count; i++) {
			v.fromBufferAttribute(src, i).applyMatrix4(m);
			positions[i * 3] = v.x;
			positions[i * 3 + 1] = v.y;
			positions[i * 3 + 2] = v.z;
		}
		const g = new THREE.BufferGeometry();
		g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
		// Keep the topology (a copy, detached from the source buffer). Normals are
		// dropped: the server strips them and /pano recomputes them on the proxy.
		const idx = mesh.geometry.getIndex();
		if (idx) g.setIndex(new THREE.BufferAttribute(idx.array.slice(), 1));
		return g;
	}

	// The owning object's id for a mesh: walk up to the node the loader tagged
	// with `pickId` (set on every placed `gltf.scene`). Carries object identity
	// into the baked proxy so the walkthrough can name / address individual
	// objects after decimation (gltf-transform preserves node names).
	function pickIdOf(obj) {
		for (let cur = obj; cur; cur = cur.parent) {
			if (cur.userData?.pickId) return cur.userData.pickId;
		}
		return null;
	}

	// Bake the live scene into one material-free, world-space GLB (geometry only:
	// each placed mesh baked into world space — the same frame the pano positions
	// were captured in). This is the merged stand-in the server's /proxy
	// decimator reduces to a few-thousand-triangle projection proxy. Each source
	// object's baked meshes hang under their own node, named with the object id,
	// so the proxy keeps per-object identity through decimation. Returns the
	// binary GLB ArrayBuffer, or null when the scene has no meshes.
	async function buildMergedSceneGlbBuffer() {
		if (modelsById.size === 0) return null;
		const root = new THREE.Group();
		const mat = new THREE.MeshStandardMaterial();
		const geoms = [];
		const objNodes = new Map(); // object id (or null) -> its node under `root`
		sceneRoot.updateWorldMatrix(true, true);
		sceneRoot.traverse((o) => {
			if (!o.isMesh || !o.geometry) return;
			const g = bakeWorldGeometry(o);
			if (!g) return;
			const id = pickIdOf(o);
			const key = id ?? "";
			let node = objNodes.get(key);
			if (!node) {
				node = new THREE.Group();
				if (id) node.name = id;
				objNodes.set(key, node);
				root.add(node);
			}
			node.add(new THREE.Mesh(g, mat));
			geoms.push(g);
		});
		if (geoms.length === 0) return null;
		try {
			const exporter = new GLTFExporter();
			return await exporter.parseAsync(root, {
				binary: true,
				onlyVisible: false,
			});
		} finally {
			for (const g of geoms) g.dispose();
			mat.dispose();
		}
	}

	// Build the merged scene and hand it to the stateless /proxy endpoint,
	// returning the decimated proxy blob for the downloadable tour bundle.
	async function buildProxyGlbBlob() {
		const glb = await buildMergedSceneGlbBuffer();
		if (!glb) return null;
		const res = await fetch(new URL("/proxy", serverUrl).toString(), {
			method: "POST",
			headers: { "Content-Type": "model/gltf-binary" },
			body: glb,
		});
		if (!res.ok) throw new Error(`server /proxy → ${res.status}`);
		return new Blob([await res.arrayBuffer()], { type: "model/gltf-binary" });
	}

	// --- bird's-eye minimap slices (one per Y level) --------------------------
	//
	// Group the anchors into Y "levels" (storeys — anchors within
	// MINIMAP_LEVEL_EPS metres of each other) and render one top-down orthographic
	// slice per level. The slice is a horizontal SLAB at camera level: cut above
	// the head (drops the roof, so we see in) AND a bit below the lowest camera in
	// the group (drops the floor-and-below, so an upper storey's slice can't show
	// the floor beneath it). The prod client shows the matching slice as a minimap
	// and dots the level's anchors onto it, mapping each anchor's world XZ through
	// the stored `bounds`.

	// Cluster anchor Ys into levels by gap; returns [{ y, minY, indices }] low→high,
	// where y is the level's median camera height (its top slice-cut + client match
	// key) and minY is its lowest camera (the bottom slice-cut rides just under it).
	function groupAnchorLevels(positions) {
		const order = positions
			.map((_, i) => i)
			.sort((a, b) => positions[a][1] - positions[b][1]);
		const groups = [];
		let cur = null;
		for (const i of order) {
			const y = positions[i][1];
			if (!cur || y - cur.lastY > MINIMAP_LEVEL_EPS) {
				cur = { indices: [], ys: [], lastY: y };
				groups.push(cur);
			}
			cur.indices.push(i);
			cur.ys.push(y);
			cur.lastY = y;
		}
		return groups.map((g) => {
			const ys = g.ys.slice().sort((a, b) => a - b);
			return { y: ys[(ys.length - 1) >> 1], minY: ys[0], indices: g.indices };
		});
	}

	// Render one top-down slice of the live scene into a PNG blob. Reuses the main
	// renderer/canvas (so tonemapping + sRGB match the panos), scissored to a
	// footprint-aspect viewport then read back. The ortho camera looks straight
	// down with -Z "up" in the image, so the stored `bounds` map world (x,z) →
	// image (left,top) as ((x-minX)/W, (z-minZ)/D).
	async function captureMinimapBlob(bounds, cutTop, cutBottom, yTop, yBot) {
		const canvas = renderer.domElement;
		const dpr = renderer.getPixelRatio();
		const prevSize = renderer.getSize(new THREE.Vector2());

		const W = bounds.maxX - bounds.minX;
		const D = bounds.maxZ - bounds.minZ;
		const cx = (bounds.minX + bounds.maxX) / 2;
		const cz = (bounds.minZ + bounds.maxZ) / 2;

		// Output pixels preserve the footprint aspect, capped by MINIMAP_RES and the
		// drawing buffer (we read back from the canvas, so we can't exceed it).
		const cap = Math.min(MINIMAP_RES, canvas.width, canvas.height);
		let pw;
		let ph;
		if (W >= D) {
			pw = cap;
			ph = Math.max(1, Math.round((cap * D) / W));
		} else {
			ph = cap;
			pw = Math.max(1, Math.round((cap * W) / D));
		}

		const cam = new THREE.OrthographicCamera(
			-W / 2,
			W / 2,
			D / 2,
			-D / 2,
			0.1,
			yTop - yBot + 4,
		);
		cam.position.set(cx, yTop + 2, cz);
		cam.up.set(0, 0, -1);
		cam.lookAt(cx, yBot, cz);
		cam.updateProjectionMatrix();

		// World clip planes bounding a horizontal SLAB: keep cutBottom <= y <= cutTop.
		// The top cut opens the roof; the bottom cut drops the floor-and-below —
		// including lower storeys. Global clipping planes intersect (a fragment
		// outside EITHER is dropped).
		const planeTop = new THREE.Plane(new THREE.Vector3(0, -1, 0), cutTop);
		const planeBottom = new THREE.Plane(new THREE.Vector3(0, 1, 0), -cutBottom);
		const prevClip = renderer.clippingPlanes;
		const prevBg = scene.background;
		const prevClear = renderer.getClearColor(new THREE.Color());
		const prevAlpha = renderer.getClearAlpha();
		const prevShadow = renderer.shadowMap.enabled;
		const bboxWasVisible = bboxRoot.visible;

		const crop = document.createElement("canvas");
		crop.width = pw;
		crop.height = ph;

		try {
			bboxRoot.visible = false; // debug wireframes don't belong on the map
			// Flat, evenly-lit floor plan reads clearer than a top-down cast-shadow
			// render; the clipping-plane swap below forces a program refresh, so this
			// toggle takes effect for the slice pass.
			renderer.shadowMap.enabled = false;
			renderer.clippingPlanes = [planeTop, planeBottom];
			scene.background = null;
			renderer.setClearColor(0x0c0d10, 1);
			renderer.setScissorTest(true);
			renderer.setViewport(0, 0, pw / dpr, ph / dpr);
			renderer.setScissor(0, 0, pw / dpr, ph / dpr);
			renderer.render(scene, cam);
			// Viewport (0,0) is the canvas' bottom-left; drawImage's source rect is
			// top-left-origin device pixels.
			crop.getContext("2d").drawImage(
				canvas,
				0,
				canvas.height - ph,
				pw,
				ph,
				0,
				0,
				pw,
				ph,
			);
		} finally {
			renderer.setScissorTest(false);
			renderer.setViewport(0, 0, prevSize.x, prevSize.y);
			renderer.setScissor(0, 0, prevSize.x, prevSize.y);
			renderer.clippingPlanes = prevClip;
			renderer.shadowMap.enabled = prevShadow;
			scene.background = prevBg;
			renderer.setClearColor(prevClear, prevAlpha);
			bboxRoot.visible = bboxWasVisible;
		}
		return new Promise((resolve, reject) =>
			crop.toBlob(
				(blob) =>
					blob ? resolve(blob) : reject(new Error("minimap encode failed")),
				"image/png",
			),
		);
	}

	async function uploadMinimap(cell, minimapId, blob) {
		const res = await fetch(
			new URL(
				`${cell.base}/tour/minimap/${encodeURIComponent(minimapId)}?${cell.run}`,
				serverUrl,
			).toString(),
			{
				method: "PUT",
				headers: { "Content-Type": "image/png" },
				body: blob,
			},
		);
		if (!res.ok) throw new Error(`upload ${minimapId} → ${res.status}`);
	}

	// Group the captured anchors by level, render + persist one bird's-eye slice
	// per level, and return the manifest `minimaps` array (empty on any failure —
	// the tour stays valid without them).
	async function buildMinimaps(cell, panoMeta, onLevel) {
		const positions = panoMeta.map((p) => p.position);
		if (positions.length === 0) return [];
		const box = new THREE.Box3().setFromObject(sceneRoot);
		if (box.isEmpty()) return [];
		const pad =
			MINIMAP_PAD_FRAC *
			Math.max(box.max.x - box.min.x, box.max.z - box.min.z, 1);
		const bounds = {
			minX: box.min.x - pad,
			maxX: box.max.x + pad,
			minZ: box.min.z - pad,
			maxZ: box.max.z + pad,
		};
		const levels = groupAnchorLevels(positions);
		const minimaps = [];
		for (let li = 0; li < levels.length; li++) {
			onLevel?.(li, levels.length);
			const file = `minimap-${li}.png`;
			// Top cut above the head (median camera height); bottom cut a bit below
			// the level's lowest camera — a slab at camera level, isolated from other
			// storeys.
			const blob = await captureMinimapBlob(
				bounds,
				levels[li].y,
				levels[li].minY - MINIMAP_SLICE_BELOW,
				box.max.y,
				box.min.y,
			);
			await uploadMinimap(cell, `minimap-${li}`, blob);
			minimaps.push({ level: li, y: levels[li].y, file, bounds });
		}
		return minimaps;
	}

	// --- shared helpers -------------------------------------------------------

	function downloadPanoBlob(blob, filename) {
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = filename;
		a.click();
		URL.revokeObjectURL(url);
	}

	// The active cell as request parts, or null when no run/slot/model is active.
	function resolveCell() {
		const c = getCell?.();
		if (!c || !c.run || !c.slotId || !c.model) return null;
		return {
			base: `/slots/${encodeURIComponent(c.slotId)}/${encodeURIComponent(c.model)}`,
			run: `run=${encodeURIComponent(c.run)}`,
		};
	}

	async function uploadPano(cell, panoId, blob) {
		const res = await fetch(
			new URL(
				`${cell.base}/tour/pano/${encodeURIComponent(panoId)}?${cell.run}`,
				serverUrl,
			).toString(),
			{
				method: "PUT",
				headers: { "Content-Type": "image/jpeg" },
				body: blob,
			},
		);
		if (!res.ok) throw new Error(`upload ${panoId} → ${res.status}`);
	}

	// =========================================================================
	// Public ops. Each guards on `busy`, flips it around the work, and pings
	// onChange so the toolbar can re-render. `onPhase(text)` reports progress.
	// =========================================================================

	// Capture one 360 from the current camera and add it to the manual tour.
	async function captureManual({ onPhase } = {}) {
		if (busy) return;
		busy = true;
		onChange();
		try {
			onPhase?.("rendering…");
			const blob = await capturePanoBlob((frac) =>
				onPhase?.(`stitching ${Math.round(frac * 100)}%`),
			);
			const fwd = camera.getWorldDirection(new THREE.Vector3());
			panoTour.push({
				id: `pano-${String(panoTour.length).padStart(3, "0")}`,
				position: camera.position.toArray(),
				forward: fwd.toArray(),
				blob,
			});
		} catch (e) {
			onEvent({ kind: "run.error", message: `360 capture failed: ${e.message}` });
		} finally {
			busy = false;
			onChange();
		}
	}

	// Decimate the manual tour's scene into a proxy, then download tour.json +
	// proxy.glb + one JPEG per pano. The manifest only advertises proxy.glb if it
	// actually built, so /pano falls back to its sphere mode when it's missing.
	async function saveTour({ onPhase } = {}) {
		if (panoTour.length === 0 || busy) return;
		busy = true;
		onChange();
		let proxyBlob = null;
		try {
			onPhase?.("building proxy…");
			proxyBlob = await buildProxyGlbBlob();
		} catch (e) {
			onEvent({
				kind: "run.error",
				message: `proxy build failed (saving panos without it): ${e.message}`,
			});
		}
		const manifest = {
			version: 1,
			proxy: proxyBlob ? "proxy.glb" : null,
			panos: panoTour.map((p) => ({
				id: p.id,
				file: `${p.id}.jpg`,
				position: p.position,
				forward: p.forward,
			})),
		};
		try {
			onPhase?.("saving…");
			downloadPanoBlob(
				new Blob([JSON.stringify(manifest, null, 2)], {
					type: "application/json",
				}),
				"tour.json",
			);
			// Space the downloads out so the browser doesn't coalesce/drop them.
			if (proxyBlob) {
				await sleep(250);
				downloadPanoBlob(proxyBlob, "proxy.glb");
			}
			for (const p of panoTour) {
				await sleep(250);
				downloadPanoBlob(p.blob, `${p.id}.jpg`);
			}
		} finally {
			busy = false;
			onChange();
		}
	}

	function clearTour() {
		if (busy) return;
		panoTour.length = 0;
		onChange();
	}

	// The "other side of the coin": the server reads the active cell's scene
	// hierarchy, has a lightweight model propose capture anchors, and returns
	// them. We drive the capture machinery over those anchors (set the camera,
	// render a 360, upload it), decimate + upload the proxy, render the minimaps,
	// and write the manifest — so the whole tour persists under
	// /artifacts/<cell>/tour/ for /pano to load by URL.
	async function runAutoTour({ onPhase } = {}) {
		if (busy) return;
		const cell = resolveCell();
		if (!cell) {
			onEvent({ kind: "run.error", message: "auto-tour: no active run/slot/model" });
			return;
		}
		if (modelsById.size === 0) {
			onEvent({
				kind: "run.error",
				message: "auto-tour: scene has no meshes loaded yet",
			});
			return;
		}
		busy = true;
		onChange();
		// Lock the camera so the animate loop's OrbitControls.update() doesn't fight
		// the positions we set per anchor; restore the view afterwards.
		const camSnapshot = {
			pos: camera.position.clone(),
			target: controls.target.clone(),
			userMoved: getCameraUserMoved(),
		};
		setCameraUserMoved(true);
		try {
			onPhase?.("planning anchors…");
			const planRes = await fetch(
				new URL(`${cell.base}/anchors?${cell.run}`, serverUrl).toString(),
				{ method: "POST" },
			);
			if (!planRes.ok) throw new Error(`/anchors → ${planRes.status}`);
			const plan = await planRes.json();
			const anchors = Array.isArray(plan.anchors) ? plan.anchors : [];
			if (anchors.length === 0) throw new Error("planner returned no anchors");

			await fetch(
				new URL(`${cell.base}/tour/reset?${cell.run}`, serverUrl).toString(),
				{ method: "POST" },
			);

			const panoMeta = [];
			for (let i = 0; i < anchors.length; i++) {
				const a = anchors[i];
				const pos = Array.isArray(a.position) ? a.position : [0, 0, 0];
				const id =
					typeof a.id === "string" && a.id
						? a.id
						: `anchor-${String(i).padStart(3, "0")}`;
				camera.position.set(pos[0], pos[1], pos[2]);
				// Each capture is a full 360°; forward only seeds /pano's initial view.
				const forward = [0, 0, -1];
				onPhase?.(`capturing ${i + 1}/${anchors.length}…`);
				const blob = await capturePanoBlob();
				await uploadPano(cell, id, blob);
				panoMeta.push({
					id,
					file: `${id}.jpg`,
					position: pos,
					forward,
					reason: typeof a.reason === "string" ? a.reason : undefined,
					name: typeof a.name === "string" ? a.name : undefined,
				});
			}

			// Decimate + persist the proxy from the merged scene.
			let hasProxy = false;
			onPhase?.("building proxy…");
			const merged = await buildMergedSceneGlbBuffer();
			if (merged) {
				const proxyRes = await fetch(
					new URL(`${cell.base}/tour/proxy?${cell.run}`, serverUrl).toString(),
					{
						method: "POST",
						headers: { "Content-Type": "model/gltf-binary" },
						body: merged,
					},
				);
				hasProxy = proxyRes.ok;
				if (!proxyRes.ok) {
					onEvent({
						kind: "run.error",
						message: `auto-tour proxy → ${proxyRes.status} (tour saved without it)`,
					});
				}
			}

			// Bird's-eye minimap slices, grouped by Y level. Best-effort: a failure
			// here leaves the tour fully usable, just without the minimap overlay.
			let minimaps = [];
			try {
				minimaps = await buildMinimaps(cell, panoMeta, (li, n) => {
					onPhase?.(`rendering minimap ${li + 1}/${n}…`);
				});
			} catch (e) {
				minimaps = [];
				onEvent({
					kind: "run.error",
					message: `auto-tour minimaps failed (tour saved without them): ${e.message}`,
				});
			}

			onPhase?.("writing manifest…");
			const manifest = {
				version: 1,
				proxy: hasProxy ? "proxy.glb" : null,
				planner_model: typeof plan.model === "string" ? plan.model : null,
				namer_model:
					typeof plan.namer_model === "string" ? plan.namer_model : null,
				planner_reasoning:
					typeof plan.reasoning === "string" ? plan.reasoning : null,
				panos: panoMeta,
				minimaps,
			};
			const manRes = await fetch(
				new URL(`${cell.base}/tour/manifest?${cell.run}`, serverUrl).toString(),
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(manifest),
				},
			);
			if (!manRes.ok) throw new Error(`/tour/manifest → ${manRes.status}`);
			const { tour_url } = await manRes.json();
			const absUrl = new URL(tour_url, serverUrl).toString();
			setStatus(
				`auto-tour ready · ${panoMeta.length} panos · open /pano?tour=${absUrl}`,
				"hdr",
			);
			onEvent({ kind: "run.done", message: `auto-tour persisted: ${absUrl}` });
		} catch (e) {
			onEvent({ kind: "run.error", message: `auto-tour failed: ${e.message}` });
		} finally {
			camera.position.copy(camSnapshot.pos);
			controls.target.copy(camSnapshot.target);
			setCameraUserMoved(camSnapshot.userMoved);
			controls.update();
			busy = false;
			onChange();
		}
	}

	return {
		get busy() {
			return busy;
		},
		get count() {
			return panoTour.length;
		},
		captureManual,
		saveTour,
		clearTour,
		runAutoTour,
	};
}
