// scene lite — browse benchmarked cells and preview each scene's web-shippable
// export: a single small, vertex-colored GLB baked server-side from the cell's
// raw meshes (textures baked into per-vertex colors, no texture maps). See
// /scene-lite.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const $ = (sel, root = document) => root.querySelector(sel);

const els = {
	list: $("#scene-list"),
	empty: $("#scene-empty"),
	refresh: $("#refresh"),
	status: $("#status"),
	stats: $("#stats"),
	selTitle: $("#sel-title"),
	selSub: $("#sel-sub"),
	download: $("#download"),
	rebuild: $("#btn-rebuild"),
	pane: $("#scene-pane"),
};

const state = { scenes: [], selected: null, busy: false };

function setStatus(text, cls = "") {
	els.status.textContent = text;
	els.status.className = cls;
}

const fmtBytes = (n) =>
	n == null
		? "?"
		: n < 1048576
			? `${(n / 1024).toFixed(0)}KB`
			: `${(n / 1048576).toFixed(2)}MB`;

const cellKey = (c) => `${c.run}/${c.slot}/${c.model}`;

// --- KTX2/Meshopt-aware loader (the lite scene uses both) --------------------
// Served locally by the studio (see studio.py /vendor/three mount) — fetching
// the Basis transcoder wasm from a CDN per worker is what stalls loads.
const ktx2 = new KTX2Loader()
	.setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
	.setWorkerLimit(Math.min(16, Math.max(4, navigator.hardwareConcurrency || 4)));
let ktx2Ready = false;

function makeLoader() {
	return new GLTFLoader().setKTX2Loader(ktx2).setMeshoptDecoder(MeshoptDecoder);
}

// --- pane helpers (overlay sits ON TOP of the permanent canvas) -------------

function paneMessage(text, { spinner = false, err = false } = {}) {
	let ov = els.pane.querySelector(".pane-overlay");
	if (!ov) {
		ov = document.createElement("div");
		ov.className = "pane-overlay";
		els.pane.appendChild(ov);
	}
	ov.innerHTML = "";
	if (spinner) {
		const sp = document.createElement("div");
		sp.className = "spinner";
		ov.appendChild(sp);
	}
	const span = document.createElement("span");
	span.className = "empty";
	span.style.marginTop = spinner ? "10px" : "0";
	if (err) span.style.color = "var(--red)";
	span.textContent = text;
	ov.appendChild(span);
}

function hidePaneMessage() {
	els.pane.querySelector(".pane-overlay")?.remove();
}

// --- three.js viewer --------------------------------------------------------

function makeViewer(pane) {
	const renderer = new THREE.WebGLRenderer({ antialias: true });
	renderer.setPixelRatio(window.devicePixelRatio);
	if (!ktx2Ready) {
		ktx2.detectSupport(renderer);
		ktx2Ready = true;
	}

	const scene = new THREE.Scene();
	scene.background = new THREE.Color(0x0c0d10);
	scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 1.0));
	const dir = new THREE.DirectionalLight(0xffffff, 1.1);
	dir.position.set(3, 5, 4);
	scene.add(dir);
	const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
	dir2.position.set(-3, 2, -2);
	scene.add(dir2);

	const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
	camera.position.set(4, 3, 5);

	const controls = new OrbitControls(camera, renderer.domElement);
	controls.enableDamping = true;
	controls.dampingFactor = 0.12;
	controls.screenSpacePanning = true;
	controls.zoomToCursor = true;
	controls.mouseButtons = {
		LEFT: THREE.MOUSE.ROTATE,
		MIDDLE: THREE.MOUSE.PAN,
		RIGHT: THREE.MOUSE.PAN,
	};
	renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

	const fitSize = () => {
		const w = pane.clientWidth;
		const h = pane.clientHeight;
		if (w === 0 || h === 0) return;
		renderer.setSize(w, h, false);
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
	};

	pane.appendChild(renderer.domElement);
	fitSize();
	const v = { renderer, scene, camera, controls, pane, model: null, frame: null, fitSize };
	pane.tabIndex = 0;
	pane.addEventListener("keydown", (e) => {
		if ((e.key === "f" || e.key === "F") && v.frame) v.frame();
	});
	pane.addEventListener("pointerenter", () => pane.focus({ preventScroll: true }));
	new ResizeObserver(fitSize).observe(pane);

	renderer.setAnimationLoop(() => {
		controls.update();
		renderer.render(scene, camera);
	});
	return v;
}

function loadModel(v, url) {
	v.fitSize();
	if (v.model) {
		v.scene.remove(v.model);
		v.model = null;
	}
	const loader = makeLoader();
	return new Promise((resolve, reject) => {
		loader.load(
			url,
			(gltf) => {
				const root = gltf.scene;
				const box = new THREE.Box3().setFromObject(root);
				const size = box.getSize(new THREE.Vector3());
				const center = box.getCenter(new THREE.Vector3());
				root.position.sub(center);
				v.scene.add(root);
				v.model = root;
				const maxDim = Math.max(size.x, size.y, size.z) || 1;
				v.frame = () => {
					const dist = maxDim * 1.6;
					v.camera.position.set(dist * 0.7, dist * 0.5, dist * 0.9);
					v.camera.near = maxDim * 0.005;
					v.camera.far = maxDim * 100;
					v.camera.updateProjectionMatrix();
					v.controls.target.set(0, 0, 0);
					v.controls.update();
				};
				v.frame();
				hidePaneMessage();
				resolve();
			},
			undefined,
			(err) => reject(new Error(err?.message ?? "GLB load failed")),
		);
	});
}

const view = makeViewer(els.pane);
paneMessage("no scene loaded");

// --- cell list --------------------------------------------------------------

async function fetchScenes() {
	let data;
	try {
		const r = await fetch("/scenes");
		if (!r.ok) throw new Error(`${r.status}`);
		data = await r.json();
	} catch (e) {
		els.list.innerHTML = "";
		const empty = document.createElement("div");
		empty.id = "scene-empty";
		empty.textContent = `failed to load cells: ${e.message ?? e}`;
		els.list.appendChild(empty);
		return;
	}
	state.scenes = data.scenes ?? [];
	renderList();
}

function renderList() {
	els.list.innerHTML = "";
	if (state.scenes.length === 0) {
		const empty = document.createElement("div");
		empty.id = "scene-empty";
		empty.textContent = "no benchmarked cells yet — run the pipeline first.";
		els.list.appendChild(empty);
		return;
	}
	const frag = document.createDocumentFragment();
	for (const c of state.scenes) frag.appendChild(sceneCard(c));
	els.list.appendChild(frag);
}

function sceneCard(c) {
	const card = document.createElement("div");
	card.className = "scene-card";
	if (state.selected && cellKey(state.selected) === cellKey(c)) card.classList.add("active");
	const head = document.createElement("div");
	head.className = "sc-head";
	const title = document.createElement("span");
	title.className = "sc-title";
	title.textContent = `${c.slot} · ${c.model}`;
	const badge = document.createElement("span");
	badge.className = `sc-badge${c.built ? " built" : ""}`;
	badge.textContent = c.built ? fmtBytes(c.bytes) : "not built";
	head.append(title, badge);
	const meta = document.createElement("div");
	meta.className = "sc-meta";
	meta.textContent = `${c.run} · ${c.objects} obj`;
	card.append(head, meta);
	card.addEventListener("click", () => selectCell(c));
	return card;
}

// --- build + load -----------------------------------------------------------

function setBusy(busy) {
	state.busy = busy;
	els.rebuild.disabled = busy || !state.selected;
}

async function selectCell(c) {
	if (state.busy) return;
	state.selected = c;
	renderList();
	els.selTitle.textContent = `${c.slot} · ${c.model}`;
	els.selSub.textContent = `${c.run} · ${c.objects} objects`;
	els.download.hidden = true;
	await buildAndLoad(c, false);
}

async function buildAndLoad(c, force) {
	setBusy(true);
	els.stats.textContent = "";
	paneMessage(force ? "rebuilding scene…" : "baking scene…", { spinner: true });
	setStatus(`${force ? "rebuilding" : "baking"} vertex-colored scene for ${cellKey(c)}…`);
	const t0 = performance.now();
	const path =
		`/scene-lite/${encodeURIComponent(c.run)}/${encodeURIComponent(c.slot)}/${encodeURIComponent(c.model)}` +
		(force ? "?force=1" : "");
	try {
		const r = await fetch(path, { method: "POST" });
		if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
		const data = await r.json();
		const bust = `${data.url}?v=${Date.now()}`;
		await loadModel(view, bust);
		const dt = ((performance.now() - t0) / 1000).toFixed(1);
		setStatus(`done (${dt}s).`, "ok");
		const tris = data.outTris != null ? `${(data.outTris / 1000).toFixed(0)}k tris · ` : "";
		const skipped = data.skipped ? ` · ${data.skipped} skipped` : "";
		els.stats.textContent = `${data.objects ?? "?"} objects${skipped} → ${tris}${fmtBytes(data.bytes)}`;
		els.download.hidden = false;
		els.download.href = bust;
		els.download.download = `${c.run}-${c.slot}-${c.model}.glb`;
		// Reflect the freshly built size in the sidebar card.
		const entry = state.scenes.find((s) => cellKey(s) === cellKey(c));
		if (entry) {
			entry.built = true;
			entry.bytes = data.bytes;
			renderList();
		}
	} catch (e) {
		paneMessage("build failed", { err: true });
		setStatus(`build failed: ${e.message ?? e}`, "err");
	} finally {
		setBusy(false);
	}
}

// --- wiring -----------------------------------------------------------------

els.refresh.addEventListener("click", fetchScenes);
els.rebuild.addEventListener("click", () => {
	if (state.selected) buildAndLoad(state.selected, true);
});

fetchScenes();
