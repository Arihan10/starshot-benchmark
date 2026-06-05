import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const $ = (sel, root = document) => root.querySelector(sel);

const els = {
	cat: $("#filter-cat"),
	search: $("#filter-search"),
	count: $("#asset-count"),
	list: $("#asset-list"),
	empty: $("#asset-empty"),
	axis: $("#axis"),
	keep: $("#keep"),
	orientation: $("#orientation"),
	useDims: $("#use-dims"),
	dimW: $("#dim-w"),
	dimH: $("#dim-h"),
	dimD: $("#dim-d"),
	btnBuild: $("#btn-build"),
	status: $("#status"),
	stats: $("#stats"),
	origPane: $("#orig-pane"),
	symPane: $("#sym-pane"),
};

const MAX_RENDER = 400;
const state = { assets: [], selectedId: null, busy: false };

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

// --- KTX2/Meshopt-aware loader (optimized GLBs use both) ---------------------

// Served locally by the studio (see studio.py /vendor/three mount) — fetching
// the Basis transcoder wasm from a CDN per worker is what stalled loads.
const ktx2 = new KTX2Loader()
	.setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
	.setWorkerLimit(
		Math.min(16, Math.max(4, navigator.hardwareConcurrency || 4)),
	);
let ktx2Ready = false;

function makeLoader() {
	return new GLTFLoader()
		.setKTX2Loader(ktx2)
		.setMeshoptDecoder(MeshoptDecoder);
}

// --- pane helpers -----------------------------------------------------------

function clearPane(pane) {
	for (const child of [...pane.children]) {
		if (
			!child.classList.contains("pane-label") &&
			!child.classList.contains("pane-help")
		) {
			child.remove();
		}
	}
}

// Status/spinner is drawn as an overlay ON TOP of the (permanently mounted)
// canvas. It must never remove the canvas: detaching the renderer is what made
// every load after the first fail to display.
function paneMessage(pane, text, { spinner = false, err = false } = {}) {
	let ov = pane.querySelector(".pane-overlay");
	if (!ov) {
		ov = document.createElement("div");
		ov.className = "pane-overlay";
		pane.appendChild(ov);
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

function hidePaneMessage(pane) {
	pane.querySelector(".pane-overlay")?.remove();
}

// --- three.js viewers (one per pane) ----------------------------------------

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

	const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
	camera.position.set(2, 1.5, 2.5);

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
	renderer.domElement.addEventListener("contextmenu", (e) =>
		e.preventDefault(),
	);

	const fitSize = () => {
		const w = pane.clientWidth;
		const h = pane.clientHeight;
		if (w === 0 || h === 0) return;
		renderer.setSize(w, h, false);
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
	};

	// Mount the canvas once and keep it for the pane's lifetime — overlays
	// (paneMessage) sit on top instead of replacing it.
	clearPane(pane);
	pane.appendChild(renderer.domElement);
	fitSize();
	const v = {
		renderer,
		scene,
		camera,
		controls,
		pane,
		model: null,
		frame: null,
		fitSize,
	};
	pane.tabIndex = 0;
	pane.addEventListener("keydown", (e) => {
		if ((e.key === "f" || e.key === "F") && v.frame) v.frame();
	});
	pane.addEventListener("pointerenter", () =>
		pane.focus({ preventScroll: true }),
	);
	new ResizeObserver(fitSize).observe(pane);

	renderer.setAnimationLoop(() => {
		controls.update();
		renderer.render(scene, camera);
	});
	return v;
}

// Try each url in order; resolves on the first that loads.
function loadModel(v, urls) {
	const list = Array.isArray(urls) ? urls : [urls];
	v.fitSize();
	if (v.model) {
		v.scene.remove(v.model);
		v.model = null;
	}
	const loader = makeLoader();
	const tryAt = (i) =>
		new Promise((resolve, reject) => {
			loader.load(
				list[i],
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
						const dist = maxDim * 2.2;
						v.camera.position.set(
							dist * 0.7,
							dist * 0.5,
							dist * 0.9,
						);
						v.camera.near = maxDim * 0.01;
						v.camera.far = maxDim * 100;
						v.camera.updateProjectionMatrix();
						v.controls.target.set(0, 0, 0);
						v.controls.update();
					};
					v.frame();
					hidePaneMessage(v.pane);
					resolve();
				},
				undefined,
				(err) => {
					if (i + 1 < list.length) resolve(tryAt(i + 1));
					else reject(new Error(err?.message ?? "GLB load failed"));
				},
			);
		});
	return tryAt(0);
}

const origView = makeViewer(els.origPane);
const symView = makeViewer(els.symPane);
paneMessage(els.origPane, "no asset selected");
paneMessage(els.symPane, "build to preview");

// --- asset list -------------------------------------------------------------

async function loadAssets() {
	let data;
	try {
		const r = await fetch("/library");
		if (!r.ok) throw new Error(`${r.status}`);
		data = await r.json();
	} catch (e) {
		els.empty.textContent = `failed to load library: ${e.message ?? e}`;
		return;
	}
	state.assets = data.assets ?? [];
	for (const c of data.categories ?? []) {
		const opt = document.createElement("option");
		opt.value = c;
		opt.textContent = c.toLowerCase();
		els.cat.appendChild(opt);
	}
	renderList();
}

function filteredAssets() {
	const cat = els.cat.value;
	const q = els.search.value.trim().toLowerCase();
	return state.assets.filter((a) => {
		if (cat && a.category !== cat) return false;
		if (
			q &&
			!a.description.toLowerCase().includes(q) &&
			!a.id.toLowerCase().includes(q)
		) {
			return false;
		}
		return true;
	});
}

function renderList() {
	const matches = filteredAssets();
	els.list.innerHTML = "";
	if (matches.length === 0) {
		const empty = document.createElement("div");
		empty.id = "asset-empty";
		empty.textContent = "no assets match your filters.";
		els.list.appendChild(empty);
		els.count.textContent = "0 assets";
		return;
	}
	const shown = matches.slice(0, MAX_RENDER);
	els.count.textContent =
		matches.length > shown.length
			? `showing ${shown.length} of ${matches.length} assets`
			: `${matches.length} asset${matches.length === 1 ? "" : "s"}`;
	const frag = document.createDocumentFragment();
	for (const a of shown) frag.appendChild(assetCard(a));
	els.list.appendChild(frag);
}

function assetCard(a) {
	const card = document.createElement("div");
	card.className = "asset-card";
	if (a.id === state.selectedId) card.classList.add("active");
	const name = document.createElement("div");
	name.className = "ac-name";
	name.textContent = a.description || a.id;
	const meta = document.createElement("div");
	meta.className = "ac-meta";
	meta.textContent = [a.category, a.id].filter(Boolean).join(" · ");
	card.append(name, meta);
	card.addEventListener("click", () => selectAsset(a, card));
	return card;
}

function selectAsset(a, card) {
	state.selectedId = a.id;
	for (const c of els.list.querySelectorAll(".asset-card.active"))
		c.classList.remove("active");
	card.classList.add("active");
	els.btnBuild.disabled = state.busy;
	els.stats.textContent = "";
	setStatus(`selected "${a.description || a.id}". set options and build.`);
	if (symView.model) {
		symView.scene.remove(symView.model);
		symView.model = null;
	}
	paneMessage(els.symPane, "build to preview");
	paneMessage(els.origPane, "loading…", { spinner: true });
	loadModel(origView, [
		`/library-optimized/${a.id}.glb`,
		`/library-raw/${a.id}.glb`,
	]).catch((e) =>
		paneMessage(els.origPane, `load failed: ${e.message ?? e}`, {
			err: true,
		}),
	);
}

// --- build ------------------------------------------------------------------

function dims() {
	if (!els.useDims.checked) return { width: null, height: null, depth: null };
	const w = parseFloat(els.dimW.value);
	const h = parseFloat(els.dimH.value);
	const d = parseFloat(els.dimD.value);
	const ok = [w, h, d].every((v) => Number.isFinite(v) && v > 0);
	return ok
		? { width: w, height: h, depth: d }
		: { width: null, height: null, depth: null };
}

function setBusy(busy) {
	state.busy = busy;
	els.btnBuild.disabled = busy || !state.selectedId;
}

async function build() {
	if (!state.selectedId) {
		setStatus("pick an asset first.", "err");
		return;
	}
	if (els.useDims.checked && dims().width === null) {
		setStatus(
			"target size is on but not all of W, H, D are positive numbers.",
			"err",
		);
		return;
	}
	setBusy(true);
	els.stats.textContent = "";
	paneMessage(els.symPane, "symmetrizing + optimizing…", { spinner: true });
	setStatus("building symmetric asset (mirror → rescale → optimize)…");
	const t0 = performance.now();
	const body = {
		library_id: state.selectedId,
		axis: parseInt(els.axis.value, 10),
		keep_positive: els.keep.value === "true",
		orientation: parseInt(els.orientation.value, 10),
		...dims(),
	};
	try {
		const r = await fetch("/symmetrize", {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(body),
		});
		if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
		const data = await r.json();
		await Promise.all([
			loadModel(origView, [data.original_url]),
			loadModel(symView, [data.symmetric_url]),
		]);
		const dt = ((performance.now() - t0) / 1000).toFixed(1);
		setStatus(`done (${dt}s).`, "ok");
		els.stats.textContent =
			`${data.triangles?.toLocaleString() ?? "?"} tris · ` +
			`raw ${fmtBytes(data.raw_bytes)} → optimized ${fmtBytes(data.optimized_bytes)}`;
	} catch (e) {
		paneMessage(els.symPane, "failed", { err: true });
		setStatus(`build failed: ${e.message ?? e}`, "err");
	} finally {
		setBusy(false);
	}
}

// --- wiring -----------------------------------------------------------------

els.cat.addEventListener("change", renderList);
els.search.addEventListener("input", renderList);
els.useDims.addEventListener("change", () => {
	const on = els.useDims.checked;
	for (const el of [els.dimW, els.dimH, els.dimD]) el.disabled = !on;
});
els.btnBuild.addEventListener("click", build);

loadAssets();
