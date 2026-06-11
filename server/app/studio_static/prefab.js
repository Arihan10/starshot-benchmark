// Prefab match inspector — a standalone studio page (kept entirely out of the
// main pipeline / client viewer). Pick a built scene, run the production prefab
// matcher (`prefabs.match_duplicates`) over its objects via the studio's
// `/prefab/*` endpoints, and inspect the grouping in a bboxes-only 3D view:
// every matched group gets a unique color + a big number baked onto the box
// faces; hover shows an object's id + description; runs are saved as versions you
// can switch between to compare.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const stage = document.getElementById("stage");
const sceneSel = document.getElementById("scene");
const versionSel = document.getElementById("version");
const rerunBtn = document.getElementById("rerun");
const statusEl = document.getElementById("status");
const tip = document.getElementById("tip");
const tipName = tip.querySelector(".name");
const tipDesc = tip.querySelector(".desc");

function setStatus(text, err = false) {
	statusEl.textContent = text;
	statusEl.classList.toggle("err", err);
}

// ── three.js scene ──────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1014);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.domElement.style.cssText = "width:100%;height:100%;display:block;";
stage.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const boxesRoot = new THREE.Group();
scene.add(boxesRoot);

function resize() {
	const w = stage.clientWidth || 1;
	const h = stage.clientHeight || 1;
	renderer.setSize(w, h, false);
	camera.aspect = w / h;
	camera.updateProjectionMatrix();
}
resize();
new ResizeObserver(resize).observe(stage);

(function tick() {
	controls.update();
	renderer.render(scene, camera);
	requestAnimationFrame(tick);
})();

// ── box building (number baked onto the faces, not a floating sprite) ────────
function makeFaceMaterial(colorHex, label) {
	const px = 256;
	const canvas = document.createElement("canvas");
	canvas.width = px;
	canvas.height = px;
	const ctx = canvas.getContext("2d");
	const c = new THREE.Color(colorHex);
	ctx.fillStyle = `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
	ctx.fillRect(0, 0, px, px);
	if (label) {
		ctx.font = `bold ${Math.round(px * 0.66)}px sans-serif`;
		ctx.textAlign = "center";
		ctx.textBaseline = "middle";
		ctx.lineWidth = px * 0.07;
		ctx.strokeStyle = "rgba(0,0,0,0.8)";
		ctx.strokeText(label, px / 2, px / 2 + px * 0.02);
		ctx.fillStyle = "#fff";
		ctx.fillText(label, px / 2, px / 2 + px * 0.02);
	}
	const tex = new THREE.CanvasTexture(canvas);
	tex.colorSpace = THREE.SRGBColorSpace;
	return new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.85 });
}

function buildBox(origin, dimensions, colorHex, filled, label, info) {
	const sx = Math.abs(dimensions[0]);
	const sy = Math.abs(dimensions[1]);
	const sz = Math.abs(dimensions[2]);
	if (sx === 0 || sy === 0 || sz === 0) return null;
	const cx = origin[0] + dimensions[0] / 2;
	const cy = origin[1] + dimensions[1] / 2;
	const cz = origin[2] + dimensions[2] / 2;
	const geom = new THREE.BoxGeometry(sx, sy, sz);
	const node = new THREE.Group();
	// A pickable mesh per box — visible face texture for matched objects, an
	// invisible box for singletons — so hover can identify ANY object.
	const mesh = new THREE.Mesh(
		geom,
		filled ? makeFaceMaterial(colorHex, label) : new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 }),
	);
	mesh.position.set(cx, cy, cz);
	mesh.userData.info = info;
	node.add(mesh);
	const edges = new THREE.LineSegments(
		new THREE.EdgesGeometry(geom),
		new THREE.LineBasicMaterial({ color: colorHex }),
	);
	edges.position.set(cx, cy, cz);
	node.add(edges);
	return node;
}

function colorMap(matchedGroups) {
	const grouped = new Map(); // node id -> { hex, label }
	const color = new THREE.Color();
	matchedGroups.forEach((g, i) => {
		// Golden-ratio hue spacing → distinct colors per group; the 1-based label
		// disambiguates groups whose colors come out similar.
		color.setHSL((i * 0.6180339887) % 1, 0.72, 0.55);
		const entry = { hex: color.getHex(), label: String(i + 1) };
		for (const id of g.member_ids ?? []) grouped.set(id, entry);
	});
	return grouped;
}

function render(objects, grouped) {
	for (const c of [...boxesRoot.children]) {
		boxesRoot.remove(c);
		c.traverse((o) => {
			o.geometry?.dispose?.();
			o.material?.map?.dispose?.();
			o.material?.dispose?.();
		});
	}
	for (const o of objects) {
		const g = grouped.get(o.id);
		// Matched-prefab objects get a filled group color + a big group number; the
		// rest stay a neutral wireframe so the grouping reads at a glance.
		const b = buildBox(
			o.origin, o.dimensions, g?.hex ?? 0x8891a0, g !== undefined, g?.label ?? null,
			{ id: o.id, prompt: o.prompt ?? "" },
		);
		if (b) boxesRoot.add(b);
	}
}

function frameAll() {
	const box = new THREE.Box3().setFromObject(boxesRoot);
	if (box.isEmpty()) return false;
	const center = new THREE.Vector3();
	box.getCenter(center);
	const size = new THREE.Vector3();
	box.getSize(size);
	const maxDim = Math.max(size.x, size.y, size.z) || 1;
	camera.position.copy(center).add(new THREE.Vector3(0.5, 0.6, 1).normalize().multiplyScalar(maxDim * 1.6));
	camera.near = Math.max(maxDim / 1000, 0.001);
	camera.far = Math.max(maxDim * 100, 100);
	camera.updateProjectionMatrix();
	controls.target.copy(center);
	controls.update();
	return true;
}

// ── hover → tooltip (id + description) ───────────────────────────────────────
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener("pointermove", (e) => {
	const rect = renderer.domElement.getBoundingClientRect();
	pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
	pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
	raycaster.setFromCamera(pointer, camera);
	const hit = raycaster
		.intersectObjects(boxesRoot.children, true)
		.find((h) => h.object.userData?.info);
	if (!hit) {
		tip.style.display = "none";
		return;
	}
	tipName.textContent = hit.object.userData.info.id;
	tipDesc.textContent = hit.object.userData.info.prompt || "(no description)";
	tip.style.display = "block";
	tip.style.left = `${e.clientX - rect.left + 14}px`;
	tip.style.top = `${e.clientY - rect.top + 14}px`;
});
renderer.domElement.addEventListener("pointerleave", () => {
	tip.style.display = "none";
});

// ── data + state ─────────────────────────────────────────────────────────────
let currentScene = null; // { run, slot, model }
let selectedVersion = null; // which saved version is shown (null = latest)
let framed = false;
let abortCtrl = null;

function populateVersions(list, current) {
	versionSel.innerHTML = "";
	for (const v of list) {
		const opt = document.createElement("option");
		opt.value = String(v.version);
		opt.textContent = `v${v.version} · ${v.groups} groups · ${v.in_group} grouped`;
		if (v.version === current) opt.selected = true;
		versionSel.appendChild(opt);
	}
	versionSel.style.display = list.length ? "" : "none";
}

// method "GET" loads a SAVED version (the `selectedVersion`, else latest);
// "POST" re-runs the matching fresh and saves it as a NEW version.
async function request(method) {
	if (!currentScene) return;
	abortCtrl?.abort();
	const ctrl = new AbortController();
	abortCtrl = ctrl;
	setStatus(method === "POST" ? "matching…" : "loading…");
	rerunBtn.disabled = true;
	try {
		const { run, slot, model } = currentScene;
		const vq =
			method === "GET" && selectedVersion != null
				? `&version=${encodeURIComponent(selectedVersion)}`
				: "";
		const url = `/prefab/match?run=${encodeURIComponent(run)}&slot=${encodeURIComponent(
			slot,
		)}&model=${encodeURIComponent(model)}${vq}`;
		const resp = await fetch(url, { method, signal: ctrl.signal });
		if (!resp.ok) {
			setStatus(`failed: ${resp.status} ${await resp.text()}`, true);
			return;
		}
		const data = await resp.json();
		selectedVersion = data.version ?? null;
		populateVersions(data.versions ?? [], selectedVersion);
		const groups = data.matched_groups ?? [];
		const grouped = colorMap(groups);
		render(data.objects ?? [], grouped);
		if (!framed) framed = frameAll();
		const note = selectedVersion == null ? " — none yet, click Re-run" : "";
		const vlabel = selectedVersion == null ? "" : `v${selectedVersion} · `;
		setStatus(
			`${vlabel}${groups.length} groups · ${grouped.size}/${data.total_objects} grouped · ${data.distinct_canonicals} distinct${note}`,
		);
		console.log("[prefab]", method, data);
	} catch (e) {
		if (e.name === "AbortError") return;
		setStatus(`error: ${e.message ?? e}`, true);
		console.error("[prefab]", e);
	} finally {
		if (abortCtrl === ctrl) rerunBtn.disabled = currentScene == null;
	}
}

// ── scene picker ─────────────────────────────────────────────────────────────
async function loadScenes() {
	try {
		const resp = await fetch("/prefab/scenes");
		const data = await resp.json();
		const scenes = data.scenes ?? [];
		sceneSel.innerHTML = "";
		if (!scenes.length) {
			const opt = document.createElement("option");
			opt.value = "";
			opt.textContent = "no built scenes found";
			sceneSel.appendChild(opt);
			setStatus("no built scenes under runs/ — build a scene first.");
			return;
		}
		const ph = document.createElement("option");
		ph.value = "";
		ph.textContent = "pick a scene…";
		sceneSel.appendChild(ph);
		for (const s of scenes) {
			const opt = document.createElement("option");
			opt.value = JSON.stringify(s);
			opt.textContent = `${s.run} / ${s.slot} / ${s.model}`;
			sceneSel.appendChild(opt);
		}
		setStatus("pick a scene to begin.");
	} catch (e) {
		setStatus(`failed to load scenes: ${e.message ?? e}`, true);
	}
}

sceneSel.addEventListener("change", () => {
	if (!sceneSel.value) {
		currentScene = null;
		rerunBtn.disabled = true;
		return;
	}
	currentScene = JSON.parse(sceneSel.value);
	selectedVersion = null;
	framed = false; // re-frame the camera for the new scene
	rerunBtn.disabled = false;
	request("GET"); // load this scene's latest saved version (no re-match)
});

versionSel.addEventListener("change", () => {
	selectedVersion = Number(versionSel.value);
	request("GET"); // switch version (no re-match), keep the viewpoint
});

rerunBtn.addEventListener("click", () => request("POST"));

loadScenes();
