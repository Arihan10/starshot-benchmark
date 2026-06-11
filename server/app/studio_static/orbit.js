// orbit — a combined dollhouse + interior walkthrough. The OVERVIEW orbits the
// cell's vertex-colored lite scene (the /lite export) with a free-pan camera;
// stepping INSIDE drops into the /pano projection walkthrough (panos projected
// onto the low-poly proxy). Both live in the SAME world frame — the lite scene's
// objects are world-placed, the proxy is baked from world matrices, and pano
// positions are captured camera positions — so a "you are here" marker dropped
// at the interior camera maps onto the dollhouse with no coordinate fixup.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const $ = (sel) => document.querySelector(sel);
const enc = encodeURIComponent;
const v3 = (a) => new THREE.Vector3().fromArray(a);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

const els = {
	stage: $("#stage"),
	list: $("#cell-list"),
	empty: $("#empty"),
	hud: $("#hud"),
	modeLabel: $("#mode-label"),
	travelFade: $("#travel-fade"),
	overlay: $("#overlay"),
	overlayMsg: $("#overlay-msg"),
	overlaySpin: $("#overlay-spin"),
	refresh: $("#refresh"),
	btnEnter: $("#btn-enter"),
	btnExit: $("#btn-exit"),
	btnLocate: $("#btn-locate"),
};

// --- renderer / scene / camera ----------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.NoToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
els.stage.appendChild(renderer.domElement);
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0d10);
scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 1.0));
const dir1 = new THREE.DirectionalLight(0xffffff, 1.1);
dir1.position.set(3, 5, 4);
const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
dir2.position.set(-3, 2, -2);
scene.add(dir1, dir2);

const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 2000);
camera.position.set(4, 3, 5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;
controls.zoomToCursor = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;
controls.mouseButtons = {
	LEFT: THREE.MOUSE.ROTATE,
	MIDDLE: THREE.MOUSE.PAN,
	RIGHT: THREE.MOUSE.PAN,
};
controls.enabled = false;

let autoRotateTimer = null;
controls.addEventListener("start", () => {
	if (mode !== "overview") return;
	controls.autoRotate = false;
	clearTimeout(autoRotateTimer);
});
controls.addEventListener("end", () => {
	if (mode !== "overview") return;
	clearTimeout(autoRotateTimer);
	autoRotateTimer = setTimeout(() => {
		if (mode === "overview") controls.autoRotate = true;
	}, 2500);
});

// --- loaders ----------------------------------------------------------------

const ktx2 = new KTX2Loader()
	.setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
	.setWorkerLimit(Math.min(16, Math.max(4, navigator.hardwareConcurrency || 4)));
ktx2.detectSupport(renderer);

function loadGLB(url) {
	const loader = new GLTFLoader().setKTX2Loader(ktx2).setMeshoptDecoder(MeshoptDecoder);
	return loader.loadAsync(url).then((gltf) => gltf.scene);
}

const textureLoader = new THREE.TextureLoader();
function prepPanoTexture(tex) {
	tex.generateMipmaps = false;
	tex.minFilter = THREE.LinearFilter;
	tex.magFilter = THREE.LinearFilter;
	tex.wrapS = THREE.RepeatWrapping;
	return tex;
}
const loadTextureUrl = (url) =>
	textureLoader.loadAsync(url).then(prepPanoTexture);

// --- interior: backdrop spheres + projection material -----------------------
// Lifted from the /pano walkthrough: the captured equirects project onto the
// proxy (view-dependent texture mapping → real parallax between anchors), with a
// camera-following backdrop sphere filling sky / whatever the proxy misses.

const SPHERE_RADIUS = 60;

function makePanoMaterial() {
	return new THREE.ShaderMaterial({
		uniforms: { map: { value: null }, opacity: { value: 1.0 } },
		vertexShader: /* glsl */ `
			varying vec3 vDir;
			void main() {
				vDir = position;
				gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
			}
		`,
		fragmentShader: /* glsl */ `
			uniform sampler2D map;
			uniform float opacity;
			varying vec3 vDir;
			void main() {
				vec3 d = normalize(vDir);
				vec2 uv = vec2(
					atan(d.z, d.x) / 6.28318530718 + 0.5,
					asin(clamp(d.y, -1.0, 1.0)) / 3.14159265359 + 0.5
				);
				gl_FragColor = vec4(texture2D(map, uv).rgb, opacity);
			}
		`,
		side: THREE.BackSide,
		transparent: true,
		depthWrite: false,
		depthTest: false,
	});
}

const sphereGeo = new THREE.SphereGeometry(SPHERE_RADIUS, 64, 32);
const sphereA = new THREE.Mesh(sphereGeo, makePanoMaterial());
const sphereB = new THREE.Mesh(sphereGeo, makePanoMaterial());
sphereA.renderOrder = 0;
sphereB.renderOrder = 1;
sphereA.visible = false;
sphereB.visible = false;
scene.add(sphereA, sphereB);

const PROJ_K = 4;
const DUMMY_TEX = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, THREE.RGBAFormat);
DUMMY_TEX.needsUpdate = true;

function makeProjectionMaterial() {
	return new THREE.ShaderMaterial({
		uniforms: {
			uTex: { value: Array.from({ length: PROJ_K }, () => DUMMY_TEX) },
			uCenter: { value: Array.from({ length: PROJ_K }, () => new THREE.Vector3()) },
			uWeight: { value: new Float32Array(PROJ_K) },
			uCount: { value: 0 },
		},
		side: THREE.DoubleSide,
		vertexShader: /* glsl */ `
			varying vec3 vWorldPos;
			varying vec3 vWorldNormal;
			void main() {
				vec4 wp = modelMatrix * vec4(position, 1.0);
				vWorldPos = wp.xyz;
				vWorldNormal = mat3(modelMatrix) * normal;
				gl_Position = projectionMatrix * viewMatrix * wp;
			}
		`,
		fragmentShader: /* glsl */ `
			uniform sampler2D uTex[${PROJ_K}];
			uniform vec3 uCenter[${PROJ_K}];
			uniform float uWeight[${PROJ_K}];
			uniform int uCount;
			varying vec3 vWorldPos;
			varying vec3 vWorldNormal;
			const float TAU = 6.28318530718;
			const float PI = 3.14159265359;

			vec2 dirToEquirect(vec3 d) {
				return vec2(atan(d.z, d.x) / TAU + 0.5, asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5);
			}
			vec3 accumOne(int i, sampler2D tex, vec3 n, inout float wsum) {
				if (i >= uCount) return vec3(0.0);
				vec3 dir = normalize(vWorldPos - uCenter[i]);
				float face = clamp(dot(n, -dir), 0.0, 1.0);
				float w = uWeight[i] * (0.2 + 0.8 * face);
				wsum += w;
				return w * texture2D(tex, dirToEquirect(dir)).rgb;
			}
			void main() {
				vec3 n = normalize(vWorldNormal);
				float wsum = 0.0;
				vec3 col = vec3(0.0);
				col += accumOne(0, uTex[0], n, wsum);
				col += accumOne(1, uTex[1], n, wsum);
				col += accumOne(2, uTex[2], n, wsum);
				col += accumOne(3, uTex[3], n, wsum);
				gl_FragColor = vec4(col / max(wsum, 1e-4), 1.0);
			}
		`,
	});
}

const projMaterial = makeProjectionMaterial();
const polyMaterial = new THREE.MeshStandardMaterial({
	color: 0x9aa7b4,
	roughness: 0.9,
	metalness: 0.0,
	flatShading: true,
	side: THREE.DoubleSide,
});

let backdropRadius = SPHERE_RADIUS;

function setupProjection(root) {
	root.traverse((o) => {
		if (!o.isMesh || !o.geometry) return;
		o.geometry.computeVertexNormals();
		o.material = projMaterial;
		o.frustumCulled = false;
	});
	const box = new THREE.Box3().setFromObject(root);
	const sph = box.getBoundingSphere(new THREE.Sphere());
	backdropRadius = Math.max(80, sph.radius * 4);
	sphereA.scale.setScalar(backdropRadius / SPHERE_RADIUS);
	sphereA.renderOrder = -1;
	sphereA.material.uniforms.opacity.value = 1;
	sphereA.material.depthTest = true; // let the opaque proxy occlude the backdrop
}

const _camDist2 = [];
function updateProjection() {
	if (panos.length === 0) return;
	const u = projMaterial.uniforms;
	const cam = camera.position;
	_camDist2.length = panos.length;
	for (let i = 0; i < panos.length; i++) {
		const p = panos[i].position;
		const dx = cam.x - p[0];
		const dy = cam.y - p[1];
		const dz = cam.z - p[2];
		_camDist2[i] = dx * dx + dy * dy + dz * dz;
	}
	const order = panos.map((_, i) => i).sort((a, b) => _camDist2[a] - _camDist2[b]);
	const K = Math.min(PROJ_K, panos.length);
	let wsum = 0;
	const w = [];
	for (let k = 0; k < K; k++) {
		const ww = 1 / (_camDist2[order[k]] + 0.25);
		w.push(ww);
		wsum += ww;
	}
	for (let k = 0; k < PROJ_K; k++) {
		if (k < K) {
			const idx = order[k];
			u.uTex.value[k] = panos[idx].texture;
			u.uCenter.value[k].fromArray(panos[idx].position);
			u.uWeight.value[k] = w[k] / wsum;
		} else {
			u.uTex.value[k] = DUMMY_TEX;
			u.uWeight.value[k] = 0;
		}
	}
	u.uCount.value = K;
	sphereA.material.uniforms.map.value = panos[order[0]].texture;
	sphereA.position.copy(cam);
}

// --- markers (interior travel hotspots, overview entry points, you-are-here) -

const HOTSPOT_FLOOR_DROP = 1.3;
const HOTSPOT_REACH = 30;
const HOTSPOT_MAX_VISIBLE = 10;
const HOTSPOT_MAX_OCCLUDED = 6;
const HOTSPOT_OCCLUDE_EPS = 0.2;
const HOTSPOT_TARGET_PX = 24;
const ENTRY_TARGET_PX = 12; // overview entry discs render smaller than interior hotspots
const AUTO_AIM_PX = 42;
const ENTRY_AIM_PX = 26; // tighter pick/hover radius to match the smaller entry discs
const HOTSPOT_BASE_RADIUS = 0.16;
const CAPTURE_EYE_HEIGHT = 1.6; // panos are shot at eye height; the floor sits this far below
const PEEK_ROTATE_SPEED = 0.5; // rad/s the dollhouse spins while locating (a slow 360 scan)

const hotspotGroup = new THREE.Group();
const entryGroup = new THREE.Group();
scene.add(hotspotGroup, entryGroup);

function makeDisc(targetIndex, color, ringColor) {
	const group = new THREE.Group();
	const disc = new THREE.Mesh(
		new THREE.CircleGeometry(HOTSPOT_BASE_RADIUS, 40),
		new THREE.MeshBasicMaterial({
			color,
			transparent: true,
			opacity: 0.55,
			side: THREE.DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	const ring = new THREE.Mesh(
		new THREE.RingGeometry(HOTSPOT_BASE_RADIUS * 1.38, HOTSPOT_BASE_RADIUS * 1.69, 48),
		new THREE.MeshBasicMaterial({
			color: ringColor,
			transparent: true,
			opacity: 0.9,
			side: THREE.DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	disc.rotation.x = -Math.PI / 2;
	ring.rotation.x = -Math.PI / 2;
	disc.renderOrder = 2;
	ring.renderOrder = 2;
	group.add(disc, ring);
	group.renderOrder = 2;
	group.userData.targetIndex = targetIndex;
	return group;
}

const _occluder = new THREE.Raycaster();
function anchorOccluded(fromPos, toPos) {
	if (!proxyGroup) return false;
	const from = v3(fromPos);
	const d = v3(toPos).sub(from);
	const dist = d.length();
	if (dist < 1e-3) return false;
	d.divideScalar(dist);
	_occluder.set(from, d);
	_occluder.near = HOTSPOT_OCCLUDE_EPS;
	_occluder.far = dist - HOTSPOT_OCCLUDE_EPS;
	if (_occluder.far <= _occluder.near) return false;
	return _occluder.intersectObject(proxyGroup, true).length > 0;
}

function neighborsByDistance() {
	const cur = v3(panos[currentIndex].position);
	const out = [];
	for (let i = 0; i < panos.length; i++) {
		if (i === currentIndex) continue;
		const d2 = cur.distanceToSquared(v3(panos[i].position));
		if (projectionMode && d2 > HOTSPOT_REACH * HOTSPOT_REACH) continue;
		out.push([i, d2]);
	}
	out.sort((a, b) => a[1] - b[1]);
	return out.map((o) => o[0]);
}

function rebuildHotspots() {
	hotspotGroup.clear();
	if (currentIndex < 0) return;
	const cur = panos[currentIndex];
	let nVisible = 0;
	let nOccluded = 0;
	for (const i of neighborsByDistance()) {
		const occluded = anchorOccluded(cur.position, panos[i].position);
		if (occluded) {
			if (nOccluded >= HOTSPOT_MAX_OCCLUDED) continue;
			nOccluded++;
		} else {
			if (nVisible >= HOTSPOT_MAX_VISIBLE) continue;
			nVisible++;
		}
		const color = occluded ? 0xe0c271 : 0xffffff;
		const ringColor = occluded ? 0xe0c271 : 0x9ad4ff;
		const spot = makeDisc(i, color, ringColor);
		spot.userData.occluded = occluded;
		spot.position.fromArray(panos[i].position);
		spot.position.y -= HOTSPOT_FLOOR_DROP;
		hotspotGroup.add(spot);
		if (nVisible >= HOTSPOT_MAX_VISIBLE && nOccluded >= HOTSPOT_MAX_OCCLUDED) break;
	}
}

function buildEntryMarkers() {
	entryGroup.clear();
	for (let i = 0; i < panos.length; i++) {
		const spot = makeDisc(i, 0x9ad4ff, 0x4a8fd8);
		spot.position.fromArray(panos[i].position);
		spot.position.y -= HOTSPOT_FLOOR_DROP;
		entryGroup.add(spot);
	}
}

function hotspotScaleForDistance(d, targetPx = HOTSPOT_TARGET_PX) {
	const h = els.stage.clientHeight || 1;
	const worldRadius = (targetPx * 2 * d * Math.tan((camera.fov * Math.PI) / 360)) / h;
	return THREE.MathUtils.clamp(worldRadius / HOTSPOT_BASE_RADIUS, 0.15, 14);
}

const _aimWorld = new THREE.Vector3();
function pickByScreen(ev, group, maxPx) {
	const rect = renderer.domElement.getBoundingClientRect();
	const cx = ev.clientX - rect.left;
	const cy = ev.clientY - rect.top;
	let best = null;
	let bestPx = maxPx;
	for (const spot of group.children) {
		spot.getWorldPosition(_aimWorld).project(camera);
		if (_aimWorld.z > 1) continue;
		const sx = (_aimWorld.x * 0.5 + 0.5) * rect.width;
		const sy = (-_aimWorld.y * 0.5 + 0.5) * rect.height;
		const px = Math.hypot(sx - cx, sy - cy);
		if (px < bestPx) {
			bestPx = px;
			best = spot;
		}
	}
	return best;
}

// You-are-here: a red pin (eye-height sphere + floor ring + connector) that
// draws over everything (depthTest off) so it's never hidden by the dollhouse.
const youMarker = new THREE.Group();
youMarker.visible = false;
youMarker.renderOrder = 999;
const youSphere = new THREE.Mesh(
	new THREE.SphereGeometry(1, 24, 16),
	new THREE.MeshBasicMaterial({ color: 0xff3030, depthTest: false, depthWrite: false }),
);
const youRing = new THREE.Mesh(
	new THREE.RingGeometry(1.4, 1.9, 40),
	new THREE.MeshBasicMaterial({
		color: 0xff3030,
		transparent: true,
		opacity: 0.85,
		side: THREE.DoubleSide,
		depthTest: false,
		depthWrite: false,
	}),
);
youRing.rotation.x = -Math.PI / 2;
const youLine = new THREE.Line(
	new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
	new THREE.LineBasicMaterial({ color: 0xff3030, transparent: true, opacity: 0.8, depthTest: false }),
);
for (const m of [youSphere, youRing, youLine]) m.renderOrder = 999;
youMarker.add(youSphere, youRing, youLine);
scene.add(youMarker);

function sizeYouMarker() {
	const r = Math.max(0.05, sceneMaxDim * 0.014);
	youSphere.geometry.dispose();
	youSphere.geometry = new THREE.SphereGeometry(r, 24, 16);
	youRing.geometry.dispose();
	youRing.geometry = new THREE.RingGeometry(r * 1.6, r * 2.2, 40);
}

function positionYouMarker(p) {
	// Floor directly beneath the user (panos sit at eye height), not the global
	// scene minimum — so the base lands on the level you're standing on.
	const floorY = p.y - CAPTURE_EYE_HEIGHT;
	youSphere.position.copy(p);
	youRing.position.set(p.x, floorY, p.z);
	youLine.geometry.setFromPoints([new THREE.Vector3(p.x, floorY, p.z), p.clone()]);
}

// --- tour / cell state ------------------------------------------------------

let panos = []; // { id, position, forward, texture }
let currentIndex = -1;
let projectionMode = false;
let liteRoot = null;
let proxyGroup = null;
let sharedOverview = false; // no lite export: the proxy doubles as the dollhouse

let mode = "empty"; // empty | loading | overview | interior | peek | transition
let sceneCenter = new THREE.Vector3();
let sceneMaxDim = 1;
const browsePose = { pos: new THREE.Vector3() };

// --- view toggles (which geometry each mode shows) --------------------------

function reskinProxy(mat) {
	if (!proxyGroup) return;
	proxyGroup.traverse((o) => {
		if (o.isMesh) o.material = mat;
	});
}

function setOverviewView() {
	if (liteRoot) liteRoot.visible = true;
	if (proxyGroup) {
		if (sharedOverview) {
			reskinProxy(polyMaterial);
			proxyGroup.visible = true;
		} else {
			proxyGroup.visible = false;
		}
	}
	sphereA.visible = false;
	sphereB.visible = false;
	hotspotGroup.visible = false;
	entryGroup.visible = true;
	youMarker.visible = false;
}

function setInteriorView() {
	if (liteRoot) liteRoot.visible = false;
	if (proxyGroup) {
		if (sharedOverview) reskinProxy(projMaterial);
		proxyGroup.visible = projectionMode;
	}
	sphereA.visible = true;
	hotspotGroup.visible = true;
	entryGroup.visible = false;
	youMarker.visible = false;
}

function setPeekView() {
	if (liteRoot) liteRoot.visible = true;
	if (proxyGroup) {
		if (sharedOverview) {
			reskinProxy(polyMaterial);
			proxyGroup.visible = true;
		} else {
			proxyGroup.visible = false;
		}
	}
	sphereA.visible = false;
	sphereB.visible = false;
	hotspotGroup.visible = false;
	entryGroup.visible = false;
	youMarker.visible = true;
}

// --- camera flight (mode changes: slerp orientation + lerp position) ---------

// A camera (not a bare Object3D) so lookAt orients -Z toward the target, matching
// how the real camera faces — otherwise a static pose ends up looking backwards.
const _dummy = new THREE.PerspectiveCamera();
let transition = null;

function startFly(toPos, lookTarget, dur, { onMid, onEnd } = {}) {
	_dummy.up.copy(camera.up);
	_dummy.position.copy(toPos);
	_dummy.lookAt(lookTarget);
	_dummy.updateMatrixWorld();
	transition = {
		fromPos: camera.position.clone(),
		toPos: toPos.clone(),
		fromQuat: camera.quaternion.clone(),
		toQuat: _dummy.quaternion.clone(),
		start: performance.now(),
		dur,
		onMid,
		onEnd,
		midDone: false,
	};
	mode = "transition";
	controls.enabled = false;
	controls.autoRotate = false;
	hoveredEntryIndex = -1;
	hoveredTargetIndex = -1;
	els.stage.classList.remove("hotspot", "grabbing");
	updateModeUI();
}

// transition fade: blur + dip to bg, peaking mid-move, to mask the lite/proxy swap
function setTravelMask(t) {
	const m = Math.sin(Math.PI * THREE.MathUtils.clamp(t, 0, 1));
	renderer.domElement.style.filter = m > 0.002 ? `blur(${(m * 7).toFixed(2)}px)` : "none";
	els.travelFade.style.opacity = (m * 0.5).toFixed(3);
}
function clearTravelMask() {
	renderer.domElement.style.filter = "none";
	els.travelFade.style.opacity = "0";
}

// --- look controls (interior: lon/lat drag + fov zoom) ----------------------

let lon = 0;
let lat = 0;
let dragging = false;
let dragMoved = 0;
let downX = 0;
let downY = 0;
let downLon = 0;
let downLat = 0;

function setLookFromForward(f) {
	const v = v3(f).normalize();
	lon = Math.atan2(v.z, v.x);
	lat = Math.asin(THREE.MathUtils.clamp(v.y, -1, 1));
}
function lookTargetFrom(pos, lo, la) {
	return pos
		.clone()
		.add(new THREE.Vector3(Math.cos(la) * Math.cos(lo), Math.sin(la), Math.cos(la) * Math.sin(lo)));
}
function applyLook() {
	lat = THREE.MathUtils.clamp(lat, -1.55, 1.55);
	const t = lookTargetFrom(camera.position, lon, lat);
	camera.lookAt(t);
}

// --- travel between anchors (interior) --------------------------------------

let interiorBusy = false;
let glide = null;

function travelTo(index) {
	if (interiorBusy || index === currentIndex || !panos[index]) return;
	hoveredTargetIndex = -1;
	interiorBusy = true;
	hotspotGroup.visible = false;

	if (projectionMode) {
		// Physically glide through world space; the projection re-blends live, so
		// geometry + textures interpolate with real parallax.
		glide = {
			fromPos: camera.position.clone(),
			toPos: v3(panos[index].position),
			start: performance.now(),
			dur: 900,
			index,
		};
		return;
	}

	// Sphere mode (no proxy): crossfade the backdrop to the next pano while
	// drifting the camera onto its world position so the marker stays truthful.
	const target = panos[index];
	sphereB.material.uniforms.map.value = target.texture;
	sphereB.material.uniforms.opacity.value = 0;
	sphereB.visible = true;
	const fromPos = camera.position.clone();
	const toPos = v3(target.position);
	const start = performance.now();
	const dur = 700;
	const step = (now) => {
		const t = Math.min(1, (now - start) / dur);
		const e = easeInOut(t);
		sphereB.material.uniforms.opacity.value = e;
		camera.position.lerpVectors(fromPos, toPos, e);
		setTravelMask(t);
		if (t < 1) {
			requestAnimationFrame(step);
			return;
		}
		sphereA.material.uniforms.map.value = target.texture;
		sphereA.material.uniforms.opacity.value = 1;
		sphereB.visible = false;
		clearTravelMask();
		interiorBusy = false;
		hotspotGroup.visible = true;
		activate(index);
	};
	requestAnimationFrame(step);
}

function activate(index) {
	currentIndex = index;
	if (!projectionMode) {
		sphereA.material.uniforms.map.value = panos[index].texture;
		sphereA.material.uniforms.opacity.value = 1;
	}
	rebuildHotspots();
	setHud(`<b>${panos[index].id}</b> · ${index + 1}/${panos.length} · drag to look`);
}

// --- mode transitions -------------------------------------------------------

function nearestPanoTo(point) {
	let best = 0;
	let bestD = Infinity;
	for (let i = 0; i < panos.length; i++) {
		const d = point.distanceToSquared(v3(panos[i].position));
		if (d < bestD) {
			bestD = d;
			best = i;
		}
	}
	return best;
}

function currentUserWorldPos() {
	return projectionMode ? camera.position.clone() : v3(panos[currentIndex].position);
}

function enter(index) {
	if (mode !== "overview" || panos.length === 0) return;
	if (index == null) index = nearestPanoTo(controls.target);
	const p = panos[index];
	const fwd = p.forward && p.forward.length ? p.forward : [0, 0, 1];
	const toPos = v3(p.position);
	startFly(toPos, toPos.clone().add(v3(fwd)), 1100, {
		onMid: () => {
			setInteriorView();
			camera.fov = 75;
			camera.updateProjectionMatrix();
		},
		onEnd: () => {
			mode = "interior";
			setLookFromForward(fwd);
			activate(index);
			updateModeUI();
		},
	});
}

function exit() {
	if (mode !== "interior" || interiorBusy) return;
	startFly(browsePose.pos.clone(), sceneCenter.clone(), 1000, {
		onMid: () => {
			setOverviewView();
			camera.fov = 55;
			camera.updateProjectionMatrix();
		},
		onEnd: () => {
			mode = "overview";
			controls.target.copy(sceneCenter);
			camera.position.copy(browsePose.pos);
			controls.enabled = true;
			controls.update();
			controls.autoRotate = true;
			updateModeUI();
		},
	});
}

let savedInterior = null;
let peekHeld = false;

function peekStart() {
	if (mode !== "interior" || interiorBusy) return;
	savedInterior = { pos: camera.position.clone(), lon, lat, index: currentIndex, fov: camera.fov };
	const userPos = currentUserWorldPos();
	positionYouMarker(userPos);
	const flat = userPos.clone().sub(sceneCenter);
	flat.y = 0;
	if (flat.lengthSq() < 1e-6) flat.set(0, 0, 1);
	flat.normalize();
	const toPos = sceneCenter.clone().addScaledVector(flat, sceneMaxDim * 1.5);
	toPos.y += sceneMaxDim * 0.6;
	startFly(toPos, sceneCenter.clone(), 850, {
		onMid: () => {
			setPeekView();
			camera.fov = 55;
			camera.updateProjectionMatrix();
		},
		onEnd: () => {
			mode = "peek";
			updateModeUI();
			if (!peekHeld) peekEnd();
		},
	});
}

function peekEnd() {
	if (mode !== "peek" || !savedInterior) return;
	const s = savedInterior;
	startFly(s.pos.clone(), lookTargetFrom(s.pos, s.lon, s.lat), 800, {
		onMid: () => {
			setInteriorView();
			camera.fov = s.fov;
			camera.updateProjectionMatrix();
		},
		onEnd: () => {
			mode = "interior";
			lon = s.lon;
			lat = s.lat;
			currentIndex = s.index;
			activate(s.index);
			updateModeUI();
		},
	});
}

// --- HUD + mode UI ----------------------------------------------------------

const OVERVIEW_HELP =
	"drag to orbit · right-drag to pan · scroll to zoom · click a marker or <b>enter interior</b>";

function setHud(html) {
	els.hud.hidden = !html;
	els.hud.innerHTML = html ?? "";
}

function updateModeUI() {
	// Leave the controls untouched mid-flight so the held locate button doesn't
	// flicker away during the peek transition.
	if (mode === "transition") return;
	els.btnEnter.hidden = mode !== "overview";
	els.btnExit.hidden = mode !== "interior";
	els.btnLocate.hidden = !(mode === "interior" || mode === "peek");
	els.btnEnter.disabled = panos.length === 0;
	els.btnExit.disabled = interiorBusy;
	if (mode === "overview") {
		els.modeLabel.innerHTML = "<b>dollhouse</b> · orbit";
		setHud(OVERVIEW_HELP);
	} else if (mode === "interior") {
		els.modeLabel.innerHTML = "<b>interior</b> · walkthrough";
	} else if (mode === "peek") {
		els.modeLabel.innerHTML = "<b>locating</b> · release to return";
		setHud("you are <b>here</b> · release to drop back in");
	}
}

// --- input ------------------------------------------------------------------

renderer.domElement.addEventListener("pointerdown", (ev) => {
	if (mode !== "interior" || interiorBusy) return;
	dragging = true;
	dragMoved = 0;
	downX = ev.clientX;
	downY = ev.clientY;
	downLon = lon;
	downLat = lat;
	els.stage.classList.add("grabbing");
	renderer.domElement.setPointerCapture(ev.pointerId);
});
renderer.domElement.addEventListener("pointermove", (ev) => {
	if (mode === "overview") {
		// Highlight + hand cursor + label when hovering an entry disc (skip mid-orbit drag).
		if (ev.buttons !== 0) return;
		const spot = pickByScreen(ev, entryGroup, ENTRY_AIM_PX);
		hoveredEntryIndex = spot ? spot.userData.targetIndex : -1;
		els.stage.classList.toggle("hotspot", spot !== null);
		setHud(spot ? `enter at <b>${panos[hoveredEntryIndex].id}</b>` : OVERVIEW_HELP);
		return;
	}
	if (mode !== "interior") return;
	if (dragging) {
		const k = (0.0032 * camera.fov) / 75;
		lon = downLon + (downX - ev.clientX) * k;
		lat = downLat + (ev.clientY - downY) * k;
		dragMoved = Math.max(dragMoved, Math.hypot(ev.clientX - downX, ev.clientY - downY));
	} else if (!interiorBusy) {
		updateHover(ev);
	}
});
renderer.domElement.addEventListener("pointerup", (ev) => {
	if (mode !== "interior") return;
	dragging = false;
	els.stage.classList.remove("grabbing");
	if (dragMoved < 5 && !interiorBusy) {
		const spot = pickByScreen(ev, hotspotGroup, AUTO_AIM_PX);
		if (spot) travelTo(spot.userData.targetIndex);
	}
});
renderer.domElement.addEventListener(
	"wheel",
	(ev) => {
		if (mode !== "interior") return;
		ev.preventDefault();
		camera.fov = THREE.MathUtils.clamp(camera.fov + ev.deltaY * 0.05, 25, 100);
		camera.updateProjectionMatrix();
	},
	{ passive: false },
);
// Overview: a clean click (OrbitControls drags don't fire 'click') picks an entry.
renderer.domElement.addEventListener("click", (ev) => {
	if (mode !== "overview") return;
	const spot = pickByScreen(ev, entryGroup, ENTRY_AIM_PX);
	if (spot) enter(spot.userData.targetIndex);
});

let hoveredTargetIndex = -1;
let hoveredEntryIndex = -1;
function updateHover(ev) {
	if (currentIndex < 0) return;
	const spot = pickByScreen(ev, hotspotGroup, AUTO_AIM_PX);
	hoveredTargetIndex = spot ? spot.userData.targetIndex : -1;
	els.stage.classList.toggle("hotspot", spot !== null);
	if (spot) {
		const target = panos[spot.userData.targetIndex];
		const wall = spot.userData.occluded ? " · behind wall" : "";
		setHud(`<b>${panos[currentIndex].id}</b> · go to <b>${target.id}</b>${wall}`);
	} else {
		setHud(`<b>${panos[currentIndex].id}</b> · ${currentIndex + 1}/${panos.length} · drag to look`);
	}
}

els.btnEnter.addEventListener("click", () => enter(null));
els.btnExit.addEventListener("click", () => exit());

function holdDown() {
	if (mode !== "interior" || interiorBusy) return;
	peekHeld = true;
	els.btnLocate.classList.add("holding");
	peekStart();
}
function holdUp() {
	if (!peekHeld) return;
	peekHeld = false;
	els.btnLocate.classList.remove("holding");
	if (mode === "peek") peekEnd();
}
els.btnLocate.addEventListener("pointerdown", (ev) => {
	ev.preventDefault();
	holdDown();
});
window.addEventListener("pointerup", holdUp);
window.addEventListener("keydown", (ev) => {
	if (ev.code === "Space" && !ev.repeat) {
		ev.preventDefault();
		holdDown();
	}
});
window.addEventListener("keyup", (ev) => {
	if (ev.code === "Space") holdUp();
});

// --- cell loading -----------------------------------------------------------

function showOverlay(msg, { spinner = true, err = false } = {}) {
	els.overlay.hidden = false;
	els.overlaySpin.hidden = !spinner;
	els.overlayMsg.textContent = msg;
	els.overlayMsg.classList.toggle("err", err);
}
const hideOverlay = () => (els.overlay.hidden = true);

function disposeObject(obj) {
	obj.traverse((o) => {
		if (o.isMesh || o.isLine) {
			o.geometry?.dispose();
			const mats = Array.isArray(o.material) ? o.material : [o.material];
			for (const m of mats) {
				if (m && m !== projMaterial && m !== polyMaterial) m.dispose?.();
			}
		}
	});
}

function clearScene() {
	if (liteRoot) {
		scene.remove(liteRoot);
		disposeObject(liteRoot);
		liteRoot = null;
	}
	if (proxyGroup) {
		scene.remove(proxyGroup);
		disposeObject(proxyGroup);
		proxyGroup = null;
	}
	for (const p of panos) p.texture?.dispose();
	panos = [];
	currentIndex = -1;
	hotspotGroup.clear();
	entryGroup.clear();
	youMarker.visible = false;
	sphereA.visible = false;
	sphereB.visible = false;
	// Drop any in-flight animation so a mid-transition cell switch can't tick
	// stale callbacks against the disposed scene.
	transition = null;
	glide = null;
	interiorBusy = false;
	peekHeld = false;
	savedInterior = null;
	hoveredEntryIndex = -1;
	hoveredTargetIndex = -1;
	clearTravelMask();
}

async function loadCell(t) {
	mode = "loading";
	controls.enabled = false;
	clearScene();
	els.empty.hidden = true;
	updateModeUI();

	// Bake (or fetch the cached) vertex-colored lite export for the dollhouse.
	showOverlay("baking lite scene…");
	let liteUrl = null;
	try {
		const r = await fetch(`/scene-lite/${enc(t.run)}/${enc(t.slot)}/${enc(t.model)}`, {
			method: "POST",
		});
		if (r.ok) liteUrl = (await r.json()).url;
	} catch {
		liteUrl = null;
	}

	// Load the tour manifest (panos + proxy), resolved relative to its URL.
	showOverlay("loading tour…");
	let manifest;
	try {
		const base = new URL(t.url, location.href);
		const res = await fetch(base.toString());
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		manifest = await res.json();
		const list = Array.isArray(manifest.panos) ? manifest.panos : [];
		if (list.length === 0) throw new Error("tour has no panos");
		const entries = list.map((p) => ({
			id: p.id,
			position: p.position,
			forward: p.forward,
			url: new URL(p.file, base).toString(),
		}));
		for (const e of entries) e.texture = await loadTextureUrl(e.url);

		let proxyRoot = null;
		if (manifest.proxy) {
			try {
				proxyRoot = await loadGLB(new URL(manifest.proxy, base).toString());
			} catch (err) {
				proxyRoot = null;
			}
		}
		let lite = null;
		if (liteUrl) {
			try {
				lite = await loadGLB(`${liteUrl}?v=${Date.now()}`);
			} catch {
				lite = null;
			}
		}
		applyCell(entries, proxyRoot, lite);
	} catch (e) {
		showOverlay(`failed to load tour: ${e.message ?? e}`, { spinner: false, err: true });
		mode = "empty";
	}
}

function applyCell(entries, proxyRoot, lite) {
	panos = entries;
	projectionMode = !!proxyRoot;
	sharedOverview = !lite && !!proxyRoot;

	if (!lite && !proxyRoot) {
		showOverlay("nothing to show for this cell", { spinner: false, err: true });
		mode = "empty";
		return;
	}

	// The overview model: the lite export, or the proxy itself when there's none.
	if (lite) {
		liteRoot = lite;
		scene.add(liteRoot);
	}
	if (proxyRoot) {
		setupProjection(proxyRoot);
		proxyGroup = proxyRoot;
		scene.add(proxyGroup);
	}

	// Frame from whichever geometry we have.
	const framed = lite ?? proxyRoot;
	const box = new THREE.Box3().setFromObject(framed);
	const size = box.getSize(new THREE.Vector3());
	box.getCenter(sceneCenter);
	sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;

	camera.near = Math.max(0.02, sceneMaxDim * 0.002);
	camera.far = Math.max(500, sceneMaxDim * 60);

	sizeYouMarker();
	buildEntryMarkers();
	rebuildHotspots();

	const dist = sceneMaxDim * 1.6;
	browsePose.pos.copy(sceneCenter).add(new THREE.Vector3(dist * 0.7, dist * 0.5, dist * 0.9));
	camera.position.copy(browsePose.pos);
	camera.fov = 55;
	camera.lookAt(sceneCenter);
	camera.updateProjectionMatrix();
	controls.target.copy(sceneCenter);
	controls.enabled = true;
	controls.update();
	controls.autoRotate = true;

	setOverviewView();
	mode = "overview";
	hideOverlay();
	updateModeUI();
}

// --- tour list --------------------------------------------------------------

let cells = [];
let activeKey = null;
const cellKey = (c) => `${c.run}/${c.slot}/${c.model}`;

function renderCells() {
	els.list.innerHTML = "";
	if (cells.length === 0) {
		const empty = document.createElement("div");
		empty.id = "cell-empty";
		empty.textContent = "no rendered tours yet — capture one in the main viewer.";
		els.list.appendChild(empty);
		return;
	}
	for (const c of cells) {
		const card = document.createElement("div");
		card.className = `cell-card${cellKey(c) === activeKey ? " active" : ""}`;
		const proxy = c.has_proxy ? " · proxy" : "";
		card.innerHTML =
			`<div class="cc-title">${c.slot} · ${c.model}</div>` +
			`<div class="cc-meta">${c.run} · ${c.panos} pano${c.panos === 1 ? "" : "s"}${proxy}</div>`;
		card.addEventListener("click", () => {
			activeKey = cellKey(c);
			renderCells();
			loadCell(c);
		});
		els.list.appendChild(card);
	}
}

async function refreshCells() {
	try {
		const res = await fetch("/tours");
		if (!res.ok) throw new Error(`${res.status}`);
		const data = await res.json();
		cells = Array.isArray(data.tours) ? data.tours : [];
	} catch {
		cells = [];
	}
	renderCells();
}

els.refresh.addEventListener("click", refreshCells);

// --- resize + render loop ---------------------------------------------------

function resize() {
	const w = els.stage.clientWidth;
	const h = els.stage.clientHeight;
	if (w === 0 || h === 0) return;
	renderer.setSize(w, h, false);
	camera.aspect = w / h;
	camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(els.stage);
resize();

let _lastFrame = 0;
renderer.setAnimationLoop((time) => {
	const now = performance.now();
	const dt = _lastFrame ? Math.min(0.05, (time - _lastFrame) / 1000) : 0;
	_lastFrame = time;

	if (transition) {
		const t = Math.min(1, (now - transition.start) / transition.dur);
		const e = easeInOut(t);
		camera.position.lerpVectors(transition.fromPos, transition.toPos, e);
		camera.quaternion.slerpQuaternions(transition.fromQuat, transition.toQuat, e);
		camera.updateMatrixWorld();
		setTravelMask(t);
		if (!transition.midDone && t >= 0.5) {
			transition.midDone = true;
			transition.onMid?.();
		}
		if (proxyGroup && proxyGroup.visible && projectionMode) updateProjection();
		if (t >= 1) {
			clearTravelMask();
			const cb = transition.onEnd;
			transition = null;
			cb?.();
		}
	} else if (mode === "overview") {
		controls.update();
	} else if (mode === "interior") {
		if (glide) {
			const t = Math.min(1, (now - glide.start) / glide.dur);
			camera.position.lerpVectors(glide.fromPos, glide.toPos, easeInOut(t));
			setTravelMask(t);
			if (t >= 1) {
				clearTravelMask();
				const idx = glide.index;
				glide = null;
				interiorBusy = false;
				hotspotGroup.visible = true;
				activate(idx);
			}
		}
		if (projectionMode) updateProjection();
		else sphereA.position.copy(camera.position);
		applyLook();
	} else if (mode === "peek") {
		// Slowly orbit the dollhouse around its center so locating gives a 360 view.
		const off = camera.position.clone().sub(sceneCenter);
		const a = PEEK_ROTATE_SPEED * dt;
		const c = Math.cos(a);
		const s = Math.sin(a);
		camera.position.x = sceneCenter.x + off.x * c - off.z * s;
		camera.position.z = sceneCenter.z + off.x * s + off.z * c;
		camera.lookAt(sceneCenter);
	}

	// Constant on-screen sizing + pulse for whichever marker group is showing.
	const pulse = 1 + 0.07 * Math.sin(time * 0.004);
	for (const group of [hotspotGroup, entryGroup]) {
		if (!group.visible) continue;
		const targetPx = group === entryGroup ? ENTRY_TARGET_PX : HOTSPOT_TARGET_PX;
		const hoverIdx = group === entryGroup ? hoveredEntryIndex : hoveredTargetIndex;
		for (const spot of group.children) {
			const hovered = spot.userData.targetIndex === hoverIdx;
			const occluded = spot.userData.occluded;
			const d = camera.position.distanceTo(spot.position);
			spot.scale.setScalar(hotspotScaleForDistance(d, targetPx) * (hovered ? 1.35 : 1) * (occluded ? 0.82 : 1));
			const [disc, ring] = spot.children;
			ring.scale.setScalar(pulse);
			if (occluded) {
				disc.material.opacity = hovered ? 0.5 : 0.22;
				ring.material.opacity = hovered ? 0.8 : 0.45;
			} else {
				disc.material.opacity = hovered ? 0.9 : 0.55;
				ring.material.opacity = hovered ? 1.0 : 0.85;
			}
		}
	}

	renderer.render(scene, camera);
});

refreshCells();
