// pano walkthrough — matterport-style viewer for 360° tours captured in the
// main client. Drop tour.json + the pano JPEGs in (optionally with proxy.glb).
//
// Two modes, chosen by what's in the drop:
//
//  * PROJECTION mode (when a proxy.glb is present) — the real matterport
//    experience. The proxy is a heavily decimated, merged stand-in of the whole
//    scene, in the SAME world frame the panos were captured in. We render it
//    from a free camera and texture every fragment by PROJECTING the nearest
//    captured panoramas onto it (view-dependent texture mapping): for each of
//    the K nearest capture points we turn the fragment's world position into a
//    direction from that capture point, sample its equirect there, and blend by
//    proximity + surface-facing. Because the texture is glued to real geometry,
//    moving the camera produces correct parallax — so travelling between anchors
//    interpolates coherently instead of cross-dissolving two flat spheres. A
//    camera-following backdrop sphere (the nearest pano) fills sky / anything the
//    proxy doesn't cover.
//
//  * SPHERE mode (no proxy) — the original fallback: each pano on an inverted
//    sphere, click a floor disc to crossfade to the next with a forward warp.
//
// The JPEGs are already tone-mapped sRGB straight off the main viewer's canvas,
// so nothing here tone-maps or colour-converts — what you captured is what you
// see (raw texels written straight to the framebuffer).

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const stageEl = document.getElementById("stage");
const dropEl = document.getElementById("drop");
const fileInputEl = document.getElementById("file-input");
const listEl = document.getElementById("pano-list");
const hudEl = document.getElementById("hud");
const emptyEl = document.getElementById("empty");
const viewToggleEl = document.getElementById("view-toggle");
const travelFadeEl = document.getElementById("travel-fade");

// --- renderer / scene --------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.NoToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
stageEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0f12);

const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 300);

const SPHERE_RADIUS = 60;

// Same direction→uv convention as the capture stitch (and three's
// EquirectangularReflectionMapping): u = atan2(z,x)/2π + 0.5, v = asin(y)/π + 0.5.
// A custom shader (rather than sphere UVs) keeps the mapping exact and gives us
// an opacity uniform for crossfades without any lighting/tonemapping chunks.
function makePanoMaterial() {
	return new THREE.ShaderMaterial({
		uniforms: {
			map: { value: null },
			opacity: { value: 1.0 },
		},
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
// sphereA: the active pano in sphere mode; the camera-following backdrop in
// projection mode. sphereB: the incoming pano during a sphere-mode crossfade.
const sphereA = new THREE.Mesh(sphereGeo, makePanoMaterial());
const sphereB = new THREE.Mesh(sphereGeo, makePanoMaterial());
sphereA.renderOrder = 0;
sphereB.renderOrder = 1;
sphereB.visible = false;
scene.add(sphereA, sphereB);

// --- projection material (view-dependent texture mapping) --------------------

const PROJ_K = 4; // panos blended per fragment (the K nearest to the camera)

// 1×1 black stand-in so the sampler array is always fully bound (unused slots
// carry weight 0, so they never contribute).
const DUMMY_TEX = new THREE.DataTexture(
	new Uint8Array([0, 0, 0, 255]),
	1,
	1,
	THREE.RGBAFormat,
);
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
				return vec2(
					atan(d.z, d.x) / TAU + 0.5,
					asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5
				);
			}

			// One pano's contribution. Sampler arrays need a constant index, so
			// the caller passes the sampler explicitly; the cheaper uniforms are
			// indexed dynamically. Surfaces facing away from a capture point are
			// down-weighted (a coarse, depth-free occlusion guard) but never
			// fully dropped, so the proxy never tears into holes.
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
const gltfLoader = new GLTFLoader();

let proxyGroup = null;
let projectionMode = false;
let backdropRadius = SPHERE_RADIUS;

// --- poly view ---------------------------------------------------------------
// The view toggle (projection mode only) swaps the proxy's projection shader for
// a flat-shaded matte material so you can inspect the raw decimated geometry
// instead of the projected panoramas. Lights drive only standard materials, so
// they leave the (unlit) projection + backdrop shaders untouched.
const polyMaterial = new THREE.MeshStandardMaterial({
	color: 0x9aa7b4,
	roughness: 0.9,
	metalness: 0.0,
	flatShading: true,
	side: THREE.DoubleSide,
});
const polyHemi = new THREE.HemisphereLight(0xdfe7ff, 0x20242c, 1.1);
const polyDir = new THREE.DirectionalLight(0xffffff, 1.4);
polyDir.position.set(0.6, 1.0, 0.4);
scene.add(polyHemi, polyDir);

let polyView = false; // false = projected 360s; true = bare proxy geometry

// --- hotspots ----------------------------------------------------------------

const hotspotGroup = new THREE.Group();
scene.add(hotspotGroup);

const HOTSPOT_FLOOR_DROP = 1.3; // meters below eye level (markers sit at the floor)
const HOTSPOT_MAX_DIST = 18; // (sphere mode) clamp far panos inside the sphere
const HOTSPOT_MIN_DIST = 1.2;
// Which anchors become in-scene hotspots. Projection mode raycasts the proxy to
// split nearby anchors into line-of-sight ones ("in this room", shown brightly)
// and occluded ones ("behind a wall/floor", shown as dim ghosts) — both stay
// clickable, so reach is broad while it's obvious what you're stepping through.
// Counts are generous (vs. the old nearest-6) so you can cross a room in one hop
// instead of beading along every capture point. Markers lie FLAT on the surface
// (they read as spots on the floor) and rescale every frame to a constant
// ON-SCREEN size; pressing them is made easy by auto-aim (screen-space pick
// magnetism) instead of billboarding them upright into flat coins.
const HOTSPOT_REACH = 30; // (projection) furthest an anchor can be and still show
const HOTSPOT_MAX_VISIBLE = 10; // line-of-sight anchors shown
const HOTSPOT_MAX_OCCLUDED = 6; // behind-wall anchors shown (as ghosts)
const HOTSPOT_OCCLUDE_EPS = 0.2; // trim off each end so a hugged wall isn't a block
const HOTSPOT_TARGET_PX = 24; // target marker radius on screen, in CSS px
const AUTO_AIM_PX = 42; // click/hover snaps to the nearest marker within this radius
const HOTSPOT_BASE_RADIUS = 0.16; // the disc geometry's world radius
const HOTSPOT_VISIBLE_COLOR = 0xffffff;
const HOTSPOT_RING_COLOR = 0x9ad4ff;
const HOTSPOT_OCCLUDED_COLOR = 0xe0c271;
let hoveredTargetIndex = -1;

function makeHotspot(targetIndex) {
	const group = new THREE.Group();
	const disc = new THREE.Mesh(
		new THREE.CircleGeometry(HOTSPOT_BASE_RADIUS, 40),
		new THREE.MeshBasicMaterial({
			color: HOTSPOT_VISIBLE_COLOR,
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
			color: HOTSPOT_RING_COLOR,
			transparent: true,
			opacity: 0.9,
			side: THREE.DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	// Lie flat on the surface (normal up) so the markers read as spots ON the
	// floor; auto-aim (see pickHotspot) handles the picking, so they don't need to
	// billboard upright to stay clickable.
	disc.rotation.x = -Math.PI / 2;
	ring.rotation.x = -Math.PI / 2;
	group.add(disc, ring);
	group.renderOrder = 2;
	disc.renderOrder = 2;
	ring.renderOrder = 2;
	group.userData.targetIndex = targetIndex;
	return group;
}

// Neighbour anchors, nearest first. Projection mode drops anything past
// HOTSPOT_REACH so huge scenes don't surface the whole map at once; sphere mode
// keeps all (placement clamps them into the backdrop sphere instead).
function neighborsByDistance() {
	const cur = new THREE.Vector3().fromArray(panos[currentIndex].position);
	const tmp = new THREE.Vector3();
	const out = [];
	for (let i = 0; i < panos.length; i++) {
		if (i === currentIndex) continue;
		const d2 = cur.distanceToSquared(tmp.fromArray(panos[i].position));
		if (projectionMode && d2 > HOTSPOT_REACH * HOTSPOT_REACH) continue;
		out.push([i, d2]);
	}
	out.sort((a, b) => a[1] - b[1]);
	return out.map((o) => o[0]);
}

// Is the straight line between two capture points blocked by the proxy? That's
// our "behind a wall / floor" test. We trace eye-to-eye (capture height, not the
// floor-dropped marker) and trim a little off both ends so a wall the anchor
// stands against — or one we're standing against — doesn't read as occlusion.
const _occluder = new THREE.Raycaster();
const _occFrom = new THREE.Vector3();
const _occTo = new THREE.Vector3();
const _occDir = new THREE.Vector3();

function anchorOccluded(fromPos, toPos) {
	if (!proxyGroup) return false;
	_occFrom.fromArray(fromPos);
	_occTo.fromArray(toPos);
	_occDir.subVectors(_occTo, _occFrom);
	const dist = _occDir.length();
	if (dist < 1e-3) return false;
	_occDir.divideScalar(dist);
	_occluder.set(_occFrom, _occDir);
	_occluder.near = HOTSPOT_OCCLUDE_EPS;
	_occluder.far = dist - HOTSPOT_OCCLUDE_EPS;
	if (_occluder.far <= _occluder.near) return false;
	return _occluder.intersectObject(proxyGroup, true).length > 0;
}

// Base palette for a marker; per-frame opacity/scale (incl. hover) is applied in
// the render loop. Occluded markers go amber so "through a wall" reads at a glance.
function styleHotspot(spot, occluded) {
	const [disc, ring] = spot.children;
	disc.material.color.setHex(occluded ? HOTSPOT_OCCLUDED_COLOR : HOTSPOT_VISIBLE_COLOR);
	ring.material.color.setHex(occluded ? HOTSPOT_OCCLUDED_COLOR : HOTSPOT_RING_COLOR);
}

function rebuildHotspots() {
	hotspotGroup.clear();
	if (currentIndex < 0) return;
	const cur = panos[currentIndex];
	let nVisible = 0;
	let nOccluded = 0;
	for (const i of neighborsByDistance()) {
		const occluded = anchorOccluded(cur.position, panos[i].position);
		// Keep the nearest few of each kind: broad enough to navigate freely, but
		// capped so dense scenes don't drown in markers.
		if (occluded) {
			if (nOccluded >= HOTSPOT_MAX_OCCLUDED) continue;
			nOccluded++;
		} else {
			if (nVisible >= HOTSPOT_MAX_VISIBLE) continue;
			nVisible++;
		}
		const spot = makeHotspot(i);
		spot.userData.occluded = occluded;
		styleHotspot(spot, occluded);
		if (projectionMode) {
			// Markers sit at the panos' true world positions, dropped to the floor.
			spot.position.fromArray(panos[i].position);
			spot.position.y -= HOTSPOT_FLOOR_DROP;
		} else {
			// Sphere mode: camera sits at the origin, so place markers RELATIVE to
			// the current pano and clamp them inside the backdrop sphere.
			const rel = new THREE.Vector3()
				.fromArray(panos[i].position)
				.sub(new THREE.Vector3().fromArray(cur.position));
			rel.y -= HOTSPOT_FLOOR_DROP;
			const len = rel.length();
			if (len > HOTSPOT_MAX_DIST) rel.multiplyScalar(HOTSPOT_MAX_DIST / len);
			else if (len < HOTSPOT_MIN_DIST) rel.multiplyScalar(HOTSPOT_MIN_DIST / len);
			spot.position.copy(rel);
		}
		hotspotGroup.add(spot);
		if (nVisible >= HOTSPOT_MAX_VISIBLE && nOccluded >= HOTSPOT_MAX_OCCLUDED) break;
	}
}

// World scale that renders the disc at ~HOTSPOT_TARGET_PX on screen at distance
// `d` — perspective shrinks worldSize/distance, so scaling by distance keeps the
// click target a constant pixel size no matter how far the anchor is (or the fov).
function hotspotScaleForDistance(d) {
	const h = stageEl.clientHeight || 1;
	const worldRadius =
		(HOTSPOT_TARGET_PX * 2 * d * Math.tan((camera.fov * Math.PI) / 360)) / h;
	return THREE.MathUtils.clamp(worldRadius / HOTSPOT_BASE_RADIUS, 0.15, 14);
}

// --- tour state ---------------------------------------------------------------

let panos = []; // { id, position:[3], forward:[3], texture }
let currentIndex = -1;
let transitioning = false;

function setHud(text) {
	hudEl.hidden = !text;
	hudEl.innerHTML = text ?? "";
}

function renderList() {
	listEl.innerHTML = "";
	panos.forEach((p, i) => {
		const item = document.createElement("div");
		item.className = `pano-item${i === currentIndex ? " active" : ""}`;
		const pos = p.position.map((v) => v.toFixed(1)).join(", ");
		item.innerHTML = `<span>${p.id}</span><span class="pos">${pos}</span>`;
		item.addEventListener("click", () => travelTo(i));
		listEl.appendChild(item);
	});
}

function activate(index) {
	currentIndex = index;
	if (!projectionMode) {
		sphereA.material.uniforms.map.value = panos[index].texture;
		sphereA.material.uniforms.opacity.value = 1;
	}
	rebuildHotspots();
	renderList();
	const mode = projectionMode ? "walk" : "look";
	setHud(`<b>${panos[index].id}</b> · ${index + 1}/${panos.length} · ${mode}`);
}

// --- look controls (lon/lat drag + fov zoom) -----------------------------------

let lon = 0; // azimuth from +X toward +Z — matches the equirect u convention
let lat = 0;
let dragging = false;
let dragMoved = 0;
let downX = 0;
let downY = 0;
let downLon = 0;
let downLat = 0;

function setLookFromForward(f) {
	const v = new THREE.Vector3().fromArray(f).normalize();
	lon = Math.atan2(v.z, v.x);
	lat = Math.asin(THREE.MathUtils.clamp(v.y, -1, 1));
}

function applyLook() {
	lat = THREE.MathUtils.clamp(lat, -1.55, 1.55);
	const dx = Math.cos(lat) * Math.cos(lon);
	const dy = Math.sin(lat);
	const dz = Math.cos(lat) * Math.sin(lon);
	// Projection mode: look from the camera's world position. Sphere mode: the
	// camera sits at the origin and only ever rotates.
	if (projectionMode) {
		camera.lookAt(
			camera.position.x + dx,
			camera.position.y + dy,
			camera.position.z + dz,
		);
	} else {
		camera.lookAt(dx, dy, dz);
	}
}

renderer.domElement.addEventListener("pointerdown", (ev) => {
	dragging = true;
	dragMoved = 0;
	downX = ev.clientX;
	downY = ev.clientY;
	downLon = lon;
	downLat = lat;
	stageEl.classList.add("grabbing");
	renderer.domElement.setPointerCapture(ev.pointerId);
});

renderer.domElement.addEventListener("pointermove", (ev) => {
	if (dragging) {
		const k = (0.0032 * camera.fov) / 75; // slower turn when zoomed in
		lon = downLon + (downX - ev.clientX) * k;
		lat = downLat + (ev.clientY - downY) * k;
		dragMoved = Math.max(
			dragMoved,
			Math.hypot(ev.clientX - downX, ev.clientY - downY),
		);
	} else {
		updateHover(ev);
	}
});

renderer.domElement.addEventListener("pointerup", (ev) => {
	dragging = false;
	stageEl.classList.remove("grabbing");
	if (dragMoved < 5) handleClick(ev);
});

renderer.domElement.addEventListener(
	"wheel",
	(ev) => {
		ev.preventDefault();
		camera.fov = THREE.MathUtils.clamp(camera.fov + ev.deltaY * 0.05, 25, 100);
		camera.updateProjectionMatrix();
	},
	{ passive: false },
);

// --- hotspot picking (screen-space auto-aim) --------------------------------
// The markers lie flat on the surface, so their foreshortened discs are fiddly
// to hit dead-on. Instead of raycasting the geometry we project each marker's
// centre to the screen and lock onto the NEAREST one within AUTO_AIM_PX of the
// cursor — forgiving targeting that keeps the discs glued to the floor.
const _aimWorld = new THREE.Vector3();

function pickHotspot(ev) {
	const rect = renderer.domElement.getBoundingClientRect();
	const cx = ev.clientX - rect.left;
	const cy = ev.clientY - rect.top;
	let best = null;
	let bestPx = AUTO_AIM_PX;
	for (const spot of hotspotGroup.children) {
		spot.getWorldPosition(_aimWorld).project(camera);
		if (_aimWorld.z > 1) continue; // behind the camera
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

function updateHover(ev) {
	if (transitioning || currentIndex < 0) return;
	const spot = pickHotspot(ev);
	hoveredTargetIndex = spot ? spot.userData.targetIndex : -1;
	stageEl.classList.toggle("hotspot", spot !== null);
	if (spot) {
		const target = panos[spot.userData.targetIndex];
		const wall = spot.userData.occluded ? " · behind wall" : "";
		setHud(`<b>${panos[currentIndex].id}</b> · go to <b>${target.id}</b>${wall}`);
	} else {
		const mode = projectionMode ? "walk" : "look";
		setHud(
			`<b>${panos[currentIndex].id}</b> · ${currentIndex + 1}/${panos.length} · ${mode}`,
		);
	}
}

function handleClick(ev) {
	if (transitioning || currentIndex < 0) return;
	const spot = pickHotspot(ev);
	if (spot) travelTo(spot.userData.targetIndex);
}

// --- travel transition -----------------------------------------------------------

// Mask the interpolation artifacts that flash during a move (proxy parallax,
// VDTM ghosting, the backdrop pano swap) with two cheap, colour-safe effects
// driven by travel progress: a screen-space blur on the canvas and a dip toward
// the background colour. Both follow a sin(pi*t) bell, so they ease IN as the
// move starts and OUT as it settles, peaking mid-transition where the artifacts
// are worst. A CSS blur leaves the viewer's raw-sRGB render path untouched
// (an EffectComposer pass would re-encode the hand-managed colours).
const TRAVEL_BLUR_MAX_PX = 8;
const TRAVEL_FADE_MAX = 0.5;

function setTravelMask(t) {
	const m = Math.sin(Math.PI * THREE.MathUtils.clamp(t, 0, 1));
	renderer.domElement.style.filter =
		m > 0.002 ? `blur(${(m * TRAVEL_BLUR_MAX_PX).toFixed(2)}px)` : "none";
	if (travelFadeEl) travelFadeEl.style.opacity = (m * TRAVEL_FADE_MAX).toFixed(3);
}

function clearTravelMask() {
	renderer.domElement.style.filter = "none";
	if (travelFadeEl) travelFadeEl.style.opacity = "0";
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

function travelTo(index) {
	if (transitioning || index === currentIndex || !panos[index]) return;
	hoveredTargetIndex = -1;

	if (projectionMode) {
		// Physically glide the camera through world space; the projection
		// re-blends every frame, so the geometry + textures interpolate live.
		transitioning = true;
		hotspotGroup.visible = false;
		const from = camera.position.clone();
		const to = new THREE.Vector3().fromArray(panos[index].position);
		const DURATION = 900;
		const start = performance.now();
		const step = (now) => {
			const t = Math.min(1, (now - start) / DURATION);
			camera.position.lerpVectors(from, to, easeInOut(t));
			setTravelMask(t);
			if (t < 1) {
				requestAnimationFrame(step);
				return;
			}
			camera.position.copy(to);
			clearTravelMask();
			transitioning = false;
			hotspotGroup.visible = true;
			activate(index);
		};
		requestAnimationFrame(step);
		return;
	}

	// Sphere mode: slide the camera-centered spheres along the travel direction
	// so the outgoing world streams past while the incoming one settles in.
	const target = panos[index];
	transitioning = true;
	hotspotGroup.visible = false;
	const dir = new THREE.Vector3()
		.fromArray(target.position)
		.sub(new THREE.Vector3().fromArray(panos[currentIndex].position));
	const dist = dir.length();
	if (dist > 1e-6) dir.divideScalar(dist);
	const warp = THREE.MathUtils.clamp(dist, 1, 8) * 0.5;

	sphereB.material.uniforms.map.value = target.texture;
	sphereB.material.uniforms.opacity.value = 0;
	sphereB.visible = true;

	const DURATION = 700;
	const start = performance.now();
	const step = (now) => {
		const t = Math.min(1, (now - start) / DURATION);
		const e = easeInOut(t);
		sphereB.material.uniforms.opacity.value = e;
		sphereA.position.copy(dir).multiplyScalar(-warp * e);
		sphereB.position.copy(dir).multiplyScalar(warp * (1 - e));
		setTravelMask(t);
		if (t < 1) {
			requestAnimationFrame(step);
			return;
		}
		sphereA.material.uniforms.map.value = target.texture;
		sphereA.material.uniforms.opacity.value = 1;
		sphereA.position.set(0, 0, 0);
		sphereB.visible = false;
		sphereB.position.set(0, 0, 0);
		clearTravelMask();
		transitioning = false;
		hotspotGroup.visible = true;
		activate(index);
	};
	requestAnimationFrame(step);
}

// --- per-frame projection update ---------------------------------------------

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

	// Inverse-distance weights over the K nearest captures; the small constant
	// keeps the weight finite (and the blend smooth) when standing on an anchor.
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

	// Backdrop = the nearest pano, recentred on the camera so it reads as
	// infinitely far (no parallax) behind the projected geometry.
	sphereA.material.uniforms.map.value = panos[order[0]].texture;
	sphereA.position.copy(cam);
}

// --- file loading -----------------------------------------------------------------

const textureLoader = new THREE.TextureLoader();

function prepPanoTexture(tex) {
	// Mip generation breaks at the equirect seam (the u-derivative jumps a full
	// wrap there), painting a blurry column — plain linear filtering avoids it.
	// RepeatWrapping keeps the seam's bilinear taps continuous.
	tex.generateMipmaps = false;
	tex.minFilter = THREE.LinearFilter;
	tex.magFilter = THREE.LinearFilter;
	tex.wrapS = THREE.RepeatWrapping;
	return tex;
}

async function loadTexture(file) {
	const url = URL.createObjectURL(file);
	try {
		return prepPanoTexture(await textureLoader.loadAsync(url));
	} finally {
		URL.revokeObjectURL(url);
	}
}

async function loadTextureUrl(url) {
	return prepPanoTexture(await textureLoader.loadAsync(url));
}

async function loadProxy(file) {
	const url = URL.createObjectURL(file);
	try {
		const gltf = await gltfLoader.loadAsync(url);
		return gltf.scene;
	} finally {
		URL.revokeObjectURL(url);
	}
}

function disposeProxy() {
	if (!proxyGroup) return;
	scene.remove(proxyGroup);
	proxyGroup.traverse((o) => {
		if (o.isMesh && o.geometry) o.geometry.dispose();
	});
	proxyGroup = null;
}

// Wire a loaded proxy into projection mode: re-skin every mesh with the shared
// projection material, recompute normals (decimation discards usable ones), and
// size the backdrop sphere + camera clip planes to the scene's extent.
function setupProjection(root) {
	disposeProxy();
	proxyGroup = root;
	root.traverse((o) => {
		if (!o.isMesh || !o.geometry) return;
		o.geometry.computeVertexNormals();
		o.material = projMaterial;
		o.frustumCulled = false;
	});
	scene.add(root);

	const box = new THREE.Box3().setFromObject(root);
	const sph = box.getBoundingSphere(new THREE.Sphere());
	backdropRadius = Math.max(80, sph.radius * 4);
	camera.near = Math.max(0.02, sph.radius * 0.0008);
	camera.far = Math.max(300, backdropRadius * 2.5);
	camera.updateProjectionMatrix();

	sphereA.scale.setScalar(backdropRadius / SPHERE_RADIUS);
	sphereA.renderOrder = -1;
	sphereA.material.uniforms.opacity.value = 1;
	// CRITICAL: depth-test the backdrop so the OPAQUE proxy occludes it. Without
	// this the backdrop (a transparent material) renders after the opaque proxy
	// and, with depthTest off, paints over the whole proxy — leaving only a flat
	// camera-centred 360 sphere with no parallax. With depthTest on, the far
	// backdrop fails the depth test wherever the near proxy wrote depth, so the
	// projected geometry shows through (and parallaxes) and the backdrop only
	// fills sky / gaps the proxy doesn't cover.
	sphereA.material.depthTest = true;
	sphereA.visible = true;
	sphereB.visible = false;
}

// Restore sphere mode's renderer/camera/backdrop state (in case the previous
// drop was a projection tour).
function resetToSphereMode() {
	disposeProxy();
	sphereA.scale.setScalar(1);
	sphereA.position.set(0, 0, 0);
	sphereA.renderOrder = 0;
	sphereA.material.depthTest = false; // sphere mode: always-on background
	sphereA.visible = true;
	camera.position.set(0, 0, 0);
	camera.near = 0.1;
	camera.far = 300;
	camera.updateProjectionMatrix();
}

// Apply the current view mode to the loaded proxy: the projected panoramas
// (360) or the flat-shaded proxy geometry (poly). Only meaningful in projection
// mode — sphere-mode tours have no geometry to toggle.
function applyViewMode() {
	if (proxyGroup) {
		proxyGroup.traverse((o) => {
			if (o.isMesh) o.material = polyView ? polyMaterial : projMaterial;
		});
	}
	// Hide the pano backdrop in poly view so the bare geometry reads against the
	// plain background; in 360 view it fills sky / anything the proxy misses.
	if (projectionMode) sphereA.visible = !polyView;
	viewToggleEl.textContent = polyView ? "● poly view" : "◐ 360 view";
	viewToggleEl.classList.toggle("poly", polyView);
	viewToggleEl.title = polyView
		? "Showing the low-poly proxy geometry. Click (or press P) to project the 360° panoramas back onto it."
		: "Showing the 360° panoramas projected onto the geometry. Click (or press P) to see the raw low-poly proxy.";
}

function setPolyView(on) {
	if (!projectionMode) return;
	polyView = !!on;
	applyViewMode();
}

async function loadFiles(fileList) {
	const files = Array.from(fileList);
	let manifest = null;
	const images = new Map();
	const glbs = new Map();
	for (const f of files) {
		const lower = f.name.toLowerCase();
		if (lower.endsWith(".json")) {
			try {
				manifest = JSON.parse(await f.text());
			} catch {
				setHud(`<b>error</b> · ${f.name} is not valid JSON`);
				return;
			}
		} else if (lower.endsWith(".glb")) {
			glbs.set(f.name, f);
		} else if (f.type.startsWith("image/")) {
			images.set(f.name, f);
		}
	}
	if (images.size === 0) {
		setHud("<b>error</b> · no panorama images in the drop");
		return;
	}

	let entries;
	if (manifest?.panos?.length) {
		entries = manifest.panos
			.filter((p) => images.has(p.file))
			.map((p) => ({
				id: p.id,
				position: p.position,
				forward: p.forward,
				file: images.get(p.file),
			}));
		if (entries.length === 0) {
			setHud("<b>error</b> · tour.json references none of the dropped images");
			return;
		}
	} else {
		// Bare images, no manifest: line them up 3m apart so traversal still works.
		entries = [...images.entries()]
			.sort(([a], [b]) => a.localeCompare(b))
			.map(([name, file], i) => ({
				id: name.replace(/\.[^.]+$/, ""),
				position: [i * 3, 0, 0],
				forward: [1, 0, 0],
				file,
			}));
	}

	// A proxy (named by the manifest, else any dropped .glb) switches on
	// projection mode.
	const proxyFile =
		(manifest?.proxy && glbs.get(manifest.proxy)) ||
		(glbs.size > 0 ? [...glbs.values()][0] : null);

	setHud(
		`loading ${entries.length} pano${entries.length === 1 ? "" : "s"}${proxyFile ? " + proxy" : ""}…`,
	);
	for (const e of entries) e.texture = await loadTexture(e.file);

	let proxyRoot = null;
	if (proxyFile) {
		try {
			proxyRoot = await loadProxy(proxyFile);
		} catch (err) {
			setHud(`<b>warning</b> · proxy failed to load (${err.message}); sphere mode`);
			proxyRoot = null;
		}
	}
	applyTour(entries, proxyRoot);
}

// Commit a loaded tour: swap in the panos (+ optional proxy), choose projection
// vs sphere mode, and frame the first anchor. Shared by the file-drop and the
// persisted-URL (?tour=) load paths.
function applyTour(entries, proxyRoot) {
	clearTravelMask();
	for (const p of panos) p.texture.dispose();
	panos = entries;
	emptyEl.hidden = true;

	if (proxyRoot) {
		projectionMode = true;
		setupProjection(proxyRoot);
		camera.position.fromArray(panos[0].position);
		polyView = false;
		viewToggleEl.hidden = false;
		applyViewMode();
	} else {
		projectionMode = false;
		resetToSphereMode();
		viewToggleEl.hidden = true;
		sphereA.material.uniforms.map.value = panos[0].texture;
		sphereA.material.uniforms.opacity.value = 1;
	}
	setLookFromForward(panos[0].forward ?? [1, 0, 0]);
	activate(0);
}

// Load a persisted tour by URL. Pano + proxy files are resolved RELATIVE to the
// tour.json URL, so the same manifest works wherever it's hosted. `tourUrl` is
// normalized against the page origin first, so both a same-origin path (the
// /runs/<cell>/tour/tour.json the rendered-tour list hands us) and an absolute
// cross-origin URL (the viewer's ?tour= hand-off) work. Driven by the saved-tour
// list, the file drop's manifest, and ?tour=.
async function loadFromTourUrl(tourUrl) {
	tourUrl = new URL(tourUrl, location.href).toString();
	setHud("loading tour…");
	let manifest;
	try {
		const res = await fetch(tourUrl);
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		manifest = await res.json();
	} catch (e) {
		setHud(`<b>error</b> · could not load tour (${e.message})`);
		return;
	}
	const list = Array.isArray(manifest.panos) ? manifest.panos : [];
	if (list.length === 0) {
		setHud("<b>error</b> · tour has no panos");
		return;
	}
	const entries = list.map((p) => ({
		id: p.id,
		position: p.position,
		forward: p.forward,
		url: new URL(p.file, tourUrl).toString(),
	}));
	setHud(`loading ${entries.length} pano${entries.length === 1 ? "" : "s"}${manifest.proxy ? " + proxy" : ""}…`);
	for (const e of entries) e.texture = await loadTextureUrl(e.url);

	let proxyRoot = null;
	if (manifest.proxy) {
		try {
			proxyRoot = (await gltfLoader.loadAsync(new URL(manifest.proxy, tourUrl).toString())).scene;
		} catch (err) {
			setHud(`<b>warning</b> · proxy failed to load (${err.message}); sphere mode`);
		}
	}
	applyTour(entries, proxyRoot);
}

// --- rendered-tour browser ---------------------------------------------------
// Lists every tour persisted under the runs tree (GET /tours, served read-only
// at /runs); click one to load it directly — no file handling.
const toursListEl = document.getElementById("tours-list");
const toursRefreshEl = document.getElementById("tours-refresh");
let loadedTourUrl = null;
let lastTours = [];

function renderTours() {
	toursListEl.innerHTML = "";
	if (lastTours.length === 0) {
		const empty = document.createElement("div");
		empty.id = "tours-empty";
		empty.textContent = "no rendered tours yet";
		toursListEl.appendChild(empty);
		return;
	}
	for (const t of lastTours) {
		const item = document.createElement("div");
		item.className = `tour-item${t.url === loadedTourUrl ? " active" : ""}`;
		const proxy = t.has_proxy ? " · proxy" : "";
		item.innerHTML =
			`<span class="title">${t.slot} · ${t.model}</span>` +
			`<span class="meta">${t.run} · ${t.panos} pano${t.panos === 1 ? "" : "s"}${proxy}</span>`;
		item.addEventListener("click", () => loadSavedTour(t));
		toursListEl.appendChild(item);
	}
}

async function refreshTours() {
	try {
		const res = await fetch("/tours");
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		lastTours = Array.isArray(data.tours) ? data.tours : [];
	} catch {
		lastTours = [];
	}
	renderTours();
}

async function loadSavedTour(t) {
	loadedTourUrl = t.url;
	renderTours();
	await loadFromTourUrl(t.url);
}

toursRefreshEl?.addEventListener("click", refreshTours);

dropEl.addEventListener("dragover", (ev) => {
	ev.preventDefault();
	dropEl.classList.add("over");
});
dropEl.addEventListener("dragleave", () => dropEl.classList.remove("over"));
dropEl.addEventListener("drop", (ev) => {
	ev.preventDefault();
	dropEl.classList.remove("over");
	loadFiles(ev.dataTransfer.files);
});
fileInputEl.addEventListener("change", () => {
	if (fileInputEl.files.length) loadFiles(fileInputEl.files);
	fileInputEl.value = "";
});
// Dropping anywhere on the stage works too.
stageEl.addEventListener("dragover", (ev) => ev.preventDefault());
stageEl.addEventListener("drop", (ev) => {
	ev.preventDefault();
	loadFiles(ev.dataTransfer.files);
});

// View toggle: click the chip or press P to flip between the projected 360s and
// the bare proxy geometry (no-op outside projection mode).
viewToggleEl.addEventListener("click", () => setPolyView(!polyView));
window.addEventListener("keydown", (ev) => {
	if (ev.key === "p" || ev.key === "P") setPolyView(!polyView);
});

// --- resize + render loop -----------------------------------------------------------

function resize() {
	const w = stageEl.clientWidth;
	const h = stageEl.clientHeight;
	renderer.setSize(w, h);
	camera.aspect = w / h;
	camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stageEl);
resize();

renderer.setAnimationLoop((time) => {
	// In poly view the proxy uses a lit material and the pano backdrop is hidden,
	// so the per-frame pano re-projection is unnecessary work — skip it.
	if (projectionMode && !polyView) updateProjection();
	applyLook();
	// Size every visible hotspot to a constant on-screen size for its current
	// distance, with a gentle ring pulse and a bump + brighten on hover.
	const pulse = 1 + 0.07 * Math.sin(time * 0.004);
	for (const spot of hotspotGroup.children) {
		const hovered = spot.userData.targetIndex === hoveredTargetIndex;
		const occluded = spot.userData.occluded;
		const d = camera.position.distanceTo(spot.position);
		// Constant on-screen size; ghosts ride a touch smaller so they recede
		// behind the live anchors.
		spot.scale.setScalar(
			hotspotScaleForDistance(d) * (hovered ? 1.35 : 1) * (occluded ? 0.82 : 1),
		);
		const [disc, ring] = spot.children;
		ring.scale.setScalar(pulse);
		if (occluded) {
			// Behind a wall/floor: a faint amber ghost, so it's clearly "through"
			// something rather than a marker in the room with you.
			disc.material.opacity = hovered ? 0.5 : 0.22;
			ring.material.opacity = hovered ? 0.8 : 0.45;
		} else {
			disc.material.opacity = hovered ? 0.9 : 0.55;
			ring.material.opacity = hovered ? 1.0 : 0.85;
		}
	}
	renderer.render(scene, camera);
});

// Populate the rendered-tour list on boot, and auto-load a specific tour when
// opened as /pano?tour=<url to tour.json> (the auto-tour / website hand-off).
refreshTours();
const _tourParam = new URLSearchParams(location.search).get("tour");
if (_tourParam) loadFromTourUrl(_tourParam);
