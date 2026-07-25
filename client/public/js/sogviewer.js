// PlayCanvas-based SOG renderer for the splat viewer's "sog" view. Renders the
// SOG-encoded trained splat (trained.sog, written by client/tools/ply-to-sog.mjs)
// with the PlayCanvas engine's native gsplat support — the same library that
// encoded it — inside a canvas overlaid on the mkkellogg viewer's container.
//
// Deliberately minimal: one Application, one camera, one gsplat entity. The
// camera mirrors the three.js rig (splatcam.js) key-for-key — left-drag orbits,
// hold-right is a momentary pointer-locked look, WASD/QE walk, R/F dolly, Shift
// sprints, and the wheel zooms in orbit / trims the fly speed in first-person —
// so flipping between the two splat views never changes how the scene handles.
// The app is kept alive across view toggles (display + autoRender flip) and
// destroyed on closeSogView().

import {
    Application,
    Color,
    Entity,
    FILLMODE_NONE,
    RESOLUTION_AUTO,
} from "playcanvas";
import { bumpFlySpeed, flySpeedMultiplier, hideSpeedHud } from "./splatcam.js";

let app = null;
let canvas = null;
let camEntity = null;
let resizeObs = null;
let visible = false;

// Camera state: `pos` IS the camera, `yaw`/`pitch` are the look angles and
// `dist` is how far ahead the orbit pivot sits. The pivot is derived rather
// than stored, so orbit and first-person share one representation and flipping
// between them never snaps the view.
const pos = { x: 4, y: 3, z: 6 };
const fwd = { x: 0, y: 0, z: -1 };
let yaw = 0.6;
let pitch = -0.4;
let dist = 6;
let baseSpeed = 3; // metres/sec at fly speed ×1, sized to the scene on open

let mode = "orbit"; // "orbit" | "fp"
let rmbInitiated = false; // this lock came from the right-hold, not the toggle
let rmbHeld = false;
let unlockedAt = 0;
let onModeCb = () => {};

const MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "r", "f"]);
const pressed = new Set();
const SPRINT = 3;
const LOOK_SENSITIVITY = 0.002; // radians per pixel of locked mouse movement
const ESCAPE_GRACE_MS = 250;
let dragging = false;
let lastX = 0;
let lastY = 0;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function updateForward() {
    const cp = Math.cos(pitch);
    fwd.x = -cp * Math.sin(yaw);
    fwd.y = -Math.sin(pitch);
    fwd.z = -cp * Math.cos(yaw);
}

function applyCamera() {
    if (!camEntity) return;
    updateForward();
    camEntity.setPosition(pos.x, pos.y, pos.z);
    camEntity.lookAt(pos.x + fwd.x, pos.y + fwd.y, pos.z + fwd.z);
}

// Re-place the camera so the pivot it was orbiting stays put after yaw/pitch or
// `dist` changed.
function placeAround(px, py, pz) {
    updateForward();
    pos.x = px - fwd.x * dist;
    pos.y = py - fwd.y * dist;
    pos.z = pz - fwd.z * dist;
}

function orbitBy(dYaw, dPitch) {
    updateForward();
    const px = pos.x + fwd.x * dist;
    const py = pos.y + fwd.y * dist;
    const pz = pos.z + fwd.z * dist;
    yaw += dYaw;
    pitch = clamp(pitch + dPitch, -1.55, 1.55);
    placeAround(px, py, pz);
}

function dollyBy(factor) {
    updateForward();
    const px = pos.x + fwd.x * dist;
    const py = pos.y + fwd.y * dist;
    const pz = pos.z + fwd.z * dist;
    dist = clamp(dist * factor, 0.05, 1000);
    placeAround(px, py, pz);
}

// Initialize from a {position:[x,y,z], lookAt:[x,y,z]} view (the pose handed
// over from the three.js viewer, so the flip is seamless).
function setViewPose(view) {
    if (!view || !view.position || !view.lookAt) return;
    const [px, py, pz] = view.position;
    const [tx, ty, tz] = view.lookAt;
    pos.x = px;
    pos.y = py;
    pos.z = pz;
    const dx = tx - px;
    const dy = ty - py;
    const dz = tz - pz;
    dist = Math.max(0.05, Math.hypot(dx, dy, dz));
    pitch = clamp(Math.asin(-dy / dist), -1.55, 1.55);
    yaw = Math.atan2(-dx, -dz);
    updateForward();
}

// W/S fly along the look direction (pitch included, so looking up climbs); A/D
// strafe level; Q/E lower/raise on world Y; Shift sprints.
function applyFpMove(dt) {
    const speed =
        baseSpeed *
        flySpeedMultiplier() *
        (pressed.has("shift") ? SPRINT : 1) *
        dt;
    updateForward();
    const rx = -fwd.z;
    const rz = fwd.x;
    const rlen = Math.hypot(rx, rz) || 1;
    if (pressed.has("w")) {
        pos.x += fwd.x * speed;
        pos.y += fwd.y * speed;
        pos.z += fwd.z * speed;
    }
    if (pressed.has("s")) {
        pos.x -= fwd.x * speed;
        pos.y -= fwd.y * speed;
        pos.z -= fwd.z * speed;
    }
    if (pressed.has("d")) {
        pos.x += (rx / rlen) * speed;
        pos.z += (rz / rlen) * speed;
    }
    if (pressed.has("a")) {
        pos.x -= (rx / rlen) * speed;
        pos.z -= (rz / rlen) * speed;
    }
    if (pressed.has("e")) pos.y += speed;
    if (pressed.has("q")) pos.y -= speed;
}

// Orbit-mode walk: WASD strafes on the horizontal plane (walk, don't dive —
// vertical is Q/E), R/F dollies. Speed tracks the pivot distance, so a zoomed-in
// view steps small.
function applyOrbitMove(dt) {
    const shifted = pressed.has("shift");
    const speed = Math.max(2, dist * 0.6) * (shifted ? SPRINT : 1) * dt;
    updateForward();
    const flen = Math.hypot(fwd.x, fwd.z) || 1;
    const fx = fwd.x / flen;
    const fz = fwd.z / flen;
    if (pressed.has("w")) {
        pos.x += fx * speed;
        pos.z += fz * speed;
    }
    if (pressed.has("s")) {
        pos.x -= fx * speed;
        pos.z -= fz * speed;
    }
    if (pressed.has("d")) {
        pos.x -= fz * speed;
        pos.z += fx * speed;
    }
    if (pressed.has("a")) {
        pos.x += fz * speed;
        pos.z -= fx * speed;
    }
    if (pressed.has("e")) pos.y += speed;
    if (pressed.has("q")) pos.y -= speed;
    if (pressed.has("r") || pressed.has("f")) {
        const rate = shifted ? 4 : 1.5;
        let factor = 1;
        if (pressed.has("r")) factor *= Math.pow(1 / rate, dt);
        if (pressed.has("f")) factor *= Math.pow(rate, dt);
        dollyBy(factor);
    }
}

function onUpdate(dt) {
    if (!visible) return;
    const step = Math.min(dt, 0.05);
    if (pressed.size > 0) {
        if (mode === "fp") applyFpMove(step);
        else applyOrbitMove(step);
    }
    applyCamera();
}

// ---- first-person (pointer lock) --------------------------------------------

function lockPointer() {
    canvas?.requestPointerLock?.();
}

function unlockPointer() {
    document.exitPointerLock?.();
}

function onPointerLockChange() {
    if (document.pointerLockElement === canvas) {
        mode = "fp";
        dragging = false; // an orbit drag can't survive into the look
        // Right-hold released before the async lock engaged — exit straight out.
        if (rmbInitiated && !rmbHeld) {
            unlockPointer();
            return;
        }
        // A right-hold look is momentary; only the toggle drives the button.
        if (!rmbInitiated) onModeCb("fp");
        return;
    }
    if (mode !== "fp") return;
    mode = "orbit";
    unlockedAt = performance.now();
    hideSpeedHud(); // the fly-speed trim only applies in FP
    const wasRmb = rmbInitiated;
    rmbInitiated = false;
    rmbHeld = false;
    if (!wasRmb) onModeCb("orbit");
}

// Mouse-look under the lock. On `mousemove` at the document, mirroring
// PointerLockControls — that's where the browser delivers the raw deltas.
function onLockedMouseMove(ev) {
    if (mode !== "fp") return;
    yaw -= (ev.movementX || 0) * LOOK_SENSITIVITY;
    pitch = clamp(pitch + (ev.movementY || 0) * LOOK_SENSITIVITY, -1.55, 1.55);
}

// ---- input ------------------------------------------------------------------

function onPointerDown(ev) {
    if (!visible) return;
    // Hold RIGHT to look: a momentary first-person fly for the duration of the
    // hold, the same pointer lock the toggle uses.
    if (ev.button === 2) {
        if (mode !== "orbit") return;
        rmbInitiated = true;
        rmbHeld = true;
        lockPointer();
        return;
    }
    if (ev.button !== 0 || mode === "fp") return;
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    canvas.setPointerCapture?.(ev.pointerId);
}

function onPointerMove(ev) {
    if (!visible || !dragging || mode === "fp") return;
    orbitBy(-(ev.clientX - lastX) * 0.005, (ev.clientY - lastY) * 0.005);
    lastX = ev.clientX;
    lastY = ev.clientY;
}

function onPointerUp(ev) {
    if (ev.button === 2) {
        rmbHeld = false;
        if (mode === "fp" && rmbInitiated) unlockPointer();
        return;
    }
    dragging = false;
    canvas?.releasePointerCapture?.(ev.pointerId);
}

// A cancelled right-hold (lost pointer) exits the look like a normal release.
function onPointerCancel() {
    dragging = false;
    if (!rmbHeld) return;
    rmbHeld = false;
    if (mode === "fp" && rmbInitiated) unlockPointer();
}

function onContextMenu(ev) {
    ev.preventDefault();
}

// The wheel dollies in orbit and trims the fly speed in first-person.
function onWheel(ev) {
    if (!visible) return;
    ev.preventDefault();
    if (mode === "fp") bumpFlySpeed(ev.deltaY);
    else dollyBy(Math.exp(ev.deltaY * 0.0012));
}

function onKeyDown(ev) {
    if (!visible) return;
    const tag = ev.target && ev.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const k = ev.key.toLowerCase();
    if (MOVE_KEYS.has(k)) {
        pressed.add(k);
        ev.preventDefault();
    } else if (k === "shift") {
        pressed.add("shift");
    }
}

function onKeyUp(ev) {
    pressed.delete(ev.key.toLowerCase());
}

function onBlur() {
    pressed.clear();
    dragging = false;
}

// The camera-mode handle the splat viewer's first-person toggle drives while
// this view is up — same shape as the three.js rig's (splatcam.js).
export const sogCamera = {
    getMode: () => mode,
    setMode(next) {
        if (!app) return;
        if (next === "fp") {
            rmbInitiated = false; // a toggle-driven lock (drives the button)
            lockPointer();
        } else {
            unlockPointer();
        }
    },
    onModeChange(cb) {
        onModeCb = cb;
    },
    // Esc is the native pointer-lock exit, so the browser has usually dropped
    // the lock before the key reaches us; a lock that just ended still counts.
    consumeEscape() {
        if (mode === "fp") {
            unlockPointer();
            return true;
        }
        return performance.now() - unlockedAt < ESCAPE_GRACE_MS;
    },
};

// Open (or replace) the SOG view inside `container`: create the PlayCanvas app
// on an overlay canvas, load `url` as a gsplat asset, and frame `view`
// ({position, lookAt}). Resolves once the splat is loaded; starts hidden —
// call setSogVisible(true) to reveal.
export async function openSogView({ container, url, view, speed: spd }) {
    closeSogView();

    // Overlay the mkkellogg canvas in the exact same rect. z-index is scoped by
    // the container's `isolation: isolate`, so it sits above the base canvas but
    // never over the viewer's header / controls / help.
    canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
        position: "absolute",
        inset: "0",
        width: "100%",
        height: "100%",
        display: "none",
        zIndex: "1",
    });
    container.appendChild(canvas);

    app = new Application(canvas, {
        graphicsDeviceOptions: { antialias: false },
    });
    app.setCanvasFillMode(FILLMODE_NONE);
    app.setCanvasResolution(RESOLUTION_AUTO);
    app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
    app.start();
    app.autoRender = false; // hidden until setSogVisible(true)

    const fit = () => {
        const r = container.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) app.resizeCanvas(r.width, r.height);
    };
    fit();
    resizeObs = new ResizeObserver(fit);
    resizeObs.observe(container);

    camEntity = new Entity("sog-camera");
    camEntity.addComponent("camera", {
        clearColor: new Color(0, 0, 0, 1),
        // Matches the three.js viewers, so a first-person pass through a room
        // reads the same in both.
        fov: 75,
        nearClip: 0.02,
        farClip: 2000,
    });
    app.root.addChild(camEntity);

    if (spd) baseSpeed = spd;
    setViewPose(view);
    applyCamera();

    let asset;
    try {
        asset = await new Promise((resolve, reject) => {
            app.assets.loadFromUrl(url, "gsplat", (err, a) =>
                err ? reject(new Error(String(err))) : resolve(a),
            );
        });
    } catch (e) {
        closeSogView(); // don't leave a splat-less app behind a broken toggle
        throw e;
    }
    if (!app) throw new Error("sog view closed while loading"); // closed mid-load
    const splat = new Entity("sog-splat");
    // Colour is data-driven: PlayCanvas' gsplat renderer evaluates whatever
    // spherical-harmonic bands the SOG carries, with no toggle to set here. The
    // trained splat is flat (degree 0, splat/stage6.py) — it carries no `f_rest_*`
    // — so the SOG is one colour per Gaussian, matching the mkkellogg view.
    splat.addComponent("gsplat", { asset });
    app.root.addChild(splat);

    app.on("update", onUpdate);
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerCancel);
    canvas.addEventListener("contextmenu", onContextMenu);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    document.addEventListener("pointerlockchange", onPointerLockChange);
    document.addEventListener("mousemove", onLockedMouseMove);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
}

// Show/hide the SOG canvas (and pause rendering while hidden). Safe no-op when
// the view was never opened.
export function setSogVisible(on) {
    visible = !!on && !!app;
    if (!app) return;
    canvas.style.display = visible ? "block" : "none";
    app.autoRender = visible;
    if (!visible) {
        if (mode === "fp") unlockPointer(); // never keep the look off-screen
        pressed.clear();
        dragging = false;
    }
}

export function isSogOpen() {
    return !!app;
}

export function closeSogView() {
    visible = false;
    pressed.clear();
    dragging = false;
    if (mode === "fp") unlockPointer();
    mode = "orbit";
    rmbInitiated = false;
    rmbHeld = false;
    document.removeEventListener("pointerlockchange", onPointerLockChange);
    document.removeEventListener("mousemove", onLockedMouseMove);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
    window.removeEventListener("blur", onBlur);
    resizeObs?.disconnect();
    resizeObs = null;
    const a = app;
    app = null;
    camEntity = null;
    if (a) {
        try {
            a.destroy(); // tears down assets, entities, and the GL context
        } catch {
            /* destroying a partially-built app can throw; ignore */
        }
    }
    canvas?.remove();
    canvas = null;
}
