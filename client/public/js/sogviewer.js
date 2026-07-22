// PlayCanvas-based SOG renderer for the splat viewer's "sog" view. Renders the
// SOG-encoded trained splat (trained.sog, written by client/tools/ply-to-sog.mjs)
// with the PlayCanvas engine's native gsplat support — the same library that
// encoded it — inside a canvas overlaid on the mkkellogg viewer's container.
//
// Deliberately minimal: one Application, one camera with orbit (drag) + dolly
// (wheel) + WASD/QE/Space walk (same keys as the main viewer), one gsplat
// entity. The app is kept alive across view toggles (display + autoRender flip)
// and destroyed on closeSogView().

import {
    Application,
    Color,
    Entity,
    FILLMODE_NONE,
    RESOLUTION_AUTO,
} from "playcanvas";

let app = null;
let canvas = null;
let camEntity = null;
let resizeObs = null;
let visible = false;

// Orbit state (radians / metres): camera = target + dist * offset(yaw, pitch).
const target = { x: 0, y: 1, z: 0 };
let yaw = 0.6;
let pitch = -0.4;
let dist = 6;
let speed = 3; // WASD metres/sec (Shift sprints 3x)

const pressed = new Set();
const MOVE_CODES = new Set([
    "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE",
    "Space", "ShiftLeft", "ShiftRight",
]);
let dragging = false;
let lastX = 0;
let lastY = 0;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function applyCamera() {
    if (!camEntity) return;
    const cp = Math.cos(pitch);
    const px = target.x + dist * cp * Math.sin(yaw);
    const py = target.y + dist * Math.sin(pitch);
    const pz = target.z + dist * cp * Math.cos(yaw);
    camEntity.setPosition(px, py, pz);
    camEntity.lookAt(target.x, target.y, target.z);
}

// Initialize orbit state from a {position:[x,y,z], lookAt:[x,y,z]} view.
function setViewPose(view) {
    if (!view || !view.position || !view.lookAt) return;
    const [px, py, pz] = view.position;
    const [tx, ty, tz] = view.lookAt;
    target.x = tx;
    target.y = ty;
    target.z = tz;
    const dx = px - tx;
    const dy = py - ty;
    const dz = pz - tz;
    dist = Math.max(0.05, Math.hypot(dx, dy, dz));
    pitch = clamp(Math.asin(dy / dist), -1.55, 1.55);
    yaw = Math.atan2(dx, dz);
}

function onUpdate(dt) {
    if (!visible || pressed.size === 0) {
        applyCamera();
        return;
    }
    const step = speed * (pressed.has("ShiftLeft") || pressed.has("ShiftRight") ? 3 : 1)
        * Math.min(dt, 0.05);
    // Horizontal forward (camera → target) and screen-right, from the yaw only
    // (walk, don't dive — vertical is Q/E/Space), matching the main viewer feel.
    const fx = -Math.sin(yaw);
    const fz = -Math.cos(yaw);
    const rx = -fz;
    const rz = fx;
    let mx = 0;
    let my = 0;
    let mz = 0;
    if (pressed.has("KeyW")) { mx += fx; mz += fz; }
    if (pressed.has("KeyS")) { mx -= fx; mz -= fz; }
    if (pressed.has("KeyD")) { mx += rx; mz += rz; }
    if (pressed.has("KeyA")) { mx -= rx; mz -= rz; }
    if (pressed.has("KeyE") || pressed.has("Space")) my += 1;
    if (pressed.has("KeyQ")) my -= 1;
    const len = Math.hypot(mx, my, mz);
    if (len > 0) {
        target.x += (mx / len) * step;
        target.y += (my / len) * step;
        target.z += (mz / len) * step;
    }
    applyCamera();
}

function onPointerDown(ev) {
    if (!visible || ev.button !== 0) return;
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    canvas.setPointerCapture?.(ev.pointerId);
}

function onPointerMove(ev) {
    if (!visible || !dragging) return;
    yaw -= (ev.clientX - lastX) * 0.005;
    pitch = clamp(pitch + (ev.clientY - lastY) * 0.005, -1.55, 1.55);
    lastX = ev.clientX;
    lastY = ev.clientY;
}

function onPointerUp(ev) {
    dragging = false;
    canvas?.releasePointerCapture?.(ev.pointerId);
}

function onWheel(ev) {
    if (!visible) return;
    ev.preventDefault();
    dist = clamp(dist * Math.exp(ev.deltaY * 0.0012), 0.05, 1000);
}

function onKeyDown(ev) {
    if (!visible || !MOVE_CODES.has(ev.code)) return;
    const tag = ev.target && ev.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    pressed.add(ev.code);
    if (ev.code === "Space") ev.preventDefault();
}

function onKeyUp(ev) {
    pressed.delete(ev.code);
}

function onBlur() {
    pressed.clear();
    dragging = false;
}

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
        fov: 60,
        nearClip: 0.02,
        farClip: 2000,
    });
    app.root.addChild(camEntity);

    if (spd) speed = spd;
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
    // View-dependent colour is data-driven: PlayCanvas' gsplat renderer reads and
    // evaluates whatever spherical-harmonic bands the SOG carries, with no toggle
    // to set here. The trained splat is degree 2 (splat/stage6.py), and
    // client/tools/ply-to-sog.mjs preserves the PLY's f_rest through the
    // 2DGS→3DGS flatten so splat-transform bakes it into the SOG — so switching
    // to the SOG view shows the same moving highlights as the mkkellogg view.
    splat.addComponent("gsplat", { asset });
    app.root.addChild(splat);

    app.on("update", onUpdate);
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
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
