// Orbit + first-person camera rig for the splat viewer, ported from the main
// dashboard viewer (scene3d.js) so both clients fly the same way: left-drag
// orbits, HOLD RIGHT for a momentary pointer-locked look, a toggle enters a
// permanent first-person fly, WASD/QE walk, R/F dolly, Shift sprints, and the
// scroll wheel trims the fly speed while in first-person.
//
// mkkellogg owns the splat render loop, so unlike scene3d.js the rig is driven
// from the outside: the caller ticks update(dt) once a frame and gets back
// whether the camera moved. The library's own OrbitControls are disabled
// (useBuiltInControls:false) because its loop calls controls.update()
// unconditionally, which re-aims the camera at the orbit target every frame and
// would fight the pointer-lock look.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";

const MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "r", "f"]);
const SPEED_STEP = 1.15; // per wheel notch
const SPEED_MIN = 0.05;
const SPEED_MAX = 16;
const SPRINT = 3;
// A lock that ended within this window counts as "the Esc that just happened"
// (see consumeEscape) — the browser drops the pointer lock itself and the
// unlock event lands a frame or two after the keydown.
const ESCAPE_GRACE_MS = 250;

// Fly speed is a VIEW PREFERENCE, not per-scene state: the wheel scales the
// scene-derived base speed, and the factor survives viewer rebuilds, cell
// switches and the three.js ↔ PlayCanvas view flip so the walk never changes
// feel underfoot. Shared with sogviewer.js for exactly that reason.
let speedMultiplier = 1;

export function flySpeedMultiplier() {
    return speedMultiplier;
}

// One wheel notch → a multiplicative change, so each notch feels the same at
// any current speed.
export function bumpFlySpeed(deltaY) {
    const factor = deltaY < 0 ? SPEED_STEP : 1 / SPEED_STEP;
    speedMultiplier = Math.min(
        SPEED_MAX,
        Math.max(SPEED_MIN, speedMultiplier * factor),
    );
    showSpeedHud();
}

// Transient fly-speed readout pinned to the left of the screen: flashes on a
// wheel notch and fades out shortly after. pointer-events:none so it can never
// intercept a click; opacity (not display) toggles so it fades.
let hudEl = null;
let hudTimer = null;

function ensureHud() {
    if (hudEl) return hudEl;
    hudEl = document.createElement("div");
    hudEl.style.cssText = [
        "position: fixed",
        "left: 18px",
        "top: 50%",
        "transform: translateY(-50%)",
        "padding: 6px 11px",
        "background: rgba(22, 24, 29, 0.94)",
        "color: #e6e6e6",
        "border: 1px solid #2a2d35",
        "border-radius: 6px",
        "font: 12px ui-monospace, SFMono-Regular, Menlo, monospace",
        "pointer-events: none",
        "opacity: 0",
        "transition: opacity 0.25s ease",
        "z-index: 130", // above the splat viewer overlay (120)
        "white-space: nowrap",
    ].join("; ");
    document.body.appendChild(hudEl);
    return hudEl;
}

export function showSpeedHud() {
    const hud = ensureHud();
    hud.textContent = `fly speed ×${speedMultiplier.toFixed(2)}`;
    hud.style.opacity = "1";
    if (hudTimer) clearTimeout(hudTimer);
    hudTimer = setTimeout(() => {
        hud.style.opacity = "0";
    }, 1200);
}

export function hideSpeedHud() {
    if (!hudEl) return;
    if (hudTimer) {
        clearTimeout(hudTimer);
        hudTimer = null;
    }
    hudEl.style.opacity = "0";
}

// Half the largest dimension of a scene AABB ({min,max} arrays) — scales the
// walk speed and the first-person pivot distance so movement matches the scene.
export function radiusFor(aabb) {
    if (!aabb || !aabb.min || !aabb.max) return 10;
    const r =
        0.5 *
        Math.max(
            aabb.max[0] - aabb.min[0],
            aabb.max[1] - aabb.min[1],
            aabb.max[2] - aabb.min[2],
        );
    return isFinite(r) && r > 0 ? r : 10;
}

// Base metres/sec at speed ×1, from the scene radius.
export function baseSpeedFor(radius) {
    return Math.max(2, radius * 0.5);
}

function isTypingTarget(t) {
    return (
        t instanceof HTMLElement &&
        (t.tagName === "INPUT" ||
            t.tagName === "TEXTAREA" ||
            t.tagName === "SELECT" ||
            t.isContentEditable)
    );
}

// Build the rig around an existing camera + canvas. `isActive` gates the global
// key handlers (the canvas lives inside a full-screen overlay that other views
// share) and `radius` sizes the fly speed; both can change after construction.
export function createCameraRig({
    camera,
    domElement,
    target,
    radius = 10,
    isActive = () => true,
}) {
    const orbit = new OrbitControls(camera, domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    if (target) orbit.target.copy(target);
    // RIGHT is the first-person look handle, so orbit keeps rotate on LEFT and
    // takes pan on MIDDLE (where mkkellogg's built-in rig had dolly).
    orbit.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.PAN,
        RIGHT: null,
    };
    orbit.update();

    const fp = new PointerLockControls(camera, domElement);
    let mode = "orbit"; // "orbit" | "fp"
    let sceneRadius = radius;
    // `rmbInitiated` marks a lock that came from the right-hold, so it exits on
    // release and doesn't flip the toggle button; `rmbHeld` tracks the physical
    // button for the async-lock race guard.
    let rmbInitiated = false;
    let rmbHeld = false;
    let unlockedAt = 0;
    let onModeCb = () => {};
    const pressed = new Set();

    const _fpDir = new THREE.Vector3();
    const _fwd = new THREE.Vector3();
    const _right = new THREE.Vector3();
    const _move = new THREE.Vector3();
    const _worldUp = new THREE.Vector3(0, 1, 0);

    // Park the orbit pivot a fixed distance ahead of the first-person camera so
    // orbit resumes from the FP pose without snapping (it re-derives its angle
    // from position → target).
    function syncTargetAhead() {
        fp.getDirection(_fpDir);
        orbit.target
            .copy(camera.position)
            .addScaledVector(_fpDir, Math.max(2, sceneRadius * 0.5));
    }

    fp.addEventListener("lock", () => {
        mode = "fp";
        orbit.enabled = false; // also makes OrbitControls ignore the wheel
        // Right-hold released before the async lock engaged — exit straight out.
        if (rmbInitiated && !rmbHeld) {
            fp.unlock();
            return;
        }
        // A right-hold look is momentary; only the toggle drives the button.
        if (!rmbInitiated) onModeCb("fp");
    });

    fp.addEventListener("unlock", () => {
        mode = "orbit";
        unlockedAt = performance.now();
        syncTargetAhead(); // hand the FP pose to orbit before it takes back over
        orbit.enabled = true;
        orbit.update();
        hideSpeedHud(); // the fly-speed trim only applies in FP
        const wasRmb = rmbInitiated;
        rmbInitiated = false;
        rmbHeld = false;
        if (!wasRmb) onModeCb("orbit");
    });

    // W/S fly along the look direction (pitch included, so looking up climbs);
    // A/D strafe level; Q/E lower/raise on world Y; Shift sprints. Mouse-look
    // comes from the pointer lock. Returns whether the camera actually moved.
    function applyFpMove(dt) {
        const moving =
            pressed.has("w") ||
            pressed.has("s") ||
            pressed.has("a") ||
            pressed.has("d") ||
            pressed.has("e") ||
            pressed.has("q");
        if (!moving) return false;
        const speed =
            baseSpeedFor(sceneRadius) *
            speedMultiplier *
            (pressed.has("shift") ? SPRINT : 1) *
            dt;
        if (pressed.has("w") || pressed.has("s")) {
            fp.getDirection(_fpDir);
            if (pressed.has("w"))
                camera.position.addScaledVector(_fpDir, speed);
            if (pressed.has("s"))
                camera.position.addScaledVector(_fpDir, -speed);
        }
        if (pressed.has("d")) fp.moveRight(speed);
        if (pressed.has("a")) fp.moveRight(-speed);
        if (pressed.has("e")) camera.position.y += speed;
        if (pressed.has("q")) camera.position.y -= speed;
        return true;
    }

    // Orbit-mode walk: WASD strafes on the horizontal plane, Q/E moves world
    // down/up, R/F dollies. Camera and target translate together so the pivot
    // follows. Speed tracks the pivot distance, so a zoomed-in view steps small.
    function applyOrbitMove(dt) {
        if (pressed.size === 0) return false;
        const shifted = pressed.has("shift");
        const camDist = Math.max(1, camera.position.distanceTo(orbit.target));
        const speed = Math.max(2, camDist * 0.6) * (shifted ? SPRINT : 1) * dt;
        _fwd.subVectors(orbit.target, camera.position);
        _fwd.y = 0;
        if (_fwd.lengthSq() === 0) return false;
        _fwd.normalize();
        _right.crossVectors(_fwd, _worldUp).normalize();
        _move.set(0, 0, 0);
        if (pressed.has("w")) _move.addScaledVector(_fwd, speed);
        if (pressed.has("s")) _move.addScaledVector(_fwd, -speed);
        if (pressed.has("d")) _move.addScaledVector(_right, speed);
        if (pressed.has("a")) _move.addScaledVector(_right, -speed);
        if (pressed.has("e")) _move.addScaledVector(_worldUp, speed);
        if (pressed.has("q")) _move.addScaledVector(_worldUp, -speed);
        let moved = false;
        if (_move.lengthSq() !== 0) {
            camera.position.add(_move);
            orbit.target.add(_move);
            moved = true;
        }
        if (pressed.has("r") || pressed.has("f")) {
            const rate = shifted ? 4 : 1.5;
            let factor = 1;
            if (pressed.has("r")) factor *= Math.pow(1 / rate, dt);
            if (pressed.has("f")) factor *= Math.pow(rate, dt);
            const offset = camera.position.clone().sub(orbit.target);
            const dist = offset.length();
            if (dist > 0) {
                offset.multiplyScalar(
                    Math.max(0.05, Math.min(4000, dist * factor)) / dist,
                );
                camera.position.copy(orbit.target).add(offset);
                moved = true;
            }
        }
        return moved;
    }

    const onPointerDown = (ev) => {
        // Hold RIGHT to look: a momentary first-person fly for the duration of
        // the hold — the same pointer lock as the toggle, so the mouse-look is
        // 1:1 with no screen-edge stall. Only from orbit; if the toggle already
        // holds a lock, leave it alone.
        if (ev.button !== 2 || mode !== "orbit" || !isActive()) return;
        rmbInitiated = true;
        rmbHeld = true;
        fp.lock();
    };

    const onPointerUp = (ev) => {
        if (ev.button !== 2) return;
        rmbHeld = false;
        if (mode === "fp" && rmbInitiated) fp.unlock();
    };

    // A cancelled right-hold (lost pointer) exits the look like a normal release.
    const onPointerCancel = () => {
        if (!rmbHeld) return;
        rmbHeld = false;
        if (mode === "fp" && rmbInitiated) fp.unlock();
    };

    const onContextMenu = (ev) => ev.preventDefault();

    // The wheel trims the fly speed in first-person; in orbit it stays zoom
    // (OrbitControls owns it). Non-passive so the page can't scroll under the
    // locked pointer.
    const onWheel = (ev) => {
        if (mode !== "fp") return;
        ev.preventDefault();
        bumpFlySpeed(ev.deltaY);
    };

    // Keys are global (the canvas doesn't hold focus under pointer lock), so
    // they're gated on isActive() and skipped while typing in the side panel.
    const onKeyDown = (ev) => {
        if (!isActive() || isTypingTarget(ev.target)) return;
        const k = ev.key.toLowerCase();
        if (MOVE_KEYS.has(k)) {
            pressed.add(k);
            ev.preventDefault();
        } else if (k === "shift") {
            pressed.add("shift");
        }
    };
    const onKeyUp = (ev) => pressed.delete(ev.key.toLowerCase());
    const onBlur = () => pressed.clear();

    domElement.addEventListener("pointerdown", onPointerDown);
    domElement.addEventListener("pointerup", onPointerUp);
    domElement.addEventListener("pointercancel", onPointerCancel);
    domElement.addEventListener("contextmenu", onContextMenu);
    domElement.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);

    return {
        orbit,
        // One frame of camera input. Orbit damping needs update() every frame;
        // in FP the pivot is parked ahead instead, since update() would re-aim
        // the camera at it and undo the mouse-look.
        update(dt) {
            if (!isActive()) {
                pressed.clear();
                return false;
            }
            let moved = mode === "fp" ? applyFpMove(dt) : applyOrbitMove(dt);
            if (mode === "fp") syncTargetAhead();
            else if (orbit.update()) moved = true;
            return moved;
        },
        getMode: () => mode,
        setMode(next) {
            if (next === "fp") {
                rmbInitiated = false; // a toggle-driven lock (drives the button)
                fp.lock(); // requestPointerLock — needs the click gesture
            } else {
                fp.unlock();
            }
        },
        onModeChange(cb) {
            onModeCb = cb;
        },
        setRadius(r) {
            sceneRadius = r;
        },
        // Esc is the native pointer-lock exit, so by the time the key reaches us
        // the browser has usually dropped the lock already. Returns whether this
        // Esc belongs to first-person — the viewer's own Esc (close) must not
        // also fire.
        consumeEscape() {
            if (mode === "fp") {
                fp.unlock();
                return true;
            }
            return performance.now() - unlockedAt < ESCAPE_GRACE_MS;
        },
        dispose() {
            domElement.removeEventListener("pointerdown", onPointerDown);
            domElement.removeEventListener("pointerup", onPointerUp);
            domElement.removeEventListener("pointercancel", onPointerCancel);
            domElement.removeEventListener("contextmenu", onContextMenu);
            domElement.removeEventListener("wheel", onWheel);
            window.removeEventListener("keydown", onKeyDown);
            window.removeEventListener("keyup", onKeyUp);
            window.removeEventListener("blur", onBlur);
            if (fp.isLocked) fp.unlock();
            fp.dispose();
            orbit.dispose();
            hideSpeedHud();
            pressed.clear();
        },
    };
}
