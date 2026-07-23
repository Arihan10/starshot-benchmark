// Orbit + fly camera controls for the SOG-LOD playground.
//
// Orbit state is (target, yaw, pitch, dist): the camera sits at
// target + dist·dir(yaw, pitch). Left-drag orbits, wheel dollies, right/middle
// drag pans in the screen plane, and WASD + Q/E/Space fly the target (so the
// whole rig translates). `frame()` seats the rig OUTSIDE a bounding box looking
// down at a birdseye tilt — the playground's establishing shot.

import { Vec3 } from "playcanvas";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const DEG = Math.PI / 180;

const MOVE_CODES = new Set([
    "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE",
    "Space", "ShiftLeft", "ShiftRight",
]);

export class OrbitFlyControls {
    constructor(canvas, cameraEntity, { speed = 4 } = {}) {
        this.canvas = canvas;
        this.cam = cameraEntity;
        this.speed = speed;

        this.target = new Vec3(0, 1, 0);
        this.yaw = 0.6;
        this.pitch = -0.5;
        this.dist = 8;

        this.pressed = new Set();
        this.dragging = 0; // 0 none, 1 orbit, 2 pan
        this.lastX = 0;
        this.lastY = 0;
        this.enabled = true;

        this._onDown = this._onDown.bind(this);
        this._onMove = this._onMove.bind(this);
        this._onUp = this._onUp.bind(this);
        this._onWheel = this._onWheel.bind(this);
        this._onKeyDown = this._onKeyDown.bind(this);
        this._onKeyUp = this._onKeyUp.bind(this);
        this._onBlur = this._onBlur.bind(this);
        this._onCtx = (e) => e.preventDefault();
    }

    attach() {
        const c = this.canvas;
        c.addEventListener("pointerdown", this._onDown);
        c.addEventListener("pointermove", this._onMove);
        c.addEventListener("pointerup", this._onUp);
        c.addEventListener("wheel", this._onWheel, { passive: false });
        c.addEventListener("contextmenu", this._onCtx);
        window.addEventListener("keydown", this._onKeyDown);
        window.addEventListener("keyup", this._onKeyUp);
        window.addEventListener("blur", this._onBlur);
    }

    detach() {
        const c = this.canvas;
        c.removeEventListener("pointerdown", this._onDown);
        c.removeEventListener("pointermove", this._onMove);
        c.removeEventListener("pointerup", this._onUp);
        c.removeEventListener("wheel", this._onWheel);
        c.removeEventListener("contextmenu", this._onCtx);
        window.removeEventListener("keydown", this._onKeyDown);
        window.removeEventListener("keyup", this._onKeyUp);
        window.removeEventListener("blur", this._onBlur);
    }

    // Seat the rig from an explicit world pose (camera position + look-at point).
    setPose(px, py, pz, tx, ty, tz) {
        this.target.set(tx, ty, tz);
        const dx = px - tx;
        const dy = py - ty;
        const dz = pz - tz;
        this.dist = Math.max(0.05, Math.hypot(dx, dy, dz));
        this.pitch = clamp(Math.asin(dy / this.dist), -1.55, 1.55);
        this.yaw = Math.atan2(dx, dz);
        this.apply();
    }

    // Birdseye establishing shot: park the camera OUTSIDE `aabb` (center + half
    // extents) at `azimuth`/`elevation`, far enough that the bounding sphere fits
    // the frustum with `margin` headroom. Looks down at the scene centre.
    frame(aabb, { azimuth = 35, elevation = 34, margin = 1.35 } = {}) {
        const c = aabb.center;
        const r = Math.max(0.1, aabb.halfExtents.length());
        const cam = this.cam.camera;
        const fovV = cam.fov * DEG;
        const aspect = cam.aspectRatio || 1;
        const fovH = 2 * Math.atan(Math.tan(fovV / 2) * aspect);
        const dist = (r / Math.sin(Math.min(fovV, fovH) / 2)) * margin;

        const az = azimuth * DEG;
        const el = elevation * DEG;
        const dir = new Vec3(
            Math.cos(el) * Math.sin(az),
            Math.sin(el),
            Math.cos(el) * Math.cos(az),
        );
        this.setPose(
            c.x + dir.x * dist, c.y + dir.y * dist, c.z + dir.z * dist,
            c.x, c.y, c.z,
        );
        return { dist, radius: r };
    }

    apply() {
        const t = this.target;
        const cp = Math.cos(this.pitch);
        this.cam.setPosition(
            t.x + this.dist * cp * Math.sin(this.yaw),
            t.y + this.dist * Math.sin(this.pitch),
            t.z + this.dist * cp * Math.cos(this.yaw),
        );
        this.cam.lookAt(t.x, t.y, t.z);
    }

    update(dt) {
        if (this.enabled && this.pressed.size) {
            const sprint = this.pressed.has("ShiftLeft") || this.pressed.has("ShiftRight");
            const step = this.speed * (sprint ? 3 : 1) * Math.min(dt, 0.05);
            // Walk from the yaw only (WASD stays level; Q/E/Space handle vertical).
            const fx = -Math.sin(this.yaw);
            const fz = -Math.cos(this.yaw);
            let mx = 0, my = 0, mz = 0;
            if (this.pressed.has("KeyW")) { mx += fx; mz += fz; }
            if (this.pressed.has("KeyS")) { mx -= fx; mz -= fz; }
            if (this.pressed.has("KeyD")) { mx += -fz; mz += fx; }
            if (this.pressed.has("KeyA")) { mx -= -fz; mz -= fx; }
            if (this.pressed.has("KeyE") || this.pressed.has("Space")) my += 1;
            if (this.pressed.has("KeyQ")) my -= 1;
            const len = Math.hypot(mx, my, mz);
            if (len > 0) {
                this.target.x += (mx / len) * step;
                this.target.y += (my / len) * step;
                this.target.z += (mz / len) * step;
            }
        }
        this.apply();
    }

    _onDown(ev) {
        if (!this.enabled) return;
        this.dragging = ev.button === 0 ? 1 : 2;
        this.lastX = ev.clientX;
        this.lastY = ev.clientY;
        this.canvas.setPointerCapture?.(ev.pointerId);
    }

    _onMove(ev) {
        if (!this.enabled || !this.dragging) return;
        const dx = ev.clientX - this.lastX;
        const dy = ev.clientY - this.lastY;
        this.lastX = ev.clientX;
        this.lastY = ev.clientY;
        if (this.dragging === 1) {
            this.yaw -= dx * 0.005;
            this.pitch = clamp(this.pitch + dy * 0.005, -1.55, 1.55);
        } else {
            // Screen-plane pan of the target, scaled by distance so it tracks the
            // cursor at any zoom.
            const k = this.dist * 0.0016;
            const rx = -Math.cos(this.yaw);
            const rz = Math.sin(this.yaw);
            const fx = -Math.sin(this.yaw);
            const fz = -Math.cos(this.yaw);
            this.target.x -= (dx * rx + dy * fx) * k;
            this.target.z -= (dx * rz + dy * fz) * k;
            this.target.y += dy * k;
        }
    }

    _onUp(ev) {
        this.dragging = 0;
        this.canvas.releasePointerCapture?.(ev.pointerId);
    }

    _onWheel(ev) {
        if (!this.enabled) return;
        ev.preventDefault();
        this.dist = clamp(this.dist * Math.exp(ev.deltaY * 0.0012), 0.05, 100000);
    }

    _onKeyDown(ev) {
        if (!this.enabled || !MOVE_CODES.has(ev.code)) return;
        const tag = ev.target && ev.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        this.pressed.add(ev.code);
        if (ev.code === "Space") ev.preventDefault();
    }

    _onKeyUp(ev) {
        this.pressed.delete(ev.code);
    }

    _onBlur() {
        this.pressed.clear();
        this.dragging = 0;
    }
}
