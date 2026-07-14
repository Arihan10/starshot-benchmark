// Full-screen Gaussian-splat viewer for Stage-2 clouds, on @mkkellogg/gaussian-
// splats-3d (orbit controls + WASM sort).
//
// LOD streaming: lazy-loads the ~10 MB base cloud (cloud.ply) for a fast first
// paint, then streams the denser detail LOD (cloud.detail.ply) in the background
// and swaps it in (removeSplatScene(base)) once it's ready.
//
// Live controls: a panel exposes the sampler's global knobs (density / overlap /
// flatness / adaptive / detail) and re-splats the open cell — POST the knobs,
// poll progress, reload in place (camera preserved) — so tuning needs no server
// restart and no manual command.

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { api } from "./api.js";
import { el } from "./ui.js";

let overlay = null;
let canvasEl = null;
let subEl = null;
let statusEl = null;
let controlsEl = null;
let viewer = null;
let openSeq = 0; // bumps on open/close to cancel in-flight loads/polls
let pollTimer = null;
let current = null; // { run, slot, model, source }
let inputs = null; // control input elements, by key

const POLL_MS = 1200;
const BYTES_PER_SPLAT = 68; // 17 float32 — for the density→size estimate

function setStatus(text, color) {
    if (!statusEl) return;
    statusEl.textContent = text || "";
    statusEl.style.color = color || "";
}

function stopPoll() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

// A 3/4 exterior view framed on the scene AABB (fallback pose when no AABB).
function framing(aabb) {
    if (!aabb || !aabb.min || !aabb.max) {
        return { position: [4, 3, 6], lookAt: [0, 1, 0] };
    }
    const { min, max } = aabb;
    const c = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    const r = Math.max(0.5, Math.hypot(size[0], size[1], size[2]) / 2);
    const d = r * 1.5;
    return { position: [c[0] + d, c[1] + d * 0.6, c[2] + d], lookAt: c };
}

// Read the live camera so a re-splat reload can keep the same view.
function captureCamera() {
    try {
        const p = viewer.camera.position;
        const t = viewer.controls.target;
        return { position: [p.x, p.y, p.z], lookAt: [t.x, t.y, t.z] };
    } catch {
        return null;
    }
}

async function teardown() {
    const v = viewer;
    viewer = null;
    if (v) {
        try {
            await v.dispose();
        } catch {
            /* disposing a partially-built viewer can throw; ignore */
        }
    }
    if (canvasEl) canvasEl.replaceChildren();
}

// Build a fresh viewer, load the base cloud, then stream the detail LOD behind it.
async function loadClouds(seq, url, detailUrl, summary, camera) {
    await teardown();
    if (seq !== openSeq) return;
    const view = camera || framing(summary && summary.scene_aabb);
    const v = new GaussianSplats3D.Viewer({
        rootElement: canvasEl,
        selfDrivenMode: true,
        useBuiltInControls: true,
        sharedMemoryForWorkers: false, // no COOP/COEP needed on the static server
        dynamicScene: false,
        sphericalHarmonicsDegree: 0, // our .ply is f_dc only (unlit)
        cameraUp: [0, 1, 0],
        initialCameraPosition: view.position,
        initialCameraLookAt: view.lookAt,
    });
    viewer = v;
    try {
        await v.addSplatScene(url, {
            showLoadingUI: true,
            splatAlphaRemovalThreshold: 1,
        });
    } catch (e) {
        if (seq === openSeq) setStatus(`failed: ${e && e.message ? e.message : e}`, "var(--red)");
        return;
    }
    if (seq !== openSeq) return;
    v.start();
    const baseN = summary && summary.splats;
    setStatus(
        (baseN ? `${baseN.toLocaleString()} splats` : "loaded") +
            (detailUrl ? " · streaming detail…" : ""),
        "var(--green)",
    );
    if (!detailUrl) return;
    // Stream the denser LOD in the background; when it lands, drop the base scene.
    v.addSplatScene(detailUrl, { showLoadingUI: false, splatAlphaRemovalThreshold: 1 })
        .then(() => {
            if (seq !== openSeq || viewer !== v) return;
            try {
                if (v.getSceneCount() > 1) v.removeSplatScene(0);
            } catch {
                /* keep both if removal fails — just a touch heavier */
            }
            const dN = summary && summary.detail && summary.detail.splats;
            setStatus(
                dN ? `${dN.toLocaleString()} splats (detail LOD)` : "detail LOD loaded",
                "var(--green)",
            );
        })
        .catch(() => {
            if (seq === openSeq) setStatus("detail LOD failed — showing base", "");
        });
}

// ---- controls panel ---------------------------------------------------------

function sliderRow(key, label, min, max, step, value, fmt) {
    const val = el("span", { class: "svc-val", text: fmt(value) });
    const input = el("input", {
        type: "range",
        class: "svc-range",
        min,
        max,
        step,
        value,
    });
    input.addEventListener("input", () => {
        val.textContent = fmt(Number(input.value));
    });
    inputs[key] = input;
    return el(
        "div",
        { class: "svc-row" },
        el("span", { class: "svc-lab", text: label }),
        input,
        val,
    );
}

function checkRow(key, label, checked) {
    const input = el("input", { type: "checkbox" });
    input.checked = !!checked;
    inputs[key] = input;
    return el("label", { class: "svc-check" }, input, el("span", { text: label }));
}

const kFmt = (v) => `${Math.round(v / 1000)}k · ~${((v * BYTES_PER_SPLAT) / 1e6).toFixed(0)}MB`;

function buildControls(summary) {
    if (!controlsEl) return;
    const p = (summary && summary.params) || {};
    const detail = summary && summary.detail;
    inputs = {};
    controlsEl.replaceChildren();

    const density = sliderRow(
        "density", "density", 20000, 1200000, 10000,
        p.target_splats || 150000, kFmt,
    );
    const radius = sliderRow(
        "radius", "overlap", 0.5, 1.6, 0.05,
        p.radius_frac != null ? p.radius_frac : 0.9, (v) => v.toFixed(2),
    );
    const flat = sliderRow(
        "flatness", "flatness", 0.02, 0.4, 0.01,
        p.flatness != null ? p.flatness : 0.1, (v) => v.toFixed(2),
    );
    const adaptive = checkRow("adaptive", "adaptive density", p.adaptive !== false);
    const detailOn = checkRow("detail", "stream detail LOD", !!detail);
    const detailDensity = sliderRow(
        "detailDensity", "detail", 100000, 2000000, 50000,
        (detail && detail.splats) || 600000, kFmt,
    );

    const btn = el("button", {
        class: "svc-resplat",
        text: "re-splat",
        onclick: () => resplat(),
    });
    inputs._btn = btn;
    const actual = el("div", { class: "svc-actual" });
    inputs._actual = actual;
    if (summary && summary.splats) {
        actual.textContent = actualText(summary);
    }

    controlsEl.append(
        el("div", { class: "svc-title", text: "sampler knobs" }),
        density,
        radius,
        flat,
        adaptive,
        detailOn,
        detailDensity,
        el("div", { class: "svc-actions" }, btn),
        actual,
    );
}

function actualText(summary) {
    const mb = (summary.bytes / 1e6).toFixed(1);
    let t = `actual: ${summary.splats.toLocaleString()} · ${mb} MB`;
    if (summary.detail) {
        t += ` (detail ${summary.detail.splats.toLocaleString()} · ${(summary.detail.bytes / 1e6).toFixed(0)} MB)`;
    }
    return t;
}

function readParams() {
    const body = {
        target_splats: Math.round(Number(inputs.density.value)),
        radius_frac: Number(inputs.radius.value),
        flatness: Number(inputs.flatness.value),
        adaptive: inputs.adaptive.checked,
    };
    if (inputs.detail.checked) {
        body.detail_splats = Math.round(Number(inputs.detailDensity.value));
    }
    return body;
}

async function resplat() {
    if (!current) return;
    const body = readParams();
    if (inputs._btn) inputs._btn.disabled = true;
    setStatus("re-splatting…", "var(--purple)");
    try {
        await api.splatStage2Start(current.run, current.slot, current.model, body);
    } catch (e) {
        setStatus(`re-splat failed: ${e.message}`, "var(--red)");
        if (inputs._btn) inputs._btn.disabled = false;
        return;
    }
    pollResplat();
}

function pollResplat() {
    stopPoll();
    const seq = openSeq;
    pollTimer = setInterval(async () => {
        if (seq !== openSeq || !current) return stopPoll();
        let st;
        try {
            st = await api.splatStage2Status(current.run, current.slot, current.model);
        } catch {
            return; // transient — retry next tick
        }
        if (seq !== openSeq) return stopPoll();
        if (st.running) {
            const phase = st.phase || "sampling";
            const prog = st.total ? ` ${st.done}/${st.total}` : "";
            setStatus(`${phase}${prog}…`, "var(--purple)");
            return;
        }
        stopPoll();
        if (inputs && inputs._btn) inputs._btn.disabled = false;
        if (st.status === "error") {
            setStatus(`re-splat failed: ${st.error || ""}`, "var(--red)");
            return;
        }
        if (st.status === "done" && st.url) {
            if (inputs && inputs._actual && st.summary) {
                inputs._actual.textContent = actualText(st.summary);
            }
            const bust = `?t=${Date.now()}`;
            loadClouds(
                openSeq,
                api.absUrl(st.url + bust),
                st.detail_url ? api.absUrl(st.detail_url + bust) : null,
                st.summary,
                captureCamera(), // keep the current view across the re-splat
            );
        }
    }, POLL_MS);
}

// ---- open / close -----------------------------------------------------------

export async function openSplatViewer(opts) {
    if (!overlay || !opts || !opts.url) return;
    const seq = ++openSeq;
    stopPoll();
    current = {
        run: opts.run,
        slot: opts.slot,
        model: opts.model,
        source: opts.source,
    };
    overlay.classList.add("open");
    subEl.textContent = opts.label || "";
    setStatus("loading…", "var(--purple)");
    buildControls(opts.summary);
    await loadClouds(seq, opts.url, opts.detailUrl || null, opts.summary, null);
}

export function isSplatViewerOpen() {
    return !!overlay && overlay.classList.contains("open");
}

export function closeSplatViewer() {
    if (!isSplatViewerOpen()) return;
    openSeq++; // cancel any in-flight load/poll
    stopPoll();
    overlay.classList.remove("open");
    setStatus("");
    subEl.textContent = "";
    current = null;
    void teardown();
}

export function initSplatViewer() {
    overlay = document.getElementById("splat-viewer");
    canvasEl = document.getElementById("splat-viewer-canvas");
    subEl = document.getElementById("splat-viewer-sub");
    statusEl = document.getElementById("splat-viewer-status");
    controlsEl = document.getElementById("splat-viewer-controls");
    document
        .getElementById("splat-viewer-close")
        .addEventListener("click", closeSplatViewer);
    // Capture-phase Esc: close the viewer (topmost) and swallow the event so the
    // splat screen's own Esc handler underneath doesn't also close.
    document.addEventListener(
        "keydown",
        (ev) => {
            if (ev.key === "Escape" && isSplatViewerOpen()) {
                ev.stopImmediatePropagation();
                closeSplatViewer();
            }
        },
        true,
    );
}
