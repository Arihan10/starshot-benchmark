// SOG-LOD playground UI. Builds the control panel, drives asset loading, runs the
// live HUD, and hosts the adaptive "dynamic loading" controller that trades splat
// budget + render resolution to hold a smooth frame rate while the engine streams
// LOD chunks in the background.

import { SplatViewer } from "./viewer.js";

const $ = (id) => document.getElementById(id);

// --- formatting -------------------------------------------------------------
const nfmt = (n) => Math.round(n).toLocaleString();
const bytesFmt = (b) =>
    b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${(b / (1 << 10)).toFixed(0)} KB`;
const budgetFmt = (v) => (v <= 0 ? "uncapped" : `${(v / 1e6).toFixed(2)}M`);

// --- control specs (single source of truth for panel + viewer defaults) -----
const GROUPS = {
    "grp-lod": [
        { key: "lodBaseDistance", label: "base distance", min: 0.5, max: 80, step: 0.5, def: 5, fmt: (v) => `${v} m` },
        { key: "lodMultiplier", label: "multiplier", min: 1.2, max: 6, step: 0.1, def: 3, fmt: (v) => `${v.toFixed(1)}×` },
        { key: "lodRangeMin", label: "finest LOD (min)", min: 0, max: 6, step: 1, def: 0, int: true, fmt: (v) => String(v) },
        { key: "lodRangeMax", label: "coarsest LOD (max)", min: 0, max: 6, step: 1, def: 6, int: true, fmt: (v) => String(v) },
    ],
    "grp-stream": [
        { key: "splatBudget", label: "splat budget", min: 0, max: 3_000_000, step: 50_000, def: 0, fmt: budgetFmt },
        { key: "lodUpdateDistance", label: "update distance", min: 0, max: 10, step: 0.25, def: 1, fmt: (v) => `${v} m` },
        { key: "lodUpdateAngle", label: "update angle", min: 0, max: 90, step: 1, def: 0, int: true, fmt: (v) => `${v}°` },
        { key: "lodBehindPenalty", label: "behind penalty", min: 1, max: 8, step: 0.5, def: 1, fmt: (v) => `${v.toFixed(1)}×` },
    ],
};

const viewer = new SplatViewer($("view"));
window.__playground = { viewer }; // debug handle (console / automation)
const sliders = {}; // key -> { input, val, spec }

function buildControls() {
    for (const [groupId, items] of Object.entries(GROUPS)) {
        const host = $(groupId);
        for (const spec of items) {
            const ctl = document.createElement("div");
            ctl.className = "ctl";
            ctl.innerHTML =
                `<div class="row"><label>${spec.label}</label><span class="val"></span></div>` +
                `<input type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${spec.def}">`;
            const input = ctl.querySelector("input");
            const val = ctl.querySelector(".val");
            const apply = () => {
                const v = spec.int ? parseInt(input.value, 10) : parseFloat(input.value);
                val.textContent = spec.fmt(v);
                viewer.set(spec.key, v);
            };
            input.addEventListener("input", apply);
            host.appendChild(ctl);
            sliders[spec.key] = { input, val, spec };
            apply(); // sync viewer.config to the panel defaults
        }
    }
}

// --- source picker ----------------------------------------------------------
async function loadCatalogue(selectUrl) {
    const picker = $("src-picker");
    picker.innerHTML = "";
    let assets = [];
    try {
        assets = (await (await fetch("/api/assets", { cache: "no-store" })).json()).assets ?? [];
    } catch {
        /* no catalogue — the URL field still works */
    }
    if (assets.length === 0) {
        const o = document.createElement("option");
        o.textContent = "— no assets under client/public/assets/ —";
        o.value = "";
        picker.appendChild(o);
    }
    for (const a of assets) {
        const o = document.createElement("option");
        o.value = a.url;
        const detail =
            a.type === "lod"
                ? `LOD×${a.lodLevels ?? "?"}, ${nfmt(a.splats ?? 0)} splats, ${bytesFmt(a.bytes)}`
                : `single SOG, ${bytesFmt(a.bytes)}`;
        o.textContent = `${a.name} — ${detail}`;
        picker.appendChild(o);
    }
    // If the loaded URL isn't in the catalogue, show it as a custom entry.
    if (selectUrl && !assets.some((a) => a.url === selectUrl)) {
        const o = document.createElement("option");
        o.value = selectUrl;
        o.textContent = `custom — ${selectUrl}`;
        picker.appendChild(o);
    }
    if (selectUrl) picker.value = selectUrl;
    return assets;
}

// --- loading ----------------------------------------------------------------
let summary = { type: "—", lodLevels: 1, splats: 0, filesTotal: 1 };

function showLoader(msg) {
    const l = $("loader");
    l.hidden = false;
    l.classList.remove("err");
    l.querySelector(".spin").style.display = "";
    $("loader-msg").textContent = msg;
}
function showError(msg) {
    const l = $("loader");
    l.hidden = false;
    l.classList.add("err");
    l.querySelector(".spin").style.display = "none";
    $("loader-msg").textContent = msg;
}

async function loadSource(url) {
    if (!url) return;
    showLoader(`streaming ${url.split("/").slice(-2).join("/")} …`);
    try {
        summary = await viewer.load(url);
        $("src-url").value = url;
        // Cap the LOD-range sliders to the asset's actual levels and re-apply so
        // the labels + viewer config match (default: allow the full range).
        const maxLod = Math.max(0, summary.lodLevels - 1);
        for (const key of ["lodRangeMin", "lodRangeMax"]) {
            const s = sliders[key];
            s.input.max = String(maxLod);
            if (key === "lodRangeMax" || parseInt(s.input.value, 10) > maxLod) {
                s.input.value = String(key === "lodRangeMax" ? maxLod : 0);
            }
            s.input.dispatchEvent(new Event("input"));
        }
        $("loader").hidden = true;
    } catch (e) {
        showError(`load failed: ${e.message ?? e}`);
    }
}

// --- adaptive dynamic loading ----------------------------------------------
// Hold ~55 fps by first trimming render resolution, then the splat budget, when
// frames run long — and restoring both when there's headroom. This is the
// "optimize the client experience" half: the engine keeps streaming detail, and
// this keeps the frame rate live while it does.
const MAX_DPR = window.devicePixelRatio || 1;
const auto = { on: false, dpr: MAX_DPR, budget: 1_500_000, t: 0 };

function adaptiveStep(frameMs, now) {
    if (!auto.on || now - auto.t < 350) return;
    auto.t = now;
    if (frameMs > 22) {
        if (auto.dpr > 0.6) auto.dpr = Math.max(0.6, auto.dpr - 0.1);
        else auto.budget = Math.max(300_000, auto.budget - 250_000);
    } else if (frameMs < 15) {
        if (auto.budget < 3_000_000) auto.budget = Math.min(3_000_000, auto.budget + 250_000);
        else if (auto.dpr < MAX_DPR) auto.dpr = Math.min(MAX_DPR, auto.dpr + 0.1);
    }
    viewer.setPixelRatio(auto.dpr);
    viewer.set("splatBudget", auto.budget);
}

function setAuto(on) {
    auto.on = on;
    sliders.splatBudget.input.disabled = on;
    if (on) {
        auto.dpr = viewer.app.graphicsDevice.maxPixelRatio;
        auto.budget = viewer.config.splatBudget || 1_500_000;
    } else {
        viewer.setPixelRatio(MAX_DPR);
        viewer.set("splatBudget", parseInt(sliders.splatBudget.input.value, 10));
    }
}

// --- HUD --------------------------------------------------------------------
function fpsClass(fps) {
    return fps >= 50 ? "good" : fps >= 30 ? "warn" : "bad";
}

function hud() {
    const s = viewer.stats();
    adaptiveStep(s.frameMs, performance.now());

    $("s-type").textContent = summary.type === "lod" ? "streamed LOD" : summary.type === "single" ? "single SOG" : "—";
    $("s-lods").textContent = summary.lodLevels ?? "—";
    $("s-splats").textContent = nfmt(summary.splats ?? 0);
    $("s-files").textContent = `${s.filesLoaded} / ${s.filesTotal}`;
    $("s-files-bar").style.width = `${s.filesTotal ? (100 * s.filesLoaded) / s.filesTotal : 0}%`;
    $("s-bytes").textContent = bytesFmt(s.bytes);
    $("s-budget").textContent = budgetFmt(viewer.config.splatBudget);
    $("s-dpr").textContent = s.pixelRatio.toFixed(2) + "×";
    $("s-dist").textContent = s.camDist.toFixed(1) + " m";
    $("s-frame").textContent = s.frameMs.toFixed(1) + " ms";
    const fpsEl = $("s-fps");
    fpsEl.textContent = s.fps.toFixed(0);
    fpsEl.className = "v " + fpsClass(s.fps);

    requestAnimationFrame(hud);
}

// --- wire up ----------------------------------------------------------------
buildControls();

$("debug").addEventListener("change", (e) => viewer.set("debug", e.target.value));
$("auto-quality").addEventListener("change", (e) => setAuto(e.target.checked));
$("reframe").addEventListener("click", () => viewer.reframe());
$("src-picker").addEventListener("change", (e) => e.target.value && loadSource(e.target.value));
$("src-load").addEventListener("click", () => loadSource($("src-url").value.trim()));
$("src-url").addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadSource($("src-url").value.trim());
});

(async () => {
    const wanted = new URLSearchParams(location.search).get("src");
    const assets = await loadCatalogue(wanted);
    const initial = wanted || assets.find((a) => a.type === "lod")?.url || assets[0]?.url;
    if (initial) await loadSource(initial);
    else showError("no asset to load — generate one with tools/ply-to-lod-sog.mjs or pass ?src=<url>");
    requestAnimationFrame(hud);
})();
