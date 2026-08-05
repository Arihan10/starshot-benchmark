// SOG-LOD playground UI. Builds the control panel from the lever catalogue, picks
// a source from both asset servers, shapes the network the splats arrive over,
// compiles new streamed bundles, runs the live HUD, and hosts the adaptive
// controller that maximizes quality at a target frame rate.
//
// Two asset origins are merged into one picker:
//   * the API server's `/sog/catalog` — the real pipeline output (per-cell
//     trained/healed splats), delivered by server/app/api/sog.py with byte
//     ranges, validators, cache policy and optional network shaping. Its
//     `sources` half is what the compile panel builds from.
//   * this static server's `/api/assets` — whatever sits under
//     client/public/assets/, served plain.
// Only the API's URLs can be shaped, since only that server implements it.

import { SplatViewer } from "./viewer.js";
import { GROUPS, LEVERS, PRESETS } from "./levers.js";
import { control, group, button, note } from "./ui.js";
import { AutoQuality, OWNED_KEYS } from "./quality.js";
import { LodCompiler } from "./compile.js";

const $ = (id) => document.getElementById(id);

const API_ORIGIN = document
    .querySelector('meta[name="server-url"]')
    ?.getAttribute("content") ?? "";

const MAX_DPR = window.devicePixelRatio || 1;

// --- formatting -------------------------------------------------------------
const nfmt = (n) => Math.round(n).toLocaleString();
const bytesFmt = (b) =>
    b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${(b / (1 << 10)).toFixed(0)} KB`;
const budgetFmt = (v) => (v <= 0 ? "uncapped" : `${(v / 1e6).toFixed(2)}M`);

// --- network shaping --------------------------------------------------------
// The shape rides in the URL PATH rather than a query string because the engine
// derives every chunk URL from the manifest's directory, dropping any query. So
// a shaped manifest URL shapes the whole bundle.
const SHAPE_PRESETS = [
    { id: "direct", label: "localhost", delay: 0, kbps: 0, jitter: 0 },
    { id: "wifi", label: "fast wifi", delay: 15, kbps: 80_000, jitter: 5 },
    { id: "broadband", label: "broadband", delay: 40, kbps: 25_000, jitter: 10 },
    { id: "lte", label: "4G", delay: 70, kbps: 12_000, jitter: 30 },
    { id: "slow", label: "slow 3G", delay: 200, kbps: 1_600, jitter: 80 },
];

const SHAPE_FIELDS = [
    {
        key: "delay",
        kind: "range",
        label: "round-trip delay",
        min: 0, max: 600, step: 5, int: true, def: 0,
        fmt: (v) => (v ? `${v} ms` : "none"),
        hint:
            "Latency the server adds before the headers of EVERY file it sends, so " +
            "a refine that needs 40 chunks pays 40 round trips. This is what makes " +
            "chunk count and fetch concurrency matter — on localhost they don't.",
    },
    {
        key: "kbps",
        kind: "range",
        label: "link bandwidth",
        min: 0, max: 100_000, step: 500, int: true, def: 0,
        fmt: (v) => (v ? `${(v / 1000).toFixed(1)} Mbps` : "uncapped"),
        hint:
            "One cap SHARED by every response in flight, so it behaves like a real " +
            "link: raising fetch concurrency divides this pipe rather than widening it.",
    },
    {
        key: "jitter",
        kind: "range",
        label: "jitter",
        min: 0, max: 300, step: 5, int: true, def: 0,
        fmt: (v) => (v ? `+0-${v} ms` : "none"),
        hint:
            "Added on top of the delay, uniformly, so chunks land out of order the " +
            "way they do on a real link — which is when the underfill limit earns its keep.",
    },
];

const CACHE_SPEC = {
    key: "cache",
    kind: "select",
    label: "cache policy",
    def: "default",
    options: [
        { value: "default", label: "chunks immutable, manifest revalidates" },
        { value: "immutable", label: "everything immutable" },
        { value: "revalidate", label: "revalidate everything" },
        { value: "nostore", label: "no-store (always cold)" },
    ],
    hint:
        "Overrides the Cache-Control the server sends. Chunk data never changes in " +
        "place so it is immutable by default and a revisit costs nothing; no-store " +
        "replays a genuine first visit without clearing the browser cache.",
};

const shape = { delay: 0, kbps: 0, jitter: 0, cache: "default" };
const shapeControls = {};
const presetButtons = {};

function shapeToken() {
    const parts = [];
    if (shape.delay) parts.push(`d${shape.delay}`);
    if (shape.kbps) parts.push(`k${shape.kbps}`);
    if (shape.jitter) parts.push(`j${shape.jitter}`);
    if (shape.cache !== "default") parts.push(shape.cache);
    return parts.length ? parts.join("-") : "direct";
}

// `/sog/f/<root>/<path>` is the unshaped delivery route; `/sog/n/<shape>/…` is the
// same bytes through a profile. Anything else (this server's own /assets) has no
// shaped form and is returned untouched.
function applyShape(url) {
    const token = shapeToken();
    if (token === "direct" || !url.includes("/sog/f/")) return url;
    return url.replace("/sog/f/", `/sog/n/${token}/`);
}

const shapeable = (url) => url.includes("/sog/f/");

// --- viewer + lever controls -------------------------------------------------
// Top-level await: the graphics device is created before the Application so the
// viewer can ask for WebGPU, which is what unlocks the GPU-sort renderer and with
// it per-node frustum culling. `?device=webgl2` forces the fallback path for
// comparison (or if a driver misbehaves).
const viewer = await SplatViewer.create($("view"), {
    deviceType: new URLSearchParams(location.search).get("device") ?? undefined,
});
const controls = {}; // lever key -> control handle
const presetLeverButtons = {};
let compiler = null;

const auto = new AutoQuality({
    apply: (key, value) => driveLever(key, value),
    maxDpr: MAX_DPR,
});

window.__playground = { viewer, auto, shape, controls }; // console / automation handle

// Write a lever AND move its control, without re-firing the control's own
// listener — how the adaptive controller keeps the panel showing the truth.
function driveLever(key, value) {
    controls[key]?.set(value);
    viewer.set(key, value);
}

// LOD selection and chunk fetching only exist for a streamed bundle — a single
// .sog is one request for the whole model. Grey them out rather than let a slider
// pretend to do something.
function setGroupEnabled(groupId, on) {
    for (const handle of Object.values(controls)) {
        if (handle.spec.group === groupId) handle.setDisabled(!on);
    }
    $(`grp-${groupId}`)?.parentElement?.classList.toggle("inert", !on);
}

// A whole-panel stance. Turns adaptive quality off first — it owns several of these
// levers, so leaving it on would silently overwrite half the preset.
function applyPreset(preset) {
    if (controls.autoQuality?.read()) controls.autoQuality.reset();
    for (const lever of LEVERS) {
        const wanted = preset.values ? preset.values[lever.key] : lever.def;
        if (wanted === undefined) continue;
        driveLever(lever.key, wanted === null ? lever.def : wanted);
    }
    // The LOD range is bounded by the loaded asset, not by the preset.
    clampLodRange();
    for (const [id, b] of Object.entries(presetLeverButtons)) {
        b.classList.toggle("active", id === preset.id);
    }
}

function buildPresetRow() {
    const host = $("lever-presets");
    for (const preset of PRESETS) {
        presetLeverButtons[preset.id] = button(preset.label, {
            title: `${preset.blurb}\n\n${preset.hint}`,
            onClick: () => applyPreset(preset),
        });
        host.appendChild(presetLeverButtons[preset.id]);
    }
}

function buildPanel() {
    const host = $("groups");
    for (const spec of GROUPS) {
        const { box, body } = group(spec);
        host.appendChild(box);
        if (spec.id === "compile") {
            compiler = new LodCompiler({
                host: body,
                apiOrigin: API_ORIGIN,
                onBuilt: onBundleBuilt,
            });
            continue;
        }
        if (spec.id === "delivery") {
            buildDeliveryGroup(body);
            continue;
        }
        if (spec.id === "adaptive") {
            buildAdaptiveGroup(body);
            continue;
        }
        for (const lever of LEVERS.filter((l) => l.group === spec.id)) {
            const handle = control(lever, (v) => {
                viewer.set(lever.key, v);
                // A hand-tuned lever means the panel is no longer any preset.
                for (const b of Object.values(presetLeverButtons)) b.classList.remove("active");
            });
            controls[lever.key] = handle;
            body.appendChild(handle.ctl);
        }
    }
}

// --- delivery (server-side) --------------------------------------------------
function buildDeliveryGroup(host) {
    const presets = document.createElement("div");
    presets.className = "presets";
    for (const preset of SHAPE_PRESETS) {
        presetButtons[preset.id] = button(preset.label, {
            title: `${preset.delay} ms round trip, ${preset.kbps ? `${preset.kbps / 1000} Mbps` : "uncapped"}`,
            onClick: () => {
                for (const field of SHAPE_FIELDS) {
                    shapeControls[field.key].set(preset[field.key]);
                    shape[field.key] = preset[field.key];
                }
                markPresetActive(preset.id);
                reloadForShape();
            },
        });
        presets.appendChild(presetButtons[preset.id]);
    }
    host.appendChild(presets);

    for (const field of SHAPE_FIELDS) {
        const handle = control(field, (v) => {
            shape[field.key] = v;
            markPresetActive(null);
        });
        // Shaping only takes effect on the next load, so commit on release rather
        // than on every pixel of the drag.
        handle.input.addEventListener("change", reloadForShape);
        shapeControls[field.key] = handle;
        host.appendChild(handle.ctl);
    }

    const cache = control(CACHE_SPEC, (v) => {
        shape.cache = v;
        markPresetActive(null);
        reloadForShape();
    });
    shapeControls.cache = cache;
    host.appendChild(cache.ctl);

    const status = note("");
    status.id = "shape-note";
    host.appendChild(status);
}

function markPresetActive(id) {
    for (const [pid, b] of Object.entries(presetButtons)) {
        b.classList.toggle("active", pid === id);
    }
}

function setShapingEnabled(on) {
    for (const handle of Object.values(shapeControls)) handle.setDisabled(!on);
    for (const b of Object.values(presetButtons)) b.disabled = !on;
    $("shape-note").textContent = on
        ? "Changing any of these reloads the asset, since the shape is part of its URL."
        : "This asset comes from the static client server, which has no shaping — pick a /sog source to enable it.";
}

function reloadForShape() {
    if (current && shapeable(current.url)) loadSource(current);
}

// --- adaptive quality --------------------------------------------------------
const TARGET_FPS_SPEC = {
    key: "targetFps",
    kind: "select",
    label: "target frame rate",
    def: "60",
    options: [
        { value: "30", label: "30 fps (cinematic floor)" },
        { value: "45", label: "45 fps" },
        { value: "60", label: "60 fps" },
        { value: "90", label: "90 fps" },
        { value: "120", label: "120 fps" },
    ],
    hint:
        "The frame rate the controller must hold. Everything above it is spent on " +
        "quality, so a lower target buys a sharper, denser scene and a higher one " +
        "buys smoothness.",
};

const AUTO_SPEC = {
    key: "autoQuality",
    kind: "toggle",
    label: "adaptive quality",
    def: false,
    hint:
        "Climbs a quality ladder — render scale, splat budget, foveation, " +
        "anti-alias — to the highest rung that still holds the target, and drops a " +
        "rung the moment it doesn't. It starts high and works down, only climbs " +
        "once the network is idle (a budget raise mid-refine both skews the " +
        "measurement and storms the server), and backs off exponentially from a " +
        "rung this machine has already failed. It drives the levers it owns, so " +
        "watch them move.",
};

let autoSnapshot = null;

function buildAdaptiveGroup(host) {
    const toggle = control(AUTO_SPEC, (on) => setAuto(on));
    host.appendChild(toggle.ctl);
    controls.autoQuality = toggle;

    const target = control(TARGET_FPS_SPEC, (v) => auto.setTargetFps(parseInt(v, 10)));
    host.appendChild(target.ctl);
    controls.targetFps = target;
    auto.setTargetFps(parseInt(TARGET_FPS_SPEC.def, 10));
}

function setAuto(on) {
    if (on) {
        // Remember what the panel had, so switching back doesn't leave the user
        // with whatever rung the ladder happened to stop on.
        autoSnapshot = Object.fromEntries(OWNED_KEYS.map((k) => [k, controls[k].read()]));
    }
    for (const key of OWNED_KEYS) controls[key].setDisabled(on);
    auto.setEnabled(on);
    if (!on && autoSnapshot) {
        for (const [key, value] of Object.entries(autoSnapshot)) driveLever(key, value);
        autoSnapshot = null;
    }
}

// --- source picker ----------------------------------------------------------
async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}

async function loadCatalogue() {
    const assets = [];
    let sources = [];
    if (API_ORIGIN) {
        try {
            // `refresh` skips the server's scan cache — a rescan is always asked for
            // because something changed (usually a bundle that just finished).
            const data = await fetchJson(new URL("/sog/catalog?refresh=1", API_ORIGIN));
            sources = data.sources ?? [];
            for (const a of data.assets ?? []) {
                assets.push({
                    key: `api:${a.id}`,
                    origin: "api",
                    name: a.name,
                    url: new URL(a.url, API_ORIGIN).toString(),
                    kind: a.kind,
                    bytes: a.bytes,
                    splats: a.splats,
                    lodLevels: a.lod_levels,
                    chunkFiles: a.chunk_files,
                });
            }
        } catch (e) {
            console.warn("[playground] /sog/catalog unavailable:", e.message);
        }
    }
    try {
        const data = await fetchJson("/api/assets");
        for (const a of data.assets ?? []) {
            assets.push({
                key: `client:${a.url}`,
                origin: "client",
                name: a.name,
                url: new URL(a.url, location.origin).toString(),
                kind: a.type === "lod" ? "lod" : "sog",
                bytes: a.bytes,
                splats: a.splats,
                lodLevels: a.lodLevels ?? 1,
                chunkFiles: null,
            });
        }
    } catch {
        /* no static catalogue — the API list and the URL field still work */
    }
    return { assets, sources };
}

function describeAsset(a) {
    const bits = [a.kind === "lod" ? `LOD x${a.lodLevels ?? "?"}` : "single SOG"];
    if (a.splats) bits.push(`${nfmt(a.splats)} splats`);
    if (a.chunkFiles) bits.push(`${a.chunkFiles} chunks`);
    bits.push(bytesFmt(a.bytes ?? 0));
    return `${a.origin === "api" ? "/sog" : "static"} - ${bits.join(", ")}`;
}

function fillPicker(assets, selectKey) {
    const picker = $("src-picker");
    picker.innerHTML = "";
    if (assets.length === 0) {
        const o = document.createElement("option");
        o.textContent = "- no SOG assets found -";
        o.value = "";
        picker.appendChild(o);
        return;
    }
    for (const a of assets) {
        const o = document.createElement("option");
        o.value = a.key;
        o.textContent = `${a.name} — ${describeAsset(a)}`;
        picker.appendChild(o);
    }
    if (selectKey) picker.value = selectKey;
}

async function rescan(selectKey) {
    const found = await loadCatalogue();
    catalogue = found.assets;
    fillPicker(catalogue, selectKey ?? current?.key);
    compiler?.setSources(found.sources);
    return found;
}

// A freshly compiled bundle is what you asked for a moment ago, so stream it.
async function onBundleBuilt(source, manifestUrl) {
    const found = await rescan();
    const match = found.assets.find((a) => a.url === manifestUrl);
    await loadSource(match ?? { key: `built:${source.id}`, origin: "api", name: source.name, url: manifestUrl });
}

// --- loading ----------------------------------------------------------------
let catalogue = [];
let current = null;
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

// The LOD-range knobs are bounded by however many levels the loaded asset actually
// has, which nothing else can know. `openUp` also widens the ceiling to that max —
// what a fresh load wants, whereas a preset asking for a tighter cap must keep it.
function clampLodRange({ openUp = false } = {}) {
    const maxLod = Math.max(0, (summary.lodLevels ?? 1) - 1);
    for (const key of ["lodRangeMin", "lodRangeMax"]) {
        const handle = controls[key];
        if (!handle) continue;
        handle.setMax(maxLod);
        const wanted = openUp && key === "lodRangeMax" ? maxLod : Math.min(handle.read(), maxLod);
        handle.set(wanted);
        viewer.set(key, handle.read());
    }
}

async function loadSource(asset) {
    if (!asset) return;
    current = asset;
    const url = applyShape(asset.url);
    setShapingEnabled(shapeable(asset.url));
    const token = shapeToken();
    showLoader(`streaming ${asset.name}${token === "direct" ? "" : ` over ${token}`} …`);
    $("src-url").value = url;
    try {
        summary = await viewer.load(url);
        const streamed = summary.type === "lod";
        setGroupEnabled("fetch", streamed);
        setGroupEnabled("lod", streamed);
        clampLodRange({ openUp: true });
        $("loader").hidden = true;
    } catch (e) {
        showError(`load failed: ${e.message ?? e}`);
    }
}

// A hand-typed URL isn't in the catalogue; wrap it so the same path handles it.
function loadRawUrl(raw) {
    if (!raw) return;
    loadSource({ key: "custom", origin: "custom", name: "custom URL", url: raw });
}

// --- HUD --------------------------------------------------------------------
const fpsClass = (fps) => (fps >= 50 ? "good" : fps >= 30 ? "warn" : "bad");

function hud() {
    const s = viewer.stats();
    auto.step(s.frameMs, s.pendingLoads, performance.now());

    $("s-type").textContent =
        summary.type === "lod" ? "streamed LOD" : summary.type === "single" ? "single SOG" : "—";
    $("s-lods").textContent = summary.lodLevels ?? "—";
    $("s-splats").textContent = nfmt(summary.splats ?? 0);
    $("s-rendered").textContent = nfmt(s.splatsRendered);
    $("s-files").textContent = `${s.filesLoaded} / ${s.filesTotal}`;
    $("s-files-bar").style.width = `${s.filesTotal ? (100 * s.filesLoaded) / s.filesTotal : 0}%`;
    $("s-cooling").textContent = String(s.filesCooling);

    const inflight = $("s-inflight");
    inflight.textContent =
        s.pendingLoads > 0 ? `${s.pendingLoads} loading` : s.resolved ? "resolved" : "settling";
    inflight.className = "v " + (s.pendingLoads > 0 ? "warn" : s.resolved ? "good" : "");

    $("s-bytes").textContent = bytesFmt(s.bytes);
    $("s-requests").textContent = s.cacheHits
        ? `${nfmt(s.requests)} (${nfmt(s.cacheHits)} cached)`
        : nfmt(s.requests);
    $("s-latency").textContent = s.requests ? `${s.latencyMs.toFixed(0)} ms` : "—";
    $("s-budget").textContent = budgetFmt(viewer.config.splatBudget);
    $("s-churn").textContent = `${s.bufferChurn.toFixed(0)}%`;
    const cull = $("s-cull");
    // With culling on, the useful number is what SURVIVED it against what was
    // resident — the reduction is the whole point and it happens after the CPU's
    // own count.
    cull.textContent = !s.culling.active
        ? s.culling.reason
        : s.splatsDrawn === null
            ? `${nfmt(s.culling.boundsEntries)} bounds`
            : `${nfmt(s.splatsDrawn)} of ${nfmt(s.splatsRendered)}`;
    cull.className = "v " + (s.culling.active ? "good" : "warn");
    $("s-cull-why").textContent = s.culling.active
        ? `${nfmt(s.culling.boundsEntries)} node bounds tested per frame`
        : "";
    $("s-api").textContent = s.culling.api;
    $("s-dpr").textContent = `${s.pixelRatio.toFixed(2)}x`;
    $("s-dist").textContent = `${s.camDist.toFixed(1)} m`;
    $("s-frame").textContent = `${s.frameMs.toFixed(1)} ms`;
    const fpsEl = $("s-fps");
    fpsEl.textContent = s.fps.toFixed(0);
    fpsEl.className = "v " + fpsClass(s.fps);

    const a = auto.state();
    $("s-tier").textContent = a.enabled ? `${a.tier} / ${a.maxTier}` : "off";
    $("s-tier-bar").style.width = a.enabled ? `${(100 * a.tier) / a.maxTier}%` : "0%";
    $("s-tier-why").textContent = a.enabled ? a.reason : "";

    requestAnimationFrame(hud);
}

// --- wire up ----------------------------------------------------------------
buildPanel();
buildPresetRow();
markPresetActive("direct");

$("toggle-desc").addEventListener("change", (e) => {
    document.body.classList.toggle("lean", !e.target.checked);
});
$("reframe").addEventListener("click", () => viewer.reframe());
$("src-picker").addEventListener("change", (e) => {
    const asset = catalogue.find((a) => a.key === e.target.value);
    if (asset) loadSource(asset);
});
$("src-load").addEventListener("click", () => loadRawUrl($("src-url").value.trim()));
$("src-url").addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadRawUrl($("src-url").value.trim());
});
$("src-refresh").addEventListener("click", () => rescan());

(async () => {
    const found = await rescan();
    const wanted = new URLSearchParams(location.search).get("src");
    // Prefer a streamed bundle on boot — LOD is what there is to play with.
    const initial =
        (wanted && catalogue.find((a) => a.url.endsWith(wanted) || a.key === wanted)) ||
        catalogue.find((a) => a.kind === "lod") ||
        catalogue[0];
    fillPicker(catalogue, initial?.key);
    if (initial) await loadSource(initial);
    else if (wanted) loadRawUrl(wanted);
    else {
        setShapingEnabled(false);
        showError(
            found.sources.length
                ? "no bundle yet — open “compile a bundle” to build one from a trained PLY"
                : "no SOG assets and no trained PLY on the API host — train a cell first, or pass ?src=<url>",
        );
    }
    requestAnimationFrame(hud);
})();
