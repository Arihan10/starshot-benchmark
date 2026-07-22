// The Stage-5 reference-frame viewer: browse a cell's `refs/frames/*.szf` as a
// lazy thumbnail matrix + a full-resolution single-frame inspector. Each SZF
// carries three planes — RGB (unlit albedo), alpha (accumulated coverage), and
// depth (planar-Z, log-uint16 codes) — all decoded client-side (see splat/stage5.py
// for the container + depth codec this mirrors). Fetches raw .szf bytes from the
// artifact server and unpacks them here; the pipeline is never touched.

import { el, toast } from "./ui.js";
import { api } from "./api.js";
import { ZSTDDecoder } from "three/addons/libs/zstddec.module.js";

const FRAME_MAGIC = 0x31465a53; // "SZF1" read as little-endian u32 (S=0x53,Z=0x5a,F=0x46,1=0x31)
const HEADER_BYTES = 16;
const FILTER_SUBLEFT = 1;
const DEPTH_CODE_MAX = 65535; // matches stage5._DEPTH_CODE_MAX; code 0 = background
const CUBE_FACES = ["+x", "-x", "+y", "-y", "+z", "-z"];
const PLANES = [
    ["rgb", "RGB"],
    ["alpha", "alpha"],
    ["depth", "depth"],
];

const THUMB = 128;
const RAW_CACHE_CAP = 64; // recently-fetched .szf buffers, for snappy detail/plane flips
const TILE_CACHE_CAP = 900; // rendered thumbnail tiles kept live (~60 MB at THUMB²)
const PAGE_SIZE = 120; // thumbnails materialized per page — the rest load on scroll / "load more"
const PAGE_PREFETCH_PX = 600; // keep the grid filled this far past the viewport bottom

let decoderPromise = null;
let overlayEl = null;
let styleInjected = false;
let escBound = false;
let state = null; // per-open session (see openRefsViewer)

// --- zstd (three's KTX2 WASM decoder, reused so there's no new dependency) -----

function ensureDecoder() {
    if (!decoderPromise) {
        const d = new ZSTDDecoder();
        decoderPromise = d.init().then(() => d);
    }
    return decoderPromise;
}

// --- SZF decode ----------------------------------------------------------------

// Wrapping prefix-sum along each row — the inverse of stage5's "Sub" predictor.
// RGBA is 4-interleaved (left neighbour is 4 bytes back); depth is scalar u16.
function unfilterRgba(px, res) {
    for (let y = 0; y < res; y++) {
        let i = y * res * 4 + 4;
        for (let x = 1; x < res; x++, i += 4) {
            px[i] += px[i - 4];
            px[i + 1] += px[i - 3];
            px[i + 2] += px[i - 2];
            px[i + 3] += px[i - 1];
        }
    }
}

function unfilterDepth(codes, res) {
    for (let y = 0; y < res; y++) {
        const base = y * res;
        for (let x = 1; x < res; x++) codes[base + x] += codes[base + x - 1];
    }
}

// Parse one SZF buffer and decompress only the requested planes. Returns
// { res, rgba: Uint8Array|null, depth: Uint16Array|null }.
async function decodeFrame(buf, { rgba: wantRgba = true, depth: wantDepth = true } = {}) {
    const dv = new DataView(buf);
    if (buf.byteLength < HEADER_BYTES || dv.getUint32(0, true) !== FRAME_MAGIC) {
        throw new Error("not an SZF frame");
    }
    const res = dv.getUint16(4, true);
    const filter = dv.getUint8(6);
    const rgbaClen = dv.getUint32(8, true);
    const depthClen = dv.getUint32(12, true);
    if (buf.byteLength !== HEADER_BYTES + rgbaClen + depthClen) {
        throw new Error("truncated SZF frame");
    }
    const dec = await ensureDecoder();
    const bytes = new Uint8Array(buf);

    let rgba = null;
    if (wantRgba) {
        const comp = bytes.subarray(HEADER_BYTES, HEADER_BYTES + rgbaClen);
        rgba = dec.decode(comp, res * res * 4);
        if (filter === FILTER_SUBLEFT) unfilterRgba(rgba, res);
    }
    let depth = null;
    if (wantDepth) {
        const off = HEADER_BYTES + rgbaClen;
        const comp = bytes.subarray(off, off + depthClen);
        const raw = dec.decode(comp, res * res * 2);
        depth = new Uint16Array(raw.buffer, raw.byteOffset, res * res);
        if (filter === FILTER_SUBLEFT) unfilterDepth(depth, res);
    }
    return { res, rgba, depth };
}

// stage5.decode_depth_u16: log-spaced code → planar-Z metres (0 = background).
function depthMeters(code, near, far) {
    if (code <= 0) return 0;
    const t = (code - 1) / (DEPTH_CODE_MAX - 1);
    return near * Math.exp(t * Math.log(far / near));
}

// --- plane rendering -----------------------------------------------------------

function clamp255(v) {
    return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

// Google's "turbo" colormap (Mikhailov) — perceptually ordered, good for depth.
function turbo(t) {
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
    const g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
    const b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814.0 - t * (22569.18 - t * 6838.66))));
    return [clamp255(r), clamp255(g), clamp255(b)];
}

// Foreground metre range (over the sampled grid) for depth normalization.
function depthRange(dec, near, far, outW, outH) {
    const { res, depth } = dec;
    let mn = Infinity;
    let mx = -Infinity;
    for (let y = 0; y < outH; y++) {
        const sy = ((y * res) / outH) | 0;
        for (let x = 0; x < outW; x++) {
            const code = depth[sy * res + (((x * res) / outW) | 0)];
            if (code > 0) {
                const m = depthMeters(code, near, far);
                if (m < mn) mn = m;
                if (m > mx) mx = m;
            }
        }
    }
    return mn <= mx ? { min: mn, max: mx } : null;
}

// Nearest-sample `dec`'s chosen plane into an outW×outH ImageData (stride
// downsampling covers both full-res detail and small matrix thumbnails).
function renderPlane(dec, plane, outW, outH, near, far, range) {
    const { res, rgba, depth } = dec;
    const img = new ImageData(outW, outH);
    const o = img.data;
    let di = 0;
    if (plane === "depth") {
        const r = range || depthRange(dec, near, far, outW, outH) || { min: near, max: far };
        const span = r.max > r.min ? r.max - r.min : 1;
        for (let y = 0; y < outH; y++) {
            const sy = ((y * res) / outH) | 0;
            for (let x = 0; x < outW; x++, di += 4) {
                const code = depth[sy * res + (((x * res) / outW) | 0)];
                if (code <= 0) {
                    o[di] = 12;
                    o[di + 1] = 12;
                    o[di + 2] = 16;
                } else {
                    const [cr, cg, cb] = turbo((depthMeters(code, near, far) - r.min) / span);
                    o[di] = cr;
                    o[di + 1] = cg;
                    o[di + 2] = cb;
                }
                o[di + 3] = 255;
            }
        }
    } else if (plane === "alpha") {
        for (let y = 0; y < outH; y++) {
            const sy = ((y * res) / outH) | 0;
            for (let x = 0; x < outW; x++, di += 4) {
                const a = rgba[(sy * res + (((x * res) / outW) | 0)) * 4 + 3];
                o[di] = a;
                o[di + 1] = a;
                o[di + 2] = a;
                o[di + 3] = 255;
            }
        }
    } else {
        for (let y = 0; y < outH; y++) {
            const sy = ((y * res) / outH) | 0;
            for (let x = 0; x < outW; x++, di += 4) {
                const si = (sy * res + (((x * res) / outW) | 0)) * 4;
                o[di] = rgba[si];
                o[di + 1] = rgba[si + 1];
                o[di + 2] = rgba[si + 2];
                o[di + 3] = 255;
            }
        }
    }
    return img;
}

// --- raw-buffer fetch + tiny LRU -----------------------------------------------

async function fetchRaw(url) {
    const hit = state.rawCache.get(url);
    if (hit) {
        state.rawCache.delete(url);
        state.rawCache.set(url, hit);
        return hit;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    state.rawCache.set(url, buf);
    while (state.rawCache.size > RAW_CACHE_CAP) {
        state.rawCache.delete(state.rawCache.keys().next().value);
    }
    return buf;
}

// --- decode queue (bounded concurrency, generation-guarded) --------------------

function pumpQueue() {
    const s = state;
    while (s.active < s.maxConc && s.pending.length) {
        const task = s.pending.shift();
        s.active++;
        task().finally(() => {
            s.active--;
            pumpQueue();
        });
    }
}

function enqueue(task) {
    state.pending.push(task);
    pumpQueue();
}

// --- matrix (thumbnail grid) ---------------------------------------------------

function tileNeeds(plane) {
    return { rgba: plane !== "depth", depth: plane === "depth" };
}

function evictTilesIfNeeded() {
    const s = state;
    while (s.liveTiles.size > TILE_CACHE_CAP) {
        const [id, tile] = s.liveTiles.entries().next().value;
        s.liveTiles.delete(id);
        if (tile.isConnected && !tile._intersecting) {
            tile._decoded = false;
            tile._canvas?.remove();
            tile._canvas = null;
            tile.classList.remove("done");
        } else {
            s.liveTiles.set(id, tile); // still visible — keep, try the next oldest
            break;
        }
    }
}

function decodeTile(tile) {
    const s = state;
    const gen = s.gen;
    const frame = tile._frame;
    if (tile._decoded || tile._busy) return;
    tile._busy = true;
    enqueue(async () => {
        if (gen !== s.gen || !tile.isConnected) return;
        try {
            const buf = await fetchRaw(s.refsBase + frame.frame_path);
            if (gen !== s.gen) return;
            const dec = await decodeFrame(buf, tileNeeds(s.plane));
            if (gen !== s.gen) return;
            const cv = el("canvas");
            cv.width = THUMB;
            cv.height = THUMB;
            cv.getContext("2d").putImageData(
                renderPlane(dec, s.plane, THUMB, THUMB, s.near, s.far, null),
                0,
                0,
            );
            tile._canvas?.remove();
            tile._canvas = cv;
            tile.insertBefore(cv, tile.firstChild);
            tile.classList.add("done");
            tile._decoded = true;
            s.liveTiles.delete(frame.frame_path);
            s.liveTiles.set(frame.frame_path, tile);
            evictTilesIfNeeded();
        } catch (e) {
            if (gen === s.gen) tile.classList.add("err");
        } finally {
            tile._busy = false;
        }
    });
}

function filteredFrames() {
    const f = state.faceFilter;
    return f === "all" ? state.frames : state.frames.filter((fr) => fr.face === f);
}

function makeTile(frame) {
    const s = state;
    const tile = el("button", {
        class: "rfv-tile",
        title: `${frameId(frame)} · cam ${frame.camera_index}`,
    });
    tile._frame = frame;
    tile.appendChild(el("span", { class: "rfv-tile-lab", text: frame.face }));
    tile.addEventListener("click", () => openDetail(s.detailOrder.indexOf(frame)));
    return tile;
}

// Materialize the next page of thumbnails (kept before the footer), so the
// initial load and each scroll step only ever create PAGE_SIZE tiles — never
// all N. Decode stays lazy per-tile via the IntersectionObserver in buildMatrix.
function appendPage() {
    const s = state;
    const total = s.pageFrames.length;
    if (s.rendered >= total) return;
    const end = Math.min(s.rendered + PAGE_SIZE, total);
    for (let i = s.rendered; i < end; i++) {
        const tile = makeTile(s.pageFrames[i]);
        s.grid.insertBefore(tile, s.footer);
        s.io.observe(tile);
    }
    s.rendered = end;
    updateFooter();
}

function updateFooter() {
    const s = state;
    const total = s.pageFrames.length;
    if (s.rendered >= total) {
        s.footer.replaceChildren(
            el("span", {
                class: "rfv-more-info",
                text: total ? `all ${total.toLocaleString()} shown` : "",
            }),
        );
    } else {
        s.footer.replaceChildren(
            el("span", {
                class: "rfv-more-info",
                text: `showing ${s.rendered.toLocaleString()} of ${total.toLocaleString()}`,
            }),
            el("button", {
                class: "rfv-nav",
                text: "load more",
                onclick: () => {
                    appendPage();
                    maybeFill();
                },
            }),
        );
    }
}

// Append pages until the footer sits past the viewport (+ prefetch) or all are
// shown — fills the first screen and follows the scroll without loading all N.
function maybeFill() {
    const s = state;
    if (!s || !s.footer || !s.grid.isConnected) return;
    let guard = 0;
    while (s.rendered < s.pageFrames.length && guard++ < 80) {
        const gb = s.grid.getBoundingClientRect();
        if (!gb.height) break;
        if (s.footer.getBoundingClientRect().top > gb.bottom + PAGE_PREFETCH_PX) break;
        appendPage();
    }
}

function onGridScroll() {
    const s = state;
    if (!s || s.scrollRaf) return;
    s.scrollRaf = requestAnimationFrame(() => {
        s.scrollRaf = 0;
        maybeFill();
    });
}

function buildMatrix() {
    const s = state;
    s.gen++;
    s.pending.length = 0;
    s.liveTiles.clear();
    s.io?.disconnect();
    s.grid.replaceChildren();

    const frames = filteredFrames();
    s.pageFrames = frames;
    s.detailOrder = frames;
    s.rendered = 0;
    s.countEl.textContent = `${frames.length.toLocaleString()} frame${frames.length === 1 ? "" : "s"}`;
    s.io = new IntersectionObserver(
        (entries) => {
            for (const ent of entries) {
                const tile = ent.target;
                tile._intersecting = ent.isIntersecting;
                if (ent.isIntersecting) decodeTile(tile);
            }
        },
        { root: s.grid, rootMargin: "400px" },
    );
    // The footer (load-more + count) stays last; new tiles insert before it.
    s.footer = el("div", { class: "rfv-more" });
    s.grid.appendChild(s.footer);
    appendPage();
    requestAnimationFrame(() => {
        if (state === s) maybeFill();
    });
}

// --- detail (full-resolution single frame) -------------------------------------

function frameId(frame) {
    return frame.frame_path.replace(/^.*\//, "").replace(/\.szf$/i, "");
}

async function openDetail(index) {
    const s = state;
    const frames = s.detailOrder;
    if (index < 0 || index >= frames.length) return;
    s.detailIndex = index;
    s.view = "detail";
    s.root.classList.add("detail");

    const frame = frames[index];
    const gen = ++s.detailGen;
    s.detailMeta.textContent = "decoding…";
    s.detailCanvas.style.visibility = "hidden";
    s.detailTitle.textContent = frameId(frame);
    s.detailPos.textContent = `${index + 1} / ${frames.length}`;

    let dec;
    try {
        const buf = await fetchRaw(s.refsBase + frame.frame_path);
        if (gen !== s.detailGen) return;
        dec = await decodeFrame(buf, { rgba: true, depth: true });
    } catch (e) {
        if (gen === s.detailGen) s.detailMeta.textContent = `failed: ${e.message}`;
        return;
    }
    if (gen !== s.detailGen) return;
    s.detailDec = dec;
    s.detailRange = depthRange(dec, s.near, s.far, dec.res, dec.res);
    paintDetail();
    s.detailCanvas.style.visibility = "visible";
    renderDetailMeta();
}

function paintDetail() {
    const s = state;
    const dec = s.detailDec;
    if (!dec) return;
    s.detailCanvas.width = dec.res;
    s.detailCanvas.height = dec.res;
    s.detailCanvas
        .getContext("2d")
        .putImageData(
            renderPlane(dec, s.plane, dec.res, dec.res, s.near, s.far, s.detailRange),
            0,
            0,
        );
}

function renderDetailMeta() {
    const s = state;
    const dec = s.detailDec;
    const frame = s.detailOrder[s.detailIndex];
    const rows = [
        ["camera", String(frame.camera_index)],
        ["face", frame.face],
        ["resolution", `${dec.res}²`],
        ["near · far", `${s.near} · ${s.far} m`],
    ];
    if (s.plane === "depth" && s.detailRange) {
        rows.push(["depth range", `${s.detailRange.min.toFixed(3)} – ${s.detailRange.max.toFixed(3)} m`]);
    }
    s.detailMeta.replaceChildren(
        ...rows.map(([k, v]) =>
            el("div", { class: "rfv-meta-row" }, el("span", { text: k }), el("b", { text: v })),
        ),
    );
}

function inspectAt(ev) {
    const s = state;
    const dec = s.detailDec;
    if (!dec) return;
    const rect = s.detailCanvas.getBoundingClientRect();
    const px = Math.min(dec.res - 1, Math.max(0, ((ev.clientX - rect.left) / rect.width * dec.res) | 0));
    const py = Math.min(dec.res - 1, Math.max(0, ((ev.clientY - rect.top) / rect.height * dec.res) | 0));
    const i = py * dec.res + px;
    const r = dec.rgba[i * 4];
    const g = dec.rgba[i * 4 + 1];
    const b = dec.rgba[i * 4 + 2];
    const a = dec.rgba[i * 4 + 3];
    const code = dec.depth[i];
    const m = depthMeters(code, s.near, s.far);
    s.pixel.textContent =
        `(${px}, ${py})  rgb ${r},${g},${b}  α ${(a / 255).toFixed(3)}  ` +
        `depth ${code === 0 ? "—" : `${m.toFixed(4)} m`}`;
}

function stepDetail(delta) {
    openDetail(state.detailIndex + delta);
}

function closeDetail() {
    state.detailGen++;
    state.detailDec = null;
    state.view = "matrix";
    state.root.classList.remove("detail");
    state.pixel.textContent = "";
}

// --- plane / face controls -----------------------------------------------------

function syncPlaneButtons() {
    for (const map of state.planeBtnMaps) {
        for (const [key, btn] of Object.entries(map)) btn.classList.toggle("on", key === state.plane);
    }
}

function setPlane(plane) {
    if (state.plane === plane) return;
    state.plane = plane;
    syncPlaneButtons();
    if (state.view === "detail" && state.detailDec) {
        state.detailRange =
            plane === "depth"
                ? depthRange(state.detailDec, state.near, state.far, state.detailDec.res, state.detailDec.res)
                : null;
        paintDetail();
        renderDetailMeta();
    } else {
        buildMatrix();
    }
}

function setFace(face) {
    if (state.faceFilter === face) return;
    state.faceFilter = face;
    for (const [key, btn] of Object.entries(state.faceBtns)) {
        btn.classList.toggle("on", key === face);
    }
    buildMatrix();
}

// --- overlay lifecycle ---------------------------------------------------------

function closeRefsViewer() {
    if (!overlayEl) return;
    if (state) {
        state.gen++;
        state.detailGen++;
        state.io?.disconnect();
        state.pending.length = 0;
    }
    overlayEl.classList.remove("open");
    overlayEl.replaceChildren();
    state = null;
}

function injectStyle() {
    if (styleInjected) return;
    styleInjected = true;
    const css = `
.rfv-overlay{position:fixed;inset:0;z-index:120;display:none;flex-direction:column;
  background:var(--bg,#101114);color:var(--accent,#9ad4ff);font:13px/1.4 ui-sans-serif,system-ui,sans-serif}
.rfv-overlay.open{display:flex}
.rfv-bar{display:flex;align-items:center;gap:10px;padding:8px 12px;flex-wrap:wrap;
  background:var(--panel,#16181d);border-bottom:1px solid rgba(255,255,255,0.12)}
.rfv-bar .rfv-title{font-weight:600;color:#fff}
.rfv-bar .rfv-sub{opacity:.6}
.rfv-count{opacity:.7;margin-left:auto}
.rfv-seg{display:inline-flex;border:1px solid rgba(255,255,255,0.14);border-radius:5px;overflow:hidden}
.rfv-seg button{background:var(--panel-2,#1c1e24);color:#cfcfe0;border:none;padding:4px 10px;cursor:pointer;font:inherit}
.rfv-seg button+button{border-left:1px solid rgba(255,255,255,0.14)}
.rfv-seg button.on{background:var(--purple,#c9a6ff);color:#12121a}
.rfv-lab{opacity:.55;margin-right:2px}
.rfv-x{background:var(--panel-2,#1c1e24);color:#fff;border:1px solid rgba(255,255,255,0.14);
  border-radius:5px;cursor:pointer;padding:4px 10px;font:inherit}
.rfv-x:hover{border-color:var(--red,#ff8080)}
.rfv-body{flex:1;min-height:0;position:relative}
.rfv-grid{position:absolute;inset:0;overflow:auto;padding:10px;
  display:grid;grid-template-columns:repeat(auto-fill,minmax(${THUMB}px,1fr));gap:8px;align-content:start}
.rfv-tile{position:relative;aspect-ratio:1/1;padding:0;border:1px solid rgba(255,255,255,0.10);
  border-radius:4px;background:#0b0b0e;overflow:hidden;cursor:pointer;min-height:${THUMB}px}
.rfv-tile canvas{width:100%;height:100%;display:block;image-rendering:auto}
.rfv-tile:hover{border-color:var(--accent-line,#4a8fd8)}
.rfv-tile.err{border-color:var(--red,#ff8080)}
.rfv-tile-lab{position:absolute;left:3px;bottom:2px;font-size:10px;color:#cfcfe0;
  background:rgba(0,0,0,0.55);padding:0 4px;border-radius:3px;pointer-events:none}
.rfv-more{grid-column:1 / -1;display:flex;align-items:center;justify-content:center;gap:12px;padding:16px 8px 26px}
.rfv-more-info{opacity:.65;font-size:12px}
.rfv-overlay:not(.detail) .rfv-detail{display:none}
.rfv-overlay.detail .rfv-grid{display:none}
.rfv-detail{position:absolute;inset:0;display:flex;flex-direction:column}
.rfv-detail-top{display:flex;align-items:center;gap:12px;padding:6px 12px;flex-wrap:wrap;
  border-bottom:1px solid rgba(255,255,255,0.10)}
.rfv-detail-main{flex:1;min-height:0;display:flex}
.rfv-stage{flex:1;min-width:0;display:flex;align-items:center;justify-content:center;
  background:#08080b;padding:12px;overflow:hidden}
.rfv-stage canvas{max-width:100%;max-height:100%;object-fit:contain;image-rendering:auto;
  border:1px solid rgba(255,255,255,0.12);background:#000}
.rfv-side{width:250px;flex:none;padding:12px;overflow-y:auto;
  background:var(--panel,#16181d);border-left:1px solid rgba(255,255,255,0.12)}
.rfv-meta-row{display:flex;justify-content:space-between;gap:10px;padding:3px 0;
  border-bottom:1px dotted rgba(255,255,255,0.08)}
.rfv-meta-row b{color:#fff;font-weight:600}
.rfv-pixel{margin-top:10px;font:11px/1.5 ui-monospace,Menlo,monospace;color:#cfcfe0;min-height:1.5em}
.rfv-nav{background:var(--panel-2,#1c1e24);color:#fff;border:1px solid rgba(255,255,255,0.14);
  border-radius:5px;cursor:pointer;padding:4px 12px;font:inherit}
.rfv-nav:hover{border-color:var(--accent-line,#4a8fd8)}
.rfv-pos{opacity:.7;min-width:80px;text-align:center}`;
    document.head.appendChild(el("style", { text: css }));
}

function planeSeg(onPick) {
    const btns = {};
    const seg = el("div", { class: "rfv-seg" });
    for (const [key, label] of PLANES) {
        const b = el("button", {
            text: label,
            class: key === "rgb" ? "on" : "",
            onclick: () => onPick(key),
        });
        btns[key] = b;
        seg.appendChild(b);
    }
    return { seg, btns };
}

function faceSeg(onPick) {
    const btns = {};
    const seg = el("div", { class: "rfv-seg" });
    for (const face of ["all", ...CUBE_FACES]) {
        const b = el("button", {
            text: face,
            class: face === "all" ? "on" : "",
            onclick: () => onPick(face),
        });
        btns[face] = b;
        seg.appendChild(b);
    }
    return { seg, btns };
}

function ensureOverlay() {
    injectStyle();
    if (!overlayEl) {
        overlayEl = el("div", { class: "rfv-overlay" });
        document.body.appendChild(overlayEl);
    }
    if (!escBound) {
        escBound = true;
        // Capture on `window` (above `document` in the capture path) so this fires
        // before splatviewer's own Esc handler and can swallow it — otherwise Esc
        // would close the splat viewer behind us instead of this overlay.
        window.addEventListener(
            "keydown",
            (ev) => {
                if (ev.key !== "Escape" || !overlayEl?.classList.contains("open")) return;
                ev.stopImmediatePropagation();
                if (state?.view === "detail") closeDetail();
                else closeRefsViewer();
            },
            true,
        );
    }
    return overlayEl;
}

// Build the whole DOM for a fresh session and stash the live refs on `state`.
function buildUI(label) {
    const s = state;
    const plane = planeSeg(setPlane);
    const face = faceSeg(setFace);
    s.faceBtns = face.btns;
    s.countEl = el("span", { class: "rfv-count" });

    const bar = el(
        "div",
        { class: "rfv-bar" },
        el("span", { class: "rfv-title", text: "reference frames" }),
        el("span", { class: "rfv-sub", text: label || "" }),
        el("span", { class: "rfv-lab", text: "plane" }),
        plane.seg,
        el("span", { class: "rfv-lab", text: "face" }),
        face.seg,
        s.countEl,
        el("button", { class: "rfv-x", text: "✕ close", onclick: closeRefsViewer }),
    );

    s.grid = el("div", { class: "rfv-grid" });
    s.grid.addEventListener("scroll", onGridScroll);

    // detail sub-view (hidden until a tile is opened)
    s.detailCanvas = el("canvas");
    s.detailCanvas.addEventListener("mousemove", inspectAt);
    s.detailCanvas.addEventListener("mouseleave", () => (s.pixel.textContent = ""));
    s.detailTitle = el("span", { class: "rfv-title" });
    s.detailPos = el("span", { class: "rfv-pos" });
    s.detailMeta = el("div", { class: "rfv-meta" });
    s.pixel = el("div", { class: "rfv-pixel" });
    const detailPlane = planeSeg(setPlane);
    s.planeBtnMaps = [plane.btns, detailPlane.btns];

    const detail = el(
        "div",
        { class: "rfv-detail" },
        el(
            "div",
            { class: "rfv-detail-top" },
            el("button", { class: "rfv-nav", text: "‹ matrix", onclick: closeDetail }),
            s.detailTitle,
            detailPlane.seg,
            el("button", { class: "rfv-nav", text: "‹ prev", onclick: () => stepDetail(-1) }),
            s.detailPos,
            el("button", { class: "rfv-nav", text: "next ›", onclick: () => stepDetail(1) }),
        ),
        el(
            "div",
            { class: "rfv-detail-main" },
            el("div", { class: "rfv-stage" }, s.detailCanvas),
            el("div", { class: "rfv-side" }, s.detailMeta, s.pixel),
        ),
    );

    s.root.replaceChildren(bar, el("div", { class: "rfv-body" }, s.grid, detail));
}

// --- public API ----------------------------------------------------------------

// Open the reference-frame viewer for a cell. Resolves the Stage-5 output on its
// own (via the status API → transforms.json), so callers only pass the cell.
export async function openRefsViewer({ run, slot, model, label } = {}) {
    if (!run || !slot || !model) return;
    if (state) {
        // Neutralize any prior session's in-flight decodes before reusing the overlay.
        state.gen++;
        state.detailGen++;
        state.io?.disconnect();
        state.pending.length = 0;
    }
    const root = ensureOverlay();
    root.classList.add("open");
    root.replaceChildren(
        el("div", { class: "rfv-bar" },
            el("span", { class: "rfv-title", text: "reference frames" }),
            el("span", { class: "rfv-sub", text: label || `${slot} · ${model}` }),
            el("span", { class: "rfv-count", text: "loading…" }),
            el("button", { class: "rfv-x", text: "✕ close", onclick: closeRefsViewer }),
        ),
    );

    let s5;
    try {
        s5 = await api.splatStage5Status(run, slot, model);
    } catch (e) {
        toast(`refs: ${e.message}`, "err");
        return closeRefsViewer();
    }
    if (!root.classList.contains("open")) return;
    if (s5.status !== "done" || !s5.url) {
        root.querySelector(".rfv-count").textContent =
            "no references yet — render Stage 5 first";
        return;
    }

    const transformsUrl = api.absUrl(s5.url);
    const refsBase = transformsUrl.replace(/transforms\.json(\?.*)?$/, "");
    let doc;
    try {
        doc = await fetch(transformsUrl, { cache: "no-store" }).then((r) => r.json());
    } catch (e) {
        toast(`refs: ${e.message}`, "err");
        return closeRefsViewer();
    }
    if (!root.classList.contains("open")) return;

    const frames = (doc.frames || []).filter((f) => f && f.frame_path);
    if (!frames.length) {
        root.querySelector(".rfv-count").textContent = "no frames in transforms.json";
        return;
    }

    state = {
        run,
        slot,
        model,
        root,
        refsBase,
        frames,
        detailOrder: frames,
        near: Number(doc.near) || 0.01,
        far: Number(doc.far) || 100,
        plane: "rgb",
        faceFilter: "all",
        view: "matrix",
        gen: 0,
        detailGen: 0,
        detailIndex: -1,
        detailDec: null,
        detailRange: null,
        active: 0,
        maxConc: Math.min(6, Math.max(2, navigator.hardwareConcurrency || 4)),
        pending: [],
        rawCache: new Map(),
        liveTiles: new Map(),
        io: null,
        pageFrames: frames,
        rendered: 0,
        footer: null,
        scrollRaf: 0,
    };
    buildUI(label || `${slot} · ${model}`);
    buildMatrix();
    syncPlaneButtons();
}
