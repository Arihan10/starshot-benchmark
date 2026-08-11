#!/usr/bin/env node
// Tiny static server for the debug viewer + the SOG-LOD streaming playground.
// Serves public/ and exposes the installed three.js / PlayCanvas engine under
// /vendor/. Reads SERVER_URL from env and injects it into the served HTML so the
// browser knows where to POST /generate.
//
// Splat assets under public/assets/ (single .sog files and streamed-SOG LOD
// bundles — a lod-meta.json manifest beside its per-LOD .sog/.webp chunks,
// produced by tools/ply-to-lod-sog.mjs) are streamed with HTTP Range support,
// long-lived immutable caching, and open CORS, so the playground can page LOD
// chunks over the network exactly as it would from a CDN. `GET /api/assets`
// enumerates them for the playground's source picker.

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { createReadStream, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = resolve(__dirname, "public");
const ASSETS_DIR = resolve(PUBLIC_DIR, "assets");
const THREE_DIR = resolve(__dirname, "node_modules", "three");
const GIFJS_DIR = resolve(__dirname, "node_modules", "gif.js", "dist");
// The Gaussian-splat viewer (@mkkellogg/gaussian-splats-3d) for the Stage-2
// cloud preview — an ESM build that imports 'three' (resolved via the importmap).
const GSPLAT_DIR = resolve(
    __dirname,
    "node_modules",
    "@mkkellogg",
    "gaussian-splats-3d",
    "build",
);
// PlayCanvas engine (ESM bundle) — renders the SOG-encoded splat + streams LOD.
const PLAYCANVAS_DIR = resolve(__dirname, "node_modules", "playcanvas", "build");

const PORT = Number(process.env.PORT ?? 8766);
const SERVER_URL = process.env.SERVER_URL ?? "http://127.0.0.1:8765";
// "bfs" | "dfs" — flips the dashboard's main-screen backdrop (light vs dark)
// so a BFS run is instantly distinguishable from a DFS one. Set by the launch
// script (scripts/run_bfs.py); defaults to dfs (dark) for every other entry.
const PIPELINE_MODE = process.env.PIPELINE_MODE ?? "dfs";
// Which page to open on boot: "/" (pipeline dashboard), "/oneshot" (one-shot
// bench), or "/playground" (the SOG-LOD streaming playground).
const VIEWER_PATH = process.env.VIEWER_PATH ?? "/";

const MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".wasm": "application/wasm",
    // Bundled super-compressed splat; fetched as an ArrayBuffer by the engine.
    ".sog": "application/octet-stream",
    ".bin": "application/octet-stream",
};

// Immutable, cacheable content: the vendored engines and the (content-stable)
// splat assets. Everything else (HTML/JS under dev) stays no-store so edits show
// up on reload.
const IMMUTABLE = "public, max-age=31536000, immutable";
const cacheFor = (url) =>
    url.startsWith("/vendor/") || url.startsWith("/assets/") ? IMMUTABLE : "no-store";

const CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Range, Content-Type",
    "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
};

function resolveUnder(root, urlPath) {
    const decoded = decodeURIComponent(urlPath.split("?")[0]);
    const rel = normalize(decoded).replace(/^(\.\.[\/\\])+/, "");
    const abs = join(root, rel);
    if (!abs.startsWith(root)) return null;
    return abs;
}

// Serve a file with correct MIME, open CORS, cache policy, and HTTP Range support
// (206 partial content) — the last is what lets the engine page large SOG chunks
// and lets the browser resume/parallelize the transfer.
async function serveFile(req, res, filePath, cache) {
    let st;
    try {
        st = await stat(filePath);
        if (st.isDirectory()) throw Object.assign(new Error("is a directory"), { code: "EISDIR" });
    } catch (e) {
        res.writeHead(404, { "Content-Type": "text/plain", ...CORS });
        return res.end(`not found: ${e.code ?? e.message}`);
    }

    const headers = {
        "Content-Type": MIME[extname(filePath).toLowerCase()] ?? "application/octet-stream",
        "Cache-Control": cache ?? "no-store",
        "Accept-Ranges": "bytes",
        ...CORS,
    };

    if (req.method === "HEAD") {
        res.writeHead(200, { ...headers, "Content-Length": st.size });
        return res.end();
    }

    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
    if (range) {
        const size = st.size;
        let start = range[1] === "" ? undefined : parseInt(range[1], 10);
        let end = range[2] === "" ? undefined : parseInt(range[2], 10);
        if (start === undefined) {
            start = Math.max(0, size - (end ?? 0)); // suffix range: bytes=-N
            end = size - 1;
        } else if (end === undefined || end >= size) {
            end = size - 1;
        }
        if (start > end || start >= size) {
            res.writeHead(416, { ...headers, "Content-Range": `bytes */${size}` });
            return res.end();
        }
        res.writeHead(206, {
            ...headers,
            "Content-Range": `bytes ${start}-${end}/${size}`,
            "Content-Length": end - start + 1,
        });
        return createReadStream(filePath, { start, end }).pipe(res);
    }

    res.writeHead(200, { ...headers, "Content-Length": st.size });
    createReadStream(filePath).pipe(res);
}

async function serveIndex(res, relHtmlPath = "index.html") {
    try {
        let html = await readFile(join(PUBLIC_DIR, relHtmlPath), "utf8");
        html = html.replace(
            /<meta name="server-url"[^>]*>/,
            `<meta name="server-url" content="${SERVER_URL}">`,
        );
        // Flip the BFS/DFS backdrop by stamping the active pipeline onto <html>.
        html = html.replace(
            /(<html\b[^>]*\bdata-pipeline=")[^"]*(")/,
            `$1${PIPELINE_MODE}$2`,
        );
        res.writeHead(200, { "Content-Type": MIME[".html"], "Cache-Control": "no-store" });
        res.end(html);
    } catch (e) {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end(`failed to render ${relHtmlPath}: ${e.message}`);
    }
}

function sendJson(res, obj, status = 200) {
    res.writeHead(status, { "Content-Type": MIME[".json"], "Cache-Control": "no-store", ...CORS });
    res.end(JSON.stringify(obj));
}

function dirBytes(dir) {
    let sum = 0;
    for (const e of readdirSync(dir, { withFileTypes: true })) {
        const p = join(dir, e.name);
        sum += e.isDirectory() ? dirBytes(p) : statSync(p).size;
    }
    return sum;
}

// Enumerate splat assets under public/assets/ for the playground picker: every
// streamed-SOG LOD bundle (a dir holding lod-meta.json) and every standalone
// .sog. Returns the URL to hand the viewer plus lightweight metadata.
function listAssets() {
    const out = [];
    if (!existsSync(ASSETS_DIR)) return out;
    const toUrl = (abs) => `/assets/${relative(ASSETS_DIR, abs).split(sep).join("/")}`;
    const walk = (dir) => {
        for (const e of readdirSync(dir, { withFileTypes: true })) {
            const abs = join(dir, e.name);
            if (e.isDirectory()) {
                const meta = join(abs, "lod-meta.json");
                if (existsSync(meta)) {
                    let info = {};
                    try {
                        const m = JSON.parse(readFileSync(meta, "utf8"));
                        info = { lodLevels: m.lodLevels, counts: m.counts, splats: m.counts?.[0] };
                    } catch {
                        /* a malformed manifest just lists without stats */
                    }
                    out.push({
                        name: relative(ASSETS_DIR, abs).split(sep).join("/"),
                        type: "lod",
                        url: toUrl(meta),
                        bytes: dirBytes(abs),
                        ...info,
                    });
                } else {
                    walk(abs);
                }
            } else if (e.name.toLowerCase().endsWith(".sog")) {
                out.push({
                    name: relative(ASSETS_DIR, abs).split(sep).join("/").replace(/\.sog$/i, ""),
                    type: "sog",
                    url: toUrl(abs),
                    bytes: statSync(abs).size,
                });
            }
        }
    };
    walk(ASSETS_DIR);
    return out.sort((a, b) => a.name.localeCompare(b.name));
}

const VENDORS = [
    ["/vendor/three/", THREE_DIR],
    ["/vendor/gifjs/", GIFJS_DIR],
    ["/vendor/gsplat/", GSPLAT_DIR],
    ["/vendor/playcanvas/", PLAYCANVAS_DIR],
];

const server = createServer(async (req, res) => {
    const url = req.url ?? "/";
    const path = url.split("?")[0];

    if (req.method === "OPTIONS") {
        res.writeHead(204, CORS);
        return res.end();
    }
    if (req.method !== "GET" && req.method !== "HEAD") {
        res.writeHead(405, { "Content-Type": "text/plain", ...CORS });
        return res.end("method not allowed");
    }

    if (path === "/" || path === "/index.html") return serveIndex(res);
    // The experimental one-shot benchmark page (own slots/models, single-call
    // scene design). Same meta-injection so it knows the API origin.
    if (path === "/oneshot" || path === "/oneshot/" || path === "/oneshot/index.html") {
        return serveIndex(res, join("oneshot", "index.html"));
    }
    // The SOG-LOD streaming playground.
    if (path === "/playground" || path === "/playground/" || path === "/playground/index.html") {
        return serveIndex(res, join("playground", "index.html"));
    }
    // The matterport tours index: every planned/captured walkthrough + publish.
    if (path === "/tours" || path === "/tours/" || path === "/tours.html") {
        return serveIndex(res, "tours.html");
    }
    // The tour mask inspector: a captured tour's panos + their per-pixel
    // object-ID masks, hovered exactly the way the prod walkthrough will.
    if (path === "/tourview" || path === "/tourview/" || path === "/tourview.html") {
        return serveIndex(res, "tourview.html");
    }
    // The parallel-flow debugger: one cell's call schedule + why each zone's
    // interior is or isn't released.
    if (path === "/paralleldebug" || path === "/paralleldebug/" || path === "/paralleldebug.html") {
        return serveIndex(res, "paralleldebug.html");
    }
    // Splat asset catalogue for the playground picker.
    if (path === "/api/assets") return sendJson(res, { assets: listAssets() });

    for (const [prefix, root] of VENDORS) {
        if (path.startsWith(prefix)) {
            const abs = resolveUnder(root, path.slice(prefix.length));
            if (abs) return serveFile(req, res, abs, cacheFor(path));
        }
    }

    const pub = resolveUnder(PUBLIC_DIR, path);
    if (pub) return serveFile(req, res, pub, cacheFor(path));

    res.writeHead(404, { "Content-Type": "text/plain", ...CORS });
    res.end("not found");
});

if (!existsSync(THREE_DIR)) {
    console.error(
        `[client] three.js not installed at ${THREE_DIR}. Run \`npm install\` in client/ first.`,
    );
    process.exit(1);
}

server.listen(PORT, "127.0.0.1", () => {
  const viewerUrl = `http://127.0.0.1:${PORT}${VIEWER_PATH}`;
  console.log(`[client] viewer at ${viewerUrl} (server=${SERVER_URL})`);
  console.log(`[client] SOG-LOD playground at http://127.0.0.1:${PORT}/playground`);
  console.log(`[client] tour mask inspector at http://127.0.0.1:${PORT}/tourview`);
  openBrowser(viewerUrl);
});

function openBrowser(url) {
    const cmd =
        process.platform === "darwin"
            ? ["open", [url]]
            : process.platform === "win32"
              ? ["cmd", ["/c", "start", "", url]]
              : ["xdg-open", [url]];
    try {
        spawn(cmd[0], cmd[1], { detached: true, stdio: "ignore" }).unref();
    } catch {
        // non-fatal: user can open the URL themselves
    }
}

for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => {
        // Open EventSource connections would otherwise keep server.close() pending
        // until the browser disconnects, so force them shut.
        server.closeAllConnections?.();
        server.close(() => process.exit(0));
        setTimeout(() => process.exit(0), 500).unref();
    });
}
