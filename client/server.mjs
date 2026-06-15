#!/usr/bin/env node
// Tiny static server for the debug viewer. Serves public/ and exposes the
// installed three.js at /vendor/three/. Reads SERVER_URL from env and injects
// it into index.html so the browser knows where to POST /generate.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = resolve(__dirname, "public");
const THREE_DIR = resolve(__dirname, "node_modules", "three");
const GIFJS_DIR = resolve(__dirname, "node_modules", "gif.js", "dist");

const PORT = Number(process.env.PORT ?? 8766);
const SERVER_URL = process.env.SERVER_URL ?? "http://127.0.0.1:8765";
// Which page to open on boot: "/" (pipeline dashboard) or "/oneshot"
// (the one-shot bench, used by scripts/run_oneshot.py).
const VIEWER_PATH = process.env.VIEWER_PATH ?? "/";

const MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".wasm": "application/wasm",
};

function resolveUnder(root, urlPath) {
    const decoded = decodeURIComponent(urlPath.split("?")[0]);
    const rel = normalize(decoded).replace(/^(\.\.[\/\\])+/, "");
    const abs = join(root, rel);
    if (!abs.startsWith(root)) return null;
    return abs;
}

async function serveFile(res, path) {
    try {
        const body = await readFile(path);
        const type =
            MIME[extname(path).toLowerCase()] ?? "application/octet-stream";
        res.writeHead(200, {
            "Content-Type": type,
            "Cache-Control": "no-store",
        });
        res.end(body);
    } catch (e) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end(`not found: ${e.code ?? e.message}`);
    }
}

async function serveIndex(res, relHtmlPath = "index.html") {
    try {
        let html = await readFile(join(PUBLIC_DIR, relHtmlPath), "utf8");
        html = html.replace(
            /<meta name="server-url"[^>]*>/,
            `<meta name="server-url" content="${SERVER_URL}">`,
        );
        res.writeHead(200, {
            "Content-Type": MIME[".html"],
            "Cache-Control": "no-store",
        });
        res.end(html);
    } catch (e) {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end(`failed to render ${relHtmlPath}: ${e.message}`);
    }
}

const server = createServer(async (req, res) => {
    const url = req.url ?? "/";
    if (url === "/" || url === "/index.html") {
        return serveIndex(res);
    }
    // The experimental one-shot benchmark page (own slots/models, single-call
    // scene design). Same meta-injection so it knows the API origin.
    if (
        url === "/oneshot" ||
        url === "/oneshot/" ||
        url === "/oneshot/index.html"
    ) {
        return serveIndex(res, join("oneshot", "index.html"));
    }
    if (url.startsWith("/vendor/three/")) {
        const sub = url.slice("/vendor/three/".length);
        const abs = resolveUnder(THREE_DIR, sub);
        if (abs) return serveFile(res, abs);
    }
    if (url.startsWith("/vendor/gifjs/")) {
        const sub = url.slice("/vendor/gifjs/".length);
        const abs = resolveUnder(GIFJS_DIR, sub);
        if (abs) return serveFile(res, abs);
    }
    const pub = resolveUnder(PUBLIC_DIR, url);
    if (pub) return serveFile(res, pub);
    res.writeHead(404, { "Content-Type": "text/plain" });
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
