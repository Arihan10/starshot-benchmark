// Frame post-processing worker for the Stage-5 capture page (splatcapture.js).
//
// The capture measurement showed the page's single JS thread was the pipeline
// wall (~6 ms/view of swizzle + pack + upload vs ~0.3 ms of actual rendering),
// so everything after the GPU readback lands here instead: raw buffers arrive
// as zero-copy transfers, frames are encoded, and the result is POSTed to the API.
// The WebGL thread only renders, fences, and copies buffers out of the PBOs.
//
// WIRE: raw planes (SRF1), host encodes them — optionally gzip-wrapped as SZC1, which
// is a win over a real socket and pure cost over loopback (`&nozip=1`).
//
// This worker deliberately does NOT compress pixels, and that is a measured decision
// rather than an omission. Having it build each view's finished .szf (deflate, written
// verbatim by the host) was tried and reverted: it ran at 6.4 img/s against 58 for raw
// planes. The reasons are worth keeping, because they apply to any variant of the idea:
//
//   - It does not remove work, it relocates it. The post workers, the render thread and
//     the host all share the container's 8 cores, so a codec here competes with the
//     render loop it was supposed to unblock.
//   - Over loopback the bytes were never the constraint, so there is nothing to buy.
//   - Browsers ship no zstd encoder, so client-side encoding forces the STORED frames
//     to deflate. Stage 6 re-reads every frame every epoch, making that a permanent
//     tax on training in exchange for a capture-time win that did not materialize.
//
// Protocol (messages in):
//   { cfg: { framesUrl, resolution, batchViews, gzip } } — once, before any frame
//   { segment: ArrayBuffer, ids: [view ids] }        — one readback segment:
//       per view, [RGBA8 color plane (4·n)][RG8 depth plane (2·n)], packed
//       back-to-back in id order (transferred zero-copy from the render thread).
//       The RG8 depth bytes ARE the log-u16 codes little-endian — no swizzle.
//   { flush: true }                                   — end of stream: drain
// Messages out:
//   { posted: n }    — a batch landed (n views)
//   { drained: true }— every queued frame is posted (follows a flush)
//   { error: "…" }   — a POST failed; the page aborts the session

const MAX_POSTS_INFLIGHT = 2;

let cfg = null;
let ready = [];
let postsInFlight = 0;
let draining = false;

// --- SRF1 encode --------------------------------------------------------------

function packBatch(frames, resolution) {
    const enc = new TextEncoder();
    const parts = [enc.encode("SRF1")];
    const head = new DataView(new ArrayBuffer(4));
    head.setUint32(0, frames.length, true);
    parts.push(new Uint8Array(head.buffer));
    for (const f of frames) {
        const idB = enc.encode(f.id);
        const h = new DataView(new ArrayBuffer(12));
        h.setUint32(0, idB.length, true);
        h.setUint32(4, resolution, true);
        h.setUint32(8, resolution, true);
        parts.push(new Uint8Array(h.buffer), idB, f.rgba, f.depth);
    }
    return new Blob(parts, { type: "application/octet-stream" });
}

// Wrap the SRF1 batch as "SZC1" + gzip(SRF1). The server detects the marker and
// inflates back to the byte-identical SRF1 body (refcapture.py / modal_capture.py),
// so the SZF encode is unchanged — this only lifts the raw payload off the single-
// threaded ingest. A win over a real socket, pure cost over loopback.
const GZIP_MARKER = new Uint8Array([0x53, 0x5a, 0x43, 0x31]); // "SZC1"

async function encodeBody(frames, resolution) {
    const srf1 = packBatch(frames, resolution);
    if (cfg.gzip === false || typeof CompressionStream === "undefined") return srf1;
    try {
        const gz = await new Response(
            srf1.stream().pipeThrough(new CompressionStream("gzip")),
        ).blob();
        return new Blob([GZIP_MARKER, gz], { type: "application/octet-stream" });
    } catch {
        return srf1; // any failure → raw SRF1
    }
}

function maybeDrained() {
    if (draining && ready.length === 0 && postsInFlight === 0) {
        postMessage({ drained: true });
    }
}

async function flush() {
    if (ready.length === 0) return maybeDrained();
    if (!draining && ready.length < cfg.batchViews) return;
    if (postsInFlight >= MAX_POSTS_INFLIGHT) return; // retried on completion
    const frames = ready;
    ready = [];
    postsInFlight += 1;
    try {
        const res = await fetch(cfg.framesUrl, {
            method: "POST",
            body: await encodeBody(frames, cfg.resolution),
        });
        if (!res.ok) {
            let detail = `HTTP ${res.status}`;
            try {
                detail = (await res.json()).detail || detail;
            } catch {
                /* non-JSON error body */
            }
            throw new Error(`frames POST failed: ${detail}`);
        }
        postMessage({ posted: frames.length });
    } catch (e) {
        postMessage({ error: String((e && e.message) || e) });
    } finally {
        postsInFlight -= 1;
        void flush();
    }
}

onmessage = (ev) => {
    const m = ev.data;
    if (m.cfg) {
        cfg = m.cfg;
        return;
    }
    if (m.flush) {
        draining = true;
        void flush();
        return;
    }
    // One readback segment: per view, [RGBA8][RG8 depth] back to back.
    const res = cfg.resolution;
    const n = res * res;
    const viewBytes = n * 6;
    const seg = new Uint8Array(m.segment);

    // Pass the raw planes through as zero-copy subarray views.
    const colorBytes = n * 4;
    for (let vi = 0; vi < m.ids.length; vi++) {
        const base = vi * viewBytes;
        ready.push({
            id: m.ids[vi],
            rgba: seg.subarray(base, base + colorBytes),
            depth: seg.subarray(base + colorBytes, base + viewBytes),
        });
    }
    void flush();
};
