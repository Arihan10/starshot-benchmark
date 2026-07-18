// Frame post-processing worker for the Stage-5 capture page (splatcapture.js).
//
// The capture measurement showed the page's single JS thread was the pipeline
// wall (~6 ms/view of swizzle + pack + upload vs ~0.3 ms of actual rendering),
// so everything after the GPU readback lands here instead: raw buffers arrive
// as zero-copy transfers, the depth pack-pass RGBA is swizzled to uint16 codes,
// frames are packed into SRF1 batches, and the batches are POSTed to the API.
// The WebGL thread only renders, fences, and copies buffers out of the PBOs.
//
// Protocol (messages in):
//   { cfg: { framesUrl, resolution, batchViews } }   — once, before any frame
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
            body: packBatch(frames, cfg.resolution),
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
    // One readback segment: slice each view's planes out of the transferred
    // buffer (subarray views — no copies). The RG8 depth plane already IS the
    // little-endian uint16 codes the SRF1 wire + SZF store carry, so there is no
    // per-pixel swizzle here — just pass the bytes through.
    const n = cfg.resolution * cfg.resolution;
    const colorBytes = n * 4;
    const depthBytes = n * 2;
    const viewBytes = colorBytes + depthBytes;
    const seg = new Uint8Array(m.segment);
    for (let vi = 0; vi < m.ids.length; vi++) {
        const base = vi * viewBytes;
        const rgba = seg.subarray(base, base + colorBytes);
        const depth = seg.subarray(base + colorBytes, base + viewBytes);
        ready.push({ id: m.ids[vi], rgba, depth });
    }
    void flush();
};
