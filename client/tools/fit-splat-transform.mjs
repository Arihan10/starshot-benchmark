#!/usr/bin/env node
// fit-splat-transform.mjs — recover where a trained splat actually belongs, by
// measuring it against the point cloud it was INITIALIZED from.
//
// WHY THIS EXISTS. An external trainer (Postshot) is handed a COLMAP model already in
// the repo's world frame — stage5 locks it, splat/colmap.py writes the poses verbatim
// and copies cloud.ply's xyz straight into points3D.txt — and still returns a model
// whose origin is somewhere else. Measured across two cells the offsets have nothing
// in common:
//
//     modern-house / gemini-flash    (-4.080, -22.840,  6.940)
//     platformer-level / opus-new    (10.980,  -1.920,  0.240)
//
// so it is per-project and cannot be applied blindly. Every splat has to be measured,
// which is why this is a tool and not a note in a commit message.
//
// WHY cloud.ply IS THE REFERENCE. It is a real correspondence, not a similarity of
// silhouettes: every trained Gaussian descends from a surfel in that file, so the two
// sets differ by training drift, densification and pruning — centimetres of scatter,
// not metres of offset. Fitting against a decimated proxy instead bounds the answer to
// about +/-2 m, which is useless for a seamless transition.
//
// THE METHOD, and the two mistakes it is built to avoid:
//
//   • The metric is SHARP: the fraction of Gaussians landing within one small voxel of
//     a surfel. The obvious alternative — "fraction within 1.5 m of anything" — scores
//     ~100% at almost any offset inside a dense scene, so it validates nothing. That
//     false positive once shipped a 0.8 m error.
//   • The search is EXHAUSTIVE, coarse to fine, seeded from the bounds offset. ICP has
//     a local minimum every wall-to-wall spacing and will happily converge into the
//     wrong one while reporting a healthy residual.
//
// THE PROOF IS THE IDENTITY RATIO. Not the absolute score — that depends on how
// densely the cloud sampled the surfaces, so it varies by scene and says little on its
// own. What matters is peak vs leaving the splat exactly where it was written. A large
// ratio means displaced; near 1 means it was already right and nothing should be baked.
//
// WHERE IT LOOKS. A cell keeps `trained.ply` / `trained.sog` and `cloud.ply` either at
// the cell root or under `splat/`, and not necessarily in the same one, so the model and
// the cloud are searched for independently across both.
//
// Usage:
//   node tools/fit-splat-transform.mjs <cell-dir | trained.ply|.sog> [--to cloud.ply]
//        [--radius 6] [--voxel 0.15] [--no-rotation-check] [--json]

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import { readPlyHeader } from "./ply-utils.mjs";

const SPLAT_TRANSFORM_CLI = fileURLToPath(
    new URL("../node_modules/@playcanvas/splat-transform/bin/cli.mjs", import.meta.url),
);

export const FIT_DEFAULTS = {
    // Half-width of the coarse sweep, in metres, around the bounds-offset seed. The
    // seed is normally within a metre, so this is generous.
    radius: 6,
    // "On the same surface" tolerance. Tight enough to discriminate, loose enough to
    // absorb the drift a trained Gaussian legitimately has from its init surfel.
    voxel: 0.15,
    // Points sampled per stage. The coarse sweep visits thousands of offsets, so it
    // gets fewer; by the fine stage there are only a hundred-odd offsets left and
    // accuracy matters more than speed.
    coarsePts: 8000,
    finePts: 40000,
    rotationCheck: true,
};

// --- positions --------------------------------------------------------------

// A cell may only have kept its `.sog`, so the solver has to be able to read one. SOG
// stores positions log-companded into WebP textures, which would mean shipping an image
// decoder and re-deriving the companding here; converting through splat-transform costs
// a few seconds and reuses the encoder's own reader instead of a second implementation
// of it that could disagree. No spatial actions are passed, so nothing is transformed —
// splat-transform negates X/Y for those, and that trap is avoided by not going near it.
//
// Returns a temp path the caller owns, plus a cleanup.
function sogToTempPly(sog) {
    const tmp = path.join(os.tmpdir(), `fit-sog-${process.pid}-${path.basename(sog)}.ply`);
    const r = spawnSync(process.execPath, [SPLAT_TRANSFORM_CLI, sog, tmp, "-w"], {
        stdio: ["ignore", "ignore", "pipe"],
    });
    if (r.status !== 0 || !fs.existsSync(tmp)) {
        fs.rmSync(tmp, { force: true });
        throw new Error(
            `could not read ${path.basename(sog)} — splat-transform failed:\n` +
                String(r.stderr ?? "").trim(),
        );
    }
    return { ply: tmp, cleanup: () => fs.rmSync(tmp, { force: true }) };
}

// --- PLY positions ----------------------------------------------------------

// x/y/z of every (or every Nth) vertex. `minOpacity` drops the low-opacity haze a
// trained splat carries — it is spread through the volume rather than sitting on
// surfaces, so it only blurs the measurement. Opacity is stored pre-sigmoid, so 0 is
// the halfway point.
function readPositions(file, { maxPts = Infinity, minOpacity = null } = {}) {
    const header = readPlyHeader(file);
    const { count, props, headerBytes } = header;
    const stride = props.length * 4;
    const opIdx = props.indexOf("opacity");
    // x/y/z are LOOKED UP, never assumed to lead the record. 3DGS PLYs from the
    // reference implementation put them first, but Brush writes its properties
    // alphabetically — x,y,z land last, after 45 f_rest_* — and reading offset 0
    // there silently yields f_dc_0, i.e. colour parsed as geometry.
    const [xIdx, yIdx, zIdx] = ["x", "y", "z"].map((a) => props.indexOf(a));
    if (xIdx < 0 || yIdx < 0 || zIdx < 0) {
        throw new Error(`${path.basename(file)} has no x/y/z properties`);
    }
    const step = Math.max(1, Math.ceil(count / maxPts));
    const out = new Float32Array(Math.ceil(count / step) * 3);
    const rec = Buffer.alloc(stride);
    const fd = fs.openSync(file, "r");
    let k = 0;
    try {
        for (let i = 0; i < count; i += step) {
            fs.readSync(fd, rec, 0, stride, headerBytes + i * stride);
            if (minOpacity !== null && opIdx >= 0 && rec.readFloatLE(opIdx * 4) < minOpacity) {
                continue;
            }
            const x = rec.readFloatLE(xIdx * 4);
            const y = rec.readFloatLE(yIdx * 4);
            const z = rec.readFloatLE(zIdx * 4);
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
            out[k * 3] = x;
            out[k * 3 + 1] = y;
            out[k * 3 + 2] = z;
            k++;
        }
    } finally {
        fs.closeSync(fd);
    }
    return { xyz: out.subarray(0, k * 3), n: k, count, props: props.length };
}

// Bounds and centre. Used for the search SEED and for the printed table — never for
// comparing the two models' sizes, which is what the perturbation checks are for; see
// the note above perturbScore for why a span ratio cannot answer that question.
function extent(p) {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.n; i++) {
        for (let a = 0; a < 3; a++) {
            const v = p.xyz[i * 3 + a];
            if (v < lo[a]) lo[a] = v;
            if (v > hi[a]) hi[a] = v;
        }
    }
    return {
        lo,
        hi,
        mid: lo.map((v, i) => (v + hi[i]) / 2),
        span: hi.map((v, i) => v - lo[i]),
    };
}

// --- occupancy --------------------------------------------------------------

// Voxel keys are packed into ONE number rather than a "i,j,k" string. Membership is
// tested tens of millions of times across the sweep, and string keys make that the
// dominant cost. 17 bits per axis keeps the key inside the 53-bit safe integer range
// while covering +/-65536 voxels — about +/-9.8 km at the default voxel size.
const BITS = 17;
const HALF = 1 << (BITS - 1);
const SPAN = 1 << BITS;
const packKey = (i, j, k) => ((i + HALF) * SPAN + (j + HALF)) * SPAN + (k + HALF);

function occupancy(cloud, voxel) {
    const set = new Set();
    for (let i = 0; i < cloud.n; i++) {
        set.add(
            packKey(
                Math.floor(cloud.xyz[i * 3] / voxel),
                Math.floor(cloud.xyz[i * 3 + 1] / voxel),
                Math.floor(cloud.xyz[i * 3 + 2] / voxel),
            ),
        );
    }
    return set;
}

// Fraction of `splat` landing in an occupied voxel once shifted by t.
function onSurface(splat, occ, voxel, tx, ty, tz, limit) {
    const n = limit ? Math.min(limit, splat.n) : splat.n;
    const stride = Math.max(1, Math.floor(splat.n / n));
    let hit = 0;
    let seen = 0;
    for (let i = 0; i < splat.n; i += stride) {
        seen++;
        if (
            occ.has(
                packKey(
                    Math.floor((splat.xyz[i * 3] + tx) / voxel),
                    Math.floor((splat.xyz[i * 3 + 1] + ty) / voxel),
                    Math.floor((splat.xyz[i * 3 + 2] + tz) / voxel),
                ),
            )
        ) {
            hit++;
        }
    }
    return hit / Math.max(1, seen);
}

function sweep(splat, occ, voxel, centre, half, step, pts) {
    let best = { score: -1, t: centre };
    for (let x = centre[0] - half; x <= centre[0] + half + 1e-9; x += step) {
        for (let y = centre[1] - half; y <= centre[1] + half + 1e-9; y += step) {
            for (let z = centre[2] - half; z <= centre[2] + half + 1e-9; z += step) {
                const s = onSurface(splat, occ, voxel, x, y, z, pts);
                if (s > best.score) best = { score: s, t: [x, y, z] };
            }
        }
    }
    return best;
}

// --- is a translation the WHOLE answer? -------------------------------------
//
// A translation-only fit can hide a rotation or a scale error, and a translation baked
// over either is wrong everywhere except the middle. Both are tested the same way:
// perturb the model, re-optimize a small translation so the perturbation is judged at
// ITS best rather than penalized for shifting things, and see whether the unperturbed
// case still wins.
//
// This is deliberately NOT done by comparing the two models' spans. That was tried and
// it lies in both directions: raw min/max spans are dominated by trained floaters (a
// thin scene reads 30% "larger" than its own surfels), and percentile spans compare
// percentiles of two differently-shaped distributions — the cloud samples surfaces
// uniformly by area while the trainer densifies where it needs detail, so the two
// tails are not comparable. Measuring the ALIGNMENT under a perturbation asks the
// question directly instead of inferring it from a proxy that does not hold.
function perturbScore(splat, occ, voxel, t, centre, pts, apply) {
    const stride = Math.max(1, Math.floor(splat.n / pts));
    let best = -1;
    for (let dx = -0.2; dx <= 0.2001; dx += 0.2) {
        for (let dy = -0.2; dy <= 0.2001; dy += 0.2) {
            for (let dz = -0.2; dz <= 0.2001; dz += 0.2) {
                let hit = 0;
                let seen = 0;
                for (let i = 0; i < splat.n; i += stride) {
                    seen++;
                    const p = apply(
                        splat.xyz[i * 3] + t[0] - centre[0],
                        splat.xyz[i * 3 + 1] + t[1] - centre[1],
                        splat.xyz[i * 3 + 2] + t[2] - centre[2],
                    );
                    if (
                        occ.has(
                            packKey(
                                Math.floor((p[0] + centre[0] + dx) / voxel),
                                Math.floor((p[1] + centre[1] + dy) / voxel),
                                Math.floor((p[2] + centre[2] + dz) / voxel),
                            ),
                        )
                    ) {
                        hit++;
                    }
                }
                best = Math.max(best, hit / Math.max(1, seen));
            }
        }
    }
    return best;
}

// Long thin scenes make the rotation test strict for free: a degree over 100 m
// displaces the far end by ~1.7 m, so nothing subtle survives.
const ROT_ANGLES = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1];
const SCALE_FACTORS = [0.9, 0.95, 1, 1.05, 1.1];

function rotationCheck(splat, occ, voxel, t, centre, pts) {
    const rows = [];
    let worry = null;
    for (const [axis, name] of [
        [0, "x"],
        [1, "y"],
        [2, "z"],
    ]) {
        const scores = ROT_ANGLES.map((deg) => {
            const r = (deg * Math.PI) / 180;
            const s = Math.sin(r);
            const c = Math.cos(r);
            const apply =
                axis === 0
                    ? (x, y, z) => [x, c * y - s * z, s * y + c * z]
                    : axis === 1
                        ? (x, y, z) => [c * x + s * z, y, -s * x + c * z]
                        : (x, y, z) => [c * x - s * y, s * x + c * y, z];
            return perturbScore(splat, occ, voxel, t, centre, pts, apply);
        });
        const bestIdx = scores.indexOf(Math.max(...scores));
        if (ROT_ANGLES[bestIdx] !== 0) worry = `rot ${name} prefers ${ROT_ANGLES[bestIdx]}°`;
        rows.push({ axis: name, angles: ROT_ANGLES, scores });
    }
    return { rows, worry };
}

function scaleCheck(splat, occ, voxel, t, centre, pts) {
    const scores = SCALE_FACTORS.map((f) =>
        perturbScore(splat, occ, voxel, t, centre, pts, (x, y, z) => [x * f, y * f, z * f]),
    );
    const bestIdx = scores.indexOf(Math.max(...scores));
    const best = SCALE_FACTORS[bestIdx];
    return {
        factors: SCALE_FACTORS,
        scores,
        worry: best !== 1 ? `scale prefers x${best}` : null,
    };
}

// --- discovery --------------------------------------------------------------

// A cell keeps its splat artifacts either at the cell root or under `splat/`, and NOT
// necessarily in the same one — modern-house/opus-new has `trained.sog` at the root
// with its `cloud.ply` in `splat/`. So the model and the cloud are searched for
// independently across both, rather than the cloud being required to sit beside the
// model. Cell root first: a file promoted to the root is the published one.
const CELL_DIRS = (cell) => [cell, path.join(cell, "splat")];

// Least lossy first. `.ply` carries float32 positions; a `.sog` has been quantized to
// 16 bits over the scene bounds, which is ~2 mm on a 120 m scene — well under the
// 0.15 m voxel, so it fits fine, but there is no reason to prefer it. `.web.sog` is
// this pipeline's own OUTPUT and may already have a correction baked in; fitting
// against it would measure the correction rather than the model.
const TRAINED_NAMES = ["trained.ply", "trained.sog"];

function findIn(dirs, names) {
    for (const dir of dirs) {
        for (const name of names) {
            const p = path.join(dir, name);
            if (fs.existsSync(p)) return p;
        }
    }
    return null;
}

// Accepts a cell directory, its splat/ subdirectory, or the model file itself, because
// all three are things someone reasonably has to hand.
export function resolveInputs(target, explicitCloud = null) {
    const stat = fs.existsSync(target) ? fs.statSync(target) : null;
    if (!stat) throw new Error(`no such path: ${target}`);

    // Normalize whatever was handed over to the CELL root, so `<cell>`, `<cell>/splat`
    // and `<cell>/splat/trained.ply` all search the same two directories. Climbing to
    // the parent is only ever right for a directory literally named `splat` — the
    // parent of a cell is the slot, and its other cells' clouds are not ours.
    let cell = stat.isFile() ? path.dirname(target) : target;
    if (path.basename(cell) === "splat") cell = path.dirname(cell);
    const dirs = CELL_DIRS(cell);

    const trained = stat.isFile() ? target : findIn(dirs, TRAINED_NAMES);
    if (!trained) {
        throw new Error(
            `no ${TRAINED_NAMES.join(" / ")} in ${cell} or its splat/ — point at the ` +
                "model file directly if it is named something else",
        );
    }

    const cloud = explicitCloud ?? findIn(dirs, ["cloud.ply"]);
    if (!cloud || !fs.existsSync(cloud)) {
        throw new Error(
            `no cloud.ply in ${cell} or its splat/ — pass --to <cloud.ply>. It is the ` +
                "point cloud the trainer was initialized from, and the only reference " +
                "that pins the answer to centimetres.",
        );
    }
    return { trained, cloud };
}

// --- the fit ----------------------------------------------------------------

/**
 * Solve the splat -> world translation. Returns the transform plus everything needed
 * to judge whether to trust it.
 */
export function fitTranslation(target, opts = {}) {
    const o = { ...FIT_DEFAULTS, ...opts };
    const log = o.onLog ?? (() => {});
    const { trained, cloud: cloudFile } = resolveInputs(target, o.to ?? null);

    const cloud = readPositions(cloudFile);
    let sogTemp = null;
    let splat;
    try {
        let modelPly = trained;
        if (trained.toLowerCase().endsWith(".sog")) {
            log(`${path.basename(trained)} is a SOG — converting to PLY to read positions`);
            sogTemp = sogToTempPly(trained);
            modelPly = sogTemp.ply;
        }
        splat = readPositions(modelPly, { maxPts: o.finePts, minOpacity: 0 });
    } finally {
        sogTemp?.cleanup();
    }
    log(
        `cloud ${cloud.count.toLocaleString()} surfels · ` +
            `splat ${splat.count.toLocaleString()} gaussians (${splat.n.toLocaleString()} sampled)`,
    );

    const ce = extent(cloud);
    const se = extent(splat);

    const occ = occupancy(cloud, o.voxel);
    log(`cloud occupies ${occ.size.toLocaleString()} voxels @ ${o.voxel} m`);

    // Seeded from the bounds offset: cheap, and reliably inside the coarse sweep.
    const seed = ce.mid.map((v, i) => Math.round((v - se.mid[i]) * 2) / 2);
    log(`seed (bounds centres): ${seed.map((v) => v.toFixed(2)).join(", ")}`);

    const coarse = sweep(splat, occ, o.voxel, seed, o.radius, 0.5, o.coarsePts);
    log(`coarse ±${o.radius} m / 0.5 m  -> ${fmtT(coarse.t)}  ${pct(coarse.score)}`);
    const refine = sweep(splat, occ, o.voxel, coarse.t, 0.5, 0.1, o.finePts);
    log(`refine ±0.5 m / 0.1 m       -> ${fmtT(refine.t)}  ${pct(refine.score)}`);
    const fine = sweep(splat, occ, o.voxel, refine.t, 0.1, 0.02, o.finePts);
    log(`fine   ±0.1 m / 0.02 m      -> ${fmtT(fine.t)}  ${pct(fine.score)}`);

    const identity = onSurface(splat, occ, o.voxel, 0, 0, 0, o.finePts);
    // Floored at one hit in the sample, not at epsilon. A splat written far enough away
    // scores a clean ZERO at identity, and dividing by 1e-9 then reports "113942893x",
    // which reads as overwhelming proof when it is really just a divide-by-nothing. The
    // smallest rate this sample can resolve is 1/n, so that is the honest denominator.
    const ratio = fine.score / Math.max(identity, 1 / Math.max(splat.n, 1));

    // Sharpness along each axis: a real alignment falls away fast, a flat profile means
    // the answer is not actually localized and should not be trusted.
    const profile = [0, 1, 2].map((axis) => {
        const samples = [];
        for (let d = -1; d <= 1.0001; d += 0.25) {
            const t = [...fine.t];
            t[axis] += d;
            samples.push({ d: +d.toFixed(2), score: onSurface(splat, occ, o.voxel, ...t, o.finePts) });
        }
        return { axis: "xyz"[axis], samples };
    });

    // Both ask the same question — is a plain translation the whole answer? — by
    // perturbing the model and seeing whether the unperturbed case still wins.
    const rotation = o.rotationCheck
        ? rotationCheck(splat, occ, o.voxel, fine.t, ce.mid, 6000)
        : null;
    const scale = o.rotationCheck
        ? scaleCheck(splat, occ, o.voxel, fine.t, ce.mid, 6000)
        : null;

    return {
        trained,
        cloud: cloudFile,
        translate: fine.t.map((v) => +v.toFixed(3)),
        score: fine.score,
        identity,
        ratio,
        scale,
        cloudExtent: ce,
        splatExtent: se,
        profile,
        rotation,
        voxel: o.voxel,
    };
}

const round3 = (e) =>
    Object.fromEntries(Object.entries(e).map(([k, v]) => [k, v.map((n) => +n.toFixed(3))]));

const fmtT = (t) => `(${t.map((v) => v.toFixed(3)).join(", ")})`;
const pct = (v) => `${(v * 100).toFixed(1)}%`;

// Whether the result is worth baking, and why not when it isn't. Kept here rather
// than in the caller so the encoder and the CLI apply the same bar.
export function fitVerdict(fit) {
    const problems = [];
    if (fit.ratio < 3) {
        problems.push(
            `peak is only ${fit.ratio.toFixed(1)}x identity — the splat may already be ` +
                "in world space, or the cloud does not correspond to it",
        );
    }
    if (fit.scale?.worry) {
        problems.push(
            `${fit.scale.worry} over x1 — the model is rescaled, and a translation ` +
                "alone cannot bring a rescaled model into register",
        );
    }
    if (fit.rotation?.worry) {
        problems.push(
            `${fit.rotation.worry} over 0° — a rotation is present and a translation ` +
                "baked over it will only be right near the centre",
        );
    }
    return { ok: problems.length === 0, problems };
}

// --- CLI --------------------------------------------------------------------

function report(fit) {
    console.log("");
    console.log("axis        cloud extent              splat extent (as written)");
    for (const [i, ax] of ["x", "y", "z"].entries()) {
        const c = fit.cloudExtent;
        const s = fit.splatExtent;
        console.log(
            `  ${ax}   [${c.lo[i].toFixed(2).padStart(9)},${c.hi[i].toFixed(2).padStart(9)} ]` +
                `  [${s.lo[i].toFixed(2).padStart(9)},${s.hi[i].toFixed(2).padStart(9)} ]`,
        );
    }
    console.log("");
    console.log(`on-surface at the fit : ${pct(fit.score)}`);
    console.log(`on-surface at identity: ${pct(fit.identity)}`);
    console.log(`peak / identity       : ${fit.ratio.toFixed(1)}x`);
    console.log("");
    console.log("sharpness (on-surface % along each axis through the optimum):");
    for (const p of fit.profile) {
        console.log(
            `  ${p.axis}  ` +
                p.samples
                    .map((s) => `${s.d >= 0 ? "+" : ""}${s.d.toFixed(2)}:${(s.score * 100).toFixed(1)}`)
                    .join("  "),
        );
    }
    if (fit.rotation) {
        console.log("");
        console.log("rotation check (0° should win on every axis):");
        for (const r of fit.rotation.rows) {
            console.log(
                `  ${r.axis}  ` +
                    r.angles
                        .map((a, i) => `${a >= 0 ? "+" : ""}${a}°:${(r.scores[i] * 100).toFixed(1)}`)
                        .join("  "),
            );
        }
    }
    if (fit.scale) {
        console.log("");
        console.log("scale check (x1 should win):");
        console.log(
            "  s  " +
                fit.scale.factors
                    .map((f, i) => `x${f}:${(fit.scale.scores[i] * 100).toFixed(1)}`)
                    .join("  "),
        );
    }
    const verdict = fitVerdict(fit);
    console.log("");
    if (verdict.ok) {
        console.log(`=== bake it:  --translate ${fit.translate.join(",")}`);
    } else {
        console.log("=== DO NOT BAKE THIS YET:");
        for (const p of verdict.problems) console.log(`      - ${p}`);
        console.log(`    (the fit itself was --translate ${fit.translate.join(",")})`);
    }
}

async function main() {
    const argv = process.argv.slice(2);
    const pos = [];
    const opt = {};
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--to") opt.to = argv[++i];
        else if (a === "--radius") opt.radius = parseFloat(argv[++i]);
        else if (a === "--voxel") opt.voxel = parseFloat(argv[++i]);
        else if (a === "--no-rotation-check") opt.rotationCheck = false;
        else if (a === "--json") opt.json = true;
        else pos.push(a);
    }
    if (pos.length < 1) {
        console.error(
            "Usage: node tools/fit-splat-transform.mjs <cell-dir | trained.ply> " +
                "[--to cloud.ply] [--radius 6] [--voxel 0.15] [--no-rotation-check] [--json]",
        );
        process.exit(1);
    }
    let fit;
    try {
        fit = fitTranslation(pos[0], {
            ...opt,
            onLog: opt.json ? () => {} : (m) => console.error(m),
        });
    } catch (e) {
        console.error(`[fit] ${e.message}`);
        process.exit(1);
    }
    if (opt.json) {
        const verdict = fitVerdict(fit);
        console.log(
            JSON.stringify(
                {
                    trained: fit.trained,
                    cloud: fit.cloud,
                    translate: fit.translate,
                    on_surface: +(fit.score * 100).toFixed(1),
                    identity: +(fit.identity * 100).toFixed(1),
                    ratio: +fit.ratio.toFixed(1),
                    rotation_ok: !fit.rotation?.worry,
                    scale_ok: !fit.scale?.worry,
                    ok: verdict.ok,
                    problems: verdict.problems,
                    // The bounds a consumer can check the baked asset against: the
                    // cloud is the world truth, and the splat once corrected should
                    // land on it. Emitted so verifying an encode does not mean
                    // re-parsing two PLYs by hand.
                    cloud_bounds: round3(fit.cloudExtent),
                    corrected_bounds: round3({
                        lo: fit.splatExtent.lo.map((v, i) => v + fit.translate[i]),
                        hi: fit.splatExtent.hi.map((v, i) => v + fit.translate[i]),
                    }),
                },
                null,
                1,
            ),
        );
    } else {
        report(fit);
    }
    if (!fitVerdict(fit).ok) process.exit(2);
}

if (path.resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url)) main();
