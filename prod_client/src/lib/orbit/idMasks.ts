import {
	ClampToEdgeWrapping,
	DataTexture,
	NearestFilter,
	NoColorSpace,
	RepeatWrapping,
	RGBAFormat,
	RGFormat,
	type Texture,
	UnsignedByteType,
} from "three";

// Per-pixel object-ID masks: the walkthrough's answer to "what am I pointing at",
// and the thing it draws a silhouette around.
//
// The alternative was raycasting the projection proxy, but that proxy is the whole
// scene decimated to 40k triangles — roughly a hundred per object — so its
// silhouettes don't match what you see and its gaps let rays through solid
// surfaces. Instead the capture worker, which is holding the full-resolution
// meshes, records the visible object at every pixel of every pano. Occlusion is
// resolved there, once, by the same depth buffer that drew the photo.
//
// That works because the walkthrough is pinned: it only ever stands ON a capture
// point, so from here a screen pixel is nothing but a ray direction and the answer
// is a fixed function of it. Hover is one array read; the highlight is one texture
// lookup. Neither touches geometry, and neither degrades as objects get complex.
//
// The container is SID1, written by the capture worker. Its byte layout and the
// equirect convention live in client/public/js/idmask.js — that file is the
// definition, this is its reader, and the two have to move together.
//
//   header (24 bytes, little-endian)
//     0   char[4] magic "SID1"      4  u16 width        6  u16 height
//     8   u8 planes                 9  u8 id filter    10  u8 index bytes
//     11  u8 coverage filter       12  u16 palette     14  u16 supersample
//     16  u32 id bytes             20  u32 coverage bytes
//   then u16[palette] (local index -> global object index), then two
//   deflate streams: the id plane, then the coverage plane.
//
// Coverage is what keeps a boundary from stair-stepping. Ids can't be antialiased —
// averaging two object indices is meaningless — so the capture supersamples and
// stores, per texel, the fraction of samples its winning id took. That fraction
// puts the edge back at its true sub-texel position when reconstructed.

const MAGIC = 0x31444953; // "SID1" read as a little-endian uint32
const HEADER_BYTES = 24;
const FILTER_SUBLEFT = 1;
export const ID_BACKGROUND = 0;

// Ids and coverage are uploaded INTERLEAVED, so a shader reads both in one tap
// (it takes four neighbours per pixel already) and the CPU hit-test reads ids out
// of the very same buffer — one copy serving both, rather than two.
const STRIDE_8 = 2; // RG8:   id, coverage
const STRIDE_16 = 4; // RGBA8: id lo, id hi, coverage, 255

export type IdMask = {
	width: number;
	height: number;
	indexBytes: number;
	supersample: number;
	palette: Uint16Array;
	localOf: Map<number, number>;
	data: Uint8Array;
	stride: number;
	texture: Texture;
	bytes: number;
};

const nextFrame = () => new Promise<void>((r) => setTimeout(r, 0));

async function inflate(bytes: Uint8Array): Promise<Uint8Array> {
	const stream = new Blob([bytes as BlobPart])
		.stream()
		.pipeThrough(new DecompressionStream("deflate"));
	return new Uint8Array(await new Response(stream).arrayBuffer());
}

// PNG's "Sub" predictor, undone in place: each sample was stored as the delta to
// its left neighbour, wrapping in the plane's own dtype (which typed-array
// assignment does for us).
function unSubLeft(
	plane: Uint8Array | Uint16Array,
	width: number,
	height: number,
) {
	for (let row = 0; row < height; row++) {
		const o = row * width;
		for (let x = 1; x < width; x++) plane[o + x] += plane[o + x - 1];
	}
}

/** Read one SID1 container. Yields between phases so a 4096×2048 mask can't
 * swallow a frame — it lands a moment after arrival rather than as a hitch. */
export async function decodeIdMask(buffer: ArrayBuffer): Promise<IdMask> {
	if (buffer.byteLength < HEADER_BYTES) throw new Error("idmask: truncated header");
	const view = new DataView(buffer);
	if (view.getUint32(0, true) !== MAGIC) throw new Error("idmask: not a SID1 container");
	const width = view.getUint16(4, true);
	const height = view.getUint16(6, true);
	const planes = view.getUint8(8);
	const idFilter = view.getUint8(9);
	const indexBytes = view.getUint8(10);
	const coverageFilter = view.getUint8(11);
	const paletteCount = view.getUint16(12, true);
	const supersample = view.getUint16(14, true);
	const idBytes = view.getUint32(16, true);
	const coverageBytes = view.getUint32(20, true);
	const paletteEnd = HEADER_BYTES + paletteCount * 2;
	if (buffer.byteLength < paletteEnd + idBytes + coverageBytes) {
		throw new Error("idmask: truncated payload");
	}

	const count = width * height;
	const palette = new Uint16Array(buffer.slice(HEADER_BYTES, paletteEnd));

	const rawIds = await inflate(new Uint8Array(buffer, paletteEnd, idBytes));
	const ids =
		indexBytes === 1
			? new Uint8Array(rawIds.buffer, rawIds.byteOffset, count)
			: new Uint16Array(rawIds.buffer, rawIds.byteOffset, count);
	if (idFilter === FILTER_SUBLEFT) unSubLeft(ids, width, height);
	await nextFrame();

	let coverage: Uint8Array | null = null;
	if (planes >= 2 && coverageBytes > 0) {
		const raw = await inflate(
			new Uint8Array(buffer, paletteEnd + idBytes, coverageBytes),
		);
		coverage = new Uint8Array(raw.buffer, raw.byteOffset, count);
		if (coverageFilter === FILTER_SUBLEFT) unSubLeft(coverage, width, height);
		await nextFrame();
	}

	const stride = indexBytes === 1 ? STRIDE_8 : STRIDE_16;
	const data = new Uint8Array(count * stride);
	if (stride === STRIDE_8) {
		for (let i = 0, o = 0; i < count; i++, o += STRIDE_8) {
			data[o] = ids[i];
			data[o + 1] = coverage ? coverage[i] : 255;
		}
	} else {
		for (let i = 0, o = 0; i < count; i++, o += STRIDE_16) {
			const local = ids[i];
			data[o] = local & 255;
			data[o + 1] = local >> 8;
			data[o + 2] = coverage ? coverage[i] : 255;
			data[o + 3] = 255;
		}
	}

	// Nearest and no mips because ids don't interpolate; repeat in u so the
	// equirect seam stays continuous; flipY OFF, which is why a shader samples at
	// t = 1 - v (the plane is stored top-down, row 0 pointing straight up).
	const texture = new DataTexture(
		data,
		width,
		height,
		stride === STRIDE_8 ? RGFormat : RGBAFormat,
		UnsignedByteType,
	);
	texture.magFilter = NearestFilter;
	texture.minFilter = NearestFilter;
	texture.generateMipmaps = false;
	texture.wrapS = RepeatWrapping;
	texture.wrapT = ClampToEdgeWrapping;
	texture.unpackAlignment = 1;
	texture.colorSpace = NoColorSpace;
	texture.needsUpdate = true;

	const localOf = new Map<number, number>();
	for (let i = 0; i < palette.length; i++) localOf.set(palette[i], i + 1);

	return {
		width,
		height,
		indexBytes,
		supersample,
		palette,
		localOf,
		data,
		stride,
		texture,
		bytes: buffer.byteLength,
	};
}

function globalAt(mask: IdMask, col: number, row: number): number {
	const x = ((col % mask.width) + mask.width) % mask.width; // the equirect wraps in u
	const y = row < 0 ? 0 : row >= mask.height ? mask.height - 1 : row;
	const o = (y * mask.width + x) * mask.stride;
	const local =
		mask.stride === STRIDE_8 ? mask.data[o] : mask.data[o] | (mask.data[o + 1] << 8);
	return local === 0 ? ID_BACKGROUND : mask.palette[local - 1];
}

/** The global object index a view direction lands on (0 = background). `radius`
 * takes a majority over a (2r+1)² block, which stops a sliver under the cursor
 * flickering the label between two objects. */
export function sampleIdMask(
	mask: IdMask,
	dx: number,
	dy: number,
	dz: number,
	radius = 0,
): number {
	const len = Math.hypot(dx, dy, dz) || 1;
	const u = Math.atan2(dz / len, dx / len) / (Math.PI * 2) + 0.5;
	const v = Math.asin(Math.min(1, Math.max(-1, dy / len))) / Math.PI + 0.5;
	const col = Math.floor(u * mask.width);
	const row = Math.floor((1 - v) * mask.height);
	if (radius <= 0) return globalAt(mask, col, row);

	let win = globalAt(mask, col, row);
	let best = 0;
	const tally = new Map<number, number>();
	for (let y = row - radius; y <= row + radius; y++) {
		for (let x = col - radius; x <= col + radius; x++) {
			const g = globalAt(mask, x, y);
			const n = (tally.get(g) ?? 0) + 1;
			tally.set(g, n);
			if (n > best) {
				best = n;
				win = g;
			}
		}
	}
	return win;
}

// A decoded 4096×2048 mask costs ~17 MB (8-bit ids) to ~34 MB (16-bit) — the
// interleaved buffer, shared by the GPU and the hit-test. So they are evicted,
// unlike the panos: keep the anchor you're standing on plus a couple you might
// step back to. The A/B workspace runs two engines, so this budget is doubled in
// practice.
const RESIDENT = 3;

/** Streams the current anchor's mask and keeps a few recent ones warm. Nothing is
 * prefetched: a mask is a couple of hundred KB behind a pano that is already
 * loading, and speculatively decoding neighbours would cost more memory than the
 * wait is worth. */
export class MaskStreamer {
	private masks = new Map<number, IdMask>();
	private lru: number[] = [];
	private loading = new Set<number>();

	constructor(
		private readonly token: () => number,
		private readonly urlFor: (index: number) => string | undefined,
		private readonly onReady: (index: number) => void,
	) {}

	/** The decoded mask for an anchor, or null while it is missing or in flight. */
	get(index: number): IdMask | null {
		const mask = this.masks.get(index);
		if (!mask) return null;
		this.touch(index);
		return mask;
	}

	/** Residency without touching the LRU — for state the UI polls every frame. */
	has(index: number): boolean {
		return this.masks.has(index);
	}

	/** Fire-and-forget: start decoding this anchor's mask if it isn't resident. */
	request(index: number) {
		if (this.masks.has(index) || this.loading.has(index)) return;
		const url = this.urlFor(index);
		if (!url) return;
		this.loading.add(index);
		const token = this.token();
		void (async () => {
			try {
				const res = await fetch(url);
				if (!res.ok) throw new Error(`mask HTTP ${res.status}`);
				const mask = await decodeIdMask(await res.arrayBuffer());
				if (token !== this.token()) {
					mask.texture.dispose();
					return;
				}
				this.masks.set(index, mask);
				this.touch(index);
				this.evict();
				this.onReady(index);
			} catch {
				// A missing or malformed mask just means this anchor has no object
				// addressing; the walkthrough is unaffected.
			} finally {
				this.loading.delete(index);
			}
		})();
	}

	reset() {
		for (const mask of this.masks.values()) mask.texture.dispose();
		this.masks.clear();
		this.lru = [];
		this.loading.clear();
	}

	private touch(index: number) {
		const at = this.lru.indexOf(index);
		if (at >= 0) this.lru.splice(at, 1);
		this.lru.push(index);
	}

	private evict() {
		while (this.lru.length > RESIDENT) {
			const drop = this.lru.shift();
			if (drop === undefined) break;
			this.masks.get(drop)?.texture.dispose();
			this.masks.delete(drop);
		}
	}
}
