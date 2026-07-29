import type { Texture } from "three";
import { loadPanoTexture } from "./loaders";

// A capture point. Textures are loaded lazily (on enter / on movement): a low-res
// blurred placeholder shows first, then the full image swaps in. `texture` is the
// current best (placeholder or full); `placeholderTexture` is kept only so it can
// be disposed at teardown without racing a bound shader uniform.
export type PanoEntry = {
	id: string;
	name?: string;
	zone?: string; // which manifest zone this capture sits in (connector travel)
	level?: number; // the storey it stands on, per the manifest; absent on older tours
	position: [number, number, number];
	forward?: [number, number, number];
	url: string;
	placeholderUrl: string;
	texture: Texture | null;
	placeholderTexture: Texture | null;
	hasFull: boolean;
	requested: boolean;
	ready?: Promise<void>;
	resolveReady?: () => void;
};

// Owns the pano list and their textures: lazy LQIP→full loading, in-flight
// invalidation (via the engine's scene-load token), and disposal. The engine
// reads positions/ids/textures off `list`; the streamer only mutates the loading
// fields. `onReady(i)` lets the engine refresh the sphere backdrop when the
// current capture's texture arrives (projection mode re-reads every frame).
export class PanoStreamer {
	private entries: PanoEntry[] = [];

	constructor(
		private readonly token: () => number,
		private readonly onReady: (i: number) => void,
	) {}

	get list(): PanoEntry[] {
		return this.entries;
	}

	// Swap in a new scene's panos, disposing the previous set's textures.
	reset(entries: PanoEntry[] = []) {
		for (const p of this.entries) {
			p.texture?.dispose();
			if (p.placeholderTexture && p.placeholderTexture !== p.texture)
				p.placeholderTexture.dispose();
		}
		this.entries = entries;
	}

	// Fire-and-forget trigger (per-frame safe): start loading pano `i` if needed.
	request(i: number) {
		const p = this.entries[i];
		if (!p || p.texture || p.requested) return;
		this.startLoad(i);
	}

	// Resolves once a texture (placeholder or full) is set — for paths that need
	// something to show before animating (sphere-mode travel).
	ensure(i: number): Promise<void> {
		const p = this.entries[i];
		if (!p || p.texture) return Promise.resolve();
		if (!p.ready) {
			p.ready = new Promise<void>((res) => {
				p.resolveReady = res;
			});
		}
		const ready = p.ready;
		this.request(i);
		return ready;
	}

	private startLoad(i: number) {
		const p = this.entries[i];
		if (p.requested) return;
		p.requested = true;
		const token = this.token();
		// Low-res blurred preview first (streams in fast), then the full image
		// sharpens in place — the panorama page's LQIP→full swap.
		loadPanoTexture(p.placeholderUrl)
			.then((tex) => {
				if (token !== this.token() || p.hasFull) {
					tex.dispose();
					return;
				}
				p.placeholderTexture = tex;
				if (!p.texture) {
					p.texture = tex;
					this.settle(i);
				}
			})
			.catch(() => {});
		loadPanoTexture(p.url)
			.then((tex) => {
				if (token !== this.token()) {
					tex.dispose();
					return;
				}
				p.hasFull = true;
				p.texture = tex;
				this.settle(i);
			})
			.catch(() => {});
	}

	private settle(i: number) {
		const p = this.entries[i];
		p.resolveReady?.();
		p.resolveReady = undefined;
		this.onReady(i);
	}
}
