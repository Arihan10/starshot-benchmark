import type { Texture } from "three";
import { loadPanoTexture } from "./loaders";

export type PanoEntry = {
	id: string;
	name?: string;
	zone?: string;
	level?: number;
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

export class PanoStreamer {
	private entries: PanoEntry[] = [];

	constructor(
		private readonly token: () => number,
		private readonly onReady: (i: number) => void,
	) {}

	get list(): PanoEntry[] {
		return this.entries;
	}

	reset(entries: PanoEntry[] = []) {
		for (const p of this.entries) {
			p.texture?.dispose();
			if (p.placeholderTexture && p.placeholderTexture !== p.texture)
				p.placeholderTexture.dispose();
		}
		this.entries = entries;
	}

	request(i: number) {
		const p = this.entries[i];
		if (!p || p.texture || p.requested) return;
		this.startLoad(i);
	}

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
