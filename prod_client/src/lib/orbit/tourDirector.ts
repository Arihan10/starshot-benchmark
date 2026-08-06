import { type NavGraph, routeCosts, shortestPath } from "./navGraph";

export type TourStop = { zone: string; index: number; route: number[] };

export type TourProgress = { stop: number; stops: number; zone: string };

const PAN_TURN = Math.PI * 2;
const PAN_RAMP = 0.15;
const LEVEL_AT = 0.2;

export const PASS_DUR_SCALE = 0.55;

export type TourHost = {
	busy: () => boolean;
	hop: (index: number, pass: boolean) => void;
	getLook: () => { lon: number; lat: number };
	setLook: (lon: number, lat: number) => void;
	onProgress: () => void;
};

function zoneCenterpoint(graph: NavGraph, members: number[]): number {
	if (members.length < 3) return members[0];
	let best = members[0];
	let bestCost = Infinity;
	for (const candidate of members) {
		const cost = routeCosts(graph, candidate);
		let total = 0;
		for (const m of members) {
			if (m !== candidate && Number.isFinite(cost[m])) total += cost[m];
		}
		if (total < bestCost) {
			bestCost = total;
			best = candidate;
		}
	}
	return best;
}

export function planZoneTour(
	graph: NavGraph,
	zoneOf: (index: number) => string,
	startIndex: number,
): TourStop[] {
	const byZone = new Map<string, number[]>();
	for (const node of graph.nodes) {
		const zone = zoneOf(node.index);
		const members = byZone.get(zone);
		if (members) members.push(node.index);
		else byZone.set(zone, [node.index]);
	}
	if (byZone.size === 0) return [];

	const centers = new Map<string, number>();
	for (const [zone, members] of byZone) {
		centers.set(zone, zoneCenterpoint(graph, members));
	}

	const pending = new Set(centers.keys());
	const order: string[] = [];
	const startZone = zoneOf(startIndex);
	if (pending.delete(startZone)) order.push(startZone);
	let at = centers.get(startZone) ?? startIndex;
	while (pending.size > 0) {
		const cost = routeCosts(graph, at);
		let nextZone: string | null = null;
		let best = Infinity;
		for (const zone of pending) {
			const c = cost[centers.get(zone) as number];
			if (c < best) {
				best = c;
				nextZone = zone;
			}
		}
		if (nextZone === null) break;
		pending.delete(nextZone);
		order.push(nextZone);
		at = centers.get(nextZone) as number;
	}

	const stops: TourStop[] = [];
	let from = startIndex;
	for (const zone of order) {
		const index = centers.get(zone) as number;
		stops.push({ zone, index, route: shortestPath(graph, from, index) ?? [index] });
		from = index;
	}
	return stops;
}

function rampProgress(t: number, ramp: number): number {
	const area = 1 - ramp;
	if (t <= ramp) return (t * t) / (2 * ramp) / area;
	if (t >= 1 - ramp) return (area - ((1 - t) * (1 - t)) / (2 * ramp)) / area;
	return (ramp / 2 + (t - ramp)) / area;
}

export class TourDirector {
	private stops: TourStop[] = [];
	private stopIdx = 0;
	private hopIdx = 0;
	private sweeping = false;
	private stopping = false;
	private running = false;
	private panStart = 0;
	private panLon = 0;
	private panLat = 0;

	constructor(
		private readonly host: TourHost,
		private readonly panMs: number,
	) {}

	get active(): boolean {
		return this.running;
	}

	get progress(): TourProgress | null {
		const stop = this.running ? this.stops[this.stopIdx] : undefined;
		if (!stop) return null;
		return { stop: this.stopIdx + 1, stops: this.stops.length, zone: stop.zone };
	}

	start(stops: TourStop[]) {
		if (stops.length === 0) return;
		this.stops = stops;
		this.stopIdx = 0;
		this.hopIdx = 0;
		this.sweeping = false;
		this.stopping = false;
		this.running = true;
		this.host.onProgress();
	}

	stop() {
		if (!this.running) return;
		if (!this.sweeping && this.host.busy()) {
			this.stopping = true;
			return;
		}
		this.abort();
	}

	abort() {
		if (!this.running) return;
		this.running = false;
		this.stopping = false;
		this.stops = [];
		this.host.onProgress();
	}

	tick(now: number) {
		if (!this.running) return;
		const stop = this.stops[this.stopIdx];
		if (!stop) {
			this.abort();
			return;
		}
		if (!this.sweeping) {
			if (this.host.busy()) return;
			if (this.stopping) {
				this.abort();
				return;
			}
			if (this.hopIdx < stop.route.length) {
				const index = stop.route[this.hopIdx];
				const arriving = this.hopIdx === stop.route.length - 1;
				this.hopIdx++;
				this.host.hop(index, !arriving);
				return;
			}
			const look = this.host.getLook();
			this.sweeping = true;
			this.panStart = now;
			this.panLon = look.lon;
			this.panLat = look.lat;
			return;
		}
		const t = Math.min(1, (now - this.panStart) / this.panMs);
		const swept = rampProgress(t, PAN_RAMP);
		const levelled = Math.min(1, swept / LEVEL_AT);
		this.host.setLook(this.panLon + swept * PAN_TURN, this.panLat * (1 - levelled));
		if (t < 1) return;
		this.stopIdx++;
		this.hopIdx = 0;
		this.sweeping = false;
		if (this.stopIdx >= this.stops.length) {
			this.abort();
			return;
		}
		this.host.onProgress();
	}
}
