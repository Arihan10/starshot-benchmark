import { type NavGraph, routeCosts, shortestPath } from "./navGraph";

// One beat of the tour: a zone, the anchor at its centre, and the hops that walk
// there from the previous beat (the last entry of `route` is the centrepoint).
export type TourStop = { zone: string; index: number; route: number[] };

export type TourProgress = { stop: number; stops: number; zone: string };

// A full revolution, so the heading ends exactly where it began — the walkthrough
// carries your heading across a hop, so finishing the sweep where it started
// means the next leg departs facing the way you already were.
const PAN_TURN = Math.PI * 2;
const PAN_RAMP = 0.15; // eased start/stop, as a fraction of the sweep
const LEVEL_AT = 0.2; // pitch is back to level a fifth of the way round

// Anchors in between are walked through, not arrived at: the hop is quicker and
// it doesn't narrate, so a three-anchor hallway doesn't fire three toasts.
export const PASS_DUR_SCALE = 0.55;

// What the director needs from the engine. It deliberately never touches the
// camera itself: hops go through the walkthrough's own typed traversal, so the
// transitions, FX and narration are exactly the ones a user gets by clicking,
// and the sweep is written onto the same yaw/pitch that drag-look writes.
export type TourHost = {
	busy: () => boolean;
	hop: (index: number, pass: boolean) => void;
	getLook: () => { lon: number; lat: number };
	setLook: (lon: number, lat: number) => void;
	onProgress: () => void;
};

// A zone's centrepoint: the anchor with the lowest total walking cost to the rest
// of its zone (the medoid). Deliberately not the centroid of the coordinates —
// that lands wherever the average happens to fall, which in an L-shaped room or a
// zone wrapped round a stairwell is often inside a wall.
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

// Plan a zone-by-zone tour from wherever the user is standing: one centrepoint
// per zone, ordered nearest-first by walking cost so the route never crosses the
// scene and doubles back, each carrying the anchors that walk you there.
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

	// Open in the zone the user is already in, then chain to whichever remaining
	// centrepoint is the shortest walk from the one we just showed.
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
		if (nextZone === null) break; // the rest is unreachable — show what we can
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

// Trapezoidal ease: accelerate over the first `ramp`, hold a constant rate, then
// decelerate. A constant-rate sweep is what reads as "looking around", but
// starting and stopping it dead reads as a machine — this gives the steady middle
// without the jerk at either end. Returns normalized displacement for t in [0,1].
function rampProgress(t: number, ramp: number): number {
	const area = 1 - ramp; // total under the velocity profile
	if (t <= ramp) return (t * t) / (2 * ramp) / area;
	if (t >= 1 - ramp) return (area - ((1 - t) * (1 - t)) / (2 * ramp)) / area;
	return (ramp / 2 + (t - ramp)) / area;
}

// Runs a planned tour as a two-phase loop per stop: walk the route, then sweep.
// Stopping drops the queue and never moves the camera, so the user keeps exactly
// the spot they're standing in. The one exception is a hop already in flight: the
// walkthrough is anchored to capture points (projection, affordances, exits all
// assume you're AT one), so we let that hop land and stop on arrival rather than
// abandoning the camera between anchors.
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

	// Let go as soon as it's safe to: right now if we're sweeping or idle, else
	// once the hop in flight has landed.
	stop() {
		if (!this.running) return;
		if (!this.sweeping && this.host.busy()) {
			this.stopping = true;
			return;
		}
		this.abort();
	}

	// Drop everything now — scene swap or teardown, where there's no camera left
	// to be considerate of.
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
