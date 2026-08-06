
import { Vector3 } from "three";

export type EdgeType = "walk" | "portal" | "phase" | "vertical" | "far";

export type NavEdge = {
	to: number;
	type: EdgeType;
	dist: number;
	planDist: number;
	dy: number;
	bearing: number;
};

export type NavNode = {
	index: number;
	rendered: NavEdge[];
	all: NavEdge[];
	trapped: boolean;
};

export type NavTuning = {
	reach: number;
	farDist: number;
	verticalDy: number;
	hopPenalty: number;
};

export type NavGraph = {
	nodes: NavNode[];
	zones: string[];
	tuning: NavTuning;
};

export type NavPano = {
	position: [number, number, number];
	zone?: string;
};

const LOS_CANDIDATES = 14;
const DEGREE_CAP = 8;
const NEAREST_GUARANTEED = 3;
const VERTICAL_CAP = 4;
const SPREAD_RAD = (20 * Math.PI) / 180;

const PRIORITY: Record<EdgeType, number> = {
	walk: 0,
	portal: 1,
	vertical: 2,
	phase: 3,
	far: 4,
};

class DSU {
	private readonly parent: number[];
	constructor(n: number) {
		this.parent = Array.from({ length: n }, (_, i) => i);
	}
	find(x: number): number {
		let r = x;
		while (this.parent[r] !== r) r = this.parent[r];
		while (this.parent[x] !== r) {
			const next = this.parent[x];
			this.parent[x] = r;
			x = next;
		}
		return r;
	}
	union(a: number, b: number): boolean {
		const ra = this.find(a);
		const rb = this.find(b);
		if (ra === rb) return false;
		this.parent[ra] = rb;
		return true;
	}
}

const SLICE_MS = 4;

export async function buildNavGraph(
	panos: NavPano[],
	panoLevel: number[] | null,
	isBlocked: (
		a: [number, number, number],
		b: [number, number, number],
	) => boolean,
	tuning: NavTuning,
	breathe?: () => Promise<void>,
): Promise<NavGraph> {
	const { reach: REACH, farDist: FAR_DIST, verticalDy: VERTICAL_DY } = tuning;
	const n = panos.length;
	const pos = panos.map((p) => new Vector3().fromArray(p.position));
	const zones: string[] = [];
	for (const p of panos) {
		if (p.zone && !zones.includes(p.zone)) zones.push(p.zone);
	}

	const sameLevel = (i: number, j: number): boolean => {
		if (panoLevel && panoLevel[i] >= 0 && panoLevel[j] >= 0)
			return panoLevel[i] === panoLevel[j];
		return Math.abs(pos[j].y - pos[i].y) < VERTICAL_DY;
	};

	const adjacentLevel = (i: number, j: number): boolean => {
		if (panoLevel && panoLevel[i] >= 0 && panoLevel[j] >= 0)
			return Math.abs(panoLevel[i] - panoLevel[j]) === 1;
		return true;
	};

	const all: NavEdge[][] = Array.from({ length: n }, () => []);
	let sliceStart = performance.now();
	for (let i = 0; i < n; i++) {
		if (breathe && performance.now() - sliceStart > SLICE_MS) {
			await breathe();
			sliceStart = performance.now();
		}
		const near: Array<{
			j: number;
			dist: number;
			plan: number;
			dy: number;
		}> = [];
		for (let j = 0; j < n; j++) {
			if (j === i) continue;
			const dx = pos[j].x - pos[i].x;
			const dy = pos[j].y - pos[i].y;
			const dz = pos[j].z - pos[i].z;
			const plan = Math.hypot(dx, dz);
			const dist = Math.hypot(dx, dy, dz);
			near.push({ j, dist, plan, dy });
		}
		near.sort((a, b) => a.dist - b.dist);
		let losBudget = LOS_CANDIDATES;
		for (const c of near) {
			const { j, dist, plan, dy } = c;
			const bearing = Math.atan2(
				pos[j].z - pos[i].z,
				pos[j].x - pos[i].x,
			);
			const level = sameLevel(i, j);
			let type: EdgeType;
			if (!level) {
				type =
					Math.abs(dy) >= VERTICAL_DY &&
					dist <= FAR_DIST &&
					adjacentLevel(i, j)
						? "vertical"
						: "far";
			} else if (dist > FAR_DIST || plan > REACH) {
				type = "far";
			} else {
				let occluded = true;
				if (losBudget > 0) {
					losBudget--;
					occluded = isBlocked(panos[i].position, panos[j].position);
				}
				type = occluded ? "portal" : "walk";
			}
			all[i].push({ to: j, type, dist, planDist: plan, dy, bearing });
		}
	}

	const dsu = new DSU(n);
	for (let i = 0; i < n; i++) {
		for (const e of all[i]) {
			if (
				e.type === "walk" ||
				e.type === "portal" ||
				e.type === "vertical"
			)
				dsu.union(i, e.to);
		}
	}
	const phased = new Set<string>();
	const addPhase = (i: number, j: number) => {
		const key = i < j ? `${i}-${j}` : `${j}-${i}`;
		if (phased.has(key)) return;
		phased.add(key);
		const d = pos[i].distanceTo(pos[j]);
		const dyij = pos[j].y - pos[i].y;
		all[i].push({
			to: j,
			type: "phase",
			dist: d,
			planDist: Math.hypot(pos[j].x - pos[i].x, pos[j].z - pos[i].z),
			dy: dyij,
			bearing: Math.atan2(pos[j].z - pos[i].z, pos[j].x - pos[i].x),
		});
		all[j].push({
			to: i,
			type: "phase",
			dist: d,
			planDist: Math.hypot(pos[i].x - pos[j].x, pos[i].z - pos[j].z),
			dy: -dyij,
			bearing: Math.atan2(pos[i].z - pos[j].z, pos[i].x - pos[j].x),
		});
	};
	for (;;) {
		let bi = -1;
		let bj = -1;
		let best = Infinity;
		for (let i = 0; i < n; i++) {
			for (let j = i + 1; j < n; j++) {
				if (dsu.find(i) === dsu.find(j)) continue;
				const d = pos[i].distanceToSquared(pos[j]);
				if (d < best) {
					best = d;
					bi = i;
					bj = j;
				}
			}
		}
		if (bi < 0) break;
		dsu.union(bi, bj);
		addPhase(bi, bj);
	}

	const walkDsu = new DSU(n);
	for (let i = 0; i < n; i++)
		for (const e of all[i]) if (e.type === "walk") walkDsu.union(i, e.to);

	const nodes: NavNode[] = [];
	for (let i = 0; i < n; i++) {
		const edges = all[i];
		const real = edges.filter(
			(e) =>
				e.type === "walk" ||
				e.type === "portal" ||
				e.type === "vertical",
		);
		const trapped = real.length === 0;
		const shown = real.filter(
			(e) =>
				!(e.type === "portal" && walkDsu.find(i) === walkDsu.find(e.to)),
		);
		const rendered: NavEdge[] = [];
		const takenBearings: number[] = [];
		const add = (e: NavEdge) => {
			if (rendered.includes(e)) return;
			rendered.push(e);
			takenBearings.push(e.bearing);
		};
		for (const e of [...shown].sort((a, b) => a.dist - b.dist)) {
			if (rendered.length >= NEAREST_GUARANTEED) break;
			add(e);
		}
		const ordered = [...shown].sort(
			(a, b) => PRIORITY[a.type] - PRIORITY[b.type] || a.dist - b.dist,
		);
		for (const e of ordered) {
			if (rendered.length >= DEGREE_CAP) break;
			if (rendered.includes(e)) continue;
			if (
				takenBearings.some(
					(b) => Math.abs(angleDelta(b, e.bearing)) < SPREAD_RAD,
				)
			)
				continue;
			add(e);
		}
		const vSeen = new Set<number>();
		for (const e of shown
			.filter((e) => e.type === "vertical")
			.sort((a, b) => a.dist - b.dist)) {
			if (vSeen.size >= VERTICAL_CAP) break;
			const lvl = panoLevel ? panoLevel[e.to] : Math.sign(e.dy);
			if (vSeen.has(lvl)) continue;
			vSeen.add(lvl);
			add(e);
		}
		if (trapped) for (const e of edges) if (e.type === "phase") add(e);
		nodes.push({ index: i, rendered, all: edges, trapped });
	}

	return { nodes, zones, tuning };
}

export function angleDelta(a: number, b: number): number {
	let d = b - a;
	while (d > Math.PI) d -= 2 * Math.PI;
	while (d < -Math.PI) d += 2 * Math.PI;
	return d;
}

export function edgeVerb(type: EdgeType, dy = 0): string {
	switch (type) {
		case "walk":
			return "walked";
		case "portal":
			return "through the doorway";
		case "phase":
			return "phased through the wall";
		case "vertical":
			return dy >= 0 ? "up a level" : "down a level";
		case "far":
			return "traveled across";
	}
}

export function edgeNoun(type: EdgeType): string {
	return type;
}

const ROUTE_COST: Partial<Record<EdgeType, number>> = {
	walk: 1,
	portal: 1.1,
	vertical: 1.4,
	phase: 8,
};

function dijkstra(
	graph: NavGraph,
	from: number,
): { dist: number[]; prev: number[] } {
	const hopPenalty = graph.tuning.hopPenalty;
	const n = graph.nodes.length;
	const dist = new Array<number>(n).fill(Infinity);
	const prev = new Array<number>(n).fill(-1);
	const settled = new Array<boolean>(n).fill(false);
	if (from < 0 || from >= n) return { dist, prev };
	dist[from] = 0;
	for (;;) {
		let u = -1;
		let best = Infinity;
		for (let i = 0; i < n; i++) {
			if (!settled[i] && dist[i] < best) {
				best = dist[i];
				u = i;
			}
		}
		if (u < 0) break;
		settled[u] = true;
		for (const e of graph.nodes[u].all) {
			const scale = ROUTE_COST[e.type];
			if (scale === undefined) continue;
			const d = dist[u] + e.dist * scale + hopPenalty;
			if (d < dist[e.to]) {
				dist[e.to] = d;
				prev[e.to] = u;
			}
		}
	}
	return { dist, prev };
}

export function routeCosts(graph: NavGraph, from: number): number[] {
	return dijkstra(graph, from).dist;
}

export function shortestPath(
	graph: NavGraph,
	from: number,
	to: number,
): number[] | null {
	if (from === to) return [];
	const { dist, prev } = dijkstra(graph, from);
	if (!Number.isFinite(dist[to])) return null;
	const hops: number[] = [];
	for (let at = to; at !== from && at >= 0; at = prev[at]) hops.unshift(at);
	return hops;
}
