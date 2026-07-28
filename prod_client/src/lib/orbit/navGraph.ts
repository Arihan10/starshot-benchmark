// The typed navigation graph — the spine of the walkthrough UX.
//
// The design principle (see the UX brief): navigation isn't about moving a
// camera, it's about *narrating topology*. Every reachable relationship between
// two capture points is one of five edge types, and each type gets its own
// affordance shape and its own transition verb. A real building would let a
// navmesh classify these; our scenes are authored, so we classify from the
// signals the pipeline already gives us — capture positions, per-node `forward`
// headings, `zone` labels, floor `level`s, and a proxy mesh for line-of-sight —
// entirely on the client at load time.
//
//   walk     (A) same level, clear line of sight, within reach   → floor puck
//   portal   (B) same level, occluded (around a wall / doorway)   → portal glow
//   vertical (D) adjacent floor, small plan distance              → up/down halo
//   far      (E) beyond reach / a different cluster               → map / search only
//   phase    (C) no walkable path — DERIVED from connectivity     → ghost puck
//
// Phase edges are medicine, not candy: they're never minted per-pair. We build
// the connected components of the walk/portal/vertical graph, then promote the
// cheapest cross-component links (a spanning tree) to `phase` so the whole scene
// is reachable — exactly and only where a zone would otherwise be sealed. A node
// with no walk/portal/vertical exit is `trapped`; its ghost puck always shows so
// the user is never stuck.

import { Vector3 } from "three";

export type EdgeType = "walk" | "portal" | "phase" | "vertical" | "far";

export type NavEdge = {
	to: number; // destination pano index
	type: EdgeType;
	dist: number; // straight-line distance (m)
	planDist: number; // horizontal (XZ) distance (m)
	dy: number; // signed height change to the destination (m)
	bearing: number; // world azimuth from→to, atan2(dz, dx) — for gaze disclosure + spread
};

export type NavNode = {
	index: number;
	// The in-view affordances: walk/portal/vertical, capped + bearing-spread, plus
	// phase edges only when this node is trapped (they're its only way out).
	rendered: NavEdge[];
	// Every classified edge — powers click-anywhere routing and the sonar reveal.
	all: NavEdge[];
	trapped: boolean; // no walk/portal/vertical exit → the ghost puck must always show
};

export type NavGraph = {
	nodes: NavNode[];
	zones: string[]; // unique zone labels, in first-seen order
};

export type NavPano = {
	position: [number, number, number];
	zone?: string;
};

// Tunables. Distances are in world meters; the scene is authored at real scale.
const REACH = 30; // furthest an in-view walk/portal reaches (matches HOTSPOT_REACH)
const FAR_DIST = 20; // beyond this a link is "far" (lives in map/search, not in-view)
const VERTICAL_DY = 2.0; // min |Δy| to read as a floor change
const LOS_CANDIDATES = 14; // nearest same-level neighbors per node that get an LOS raycast
const DEGREE_CAP = 8; // max rendered affordances per node
const NEAREST_GUARANTEED = 3; // the closest few ALWAYS render, whatever their bearing
const VERTICAL_CAP = 4; // always render up to this many level-change beacons
// Suppress a second affordance whose bearing hugs one already chosen, so they fan
// out instead of stacking — kept gentle so genuinely nearby points aren't hidden.
const SPREAD_RAD = (20 * Math.PI) / 180;

const PRIORITY: Record<EdgeType, number> = {
	walk: 0,
	portal: 1,
	vertical: 2,
	phase: 3,
	far: 4,
};

// Minimal union-find for the connectivity guarantee.
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

export function buildNavGraph(
	panos: NavPano[],
	panoLevel: number[] | null,
	isBlocked: (
		a: [number, number, number],
		b: [number, number, number],
	) => boolean,
): NavGraph {
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

	// Whether j sits exactly ONE storey from i. Only decidable when both panos have
	// a known level; a tour with no minimap slices carries no level indices to
	// count, so there the Y-based cross-level test above stands on its own.
	const adjacentLevel = (i: number, j: number): boolean => {
		if (panoLevel && panoLevel[i] >= 0 && panoLevel[j] >= 0)
			return Math.abs(panoLevel[i] - panoLevel[j]) === 1;
		return true;
	};

	// Per ordered pair, classify everything *except* phase (which is connectivity-
	// derived below). LOS is only raycast for the nearest same-level candidates
	// within reach, so the whole build stays ~O(n · LOS_CANDIDATES) raycasts.
	const all: NavEdge[][] = Array.from({ length: n }, () => []);
	for (let i = 0; i < n; i++) {
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
		// Which same-level, in-reach neighbors are close enough to be worth an LOS
		// test (walk vs. portal). Everything else is portal/vertical/far by geometry.
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
				// Cross-level: a LEVEL CHANGE (vertical beacon + iris transition)
				// whenever it's within the far cutoff — so floor-to-floor links stay
				// visible in-view even across the room, not only when stacked directly
				// overhead. Only genuinely distant cross-level links fall to map-only far.
				//
				// ADJACENT FLOORS ONLY: a beacon promises one storey of travel, and no
				// staircase skips a floor, so a link spanning more than one level
				// (0 → 2) is never vertical. It falls to `far`, still reachable from the
				// map/search or by routing through the floor in between.
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

	// Connectivity guarantee. Union nodes joined by a real (walk/portal/vertical)
	// edge, then promote the cheapest cross-component pair to a bidirectional phase
	// edge until the graph is one component. This is the *only* source of phase.
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
	// Repeatedly bridge the closest two still-separate nodes until connected.
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
		if (bi < 0) break; // one component (or n ≤ 1)
		dsu.union(bi, bj);
		addPhase(bi, bj);
	}

	// Walk-only reachability: which nodes an unbroken chain of clear-line-of-sight
	// hops connects. If B lands in A's walk component then A can get to B on foot
	// without cutting through anything — what the portal filter below reads.
	const walkDsu = new DSU(n);
	for (let i = 0; i < n; i++)
		for (const e of all[i]) if (e.type === "walk") walkDsu.union(i, e.to);

	// Per-node selection of the rendered affordances.
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
		// A portal claims "the way there is around or through this wall" — untrue when
		// the destination already sits in our walk component, because a chain of
		// clear-LOS hops reaches it. Drop those so an orange spot never marks a place
		// you can simply walk to. `trapped` is still judged on the unfiltered set (the
		// edge does exist) and `all` is untouched, so hover, the exits panel, the
		// sonar reveal and click-anywhere routing all still see it.
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
		// 1) Always show the closest few, regardless of bearing — a nearby point must
		//    never be suppressed just because something else lies the same way.
		for (const e of [...shown].sort((a, b) => a.dist - b.dist)) {
			if (rendered.length >= NEAREST_GUARANTEED) break;
			add(e);
		}
		// 2) Fill to the cap by priority (walk < portal < vertical) then distance,
		//    fanning out by bearing so they don't stack in one direction.
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
		// 3) ALWAYS surface a level-change beacon toward EACH reachable floor — the
		//    nearest vertical to each distinct destination level. Without this, a
		//    middle floor shows only the beacons to whichever floor is CLOSER (down,
		//    usually), and "the way up" never makes the cut. One beacon per adjacent
		//    floor, nearest first, capped — since verticals span a single storey, a
		//    middle floor surfaces exactly two: the storey above and the one below.
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
		// 4) A trapped node's only exits are phase edges — always surface them.
		if (trapped) for (const e of edges) if (e.type === "phase") add(e);
		nodes.push({ index: i, rendered, all: edges, trapped });
	}

	return { nodes, zones };
}

// Signed smallest angle between two azimuths, in (-π, π].
export function angleDelta(a: number, b: number): number {
	let d = b - a;
	while (d > Math.PI) d -= 2 * Math.PI;
	while (d < -Math.PI) d += 2 * Math.PI;
	return d;
}

// Human verb for an arrival toast / exits panel / screen reader.
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

// Short affordance noun, for the exits panel / a11y ("walk", "phase", …).
export function edgeNoun(type: EdgeType): string {
	return type;
}

// --- routing --------------------------------------------------------------
//
// Travel between two distant anchors should *walk* rather than teleport, so we
// need the anchors in between. Cost is edge length scaled per type: a phase link
// is priced high enough that a route only cuts through a wall when there is
// genuinely no way round, and a flat per-hop charge prefers a few long strides
// over a zigzag through every anchor in the room. `far` is excluded outright —
// it isn't a traversal, it's the map-and-search-only class. Because phase edges
// are minted as a spanning tree across the components, what's left is always one
// connected graph: a route exists between any two nodes.
const ROUTE_COST: Partial<Record<EdgeType, number>> = {
	walk: 1,
	portal: 1.1,
	vertical: 1.4,
	phase: 8,
};
const HOP_PENALTY = 1.5; // per hop, in meters-equivalent

// Dense Dijkstra — n is in the tens, so an array scan beats carrying a heap.
function dijkstra(
	graph: NavGraph,
	from: number,
): { dist: number[]; prev: number[] } {
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
			const d = dist[u] + e.dist * scale + HOP_PENALTY;
			if (d < dist[e.to]) {
				dist[e.to] = d;
				prev[e.to] = u;
			}
		}
	}
	return { dist, prev };
}

// Walking cost from `from` to every node (Infinity where there's no route).
export function routeCosts(graph: NavGraph, from: number): number[] {
	return dijkstra(graph, from).dist;
}

// The hops that walk `from` → `to`: the anchors in between, then `to` itself.
// Empty when already there; null when no walkable route exists.
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
