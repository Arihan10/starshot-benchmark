// Tiny leaf split out of the retired in-page report.js: the per-scene
// computed-attention tally the LIVE /tf drawer reads to gate the "open workspace"
// button (the analysis workspace itself is the separate /tf-workspace iframe). By
// keeping just this here — importing only state.js — /tf no longer eagerly loads
// the ~350 KB report/reportGraphs/overview/summary bundle.

import { state } from "./state.js";

// Distinct computed OR stale event indices in a server attention-status payload.
function countComputedAttention(r) {
	return new Set([...(r?.computed || []), ...(r?.stale || [])].map(Number)).size;
}

// Record how many of a scene's steps have a stored analysis (drives the button
// enable + the ⚗ tab). Called from selectCell + every attention status poll.
export function setSceneAttentionCount(slotId, r) {
	if (!slotId) return;
	state.sceneAttentionCounts.set(String(slotId), countComputedAttention(r));
}

export function sceneAttentionCount(slotId) {
	return state.sceneAttentionCounts.get(String(slotId)) || 0;
}
