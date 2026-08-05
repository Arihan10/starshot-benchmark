// Adaptive quality controller — maximize visual quality subject to a frame-time
// budget, rather than merely survive one.
//
// The objective is an optimisation, not a panic button: find the HIGHEST tier of
// a quality ladder whose frame time still meets the target, and sit there. So it
// climbs as readily as it drops, and it starts near the top and works down
// instead of starting low and creeping up.
//
// The ladder moves four levers together, chosen because each is monotone in both
// cost and quality and none of them fights the engine:
//   * splat budget     — geometry, sort and work-buffer cost, and the FIRST thing
//                        spent. It is also the only geometry lever the controller
//                        touches: the engine's own budget enforcement already
//                        coarsens the farthest octree nodes first and rescales the
//                        whole LOD distance ladder to converge, so also driving
//                        `lodBaseDistance` would put a second feedback loop on top
//                        of that one and the two would fight.
//   * foveation        — fill rate, and perceptually the cheapest thing to spend,
//                        so it goes early and generously.
//   * render scale     — fill rate too, but it is spent LAST and never below 0.8.
//                        Splats are already soft, so dropping resolution doesn't
//                        read as "slightly softer", it reads as pixelated — the one
//                        degradation that makes a splat scene look broken rather
//                        than cheaper.
//   * anti-alias       — a small fixed cost for steadier thin geometry; only
//                        affordable at the top of the ladder.
// It also raises `lodUnderfillLimit` off the engine's default of 0, which is not a
// performance lever at all: it lets a node draw an already-resident coarser level
// while the one it wants downloads, instead of leaving a hole. That is free
// quality, so the controller always takes it — more of it at the bottom of the
// ladder, where coarsening and streaming lag are both worse.
//
// Hysteresis matters more than the thresholds. Dropping is immediate because a
// stutter is worse than a lost tier; climbing needs sustained headroom, an idle
// network (a budget raise during a refine both pollutes the frame-time reading and
// triggers a fetch storm), and a per-tier backoff so a rung this machine cannot
// hold is not retried every second forever.
//
// The backoff is deliberately SHORT and forgiving, because most drops are lies. A
// chunk decode, a shader compile or a window resize all spike one frame, and
// blaming the rung for that leaves you stuck a rung below what your machine can
// actually do — for a long time, if the penalty keeps doubling. So: a drop that
// happens while chunks are in flight never records a penalty at all (the
// measurement isn't about the rung), the penalty ceiling is seconds rather than a
// minute, and holding a rung comfortably decays the penalties above it. A wrong
// probe costs one frame; refusing to probe costs quality for as long as you sit
// there.
//
// Note on vsync: `frameMs` is a wall-clock frame delta, so a vsync-locked 60 Hz
// display reads 16.7ms no matter how much headroom is left. The thresholds are
// therefore set AT the target rather than below it — meeting the target counts as
// headroom, and the probe-and-back-off loop is what discovers the real ceiling.

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// index 0 = cheapest, last = best. `dpr` is a fraction of the device pixel ratio so
// the ladder means the same thing on a 1x and a 2x display — and it bottoms out at
// 0.8, because below that a splat scene stops looking softer and starts looking
// pixelated. Geometry and foveation carry the range instead: rung 0 draws 17x fewer
// Gaussians than rung 6, which is far more headroom than resolution could buy.
export const TIERS = [
    { dpr: 0.80, splatBudget: 400_000, foveationStrength: 0.60, antiAlias: false, lodUnderfillLimit: 2 },
    { dpr: 0.85, splatBudget: 700_000, foveationStrength: 0.45, antiAlias: false, lodUnderfillLimit: 2 },
    { dpr: 0.90, splatBudget: 1_100_000, foveationStrength: 0.30, antiAlias: false, lodUnderfillLimit: 1 },
    { dpr: 0.95, splatBudget: 1_700_000, foveationStrength: 0.18, antiAlias: false, lodUnderfillLimit: 1 },
    { dpr: 1.0, splatBudget: 2_500_000, foveationStrength: 0.08, antiAlias: false, lodUnderfillLimit: 1 },
    { dpr: 1.0, splatBudget: 3_500_000, foveationStrength: 0, antiAlias: true, lodUnderfillLimit: 1 },
    { dpr: 1.0, splatBudget: 0, foveationStrength: 0, antiAlias: true, lodUnderfillLimit: 1 },
];

// The levers the ladder owns; the panel greys these out while it is running.
export const OWNED_KEYS = [
    "pixelRatio",
    "splatBudget",
    "foveationStrength",
    "antiAlias",
    "lodUnderfillLimit",
];

// Budgeted but high — starting at the uncapped top tier would spike a
// multi-million-splat scene before the first measurement came back.
const START_TIER = TIERS.length - 2;

const EVAL_MS = 250; // how often a decision is made
const SETTLE_MS = 500; // ignored window after a change, so the frame-time EMA catches up
const CLIMB_STREAK = 2; // consecutive comfortable evaluations before promoting (~0.5s)
const OVER_FACTOR = 1.25; // frame time this far past target counts as missing it
const OK_FACTOR = 1.05; // ...and this close to target counts as meeting it
const BACKOFF_BASE_MS = 1500;
const BACKOFF_MAX_MS = 8000; // a rung is never untouchable for more than a few seconds
// Hold a rung comfortably this long and one failure is forgiven above it, so a rung
// lost to a one-off hitch comes back on its own.
const FORGIVE_MS = 6000;

export class AutoQuality {
    // `apply(key, value)` writes a lever AND moves its control, so the panel keeps
    // showing the truth. `maxDpr` is the display's device pixel ratio.
    constructor({ apply, maxDpr = 1, targetFps = 60 }) {
        this.apply = apply;
        this.maxDpr = maxDpr;
        this.targetFps = targetFps;
        this.enabled = false;
        this.tier = START_TIER;
        this.reason = "off";
        this._nextEval = 0;
        this._settleUntil = 0;
        this._streak = 0;
        // When the current rung was reached, for deciding a penalty has been served.
        this._stableSince = 0;
        // tier index -> { failures, retryAt } for tiers this machine has missed.
        this._blocked = new Map();
    }

    get maxTier() {
        return TIERS.length - 1;
    }

    setEnabled(on) {
        this.enabled = on;
        if (!on) {
            this.reason = "off";
            return;
        }
        this.tier = START_TIER;
        this._streak = 0;
        this._blocked.clear();
        this._nextEval = 0;
        this._settleUntil = 0;
        this._stableSince = 0;
        this.reason = "probing";
        this._applyTier();
    }

    setTargetFps(fps) {
        this.targetFps = fps;
        // A new target invalidates what we learned about which tiers are reachable.
        this._blocked.clear();
        this._streak = 0;
    }

    // Values the ladder is currently asking for, so the caller can show them.
    current() {
        const tier = TIERS[this.tier];
        return { ...tier, pixelRatio: this._dprFor(tier) };
    }

    state() {
        return {
            enabled: this.enabled,
            tier: this.tier,
            maxTier: this.maxTier,
            targetFps: this.targetFps,
            reason: this.reason,
        };
    }

    // Snapped to the render-scale control's 0.05 grid so the panel shows exactly
    // what was applied rather than the nearest slider notch.
    _dprFor(tier) {
        return Math.max(0.4, Math.round(this.maxDpr * tier.dpr * 20) / 20);
    }

    _applyTier() {
        const tier = TIERS[this.tier];
        this.apply("pixelRatio", this._dprFor(tier));
        this.apply("splatBudget", tier.splatBudget);
        this.apply("foveationStrength", tier.foveationStrength);
        this.apply("antiAlias", tier.antiAlias);
        this.apply("lodUnderfillLimit", tier.lodUnderfillLimit);
    }

    _block(tier, now) {
        const record = this._blocked.get(tier) ?? { failures: 0, retryAt: 0 };
        record.failures += 1;
        record.retryAt =
            now + Math.min(BACKOFF_BASE_MS * 2 ** (record.failures - 1), BACKOFF_MAX_MS);
        this._blocked.set(tier, record);
    }

    _blockedFor(tier, now) {
        const record = this._blocked.get(tier);
        return record && now < record.retryAt ? record.retryAt - now : 0;
    }

    // Sitting comfortably on a rung is evidence the rung above may be reachable
    // after all — the last failure there may have been a hitch that had nothing to
    // do with quality. Serve down one failure per FORGIVE_MS so a rung lost that way
    // returns, while a rung that keeps genuinely failing stays at the ceiling.
    _forgive(now) {
        if (!this._stableSince) this._stableSince = now;
        if (now - this._stableSince < FORGIVE_MS) return;
        this._stableSince = now;
        const record = this._blocked.get(this.tier + 1);
        if (!record) return;
        record.failures -= 1;
        if (record.failures <= 0) this._blocked.delete(this.tier + 1);
        else record.retryAt = Math.min(record.retryAt, now);
    }

    // Called every animation frame; decides at most once per EVAL_MS.
    // `frameMs` is the smoothed frame delta, `pendingLoads` the engine's own count
    // of chunk fetches still out.
    step(frameMs, pendingLoads, now) {
        if (!this.enabled) return;
        if (now < this._settleUntil || now < this._nextEval) return;
        this._nextEval = now + EVAL_MS;

        const targetMs = 1000 / this.targetFps;

        if (frameMs > targetMs * OVER_FACTOR) {
            this._streak = 0;
            if (this.tier === 0) {
                this.reason = "floor — target unreachable";
                return;
            }
            // Only blame the rung when nothing was streaming. A chunk landing mid-frame
            // spikes the frame time on its own, and penalising the rung for that is how
            // a machine ends up parked below what it can actually hold.
            const trustworthy = pendingLoads === 0;
            if (trustworthy) this._block(this.tier, now);
            this.tier -= 1;
            this._settleUntil = now + SETTLE_MS;
            this._stableSince = now;
            this.reason = trustworthy
                ? `missed ${this.targetFps} fps — dropped to ${this.tier}`
                : `hitch while streaming — dropped to ${this.tier}, rung not penalised`;
            this._applyTier();
            return;
        }

        if (frameMs > targetMs * OK_FACTOR) {
            // Decay rather than reset: one borderline sample shouldn't undo the whole
            // run-up, or a scene that hovers near the target never climbs at all.
            this._streak = Math.max(0, this._streak - 1);
            this.reason = "holding";
            return;
        }

        // Comfortable. Climb, if there is anywhere to go and nothing in the way.
        if (this.tier === this.maxTier) {
            this.reason = "max quality";
            return;
        }
        this._forgive(now);
        if (pendingLoads > 0) {
            this.reason = `waiting on ${pendingLoads} chunk${pendingLoads === 1 ? "" : "s"}`;
            return;
        }
        const cooling = this._blockedFor(this.tier + 1, now);
        if (cooling > 0) {
            this.reason = `rung ${this.tier + 1} cooling (${(cooling / 1000).toFixed(1)}s)`;
            return;
        }
        this._streak += 1;
        if (this._streak < CLIMB_STREAK) {
            this.reason = `headroom (${this._streak}/${CLIMB_STREAK})`;
            return;
        }
        this._streak = 0;
        this.tier = clamp(this.tier + 1, 0, this.maxTier);
        this._settleUntil = now + SETTLE_MS;
        this._stableSince = now;
        this.reason = `headroom — raised to ${this.tier}`;
        this._applyTier();
    }
}
