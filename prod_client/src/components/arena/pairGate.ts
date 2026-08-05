/**
 * Holds the two panels' scene swaps until BOTH are ready, then runs them back to
 * back.
 *
 * WHY A PAIR IS A PAIR. The two sides load independently and finish whenever their
 * bytes and their decode say so — measured, ~550 ms apart on the checked-in round,
 * because one cell carries a 12 MB splat and the other has none. Left to
 * themselves they swap one at a time, so for half a second the row shows one build
 * from the new round beside one from the old, under the new round's prompt. That
 * reads as the comparison changing under you, which is the one thing a comparison
 * must not do.
 *
 * It is NOT a loading state. Both engines are still rendering their previous scene
 * the whole time they are held here — the wait costs the faster side a few hundred
 * milliseconds of showing a scene it was showing anyway.
 *
 * THE TIMEOUT IS THE POINT, not a safety net. A side whose assets 404, or whose
 * prepare throws, may never arrive; without a deadline the other side would hold
 * its previous scene for ever and the round would silently never turn over. After
 * `timeoutMs` from the FIRST arrival, whoever is here goes, and a straggler
 * commits on its own when it lands.
 */
export class PairGate {
	private readonly waiting = new Map<string, () => void>();
	private timer: number | null = null;
	private released = false;

	constructor(
		private readonly expect: number,
		private readonly timeoutMs = 4000,
	) {}

	/**
	 * Register a side's swap. Runs it immediately if this completes the pair (or if
	 * the gate has already given up waiting), otherwise holds it.
	 */
	arrive(key: string, commit: () => void) {
		if (this.released) {
			commit();
			return;
		}
		this.waiting.set(key, commit);
		if (this.waiting.size >= this.expect) {
			this.release();
			return;
		}
		// Deadline runs from the first arrival: it is a bound on how long a side
		// that IS ready waits for one that may never be.
		if (this.timer === null) {
			this.timer = window.setTimeout(() => this.release(), this.timeoutMs);
		}
	}

	/** Drop a side that will never arrive (its load failed, or was superseded). */
	abandon(key: string) {
		if (this.released || !this.waiting.has(key)) return;
		this.waiting.delete(key);
	}

	private release() {
		if (this.released) return;
		this.released = true;
		if (this.timer !== null) {
			window.clearTimeout(this.timer);
			this.timer = null;
		}
		// Synchronously, in one task, so both panels are swapped inside a single
		// frame. Handing them to separate rAF callbacks or microtasks would put a
		// paint between them and give back the stagger this exists to remove.
		const commits = [...this.waiting.values()];
		this.waiting.clear();
		for (const commit of commits) commit();
	}

	/** Let go of everything without running it — the round moved on. */
	cancel() {
		this.released = true;
		this.waiting.clear();
		if (this.timer !== null) {
			window.clearTimeout(this.timer);
			this.timer = null;
		}
	}
}
