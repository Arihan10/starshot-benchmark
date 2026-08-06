export class PairGate {
	private readonly waiting = new Map<string, () => void>();
	private timer: number | null = null;
	private released = false;

	constructor(
		private readonly expect: number,
		private readonly timeoutMs = 4000,
	) {}

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
		if (this.timer === null) {
			this.timer = window.setTimeout(() => this.release(), this.timeoutMs);
		}
	}

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
		const commits = [...this.waiting.values()];
		this.waiting.clear();
		for (const commit of commits) commit();
	}

	cancel() {
		this.released = true;
		this.waiting.clear();
		if (this.timer !== null) {
			window.clearTimeout(this.timer);
			this.timer = null;
		}
	}
}
