// A tiny cross-module bus so the global ops bar (opsbar.js) can show + sync the
// Modal attention-compute queue that the /tf attention modules own, WITHOUT the
// ops bar importing /tf-only state. /tf registers a provider on boot; the ops bar
// reads it each poll. On pages with no attention compute (the dashboard) no
// provider is set, so the ops bar simply omits the attention segment + its sync
// button.
//
// provider shape:
//   {
//     snapshot(): { running, queued, computed, total, main, abl, cell } | null,
//     sync(): Promise<void>,   // re-pull the Modal queue for the current cell
//   }

let _provider = null;

export function setAttnProvider(p) { _provider = p; }
export function getAttnProvider() { return _provider; }
