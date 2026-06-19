// THE single client-side status renderer. Every panel (board, overlay,
// prompt-lab sims, compare) turns a cell/branch summary into its dot + message
// through here, so the status shown never drifts between views.
//
// The server already resolves `status` into one of idle | running | paused |
// error | done (folding a cell parked at a step gate into `paused`), so the dot
// is just that value. This module only adds the matching human label and the
// relevant step:
//
//   paused  → "awaiting <next step>"   (the gated call that runs on the next step)
//   running → "running <current step>" (the call in flight)
//   done    → "done"
//   error   → "error"
//   idle    → "not started"
//
// Cells and branches share the summary shape, so one function serves both.

// Format a step descriptor. Gate calls (`pending`/`current`) carry
// `template`/`step`; the `last_step` marker carries `phase`. The node is
// appended when known. Returns null when there's nothing to name.
export function stepName(call) {
  if (!call) return null;
  const name = call.template ?? call.step ?? call.phase ?? null;
  if (!name) return null;
  return call.node ? `${name} @ ${call.node}` : name;
}

// { state, dot, label } for a cell OR branch summary. `summary` may be null
// (e.g. a just-forked branch the poll hasn't reported yet) → treated as idle.
export function statusView(summary) {
  const state = summary?.status ?? "idle";
  let label;
  switch (state) {
    case "running":
      // The call in flight; for an ungated (full-run) cell there's no live
      // gate, so fall back to the last pipeline-phase marker.
      label = `running ${stepName(summary.current) ?? stepName(summary.last_step) ?? "…"}`;
      break;
    case "paused": {
      // The step that runs on the next advance. Known only while parked at a
      // live gate; a hard pause / post-restart cell has none → plain "paused".
      const next = stepName(summary.pending);
      label = next ? `awaiting ${next}` : "paused";
      break;
    }
    case "done": label = "done"; break;
    case "error": label = "error"; break;
    case "idle": label = "not started"; break;
    default: label = state;
  }
  return { state, dot: state, label };
}
