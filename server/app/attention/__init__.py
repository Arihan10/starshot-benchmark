"""Attention-analysis pipeline.

Teacher-forced attention instrumentation for the text-to-3D reasoning steps:
recompute, for each generated token, how much it attended to each scene
entity (region / object / attribute) — resolved back onto semantic entities
rather than raw token ids.

Layout (each module documents its role + what is real vs mock/scaffold):

  schema.py    wire contracts: the GenerationTrace streamed to a worker, and
               the sparse AnalysisResult streamed back.
  semantic.py  tokenizer abstraction (mock + optional HF offset-mapping) and
               the token<->char<->entity/component remapping built from the
               tf-export maps.
  stats.py     the attention statistics (scale, renormalized scene map,
               entropy ratio) + span/hierarchical aggregation. Pure functions.
  worker.py    the compute worker: reconstruct -> tokenize -> forward pass
               (mock now; real HF/GPU behind the same interface) -> stats ->
               semantic remap -> sparse records. Runs locally today.
  store.py     sparse per-token result storage under the cell dir.
  modal_app.py Modal GPU deployment scaffolding + the KV-cache / FlashAttention
               / parallelism design (guarded import; not deployed from here).
"""
