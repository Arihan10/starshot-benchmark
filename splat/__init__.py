"""SceneBench mesh → Gaussian-splat delivery pipeline (see `splat/overview.md`).

Downstream of the parametric mesh pipeline: converts a generated cell
(`objects-generated/` + `events.jsonl`) into a compressed, free-fly Gaussian
splat for the public voting site. Built stage by stage; `stage1` (the scene
assembler) is the first.
"""
