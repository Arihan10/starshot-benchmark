"""Server entry that runs the experimental BREADTH-FIRST divider.

Identical to `app.main` (the full Trellis 2 + Nano Banana pipeline) except
Phase 1 walks the zone tree breadth-first (`pipeline.divider_bfs.run`) instead
of depth-first (`pipeline.divider.run`). Everything downstream — generation,
framing, negative space, resume, stepped mode, the dashboard contract — is
unchanged, so a run launched here is directly comparable to an `app.main` run
of the same prompt + model.

The swap is a single module-attribute rebind: `routes._run` resolves
`divider.run(...)` by attribute at request time, so rebinding it before the
app is created routes every run through the BFS walk without touching routes.

Launched by `scripts/run_bfs.py`.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from app.utils.rlimit import raise_nofile_limit  # noqa: E402

raise_nofile_limit()

from app.pipeline import divider, divider_bfs  # noqa: E402

divider.run = divider_bfs.run

from app.api.routes import create_app  # noqa: E402

app = create_app()
