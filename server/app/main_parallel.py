"""Server entry that runs the SPLIT-PHASE (parallel-interior) divider.

Identical to `app.main` — the full Trellis 2 + Nano Banana pipeline, the same
generation, resume, stepped mode and dashboard contract — except Phase 1 frames
the WHOLE tree first and then builds every zone's interior concurrently
(`pipeline.divider_parallel.run`), instead of building each zone's interior as
the depth-first walk reaches it. A run launched here is directly comparable to
an `app.main` run of the same prompt + model.

The swap is a single module-attribute rebind: `routes._run` resolves
`divider.run(...)` by attribute at request time, so rebinding it before the app
is created routes every run through the split walk without touching routes.

Pair it with `STARSHOT_NEXT_OBJECT_CAP=2` in `server/.env` to cap the anchor
completion loop at two rounds per zone.

Launched by `scripts/run_parallel.py`.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from app.utils.rlimit import raise_nofile_limit  # noqa: E402

raise_nofile_limit()

from app.pipeline import divider, divider_parallel  # noqa: E402

divider.run = divider_parallel.run

from app.api.routes import create_app  # noqa: E402

app = create_app()
