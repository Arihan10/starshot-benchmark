"""Process resource limits — macOS defaults to a 256-fd soft cap, which
benchmark runs can hit when many cells log events, stream SSE, and serve
artifacts concurrently. Bump the soft limit toward the hard ceiling at
startup (no-op when already sufficient or on platforms without `resource`)."""

from __future__ import annotations


def raise_nofile_limit(*, minimum: int = 4096) -> None:
    try:
        import resource
    except ImportError:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return
    # macOS often reports hard=unlimited as a large int; clamp sanely.
    if hard != resource.RLIM_INFINITY and hard > 0:
        target = min(max(minimum, soft), hard)
    else:
        target = max(minimum, soft)
    if target <= soft:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass
