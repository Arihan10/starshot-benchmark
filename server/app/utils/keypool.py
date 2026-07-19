"""Rotating API-key pools for rate-limited OpenAI-compatible providers.

One pool per provider (keyed by its `api_key_env` name), shared process-wide
across every slot task. `current()` hands out the active key plus a
generation stamp; a caller that gets HTTP 429 reports that stamp back to
`rotate()`, which advances to the next key only if no other task already did.
First reporter wins, stale reports no-op — a burst of concurrent 429s on the
same key rolls the pool exactly once, and generations increase monotonically
so a wrapped-around pool can't be advanced by a stale report either.

Methods contain no awaits, so under a single event loop they are atomic
across tasks. Keys load once, lazily, from the environment:

  * `{api_key_env}_ARRAY` — JSON array or comma/newline-separated list; else
  * `{api_key_env}` — the plain single-key var (a one-key pool never rotates).
"""

from __future__ import annotations

import json
import os

from app.utils import logging as rlog


def _mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) >= 6 else "..."


class KeyPool:
    def __init__(self, name: str, keys: list[str]) -> None:
        self.name = name
        self._keys = keys
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def size(self) -> int:
        return len(self._keys)

    def current(self) -> tuple[int, str]:
        """The active `(generation, key)`. Pass the generation back to
        `rotate()` when the request it authorized comes back 429."""
        return self._generation, self._keys[self._generation % len(self._keys)]

    def rotate(self, generation: int) -> bool:
        """Advance to the next key iff the pool is still on `generation` —
        the compare-and-swap that collapses concurrent 429 reports for one
        key into a single advance. Returns True when this call rotated."""
        if generation != self._generation or len(self._keys) == 1:
            return False
        self._generation += 1
        idx = self._generation % len(self._keys)
        rlog.console_note(
            f"[keypool] {self.name}: 429 on {_mask(self._keys[generation % len(self._keys)])}"
            f" -> rotating to key[{idx}] {_mask(self._keys[idx])} (gen {self._generation})"
        )
        return True


def _parse_array(var: str, raw: str) -> list[str]:
    if raw.strip().startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            # A malformed pool must fail loudly, not silently fall back to
            # the single key. RuntimeError so call_llm_once's retry budgets
            # (which eat ValueError) never burn resamples on a config error.
            raise RuntimeError(f"{var} is not a valid JSON array: {e}") from e
        items = data if isinstance(data, list) else []
    else:
        items = raw.replace("\n", ",").split(",")
    keys: list[str] = []
    for item in items:
        key = str(item).strip().strip("\"'")
        if key and key not in keys:
            keys.append(key)
    return keys


_POOLS: dict[str, KeyPool] = {}


def get_pool(api_key_env: str) -> KeyPool:
    """The shared pool for `api_key_env`, created from the environment on
    first use: `{api_key_env}_ARRAY` when set, else the single-key var."""
    pool = _POOLS.get(api_key_env)
    if pool is not None:
        return pool
    array_var = f"{api_key_env}_ARRAY"
    raw = os.environ.get(array_var, "")
    keys = _parse_array(array_var, raw) if raw.strip() else []
    if not keys:
        single = os.environ.get(api_key_env, "").strip()
        keys = [single] if single else []
    if not keys:
        raise RuntimeError(
            f"{array_var} / {api_key_env} is not set — required for a rotate-enabled model"
        )
    pool = KeyPool(api_key_env, keys)
    _POOLS[api_key_env] = pool
    rlog.console_note(
        f"[keypool] {api_key_env}: pool of {len(keys)} key(s):"
        f" {', '.join(_mask(k) for k in keys)}"
    )
    return pool
