"""Cell identity + freshness stamp — shared by the API server and Modal backend.

A CELL is (run, slot, full-model-name, prompt_version). Its hash namespaces ALL
attention queue state, so a prompt-version bump (or a different run/slot/model)
can never collide with — or serve a stale result from — an older version. The
server recovers the whole queue after a reload/restart with one pull by this hash.

A STAMP is an HMAC-SHA256 (a keyed / *symmetric* hash) over the cell identity +
the step's content key + prompt version. It is the up-to-date proof: the server
only accepts a computed result if its stamp re-verifies against what the server
currently believes (current prompt version + current step content). A result for
an edited step or a superseded prompt version fails verification and is rejected —
so a displayed attention map is always cryptographically bound to its exact inputs.

NOTE: this is authentication/integrity (HMAC), not secrecy — we don't need to hide
the payload, only to verify it's current. HMAC with a shared symmetric secret is
the correct tool for that. Override the secret on BOTH sides for real deployments
via `ATTENTION_STAMP_SECRET`; the default keeps local dev zero-config.
"""

from __future__ import annotations

import hashlib
import hmac
import os

_SCHEME = "v1"
_SEP = "\x1f"  # unit separator — never appears in ids/keys, so components can't collide
_DEFAULT_SECRET = "starshot-attention-stamp-v1"


def _secret() -> bytes:
    return os.environ.get("ATTENTION_STAMP_SECRET", _DEFAULT_SECRET).encode("utf-8")


def cell_hash(run: str, slot: str, model_full_name: str, prompt_version: str | None) -> str:
    """Stable id for a (run, slot, model, prompt_version) cell (hex, 20 chars)."""
    raw = _SEP.join((_SCHEME, str(run), str(slot), str(model_full_name), str(prompt_version or "")))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def stamp(*, cell_hash: str, event_index: int, input_key: str | None,
          prompt_version: str | None) -> str:
    """Keyed HMAC binding a result to its exact cell + prompt version + step content."""
    msg = _SEP.join((
        _SCHEME, str(cell_hash), str(int(event_index)),
        str(input_key or ""), str(prompt_version or ""),
    ))
    return hmac.new(_secret(), msg.encode("utf-8"), hashlib.sha256).hexdigest()


def verify(candidate: str | None, *, cell_hash: str, event_index: int,
           input_key: str | None, prompt_version: str | None) -> bool:
    """Constant-time check that `candidate` is the valid stamp for these inputs."""
    if not candidate:
        return False
    expected = stamp(cell_hash=cell_hash, event_index=event_index,
                     input_key=input_key, prompt_version=prompt_version)
    return hmac.compare_digest(candidate, expected)


def req_token(*, input_key: str | None, max_heads: int, top_k: int,
              max_query_tokens: int, analysis_version: int, to_place_version: int,
              agg_version: int = 0, force_nonce: str | None = None) -> str:
    """Opaque identity of ONE compute request for a step: its content key + the
    compute params + the analysis/to-place/aggregation code versions (+ an optional
    force nonce). This is the DEDUP KEY the GPU worker keys on — the SAME token is
    never computed twice, while ANY change that should recompute (edited content →
    new `input_key`, a raised head budget, a new analysis/to-place/agg version, or
    an explicit force → fresh `force_nonce`) yields a NEW token that recomputes.

    The server owns it so ALL freshness logic stays server-side; the queue and the
    worker treat it as an opaque string. Plain SHA-256 (not HMAC): it's a cache key,
    not an integrity proof — the HMAC `stamp` is what authenticates a result."""
    raw = _SEP.join((
        _SCHEME, str(input_key or ""), str(int(max_heads)), str(int(top_k)),
        str(int(max_query_tokens)), str(int(analysis_version)),
        str(int(to_place_version)), str(int(agg_version)), str(force_nonce or ""),
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
