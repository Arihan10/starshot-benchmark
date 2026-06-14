"""Upload published scene assets to the Cloudflare R2 bucket.

R2 speaks the S3 API, so we talk to it with boto3 against the account's R2
endpoint. Credentials come from the environment:

  CLOUDFLARE_ACCOUNT_ID   — picks the `<account>.r2.cloudflarestorage.com` host
  R2_ACCESS_KEY_ID        — an R2 API token's access key id
  R2_SECRET_ACCESS_KEY    — its secret
  R2_BUCKET               — target bucket (defaults to benchmark-assets-prod)

Uploads are plain PUTs at deterministic keys (see services/publish.py), so
re-publishing a (run, slot, model, version) overwrites the objects in place.
boto3 is imported lazily so the server still starts without it / without creds;
the import/credential error only surfaces when a publish is actually attempted.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

_DEFAULT_BUCKET = "benchmark-assets-prod"

_client: Any = None
_client_lock = threading.Lock()


def _bucket() -> str:
    return os.environ.get("R2_BUCKET", _DEFAULT_BUCKET)


def _get_client() -> Any:
    """The shared S3 client for R2, built once. boto3 clients are safe to share
    across threads, which is what the gathered to_thread uploads below do."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
                key_id = os.environ.get("R2_ACCESS_KEY_ID")
                secret = os.environ.get("R2_SECRET_ACCESS_KEY")
                if not (account and key_id and secret):
                    raise RuntimeError(
                        "R2 upload needs CLOUDFLARE_ACCOUNT_ID, R2_ACCESS_KEY_ID and "
                        "R2_SECRET_ACCESS_KEY in the environment"
                    )
                import boto3  # lazy: optional dependency, only needed to publish
                from botocore.config import Config

                _client = boto3.client(
                    "s3",
                    endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
                    aws_access_key_id=key_id,
                    aws_secret_access_key=secret,
                    region_name="auto",
                    config=Config(
                        signature_version="s3v4",
                        retries={"max_attempts": 3, "mode": "standard"},
                    ),
                )
    return _client


def _put_sync(key: str, path: Path, content_type: str) -> None:
    _get_client().put_object(
        Bucket=_bucket(), Key=key, Body=path.read_bytes(), ContentType=content_type
    )


async def put_file(key: str, path: Path, content_type: str) -> None:
    """Upload one local file to `key`, overwriting any existing object. Runs the
    blocking boto3 call off the event loop so callers can gather many at once."""
    await asyncio.to_thread(_put_sync, key, path, content_type)
