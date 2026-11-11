"""API-key authentication.

Keys come from CHURN_API_KEYS (comma separated). An entry is either the key
itself or `sha256:<hex digest of the key>`, so the plaintext need not sit in the
environment. Comparison is constant-time. Keys are never logged; requests are
attributed to a short fingerprint instead.
"""
from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="API key issued to the calling system")


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def key_matches(candidate: str, entry: str) -> bool:
    if entry.startswith("sha256:"):
        digest = hashlib.sha256(candidate.encode()).hexdigest()
        return hmac.compare_digest(digest, entry[7:].lower())
    return hmac.compare_digest(candidate.encode(), entry.encode())


def require_api_key(request: Request, key: str | None = Security(api_key_header)) -> str:
    if not key:
        raise HTTPException(401, "Missing API key. Send it in the X-API-Key header.")
    # check every entry (no early exit) so timing does not reveal which matched
    ok = [key_matches(key, e) for e in request.app.state.settings.api_keys]
    if not any(ok):
        raise HTTPException(401, "Invalid API key.")
    request.state.key_id = fingerprint(key)
    return request.state.key_id
