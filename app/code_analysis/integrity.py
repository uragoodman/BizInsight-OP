"""Canonical fingerprints shared by the Phase 11 API and worker."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def plan_fingerprint(value: Any) -> str:
    """Hash the persisted plan using the original Phase 11 canonical form.

    Keep the default JSON separators for compatibility with plans created
    before this helper was introduced.
    """
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compact_fingerprint(value: Any) -> str:
    """Hash other structured evidence in a whitespace-independent form."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
