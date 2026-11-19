"""Safety boundary for optional test artifacts: never write into the repo."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def isolated_output_path(env_name: str) -> Path | None:
    """Resolve an optional output override and require it to live in OS temp."""
    raw = os.getenv(env_name)
    if not raw:
        return None
    target = Path(raw).expanduser().resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        target.relative_to(temp_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{env_name} must point under the system temporary directory; "
            "test outputs may not overwrite repository or historical artifacts."
        ) from exc
    return target
