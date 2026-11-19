"""Shared deterministic operation cues used by compilation and preference rules."""

from __future__ import annotations

import re


def is_explicit_compare_request(text: str) -> bool:
    """Recognize comparison constructions without treating 比较 + predicate as compare."""
    folded = str(text or "").casefold()
    if any(marker in folded for marker in ("对比", "同比", "环比")):
        return True
    if re.search(r"(?<![a-z0-9_])compare(?![a-z0-9_])|comparison", folded):
        return True
    return bool(
        re.search(r"(?:进行|做|做个|做一次)\s*比较", folded)
        or re.search(r"比较\s*(?:一下|两者|它们|两期|两个|gmv|订单|商品|类目)", folded)
        or re.search(r".+(?:和|与|跟).+比较", folded)
    )
