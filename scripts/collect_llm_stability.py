"""Measure real OpenAI-compatible LLM availability and latency.

This is intentionally separate from the Agent quality evaluator: it measures
the provider transport itself, using the application's configured LLM client.
It never writes prompts, replies, API keys, or response bodies to the artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.llm.client import llm_chat


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, max(0, int(len(values) * ratio + 0.999999) - 1))]


def collect(runs: int, max_tokens: int) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for index in range(max(1, runs)):
        started = time.perf_counter()
        try:
            reply = llm_chat(
                "You are a connectivity probe. Reply with exactly OK.",
                "Reply with OK.",
                temperature=0.0,
                max_tokens=max_tokens,
            )
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            samples.append({"run": index + 1, "ok": bool(reply.strip()), "latency_ms": elapsed, "response_chars": len(reply)})
        except Exception as exc:  # defensive: client normally degrades to empty
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            samples.append({"run": index + 1, "ok": False, "latency_ms": elapsed, "error_type": type(exc).__name__})
    latencies = [float(item["latency_ms"]) for item in samples]
    successful = [float(item["latency_ms"]) for item in samples if item["ok"]]
    return {
        "runs": len(samples),
        "successes": sum(1 for item in samples if item["ok"]),
        "success_rate": round(sum(1 for item in samples if item["ok"]) / len(samples), 4),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "p95": round(percentile(latencies, 0.95) or 0, 3) if latencies else None,
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "successful_p95": round(percentile(successful, 0.95) or 0, 3) if successful else None,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    # Reasoning-capable providers may spend completion budget on hidden
    # reasoning before emitting visible content. 32 tokens can therefore make
    # a healthy response look empty; keep the probe budget aligned with the
    # /chat/test-llm endpoint.
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "quality" / "llm-api-stability-latest.json")
    args = parser.parse_args()
    settings = get_settings()
    result = {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "base_url_host": (settings.deepseek_base_url or "").split("/")[2] if "://" in settings.deepseek_base_url else "",
        "timeout_seconds": settings.llm_timeout_seconds,
        "resilience_attempts": settings.llm_resilience_attempts,
        "measurement": collect(args.runs, args.max_tokens),
        "limitations": [
            "The current client exposes completion latency, not first-token latency.",
            "An empty client response is recorded as unavailable; the client intentionally redacts provider error details.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
