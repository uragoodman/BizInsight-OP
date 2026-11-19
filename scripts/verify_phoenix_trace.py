"""Emit a harmless synthetic OpenTelemetry span for a Phoenix connectivity check."""

from __future__ import annotations

import sys

from opentelemetry import trace

from app.core.config import get_settings
from app.observability.tracing import span


def main() -> int:
    endpoint = get_settings().otel_exporter_otlp_endpoint.strip()
    if not endpoint:
        print(
            "OTEL_EXPORTER_OTLP_ENDPOINT is empty. Set it to "
            "http://127.0.0.1:4318/v1/traces and retry.",
            file=sys.stderr,
        )
        return 2

    with span("phoenix.smoke_test", {"phoenix.smoke_test": True}) as active_span:
        trace_id = f"{active_span.get_span_context().trace_id:032x}"

    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if callable(force_flush) and not force_flush(timeout_millis=10_000):
        print("OTLP exporter did not flush the smoke trace.", file=sys.stderr)
        return 1

    print(f"PHOENIX_TRACE_SENT trace_id={trace_id}")
    print("Open Phoenix UI and find the phoenix.smoke_test span.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
