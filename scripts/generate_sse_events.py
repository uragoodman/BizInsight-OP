#!/usr/bin/env python3
"""从后端 Pydantic SSE 事件模型生成 TypeScript 类型定义。

在 npm run build 前执行，确保前端类型与后端事件契约一致。
生成文件：web/src/generated/sse-events.ts
"""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    check_only = "--check" in sys.argv
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from app.schemas.events import (
        SseCompletedEvent,
        SseFailedEvent,
        SseHeartbeatEvent,
        SseInvestigationProgressEvent,
        SseNodeCompletedEvent,
        SseNodeStartedEvent,
        SsePlanEvent,
        SseStartedEvent,
        SseSuspendedEvent,
    )

    models = [
        SseStartedEvent,
        SseNodeStartedEvent,
        SseNodeCompletedEvent,
        SsePlanEvent,
        SseInvestigationProgressEvent,
        SseHeartbeatEvent,
        SseCompletedEvent,
        SseSuspendedEvent,
        SseFailedEvent,
    ]

    def _to_ts(schema: dict) -> str:
        t = schema.get("type", "")
        if t == "string":
            if "const" in schema:
                return f"'{schema['const']}'"
            ev = schema.get("enum")
            return " | ".join(f"'{v}'" for v in ev) if ev else "string"
        if t == "boolean":
            return "boolean"
        if t in ("integer", "number"):
            return "number"
        if t == "array":
            return "string[]" if schema.get("items", {}).get("type") == "string" else "unknown[]"
        if t == "object":
            return "Record<string, unknown>"
        for key in ("anyOf", "oneOf"):
            subs = schema.get(key, [])
            if subs:
                return " | ".join("null" if s.get("type") == "null" else _to_ts(s) for s in subs)
        return "unknown"

    lines = [
        "// Auto-generated from app/schemas/events.py — do not edit by hand.",
        "// Run: npm run gen:events",
        "",
    ]
    for m in models:
        sc = m.model_json_schema()
        req = set(sc.get("required", []))
        # Fields with defaults always have values at runtime
        for pn, ps in sc.get("properties", {}).items():
            if "default" in ps:
                req.add(pn)
        lines.append(f"export interface {m.__name__} {{")
        for pn, ps in sc.get("properties", {}).items():
            o = "" if pn in req else "?"
            lines.append(f"  {pn}{o}: {_to_ts(ps)};")
        lines.append("}")
        lines.append("")

    lines.append("export type SseEvent =")
    for i, m in enumerate(models):
        lines.append(f"  | {m.__name__}" + (";" if i == len(models) - 1 else ""))
    lines.append("")
    for m in models:
        lit = m.model_json_schema()["properties"]["event"].get("default")
        if lit:
            lines.append(
                f"export const is{m.__name__} = (e: SseEvent): e is {m.__name__} => e.event === \"{lit}\";"
            )
    lines.append("")
    lines.append(
        "export type ChatResult = SseCompletedEvent | (SseSuspendedEvent & { "
        "error: boolean; error_detail: string | null; presentation: unknown | null; "
        "complex_plan: Record<string, unknown> | null; durable_turn_id: string | null; "
        "execution_id: string | null; commit_version: number | null; "
        "clarification: Record<string, unknown> | null });"
    )
    lines.append("")

    out_path = repo / "web" / "src" / "generated" / "sse-events.ts"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if check_only:
        if out_path.exists() and out_path.read_text(encoding="utf-8") == content:
            print(f"OK {out_path.relative_to(repo)} is up to date")
            return
        else:
            print(f"ERROR: {out_path.relative_to(repo)} is stale. Run: npm run gen:events", file=sys.stderr)
            sys.exit(1)
    out_path.write_text(content, encoding="utf-8")
    print(f"→ Generated {out_path.relative_to(repo)} ({len(models)} models)")


if __name__ == "__main__":
    main()
