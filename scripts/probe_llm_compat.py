"""Compare the provider's plain and JSON-mode Chat Completions behavior."""

import os

from openai import OpenAI


client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL"),
    timeout=30,
)
model = os.environ.get("LLM_MODEL", "hy3")
for name, extra in (("plain", {}), ("json", {"response_format": {"type": "json_object"}})):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Return exactly JSON: {"ok":true}'}],
            max_tokens=64,
            **extra,
        )
        content = getattr(response.choices[0].message, "content", None) if response.choices else None
        print(name, "OK", repr(content)[:200])
    except Exception as exc:  # noqa: BLE001 - diagnostic probe
        print(
            name,
            "FAIL",
            type(exc).__name__,
            getattr(exc, "status_code", None),
            getattr(exc, "request_id", None),
            repr(getattr(exc, "body", None))[:500],
        )
