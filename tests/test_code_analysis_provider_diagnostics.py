"""Safe diagnostics for malformed provider artifact envelopes."""

from __future__ import annotations

import hashlib

import pytest

from app.code_analysis.providers import ArtifactEnvelopeError, _extract_json_payload


def test_invalid_provider_json_exposes_only_bounded_shape_metadata():
    response = "Here is the result:\n```json\n{\"analysis.py\":\"def analyze("

    with pytest.raises(ArtifactEnvelopeError) as raised:
        _extract_json_payload(response)

    diagnostics = raised.value.diagnostics
    assert diagnostics["parser_error"] == "invalid_json"
    assert diagnostics["response_shape"] == "markdown_fenced"
    assert diagnostics["likely_truncated"] is True
    assert diagnostics["response_bytes"] == len(response.encode("utf-8"))
    assert diagnostics["response_sha256"] == hashlib.sha256(
        response.encode("utf-8")
    ).hexdigest()
    assert "def analyze" not in str(diagnostics)


def test_missing_analysis_key_is_distinguished_from_invalid_json():
    with pytest.raises(ArtifactEnvelopeError) as raised:
        _extract_json_payload('{"explanation.json": {}, "requirements.json": []}')

    assert raised.value.diagnostics["parser_error"] == "missing_analysis_py"
    assert raised.value.diagnostics["has_analysis_py_key"] is False
