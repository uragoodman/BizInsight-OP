"""Candidate-code providers. Providers generate; they never execute code."""
from __future__ import annotations

import json
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

import httpx

from app.code_analysis.policy import ALLOWED_IMPORT_ROOTS, ALLOWED_REQUIREMENTS
from app.core.config import get_settings


@dataclass(frozen=True)
class CandidateArtifact:
    code: str
    explanation: dict
    requirements: list[str]
    provider: str
    model: str
    provider_version: str
    provider_request_id: str = ""


class ArtifactEnvelopeError(ValueError):
    """A bounded, non-sensitive diagnostic for an invalid provider envelope."""

    def __init__(self, message: str, *, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class SecurityReview:
    """A model review is advisory evidence; the worker remains the authority."""

    decision: str
    risk_level: str
    findings: list[str]
    summary: str
    provider_request_id: str = ""

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "risk_level": self.risk_level,
            "findings": list(self.findings),
            "summary": self.summary,
            "provider_request_id": self.provider_request_id,
        }


@dataclass
class ProviderContext:
    """Minimal server-issued context passed to a code-generation provider."""

    job_id: str
    tenant_id: str
    actor_id: str
    trace_id: str
    plan_hash: str
    data_contract_hash: str
    deadline_monotonic: float | None = None
    # Populated as soon as a provider creates its own correlation id, so a
    # failure before CandidateArtifact is returned can still be audited.
    provider_request_id: str = ""
    security_review_request_id: str = ""


class CodeAgentProvider(Protocol):
    def propose_analysis(
        self,
        plan: dict,
        data_contract: dict,
        *,
        context: ProviderContext | None = None,
        previous_code: str | None = None,
        failure_feedback: dict | None = None,
    ) -> CandidateArtifact: ...

    def review_analysis(
        self,
        plan: dict,
        data_contract: dict,
        candidate: CandidateArtifact,
        *,
        context: ProviderContext | None = None,
    ) -> SecurityReview: ...


class FakeCodeAgentProvider:
    """Deterministic provider for local development and security tests."""
    def propose_analysis(
        self,
        plan: dict,
        data_contract: dict,
        *,
        context: ProviderContext | None = None,
        previous_code: str | None = None,
        failure_feedback: dict | None = None,
    ) -> CandidateArtifact:
        reader = {"csv": "pd.read_csv(input_path)", "xlsx": "pd.read_excel(input_path)", "parquet": "pd.read_parquet(input_path)"}.get(str(data_contract.get("file_type")), "pd.read_csv(input_path)")
        code = f'''import pandas as pd\n\ndef analyze(input_path, output_dir):\n    df = {reader}\n    numeric = df.select_dtypes(include="number")\n    numeric_summary = numeric.describe().fillna(0).to_dict()\n    summary = {{"rows": int(len(df)), "columns": list(df.columns), "numeric_summary": numeric_summary}}\n    return {{"conclusion": "上传文件包含 " + str(len(df)) + " 行、" + str(len(df.columns)) + " 列。本地回退分析仅完成描述性统计，无法支持针对具体问题的因果结论。", "method": "对行数、列数和数值列执行确定性的描述性统计。", "summary": summary, "evidence": [{{"metric": "row_count", "value": int(len(df))}}, {{"metric": "column_count", "value": int(len(df.columns))}}], "charts": [], "limitations": ["本地确定性 Provider 不推断业务因果关系，也不执行模型生成的自定义分析。"], "recommendations": ["配置受控的 OpenCode Provider，以支持针对具体问题的分析。"]}}\n'''
        return CandidateArtifact(
            code=code,
            explanation={"method": "descriptive statistics", "plan": plan.get("question", "")},
            requirements=["pandas"],
            provider="fake",
            model="deterministic",
            provider_version="builtin",
        )

    def review_analysis(
        self,
        plan: dict,
        data_contract: dict,
        candidate: CandidateArtifact,
        *,
        context: ProviderContext | None = None,
    ) -> SecurityReview:
        return SecurityReview(
            decision="allow",
            risk_level="low",
            findings=[],
            summary="Deterministic local provider; static policy remains the hard security gate.",
        )


def _response_diagnostics(
    text: str,
    *,
    parser_error: str,
    json_error: str | None = None,
) -> dict[str, object]:
    """Describe a malformed response without persisting its contents."""
    raw = str(text or "")
    stripped = raw.strip()
    encoded = raw.encode("utf-8", "replace")
    has_fence = "```" in stripped
    has_json_start = bool(re.search(r"\{", stripped))
    has_analysis_key = bool(re.search(r"[\"']analysis\.py[\"']\s*:", raw))
    has_explanation_key = bool(re.search(r"[\"']explanation\.json[\"']\s*:", raw))
    has_requirements_key = bool(re.search(r"[\"']requirements\.json[\"']\s*:", raw))
    if not stripped:
        response_shape = "empty"
    elif has_fence:
        response_shape = "markdown_fenced"
    elif stripped.startswith("{"):
        response_shape = "json_like"
    else:
        response_shape = "plain_text_or_preamble"
    return {
        "parser_error": parser_error,
        "response_shape": response_shape,
        "response_bytes": len(encoded),
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
        "has_json_object_start": has_json_start,
        "has_markdown_fence": has_fence,
        "has_analysis_py_key": has_analysis_key,
        "has_explanation_json_key": has_explanation_key,
        "has_requirements_json_key": has_requirements_key,
        "likely_truncated": bool(stripped) and (
            (has_json_start and not stripped.endswith("}"))
            or (has_fence and not stripped.endswith("```"))
        ),
        **({"json_error": json_error[:160]} if json_error else {}),
    }


def _extract_json_payload(text: str) -> dict:
    """Parse a model response without accepting prose around the JSON object."""
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    json_error: str | None = None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        json_error = exc.msg
        # Some OpenAI-compatible providers add a short preamble despite the
        # strict prompt. Recover only a complete JSON object; the structural
        # checks below remain mandatory and candidate code is still scanned by
        # the worker policy before execution.
        starts = [match.start() for match in re.finditer(r"\{", candidate)]
        if not starts:
            raise ValueError("OpenCode did not return the required JSON artifact envelope") from exc
        decoder = json.JSONDecoder()
        payload = None
        for start in starts:
            try:
                parsed, end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("analysis.py"), str)
                and not candidate[start + end :].strip()
            ):
                payload = parsed
                break
        if payload is None:
            raise ArtifactEnvelopeError(
                "OpenCode did not return the required JSON artifact envelope",
                diagnostics=_response_diagnostics(
                    text,
                    parser_error="invalid_json",
                    json_error=json_error,
                ),
            ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("analysis.py"), str):
        raise ArtifactEnvelopeError(
            "OpenCode response is missing analysis.py",
            diagnostics=_response_diagnostics(
                text,
                parser_error="missing_analysis_py",
                json_error=json_error,
            ),
        )
    return payload


def _extract_json_object(
    text: str,
    *,
    required_keys: set[str] | None = None,
) -> dict:
    """Extract one bounded JSON object from a model response.

    Review models sometimes add a short reasoning preamble or Markdown fences.
    That text is not trusted; only the extracted object is accepted, and the
    caller still performs the schema and decision validation.
    """
    candidate = text.strip()
    objects: list[dict] = []
    decoder = json.JSONDecoder()

    def collect(value: str) -> None:
        try:
            parsed = json.loads(value.strip())
        except json.JSONDecodeError:
            return
        if isinstance(parsed, dict):
            objects.append(parsed)

    collect(candidate)
    for fenced in re.finditer(
        r"```(?:json)?\s*(.*?)\s*```",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        collect(fenced.group(1))

    for match in re.finditer(r"\{", candidate):
        try:
            parsed, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)

    if required_keys:
        for payload in objects:
            if required_keys.issubset(payload):
                return payload
    if objects:
        return objects[0]
    raise ValueError("OpenCode security review was not valid JSON")


def _response_text(payload: dict) -> str:
    parts = payload.get("parts")
    if not isinstance(parts, list):
        return str(payload.get("text") or "")
    return "\n".join(
        str(part.get("text"))
        for part in parts
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
    )


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_explanation(value: object) -> dict:
    parsed = _json_value(value)
    if not isinstance(parsed, dict):
        raise ValueError("OpenCode explanation.json must contain a JSON object")
    return parsed


def _normalize_requirements(value: object) -> list[str]:
    parsed = _json_value(value)
    if isinstance(parsed, dict):
        parsed = parsed.get(
            "dependencies",
            parsed.get("requirements", parsed.get("required")),
        )
    if not isinstance(parsed, list):
        raise ValueError("OpenCode requirements.json must contain a JSON array")
    return [str(item) for item in parsed]


class OpenCodeProvider:
    """HTTP adapter for a headless OpenCode server on an internal network."""

    @staticmethod
    def _request_timeout(context: ProviderContext | None) -> float:
        configured = max(1.0, get_settings().code_analysis_opencode_timeout_seconds)
        if context is None or context.deadline_monotonic is None:
            return configured
        remaining = context.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Code analysis job deadline exceeded before provider call")
        return min(configured, remaining)

    def propose_analysis(
        self,
        plan: dict,
        data_contract: dict,
        *,
        context: ProviderContext | None = None,
        previous_code: str | None = None,
        failure_feedback: dict | None = None,
    ) -> CandidateArtifact:
        settings = get_settings()
        provider_request_id = f"oc_{uuid4().hex}"
        if context is not None:
            context.provider_request_id = provider_request_id
        prompt = {
            "instruction": (
                "Generate candidate Python analysis code. Return exactly one JSON object with keys "
                "'analysis.py', 'explanation.json', and 'requirements.json'; do not use Markdown fences. "
                "The code must only define helpers and analyze(input_path, output_dir), may only use the "
                "registered data-science libraries, and must not use shell, network, dynamic execution, "
                "environment variables, or arbitrary file reads. Do not execute functions at module scope "
                "except the exact optional matplotlib.use('Agg') headless setup; put all other work inside "
                "analyze(input_path, output_dir). If the plan contains conversation_context, use it only "
                "as context for the follow-up question and re-check all claims directly from the input file. "
                "Write all user-facing natural-language output in Simplified Chinese. Keep professional "
                "terms, metric names, column names, file names, model IDs, code, JSON keys, units, URLs, "
                "numbers, and necessary special symbols in their original form. Do not use English sentences "
                "when a clear Chinese expression is available. For every explicit requirement in the question, "
                "compute and expose the corresponding metric in the result; do not substitute an adjacent analysis "
                "such as country rankings for a requested monthly metric or period change."
            ),
            "contract": data_contract,
            "plan": plan,
            "entrypoint": "def analyze(input_path, output_dir) -> dict",
            "request_context": {
                "job_id": context.job_id if context else "",
                "trace_id": context.trace_id if context else "",
                "plan_hash": context.plan_hash if context else "",
                "data_contract_hash": context.data_contract_hash if context else "",
            },
            "static_policy": {
                "allowed_import_roots": sorted(ALLOWED_IMPORT_ROOTS),
                "allowed_requirements": sorted(ALLOWED_REQUIREMENTS),
                "forbidden_imports": [
                    "os",
                    "sys",
                    "pathlib",
                    "csv",
                    "subprocess",
                    "socket",
                    "requests",
                    "httpx",
                    "urllib",
                ],
                "forbidden_builtins": [
                    "open",
                    "eval",
                    "exec",
                    "compile",
                    "__import__",
                    "getattr",
                    "setattr",
                ],
                "forbidden_constructs": [
                    "with",
                    "async with",
                    "lambda",
                    "class",
                    "async def",
                    "top-level function calls other than matplotlib.use('Agg')",
                ],
                "output_path_rule": (
                    "output_dir already exists; create chart paths using "
                    "output_dir + '/chart.png'. Do not create directories."
                ),
            },
            "candidate_result_contract": {
                "conclusion": (
                    "non-empty natural-language conclusion that directly answers the confirmed question; "
                    "this narrative is mandatory and must not be replaced by summary cards; "
                    "state explicitly when the data is insufficient"
                ),
                "method": "non-empty short string or non-empty JSON object describing the performed calculation",
                "summary": "non-empty JSON object containing structured metrics used by optional result cards",
                "evidence": "JSON list of compact objects that support the conclusion; use [] only when no evidence is available",
                "charts": "JSON list of PNG file-path strings created under output_dir",
                "limitations": "JSON list of non-empty strings",
                "recommendations": (
                    "JSON list of cautious, testable next steps; do not claim causality without evidence"
                ),
            },
            "language_policy": {
                "natural_language": "Simplified Chinese",
                "preserve_original": [
                    "professional terms",
                    "metric and column names",
                    "file names",
                    "model IDs",
                    "code",
                    "JSON keys",
                    "units",
                    "URLs",
                    "numbers",
                    "necessary special symbols",
                ],
                "rule": "Use Chinese for all other user-facing text, including conclusion, method, evidence, limitations, recommendations, and reproducibility.",
            },
            "trusted_result_enrichment": {
                "reproducibility": (
                    "reserved for the trusted worker; analysis.py must not return this key. "
                    "The worker adds input, plan, candidate-code, image, provider, and model fingerprints."
                )
            },
            "response_schema": {
                "analysis.py": "Python source string",
                "explanation.json": {"method": "string", "limitations": ["string"]},
                "requirements.json": ["pandas", "matplotlib"],
            },
        }
        if previous_code is not None and failure_feedback is not None:
            prompt["repair"] = {
                "instruction": (
                    "Repair the previous candidate using only the sanitized failure information. "
                    "Every listed policy error must be removed. Do not replace a forbidden API with another "
                    "file, shell, reflection, or network API. Preserve the entrypoint and output contract. "
                    "Do not broaden permissions or dependencies."
                ),
                "previous_analysis.py": previous_code,
                "failure": failure_feedback,
            }
        auth = None
        if settings.code_analysis_opencode_password:
            auth = (settings.code_analysis_opencode_username, settings.code_analysis_opencode_password)
        protocol_retries = max(0, int(settings.code_analysis_opencode_protocol_retries))
        last_protocol_error: ValueError | None = None
        for protocol_attempt in range(protocol_retries + 1):
            if protocol_attempt:
                prompt["instruction"] += (
                    " Previous output could not be parsed. Return the artifact JSON object only, "
                    "with no preamble, Markdown, commentary, or truncation."
                )
            with httpx.Client(auth=auth) as client:
                response = client.post(
                    f"{settings.code_analysis_opencode_url.rstrip('/')}/session",
                    json={"title": "governed-analysis"},
                    timeout=self._request_timeout(context),
                )
                response.raise_for_status()
                session_id = response.json()["id"]
                try:
                    message = client.post(
                        f"{settings.code_analysis_opencode_url.rstrip('/')}/session/{session_id}/message",
                        headers={"X-Request-ID": f"{provider_request_id}-{protocol_attempt}"},
                        json={
                            "agent": "governed-analysis",
                            "tools": {},
                            "parts": [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}],
                        },
                        timeout=self._request_timeout(context),
                    )
                    message.raise_for_status()
                finally:
                    # A persistent headless server must not accumulate one session
                    # database entry per analysis job.
                    try:
                        client.delete(
                            f"{settings.code_analysis_opencode_url.rstrip('/')}/session/{session_id}",
                            timeout=5.0,
                        )
                    except httpx.HTTPError:
                        pass
            try:
                data = _extract_json_payload(_response_text(message.json()))
                return CandidateArtifact(
                    code=data["analysis.py"],
                    explanation=_normalize_explanation(data.get("explanation.json", {})),
                    requirements=_normalize_requirements(data.get("requirements.json", [])),
                    provider="opencode",
                    model=settings.llm_model,
                    provider_version=settings.code_analysis_opencode_version,
                    provider_request_id=provider_request_id,
                )
            except ValueError as exc:
                last_protocol_error = exc
                if protocol_attempt >= protocol_retries:
                    raise
        raise last_protocol_error or ValueError("OpenCode artifact protocol failed")

    def review_analysis(
        self,
        plan: dict,
        data_contract: dict,
        candidate: CandidateArtifact,
        *,
        context: ProviderContext | None = None,
    ) -> SecurityReview:
        """Ask a no-tools OpenCode agent to review the candidate before execution."""
        settings = get_settings()
        request_id = f"ocr_{uuid4().hex}"
        if context is not None:
            context.security_review_request_id = request_id
        prompt = {
            "instruction": (
                "Review the proposed analysis.py for security and policy risk. Do not execute it, "
                "do not use tools, do not request broader permissions, and do not rewrite the code. "
                "The deterministic worker already enforces its static policy; identify only concrete "
                "risks that would make running this candidate unsafe or outside the confirmed contract. "
                "Return exactly one JSON object with decision allow, repair, or reject. Use reject for "
                "a clearly dangerous candidate and repair for a concrete fixable violation."
            ),
            "plan": plan,
            "data_contract": data_contract,
            "candidate": {
                "analysis.py": candidate.code,
                "requirements": candidate.requirements,
            },
            "trusted_worker_contract": {
                "reproducibility": (
                    "The worker adds this field after sandbox execution from trusted server-side facts. "
                    "The candidate analysis.py must not return reproducibility, and its absence is not a finding."
                ),
                "required_candidate_keys": [
                    "conclusion",
                    "method",
                    "summary",
                    "evidence",
                    "charts",
                    "limitations",
                    "recommendations",
                ],
            },
            "review_schema": {
                "decision": "allow|repair|reject",
                "risk_level": "low|medium|high",
                "findings": ["short concrete finding"],
                "summary": "short explanation",
            },
            "hard_rules": [
                "No network, subprocess, shell, dynamic execution, reflection, environment access, or arbitrary file reads.",
                "Only the declared input_path and output_dir contract may be used.",
                "Only approved imports and requirements are allowed.",
                "Do not reject a candidate for omitting reproducibility; the trusted worker adds it after execution.",
                "Do not treat an unclear response as safe; return repair or reject when evidence is insufficient.",
            ],
        }
        auth = None
        if settings.code_analysis_opencode_password:
            auth = (
                settings.code_analysis_opencode_username,
                settings.code_analysis_opencode_password,
            )
        protocol_retries = max(0, int(settings.code_analysis_opencode_protocol_retries))
        last_error: ValueError | None = None
        for protocol_attempt in range(protocol_retries + 1):
            if protocol_attempt:
                prompt["instruction"] += (
                    " Return the review JSON object only, without Markdown, commentary, or preamble."
                )
            with httpx.Client(auth=auth) as client:
                response = client.post(
                    f"{settings.code_analysis_opencode_url.rstrip('/')}/session",
                    json={"title": "governed-security-review"},
                    timeout=self._request_timeout(context),
                )
                response.raise_for_status()
                session_id = response.json()["id"]
                try:
                    message = client.post(
                        f"{settings.code_analysis_opencode_url.rstrip('/')}/session/{session_id}/message",
                        headers={"X-Request-ID": f"{request_id}-{protocol_attempt}"},
                        json={
                            "agent": "governed-security-review",
                            "tools": {},
                            "parts": [{
                                "type": "text",
                                "text": json.dumps(prompt, ensure_ascii=False),
                            }],
                        },
                        timeout=self._request_timeout(context),
                    )
                    message.raise_for_status()
                finally:
                    try:
                        client.delete(
                            f"{settings.code_analysis_opencode_url.rstrip('/')}/session/{session_id}",
                            timeout=5.0,
                        )
                    except httpx.HTTPError:
                        pass
            try:
                data = _extract_json_object(
                    _response_text(message.json()),
                    required_keys={"decision", "risk_level", "findings", "summary"},
                )
                decision = str(data.get("decision") or "").strip().lower()
                risk_level = str(data.get("risk_level") or "").strip().lower()
                if decision not in {"allow", "repair", "reject"}:
                    raise ValueError("OpenCode security review returned an invalid decision")
                if risk_level not in {"low", "medium", "high"}:
                    raise ValueError("OpenCode security review returned an invalid risk level")
                raw_findings = data.get("findings", [])
                if not isinstance(raw_findings, list):
                    raise ValueError("OpenCode security review findings must be a list")
                findings = [str(item).strip()[:500] for item in raw_findings if str(item).strip()]
                summary = str(data.get("summary") or "").strip()[:1_000]
                if not summary:
                    raise ValueError("OpenCode security review returned no summary")
                return SecurityReview(
                    decision=decision,
                    risk_level=risk_level,
                    findings=findings[:20],
                    summary=summary,
                    provider_request_id=request_id,
                )
            except ValueError as exc:
                last_error = exc
                if protocol_attempt >= protocol_retries:
                    raise
        raise last_error or ValueError("OpenCode security review protocol failed")


def get_provider() -> CodeAgentProvider:
    return OpenCodeProvider() if get_settings().code_analysis_provider == "opencode" else FakeCodeAgentProvider()
