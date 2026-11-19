"""Deterministic, fail-closed runner primitives for the 17-case Provider study.

The module deliberately separates the public user question sent to the Agent
from the candidate truth supplied only to the post-run scorer. It does not
start experiments on import; live execution is an explicit API call.
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
import contextlib
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from evaluation.provider_candidate_protocol import (
    PROVIDER_CANDIDATE_SCORER_VERSION,
    score_provider_candidate_run,
)


SCORER_VERSION = PROVIDER_CANDIDATE_SCORER_VERSION
DEFAULT_SEED = 20261015
ARMS = ("full", "critic_llm_review_off")
EXPERIMENT_ID = "provider-candidate-17-full-off-r3-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    provider_id: str = "primary"
    model: str = "qwen3.8-flash"
    max_graph_steps: int = 48
    max_task_tokens: int = 60_000
    max_cost_cny_per_workflow: float = 5.0
    max_completion_tokens: int = 4_096
    max_sdk_sends_per_workflow: int = 10
    sdk_retries: int = 0
    application_attempts: int = 1
    fallback_enabled: bool = False
    max_tool_calls: int = 8
    max_latency_ms: int = 240_000
    model_timeout_seconds: float = 60.0
    max_revisions: int = 1
    critic_thinking_mode: str = "non_thinking_experiment"
    input_token_estimate_cap_per_call: int = 200_000
    input_byte_factor: int = 4
    input_protocol_token_margin: int = 16_384

    def __post_init__(self) -> None:
        if self.input_token_estimate_cap_per_call < 1 or self.input_byte_factor < 1 or self.input_protocol_token_margin < 0:
            raise ValueError("Input token reservation must have a positive enforced cap and valid estimator bounds.")

    @classmethod
    def from_frozen_arm_configs(cls, arms: Mapping[str, Mapping[str, Any]]) -> "ExperimentConfig":
        """Build the runtime budget/call config only from the frozen B/G settings.

        Shared execution fields must be present and identical in both arms;
        arm-specific Critic fields remain outside this shared config.
        """
        baseline = arms.get("B") if isinstance(arms, Mapping) else None
        governed = arms.get("G") if isinstance(arms, Mapping) else None
        if not isinstance(baseline, Mapping) or not isinstance(governed, Mapping):
            raise ValueError("Frozen B/G execution settings are missing")
        names = [item.name for item in fields(cls)]
        missing = [name for name in names if name not in baseline or name not in governed]
        if missing:
            raise ValueError("Frozen B/G execution settings omit required fields: " + ", ".join(missing))
        mismatched = [name for name in names if baseline[name] != governed[name]]
        if mismatched:
            raise ValueError("Frozen B/G shared execution settings differ: " + ", ".join(mismatched))
        if "max_rounds" in baseline and baseline["max_rounds"] != baseline["max_graph_steps"]:
            raise ValueError("Frozen B max_rounds does not match max_graph_steps")
        return cls(**{name: baseline[name] for name in names})

    def build_llm_call_config(self, settings: Any) -> Any:
        """Materialize the SDK policy from the same frozen experiment config."""
        from app.llm.call_config import LLMCallConfig

        return LLMCallConfig.provider_ab_experiment(
            settings,
            provider_id=self.provider_id,
            max_completion_tokens=self.max_completion_tokens,
            timeout_seconds=self.model_timeout_seconds,
        )

    def summary(self, arm: str) -> dict[str, Any]:
        return {
            **self.__dict__,
            "arm": arm,
            "critic_review_mode": "full" if arm == "full" else "critic_llm_review_off",
            "optional_critic_reviews": ["groundedness", "completeness", "consistency"] if arm == "full" else [],
            "only_arm_difference": "three non-thinking optional LLM Critic reviews" if arm == "full" else "optional LLM Critic reviews marked evaluation_ablation",
        }


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_manifest(cases: list[Mapping[str, Any]], *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Create a seeded, paired schedule with arm-order counts differing by at most one."""
    case_ids = [str(row["case_id"]) for row in cases]
    if len(case_ids) != 17 or len(set(case_ids)) != len(case_ids):
        raise ValueError("Frozen candidate set must contain 17 unique cases.")
    pairs = [(case_id, replicate) for case_id in case_ids for replicate in (1, 2, 3)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    first_arm = ["full"] * (len(pairs) // 2) + ["critic_llm_review_off"] * (len(pairs) - len(pairs) // 2)
    rng.shuffle(first_arm)

    entries: list[dict[str, Any]] = []
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, EXPERIMENT_ID)
    for (case_id, replicate), first in zip(pairs, first_arm):
        ordered_arms = (first, ARMS[1] if first == ARMS[0] else ARMS[0])
        pair_id = f"{case_id}-r{replicate}"
        for arm in ordered_arms:
            run_key = f"{EXPERIMENT_ID}:{case_id}:{arm}:{replicate}"
            run_id = str(uuid.uuid5(namespace, run_key))
            entries.append({
                "case_id": case_id,
                "arm": arm,
                "replicate": replicate,
                "pair_id": pair_id,
                "run_id": run_id,
                "schedule_index": len(entries) + 1,
                "status": "not_started",
                "started_at": None,
                "completed_at": None,
                "provider_sends": 0,
                "score": None,
            })
    order_counts: dict[str, int] = {arm: 0 for arm in ARMS}
    for index in range(0, len(entries), 2):
        order_counts[entries[index]["arm"]] += 1
    return {
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "schedule_mode": "serial_paired; pair order and arm-first assignment seeded; full/off first counts differ by at most one",
        "planned_units": len(entries),
        "arm_order_pair_counts": order_counts,
        "entries": entries,
    }


@dataclass(frozen=True)
class CostPolicy:
    """Reservation model copied from the accepted AOV preflight policy."""

    max_sdk_calls_per_workflow: int = 10
    input_token_estimate_cap_per_call: int = 200_000
    input_byte_factor: int = 4
    input_protocol_token_margin: int = 16_384
    max_completion_tokens: int = 4_096
    documented_output_variance_tokens: int = 10
    input_cny_per_million: float = 1.094
    output_cny_per_million: float = 3.427
    cumulative_estimate_limit_cny: float = 250.0
    basis: str = "maximum published qwen3.8-flash regional on-demand rates; no cache discount"

    def validate_enforcement_config(self, config: ExperimentConfig) -> None:
        """Refuse execution unless the preflight estimate matches send-time guards."""
        checks = {
            "max_sdk_calls_per_workflow": (self.max_sdk_calls_per_workflow, config.max_sdk_sends_per_workflow),
            "input_token_estimate_cap_per_call": (self.input_token_estimate_cap_per_call, config.input_token_estimate_cap_per_call),
            "input_byte_factor": (self.input_byte_factor, config.input_byte_factor),
            "input_protocol_token_margin": (self.input_protocol_token_margin, config.input_protocol_token_margin),
            "max_completion_tokens": (self.max_completion_tokens, config.max_completion_tokens),
        }
        mismatches = [name for name, (estimated, enforced) in checks.items() if estimated != enforced]
        if mismatches:
            raise ValueError("Cost preflight differs from SDK send-time enforcement: " + ", ".join(mismatches))
        if self.max_per_run_cny() > config.max_cost_cny_per_workflow + 1e-12:
            raise ValueError("Worst-case cost reservation exceeds per-workflow configured estimate limit.")

    def max_per_call_cny(self) -> float:
        input_tokens = self.input_token_estimate_cap_per_call
        output_tokens = self.max_completion_tokens + self.documented_output_variance_tokens
        return (input_tokens * self.input_cny_per_million + output_tokens * self.output_cny_per_million) / 1_000_000

    def max_per_run_cny(self) -> float:
        return self.max_per_call_cny() * self.max_sdk_calls_per_workflow

    def estimate(self, planned_runs: int) -> dict[str, Any]:
        amount = self.max_per_run_cny() * planned_runs
        return {
            "runs": planned_runs,
            "sdk_calls_upper_bound": planned_runs * self.max_sdk_calls_per_workflow,
            "input_token_upper_bound": planned_runs * self.max_sdk_calls_per_workflow * self.input_token_estimate_cap_per_call,
            "output_token_upper_bound_including_variance": planned_runs * self.max_sdk_calls_per_workflow * (self.max_completion_tokens + self.documented_output_variance_tokens),
            "per_run_cny": round(self.max_per_run_cny(), 8),
            "estimated_cny": round(amount, 8),
            "cumulative_estimate_limit_cny": self.cumulative_estimate_limit_cny,
            "within_estimate_limit": amount <= self.cumulative_estimate_limit_cny,
            "is_account_hard_cap": False,
            "basis": self.basis,
        }


class RunLedger:
    """Append-only event ledger; anything started is never silently rescheduled."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        if path.exists():
            self.events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, run_id: str, event: str, **fields: Any) -> dict[str, Any]:
        row = {"run_id": run_id, "event": event, "recorded_at": datetime.now(timezone.utc).isoformat(), **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.events.append(row)
        return row

    def latest(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        sends: dict[str, int] = {}
        for event in self.events:
            run_id = event["run_id"]
            if event["event"] == "sdk_send_started":
                sends[run_id] = sends.get(run_id, 0) + 1
            if event["event"] in {
                "cost_reserved", "started", "completed", "failed",
                "interrupted_before_send", "interrupted_result_unknown",
            }:
                result[run_id] = dict(event)
        for run_id, row in result.items():
            row["provider_sends"] = sends.get(run_id, 0)
        return result

    def reconcile_interrupted(self) -> dict[str, int]:
        """Mark started-without-terminal units interrupted; never classify as not-started."""
        latest = self.latest()
        counts = {"interrupted_before_send": 0, "interrupted_result_unknown": 0}
        started_ids = {event["run_id"] for event in self.events if event["event"] == "started"}
        terminal = {"completed", "failed", "interrupted_before_send", "interrupted_result_unknown"}
        for run_id, row in latest.items():
            if run_id not in started_ids or row.get("event") in terminal:
                continue
            target = "interrupted_result_unknown" if row.get("provider_sends", 0) else "interrupted_before_send"
            self.append(run_id, target, provider_sends=row.get("provider_sends", 0), retry_forbidden=True)
            counts[target] += 1
        return counts

    def resume_candidates(self, entries: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        # Selection/dry-run is read-only. Recovery is an explicit mutation via
        # reconcile_interrupted, not a side effect of inspecting pending units.
        states = self.latest()
        return [entry for entry in entries if states.get(str(entry["run_id"]), {}).get("event") in (None, "cost_reserved")]


class EstimatedCostLedger:
    """Reserve a whole run before starting; unknown usage never releases a reservation."""

    def __init__(self, policy: CostPolicy, path: Path):
        self.policy, self.path = policy, path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reservations: list[dict[str, Any]] = []
        if path.exists():
            self.reservations = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @property
    def reserved_cny(self) -> float:
        return sum(float(row.get("reserved_cny", 0) or 0) for row in self.reservations)

    def reserve(self, run_id: str) -> float:
        existing = next((row for row in self.reservations if row["run_id"] == run_id), None)
        if existing is not None:
            return float(existing["reserved_cny"])
        amount = self.policy.max_per_run_cny()
        if self.reserved_cny + amount > self.policy.cumulative_estimate_limit_cny + 1e-12:
            raise CumulativeCostLimitExceeded("Cumulative estimated cost limit would be exceeded; no workflow started.")
        row = {"run_id": run_id, "reserved_cny": amount, "usage_status": "unknown_until_completed", "created_at": datetime.now(timezone.utc).isoformat()}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.reservations.append(row)
        return amount

    def record_usage(self, run_id: str, usage: Mapping[str, Any]) -> dict[str, Any]:
        reserve = next((float(row["reserved_cny"]) for row in self.reservations if row.get("run_id") == run_id and row.get("reserved_cny") is not None), None)
        if reserve is None:
            raise RuntimeError("Cannot account for a run without a persisted worst-case reservation.")
        actual: float | None = None
        overrun = False
        if usage.get("status") == "complete":
            input_tokens = int(usage["total_input_tokens"])
            output_tokens = int(usage["total_output_tokens"])
            actual = (input_tokens * self.policy.input_cny_per_million + output_tokens * self.policy.output_cny_per_million) / 1_000_000
            overrun = actual > reserve + 1e-12
        event = {
            "run_id": run_id,
            "event": "actual_usage_recorded",
            "usage_status": usage.get("status"),
            "known_input_token_subtotal": usage.get("known_input_token_subtotal"),
            "known_output_token_subtotal": usage.get("known_output_token_subtotal"),
            "actual_estimated_cny": actual,
            "reserved_cny_retained": reserve,
            "reservation_exceeded": overrun,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.reservations.append(event)
        return event


class CumulativeCostLimitExceeded(RuntimeError):
    pass


def verify_hashes(expected: Mapping[str, str], root: Path) -> None:
    for relpath, expected_sha in expected.items():
        path = root / relpath
        if not path.is_file():
            raise RuntimeError(f"Frozen input missing: {relpath}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha:
            raise RuntimeError(f"Frozen input hash mismatch: {relpath}")


def run_workflow_unit(
    entry: Mapping[str, Any], candidate: Mapping[str, Any], *, tenant_context: Any,
    trace_dir: Path, ledger: RunLedger, cost_ledger: EstimatedCostLedger,
    experiment_config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    workflow_fn: Callable[..., dict[str, Any]] | None = None,
    scorer_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Invoke the normal production workflow; candidate truth is scorer-only."""
    if workflow_fn is None:
        from app.graph.workflow import run_chat_workflow
        workflow_fn = run_chat_workflow
    if scorer_fn is None:
        scorer_fn = score_provider_candidate_run
    cost_ledger.policy.validate_enforcement_config(experiment_config)
    from app.graph.critic_review_mode import CriticLLMReviewMode
    from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
    from app.graph import workflow as workflow_module

    mode = entry["arm"]
    review_mode = CriticLLMReviewMode.FULL if mode == "full" else CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF
    call_config = experiment_config.build_llm_call_config(workflow_module.get_settings())
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_id = str(entry["run_id"])
    existing = ledger.latest().get(trace_id)
    if existing and existing.get("event") not in ("cost_reserved",):
        raise RuntimeError("Run unit has already started or terminated; automatic retry is forbidden.")
    # Reserve the frozen worst-case estimate before marking the unit started.
    # A failed or interrupted unit keeps its reservation and is never retried.
    reservation = cost_ledger.reserve(trace_id)
    ledger.append(trace_id, "cost_reserved", reserved_cny=reservation, usage_status="unknown_until_completion")
    ledger.append(trace_id, "started", case_id=entry["case_id"], arm=mode, replicate=entry["replicate"])
    started_at = time.perf_counter()
    try:
        response = workflow_fn(
            str(candidate["user_question"]),
            tenant_context=tenant_context,
            memory_enabled=False,
            trace_id=trace_id,
            _critic_llm_review_mode=review_mode,
            _critic_llm_thinking_mode=CriticLLMThinkingMode.NON_THINKING_EXPERIMENT,
            _llm_call_config=call_config,
        )
        trace_path = trace_dir / f"{trace_id}.json"
        if not trace_path.is_file():
            raise RuntimeError("Workflow completed without a persisted trace; run is failed and not retried.")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        scored = scorer_fn(trace, candidate=dict(candidate), arm=mode, replicate=int(entry["replicate"]))
        run_events = [event for event in ledger.events if event.get("run_id") == trace_id]
        sql_started = sum(event.get("event") == "sql_attempt_started" for event in run_events)
        sql_succeeded = sum(event.get("event") == "sql_attempt_succeeded" for event in run_events)
        sql_failed = sum(event.get("event") == "sql_attempt_failed" for event in run_events)
        output = {
            "response": response,
            "trace": trace,
            "score": scored,
            "error": (response.get("errors") if isinstance(response, Mapping) else None),
        "measurements": {
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "sdk_sends": sum(event.get("event") == "sdk_send_started" for event in run_events),
                "sql_attempts": sql_started,
                "sql_succeeded": sql_succeeded,
                "sql_failed": sql_failed,
                "tool_calls": len(((trace.get("state_snapshot") or {}).get("tool_calls") or [])),
                "llm_calls": len(((trace.get("state_snapshot") or {}).get("llm_calls") or [])),
                "token_usage": token_usage_snapshot(trace, run_events, run_id=trace_id),
            },
        }
        accounting = cost_ledger.record_usage(trace_id, output["measurements"]["token_usage"])
        output["measurements"]["cost_accounting"] = accounting
        reservation = next(
            (row for row in cost_ledger.reservations
             if row.get("run_id") == trace_id and row.get("event") != "actual_usage_recorded" and row.get("reserved_cny") is not None),
            None,
        )
        _reconcile_cost_accounting(
            output["measurements"]["token_usage"], accounting, cost_ledger.policy,
            expected_reservation_cny=float(reservation["reserved_cny"]) if reservation else None,
        )
        output_path = trace_dir / f"{trace_id}.run_record.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        ledger.append(
            trace_id, "usage_summary",
            **{key: value for key, value in output["measurements"]["token_usage"].items() if key != "run_id"},
        )
        ledger.append(trace_id, "completed", provider_sends=output["measurements"]["sdk_sends"], score=scored)
        return output
    except BaseException as exc:
        # Keep executor records secret-safe; detailed sanitized failures remain
        # available from the normal workflow trace when one was persisted.
        failure = {
            "run_id": trace_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "retry_forbidden": True,
            "trace_present": (trace_dir / f"{trace_id}.json").is_file(),
            "provider_sends": sum(event.get("run_id") == trace_id and event.get("event") == "sdk_send_started" for event in ledger.events),
        }
        (trace_dir / f"{trace_id}.failure_record.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        ledger.append(trace_id, "failed", error_type=type(exc).__name__, retry_forbidden=True)
        raise


def _sdk_sends(trace: Mapping[str, Any]) -> int:
    state = (trace.get("state_snapshot") or {}) if isinstance(trace.get("state_snapshot"), Mapping) else {}
    calls = state.get("llm_calls") or []
    return sum(
        1 for call in calls for attempt in (call.get("attempts") or [])
        if attempt.get("attempt_id") and attempt.get("sdk_request_sent") is not False
    )


def _sdk_send_attempt_ids(trace: Mapping[str, Any]) -> set[str]:
    state = (trace.get("state_snapshot") or {}) if isinstance(trace.get("state_snapshot"), Mapping) else {}
    return {
        str(attempt["attempt_id"])
        for call in (state.get("llm_calls") or [])
        for attempt in (call.get("attempts") or [])
        if attempt.get("attempt_id") and attempt.get("sdk_request_sent") is not False
    }


def token_usage_snapshot(
    trace: Mapping[str, Any], events: Iterable[Mapping[str, Any]], *, run_id: str,
) -> dict[str, Any]:
    """Aggregate usage only after binding the explicit unit ID to the persisted trace.

    The trace schema stores its ID at ``state_snapshot.trace_id``.  A missing or
    conflicting ID is a metering error, never an empty run (and therefore never
    a fabricated zero-token result).
    """
    explicit_id = str(run_id or "").strip()
    state = trace.get("state_snapshot") if isinstance(trace.get("state_snapshot"), Mapping) else {}
    nested_id = str((state or {}).get("trace_id") or "").strip()
    legacy_id = str(trace.get("trace_id") or "").strip()
    errors: list[str] = []
    if not explicit_id:
        errors.append("execution_unit_run_id_missing")
    if not nested_id and not legacy_id:
        errors.append("trace_run_id_missing")
    if nested_id and nested_id != explicit_id:
        errors.append("execution_unit_run_id_does_not_match_state_snapshot_trace_id")
    if legacy_id and legacy_id != explicit_id:
        errors.append("execution_unit_run_id_does_not_match_top_level_trace_id")
    if nested_id and legacy_id and nested_id != legacy_id:
        errors.append("trace_id_fields_conflict")

    all_events = list(events)
    if any(not row.get("run_id") for row in all_events):
        errors.append("run_event_run_id_missing")
    run_events = [row for row in all_events if row.get("run_id") == explicit_id] if explicit_id else []
    # If a conflicting/missing trace ID is present, do not silently aggregate
    # an apparently empty subset. The selected subset remains inspectable only
    # as counts; totals are withheld below.
    sent_events = [row for row in run_events if row.get("event") == "sdk_send_started"]
    responses = [row for row in run_events if row.get("event") == "sdk_response_received"]
    failed_events = [row for row in run_events if row.get("event") == "sdk_send_failed"]
    refused_events = [row for row in run_events if row.get("event") == "sdk_send_refused" and row.get("send_started") is False]

    known_inputs: list[int] = []
    known_outputs: list[int] = []
    complete_response_ids: set[str] = set()
    unknown_response_count = 0
    for row in responses:
        input_tokens, output_tokens = row.get("input_tokens"), row.get("output_tokens")
        valid_counts = (
            isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens >= 0
            and isinstance(output_tokens, int) and not isinstance(output_tokens, bool) and output_tokens >= 0
        )
        if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens >= 0:
            known_inputs.append(input_tokens)
        if isinstance(output_tokens, int) and not isinstance(output_tokens, bool) and output_tokens >= 0:
            known_outputs.append(output_tokens)
        if row.get("usage_status") == "complete" and valid_counts:
            if row.get("attempt_id"):
                complete_response_ids.add(str(row["attempt_id"]))
        else:
            unknown_response_count += 1

    # An SDK send is matched by the attempt ID recorded at the send boundary.
    # Missing/duplicate IDs or multiple terminal events indicate an incomplete
    # or corrupted ledger; they are not papered over by arithmetic subtraction.
    sent_attempt_ids: list[str] = []
    for row in sent_events:
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id:
            errors.append("sdk_send_started_missing_attempt_id")
        sent_attempt_ids.append(attempt_id)
    if len([x for x in sent_attempt_ids if x]) != len(set(x for x in sent_attempt_ids if x)):
        errors.append("duplicate_sdk_send_attempt_id")
    terminals_by_attempt: dict[str, int] = {}
    for row in [*responses, *failed_events]:
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id:
            errors.append("sdk_send_terminal_event_missing_attempt_id")
            continue
        terminals_by_attempt[attempt_id] = terminals_by_attempt.get(attempt_id, 0) + 1
    for attempt_id in sent_attempt_ids:
        if attempt_id and terminals_by_attempt.get(attempt_id, 0) != 1:
            errors.append("sdk_send_missing_or_duplicate_terminal_event")
    if set(terminals_by_attempt) - set(sent_attempt_ids):
        errors.append("sdk_terminal_event_without_send_start")
    trace_attempt_ids = _sdk_send_attempt_ids(trace)
    if len(trace_attempt_ids) != _sdk_sends(trace) or trace_attempt_ids != {x for x in sent_attempt_ids if x}:
        errors.append("trace_llm_call_attempts_do_not_match_sdk_send_ledger")

    failed_ids = {str(row.get("attempt_id") or "") for row in failed_events}
    unknown_attempt_ids = {
        attempt_id for attempt_id in sent_attempt_ids
        if attempt_id and (attempt_id in failed_ids or attempt_id not in complete_response_ids)
    }
    unknown_count = max(unknown_response_count, len(unknown_attempt_ids))
    state_summary = (state or {}).get("llm_token_summary")
    if not isinstance(state_summary, Mapping):
        errors.append("trace_llm_token_summary_missing")
        state_summary = {}
    state_checks = {
        "sdk_attempt_count": ("sdk_attempt_count", len(sent_events)),
        "known_input_tokens_subtotal": ("known_input_tokens_subtotal", sum(known_inputs)),
        "known_output_tokens_subtotal": ("known_output_tokens_subtotal", sum(known_outputs)),
        "local_send_refusal_count": ("local_send_refusal_count", len(refused_events)),
        "unknown_usage_attempt_count": ("unknown_usage_attempt_count", unknown_count),
    }
    for label, (field, expected) in state_checks.items():
        if field not in state_summary:
            errors.append(f"trace_llm_token_summary_missing_{label}")
            continue
        if state_summary.get(field) != expected:
            errors.append(f"trace_llm_token_summary_{label}_mismatch")
    if not sent_events and refused_events and not errors:
        status, total_input, total_output = "local_send_refused", 0, 0
    elif not sent_events and not refused_events and not errors:
        status, total_input, total_output = "no_requests", 0, 0
    elif errors:
        status, total_input, total_output = "metering_error", None, None
    elif unknown_count:
        status, total_input, total_output = "sent_usage_unknown", None, None
    else:
        status, total_input, total_output = "complete", sum(known_inputs), sum(known_outputs)

    computed_state_input = total_input if sent_events else 0
    computed_state_output = total_output if sent_events else 0
    computed_state_total = (
        computed_state_input + computed_state_output
        if computed_state_input is not None and computed_state_output is not None else None
    )
    for field, actual in (
        ("input_tokens", computed_state_input),
        ("output_tokens", computed_state_output),
        ("total_tokens", computed_state_total),
    ):
        if state_summary.get(field) != actual:
            errors.append(f"trace_llm_token_summary_{field}_mismatch")
    if errors:
        status, total_input, total_output = "metering_error", None, None

    return {
        "run_id": explicit_id or None,
        "trace_run_id": nested_id or legacy_id or None,
        "sdk_sends": len(sent_events),
        "provider_failed_sends": len(failed_events),
        "responses_with_usage": len(responses) - unknown_response_count,
        "unknown_or_failed_usage_requests": unknown_count,
        "local_pre_send_refusals": len(refused_events),
        "known_input_token_subtotal": sum(known_inputs),
        "known_output_token_subtotal": sum(known_outputs),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output if total_input is not None and total_output is not None else None,
        "status": status,
        "request_status": (
            "metering_error" if errors else "requested" if sent_events
            else "not_sent" if refused_events else "no_model_request"
        ),
        "metering_errors": sorted(set(errors)),
        "state_summary_matches": not errors,
        "unrelated_run_event_count": sum(1 for row in all_events if row.get("run_id") != explicit_id),
        "verified_usage": status == "complete",
    }


def _reconcile_cost_accounting(
    usage: dict[str, Any], accounting: dict[str, Any], policy: CostPolicy,
    *, expected_reservation_cny: float | None = None,
) -> None:
    """Attach explicit cross-checks; never replace one source with another."""
    errors = list(usage.get("metering_errors") or [])
    cost_errors: list[str] = []
    if accounting.get("run_id") != usage.get("run_id"):
        cost_errors.append("cost_record_run_id_mismatch")
    if accounting.get("usage_status") != usage.get("status"):
        cost_errors.append("cost_record_usage_status_mismatch")
    if expected_reservation_cny is None:
        cost_errors.append("cost_reservation_record_missing")
    elif accounting.get("reserved_cny_retained") is None or round(float(accounting["reserved_cny_retained"]), 12) != round(expected_reservation_cny, 12):
        cost_errors.append("cost_record_reservation_amount_mismatch")
    for field in ("known_input_token_subtotal", "known_output_token_subtotal"):
        if accounting.get(field) != usage.get(field):
            cost_errors.append(f"cost_record_{field}_mismatch")
    expected_cost = None
    if usage.get("status") == "complete":
        expected_cost = (
            int(usage["total_input_tokens"]) * policy.input_cny_per_million
            + int(usage["total_output_tokens"]) * policy.output_cny_per_million
        ) / 1_000_000
    if accounting.get("actual_estimated_cny") != expected_cost:
        # Floating point serialization can vary at the final bit; compare at
        # the recorded currency precision instead of silently accepting a gap.
        actual = accounting.get("actual_estimated_cny")
        if actual is None or expected_cost is None or round(float(actual), 12) != round(expected_cost, 12):
            cost_errors.append("cost_record_estimated_amount_mismatch")
    expected_overrun = bool(expected_cost is not None and expected_reservation_cny is not None and expected_cost > expected_reservation_cny + 1e-12)
    if accounting.get("reservation_exceeded") != expected_overrun:
        cost_errors.append("cost_record_reservation_exceeded_flag_mismatch")
    errors.extend(cost_errors)
    if errors:
        usage["status"] = "metering_error"
        usage["total_input_tokens"] = None
        usage["total_output_tokens"] = None
        usage["total_tokens"] = None
    usage["metering_errors"] = sorted(set(errors))
    usage["cost_record_matches"] = not cost_errors
    usage["cost_record_reconciliation_errors"] = sorted(set(cost_errors))
    accounting["reconciliation_matches"] = not cost_errors
    accounting["reconciliation_errors"] = sorted(set(cost_errors))


class _BudgetedCompletions:
    def __init__(self, completions: Any, *, ledger: RunLedger, config: ExperimentConfig, send_counts: dict[str, int]):
        self._completions, self._ledger, self._config = completions, ledger, config
        self._sends = send_counts

    def _refuse(self, state: Mapping[str, Any] | None, reason_code: str) -> None:
        from app.llm.client import LocalSDKSendRefused

        active = dict(state or {})
        calls = active.get("llm_calls") or []
        last_call = calls[-1] if calls else {}
        attempts = last_call.get("attempts") or []
        attempt = attempts[-1] if attempts else {}
        run_id = str(active.get("trace_id") or "") or None
        # Persist only safe identifiers and a controlled reason code. This event
        # explicitly records a local refusal before SDK send-start.
        self._ledger.append(
            run_id, "sdk_send_refused", reason_code=reason_code,
            send_started=False, call_id=last_call.get("call_id") or None,
            attempt_id=attempt.get("attempt_id") or None,
            operation=last_call.get("operation") or None,
            node=last_call.get("node") or None,
        )
        raise LocalSDKSendRefused(reason_code)

    def create(self, **kwargs: Any) -> Any:
        from app.llm.client import active_state

        state = active_state()
        state = state or {}
        run_id = str(state.get("trace_id") or "")
        if not run_id:
            self._refuse(state, "missing_run_id")
        sent = self._sends.get(run_id, 0)
        if sent >= self._config.max_sdk_sends_per_workflow:
            self._refuse(state, "send_limit_reached")
        if kwargs.get("model") != self._config.model:
            self._refuse(state, "model_mismatch")
        if "max_tokens" in kwargs or kwargs.get("max_completion_tokens") != self._config.max_completion_tokens:
            self._refuse(state, "completion_limit_mismatch")
        payload = json.dumps(kwargs, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")
        estimated_input = len(payload) * self._config.input_byte_factor + self._config.input_protocol_token_margin
        if estimated_input > self._config.input_token_estimate_cap_per_call:
            self._refuse(state, "input_estimate_limit")
        calls = state.get("llm_calls") or []
        last_call = calls[-1] if calls else {}
        current_node = str(last_call.get("node") or "")
        arm = str(state.get("critic_llm_review_mode") or "")
        if current_node == "critic":
            if arm == "critic_llm_review_off":
                self._refuse(state, "critic_review_disabled")
            if arm == "full" and (kwargs.get("extra_body") or {}).get("enable_thinking") is not False:
                self._refuse(state, "critic_thinking_mismatch")
        attempts = last_call.get("attempts") or []
        active_attempt = attempts[-1] if attempts else {}
        # This fsynced event is the boundary immediately before the SDK send.
        send_number = sent + 1
        self._ledger.append(
            run_id, "sdk_send_started", send_number=send_number,
            call_id=last_call.get("call_id"), attempt_id=active_attempt.get("attempt_id"),
            operation=last_call.get("operation"), node=last_call.get("node"),
            estimated_input_tokens=estimated_input,
            input_token_estimate_cap=self._config.input_token_estimate_cap_per_call,
        )
        self._sends[run_id] = send_number
        try:
            response = self._completions.create(**kwargs)
        except Exception as exc:
            self._ledger.append(
                run_id, "sdk_send_failed", send_number=send_number,
                call_id=last_call.get("call_id"), attempt_id=active_attempt.get("attempt_id"),
                error_type=type(exc).__name__, usage_status="unknown",
            )
            raise
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion = getattr(usage, "completion_tokens", None) if usage is not None else None
        self._ledger.append(
            run_id, "sdk_response_received", send_number=send_number,
            call_id=last_call.get("call_id"), attempt_id=active_attempt.get("attempt_id"),
            input_tokens=prompt, output_tokens=completion,
            usage_status="complete" if prompt is not None and completion is not None else "unknown",
        )
        return response


class _BudgetedChat:
    def __init__(self, chat: Any, *, ledger: RunLedger, config: ExperimentConfig, send_counts: dict[str, int]):
        self.completions = _BudgetedCompletions(chat.completions, ledger=ledger, config=config, send_counts=send_counts)


class _BudgetedClient:
    def __init__(self, client: Any, *, ledger: RunLedger, config: ExperimentConfig, send_counts: dict[str, int]):
        self.chat = _BudgetedChat(client.chat, ledger=ledger, config=config, send_counts=send_counts)


@contextlib.contextmanager
def bind_isolated_experiment_runtime(*, engine: Any, trace_dir: Path, ledger: RunLedger,
                                     config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG):
    """Bind ordinary workflow modules to one loopback test DB and frozen limits.

    The caller must provide the test tenant context when invoking
    ``run_workflow_unit``. This context manager is synchronous by design: runs
    are serial, so request configuration cannot cross arms.
    """
    from app.core.config import get_settings as base_get_settings
    from app.graph import workflow
    from app.harness import storage
    from app.llm import client as llm
    import app.llm.routing as routing
    from app.data.repositories import daily_metrics
    from app.core import time_expression

    url = getattr(getattr(engine, "url", None), "render_as_string", lambda **_: "") (hide_password=True)
    db_name = str(getattr(getattr(engine, "url", None), "database", "") or "")
    if not any(host in url for host in ("127.0.0.1", "localhost", "[::1]")) or not db_name.startswith(("provider_candidate_", "platform_trend_acceptance_")):
        raise RuntimeError("Experiment requires a loopback-only isolated PostgreSQL engine.")
    settings = base_get_settings().model_copy(update={
        "max_graph_steps": config.max_graph_steps,
        "max_tokens": config.max_task_tokens,
        "max_cost": config.max_cost_cny_per_workflow,
        "max_tool_calls": config.max_tool_calls,
        "max_latency_ms": config.max_latency_ms,
        "max_revisions": config.max_revisions,
        "llm_timeout_seconds": config.model_timeout_seconds,
        "llm_resilience_attempts": config.application_attempts,
        "llm_input_cost_per_1k": 1.094 / 1000,
        "llm_output_cost_per_1k": 3.427 / 1000,
        "llm_fast_input_cost_per_1k": 1.094 / 1000,
        "llm_fast_output_cost_per_1k": 3.427 / 1000,
        "llm_quality_input_cost_per_1k": 1.094 / 1000,
        "llm_quality_output_cost_per_1k": 3.427 / 1000,
        "trace_backend": "file",
        "trace_dir": str(trace_dir),
        "llm_user_daily_token_limit": 0,
        "llm_user_monthly_token_limit": 0,
        "llm_user_daily_cost_limit": 0,
        "llm_user_monthly_cost_limit": 0,
    })
    modules = (workflow, storage, llm, routing)
    originals = {module: module.get_settings for module in modules}
    original_openai = llm.OpenAI
    original_usage = llm.record_usage
    original_engines = (daily_metrics.engine, time_expression.engine)
    original_persist = workflow._persist_chat_turn
    from app.mcp import client as mcp, tools
    original_local_transport = getattr(mcp, "_call_local_mcp_tool", None)
    send_counts: dict[str, int] = {}

    def openai_factory(**kwargs: Any) -> Any:
        if kwargs.get("max_retries") != config.sdk_retries:
            raise RuntimeError("OpenAI SDK client retries differ from frozen experiment config.")
        if float(kwargs.get("timeout", 0)) != config.model_timeout_seconds:
            raise RuntimeError(f"OpenAI SDK timeout differs from frozen experiment config: {kwargs.get('timeout')!r}.")
        return _BudgetedClient(original_openai(**kwargs), ledger=ledger, config=config, send_counts=send_counts)

    from sqlalchemy import event

    def sql_before(_conn: Any, _cursor: Any, statement: str, parameters: Any, _context: Any, _many: Any) -> None:
        from app.core.execution_context import active_state
        state = active_state() or {}
        run_id = str(state.get("trace_id") or "")
        if run_id:
            ledger.append(run_id, "sql_attempt_started", sql_sha256=hashlib.sha256(statement.encode("utf-8")).hexdigest(), parameter_names=sorted(str(k) for k in parameters) if isinstance(parameters, Mapping) else None)

    def sql_after(_conn: Any, _cursor: Any, statement: str, parameters: Any, _context: Any, _many: Any) -> None:
        from app.core.execution_context import active_state
        state = active_state() or {}
        run_id = str(state.get("trace_id") or "")
        if run_id:
            ledger.append(run_id, "sql_attempt_succeeded", sql_sha256=hashlib.sha256(statement.encode("utf-8")).hexdigest())

    def sql_error(exception_context: Any) -> None:
        from app.core.execution_context import active_state
        state = active_state() or {}
        run_id = str(state.get("trace_id") or "")
        statement = str(getattr(exception_context, "statement", "") or "")
        if run_id:
            ledger.append(run_id, "sql_attempt_failed", sql_sha256=hashlib.sha256(statement.encode("utf-8")).hexdigest(), error_type=type(exception_context.original_exception).__name__)

    event.listen(engine, "before_cursor_execute", sql_before)
    event.listen(engine, "after_cursor_execute", sql_after)
    event.listen(engine, "handle_error", sql_error)

    for module in modules:
        module.get_settings = lambda settings=settings: settings
    llm.OpenAI = openai_factory
    # Usage is retained in the request trace; do not write an experiment to a business usage DB.
    llm.record_usage = lambda **_: None
    daily_metrics.engine = engine
    time_expression.engine = engine
    workflow._persist_chat_turn = lambda *args, **kwargs: None

    async def local_transport(tool_name: str, arguments: dict[str, Any]) -> Any:
        tool = getattr(tools, tool_name, None)
        if tool is None or tool_name.startswith("_"):
            raise RuntimeError(f"No registered local tool implementation for {tool_name!r}.")
        return tool(**arguments)

    mcp._call_local_mcp_tool = local_transport
    try:
        from app.llm.routing import model_for
        actual_models = {operation: model_for(operation)[0] for operation in ("classify", "extract", "chat")}
        if set(actual_models.values()) != {config.model}:
            raise RuntimeError("Configured classify/extract/chat routes do not all match the frozen model.")
        yield settings
    finally:
        for module, getter in originals.items():
            module.get_settings = getter
        llm.OpenAI = original_openai
        llm.record_usage = original_usage
        daily_metrics.engine, time_expression.engine = original_engines
        workflow._persist_chat_turn = original_persist
        if original_local_transport is None:
            delattr(mcp, "_call_local_mcp_tool")
        else:
            mcp._call_local_mcp_tool = original_local_transport
        event.remove(engine, "before_cursor_execute", sql_before)
        event.remove(engine, "after_cursor_execute", sql_after)
        event.remove(engine, "handle_error", sql_error)


class CandidateBatchExecutor:
    """Serial executor; only explicit ``run_pending`` can invoke the workflow."""

    def __init__(self, *, root: Path, manifest: Mapping[str, Any], candidates: list[Mapping[str, Any]],
                 frozen_hashes: Mapping[str, str], ledger: RunLedger, cost_ledger: EstimatedCostLedger,
                 trace_dir: Path, config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
                 expected_units: int = 102,
                 scorer_fn: Callable[..., dict[str, Any]] | None = None):
        self.root, self.manifest = root, manifest
        self.candidates = {str(row["case_id"]): row for row in candidates}
        self.frozen_hashes, self.ledger, self.cost_ledger = dict(frozen_hashes), ledger, cost_ledger
        self.trace_dir, self.config = trace_dir, config
        self.expected_units = expected_units
        self.scorer_fn = scorer_fn

    def verify_frozen_inputs(self) -> None:
        verify_hashes(self.frozen_hashes, self.root)
        if self.expected_units < 1 or self.manifest.get("planned_units") != self.expected_units or len(self.manifest.get("entries") or []) != self.expected_units:
            raise RuntimeError(f"Run manifest must contain exactly {self.expected_units} units.")
        if len({row["run_id"] for row in self.manifest["entries"]}) != self.expected_units:
            raise RuntimeError("Frozen run manifest contains duplicate run IDs.")

    def dry_run(self) -> dict[str, Any]:
        self.verify_frozen_inputs()
        self.ledger.reconcile_interrupted()
        states = self.ledger.latest()
        statuses = {"not_started": 0, "started": 0, "completed": 0, "failed": 0,
                    "interrupted_before_send": 0, "interrupted_result_unknown": 0}
        for entry in self.manifest["entries"]:
            event_name = states.get(str(entry["run_id"]), {}).get("event")
            status = event_name if event_name in statuses else "not_started"
            statuses[status] += 1
        return {
            "status": "dry_run_only",
            "planned": len(self.manifest["entries"]),
            "started": statuses["started"],
            "completed": statuses["completed"],
            "failed": statuses["failed"],
            "interrupted_before_send": statuses["interrupted_before_send"],
            "interrupted_result_unknown": statuses["interrupted_result_unknown"],
            "not_started": statuses["not_started"],
            "provider_sends": sum(row.get("provider_sends", 0) for row in states.values()),
            "database_started": False,
            "entrypoints_invoked": 0,
        }

    def run_pending(self, *, engine: Any, tenant_context: Any, batch_size: int,
                    workflow_fn: Callable[..., dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Start at most ``batch_size`` never-started units in deterministic serial order."""
        self.verify_frozen_inputs()  # Fail before DB-bound Agent work or SDK sends.
        self.cost_ledger.policy.validate_enforcement_config(self.config)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        remaining = self.ledger.resume_candidates(self.manifest["entries"])
        outputs: list[dict[str, Any]] = []
        with bind_isolated_experiment_runtime(
            engine=engine, trace_dir=self.trace_dir, ledger=self.ledger, config=self.config,
        ):
            for entry in remaining[:batch_size]:
                candidate = self.candidates.get(str(entry["case_id"]))
                if candidate is None:
                    raise RuntimeError("Run manifest references a case absent from frozen candidate truth.")
                try:
                    outputs.append(run_workflow_unit(
                        entry, candidate, tenant_context=tenant_context,
                        trace_dir=self.trace_dir, ledger=self.ledger,
                        cost_ledger=self.cost_ledger, experiment_config=self.config,
                        workflow_fn=workflow_fn, scorer_fn=self.scorer_fn,
                    ))
                except CumulativeCostLimitExceeded:
                    raise
                except Exception:
                    # A failure is terminal for this run ID; the schedule may
                    # continue with later, never-started units in the batch.
                    continue
                if outputs[-1]["measurements"]["cost_accounting"].get("reservation_exceeded"):
                    self.ledger.append(str(entry["run_id"]), "cost_accounting_block", halt_next_units=True)
                    break
                if self.cost_ledger.reserved_cny > self.cost_ledger.policy.cumulative_estimate_limit_cny:
                    raise RuntimeError("Cost reservation ledger exceeded its configured estimate limit.")
        return outputs


def load_isolated_candidate_database(admin_url_text: str, *, expected_csv_hashes: Mapping[str, str] | None = None,
                                     dataset_dir: Path | None = None) -> tuple[Any, str, dict[str, Any]]:
    """Load complete public Olist v2 facts into a fresh loopback-only database.

    The returned database is intentionally retained for the caller's batch and
    must be disposed with ``drop_isolated_candidate_database`` in ``finally``.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    dataset_dir = dataset_dir or root / "kagglehub/datasets/olistbr/brazilian-ecommerce/versions/2"
    files = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "order_payments": "olist_order_payments_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "order_reviews": "olist_order_reviews_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "product_category_name_translation": "product_category_name_translation.csv",
    }
    source_hashes: dict[str, str] = {}
    for key, filename in files.items():
        path = dataset_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Olist source is missing: {filename}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = (expected_csv_hashes or {}).get(filename)
        if expected is not None and digest != expected:
            raise RuntimeError(f"Frozen Olist source hash mismatch: {filename}")
        source_hashes[filename] = digest

    admin_url = make_url(admin_url_text)
    if admin_url.host not in {"127.0.0.1", "localhost", "::1"} or admin_url.port not in {15433, 15437} or admin_url.database not in {"postgres", "template1"}:
        raise RuntimeError("Candidate facts may only be loaded via a loopback PostgreSQL maintenance database.")
    database_name = f"provider_candidate_{uuid.uuid4().hex[:12]}"
    admin_engine = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True,
                                 connect_args={"connect_timeout": 5})
    engine = None
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{database_name}"'))
        target_url = admin_url.set(database=database_name)
        engine = create_engine(target_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        from app.data.bootstrap import ensure_schemas, load_raw_tables
        ensure_schemas(engine)
        load_raw_tables(dataset_dir, target_engine=engine)

        helper_path = root / "docs/testing/provider_critic_candidate_db_reconcile_20261013/run_acceptance.py"
        spec = importlib.util.spec_from_file_location("provider_candidate_db_loader_helpers", helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the frozen candidate migration helper.")
        helpers = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = helpers
        spec.loader.exec_module(helpers)
        with engine.begin() as conn:
            for path, name in (
                (root / "migrations/versions/20260723_05_olist_payment_fulfillment.py", "olist_mart_20260723"),
                (root / "migrations/versions/20260725_06_olist_phase6_marts.py", "olist_views_20260725"),
                (root / "migrations/versions/20261006_54_unify_gmv_formula.py", "olist_gmv_formula_20261006_54"),
            ):
                helpers.apply_migration(conn, path, name)
        with engine.connect() as conn:
            coverage = {
                table: int(conn.execute(text(f"SELECT COUNT(*) FROM raw.{table}")).scalar_one())
                for table in files
            }
            date_bounds = conn.execute(text(
                "SELECT MIN(order_purchase_timestamp)::date, MAX(order_purchase_timestamp)::date "
                "FROM raw.orders WHERE order_purchase_timestamp IS NOT NULL"
            )).one()
        return engine, database_name, {
            "host": "loopback_only", "database_name": database_name,
            "load_policy": "all nine Olist v2 CSV files, no date clipping",
            "source_csv_sha256": source_hashes,
            "raw_table_rows": coverage,
            "orders_date_bounds": [str(value) for value in date_bounds],
            "migrations": ["20260723_05", "20260725_06", "20261006_54"],
        }
    except BaseException:
        if engine is not None:
            engine.dispose()
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
        raise
    finally:
        admin_engine.dispose()


def drop_isolated_candidate_database(admin_url_text: str, database_name: str, engine: Any) -> None:
    """Dispose and drop only a database name generated by this loader."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    if not database_name.startswith("provider_candidate_"):
        raise ValueError("Refusing to drop a database not created by the candidate loader.")
    engine.dispose()
    admin_url = make_url(admin_url_text)
    if admin_url.host not in {"127.0.0.1", "localhost", "::1"} or admin_url.port not in {15433, 15437}:
        raise RuntimeError("Refusing to drop a database through a non-loopback server.")
    admin_engine = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    try:
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)'))
    finally:
        admin_engine.dispose()
