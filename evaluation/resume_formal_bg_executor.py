"""Explicit, append-only runner for the 60-case formal B/G schedule.

Importing this module is inert. Execution needs an explicit Provider budget,
an isolated PostgreSQL admin URL, verified case qualification, and an
observation adapter. Unknown scoring fields remain manual review.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from evaluation.provider_candidate_batch import (
    CostPolicy, CumulativeCostLimitExceeded, EstimatedCostLedger, ExperimentConfig,
    RunLedger, _reconcile_cost_accounting, bind_isolated_experiment_runtime,
    drop_isolated_candidate_database, load_isolated_candidate_database,
    token_usage_snapshot, verify_hashes,
)
from evaluation.resume_formal_bg_v1 import (
    FORMAL_ARMS, candidate_operation_contract, compare_operation_contracts,
    score_formal_run, validate_paired_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_HISTORY_LEDGER_NAMES = {
    "events.jsonl", "run_events.jsonl", "cost_reservations.jsonl", "ledger.jsonl", "run_ledger.jsonl",
}


def _is_history_ledger(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in _HISTORY_LEDGER_NAMES
        or (any(token in name for token in ("event", "reservation")) and path.suffix.lower() == ".jsonl")
        or ("ledger" in name and path.suffix.lower() in {".json", ".jsonl"})
        or name.endswith((".run_record.json", ".failure_record.json"))
    )


def _history_rows(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, Mapping):
        rows: list[Mapping[str, Any]] = [document]
        # Cost snapshots commonly wrap append-only reservations in a list.
        for key in ("events", "reservations", "records", "entries"):
            nested = document.get(key)
            if isinstance(nested, list):
                rows.extend(item for item in nested if isinstance(item, Mapping))
        return rows
    if isinstance(document, list):
        return [item for item in document if isinstance(item, Mapping)]
    return []


def historical_run_id_occurrences(
    run_ids: Sequence[str], roots: Sequence[Path], *, exclude_paths: Sequence[Path] = (),
) -> dict[str, list[dict[str, str]]]:
    """Find run IDs already present in append-only ledgers and terminal records.

    Malformed recognized ledgers fail closed: an unreadable history cannot be
    treated as proof that a run ID is unused. Result JSON is limited to
    explicit run/failure records so frozen manifests and case files are not
    misclassified as executions.
    """
    wanted = {str(item) for item in run_ids}
    found: dict[str, list[dict[str, str]]] = {run_id: [] for run_id in wanted}
    excluded = {Path(path).resolve() for path in exclude_paths}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates: set[Path] = set()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.resolve() in excluded:
                continue
            if _is_history_ledger(path):
                candidates.add(path)
        for path in sorted(candidates):
            try:
                if path.suffix == ".jsonl":
                    rows = []
                    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                        if not line.strip():
                            continue
                        try:
                            rows.extend(_history_rows(json.loads(line)))
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Malformed run-history ledger {path}:{line_no}") from exc
                else:
                    rows = _history_rows(json.loads(path.read_text(encoding="utf-8")))
            except OSError as exc:
                raise ValueError(f"Unreadable run-history record {path}") from exc
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                run_id = str(row.get("run_id") or "")
                if run_id in wanted:
                    found[run_id].append({
                        "path": str(path.resolve()),
                        "event": str(row.get("event") or row.get("lifecycle_status") or "record"),
                    })
    return {key: value for key, value in found.items() if value}


def assert_run_ids_not_previously_used(
    run_ids: Sequence[str], roots: Sequence[Path], *, active_ledger_path: Path | None = None,
    active_cost_ledger_path: Path | None = None, resumable_reserved_ids: Sequence[str] = (),
) -> dict[str, list[dict[str, str]]]:
    """Refuse reuse across all supplied historical run and cost ledgers.

    The only allowed existing occurrence is an untouched cost reservation for
    the same ID in the active ledger itself. A start/send/terminal event or an
    occurrence in any other ledger makes the ID unavailable.
    """
    excluded = [path for path in (active_ledger_path, active_cost_ledger_path) if path is not None]
    occurrences = historical_run_id_occurrences(run_ids, roots, exclude_paths=excluded)
    active_path = Path(active_ledger_path).resolve() if active_ledger_path else None
    resumable = {str(item) for item in resumable_reserved_ids}
    collisions: dict[str, list[dict[str, str]]] = {}
    for run_id, rows in occurrences.items():
        if run_id in resumable and active_path is not None and all(
            Path(row["path"]).resolve() == active_path and row["event"] == "cost_reserved"
            for row in rows
        ):
            continue
        collisions[run_id] = rows
    if collisions:
        raise RuntimeError("Run IDs already exist in historical ledgers/records: " + json.dumps(collisions, ensure_ascii=False))
    return occurrences


def runtime_freeze_hashes(root: Path) -> dict[str, str]:
    """Hash every first-party runtime input that can affect an experiment.

    Tests, historical evidence and run outputs are deliberately excluded; the
    returned mapping is intended to be embedded into a new frozen manifest.
    """
    root = Path(root)
    source_roots = (
        root / "app", root / "evaluation", root / "config",
        root / "migrations", root / "docs" / "knowledge",
    )
    frozen_suffixes = {
        ".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt",
        ".ini", ".prompt", ".jinja", ".jinja2", ".j2",
    }
    paths = {
        path for source_root in source_roots if source_root.exists()
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in frozen_suffixes
        # Never include local account or secret material in a freeze manifest.
        and path.name.lower() not in {".env", "secrets.json", "credentials.json"}
    }
    paths.update(path for name in (
        "pyproject.toml", "requirements.txt", "requirements-dev.txt", "environment.yml", "environment.yaml",
        "alembic.ini", ".env.example",
    )
                 if (path := root / name).is_file())
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def assert_runtime_freeze_complete(root: Path, frozen_hashes: Mapping[str, str]) -> dict[str, str]:
    """Require a new frozen manifest to cover all current runtime inputs."""
    current = runtime_freeze_hashes(root)
    missing = sorted(set(current) - set(frozen_hashes))
    if missing:
        raise ValueError(
            "Frozen manifest omits runtime code/config/prompt inputs: "
            + ", ".join(missing[:12])
            + (f" (and {len(missing) - 12} more)" if len(missing) > 12 else "")
        )
    mismatches = sorted(key for key, digest in current.items() if frozen_hashes.get(key) != digest)
    if mismatches:
        raise ValueError(
            "Frozen manifest runtime hashes do not match current code/config/prompts: "
            + ", ".join(mismatches[:12])
        )
    return current


def _tenant_identity(context: Any) -> tuple[str, str, str, str]:
    return tuple(str(getattr(context, key, "") or "") for key in (
        "tenant_id", "user_id", "role", "scope_type",
    ))


def _case_identity(case: Mapping[str, Any]) -> tuple[str, str, str, str]:
    identity = case.get("same_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"Case {case.get('case_id')} has no frozen same_identity contract")
    normalized = tuple(str(identity.get(key) or "") for key in (
        "tenant_id", "user_id", "role", "scope_type",
    ))
    if any(not value for value in normalized):
        raise ValueError(f"Case {case.get('case_id')} has an incomplete same_identity contract")
    return normalized


def resolve_formal_tenant_context(cases: Sequence[Mapping[str, Any]], requested: Any = None) -> Any:
    """Resolve the one identity frozen for the selected experiment units."""
    identities = {_case_identity(case) for case in cases}
    if len(identities) != 1:
        raise ValueError("Selected B/G cases do not share the same frozen identity")
    expected = next(iter(identities))
    if requested is not None:
        if _tenant_identity(requested) != expected:
            raise ValueError("Runtime tenant context does not match the selected cases' frozen identity")
        return requested
    from app.tenancy import TenantContext
    tenant_id, user_id, role, scope_type = expected
    return TenantContext(tenant_id=tenant_id, user_id=user_id, role=role, scope_type=scope_type)


def validate_formal_budget(
    *, planned_units: int, max_units: int, estimate_limit_cny: float,
    provider_authorization_cny: float, cost_policy: CostPolicy,
    experiment_config: ExperimentConfig, reserved_cny: float,
) -> dict[str, Any]:
    """Validate budget before the caller starts a database or a workflow."""
    if isinstance(max_units, bool) or max_units < 0 or max_units > planned_units:
        raise ValueError("max_units must be explicit and within the frozen manifest size")
    if round(float(estimate_limit_cny), 8) != round(float(provider_authorization_cny), 8):
        raise ValueError("Estimate limit and Provider authorization must match exactly")
    if estimate_limit_cny <= 0 or provider_authorization_cny <= 0:
        raise ValueError("Provider authorization must be explicit and positive")
    if estimate_limit_cny > cost_policy.cumulative_estimate_limit_cny + 1e-12:
        raise ValueError(
            "Requested estimate authorization exceeds the append-only cost ledger's configured cumulative limit; "
            "update the ledger policy explicitly before execution."
        )


    cost_policy.validate_enforcement_config(experiment_config)
    estimate = cost_policy.estimate(max_units)
    remaining = float(estimate_limit_cny) - float(reserved_cny)
    if estimate["estimated_cny"] > remaining + 1e-12:
        raise CumulativeCostLimitExceeded("Selected units exceed remaining cumulative estimate authorization.")
    return {
        "selected_units": max_units,
        "new_estimated_reservation_cny": estimate["estimated_cny"],
        "existing_reservations_cny": round(float(reserved_cny), 8),
        "cumulative_estimate_limit_cny": float(estimate_limit_cny),
        "remaining_before_selection_cny": round(remaining, 8),
        "provider_authorization_cny": float(provider_authorization_cny),
        "is_account_hard_cap": False,
        "cost_policy": estimate,
    }


def select_pending_manifest_entries(
    entries: Sequence[Mapping[str, Any]],
    latest: Mapping[str, Mapping[str, Any]],
    *,
    max_units: int,
    run_ids: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Select only never-started frozen units, optionally by explicit run ID.

    Explicit selection supports bounded repair retests of one arm while the
    manifest remains paired and auditable. It never changes pair definitions
    or makes an unselected counterpart look executed.
    """
    if isinstance(max_units, bool) or max_units < 1 or max_units > len(entries):
        raise ValueError(f"max_units must be explicitly set from 1 through {len(entries)}")
    pending = [
        row for row in entries
        if str(row.get("run_id") or "") not in latest
        or latest[str(row.get("run_id") or "")].get("event") == "cost_reserved"
    ]
    if run_ids is None:
        return pending[:max_units]
    requested = [str(item) for item in run_ids]
    if len(requested) != max_units or len(set(requested)) != len(requested):
        raise ValueError("Explicit run_ids must be unique and match max_units exactly")
    by_id = {str(row.get("run_id") or ""): row for row in entries}
    if any(run_id not in by_id for run_id in requested):
        raise ValueError("Explicit run_ids must belong to the frozen manifest")
    pending_ids = {str(row.get("run_id") or "") for row in pending}
    started = [run_id for run_id in requested if run_id not in pending_ids]
    if started:
        raise RuntimeError("Explicit run_ids include started or terminal units; retry is forbidden")
    requested_set = set(requested)
    selected = [row for row in entries if str(row.get("run_id") or "") in requested_set]
    if [str(row.get("run_id") or "") for row in selected] != requested:
        raise ValueError("Explicit run_ids must follow frozen manifest order")
    return selected


class FormalBGBatchExecutor:
    def __init__(
        self, *, root: Path, cases: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
        frozen_hashes: Mapping[str, str], qualification: Mapping[str, Any],
        ledger: RunLedger, cost_ledger: EstimatedCostLedger, trace_dir: Path,
        config: ExperimentConfig, historical_ledger_roots: Sequence[Path] | None = None,
        arm_execution_mode: str = "bare_tool_baseline_vs_governed_workflow",
        record_arm_labels: Mapping[str, str] | None = None,
        scoring_arm_by_execution_arm: Mapping[str, str] | None = None,
    ):
        self.root = root
        self.cases = {str(row["case_id"]): dict(row) for row in cases}
        self.manifest = dict(manifest)
        self.frozen_hashes = dict(frozen_hashes)
        self.qualification = dict(qualification)
        self.ledger, self.cost_ledger = ledger, cost_ledger
        self.trace_dir, self.config = trace_dir, config
        if arm_execution_mode not in {
            "bare_tool_baseline_vs_governed_workflow",
            "critic_off_vs_full_workflow",
            "critic_adaptive_vs_full_workflow",
        }:
            raise ValueError(f"Unsupported formal arm execution mode: {arm_execution_mode}")
        self.arm_execution_mode = arm_execution_mode
        self.record_arm_labels = {str(key): str(value) for key, value in (record_arm_labels or {}).items()}
        self.scoring_arm_by_execution_arm = {
            str(key): str(value) for key, value in (scoring_arm_by_execution_arm or {}).items()
        }
        unsupported_labels = set(self.record_arm_labels) - set(FORMAL_ARMS)
        unsupported_scoring_keys = set(self.scoring_arm_by_execution_arm) - set(FORMAL_ARMS)
        unsupported_scoring_arms = set(self.scoring_arm_by_execution_arm.values()) - set(FORMAL_ARMS)
        if unsupported_labels or unsupported_scoring_keys or unsupported_scoring_arms:
            raise ValueError("Arm-label/scoring overrides must map formal execution arms to formal scoring arms")
        self.historical_ledger_roots = (
            list(historical_ledger_roots)
            if historical_ledger_roots is not None
            else [self.root / "docs" / "testing"]
        )

    def verify_inputs(self) -> dict[str, Any]:
        verify_hashes(self.frozen_hashes, self.root)
        current_runtime_hashes = assert_runtime_freeze_complete(self.root, self.frozen_hashes)
        schedule = validate_paired_manifest(self.manifest, list(self.cases.values()))
        # A formal cohort may be a deliberately pre-registered subset of the
        # original 60 public development cases.  Keep recognizing the source
        # by an explicit selected-cases filename rather than hard-coding the
        # historical 12-case cohort; the selected file is still hashed and
        # compared row-for-row below.
        candidate_key = next((key for key in self.frozen_hashes
                              if Path(key).name in {
                                  "candidate_cases_60.json", "selected_cases_12.json",
                                  "selected_cases_23.json",
                              }), None)
        qualification_key = next((key for key in self.frozen_hashes
                                  if Path(key).name == "offline_candidate_qualification_v2.json"), None)
        if candidate_key is None or qualification_key is None:
            raise ValueError("Frozen hashes must include a candidate set and offline_candidate_qualification_v2.json")
        candidate_hash = self.frozen_hashes[candidate_key]
        qualification_hash = self.frozen_hashes[qualification_key]
        frozen_candidate = json.loads((self.root / candidate_key).read_text(encoding="utf-8"))
        frozen_qualification = json.loads((self.root / qualification_key).read_text(encoding="utf-8"))
        source_rows = frozen_candidate.get("cases", [])
        source_by_id = {str(row.get("case_id")): row for row in source_rows}
        selected_by_id = {str(row["case_id"]): row for row in self.cases.values()}
        if len(source_by_id) != len(source_rows) or any(
            case_id not in source_by_id
            or _canonical_sha256(source_by_id[case_id]) != _canonical_sha256(case)
            for case_id, case in selected_by_id.items()
        ):
            raise ValueError("In-memory candidate rows do not match the hashed candidate source")
        if _canonical_sha256(frozen_qualification) != _canonical_sha256(self.qualification):
            raise ValueError("In-memory qualification does not match the hashed qualification file")
        if self.manifest.get("candidate_sha256") not in (None, candidate_hash):
            raise ValueError("Run manifest candidate hash does not match the frozen candidate file")
        if self.manifest.get("qualification_sha256") not in (None, qualification_hash):
            raise ValueError("Run manifest qualification hash does not match the frozen qualification file")
        qualification_case_count = int(self.qualification.get("candidate_count") or 0)
        if qualification_case_count != len(source_rows):
            raise ValueError("Qualification candidate count does not match the hashed candidate source")
        if self.qualification.get("candidate_sha256") != candidate_hash:
            raise ValueError("Qualification artifact was produced for a different candidate file hash")
        contract_rows: list[dict[str, Any]] = []
        contract_errors: list[dict[str, Any]] = []
        for case_id, case in sorted(self.cases.items()):
            b_contract = candidate_operation_contract(case, arm="B")
            g_contract = candidate_operation_contract(case, arm="G")
            comparison = compare_operation_contracts(case, b_contract=b_contract, g_contract=g_contract)
            routes_declared = bool(b_contract.get("semantic_route")) and bool(g_contract.get("semantic_route"))
            row = {
                "case_id": case_id,
                "status": comparison["status"] if routes_declared else "blocked_route_unresolved",
                "b_route": b_contract.get("semantic_route"),
                "g_route": g_contract.get("semantic_route"),
                "mismatches": comparison["mismatches"],
            }
            contract_rows.append(row)
            if row["status"] != "equivalent":
                contract_errors.append(row)
        if contract_errors:
            raise ValueError("Formal B/G candidate contract mismatch: " + json.dumps(contract_errors, ensure_ascii=False))
        identity_by_case = {case_id: _case_identity(case) for case_id, case in self.cases.items()}
        if len(set(identity_by_case.values())) != 1:
            raise ValueError("Selected cases do not share one identical frozen tenant identity")
        if isinstance(self.qualification.get("eligible_case_ids"), list):
            qualified_ids = {str(item) for item in self.qualification["eligible_case_ids"]}
        else:
            qualified_ids = {
                str(row.get("case_id")) for row in self.qualification.get("cases", [])
                if row.get("qualification_status") == "eligible"
            }
        manifest_case_ids = {str(row["case_id"]) for row in self.manifest["entries"]}
        if manifest_case_ids != set(self.cases):
            raise ValueError("Formal manifest case IDs do not match the selected verified candidate set")
        selected_qualified_ids = set(self.cases).intersection(qualified_ids)
        selected_blocked_ids = set(self.cases) - qualified_ids
        return {
            **schedule,
            "runtime_inventory_file_count": len(current_runtime_hashes),
            "runtime_inventory_sha256": _canonical_sha256(current_runtime_hashes),
            "candidate_sha256": candidate_hash,
            "qualified_case_count": len(qualified_ids),
            "candidate_source_count": len(source_rows),
            "selected_qualified_case_count": len(selected_qualified_ids),
            "selected_blocked_case_ids": sorted(selected_blocked_ids),
            "blocked_case_count": len(source_rows) - len(qualified_ids),
            "qualified_case_ids": sorted(qualified_ids),
            "semantic_contract_pairs_checked": len(contract_rows),
            "semantic_contract_pairs_equivalent": sum(row["status"] == "equivalent" for row in contract_rows),
            "semantic_contract_rows": contract_rows,
            "same_identity_contract": list(next(iter(identity_by_case.values()))),
            "same_identity_case_count": len(identity_by_case),
        }

    def dry_run(self, *, cost_policy: CostPolicy) -> dict[str, Any]:
        verified = self.verify_inputs()
        cost_policy.validate_enforcement_config(self.config)
        states = self.ledger.latest()
        counts = {"not_started": 0, "cost_reserved": 0, "started": 0, "scored_without_terminal_completion": 0,
                  "completed": 0, "failed": 0, "interrupted_before_send": 0,
                  "interrupted_result_unknown": 0, "unknown_ledger_event": 0}
        for row in self.manifest["entries"]:
            event = states.get(row["run_id"], {}).get("event")
            if event == "scored":
                status = "scored_without_terminal_completion"
            elif event in counts:
                status = str(event)
            elif event is None:
                status = "not_started"
            else:
                status = "unknown_ledger_event"
            counts[status] += 1
        return {
            "status": "dry_run_only", **verified,
            "lifecycle_counts": counts,
            "denominator_contract": {
                "planned_unit_denominator": len(self.manifest["entries"]),
                "failed_timeout_interrupted_and_sent_unknown_usage_remain_in_denominator": True,
                "unknown_usage_is_not_zero": True,
                "automatic_retry_or_case_replacement": False,
                "complete_pair_count_reported_separately": True,
            },
            "planned_cost_estimate": cost_policy.estimate(len(self.manifest["entries"])),
            "existing_cost_reservation_cny": round(self.cost_ledger.reserved_cny, 8),
            "database_started": False,
            "provider_sends": sum(row.get("event") == "sdk_send_started" for row in self.ledger.events),
        }

    def execute(
        self, *, max_units: int, estimate_limit_cny: float, provider_authorization_cny: float,
        admin_url_text: str, expected_csv_hashes: Mapping[str, str],
        observation_builder: Callable[[Mapping[str, Any], str, Mapping[str, Any]], Mapping[str, Any]],
        fault_context_factory: Callable[[Mapping[str, Any], str], Any] | None = None,
        tenant_context: Any = None, run_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Explicit live entry. It refuses before DB startup unless all preconditions pass."""
        verified = self.verify_inputs()
        manifest_unit_count = len(self.manifest.get("entries") or [])
        if isinstance(max_units, bool) or max_units < 1 or max_units > manifest_unit_count:
            raise ValueError(f"max_units must be explicitly set from 1 through {manifest_unit_count}")
        states = self.ledger.latest()
        selected = select_pending_manifest_entries(
            self.manifest["entries"], states, max_units=max_units, run_ids=run_ids,
        )
        if not selected:
            return []
        # Formal execution always scans the repository's complete historical
        # evidence tree. The injectable roots are additive (for isolated
        # smoke histories), never a way to narrow the duplicate-ID search.
        history_roots = list(dict.fromkeys([
            *(Path(item) for item in self.historical_ledger_roots),
            self.root / "docs" / "testing",
        ]))
        assert_run_ids_not_previously_used(
            [str(row["run_id"]) for row in selected],
            history_roots,
            active_ledger_path=self.ledger.path,
            active_cost_ledger_path=self.cost_ledger.path,
        )
        cost_policy = self.cost_ledger.policy
        reserved_run_ids = {
            str(row.get("run_id")) for row in self.cost_ledger.reservations
            if row.get("event") is None and row.get("reserved_cny") is not None
        }
        selected_reserved = [row for row in selected if str(row["run_id"]) in reserved_run_ids]
        selected_new = [row for row in selected if str(row["run_id"]) not in reserved_run_ids]
        for row in selected_reserved:
            latest = states.get(str(row["run_id"]), {})
            if latest.get("event") != "cost_reserved":
                raise RuntimeError("A cost-reserved run has already entered execution and cannot be resumed.")
            run_events = [item for item in self.ledger.events if item.get("run_id") == row["run_id"]]
            if any(item.get("event") == "sdk_send_started" for item in run_events):
                raise RuntimeError("A reserved run has an SDK send but no start state; result is unknown and retry is forbidden.")
        budget = validate_formal_budget(
            planned_units=manifest_unit_count, max_units=len(selected_new),
            estimate_limit_cny=estimate_limit_cny,
            provider_authorization_cny=provider_authorization_cny,
            cost_policy=cost_policy, experiment_config=self.config,
            reserved_cny=self.cost_ledger.reserved_cny,
        )
        qualified_ids = set(verified["qualified_case_ids"])
        blocked = [row["case_id"] for row in selected if row["case_id"] not in qualified_ids]
        if blocked:
            raise RuntimeError("Selected candidates have not passed data/query qualification: " + ", ".join(sorted(set(blocked))))
        if any(self.cases[row["case_id"]].get("stratum") == "fault_or_budget" for row in selected) and fault_context_factory is None:
            raise RuntimeError("Fault/budget candidate execution requires its verified fault-injection context factory.")
        tenant_context = resolve_formal_tenant_context(
            [self.cases[row["case_id"]] for row in selected], tenant_context,
        )

        # Both the bare-tool arm and the governed workflow persist traces into
        # this directory.  The workflow's file storage creates its own parent,
        # but the bare-tool arm writes directly with Path.write_text; create the
        # shared output directory before starting the database or any model call.
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        engine = None
        database_name = None
        outputs: list[dict[str, Any]] = []
        try:
            engine, database_name, _coverage = load_isolated_candidate_database(
                admin_url_text, expected_csv_hashes=expected_csv_hashes,
            )
            with bind_isolated_experiment_runtime(
                engine=engine, trace_dir=self.trace_dir, ledger=self.ledger, config=self.config,
            ):
                for entry in selected:
                    case = self.cases[entry["case_id"]]
                    outputs.append(self._run_one(
                        entry, case, tenant_context=tenant_context,
                        observation_builder=observation_builder,
                        fault_context_factory=fault_context_factory,
                    ))
            return outputs
        finally:
            if engine is not None and database_name:
                engine.dispose()
                drop_isolated_candidate_database(admin_url_text, database_name, engine)
            if outputs:
                try:
                    from evaluation.resume_bg_companion_audit import summarize_companion_quality_records
                    companion_summary = summarize_companion_quality_records(outputs)
                    companion_summary.update({
                        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "batch_run_ids": [str(item.get("run_id")) for item in outputs],
                        "summary_kind": "this_execute_call_only",
                    })
                    summary_path = self.trace_dir / f"companion_quality_summary_{uuid4()}.json"
                    summary_path.write_text(
                        json.dumps(companion_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
                    )
                except Exception as summary_error:
                    try:
                        error_path = self.trace_dir / "companion_quality_summary_errors.jsonl"
                        with error_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "error_type": type(summary_error).__name__,
                                "run_ids": [str(item.get("run_id")) for item in outputs],
                            }, ensure_ascii=False) + "\n")
                    except OSError:
                        pass

    def _run_one(
        self, entry: Mapping[str, Any], case: Mapping[str, Any], *, tenant_context: Any,
        observation_builder: Callable[[Mapping[str, Any], str, Mapping[str, Any]], Mapping[str, Any]],
        fault_context_factory: Callable[[Mapping[str, Any], str], Any] | None,
    ) -> dict[str, Any]:
        from app.graph.critic_review_mode import CriticLLMReviewMode
        from app.graph.critic_llm_thinking_mode import CriticLLMThinkingMode
        from app.graph import workflow

        run_id, arm = str(entry["run_id"]), str(entry["arm"])
        if arm not in FORMAL_ARMS:
            raise ValueError("Formal manifest has an invalid arm")
        record_arm = self.record_arm_labels.get(arm, arm)
        evaluation_arm = self.scoring_arm_by_execution_arm.get(arm, arm)
        prior = self.ledger.latest().get(run_id)
        if prior and prior.get("event") != "cost_reserved":
            raise RuntimeError("A started or terminal run ID cannot be retried")
        if prior and prior.get("provider_sends", 0):
            raise RuntimeError("A cost-reserved run already has an SDK send; retry is forbidden")
        # Keep _run_one safe when invoked directly by a focused runner as well
        # as through execute().  In particular, B writes the trace itself and
        # does not pass through app.harness.storage._trace_dir().
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        reserve = self.cost_ledger.reserve(run_id)
        self.ledger.append(run_id, "cost_reserved", reserved_cny=reserve)
        self.ledger.append(run_id, "started", case_id=entry["case_id"], arm=arm, replicate=entry["replicate"])
        start = time.perf_counter()
        trace_path = self.trace_dir / f"{run_id}.json"
        try:
            from evaluation.resume_bg_tool_surface import build_evaluation_tool_surface
            evaluation_tool_surface = build_evaluation_tool_surface(case, tenant_context)
            call_config = self.config.build_llm_call_config(workflow.get_settings())
            injection = fault_context_factory(case, arm) if fault_context_factory is not None else nullcontext()
            with injection:
                if arm == "B" and self.arm_execution_mode == "bare_tool_baseline_vs_governed_workflow":
                    from evaluation.resume_bare_tool_loop import run_bare_tool_loop
                    state = run_bare_tool_loop(
                        str(case["user_question"]), tenant_context=tenant_context, run_id=run_id,
                        max_rounds=self.config.max_graph_steps, max_tool_calls=self.config.max_tool_calls,
                        max_latency_ms=self.config.max_latency_ms, max_task_tokens=self.config.max_task_tokens,
                        max_cost_cny=self.config.max_cost_cny_per_workflow, llm_call_config=call_config,
                        evaluation_tool_surface=evaluation_tool_surface,
                    )
                    trace = {"trace_id": run_id, "state_snapshot": state,
                             "workflow_result": {"answer": (state.get("final") or {}).get("answer", ""),
                                                  "presentation": (state.get("final_display") or {}).get("presentation")}}
                    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                else:
                    critic_mode = (
                        CriticLLMReviewMode.CRITIC_LLM_REVIEW_OFF
                        if arm == "B" and self.arm_execution_mode == "critic_off_vs_full_workflow"
                        else CriticLLMReviewMode.ADAPTIVE
                        if arm == "B" and self.arm_execution_mode == "critic_adaptive_vs_full_workflow"
                        else CriticLLMReviewMode.FULL
                    )
                    workflow.run_chat_workflow(
                        str(case["user_question"]), tenant_context=tenant_context, memory_enabled=False,
                        trace_id=run_id, _critic_llm_review_mode=critic_mode,
                        _critic_llm_thinking_mode=CriticLLMThinkingMode.NON_THINKING_EXPERIMENT,
                        _llm_call_config=call_config,
                        _evaluation_tool_surface=evaluation_tool_surface,
                    )
                    if not trace_path.is_file():
                        raise RuntimeError("Governed workflow completed without a persisted trace")
                    trace = json.loads(trace_path.read_text(encoding="utf-8"))
            from evaluation.resume_bg_observation import extract_formal_observation
            observation = extract_formal_observation(case, evaluation_arm, trace)
            # Keep explicitly supplied, non-conflicting metadata adapters for
            # compatibility, but never let them overwrite extracted task facts.
            supplemental = dict(observation_builder(case, evaluation_arm, trace)) if observation_builder else {}
            for key, value in supplemental.items():
                if key not in observation:
                    observation[key] = value
            score = score_formal_run(case, observation=observation, lifecycle_status="completed", arm=evaluation_arm)
            # Store granular per-object evidence/authorization/call facts with
            # each new run record. This is an evaluation projection only and
            # does not change scoring or historical records.
            from evaluation.resume_bg_companion_audit import build_companion_quality_audit
            try:
                companion_audit = build_companion_quality_audit(
                    case, evaluation_arm, trace, observation, expected_run_id=run_id,
                )
                companion_audit_error = None
            except Exception as audit_error:
                # An audit projection must never turn a completed business run
                # into a failed run, but its own failure must remain visible.
                companion_audit = None
                companion_audit_error = {
                    "error_type": type(audit_error).__name__, "error": str(audit_error),
                }
            events = [row for row in self.ledger.events if row.get("run_id") == run_id]
            usage = token_usage_snapshot(trace, events, run_id=run_id)
            accounting = self.cost_ledger.record_usage(run_id, usage)
            reserve_row = next(row for row in self.cost_ledger.reservations
                               if row.get("run_id") == run_id and row.get("reserved_cny") is not None)
            _reconcile_cost_accounting(usage, accounting, self.cost_ledger.policy,
                                       expected_reservation_cny=float(reserve_row["reserved_cny"]))
            record = {
                "run_id": run_id, "case_id": entry["case_id"], "arm": record_arm,
                "execution_arm": arm, "evaluation_arm": evaluation_arm,
                "replicate": entry["replicate"], "trace": trace, "observation": observation,
                "score": score, "lifecycle_status": "completed",
                "cohort_id": self.manifest.get("experiment_id") or _canonical_sha256(self.manifest),
                "manifest_sha256": _canonical_sha256(self.manifest),
                "scorer_version": score.get("scorer_version"),
                "companion_quality_audit": companion_audit,
                "companion_quality_audit_error": companion_audit_error,
                "measurements": {
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
                    "sdk_sends": sum(row.get("event") == "sdk_send_started" for row in events),
                    "token_usage": usage, "cost_accounting": accounting,
                },
            }
            (self.trace_dir / f"{run_id}.run_record.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
            )
            self.ledger.append(run_id, "scored", scorer_version=score["scorer_version"],
                               task_acceptance_status=score["task_acceptance_status"],
                               manual_review_required=score["manual_review_required"])
            self.ledger.append(run_id, "completed", provider_sends=record["measurements"]["sdk_sends"])
            return record
        except BaseException as exc:
            provider_sends = sum(
                row.get("run_id") == run_id and row.get("event") == "sdk_send_started"
                for row in self.ledger.events
            )
            failure_record = {
                "run_id": run_id,
                "case_id": entry.get("case_id"),
                "arm": record_arm,
                "execution_arm": arm,
                "evaluation_arm": evaluation_arm,
                "replicate": entry.get("replicate"),
                "lifecycle_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "provider_sends": provider_sends,
                "trace_present": trace_path.is_file(),
                "retry_forbidden": True,
            }
            if trace_path.is_file():
                try:
                    failed_trace = json.loads(trace_path.read_text(encoding="utf-8"))
                    from evaluation.resume_bg_observation import extract_formal_observation
                    from evaluation.resume_bg_companion_audit import build_companion_quality_audit
                    failed_observation = extract_formal_observation(case, evaluation_arm, failed_trace)
                    failure_record["companion_quality_audit"] = build_companion_quality_audit(
                        case, evaluation_arm, failed_trace, failed_observation, expected_run_id=run_id,
                    )
                except Exception as audit_error:
                    # The execution failure remains primary; projection failure
                    # is preserved separately and does not fabricate facts.
                    failure_record["companion_quality_audit_error"] = {
                        "error_type": type(audit_error).__name__, "error": str(audit_error),
                    }
            failure_record_path = self.trace_dir / f"{run_id}.failure_record.json"
            try:
                failure_record_path.write_text(
                    json.dumps(failure_record, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            except OSError as write_error:
                # Preserve the original execution exception; report the
                # secondary persistence failure in the append-only ledger.
                self.ledger.append(
                    run_id, "failure_record_write_failed",
                    error_type=type(write_error).__name__,
                )
            self.ledger.append(run_id, "failed", error_type=type(exc).__name__, retry_forbidden=True,
                               trace_present=trace_path.is_file(), provider_sends=provider_sends,
                               failure_record_present=failure_record_path.is_file())
            raise
