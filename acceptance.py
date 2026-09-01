from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from production_contract import ProductionContract, ProductionContractError, load_resolved_config, validate_production_contract


class AcceptanceError(RuntimeError):
    """Raised when acceptance evidence is malformed or cannot be evaluated safely."""


class AcceptanceStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"


REQUIRED_SCENARIOS: tuple[str, ...] = (
    "backend_connection_refused",
    "generation_hang",
    "backend_500",
    "stream_disconnect",
    "queue_full",
    "concurrent_discord_requests",
    "backend_process_kill",
    "restart_failure",
    "gpu_unavailable",
    "cpu_fallback",
    "oversized_context",
    "repeated_compaction",
    "large_text_attachment",
    "unsupported_attachment",
    "bot_restart_conversation_continuity",
)

SCENARIO_BASELINE_EVENTS: dict[str, tuple[str, ...]] = {
    "backend_connection_refused": ("generation.failure",),
    "generation_hang": ("generation.timeout",),
    "backend_500": ("generation.failure",),
    "stream_disconnect": ("generation.failure",),
    "queue_full": ("queue.rejected",),
    "concurrent_discord_requests": ("queue.admitted",),
    "backend_process_kill": ("watchdog.failure", "watchdog.restart", "probe.success"),
    "restart_failure": ("watchdog.restart", "watchdog.failure"),
    "gpu_unavailable": ("watchdog.failure",),
    "cpu_fallback": ("watchdog.failure",),
    "oversized_context": (),
    "repeated_compaction": ("context.compacted",),
    "large_text_attachment": ("request.received",),
    "unsupported_attachment": ("request.received",),
    "bot_restart_conversation_continuity": ("request.received",),
}

FORBIDDEN_EVIDENCE_KEYS = {
    "api_key",
    "authorization",
    "bot_token",
    "content",
    "message_content",
    "prompt",
    "request_body",
    "response_body",
    "response_text",
    "system_prompt",
    "attachment_body",
}

REQUIRED_SOAK_METRICS: tuple[str, ...] = (
    "duration_hours",
    "total_requests",
    "successes",
    "failures",
    "timeouts",
    "hangs",
    "restarts",
    "manual_interventions",
    "peak_ram_mib",
    "peak_vram_mib",
    "queue_peak",
    "queue_overflow_incidents",
    "compaction_count",
    "compaction_failures",
    "summary_regressions",
    "unreaped_orphan_processes",
    "restart_storm_incidents",
    "context_overflow_requests_sent",
    "silent_truncation_or_drop_incidents",
    "gpu_cpu_fallback_misclassifications",
    "secret_or_token_leaks",
)


@dataclass(frozen=True)
class CaseEvaluation:
    scenario: str
    status: AcceptanceStatus
    reasons: tuple[str, ...]
    observed_events: tuple[str, ...]
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class SoakEvaluation:
    status: AcceptanceStatus
    reasons: tuple[str, ...]
    metrics: dict[str, int | float]


@dataclass(frozen=True)
class AcceptanceEvaluation:
    status: AcceptanceStatus
    runtime: dict[str, Any]
    cases: tuple[CaseEvaluation, ...]
    soak: SoakEvaluation
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        for case in payload["cases"]:
            case["status"] = case["status"].value
        payload["soak"]["status"] = self.soak.status.value
        return payload


def _parse_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceError(f"{name} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcceptanceError(f"{name} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise AcceptanceError(f"{name} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_json_events(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AcceptanceError(f"event log line {line_number} is not JSON") from exc
            if not isinstance(payload, dict):
                raise AcceptanceError(f"event log line {line_number} must be a JSON object")
            if not isinstance(payload.get("event"), str) or not payload["event"]:
                raise AcceptanceError(f"event log line {line_number} has no event name")
            _parse_timestamp(payload.get("timestamp"), f"event log line {line_number}.timestamp")
            _assert_no_sensitive_keys(payload, f"event log line {line_number}")
            events.append(payload)
    if not events:
        raise AcceptanceError("event log contains no events")
    return events


def _assert_no_sensitive_keys(value: Any, location: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_EVIDENCE_KEYS:
                raise AcceptanceError(f"{location} contains forbidden sensitive key {key!r}")
            _assert_no_sensitive_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_sensitive_keys(child, f"{location}[{index}]")


def _events_in_window(
    events: Sequence[Mapping[str, Any]],
    *,
    started_at: datetime,
    ended_at: datetime,
) -> list[Mapping[str, Any]]:
    if ended_at < started_at:
        raise AcceptanceError("case ended_at must not precede started_at")
    selected: list[Mapping[str, Any]] = []
    for event in events:
        timestamp = _parse_timestamp(event.get("timestamp"), "event.timestamp")
        if started_at <= timestamp <= ended_at:
            selected.append(event)
    return selected


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise AcceptanceError(f"{name} must be a list of non-empty strings")
    return tuple(value)


def _require_automated_assertions(case: Mapping[str, Any], reasons: list[str]) -> None:
    if case.get("assertion_mode") != "automated":
        reasons.append("assertion_mode must be 'automated'")
    assertions = case.get("assertions")
    if not isinstance(assertions, Mapping):
        reasons.append("automated assertions mapping is missing")
        return
    for key in ("stimulus_applied", "expected_behavior_observed", "automatic_recovery_observed"):
        if assertions.get(key) is not True:
            reasons.append(f"automated assertion not satisfied: {key}")


def evaluate_case(case: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> CaseEvaluation:
    scenario = case.get("scenario")
    if not isinstance(scenario, str) or scenario not in REQUIRED_SCENARIOS:
        raise AcceptanceError(f"unknown acceptance scenario: {scenario!r}")

    started_at = _parse_timestamp(case.get("started_at"), f"{scenario}.started_at")
    ended_at = _parse_timestamp(case.get("ended_at"), f"{scenario}.ended_at")
    window = _events_in_window(events, started_at=started_at, ended_at=ended_at)
    observed_events = tuple(sorted({str(event["event"]) for event in window}))
    request_ids = tuple(sorted({str(event["request_id"]) for event in window if event.get("request_id") is not None}))

    configured_required = _string_list(case.get("required_events"), f"{scenario}.required_events")
    required_events = tuple(dict.fromkeys((*SCENARIO_BASELINE_EVENTS[scenario], *configured_required)))
    forbidden_events = _string_list(case.get("forbidden_events"), f"{scenario}.forbidden_events")

    reasons: list[str] = []
    for event_name in required_events:
        if event_name not in observed_events:
            reasons.append(f"required event missing: {event_name}")
    for event_name in forbidden_events:
        if event_name in observed_events:
            reasons.append(f"forbidden event observed: {event_name}")

    _require_automated_assertions(case, reasons)

    manual_interventions = case.get("manual_interventions")
    if not isinstance(manual_interventions, int) or isinstance(manual_interventions, bool) or manual_interventions < 0:
        reasons.append("manual_interventions is missing or invalid")
    elif manual_interventions != 0:
        reasons.append(f"manual_interventions={manual_interventions}, expected 0")

    expected_request_ids = _string_list(case.get("request_ids"), f"{scenario}.request_ids")
    if not expected_request_ids:
        reasons.append("at least one real Discord request_id is required")
    else:
        missing_ids = sorted(set(expected_request_ids) - set(request_ids))
        if missing_ids:
            reasons.append(f"request_id evidence missing: {', '.join(missing_ids)}")

    if scenario in {"concurrent_discord_requests", "bot_restart_conversation_continuity"} and len(set(expected_request_ids)) < 2:
        reasons.append("scenario requires at least two distinct Discord request_ids")

    if scenario == "repeated_compaction":
        compaction_count = sum(1 for event in window if event.get("event") == "context.compacted")
        if compaction_count < 2:
            reasons.append("repeated_compaction requires at least two context.compacted events")

    if not window:
        status = AcceptanceStatus.UNVERIFIED
        reasons.append("no machine-readable events in case time window")
    elif reasons:
        status = AcceptanceStatus.FAIL
    else:
        status = AcceptanceStatus.PASS

    return CaseEvaluation(
        scenario=scenario,
        status=status,
        reasons=tuple(reasons),
        observed_events=observed_events,
        request_ids=request_ids,
    )


def _non_negative_number(metrics: Mapping[str, Any], key: str, reasons: list[str]) -> int | float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        reasons.append(f"{key} is missing or invalid")
        return None
    return value


def evaluate_soak(metrics: Mapping[str, Any], *, queue_capacity: int) -> SoakEvaluation:
    if not isinstance(metrics, Mapping):
        raise AcceptanceError("soak metrics must be a mapping")
    if not isinstance(queue_capacity, int) or isinstance(queue_capacity, bool) or queue_capacity < 0:
        raise AcceptanceError("queue_capacity must be a non-negative integer")

    reasons: list[str] = []
    normalized: dict[str, int | float] = {}
    for key in REQUIRED_SOAK_METRICS:
        value = _non_negative_number(metrics, key, reasons)
        if value is not None:
            normalized[key] = value

    if reasons:
        return SoakEvaluation(AcceptanceStatus.UNVERIFIED, tuple(reasons), normalized)

    checks: tuple[tuple[bool, str], ...] = (
        (normalized["duration_hours"] >= 24, "duration_hours must be at least 24"),
        (normalized["total_requests"] > 0, "total_requests must be greater than zero"),
        (
            normalized["successes"] + normalized["failures"] == normalized["total_requests"],
            "successes + failures must equal total_requests",
        ),
        (normalized["manual_interventions"] == 0, "manual_interventions must be 0"),
        (normalized["hangs"] == 0, "unrecovered hangs must be 0"),
        (normalized["queue_peak"] <= queue_capacity, f"queue_peak must not exceed bounded waiting queue {queue_capacity}"),
        (normalized["queue_overflow_incidents"] == 0, "queue_overflow_incidents must be 0"),
        (normalized["compaction_failures"] == 0, "compaction_failures must be 0"),
        (normalized["summary_regressions"] == 0, "summary_regressions must be 0"),
        (normalized["unreaped_orphan_processes"] == 0, "unreaped_orphan_processes must be 0"),
        (normalized["restart_storm_incidents"] == 0, "restart_storm_incidents must be 0"),
        (normalized["context_overflow_requests_sent"] == 0, "context_overflow_requests_sent must be 0"),
        (
            normalized["silent_truncation_or_drop_incidents"] == 0,
            "silent_truncation_or_drop_incidents must be 0",
        ),
        (
            normalized["gpu_cpu_fallback_misclassifications"] == 0,
            "gpu_cpu_fallback_misclassifications must be 0",
        ),
        (normalized["secret_or_token_leaks"] == 0, "secret_or_token_leaks must be 0"),
    )
    failed = [message for ok, message in checks if not ok]
    if failed:
        return SoakEvaluation(AcceptanceStatus.FAIL, tuple(failed), normalized)
    return SoakEvaluation(AcceptanceStatus.PASS, (), normalized)


def _runtime_identity(contract: ProductionContract) -> dict[str, Any]:
    return {
        "backend": contract.backend,
        "backend_version_or_commit": contract.backend_version_or_commit,
        "model": contract.model,
        "model_upstream": contract.model_upstream,
        "model_artifact": contract.model_artifact,
        "model_revision": contract.model_revision,
        "model_sha256": contract.model_sha256,
        "quantization_or_dtype": contract.quantization_or_dtype,
        "context_window_tokens": contract.context_window_tokens,
        "network_mode": contract.network_mode,
        "network_endpoint": contract.network_endpoint,
        "supervisor_kind": contract.supervisor_kind,
        "gpu_required": contract.gpu_required,
        "gpu_device_index": contract.gpu_device_index,
        "gpu_process_name_pattern": contract.gpu_process_name_pattern,
    }


def evaluate_acceptance(
    *,
    config: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    cases: Iterable[Mapping[str, Any]],
    soak_metrics: Mapping[str, Any],
) -> AcceptanceEvaluation:
    try:
        contract = validate_production_contract(config)
    except ProductionContractError as exc:
        raise AcceptanceError(f"production contract is invalid: {exc}") from exc
    if contract is None:
        raise AcceptanceError("production contract must be enabled before acceptance evaluation")

    runtime_control = config.get("runtime_control")
    if not isinstance(runtime_control, Mapping):
        raise AcceptanceError("runtime_control must be configured")
    max_concurrency = runtime_control.get("max_concurrency")
    max_queue_size = runtime_control.get("max_queue_size")
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise AcceptanceError("runtime_control.max_concurrency must be a positive integer")
    if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int) or max_queue_size < 0:
        raise AcceptanceError("runtime_control.max_queue_size must be a non-negative integer")

    case_results: list[CaseEvaluation] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise AcceptanceError("each case must be a mapping")
        result = evaluate_case(case, events)
        if result.scenario in seen:
            raise AcceptanceError(f"duplicate scenario evidence: {result.scenario}")
        seen.add(result.scenario)
        case_results.append(result)

    for scenario in REQUIRED_SCENARIOS:
        if scenario not in seen:
            case_results.append(
                CaseEvaluation(
                    scenario=scenario,
                    status=AcceptanceStatus.UNVERIFIED,
                    reasons=("required failure-injection scenario has no evidence",),
                    observed_events=(),
                    request_ids=(),
                )
            )

    case_results.sort(key=lambda result: REQUIRED_SCENARIOS.index(result.scenario))
    soak = evaluate_soak(soak_metrics, queue_capacity=max_queue_size)

    statuses = [result.status for result in case_results] + [soak.status]
    if AcceptanceStatus.FAIL in statuses:
        status = AcceptanceStatus.FAIL
    elif AcceptanceStatus.UNVERIFIED in statuses:
        status = AcceptanceStatus.UNVERIFIED
    else:
        status = AcceptanceStatus.PASS

    return AcceptanceEvaluation(
        status=status,
        runtime=_runtime_identity(contract),
        cases=tuple(case_results),
        soak=soak,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate llmcord production acceptance evidence")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--events", required=True, help="production JSONL event stream")
    parser.add_argument("--cases", required=True, help="JSON array of automated failure-injection case evidence")
    parser.add_argument("--soak", required=True, help="JSON object containing measured 24h soak metrics")
    parser.add_argument("--output", required=True, help="path for the canonical machine-readable acceptance result")
    args = parser.parse_args(argv)

    try:
        config = load_resolved_config(args.config)
        events = load_json_events(args.events)
        cases = _load_json(args.cases)
        soak = _load_json(args.soak)
        if not isinstance(cases, list):
            raise AcceptanceError("cases file must contain a JSON array")
        if not isinstance(soak, Mapping):
            raise AcceptanceError("soak file must contain a JSON object")
        result = evaluate_acceptance(config=config, events=events, cases=cases, soak_metrics=soak)
    except (AcceptanceError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": AcceptanceStatus.UNVERIFIED.value, "error": str(exc)}, ensure_ascii=False))
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.status.value, "output": str(output)}, ensure_ascii=False))
    if result.status is AcceptanceStatus.PASS:
        return 0
    if result.status is AcceptanceStatus.FAIL:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
