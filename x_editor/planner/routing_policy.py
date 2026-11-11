"""Queryable non-legacy planner skills for X-Editor routing context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

from x_editor.core.schemas import EnrichedIntent, ToolCapability

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment only
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


SKILL_DIR = Path(__file__).with_name("skills")


@dataclass(frozen=True)
class RoutingPolicyCandidate:
    """One policy-backed candidate workflow for a request."""

    policy_id: str
    priority: int
    score: int
    selected_tools: list[str]
    workflow: list[str]
    rationale: str
    required_evidence: list[str]
    guardrails: list[str]
    matched_signals: list[str]
    missing_tools: list[str]

    def model_dump(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "priority": self.priority,
            "score": self.score,
            "selected_tools": self.selected_tools,
            "workflow": self.workflow,
            "rationale": self.rationale,
            "required_evidence": self.required_evidence,
            "guardrails": self.guardrails,
            "matched_signals": self.matched_signals,
            "missing_tools": self.missing_tools,
        }


def load_planner_skills(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load declarative planner skills from the dedicated non-legacy registry."""

    if yaml is None:
        raise RuntimeError("PyYAML is required to load planner skills") from YAML_IMPORT_ERROR
    root = Path(path or SKILL_DIR)
    if root.is_file():
        data = yaml.safe_load(root.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and "policies" in data:
            policies = data.get("policies")
            if not isinstance(policies, list):
                raise ValueError(f"Routing policy file must contain a policies list: {root}")
            return [item for item in policies if isinstance(item, dict)]
        return [data] if isinstance(data, dict) and data.get("skill_id") else []
    if not root.exists():
        return []
    skills: list[dict[str, Any]] = []
    for skill_path in sorted(root.glob("*.yaml")):
        if skill_path.name.startswith("_"):
            continue
        data = yaml.safe_load(skill_path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and data.get("skill_id"):
            data.setdefault("source_path", str(skill_path))
            skills.append(data)
    return skills


def load_routing_policies(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Compatibility alias for code that still says routing policy."""

    return load_planner_skills(path)


def query_routing_policies(
    intent: EnrichedIntent | str,
    *,
    capabilities: dict[str, ToolCapability] | Iterable[str] | None = None,
    reference_audio_paths: list[str] | None = None,
    limit: int = 4,
    min_score: int = 1,
) -> list[RoutingPolicyCandidate]:
    """Return prioritized policy candidates without making the final routing decision."""

    text = intent.clarified_goal if isinstance(intent, EnrichedIntent) else str(intent)
    lowered = text.lower()
    available = _available_tool_ids(capabilities)
    refs = reference_audio_paths or []
    candidates: list[RoutingPolicyCandidate] = []
    for policy in load_planner_skills():
        signals = [str(item).lower() for item in policy.get("positive_signals") or []]
        negative_signals = [str(item).lower() for item in policy.get("negative_signals") or []]
        if any(_matches_signal(lowered, signal) for signal in negative_signals):
            continue
        matched = [signal for signal in signals if _matches_signal(lowered, signal)]
        score = len(matched)
        if policy.get("requires_reference_audio") and refs:
            score += 1
            matched.append("reference_audio_paths")
        if score < min_score:
            continue
        declared_selected = [str(item) for item in policy.get("selected_tools") or []]
        runtime_ops = {str(item) for item in policy.get("runtime_operations") or []}
        missing = [
            tool_id
            for tool_id in declared_selected
            if available is not None and tool_id not in available and tool_id not in runtime_ops
        ]
        selected = [
            tool_id
            for tool_id in declared_selected
            if available is None or tool_id in available or tool_id in runtime_ops
        ]
        candidates.append(
            RoutingPolicyCandidate(
                policy_id=str(policy.get("policy_id") or policy.get("skill_id") or "unknown"),
                priority=int(policy.get("priority") or 0),
                score=score,
                selected_tools=selected,
                workflow=[str(item) for item in policy.get("workflow") or []],
                rationale=str(policy.get("rationale") or ""),
                required_evidence=[str(item) for item in policy.get("required_evidence") or []],
                guardrails=[str(item) for item in policy.get("guardrails") or []],
                matched_signals=matched,
                missing_tools=missing,
            )
        )
    candidates.sort(key=lambda item: (item.score, item.priority), reverse=True)
    return candidates[:limit]


def routing_policy_context(
    intent: EnrichedIntent | str,
    *,
    capabilities: dict[str, ToolCapability] | Iterable[str] | None = None,
    reference_audio_paths: list[str] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return JSON-serializable policy context for planner prompts or traces."""

    return [
        candidate.model_dump()
        for candidate in query_routing_policies(
            intent,
            capabilities=capabilities,
            reference_audio_paths=reference_audio_paths,
            limit=limit,
        )
    ]


def _available_tool_ids(capabilities: dict[str, ToolCapability] | Iterable[str] | None) -> set[str] | None:
    if capabilities is None:
        return None
    if isinstance(capabilities, dict):
        return set(capabilities)
    return {str(item) for item in capabilities}


def _matches_signal(lowered_text: str, lowered_signal: str) -> bool:
    signal = str(lowered_signal or "").lower().strip()
    if not signal:
        return False
    if _has_ascii_alnum(signal):
        prefix = r"(?<![a-z0-9])" if signal[0].isalnum() and signal[0].isascii() else ""
        suffix = r"(?![a-z0-9])" if signal[-1].isalnum() and signal[-1].isascii() else ""
        return re.search(prefix + re.escape(signal) + suffix, lowered_text) is not None
    return signal in lowered_text


def _has_ascii_alnum(value: str) -> bool:
    return any(ch.isascii() and ch.isalnum() for ch in value)
