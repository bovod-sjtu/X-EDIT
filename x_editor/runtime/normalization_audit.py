"""Auditable invariants for semantics-preserving runtime normalization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from html import unescape
from pathlib import Path
from typing import Any

from x_editor.runtime.operation_transport import voice_clone_transport_equivalent


class NormalizationTransformClass(StrEnum):
    """Classes of transformations allowed at the tool-call boundary."""

    ENUM_CANONICALIZATION = "enum_canonicalization"
    REFERENCE_OR_PATH_RESOLUTION = "reference_or_path_resolution"
    DEFAULT_INJECTION = "default_injection"
    BIJECTIVE_ENCODING = "bijective_encoding"
    STRUCTURAL_CANONICALIZATION = "structural_canonicalization"
    SEMANTIC_CHANGE = "semantic_change"


@dataclass(frozen=True)
class NormalizationSnapshot:
    operation: str
    inputs: dict[str, Any]
    params: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "inputs": _jsonable(self.inputs),
            "params": _jsonable(self.params),
        }


@dataclass(frozen=True)
class NormalizationTransformation:
    path: str
    before: Any
    after: Any
    classification: NormalizationTransformClass
    reversible: bool
    semantics_preserving: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "before": _jsonable(self.before),
            "after": _jsonable(self.after),
            "classification": self.classification.value,
            "reversible": self.reversible,
            "semantics_preserving": self.semantics_preserving,
        }


@dataclass(frozen=True)
class NormalizationAudit:
    before: NormalizationSnapshot
    after: NormalizationSnapshot
    transformations: tuple[NormalizationTransformation, ...]
    semantic_fingerprint_before: str
    semantic_fingerprint_after: str
    behavior_changed: bool
    semantics_preserved: bool

    def as_dict(self) -> dict[str, Any]:
        """Keep legacy ``inputs``/``params`` keys while adding full audit data."""

        return {
            "operation_before": self.before.operation,
            "operation_after": self.after.operation,
            "inputs_before": _jsonable(self.before.inputs),
            "params_before": _jsonable(self.before.params),
            "inputs": _jsonable(self.after.inputs),
            "params": _jsonable(self.after.params),
            "transformations": [item.as_dict() for item in self.transformations],
            "semantic_fingerprint_before": self.semantic_fingerprint_before,
            "semantic_fingerprint_after": self.semantic_fingerprint_after,
            "behavior_changed": self.behavior_changed,
            "semantics_preserved": self.semantics_preserved,
        }


_MISSING = object()
_PATH_TOKENS = ("path", "audio", "ref", "artifact", "output_dir")
_ENUM_KEYS = {
    "aligner_backend",
    "edited_artifact_scope",
    "emotion_condition_mode",
    "foreground_action",
    "language",
    "method",
    "mode",
    "reconstruction_mode",
    "smooth_policy",
    "synthesis_mode",
}

_OPERATION_ENUM_ALIASES: dict[tuple[str, str], dict[str, str]] = {
    ("foreground_background_remix", "foreground_action"): {
        "background_only": "remove_foreground",
        "extract_foreground": "isolate_foreground",
        "foreground_only": "isolate_foreground",
        "keep_background": "remove_foreground",
        "remove_voice": "remove_foreground",
    },
}
_SAFE_DEFAULT_KEYS = {
    "crossfade_ms",
    "debug",
    "disable_prepend_context",
    "evaluate",
    "expand_radius",
    "fail_if_cannot_prepend_context",
    "headroom_db",
    "max_candidates",
    "max_f5_edit_seconds",
    "merge_gap",
    "min_unmasked_context_ratio",
    "min_unmasked_context_sec",
    "persist",
    "preserve_duration",
    "return_synthetic_target_audio",
    "seed",
}
_NON_SEMANTIC_PARAM_KEYS = _ENUM_KEYS | _SAFE_DEFAULT_KEYS | {
    "allow_method_fallback",
    "debug_dir",
    "debug_output_dir",
    "generation_id",
    "input_audio_path",
    "instruction",
    "language",
    "output_dir",
    "output_path",
    "profile_id",
    "service_url",
    "source_audio_path",
    "source_text",
    "speaker_audio_path",
    "text",
    "trace_id",
}
_AUDIO_ROLE_KEYS = {
    "input_audio",
    "input_audio_path",
    "reference_audio",
    "reference_audio_path",
    "ref_audio_path",
    "speaker_audio_path",
    "target_audio_path",
}


def normalization_snapshot(
    *, operation: str, inputs: dict[str, Any], params: dict[str, Any]
) -> NormalizationSnapshot:
    return NormalizationSnapshot(
        operation=str(operation),
        inputs=_jsonable(inputs),
        params=_jsonable(params),
    )


def build_normalization_audit(
    before: NormalizationSnapshot,
    after: NormalizationSnapshot,
) -> NormalizationAudit:
    before_payload = _semantic_payload(before)
    after_payload = _semantic_payload(after)
    before_fingerprint = _fingerprint(before_payload)
    after_fingerprint = _fingerprint(after_payload)
    semantics_preserved = (
        before_fingerprint == after_fingerprint
        or _text_edit_context_embedding_equivalent(before, after)
        or _text_edit_instruction_canonicalization_equivalent(before, after)
    )
    if before.operation == "qwen3_tts" and before.params.get("mode") == "voice_clone":
        # Compare the complete service call, not only its semantic summary.
        semantics_preserved = before == after or (
            after.operation == before.operation
            and voice_clone_transport_equivalent(
                before_inputs=before.inputs,
                before_params=before.params,
                after_inputs=after.inputs,
                after_params=after.params,
            )
        )
    raw_changes = _diff_values(before.as_dict(), after.as_dict())
    transformations = tuple(
        _classify_change(
            path=path,
            before=value_before,
            after=value_after,
            semantics_preserved=semantics_preserved,
        )
        for path, value_before, value_after in raw_changes
    )
    return NormalizationAudit(
        before=before,
        after=after,
        transformations=transformations,
        semantic_fingerprint_before=before_fingerprint,
        semantic_fingerprint_after=after_fingerprint,
        behavior_changed=bool(raw_changes),
        semantics_preserved=semantics_preserved,
    )


def _semantic_payload(snapshot: NormalizationSnapshot) -> dict[str, Any]:
    operation = snapshot.operation.removeprefix("edit_data_pipeline.").replace("-", "_")
    params = snapshot.params
    payload: dict[str, Any] = {
        "operation": snapshot.operation,
        "bound_audio_roles": sorted(
            _canonical_audio_role(key)
            for mapping in (snapshot.inputs, params)
            for key, value in mapping.items()
            if key in _AUDIO_ROLE_KEYS and value not in (None, "", [], {})
        ),
    }
    payload["bound_audio_roles"] = sorted(set(payload["bound_audio_roles"]))

    instruction_st = params.get("instruction_st")
    handled = {"instruction_st"}
    if operation == "text_edit":
        semantic_transform = params.get("semantic_transform")
        source_text = _non_empty_text(params.get("source_text"))
        target_text = _non_empty_text(params.get("target_text"))
        if isinstance(instruction_st, str) and instruction_st.strip():
            rendered_source = _render_text_edit_source(instruction_st)
            rendered_target = _render_text_edit_target(instruction_st)
            source_text = source_text or rendered_source
            target_text = target_text or rendered_target
            target_conflicts = (
                target_text != rendered_target
                and not _sentence_boundary_spacing_equivalent(
                    target_text,
                    rendered_target,
                )
            )
            if source_text != rendered_source or target_conflicts:
                payload["text_edit_contract_conflict"] = {
                    "rendered_source": rendered_source,
                    "rendered_target": rendered_target,
                }
        if semantic_transform is not None:
            payload["text_edit"] = {
                "semantic_transform": _canonical_semantic_value(semantic_transform)
            }
        else:
            payload["text_edit"] = {"target_text": target_text}
        handled.update({"semantic_transform", "semantic_transform_id", "source_text", "target_text"})
    elif isinstance(instruction_st, str) and instruction_st.strip():
        explicitly_encoded_pitch_rate = operation == "pitch_rate_edit" and any(
            params.get(key) is not None for key in ("semitones", "rate", "factor", "speed")
        )
        if not explicitly_encoded_pitch_rate:
            payload["tagged_instruction"] = _tagged_instruction_semantics(
                operation, instruction_st
            )

    if snapshot.operation == "artifact_ops.speaker_scoped_edit":
        nested_operation = str(params.get("edit_op") or params.get("edit_operation") or "")
        nested_params = params.get("edit_op_params") or params.get("edit_params") or {}
        if nested_operation and isinstance(nested_params, dict):
            payload["nested_edit"] = _canonical_nested_edit_semantics(
                nested_operation, nested_params
            )
            handled.update({"edit_op", "edit_operation", "edit_op_params", "edit_params"})

    semantic_params = {
        key: _canonical_operation_param(snapshot.operation, key, value)
        for key, value in params.items()
        if key not in handled and key not in _NON_SEMANTIC_PARAM_KEYS
    }
    if semantic_params:
        payload["semantic_params"] = semantic_params
    return payload


def _tagged_instruction_semantics(operation: str, value: str) -> dict[str, Any]:
    source = (
        _strip_emotion_instruction_tags(value)
        if operation == "emotion_style_edit"
        else _strip_instruction_tags(value)
    )
    if operation == "pitch_rate_edit":
        events: list[dict[str, Any]] = []
        pattern = re.compile(
            r"<(?P<kind>pitch|rate),\s*(?P<key>semitones|factor)="
            r"(?P<value>[+-]?[0-9]+(?:\.[0-9]+)?)\s*>(?P<body>.*?)</(?P=kind)>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(value):
            events.append(
                {
                    "kind": match.group("kind").lower(),
                    "key": match.group("key").lower(),
                    "value": float(match.group("value")),
                    "body": unescape(match.group("body")),
                    "source_offset": len(_strip_instruction_tags(value[: match.start()])),
                }
            )
        return {"source_text": source, "events": events}
    if operation == "emotion_style_edit":
        return {"source_text": source, "events": _emotion_events(value)}
    if operation == "pause_edit":
        events = []
        for match in re.finditer(r"<pause\s+(?P<attrs>[^>]*)/>", value, flags=re.IGNORECASE):
            events.append(
                {
                    "attrs": _attribute_map(match.group("attrs")),
                    "source_offset": len(_strip_instruction_tags(value[: match.start()])),
                }
            )
        return {"source_text": source, "events": events}
    return {"source_text": source, "instruction_st": value}


def _emotion_events(value: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    canonical = re.compile(
        r"<emo\b(?P<attrs>[^>]*)>(?P<body>.*?)</emo>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    legacy = re.compile(
        r"<(?P<emotion>[a-z_]+)(?:,\s*level=(?P<level>[1-3]))?>"
        r"(?P<body>.*?)</(?P=emotion)>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    desc = re.compile(
        r"<desc\b(?P<attrs>[^>]*)>(?P<body>.*?)</desc>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    bracket = re.compile(
        r"\[(?P<emotion>[a-z_]+)\](?P<body>.*?)\[/(?P=emotion)\]",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for pattern, kind in (
        (canonical, "emo"),
        (desc, "desc"),
        (legacy, "legacy"),
        (bracket, "bracket"),
    ):
        for match in pattern.finditer(value):
            attrs = _attribute_map(match.groupdict().get("attrs") or "")
            if kind == "legacy":
                attrs = {
                    "type": str(match.group("emotion")).lower(),
                    "level": match.group("level") or "2",
                }
            if kind == "desc" and "text" in attrs:
                attrs = {"desc": attrs["text"]}
            if kind == "bracket":
                attrs = {"desc": str(match.group("emotion")).lower()}
            events.append(
                {
                    "attrs": attrs,
                    "body": unescape(match.group("body")),
                    "source_offset": len(
                        _strip_emotion_instruction_tags(value[: match.start()])
                    ),
                }
            )
    return sorted(events, key=lambda item: (item["source_offset"], json.dumps(item["attrs"], sort_keys=True)))


def _strip_emotion_instruction_tags(value: str) -> str:
    """Render source wording for either XML or explicit paired bracket markup."""

    return re.sub(
        r"\[/?[a-z_]+\]",
        "",
        _strip_instruction_tags(value),
        flags=re.IGNORECASE,
    )


def _classify_change(
    *, path: str, before: Any, after: Any, semantics_preserved: bool
) -> NormalizationTransformation:
    key = path.rsplit(".", 1)[-1]
    added = before is _MISSING
    if not semantics_preserved and _semantic_path(path):
        classification = NormalizationTransformClass.SEMANTIC_CHANGE
    elif key in _ENUM_KEYS:
        classification = NormalizationTransformClass.ENUM_CANONICALIZATION
    elif any(token in key for token in _PATH_TOKENS):
        classification = NormalizationTransformClass.REFERENCE_OR_PATH_RESOLUTION
    elif added and (
        key in _SAFE_DEFAULT_KEYS
        or (key == "x_vector_only_mode" and semantics_preserved)
    ):
        classification = NormalizationTransformClass.DEFAULT_INJECTION
    elif key in {"instruction_st", "edit_tagged_text", "edits"}:
        classification = NormalizationTransformClass.BIJECTIVE_ENCODING
    else:
        classification = NormalizationTransformClass.STRUCTURAL_CANONICALIZATION
    preserving = semantics_preserved or classification != NormalizationTransformClass.SEMANTIC_CHANGE
    return NormalizationTransformation(
        path=path,
        before=None if before is _MISSING else before,
        after=None if after is _MISSING else after,
        classification=classification,
        reversible=not added and after is not _MISSING,
        semantics_preserving=preserving,
    )


def _semantic_path(path: str) -> bool:
    if path == "operation":
        return True
    key = path.rsplit(".", 1)[-1]
    return key not in _NON_SEMANTIC_PARAM_KEYS and not any(
        token in key for token in _PATH_TOKENS
    )


def _diff_values(before: Any, after: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        rows: list[tuple[str, Any, Any]] = []
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}" if path else key
            rows.extend(
                _diff_values(before.get(key, _MISSING), after.get(key, _MISSING), child_path)
            )
        return rows
    if before is _MISSING or after is _MISSING or before != after:
        return [(path, before, after)]
    return []


def _render_text_edit_source(value: str) -> str:
    without_insertions = re.sub(r"<ins\b[^>]*>.*?</ins>", "", value, flags=re.DOTALL)
    return _strip_instruction_tags(without_insertions)


def _render_text_edit_target(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        target = _attribute_map(attrs).get("targ")
        return target if target is not None else unescape(match.group("body"))

    rendered = re.sub(
        r"<sub\s+(?P<attrs>[^>]*)>(?P<body>.*?)</sub>",
        replace,
        value,
        flags=re.DOTALL,
    )
    rendered = re.sub(r"<del\b[^>]*>.*?</del>", "", rendered, flags=re.DOTALL)
    return _strip_instruction_tags(rendered)


def _text_edit_context_embedding_equivalent(
    before: NormalizationSnapshot,
    after: NormalizationSnapshot,
) -> bool:
    """Prove that one exact local text edit was embedded in unchanged context."""

    if before.operation != after.operation or not before.operation.endswith("text_edit"):
        return False
    before_instruction = before.params.get("instruction_st")
    after_instruction = after.params.get("instruction_st")
    if not isinstance(before_instruction, str) or not isinstance(after_instruction, str):
        return False
    local_source = _render_text_edit_source(before_instruction)
    local_target = _render_text_edit_target(before_instruction)
    full_source = _non_empty_text(after.params.get("source_text"))
    if not local_source or full_source is None:
        return False
    if before.params.get("source_text") != local_source:
        return False
    before_target = before.params.get("target_text")
    if before_target is not None and before_target != local_target:
        return False
    if full_source.count(local_source) != 1:
        return False
    if _render_text_edit_source(after_instruction) != full_source:
        return False
    start = full_source.index(local_source)
    expected_target = full_source[:start] + local_target + full_source[start + len(local_source) :]
    return (
        _render_text_edit_target(after_instruction) == expected_target
        and after.params.get("target_text") == expected_target
    )


def _text_edit_instruction_canonicalization_equivalent(
    before: NormalizationSnapshot,
    after: NormalizationSnapshot,
) -> bool:
    """Prove that tag placement changed while declared source and target did not."""

    if before.operation != after.operation or not before.operation.endswith("text_edit"):
        return False
    before_instruction = before.params.get("instruction_st")
    after_instruction = after.params.get("instruction_st")
    source_text = _non_empty_text(before.params.get("source_text"))
    target_text = _non_empty_text(before.params.get("target_text"))
    if not all(
        isinstance(value, str)
        for value in (before_instruction, after_instruction, source_text, target_text)
    ):
        return False
    if after.params.get("source_text") != source_text or after.params.get("target_text") != target_text:
        return False

    def compact(value: str) -> str:
        return re.sub(r"\s+", " ", unescape(value)).strip()

    if compact(_render_text_edit_source(before_instruction)) != compact(source_text):
        return False
    if compact(_render_text_edit_target(before_instruction)) != compact(target_text):
        return False
    if _render_text_edit_source(after_instruction) != source_text:
        return False
    if _render_text_edit_target(after_instruction) != target_text:
        return False

    projected_before = NormalizationSnapshot(
        operation=before.operation,
        inputs=before.inputs,
        params={**before.params, "instruction_st": after_instruction},
    )
    return _semantic_payload(projected_before) == _semantic_payload(after)


def _sentence_boundary_spacing_equivalent(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left == right

    def canonical(value: str) -> str:
        compact = re.sub(r"\s+", " ", unescape(value)).strip()
        return re.sub(
            r"([.!?。！？][\"'”’]?)\s+(?=\S)",
            r"\1",
            compact,
        )

    return canonical(left) == canonical(right)


def _strip_instruction_tags(value: str) -> str:
    text = re.sub(r"<pause\s+[^>]*/>", "", value, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text)


def _attribute_map(value: str) -> dict[str, str]:
    return {
        match.group("name").lower(): unescape(match.group("value"))
        for match in re.finditer(
            r"(?P<name>[a-zA-Z_][\w-]*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
            value,
            flags=re.DOTALL,
        )
    }


def _canonical_semantic_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {str(key): _canonical_semantic_value(child) for key, child in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_semantic_value(child) for child in value]
    return _jsonable(value)


def _canonical_operation_param(operation: str, key: str, value: Any) -> Any:
    normalized_operation = operation.removeprefix("edit_data_pipeline.").replace("-", "_")
    aliases = _OPERATION_ENUM_ALIASES.get((normalized_operation, key))
    if aliases and isinstance(value, str):
        value = aliases.get(value.strip().casefold(), value)
    return _canonical_semantic_value(value)


def _canonical_audio_role(key: str) -> str:
    if key in {"reference_audio", "reference_audio_path", "ref_audio_path", "speaker_audio_path"}:
        return "reference_audio"
    if key == "target_audio_path":
        return "target_audio"
    return "input_audio"


def _canonical_nested_edit_semantics(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    normalized_operation = operation.removeprefix("edit_data_pipeline.")
    isolated = dict(params)
    foreign_values: list[Any] = []
    owners = {
        "pitch_method": {"pitch_rate_edit", "pitch_shift"},
        "rate_method": {"pitch_rate_edit", "speech_rate"},
    }
    for key, allowed_operations in owners.items():
        if key in isolated and normalized_operation not in allowed_operations:
            foreign_values.append(isolated.pop(key))
    if "method" in isolated and any(isolated["method"] == value for value in foreign_values):
        isolated.pop("method", None)
    if normalized_operation == "volume_normalize":
        isolated.setdefault("target_db", -18.0)
    semantic_params = {
        key: _canonical_semantic_value(value)
        for key, value in isolated.items()
        if key not in _NON_SEMANTIC_PARAM_KEYS
    }
    return {"operation": operation, "params": semantic_params}


def _non_empty_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is _MISSING:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, set):
        return sorted(_jsonable(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    value_type = type(value)
    return {
        "__runtime_dependency_type__": (
            f"{value_type.__module__}.{value_type.__qualname__}"
        )
    }
