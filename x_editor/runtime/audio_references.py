"""Resolve planner-facing audio aliases to concrete runtime paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

AUDIO_REFERENCE_KEYS = {
    "audio",
    "audio_path",
    "audio_paths",
    "input_audio",
    "input_audios",
    "input_audio_path",
    "input_audio_paths",
    "source_audio",
    "source_audio_path",
    "target_audio",
    "target_audio_path",
    "reference_audio",
    "reference_audio_path",
    "reference_audio_paths",
    "ref_audio",
    "ref_audio_path",
    "mixture_audio",
    "mixture_audio_path",
    "speaker_audio",
    "speaker_audio_path",
    "target_speaker_reference_path",
    "edited_spans_audio",
    "edited_spans_audio_paths",
    "replacement_audio",
    "replacement_audio_path",
}

REFERENCE_AUDIO_ARGUMENT_KEYS = {
    "edited_spans_audio",
    "edited_spans_audio_paths",
    "replacement_audio",
    "replacement_audio_path",
}

AudioReferenceRole = Literal["source", "reference", "authoritative_output"]
ExpectedAudioReferenceRole = Literal["source", "reference"]


class AudioReferenceResolutionError(ValueError):
    """Reject an unsafe or ambiguous opaque runtime audio reference."""


@dataclass(frozen=True)
class _OpaqueAudioTarget:
    canonical_ref: str
    materialized_path: str
    role: AudioReferenceRole
    order: int
    producer_step_id: str | None = None


def collect_input_audios(state: dict[str, Any]) -> list[str]:
    """Return the ordered runtime input audios known to this execution."""

    candidates = state.get("input_audios") or state.get("input_audio_paths") or []
    audios = _as_str_list(candidates)
    if not audios:
        primary = state.get("input_audio")
        if isinstance(primary, str) and primary:
            audios.append(primary)
        audios.extend(_as_str_list(state.get("reference_audio_paths") or []))
    return _dedupe_preserve_order(audios)


def build_audio_alias_map(state: dict[str, Any]) -> dict[str, str]:
    """Build case-insensitive aliases for primary, reference, and artifact audio."""

    aliases: dict[str, str] = {}
    audios = collect_input_audios(state)
    for index, path in enumerate(audios):
        _add_alias(aliases, f"input:{index}", path)
        _add_alias(aliases, f"input_audio:{index}", path)
        _add_alias(aliases, f"input_audio_path:{index}", path)
        _add_alias(aliases, f"input_audios[{index}]", path)
        _add_alias(aliases, f"input_audio_paths[{index}]", path)
        _add_alias(aliases, f"input_audio_{index}", path)
        _add_alias(aliases, f"input_audio_path_{index}", path)
        _add_alias(aliases, f"audio_paths[{index}]", path)
        _add_alias(aliases, f"audio:{index}", path)
        _add_alias(aliases, str(index), path)
        _add_alias(aliases, f"input_{index}", path)
        _add_alias(aliases, f"input_audio_{index}", path)
        _add_alias(aliases, f"input_audio_path_{index}", path)
        _add_alias(aliases, f"audio_{index}", path)
        _add_alias(aliases, f"audio_path_{index}", path)
        _add_alias(aliases, f"session_input_audio_{index}", path)
        _add_alias(aliases, f"session_input_audio_path_{index}", path)
        _add_alias(aliases, f"session.input_audio[{index}]", path)
        _add_alias(aliases, f"session.input_audios[{index}]", path)
        _add_alias(aliases, f"session.audio[{index}]", path)
        _add_alias(aliases, f"session.audios[{index}]", path)
        _add_alias(aliases, f"session.round[0].audio[{index}]", path)
        _add_alias(aliases, f"session.round[0].audios[{index}]", path)

        human_index = index + 1
        _add_alias(aliases, f"audio{human_index}", path)
        _add_alias(aliases, f"audio_{human_index}", path)
        _add_alias(aliases, f"audio_path{human_index}", path)
        _add_alias(aliases, f"audio_path_{human_index}", path)
        _add_alias(aliases, f"input{human_index}", path)
        _add_alias(aliases, f"input_audio{human_index}", path)
        _add_alias(aliases, f"input_audio_path{human_index}", path)
        _add_alias(aliases, f"session_input_audio{human_index}", path)
        if index == 0:
            _add_alias(aliases, "input", path)
            _add_alias(aliases, "initial", path)
            _add_alias(aliases, "initial_audio", path)
            _add_alias(aliases, "initial_audio_path", path)
            _add_alias(aliases, "source", path)
            _add_alias(aliases, "source_audio", path)
            _add_alias(aliases, "source_audio_path", path)
            _add_alias(aliases, "original", path)
            _add_alias(aliases, "original_audio", path)
            _add_alias(aliases, "original_audio_path", path)

    refs = _as_str_list(state.get("reference_audio_paths") or [])
    for index, path in enumerate(refs):
        _add_alias(aliases, f"reference:{index}", path)
        _add_alias(aliases, f"reference_audio:{index}", path)
        _add_alias(aliases, f"reference_audio_{index}", path)
        _add_alias(aliases, f"reference_audio_path_{index}", path)
        _add_alias(aliases, f"reference_audio_paths[{index}]", path)
        _add_alias(aliases, f"reference_audio_path[{index}]", path)
        _add_alias(aliases, f"ref:{index}", path)
        _add_alias(aliases, f"ref_{index}", path)
        _add_alias(aliases, f"reference{index + 1}", path)
        _add_alias(aliases, f"reference_audio{index + 1}", path)
        _add_alias(aliases, f"ref{index + 1}", path)

    reference_items = state.get("reference_audio_items") or []
    if isinstance(reference_items, list):
        for index, item in enumerate(reference_items):
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("audio_path") or item.get("reference_audio_path")
            if not isinstance(path, str) or not path:
                continue
            fallback_id = f"ref{index + 1}"
            for alias in _reference_item_aliases(item, fallback_id=fallback_id):
                _add_alias(aliases, alias, path)

    reference_aliases = state.get("reference_audio_aliases") or {}
    if isinstance(reference_aliases, dict):
        for alias, path in reference_aliases.items():
            if isinstance(alias, str) and isinstance(path, str):
                _add_alias(aliases, alias, path)

    session_aliases = state.get("edit_session_audio_aliases") or state.get("session_audio_aliases") or {}
    if isinstance(session_aliases, dict):
        for alias, path in session_aliases.items():
            if isinstance(alias, str) and isinstance(path, str):
                _add_alias(aliases, alias, path)

    primary = state.get("input_audio")
    if isinstance(primary, str) and primary:
        _add_alias(aliases, "input_audio", primary)
        _add_alias(aliases, "base_audio", primary)
        _add_alias(aliases, "base", primary)

    source_text_audio = state.get("source_text_audio_path")
    if isinstance(source_text_audio, str) and source_text_audio:
        _add_alias(aliases, "source_text_audio_path", source_text_audio)
        _add_alias(aliases, "source_text_audio", source_text_audio)
        _add_alias(aliases, "transcribed_audio", source_text_audio)

    current = state.get("current_audio_path")
    if isinstance(current, str) and current:
        _add_alias(aliases, "current", current)
        _add_alias(aliases, "<current>", current)
        _add_alias(aliases, "current_audio", current)
        _add_alias(aliases, "current_audio_path", current)
        _add_alias(aliases, "session_current", current)
        _add_alias(aliases, "session_current_audio", current)

    artifacts = state.get("artifacts")
    if isinstance(artifacts, dict):
        for key, value in artifacts.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                _add_alias(aliases, key, value)
    return aliases


def resolve_audio_reference(
    value: Any,
    state: dict[str, Any],
    *,
    expected_role: ExpectedAudioReferenceRole | None = None,
    allow_runtime_paths: bool = True,
) -> Any:
    """Resolve a single string/list alias while leaving ordinary values intact."""

    if opaque_audio_resolution_enabled(state):
        return _resolve_opaque_audio_reference(
            value,
            state,
            expected_role=expected_role,
            allow_runtime_paths=allow_runtime_paths,
        )
    if isinstance(value, dict):
        structured = _resolve_structured_step_audio_reference(value, state)
        if structured is not None:
            return structured
        wrapped = _resolve_single_audio_reference_wrapper(value, state)
        if wrapped is not None:
            return wrapped
        return value
    if isinstance(value, list):
        return [resolve_audio_reference(item, state) for item in value]
    if not isinstance(value, str):
        return value
    token = value.strip()
    if not token:
        return value
    token, unwrapped = _unwrap_template_token(token)
    aliases = build_audio_alias_map(state)
    resolved = aliases.get(token.lower())
    if resolved is not None:
        return resolved
    step_output = _resolve_string_step_audio_reference(token, state)
    if step_output is not None:
        return step_output
    perception_step_audio = _resolve_perception_step_current_audio_reference(token, state)
    if perception_step_audio is not None:
        return perception_step_audio
    if _looks_like_path(token):
        return token if unwrapped else value
    return value


def resolve_audio_references_in_mapping(
    mapping: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Resolve known audio-reference fields in an inputs/params mapping."""

    return {
        key: _resolve_audio_reference_value(key, value, state)
        for key, value in dict(mapping).items()
    }


def _resolve_audio_reference_value(key: Any, value: Any, state: dict[str, Any]) -> Any:
    key_text = str(key)
    if opaque_audio_resolution_enabled(state) and _is_audio_reference_key(key_text):
        return resolve_audio_reference(
            value,
            state,
            expected_role=_expected_audio_reference_role(key_text),
            allow_runtime_paths=bool(
                state.get("_runtime_audio_boundary_materialized")
            ),
        )
    if isinstance(value, dict):
        structured = _resolve_structured_step_audio_reference(value, state)
        if structured is not None:
            return structured
        return {
            nested_key: _resolve_audio_reference_value(nested_key, nested_value, state)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_audio_reference_value(key_text, item, state)
            for item in value
        ]
    if _is_audio_reference_key(key_text) or _is_known_alias(value, state):
        return resolve_audio_reference(value, state)
    return value


def opaque_audio_resolution_enabled(state: dict[str, Any]) -> bool:
    context = state.get("context")
    if not isinstance(context, dict):
        context = {}
    mode = state.get("_runtime_audio_resolution_mode") or context.get(
        "_runtime_audio_resolution_mode"
    )
    return mode == "opaque_ref_v1"


def bound_audio_refs(
    state: dict[str, Any],
    *,
    roles: set[AudioReferenceRole] | None = None,
) -> list[str]:
    bindings = _opaque_audio_bindings(state)
    return [
        item.canonical_ref
        for item in sorted(bindings, key=lambda binding: binding.order)
        if roles is None or item.role in roles
    ]


def canonicalize_perception_audio_bindings(
    value: Any,
    state: dict[str, Any],
    *,
    audio_refs: list[str] | None = None,
) -> Any:
    """Map private media bindings in Perception results back to canonical refs."""

    if not opaque_audio_resolution_enabled(state):
        return value
    bindings = _opaque_audio_bindings(state)
    by_ref = {item.canonical_ref: item for item in bindings}
    by_path = {item.materialized_path: item for item in bindings}
    allowed_refs = list(audio_refs) if audio_refs is not None else None
    if allowed_refs is not None:
        if len(allowed_refs) != len(set(allowed_refs)):
            raise AudioReferenceResolutionError(
                "Perception audio_refs contain duplicate canonical refs"
            )
        for audio_ref in allowed_refs:
            if audio_ref not in by_ref:
                raise AudioReferenceResolutionError(
                    "Perception audio_refs contain an unknown canonical ref"
                )
    return _canonicalize_perception_audio_value(
        value,
        by_ref=by_ref,
        by_path=by_path,
        allowed_refs=allowed_refs,
        path="perception_result",
    )


def canonical_audio_ref_for_runtime_path(
    value: Any,
    state: dict[str, Any],
) -> str | None:
    if not opaque_audio_resolution_enabled(state):
        return str(value) if isinstance(value, (str, Path)) and str(value) else None
    target = _opaque_target_for_runtime_path(
        str(value) if isinstance(value, (str, Path)) else value,
        state,
        _opaque_audio_bindings(state),
    )
    return target.canonical_ref if target is not None else None


def canonicalize_runtime_audio_paths(value: Any, state: dict[str, Any]) -> Any:
    """Replace known private runtime paths with stable public audio refs."""

    if not opaque_audio_resolution_enabled(state):
        return value
    bindings = _opaque_audio_bindings(state)
    return _canonicalize_runtime_audio_paths(
        value,
        state=state,
        bindings=bindings,
        key=None,
    )


def _canonicalize_runtime_audio_paths(
    value: Any,
    *,
    state: dict[str, Any],
    bindings: list[_OpaqueAudioTarget],
    key: str | None,
) -> Any:
    if isinstance(value, str):
        target = _opaque_target_for_runtime_path(value, state, bindings)
        if target is not None:
            return target.canonical_ref
        if key is not None and _is_audio_reference_key(key) and _looks_like_path(value):
            return "invalid_opaque_audio_literal"
        return value
    if isinstance(value, Path):
        text = str(value)
        target = _opaque_target_for_runtime_path(text, state, bindings)
        if target is not None:
            return target.canonical_ref
        if key is not None and _is_audio_reference_key(key):
            return "invalid_opaque_audio_literal"
        return text
    if isinstance(value, dict):
        return {
            child_key: _canonicalize_runtime_audio_paths(
                child,
                state=state,
                bindings=bindings,
                key=str(child_key),
            )
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _canonicalize_runtime_audio_paths(
                child,
                state=state,
                bindings=bindings,
                key=key,
            )
            for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _canonicalize_runtime_audio_paths(
                child,
                state=state,
                bindings=bindings,
                key=key,
            )
            for child in value
        )
    return value


def register_runtime_audio_paths(state: dict[str, Any], value: Any) -> None:
    paths = state.setdefault("_runtime_trusted_audio_paths", set())
    if not isinstance(paths, set):
        paths = set()
        state["_runtime_trusted_audio_paths"] = paths
    _collect_runtime_audio_paths(value, paths, state=state)


def validate_materialized_audio_reference_roles(
    mapping: dict[str, Any],
    state: dict[str, Any],
) -> None:
    if not opaque_audio_resolution_enabled(state):
        return
    bindings = _opaque_audio_bindings(state)
    _validate_materialized_audio_value(
        mapping,
        state=state,
        bindings=bindings,
        key=None,
    )


def restore_canonical_audio_references(
    resolved: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    """Restore planner-authored audio fields after a private boundary call."""

    restored = dict(resolved)
    for key in set(resolved) | set(canonical):
        key_text = str(key)
        if _is_audio_reference_key(key_text) or key_text in {
            "artifact_ref_id",
            "replacement_audio",
            "replacement_audio_path",
            "edited_audio",
            "edited_audio_path",
            "insert_audio",
            "insert_audio_path",
            "path",
        }:
            if key in canonical:
                restored[key] = canonical[key]
            else:
                restored.pop(key, None)
            continue
        resolved_value = restored.get(key)
        canonical_value = canonical.get(key)
        if isinstance(resolved_value, dict) and isinstance(canonical_value, dict):
            restored[key] = restore_canonical_audio_references(
                resolved_value,
                canonical_value,
            )
        elif isinstance(resolved_value, list) and isinstance(canonical_value, list):
            restored[key] = _restore_canonical_audio_reference_list(
                resolved_value,
                canonical_value,
            )
    return restored


def _resolve_opaque_audio_reference(
    value: Any,
    state: dict[str, Any],
    *,
    expected_role: ExpectedAudioReferenceRole | None,
    allow_runtime_paths: bool,
) -> Any:
    if isinstance(value, list):
        return [
            _resolve_opaque_audio_reference(
                item,
                state,
                expected_role=expected_role,
                allow_runtime_paths=allow_runtime_paths,
            )
            for item in value
        ]
    if isinstance(value, dict):
        target = _opaque_structured_audio_target(value, state)
        if target is None and len(value) == 1:
            key, nested = next(iter(value.items()))
            key_text = str(key)
            if _is_audio_reference_key(key_text) or key_text == "artifact_ref_id":
                nested_role = (
                    _expected_audio_reference_role(key_text)
                    if _is_audio_reference_key(key_text)
                    else expected_role
                )
                return _resolve_opaque_audio_reference(
                    nested,
                    state,
                    expected_role=nested_role,
                    allow_runtime_paths=allow_runtime_paths,
                )
        if target is None:
            raise AudioReferenceResolutionError(
                "Opaque audio reference object is ambiguous or unknown"
            )
        _require_opaque_audio_role(target, expected_role, state)
        return target.materialized_path
    if not isinstance(value, str) or not value.strip():
        raise AudioReferenceResolutionError(
            "Opaque audio reference must be a non-empty canonical ref"
        )

    token, _ = _unwrap_template_token(value.strip())
    bindings = _opaque_audio_bindings(state)
    target = _opaque_bound_target(token, bindings)
    if target is None:
        target = _opaque_alias_target(token, state, bindings)
    if target is None:
        target = _opaque_step_or_artifact_target(token, state)
    if target is not None:
        _require_opaque_audio_role(target, expected_role, state)
        return target.materialized_path

    if _looks_like_path(token):
        trusted_paths = state.get("_runtime_trusted_audio_paths")
        if (
            allow_runtime_paths
            and isinstance(trusted_paths, set)
            and token in trusted_paths
        ):
            return token
        raise AudioReferenceResolutionError(
            "Physical audio path literals are forbidden in opaque_ref_v1"
        )
    if token.casefold().startswith("aud:v1:"):
        raise AudioReferenceResolutionError("Unknown opaque canonical audio ref")
    raise AudioReferenceResolutionError("Unknown opaque audio alias or output ref")


def _opaque_audio_bindings(state: dict[str, Any]) -> list[_OpaqueAudioTarget]:
    context = state.get("context")
    if not isinstance(context, dict):
        context = {}
    raw_bindings = state.get("_runtime_audio_bindings")
    if raw_bindings is None:
        raw_bindings = context.get("_runtime_audio_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise AudioReferenceResolutionError(
            "opaque_ref_v1 requires runtime audio bindings"
        )

    bindings: list[_OpaqueAudioTarget] = []
    exact_refs: set[str] = set()
    folded_refs: dict[str, str] = {}
    materialized_paths: set[str] = set()
    orders: set[int] = set()
    for raw in raw_bindings:
        audio_ref = _binding_field(raw, "audio_ref")
        materialized_path = _binding_field(raw, "materialized_path")
        role = _binding_field(raw, "role")
        order = _binding_field(raw, "order")
        if not isinstance(audio_ref, str) or not re.fullmatch(
            r"aud:v1:[A-Za-z0-9][A-Za-z0-9._-]*",
            audio_ref,
            flags=re.IGNORECASE,
        ):
            raise AudioReferenceResolutionError(
                "Runtime audio binding has an invalid canonical ref"
            )
        if not isinstance(materialized_path, str) or not materialized_path:
            raise AudioReferenceResolutionError(
                "Runtime audio binding has no materialized path"
            )
        if role not in {"source", "reference", "authoritative_output"}:
            raise AudioReferenceResolutionError(
                "Runtime audio binding has an invalid role"
            )
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise AudioReferenceResolutionError(
                "Runtime audio binding has an invalid order"
            )
        folded = audio_ref.casefold()
        if audio_ref in exact_refs:
            raise AudioReferenceResolutionError(
                "Runtime audio bindings contain a duplicate canonical ref"
            )
        if folded in folded_refs:
            raise AudioReferenceResolutionError(
                "Runtime audio bindings contain a case-conflicting canonical ref"
            )
        if not audio_ref.startswith("aud:v1:"):
            raise AudioReferenceResolutionError(
                "Runtime audio binding canonical ref has invalid letter case"
            )
        if materialized_path in materialized_paths:
            raise AudioReferenceResolutionError(
                "Runtime audio bindings contain a duplicate materialized path"
            )
        if order in orders:
            raise AudioReferenceResolutionError(
                "Runtime audio bindings contain a duplicate order"
            )
        exact_refs.add(audio_ref)
        folded_refs[folded] = audio_ref
        materialized_paths.add(materialized_path)
        orders.add(order)
        bindings.append(
            _OpaqueAudioTarget(
                canonical_ref=audio_ref,
                materialized_path=materialized_path,
                role=role,
                order=order,
            )
        )
    return bindings


def _binding_field(binding: Any, field: str) -> Any:
    if isinstance(binding, dict):
        return binding.get(field)
    return getattr(binding, field, None)


def _opaque_bound_target(
    token: str,
    bindings: list[_OpaqueAudioTarget],
) -> _OpaqueAudioTarget | None:
    by_ref = {item.canonical_ref: item for item in bindings}
    if token in by_ref:
        return by_ref[token]
    folded_matches = [
        item for item in bindings if item.canonical_ref.casefold() == token.casefold()
    ]
    if folded_matches:
        raise AudioReferenceResolutionError(
            "Opaque canonical audio ref has incorrect letter case"
        )
    return None


def _opaque_alias_target(
    token: str,
    state: dict[str, Any],
    bindings: list[_OpaqueAudioTarget],
) -> _OpaqueAudioTarget | None:
    aliases: dict[str, _OpaqueAudioTarget] = {}
    sources = sorted(
        (
            item
            for item in bindings
            if item.role in {"source", "authoritative_output"}
        ),
        key=lambda item: item.order,
    )
    references = sorted(
        (item for item in bindings if item.role == "reference"),
        key=lambda item: item.order,
    )
    for index, target in enumerate(sources):
        for alias in _source_aliases(index):
            _add_opaque_alias(aliases, alias, target)
    for index, target in enumerate(references):
        for alias in _reference_aliases(index):
            _add_opaque_alias(aliases, alias, target)

    current = state.get("current_audio_path")
    current_target = _opaque_target_for_runtime_path(current, state, bindings)
    if current_target is not None:
        for alias in (
            "current",
            "current_audio",
            "current_audio_path",
            "session_current",
            "session_current_audio",
        ):
            aliases[alias] = current_target

    for mapping_key in (
        "reference_audio_aliases",
        "edit_session_audio_aliases",
        "session_audio_aliases",
    ):
        raw_aliases = state.get(mapping_key)
        if not isinstance(raw_aliases, dict):
            continue
        for raw_alias, raw_ref in raw_aliases.items():
            if not isinstance(raw_alias, str) or not isinstance(raw_ref, str):
                raise AudioReferenceResolutionError(
                    "Opaque audio aliases must map strings to canonical refs"
                )
            target = _opaque_bound_target(raw_ref, bindings)
            if target is None:
                raise AudioReferenceResolutionError(
                    "Opaque audio alias maps to an unknown canonical ref"
                )
            _add_opaque_alias(aliases, raw_alias, target)
    return aliases.get(token.casefold())


def _source_aliases(index: int) -> tuple[str, ...]:
    aliases = (
        f"input:{index}",
        f"input_audio:{index}",
        f"input_audio_{index}",
        f"input_audio_paths[{index}]",
        f"input_audios[{index}]",
        f"audio:{index}",
        f"audio{index + 1}",
        f"audio_{index + 1}",
        str(index),
    )
    if index != 0:
        return aliases
    return (
        *aliases,
        "input",
        "input_audio",
        "source",
        "source_audio",
        "source_audio_path",
        "base",
        "base_audio",
        "original",
        "original_audio",
    )


def _reference_aliases(index: int) -> tuple[str, ...]:
    return (
        f"reference:{index}",
        f"reference_audio:{index}",
        f"reference_audio_{index}",
        f"reference_audio_paths[{index}]",
        f"reference{index + 1}",
        f"reference_audio{index + 1}",
        f"ref:{index}",
        f"ref_{index}",
        f"ref{index + 1}",
    )


def _add_opaque_alias(
    aliases: dict[str, _OpaqueAudioTarget],
    alias: str,
    target: _OpaqueAudioTarget,
) -> None:
    folded = alias.casefold()
    existing = aliases.get(folded)
    if existing is not None and existing.canonical_ref != target.canonical_ref:
        raise AudioReferenceResolutionError(
            "Opaque audio alias resolves to multiple canonical refs"
        )
    aliases[folded] = target


def _opaque_structured_audio_target(
    value: dict[str, Any],
    state: dict[str, Any],
) -> _OpaqueAudioTarget | None:
    step_ref = _first_present(
        value,
        "from_step",
        "step_id",
        "step",
        "source_step",
        "input_from_step",
        "audio_from_step",
        "replacement_from_step",
        "edited_audio_from_step",
    )
    if not isinstance(step_ref, str) or not step_ref:
        return None
    output_key = _first_present(
        value,
        "from_output",
        "output",
        "output_key",
        "result_key",
        "key",
    )
    token = step_ref
    if output_key is not None:
        token = f"{step_ref}:{output_key}"
    return _opaque_step_or_artifact_target(token, state)


def _opaque_step_or_artifact_target(
    token: str,
    state: dict[str, Any],
) -> _OpaqueAudioTarget | None:
    step_outputs = state.get("step_outputs")
    if not isinstance(step_outputs, dict):
        step_outputs = {}
    step_ref = token
    output_key: str | None = None
    for separator in (":", "."):
        if separator in token:
            step_ref, output_key = token.split(separator, 1)
            break
    output = _exact_opaque_mapping_value(step_outputs, step_ref, label="step output")
    if isinstance(output, dict):
        key = output_key
        if key is not None and key.casefold() in {"result", "output", "audio"}:
            key = None
        path = _audio_path_from_step_output(output, key)
        if path:
            return _OpaqueAudioTarget(
                canonical_ref=(
                    f"{step_ref}:{output_key}" if output_key else step_ref
                ),
                materialized_path=path,
                role="authoritative_output",
                order=10**9,
                producer_step_id=step_ref,
            )

    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    artifact_path = _exact_opaque_mapping_value(
        artifacts,
        token,
        label="artifact output",
    )
    if not isinstance(artifact_path, str) or not artifact_path:
        return None
    producer_step_id = _runtime_path_step_output_producer(
        artifact_path,
        step_outputs,
    )
    if producer_step_id is None:
        return None
    return _OpaqueAudioTarget(
        canonical_ref=token,
        materialized_path=artifact_path,
        role="authoritative_output",
        order=10**9,
        producer_step_id=producer_step_id,
    )


def _exact_opaque_mapping_value(
    mapping: dict[str, Any],
    key: str,
    *,
    label: str,
) -> Any:
    if key in mapping:
        return mapping[key]
    matches = [
        candidate
        for candidate in mapping
        if isinstance(candidate, str) and candidate.casefold() == key.casefold()
    ]
    if matches:
        raise AudioReferenceResolutionError(f"Opaque {label} ref has incorrect case")
    return None


def _opaque_target_for_runtime_path(
    value: Any,
    state: dict[str, Any],
    bindings: list[_OpaqueAudioTarget],
) -> _OpaqueAudioTarget | None:
    if not isinstance(value, str) or not value:
        return None
    for binding in bindings:
        if value == binding.materialized_path:
            return binding
    step_outputs = state.get("step_outputs")
    if isinstance(step_outputs, dict):
        for step_id, output in step_outputs.items():
            if not isinstance(step_id, str) or not isinstance(output, dict):
                continue
            path = _audio_path_from_step_output(output)
            if path == value:
                return _OpaqueAudioTarget(
                    canonical_ref=f"{step_id}:output_audio_path",
                    materialized_path=value,
                    role="authoritative_output",
                    order=10**9,
                    producer_step_id=step_id,
                )
    return None


def _runtime_path_is_step_output(
    path: str,
    step_outputs: dict[str, Any],
) -> bool:
    return _runtime_path_step_output_producer(path, step_outputs) is not None


def _runtime_path_step_output_producer(
    path: str,
    step_outputs: dict[str, Any],
) -> str | None:
    for step_id, output in step_outputs.items():
        if (
            isinstance(step_id, str)
            and isinstance(output, dict)
            and _audio_path_from_step_output(output) == path
        ):
            return step_id
    return None


def _require_opaque_audio_role(
    target: _OpaqueAudioTarget,
    expected_role: ExpectedAudioReferenceRole | None,
    state: dict[str, Any],
) -> None:
    perception_refs = state.get("_runtime_perception_audio_refs")
    if not isinstance(perception_refs, list):
        context = state.get("context")
        perception_refs = (
            context.get("_runtime_perception_audio_refs")
            if isinstance(context, dict)
            else None
        )
    if (
        isinstance(perception_refs, list)
        and target.canonical_ref in perception_refs
    ):
        # Perception's input/reference keys describe observation roles, not
        # edit authority. This exception is scoped to the gateway's frozen
        # request namespace and is never enabled for forward edit execution.
        return
    if target.role == "authoritative_output":
        if expected_role == "reference" and target.producer_step_id is None:
            raise AudioReferenceResolutionError(
                "Reference audio field requires a reference binding or declared "
                "authoritative step output"
            )
        _require_declared_authoritative_output_dependency(target, state)
        return
    if _admitted_audio_usage_allows_role(target, expected_role, state):
        return
    if expected_role == "source" and target.role != "source":
        raise AudioReferenceResolutionError(
            "Source audio field cannot consume a reference binding"
        )
    if expected_role == "reference" and target.role != "reference":
        raise AudioReferenceResolutionError(
            "Reference audio field requires a reference binding"
        )


def _admitted_audio_usage_allows_role(
    target: _OpaqueAudioTarget,
    expected_role: ExpectedAudioReferenceRole | None,
    state: dict[str, Any],
) -> bool:
    if expected_role is None:
        return False
    context = state.get("context")
    raw_bindings = (
        context.get("_runtime_admitted_audio_usage_bindings")
        if isinstance(context, dict)
        else None
    )
    current_step_id = state.get("_current_step_id")
    if not isinstance(raw_bindings, list) or not isinstance(current_step_id, str):
        return False
    allowed_usages = (
        {"edit_input", "assembly_member", "comparison_input"}
        if expected_role == "source"
        else {"reference_input", "speaker_grounded_reference", "comparison_input"}
    )
    return any(
        isinstance(binding, dict)
        and binding.get("plan_step_id") == current_step_id
        and binding.get("audio_ref") == target.canonical_ref
        and binding.get("usage") in allowed_usages
        for binding in raw_bindings
    )


def _require_declared_authoritative_output_dependency(
    target: _OpaqueAudioTarget,
    state: dict[str, Any],
) -> None:
    producer_step_id = target.producer_step_id
    if producer_step_id is None:
        return
    current_step_id = state.get("_current_step_id")
    current_step = (
        _plan_step_for_ref(current_step_id, state)
        if isinstance(current_step_id, str) and current_step_id
        else None
    )
    dependencies = getattr(current_step, "depends_on", None)
    if not isinstance(dependencies, list) or producer_step_id not in dependencies:
        raise AudioReferenceResolutionError(
            "Authoritative step output requires a declared dependency"
        )


def _expected_audio_reference_role(key: str) -> ExpectedAudioReferenceRole:
    lowered = key.casefold()
    if (
        lowered in REFERENCE_AUDIO_ARGUMENT_KEYS
        or "reference" in lowered
        or lowered.startswith(("ref_audio", "speaker_audio"))
    ):
        return "reference"
    return "source"


def _canonicalize_perception_audio_value(
    value: Any,
    *,
    by_ref: dict[str, _OpaqueAudioTarget],
    by_path: dict[str, _OpaqueAudioTarget],
    allowed_refs: list[str] | None,
    path: str,
) -> Any:
    if isinstance(value, str):
        if value in by_path:
            canonical_ref = by_path[value].canonical_ref
            if allowed_refs is not None and canonical_ref not in allowed_refs:
                raise AudioReferenceResolutionError(
                    "Perception evidence audio reference is outside the request namespace"
                )
            return canonical_ref
        return value
    if isinstance(value, list):
        return [
            _canonicalize_perception_audio_value(
                item,
                by_ref=by_ref,
                by_path=by_path,
                allowed_refs=allowed_refs,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value

    normalized = dict(value)
    for key, child in list(normalized.items()):
        child_path = f"{path}.{key}"
        if key in {"output_audio_path", "output_audio_paths"}:
            # Perception frontends may retain generated review-slice paths in
            # audit metadata. They are outputs of observation, not workflow
            # input references, and reviewed evidence projection removes them.
            continue
        if key in {"audio_ref_id", "source_text_audio_path"}:
            if child in (None, ""):
                continue
            normalized[key] = _canonical_perception_audio_ref(
                child,
                by_ref=by_ref,
                by_path=by_path,
                allowed_refs=allowed_refs,
                path=child_path,
            )
            continue
        if key in {"audio_index", "source_text_audio_index"}:
            if child is None:
                continue
            ref_key = (
                "audio_ref_id" if key == "audio_index" else "source_text_audio_path"
            )
            declared_audio_ref = normalized.get(ref_key)
            if declared_audio_ref not in (None, ""):
                declared_audio_ref = _canonical_perception_audio_ref(
                    declared_audio_ref,
                    by_ref=by_ref,
                    by_path=by_path,
                    allowed_refs=allowed_refs,
                    path=f"{path}.{ref_key}",
                )
                normalized[ref_key] = declared_audio_ref
            normalized[key] = _normalize_perception_audio_index(
                child,
                by_ref=by_ref,
                allowed_refs=allowed_refs,
                declared_audio_ref=declared_audio_ref,
                path=child_path,
            )
            continue
        if _is_audio_reference_key(str(key)):
            normalized[key] = _canonicalize_perception_audio_reference_value(
                child,
                by_ref=by_ref,
                by_path=by_path,
                allowed_refs=allowed_refs,
                path=child_path,
            )
            continue
        normalized[key] = _canonicalize_perception_audio_value(
            child,
            by_ref=by_ref,
            by_path=by_path,
            allowed_refs=allowed_refs,
            path=child_path,
        )

    if allowed_refs is not None:
        _validate_perception_mapping_binding(
            normalized,
            allowed_refs=allowed_refs,
            path=path,
        )
    return normalized


def _canonicalize_perception_audio_reference_value(
    value: Any,
    *,
    by_ref: dict[str, _OpaqueAudioTarget],
    by_path: dict[str, _OpaqueAudioTarget],
    allowed_refs: list[str] | None,
    path: str = "perception",
) -> Any:
    if isinstance(value, list):
        return [
            _canonical_perception_audio_ref(
                item,
                by_ref=by_ref,
                by_path=by_path,
                allowed_refs=allowed_refs,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
            if item not in (None, "")
        ]
    if value in (None, ""):
        return value
    return _canonical_perception_audio_ref(
        value,
        by_ref=by_ref,
        by_path=by_path,
        allowed_refs=allowed_refs,
        path=path,
    )


def _canonical_perception_audio_ref(
    value: Any,
    *,
    by_ref: dict[str, _OpaqueAudioTarget],
    by_path: dict[str, _OpaqueAudioTarget],
    allowed_refs: list[str] | None,
    path: str = "perception",
) -> str:
    if not isinstance(value, str) or not value:
        raise AudioReferenceResolutionError(
            f"Perception evidence has a missing audio_ref_id namespace at {path}"
        )
    if value in by_ref:
        canonical_ref = value
    elif value in by_path:
        canonical_ref = by_path[value].canonical_ref
    else:
        raise AudioReferenceResolutionError(
            f"Perception evidence contains an unknown audio reference at {path}"
        )
    if allowed_refs is not None and canonical_ref not in allowed_refs:
        raise AudioReferenceResolutionError(
            "Perception evidence audio reference is outside the request namespace "
            f"at {path}"
        )
    return canonical_ref


def _normalize_perception_audio_index(
    value: Any,
    *,
    by_ref: dict[str, _OpaqueAudioTarget],
    allowed_refs: list[str] | None,
    declared_audio_ref: str | None,
    path: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AudioReferenceResolutionError(
            f"Perception evidence audio_index at {path} must be an integer"
        )
    if allowed_refs is None:
        return value
    if declared_audio_ref is not None:
        local_index = allowed_refs.index(declared_audio_ref)
        indexed_refs = {
            audio_ref
            for audio_ref in allowed_refs
            if value in {allowed_refs.index(audio_ref), by_ref[audio_ref].order}
        }
        if not indexed_refs:
            raise AudioReferenceResolutionError(
                f"Perception evidence audio_index at {path} is outside the request namespace"
            )
        if declared_audio_ref not in indexed_refs:
            raise AudioReferenceResolutionError(
                f"Perception evidence audio ref and index disagree at {path}"
            )
        return local_index
    matching_refs = [
        audio_ref for audio_ref in allowed_refs if by_ref[audio_ref].order == value
    ]
    local_ref = allowed_refs[value] if 0 <= value < len(allowed_refs) else None
    if len(matching_refs) > 1:
        raise AudioReferenceResolutionError(
            f"Perception evidence audio_index at {path} is ambiguous in the runtime namespace"
        )
    if matching_refs and local_ref is not None and matching_refs[0] != local_ref:
        raise AudioReferenceResolutionError(
            f"Perception evidence audio_index at {path} is ambiguous in the request namespace"
        )
    if len(matching_refs) == 1:
        return allowed_refs.index(matching_refs[0])
    if local_ref is not None:
        return value
    raise AudioReferenceResolutionError(
        f"Perception evidence audio_index at {path} is outside the request namespace"
    )


def _validate_perception_mapping_binding(
    value: dict[str, Any],
    *,
    allowed_refs: list[str],
    path: str,
) -> None:
    for ref_key, index_key in (
        ("audio_ref_id", "audio_index"),
        ("source_text_audio_path", "source_text_audio_index"),
    ):
        audio_ref = value.get(ref_key)
        audio_index = value.get(index_key)
        if audio_ref is None or audio_index is None:
            continue
        if allowed_refs[audio_index] != audio_ref:
            raise AudioReferenceResolutionError(
                f"Perception evidence audio ref and index disagree at {path}"
            )


def _collect_runtime_audio_paths(
    value: Any,
    paths: set[str],
    *,
    state: dict[str, Any],
) -> None:
    if isinstance(value, str) and canonical_audio_ref_for_runtime_path(value, state):
        paths.add(value)
        return
    if isinstance(value, dict):
        for child in value.values():
            _collect_runtime_audio_paths(child, paths, state=state)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _collect_runtime_audio_paths(child, paths, state=state)


def _validate_materialized_audio_value(
    value: Any,
    *,
    state: dict[str, Any],
    bindings: list[_OpaqueAudioTarget],
    key: str | None,
) -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _validate_materialized_audio_value(
                child,
                state=state,
                bindings=bindings,
                key=str(child_key),
            )
        return
    if isinstance(value, list):
        for child in value:
            _validate_materialized_audio_value(
                child,
                state=state,
                bindings=bindings,
                key=key,
            )
        return
    if key is None or not _is_audio_reference_key(key) or not isinstance(value, str):
        return
    if key.casefold().startswith(("output_", "generated_")):
        return
    target = _opaque_target_for_runtime_path(value, state, bindings)
    if target is None:
        raise AudioReferenceResolutionError(
            "Runtime audio field did not resolve to a bound or authoritative output"
        )
    _require_opaque_audio_role(
        target,
        _expected_audio_reference_role(key),
        state,
    )


def _restore_canonical_audio_reference_list(
    resolved: list[Any],
    canonical: list[Any],
) -> list[Any]:
    restored = list(resolved)
    for index, canonical_value in enumerate(canonical):
        if index >= len(restored):
            restored.append(canonical_value)
            continue
        resolved_value = restored[index]
        if isinstance(resolved_value, dict) and isinstance(canonical_value, dict):
            restored[index] = restore_canonical_audio_references(
                resolved_value,
                canonical_value,
            )
    return restored


def _resolve_single_audio_reference_wrapper(
    value: dict[str, Any],
    state: dict[str, Any],
) -> str | None:
    """Unwrap one unambiguous planner-authored audio reference object."""

    if len(value) != 1:
        return None
    key, nested = next(iter(value.items()))
    key_text = str(key)
    if not _is_audio_reference_key(key_text) and key_text != "artifact_ref_id":
        return None
    resolved = resolve_audio_reference(nested, state)
    if isinstance(resolved, str) and resolved.strip():
        if (
            key_text == "artifact_ref_id"
            and resolved == nested
            and not _looks_like_path(resolved)
        ):
            return None
        return resolved
    return None


def _resolve_structured_step_audio_reference(value: dict[str, Any], state: dict[str, Any]) -> str | None:
    step_ref = _first_present(
        value,
        "from_step",
        "step_id",
        "step",
        "source_step",
        "input_from_step",
        "audio_from_step",
        "replacement_from_step",
        "edited_audio_from_step",
    )
    if not isinstance(step_ref, str) or not step_ref.strip():
        return None

    output_key = _first_present(value, "from_output", "output", "output_key", "result_key", "key")
    output_key_text = str(output_key).strip() if output_key is not None else None
    step_outputs = state.get("step_outputs") if isinstance(state.get("step_outputs"), dict) else {}
    output = step_outputs.get(step_ref)
    if isinstance(output, dict):
        output_path = _audio_path_from_step_output(output, output_key_text)
        if output_path:
            return output_path

    resolved = resolve_audio_reference(step_ref, state)
    if isinstance(resolved, str) and resolved.strip():
        path = Path(resolved)
        if path.exists():
            return str(path)
    return None


def _resolve_string_step_audio_reference(value: str, state: dict[str, Any]) -> str | None:
    step_outputs = state.get("step_outputs") if isinstance(state.get("step_outputs"), dict) else {}
    plan = state.get("plan")
    if not step_outputs and plan is None:
        return None
    token = value.strip()
    if not token:
        return None
    for separator in (":", "."):
        if separator not in token:
            continue
        step_ref, output_key = token.split(separator, 1)
        output = _step_output_for_ref(step_ref.strip(), step_outputs)
        if not isinstance(output, dict):
            step_field = _resolve_plan_step_field_reference(step_ref.strip(), output_key, state)
            if step_field is not None:
                return step_field
            continue
        key = output_key.strip()
        if not key or key.lower() in {"result", "output", "audio"}:
            return _audio_path_from_step_output(output)
        audio_path = _audio_path_from_step_output(output, key)
        if audio_path:
            return audio_path
        if key.lower() in {"output_audio_path", "output_audio", "audio_path", "audio"}:
            return _current_audio_for_perception_step_output(output, state)
    output = _step_output_for_ref(token, step_outputs)
    if isinstance(output, dict):
        return _audio_path_from_step_output(output)
    return None


def _resolve_perception_step_current_audio_reference(value: str, state: dict[str, Any]) -> str | None:
    if any(separator in value for separator in (":", ".")):
        return None
    step_outputs = state.get("step_outputs") if isinstance(state.get("step_outputs"), dict) else {}
    output = _step_output_for_ref(value, step_outputs)
    if not isinstance(output, dict) or _audio_path_from_step_output(output):
        return None
    if not _is_perception_step_output_without_audio(output):
        return None
    return _current_audio_for_perception_step_output(output, state)


def _current_audio_for_perception_step_output(output: dict[str, Any], state: dict[str, Any]) -> str | None:
    if not _is_perception_step_output_without_audio(output):
        return None
    current = state.get("current_audio_path") or state.get("input_audio")
    if isinstance(current, str) and current and Path(current).exists():
        return current
    return None


def _is_perception_step_output_without_audio(output: dict[str, Any]) -> bool:
    if _audio_path_from_step_output(output):
        return False
    operation = str(output.get("operation") or "")
    return (
        operation.startswith("perception.")
        or isinstance(output.get("perception_report"), dict)
        or isinstance(output.get("selected_spans"), list)
        or isinstance(output.get("audio_caption_segments"), list)
        or isinstance(output.get("speaker_segments"), list)
    )


def _resolve_plan_step_field_reference(step_ref: str, field_ref: str, state: dict[str, Any]) -> str | None:
    step = _plan_step_for_ref(step_ref, state)
    if step is None:
        return None
    value = _step_field_value(step, field_ref)
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {f"{step_ref}:{field_ref}", f"{step_ref}.{field_ref}"}:
        return None
    resolved = resolve_audio_reference(value, state)
    if isinstance(resolved, str) and resolved.strip():
        return resolved
    return None


def _plan_step_for_ref(step_ref: str, state: dict[str, Any]) -> Any | None:
    plan = state.get("plan")
    steps = getattr(plan, "steps", None)
    if not isinstance(steps, list):
        return None
    if not step_ref:
        step_ref = str(state.get("_current_step_id") or "")
    lowered = step_ref.lower()
    for step in steps:
        step_id = getattr(step, "id", None)
        if isinstance(step_id, str) and step_id.lower() == lowered:
            return step
    return None


def _step_field_value(step: Any, field_ref: str) -> Any:
    parts = [part for part in re.split(r"[:.]", str(field_ref).strip()) if part]
    if not parts:
        return None
    current: Any = step
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _step_output_for_ref(step_ref: str, step_outputs: dict[str, Any]) -> Any:
    if step_ref in step_outputs:
        return step_outputs[step_ref]
    lowered = step_ref.lower()
    for key, value in step_outputs.items():
        if isinstance(key, str) and key.lower() == lowered:
            return value
    return None


def _audio_path_from_step_output(output: dict[str, Any], output_key: str | None = None) -> str | None:
    keys: list[str] = []
    if output_key:
        keys.extend(
            [
                output_key,
                f"{output_key}_path",
                output_key.replace("_audio", "_audio_path"),
            ]
        )
        if output_key == "output_audio":
            keys.append("output_audio_path")
        elif output_key == "target_audio":
            keys.append("target_audio_path")
        elif output_key == "generated_audio":
            keys.append("generated_audio_path")
        elif output_key == "audio":
            keys.append("audio_path")
        elif output_key == "output":
            keys.append("output_path")
    keys.extend(
        [
            "output_audio_path",
            "target_audio_path",
            "generated_audio_path",
            "audio_path",
            "output_path",
            "path",
        ]
    )
    seen: set[str] = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        candidate = output.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def require_existing_audio_path(value: Any, label: str, state: dict[str, Any]) -> str:
    """Resolve aliases and require that the result is an existing audio path."""

    expected_role = (
        _expected_audio_reference_role(label)
        if opaque_audio_resolution_enabled(state)
        else None
    )
    resolved = resolve_audio_reference(
        value,
        state,
        expected_role=expected_role,
    )
    if not isinstance(resolved, str) or not resolved:
        raise FileNotFoundError(f"{label} is required")
    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return str(path)


def _is_audio_reference_key(key: str) -> bool:
    return key in AUDIO_REFERENCE_KEYS or key.endswith(
        ("_audio_path", "_audio_paths")
    )


def _is_known_alias(value: Any, state: dict[str, Any]) -> bool:
    if isinstance(value, list):
        return any(_is_known_alias(item, state) for item in value)
    if not isinstance(value, str):
        return False
    token = value.strip()
    token, _ = _unwrap_template_token(token)
    return bool(token) and not _looks_like_path(token) and token.lower() in build_audio_alias_map(state)


def _unwrap_template_token(value: str) -> tuple[str, bool]:
    token = value.strip()
    if len(token) >= 3 and token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        if inner:
            return inner, True
    if len(token) >= 3 and token.startswith("<") and token.endswith(">"):
        inner = token[1:-1].strip()
        if inner:
            return inner, True
    return token, False


def _looks_like_path(value: str) -> bool:
    if "/" in value or "\\" in value:
        return True
    if value.startswith((".", "~")):
        return True
    return bool(Path(value).suffix)


def _add_alias(aliases: dict[str, str], alias: str, path: str) -> None:
    if alias and path:
        aliases.setdefault(alias.lower(), path)


def _reference_item_aliases(item: dict[str, Any], *, fallback_id: str) -> list[str]:
    aliases: list[str] = []
    for key in ("audio_id", "id", "label", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            aliases.append(value.strip())
    if not aliases:
        aliases.append(fallback_id)
    expanded: list[str] = []
    for alias in aliases:
        expanded.extend(
            [
                alias,
                f"ref:{alias}",
                f"ref_{alias}",
                f"reference:{alias}",
                f"reference_{alias}",
                f"reference_audio:{alias}",
                f"reference_audio_{alias}",
                f"reference_audio_path:{alias}",
                f"reference_audio_path_{alias}",
            ]
        )
    return expanded


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, (str, Path)) and str(item)]
    if isinstance(value, (str, Path)) and str(value):
        return [str(value)]
    return []


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
