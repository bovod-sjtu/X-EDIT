"""HTTP client for the external edit-data-pipeline services."""

from __future__ import annotations

import base64
import os
import re
import wave
from html import escape, unescape
from pathlib import Path
from typing import Any

import requests
import soundfile as sf

from x_editor.core.emotion_choices import (
    explicit_emotion_attributes,
    validate_explicit_emotion_choices,
)
from x_editor.core.tagged_text import render_tagged_source
from x_editor.core.text_equivalence import (
    TRANSCRIPT_COPY_EQUIVALENCE,
    equivalent_transcript_copy,
)
from x_editor.tools.catalog._http_retry import (
    post_with_retries,
    retry_attempts,
    retry_response_post,
)
from x_editor.tools.external_services import require_pipeline_service
from x_editor.tools.edit_backend import dots_service_url, uses_dots_model
from x_editor.tools.catalog.edit_data_pipeline.dots_client import apply_dots_edit
from x_editor.tools.catalog.edit_data_pipeline.emotion_transport import bind_emotion_separator_whitespace

DEFAULT_URLS = {
    "text_edit": "http://127.0.0.1:8035",
    "pitch_rate_edit": "http://127.0.0.1:8036",
    "emotion_style_edit": "http://127.0.0.1:8037",
    "pause_edit": "http://127.0.0.1:8038",
    "speech_rate": "http://127.0.0.1:8004",
    "volume_normalize": "http://127.0.0.1:8006",
    "pitch_shift": "http://127.0.0.1:8927",
    "boundary_smooth": "http://127.0.0.1:8018",
    "audio_assembly": "http://127.0.0.1:8021",
}

URL_ENV = {
    "text_edit": "EDIT_PIPELINE_TEXT_TAGGED_URL",
    "pitch_rate_edit": "EDIT_PIPELINE_PITCH_RATE_URL",
    "emotion_style_edit": "EDIT_PIPELINE_EMOTION_STYLE_URL",
    "pause_edit": "EDIT_PIPELINE_PAUSE_EDIT_URL",
    "speech_rate": "EDIT_PIPELINE_SPEECH_RATE_URL",
    "volume_normalize": "EDIT_PIPELINE_VOLUME_URL",
    "pitch_shift": "EDIT_PIPELINE_PITCH_URL",
    "boundary_smooth": "EDIT_PIPELINE_BOUNDARY_SMOOTHER_URL",
    "audio_assembly": "EDIT_PIPELINE_AUDIO_ASSEMBLY_URL",
}

ENDPOINTS = {
    "text_edit": "/execute",
    "pitch_rate_edit": "/execute",
    "emotion_style_edit": "/execute",
    "pause_edit": "/execute",
    "speech_rate": "/stretch",
    "volume_normalize": "/normalize",
    "pitch_shift": "/shift",
    "boundary_smooth": "/smooth",
    "audio_assembly": "/assemble",
}

SPEECH_RATE_RESYNTHESIS_METHODS = {"indextts2", "wsola+indextts2", "auto"}
PITCH_SHIFT_METHODS = {"psola", "psola+qwen3_codec", "psola+indextts2_cco", "psola+mimo_codec"}
DEFAULT_TEXT_EDIT_MAX_INPUT_SEC = 30.0

OPERATION_ALIASES = {
    "text": "text_edit",
    "text_edit_tagged": "text_edit",
    "content_edit": "text_edit",
    "substitution": "text_edit",
    "prosody_edit": "pitch_rate_edit",
    "prosody_edit_pipeline": "pitch_rate_edit",
    "pitch_rate": "pitch_rate_edit",
    "rate": "pitch_rate_edit",
    "speed": "pitch_rate_edit",
    "pitch": "pitch_rate_edit",
    "rate_pipeline": "pitch_rate_edit",
    "speed_pipeline": "pitch_rate_edit",
    "speech_rate_pipeline": "pitch_rate_edit",
    "pitch_pipeline": "pitch_rate_edit",
    "prosody": "pitch_rate_edit",
    "emotion": "emotion_style_edit",
    "emotion_edit": "emotion_style_edit",
    "style_edit": "emotion_style_edit",
    "emotion_style": "emotion_style_edit",
    "pause": "pause_edit",
    "pause_edit_pipeline": "pause_edit",
    "pause_insertion": "pause_edit",
    "speech_rate_conversion": "speech_rate",
    "time_stretch": "speech_rate",
    "stretch": "speech_rate",
    "volume": "volume_normalize",
    "loudness": "volume_normalize",
    "normalize": "volume_normalize",
    "normalise": "volume_normalize",
    "pitch_conversion": "pitch_shift",
    "shift": "pitch_shift",
    "pause_smooth": "boundary_smooth",
    "boundary_pause_smooth": "boundary_smooth",
    "smooth": "boundary_smooth",
    "assemble": "audio_assembly",
    "assembly": "audio_assembly",
    "audio_assemble": "audio_assembly",
}


class EditDataPipelineServiceError(RuntimeError):
    """Stable classification for a failed external edit-data service call."""

    def __init__(
        self,
        *,
        operation: str,
        status_code: int,
        failure_class: str,
        response_text: str,
    ) -> None:
        self.operation = operation
        self.status_code = status_code
        self.failure_class = failure_class
        self.response_text = response_text
        super().__init__(
            f"edit-data-pipeline {operation} service failure "
            f"[{failure_class}] {status_code}: {response_text[:1000]}"
        )


class EditDataPipelineClient:
    """Thin service wrapper that keeps X-Editor independent from pipeline internals."""

    def __init__(self, timeout: float = 600.0) -> None:
        self.timeout = timeout

    def apply(self, arguments: dict[str, Any]) -> dict[str, Any]:
        operation = _normalize_operation(arguments.get("operation"))
        require_pipeline_service(operation)
        output_path = _required_str(arguments, "output_path")

        if operation == "text_edit":
            return self.text_edit(arguments, output_path)
        if operation == "pitch_rate_edit":
            return self.pitch_rate_edit(arguments, output_path)
        if operation == "emotion_style_edit":
            return self.emotion_style_edit(arguments, output_path)
        if operation == "pause_edit":
            return self.pause_edit(arguments, output_path)
        if operation == "speech_rate":
            return self.speech_rate(arguments, output_path)
        if operation == "volume_normalize":
            return self.volume_normalize(arguments, output_path)
        if operation == "pitch_shift":
            return self.pitch_shift(arguments, output_path)
        if operation == "boundary_smooth":
            return self.boundary_smooth(arguments, output_path)
        if operation == "audio_assembly":
            return self.audio_assembly(arguments, output_path)
        raise ValueError(f"Unsupported edit-data-pipeline operation: {operation}")

    def health(self, operation: str) -> dict[str, Any]:
        operation = _normalize_operation(operation)
        url = self._service_url(operation)
        resp = requests.get(f"{url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def text_edit(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        if uses_dots_model("text_edit"):
            return apply_dots_edit("text_edit", arguments, output_path, self.timeout)
        require_pipeline_service("text_edit")
        _enforce_text_edit_segment_duration(arguments)
        return self._post_execute_json(
            "text_edit",
            arguments,
            output_path,
            optional_keys=[
                "language",
                "sample_id",
                "clip_id",
                "generation_id",
                "instructions_nl",
                "reverse_instructions_nl",
                "evaluate",
                "persist",
                "output_dir",
                "duration_mode",
                "synthesis_mode",
                "debug",
                "debug_output_dir",
                "expand_radius",
                "merge_gap",
                "nfe_step",
                "cfg_strength",
                "sway_sampling_coef",
                "source_crossfade_sec",
                "regen_crossfade_sec",
                "seed",
                "disable_prepend_context",
                "fail_if_cannot_prepend_context",
                "min_unmasked_context_sec",
                "min_unmasked_context_ratio",
            ],
        )

    def pitch_rate_edit(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        if uses_dots_model("pitch_rate_edit"):
            return apply_dots_edit("pitch_rate_edit", arguments, output_path, self.timeout)
        require_pipeline_service("pitch_rate_edit")
        return self._post_execute_json(
            "pitch_rate_edit",
            arguments,
            output_path,
            optional_keys=[
                "language",
                "sample_id",
                "clip_id",
                "generation_id",
                "instructions_nl",
                "reverse_instructions_nl",
                "evaluate",
                "persist",
                "output_dir",
                "aligner_backend",
                "pitch_method",
                "rate_method",
                "psola_indextts2_cco_pitch_boost",
                "crossfade_ms",
                "smooth_policy",
                "smooth_margin_sec",
            ],
            field_adapters={
                "aligner_backend": _coerce_qwen3_aligner_backend,
                "instructions_nl": _coerce_prosody_instructions_nl,
                "reverse_instructions_nl": _coerce_prosody_reverse_instructions_nl,
            },
        )

    def emotion_style_edit(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        if uses_dots_model("emotion_style_edit"):
            return apply_dots_edit("emotion_style_edit", arguments, output_path, self.timeout)
        require_pipeline_service("emotion_style_edit")
        return self._post_execute_json(
            "emotion_style_edit",
            arguments,
            output_path,
            optional_keys=[
                "language",
                "sample_id",
                "clip_id",
                "generation_id",
                "instruction_nl",
                "evaluate",
                "persist",
                "output_dir",
                "debug",
                "debug_output_dir",
                "return_synthetic_target_audio",
                "max_candidates",
                "multi_frequent_smoothing_radius_sec",
                "aligner_backend",
                "synthesis_mode",
                "emotion_condition_mode",
            ],
            field_adapters={
                "instruction_nl": _coerce_emotion_instruction_nl,
                "aligner_backend": _coerce_qwen3_aligner_backend,
            },
        )

    def pause_edit(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        if uses_dots_model("pause_edit"):
            return apply_dots_edit("pause_edit", arguments, output_path, self.timeout)
        require_pipeline_service("pause_edit")
        arguments = dict(arguments)
        method = arguments.get("synth_method", arguments.get("stage2_method"))
        # Previous names for the same upstream synthesis strategies.
        method = {"method_one": "single_region", "method_two": "dual_boundary"}.get(method, method)
        if method is not None:
            if method not in {"insert_only", "single_region", "dual_boundary"}:
                raise ValueError("pause_edit synth_method must be insert_only, single_region or dual_boundary")
            arguments["synth_method"] = method
        return self._post_execute_json(
            "pause_edit",
            arguments,
            output_path,
            optional_keys=[
                "language",
                "sample_id",
                "clip_id",
                "profile_id",
                "instructions_nl",
                "evaluate",
                "persist",
                "output_dir",
                "aligner_backend",
                "synth_method",
                "expand_radius",
                "center_regen_second_pass",
                "center_regen_margin_sec",
            ],
            field_adapters={"aligner_backend": _coerce_qwen3_aligner_backend},
        )

    def speech_rate(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        input_audio = _required_audio_path(arguments, "input_audio_path")
        method = str(arguments.get("method") or "wsola+indextts2").strip().lower()
        allow_method_fallback = bool(arguments.get("allow_method_fallback", False))
        payload: dict[str, Any] = {
            "audio": input_audio,
            "rate": float(arguments.get("rate", 1.0)),
            "method": method,
            "output_format": "wav",
        }
        _copy_optional_text(arguments, payload)
        if method in SPEECH_RATE_RESYNTHESIS_METHODS:
            if "text" not in payload:
                raise ValueError(f"text is required for speech_rate method={method!r}")
            speaker_audio = arguments.get("speaker_audio_path") or input_audio
            payload["spk_audio"] = _existing_path(speaker_audio, "speaker_audio_path")
        elif arguments.get("speaker_audio_path"):
            payload["spk_audio"] = _existing_path(arguments["speaker_audio_path"], "speaker_audio_path")
        try:
            return self._post_wav(
                "speech_rate",
                payload,
                output_path,
                arguments.get("service_url"),
                retry_transient=False,
            )
        except RuntimeError as exc:
            if not allow_method_fallback or not _should_retry_speech_rate_with_wsola(
                method, str(exc)
            ):
                raise
            fallback_payload = dict(payload)
            fallback_payload["method"] = "wsola"
            result = self._post_wav("speech_rate", fallback_payload, output_path, arguments.get("service_url"))
            result["method_fallback"] = {
                "from": method,
                "to": "wsola",
                "reason": str(exc)[:500],
            }
            result["recovered_flow"] = True
            return result

    def volume_normalize(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        input_audio = _required_audio_path(arguments, "input_audio_path")
        payload = {
            "audio": input_audio,
            "target_db": float(arguments.get("target_db", -20.0)),
            "method": arguments.get("method") or "rms",
            "headroom_db": float(arguments.get("headroom_db", -1.0)),
            "output_format": "wav",
        }
        return self._post_wav("volume_normalize", payload, output_path, arguments.get("service_url"))

    def pitch_shift(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        input_audio = _required_audio_path(arguments, "input_audio_path")
        if "semitones" not in arguments:
            raise ValueError("pitch_shift requires semitones")
        method = _canonical_pitch_shift_method(arguments.get("method"))
        if method != "psola" and not arguments.get("text"):
            method = "psola"
        payload: dict[str, Any] = {
            "audio": input_audio,
            "semitones": float(arguments["semitones"]),
            "method": method,
            "output_format": "wav",
        }
        _copy_optional(arguments, payload, "text")
        return self._post_wav("pitch_shift", payload, output_path, arguments.get("service_url"))

    def boundary_smooth(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        require_pipeline_service("boundary_smooth")
        input_audio = _required_audio_path(arguments, "input_audio_path")
        text = _required_str(arguments, "text")
        points = arguments.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError("boundary_smooth requires a non-empty points list")

        payload: dict[str, Any] = {
            "audio": _encode_audio_file(input_audio),
            "text": text,
            "points": [_prepare_smoothing_point(point) for point in points],
            "output_format": "wav",
            "debug": bool(arguments.get("debug", False)),
        }
        for key in [
            "aligner_backend",
            "language",
            "f5_steps",
            "f5_cfg_strength",
            "f5_sway_sampling_coef",
            "f5_seed",
            "debug_dir",
            "trace_id",
        ]:
            _copy_optional(arguments, payload, key)
        return self._post_wav("boundary_smooth", payload, output_path, arguments.get("service_url"))

    def audio_assembly(self, arguments: dict[str, Any], output_path: str) -> dict[str, Any]:
        require_pipeline_service("audio_assembly")
        source_audio = arguments.get("source_audio_path") or arguments.get("input_audio_path")
        if not source_audio:
            raise ValueError("audio_assembly requires source_audio_path or input_audio_path")
        target_audio = _required_audio_path(arguments, "target_audio_path")
        edit_tagged_text = _required_str(arguments, "edit_tagged_text")

        payload: dict[str, Any] = {
            "source_audio": _encode_audio_file(_existing_path(source_audio, "source_audio_path")),
            "target_audio": _encode_audio_file(target_audio),
            "edit_tagged_text": edit_tagged_text,
            "preserve_outer_boundaries": bool(arguments.get("preserve_outer_boundaries", True)),
            "aligner_backend": arguments.get("aligner_backend") or "qwen3",
            "output_format": "json",
            "debug": bool(arguments.get("debug", False)),
        }
        for key in [
            "time_spans_from_source",
            "time_spans_from_target",
            "language",
            "smoothing",
            "debug_dir",
            "trace_id",
        ]:
            _copy_optional(arguments, payload, key)
        return self._post_json_audio(
            "audio_assembly",
            payload,
            output_path,
            arguments.get("service_url"),
        )

    def _post_json_audio(
        self,
        operation: str,
        payload: dict[str, Any],
        output_path: str,
        service_url: str | None = None,
    ) -> dict[str, Any]:
        operation = _normalize_operation(operation)
        url = (service_url or self._service_url(operation)).rstrip("/")
        endpoint = ENDPOINTS[operation]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        resp = post_with_retries(
            f"{url}{endpoint}",
            json=payload,
            timeout=self.timeout,
            service_name=f"edit-data-pipeline {operation}",
            retry_env="X_EDITOR_EDIT_PIPELINE_RETRIES",
            backoff_env="X_EDITOR_EDIT_PIPELINE_RETRY_BACKOFF",
            default_retries=1,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"edit-data-pipeline {operation} service error "
                f"{resp.status_code}: {resp.text[:1000]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"edit-data-pipeline {operation} returned non-JSON response"
            ) from exc
        audio = data.get("audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
            raise RuntimeError(
                f"edit-data-pipeline {operation} response missing audio.data"
            )
        output.write_bytes(base64.b64decode(audio["data"]))
        result: dict[str, Any] = {
            "output_audio_path": str(output),
            "operation": operation,
            "service_url": url,
            "response_fields": sorted(key for key in data if key != "audio"),
        }
        result.update(_wav_file_info(output))
        for key in (
            "duration_sec",
            "n_splice_points",
            "n_target_spans",
            "target_spans",
            "details",
            "response_time",
            "trace_id",
        ):
            if key in data:
                result[key] = data[key]
        attempts = retry_attempts(resp)
        if attempts:
            result["request_retries"] = attempts
        return result

    def _post_execute_json(
        self,
        operation: str,
        arguments: dict[str, Any],
        output_path: str,
        *,
        optional_keys: list[str],
        field_aliases: dict[str, str] | None = None,
        field_adapters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retired = {"save_parquet", "parquet_output_dir", "output_dataset_name",
                   "parquet_dataset_name"}
        unsupported = sorted(k for k in retired if arguments.get(k) is not None)
        if unsupported:
            raise ValueError("Pinned pipeline no longer accepts: " + ", ".join(unsupported))
        input_audio = _required_audio_path(arguments, "input_audio_path")
        payload: dict[str, Any] = {
            "audio": _encode_audio_file(input_audio),
            "source_text": _required_str(arguments, "source_text"),
            "instruction_st": _required_str(arguments, "instruction_st"),
        }
        adapters = field_adapters or {}
        for key in optional_keys:
            if key not in arguments or arguments[key] is None:
                continue
            adapter = adapters.get(key)
            if adapter is not None:
                value = adapter(arguments)
                if value is not None:
                    payload[key] = value
                continue
            payload[key] = arguments[key]
        for source_key, target_key in (field_aliases or {}).items():
            if target_key not in payload and source_key in arguments and arguments[source_key] is not None:
                payload[target_key] = arguments[source_key]
        for target_key, adapter in adapters.items():
            if target_key not in payload:
                value = adapter(arguments)
                if value is not None:
                    payload[target_key] = value
        if operation == "pitch_rate_edit" and payload.get("language") is not None:
            payload["language"] = _prosody_service_language(payload["language"])
        emotion_transport = None
        if operation == "emotion_style_edit":
            payload["instruction_st"] = _canonicalize_emotion_instruction_st_for_service(
                payload["instruction_st"]
            )
            canonical_instruction = payload["instruction_st"]
            payload["instruction_st"] = bind_emotion_separator_whitespace(canonical_instruction)
            if payload["instruction_st"] != canonical_instruction:
                emotion_transport = {
                    "authored_instruction_st": arguments["instruction_st"],
                    "service_instruction_st": payload["instruction_st"],
                    "rule": "bind_separator_whitespace_to_adjacent_emotion_segment",
                    "rendered_source_unchanged": True,
                }

        # Services align to the exact text rendered by their label parser.
        # Only the redundant transport copy changes; authored arguments and
        # edit tags remain intact, and different spoken words still fail.
        rendered_source = render_tagged_source(
            payload["instruction_st"],
            lexical=operation == "text_edit",
            decode_entities=operation == "text_edit",
        )
        source_transport: dict[str, str] | None = None
        if payload["source_text"] != rendered_source:
            if not equivalent_transcript_copy(payload["source_text"], rendered_source):
                raise ValueError(
                    f"{operation} instruction_st source does not match source_text"
                )
            source_transport = {
                "authored_source_text": payload["source_text"],
                "service_source_text": rendered_source,
                "equivalence": TRANSCRIPT_COPY_EQUIVALENCE,
            }
            payload["source_text"] = rendered_source

        url = (arguments.get("service_url") or self._service_url(operation)).rstrip("/")
        endpoint = ENDPOINTS[operation]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload_repair: dict[str, Any] | None = None
        request_url = f"{url}{endpoint}"
        resp = post_with_retries(
            request_url,
            json=payload,
            timeout=self.timeout,
            service_name=f"edit-data-pipeline {operation}",
            retry_env="X_EDITOR_EDIT_PIPELINE_RETRIES",
            backoff_env="X_EDITOR_EDIT_PIPELINE_RETRY_BACKOFF",
            default_retries=1,
            retry_statuses=(),
        )
        if resp.status_code != 200:
            repaired_payload, payload_repair = _repair_execute_payload_for_service_error(operation, payload, resp)
            if repaired_payload is not None:
                payload = repaired_payload
                resp = post_with_retries(
                    request_url,
                    json=payload,
                    timeout=self.timeout,
                    service_name=f"edit-data-pipeline {operation}",
                    retry_env="X_EDITOR_EDIT_PIPELINE_RETRIES",
                    backoff_env="X_EDITOR_EDIT_PIPELINE_RETRY_BACKOFF",
                    default_retries=1,
                    retry_statuses=(),
                )
        if resp.status_code != 200:
            resp = retry_response_post(
                resp,
                request_url,
                json=payload,
                timeout=self.timeout,
                service_name=f"edit-data-pipeline {operation}",
                retry_env="X_EDITOR_EDIT_PIPELINE_RETRIES",
                backoff_env="X_EDITOR_EDIT_PIPELINE_RETRY_BACKOFF",
                default_retries=1,
            )
        if resp.status_code != 200:
            raise EditDataPipelineServiceError(
                operation=operation,
                status_code=resp.status_code,
                failure_class=_classify_execute_service_failure(
                    operation,
                    status_code=resp.status_code,
                    response_text=resp.text,
                ),
                response_text=resp.text,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"edit-data-pipeline {operation} returned non-JSON response") from exc
        audio = data.get("audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
            raise RuntimeError(f"edit-data-pipeline {operation} response missing audio.data")
        output.write_bytes(base64.b64decode(audio["data"]))
        wav_info = _wav_file_info(output)
        response_keys = sorted(key for key in data.keys() if key != "audio")
        result: dict[str, Any] = {
            "output_audio_path": str(output),
            "operation": operation,
            "service_url": url,
            "response_fields": response_keys,
            "backend": "edit_data_pipeline",
        }
        if payload_repair:
            result["request_payload_repair"] = payload_repair
        if source_transport:
            result["source_text_transport"] = source_transport
        if emotion_transport:
            result["emotion_label_transport"] = emotion_transport
        attempts = retry_attempts(resp)
        if attempts:
            result["request_retries"] = attempts
        result.update(wav_info)
        for key in [
            "status",
            "pipeline_type",
            "pipeline_version",
            "sample_id",
            "generation_id",
            "source_text",
            "target_text",
            "target_tagged_text",
            "edit_tagged_text",
            "generated_text",
            "pause_edit_plan",
            "changed_segments",
            "plain_text",
            "instruction_st",
            "source_spans",
            "target_spans",
            "target_aligned_spans",
            "edit_result",
            "edit_pairs",
            "eval_results",
            "metadata",
            "debug",
        ]:
            if key in data:
                result[key] = data[key]
        return result

    def _post_wav(
        self,
        operation: str,
        payload: dict[str, Any],
        output_path: str,
        service_url: str | None = None,
        *,
        retry_transient: bool = True,
    ) -> dict[str, Any]:
        operation = _normalize_operation(operation)
        url = (service_url or self._service_url(operation)).rstrip("/")
        endpoint = ENDPOINTS[operation]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if retry_transient:
            resp = post_with_retries(
                f"{url}{endpoint}",
                json=payload,
                timeout=self.timeout,
                service_name=f"edit-data-pipeline {operation}",
                retry_env="X_EDITOR_EDIT_PIPELINE_RETRIES",
                backoff_env="X_EDITOR_EDIT_PIPELINE_RETRY_BACKOFF",
                default_retries=1,
            )
        else:
            resp = post_with_retries(
                f"{url}{endpoint}",
                json=payload,
                timeout=self.timeout,
                service_name=f"edit-data-pipeline {operation}",
                retry_env="X_EDITOR_EDIT_PIPELINE_RETRIES",
                backoff_env="X_EDITOR_EDIT_PIPELINE_RETRY_BACKOFF",
                default_retries=1,
                retry_statuses=(),
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"edit-data-pipeline {operation} service error {resp.status_code}: {resp.text[:1000]}"
            )
        output.write_bytes(resp.content)
        headers = {key: value for key, value in resp.headers.items() if key.lower().startswith("x-")}
        result = {
            "output_audio_path": str(output),
            "operation": operation,
            "service_url": url,
            "sample_rate": _first_header_int(resp, ["X-Sample-Rate"]),
            "duration_seconds": _first_header_float(
                resp,
                [
                    "X-Duration-Sec",
                    "X-Duration-Seconds",
                    "X-Stretched-Duration",
                    "X-Shifted-Duration",
                ],
            ),
            "response_headers": headers,
        }
        attempts = retry_attempts(resp)
        if attempts:
            result["request_retries"] = attempts
        return result

    def _service_url(self, operation: str) -> str:
        if uses_dots_model(operation):
            return dots_service_url()
        require_pipeline_service(operation)
        env_name = URL_ENV[operation]
        value = os.getenv(env_name) or DEFAULT_URLS[operation]
        if not value:
            raise ValueError(
                f"No default service URL is configured for {operation}; set {env_name} "
                "or pass service_url explicitly."
            )
        return value


def _normalize_operation(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("operation is required")
    normalized = value.strip().lower().replace("-", "_")
    normalized = OPERATION_ALIASES.get(normalized, normalized)
    if normalized not in ENDPOINTS:
        raise ValueError(f"Unsupported operation: {value}")
    return normalized


def _repair_execute_payload_for_service_error(
    operation: str,
    payload: dict[str, Any],
    response: requests.Response,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return a one-shot compatibility retry payload for known service-schema drift."""

    repaired = dict(payload)
    dropped: list[str] = []
    forbidden = _extra_forbidden_fields(response)
    for key in forbidden:
        if key in repaired:
            repaired.pop(key, None)
            dropped.append(key)
    text = (getattr(response, "text", "") or "").lower()
    if operation == "text_edit" and "prepend context is required but disabled" in text:
        for key in [
            "disable_prepend_context",
            "fail_if_cannot_prepend_context",
            "min_unmasked_context_sec",
            "min_unmasked_context_ratio",
        ]:
            if key in repaired:
                repaired.pop(key, None)
                dropped.append(key)
    if not dropped or repaired == payload:
        return None, None
    return repaired, {
        "reason": "service rejected optional compatibility fields",
        "status_code": getattr(response, "status_code", None),
        "dropped_fields": sorted(set(dropped)),
    }


def _classify_execute_service_failure(
    operation: str,
    *,
    status_code: int,
    response_text: str,
) -> str:
    """Classify service failures without changing the requested operation."""

    lowered = str(response_text or "").lower()
    if operation in {
        "text_edit",
        "pitch_rate_edit",
        "emotion_style_edit",
        "pause_edit",
    } and any(
        marker in lowered
        for marker in (
            "alignment failed",
            "aligner failed",
            "forced alignment",
            "alignment service",
        )
    ):
        return "alignment_backend_failure"
    if status_code in {400, 409, 422}:
        return "request_contract_rejected"
    if status_code >= 500:
        return "service_internal_failure"
    return "service_http_failure"


def _extra_forbidden_fields(response: requests.Response) -> list[str]:
    try:
        payload = response.json()
    except Exception:
        return []
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(detail, list):
        return []
    fields: list[str] = []
    for item in detail:
        if not isinstance(item, dict) or item.get("type") != "extra_forbidden":
            continue
        loc = item.get("loc")
        if isinstance(loc, list) and len(loc) >= 2 and loc[0] == "body":
            fields.append(str(loc[1]))
    return fields


def _should_retry_speech_rate_with_wsola(method: str, error: str) -> bool:
    if method not in SPEECH_RATE_RESYNTHESIS_METHODS:
        return False
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in [
            "extract_tokens",
            "indextts",
            "localhost', port=8002",
            "localhost:8002",
            "connection refused",
        ]
    )


def _canonical_pitch_shift_method(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "psola_qwen3_codec": "psola+qwen3_codec",
        "psola_indextts2_cco": "psola+indextts2_cco",
        "psola_mimo_codec": "psola+mimo_codec",
        "psola+qwen3+codec": "psola+qwen3_codec",
        "psola+indextts2+cco": "psola+indextts2_cco",
        "psola+mimo+codec": "psola+mimo_codec",
    }
    canonical = aliases.get(text, text)
    if canonical in PITCH_SHIFT_METHODS:
        return canonical
    return "psola"


def _prosody_service_language(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-")
    if text in {"eng", "en", "english", "en-us", "en-gb"}:
        return "en"
    if text in {"zho", "cmn", "yue", "zh", "chinese", "mandarin", "cantonese", "zh-cn", "zh-hans"}:
        return "zh"
    return str(value)


def _required_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value


def _required_audio_path(arguments: dict[str, Any], key: str) -> str:
    return _existing_path(_required_str(arguments, key), key)


def _enforce_text_edit_segment_duration(arguments: dict[str, Any]) -> None:
    input_audio = _required_audio_path(arguments, "input_audio_path")
    max_sec = _text_edit_max_input_sec()
    duration = _audio_duration_seconds(Path(input_audio))
    if duration is None:
        raise ValueError(
            "text_edit input must be a frontend-cut audio segment shorter than "
            f"{max_sec:g}s, but duration could not be verified for {input_audio}."
        )
    if duration >= max_sec:
        raise ValueError(
            "text_edit input must be a frontend-cut audio segment shorter than "
            f"{max_sec:g}s; got {duration:.3f}s. Split on word boundaries before calling text_edit."
        )


def _text_edit_max_input_sec() -> float:
    raw = os.getenv("X_EDITOR_TEXT_EDIT_MAX_INPUT_SEC")
    if raw is None:
        return DEFAULT_TEXT_EDIT_MAX_INPUT_SEC
    value = float(raw)
    if value <= 0:
        raise ValueError("X_EDITOR_TEXT_EDIT_MAX_INPUT_SEC must be positive")
    return min(value, DEFAULT_TEXT_EDIT_MAX_INPUT_SEC)


def _audio_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            return frames / sample_rate if sample_rate else None
    except (wave.Error, OSError, EOFError):
        pass
    # Python's wave reader excludes IEEE-float WAV (including gain artifacts).
    # Preserve the PCM fast path and verify other WAV headers with libsndfile.
    try:
        with sf.SoundFile(str(path), "r") as audio:
            if audio.format not in {"WAV", "WAVEX"} or audio.samplerate <= 0 or audio.frames <= 0:
                return None
            if len(audio.read(frames=1, dtype="float64", always_2d=True)) != 1:
                return None
            return audio.frames / audio.samplerate
    except (RuntimeError, OSError, ValueError):
        return None


def _existing_path(value: str, label: str) -> str:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {value}")
    return str(path)


def _copy_optional(source: dict[str, Any], target: dict[str, Any], key: str) -> None:
    if key in source and source[key] is not None:
        target[key] = source[key]


def _copy_optional_text(source: dict[str, Any], target: dict[str, Any]) -> None:
    text = source.get("text") or source.get("source_text")
    if text is not None:
        target["text"] = text


def _coerce_natural_language_instruction(value: Any) -> dict[str, str] | None:
    """Coerce legacy NL metadata into the latest prosody service shape."""

    if isinstance(value, str) and value.strip():
        text = value.strip()
        return {"precise": text, "vague": text}
    if not isinstance(value, dict):
        return None
    precise = _first_nonempty_string(value.get("precise"))
    vague = _first_nonempty_string(value.get("vague"))
    fallback = precise or vague
    if not fallback:
        for item in value.values():
            fallback = _first_nonempty_string(item)
            if fallback:
                break
    if not fallback:
        return None
    return {"precise": precise or fallback, "vague": vague or fallback}


def _coerce_prosody_instructions_nl(arguments: dict[str, Any]) -> dict[str, str] | None:
    return _coerce_natural_language_instruction(arguments.get("instructions_nl"))


def _coerce_prosody_reverse_instructions_nl(arguments: dict[str, Any]) -> dict[str, str] | None:
    return _coerce_natural_language_instruction(arguments.get("reverse_instructions_nl"))


def _coerce_emotion_instruction_nl(arguments: dict[str, Any]) -> str | None:
    """Adapt older X-Editor instruction metadata to the latest emotion service."""

    value = arguments.get("instruction_nl")
    if isinstance(value, str) and value.strip():
        return value
    value = arguments.get("instructions_nl")
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ("precise", "vague"):
            first = _first_nonempty_string(value.get(key))
            if first:
                return first
        for item in value.values():
            first = _first_nonempty_string(item)
            if first:
                return first
    return None


def _coerce_qwen3_aligner_backend(arguments: dict[str, Any]) -> str | None:
    value = arguments.get("aligner_backend")
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if not text:
        return None
    if text in {"auto", "default", "legacy", "qwen", "qwen3", "qwen3_omni", "qwen_3"}:
        return "qwen3"
    raise ValueError("Pinned editing pipeline supports the qwen3 aligner only")


def _canonicalize_emotion_instruction_st_for_service(instruction_st: str) -> str:
    """Translate the declared emotion/intensity without quantizing or inferring it."""
    validate_explicit_emotion_choices(instruction_st)

    def native(attrs: str, body: str) -> str:
        tag, level, alpha = explicit_emotion_attributes(attrs)
        key, value = ("level", level) if level is not None else ("alpha", alpha)
        return f'<emo type="{tag}" {key}="{escape(str(value), quote=True)}">{body}</emo>'

    converted = re.sub(
        r"<emo\b(?P<attrs>[^>]*)>(?P<body>.*?)</emo>",
        lambda m: native(m.group("attrs"), m.group("body")),
        instruction_st, flags=re.IGNORECASE | re.DOTALL,
    )
    legacy = re.compile(
        r"<(?P<tag>happy|angry|sad|afraid|disgusted|melancholic|surprised|calm)"
        r"\s*,\s*(?P<kind>level|alpha)=(?P<value>[^>]+)>(?P<body>.*?)</(?P=tag)>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return legacy.sub(
        lambda m: native(f'type="{m.group("tag")}" {m.group("kind").lower()}="{m.group("value")}"',
                         m.group("body")), converted,
    )


def _first_nonempty_string(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        for item in value:
            text = _first_nonempty_string(item)
            if text:
                return text
    return None


def _encode_audio_file(path: str) -> dict[str, str]:
    audio_path = Path(path)
    suffix = audio_path.suffix.lower().lstrip(".") or "wav"
    return {
        "data": base64.b64encode(audio_path.read_bytes()).decode("utf-8"),
        "format": suffix,
    }


def _prepare_smoothing_point(point: Any) -> dict[str, Any]:
    if not isinstance(point, dict):
        raise ValueError("Each boundary_smooth point must be an object")
    prepared = dict(point)
    insert_path = prepared.pop("audio_insert_path", None)
    if insert_path:
        prepared["audio_insert"] = _encode_audio_file(_existing_path(str(insert_path), "audio_insert_path"))
    return prepared


def _first_header_int(resp: requests.Response, names: list[str]) -> int | None:
    value = _first_header(resp, names)
    return int(value) if value is not None else None


def _first_header_float(resp: requests.Response, names: list[str]) -> float | None:
    value = _first_header(resp, names)
    return float(value) if value is not None else None


def _first_header(resp: requests.Response, names: list[str]) -> str | None:
    for name in names:
        value = resp.headers.get(name)
        if value:
            return value
    return None


def _wav_file_info(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            return {
                "sample_rate": sample_rate,
                "duration_seconds": frames / sample_rate if sample_rate else None,
            }
    except (wave.Error, OSError, EOFError):
        return {}
