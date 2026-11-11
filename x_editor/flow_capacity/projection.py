"""Prompt-safe projections and leakage audits for flow-capacity releases."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .identity import content_sha256
from .schemas import RuntimeCaseRecord


_KEY_NORMALIZER = re.compile(r"[^a-z0-9]+")

FORBIDDEN_RUNTIME_KEYS = {
    "benchmark",
    "benchmarkcase",
    "benchmarkid",
    "benchmarkidentity",
    "caseid",
    "datasetid",
    "expectedaudio",
    "expectededitgraph",
    "expectedoutput",
    "expectedtranscript",
    "hiddenevaluatorfields",
    "label",
    "labels",
    "officialanswer",
    "officialrubric",
    "rubric",
    "rubrics",
    "sampleid",
    "scorer",
    "scoreroutput",
    "scorerrubric",
    "shard",
    "sidecarrecordkey",
    "split",
    "suite",
    "targetaudio",
    "targetwav",
}

FORBIDDEN_STANDARD_KEYS = {
    "benchmark",
    "benchmarkcase",
    "benchmarkid",
    "benchmarkidentity",
    "datasetid",
    "evaluationobjectives",
    "expectedscore",
    "hiddenevaluatorfields",
    "officialanswer",
    "officialrubric",
    "labels",
    "rubric",
    "rubrics",
    "sampleid",
    "scorer",
    "scorerid",
    "scoreroutput",
    "scorerrubric",
    "sidecarrecordkey",
    "shard",
    "split",
    "suite",
    "targetaudio",
    "targetwav",
}


def normalize_key(value: str) -> str:
    return _KEY_NORMALIZER.sub("", value.lower())


def find_forbidden_key_paths(
    value: Any,
    *,
    forbidden_keys: set[str],
    prefix: str = "$",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if normalize_key(str(key)) in forbidden_keys:
                paths.append(path)
            paths.extend(find_forbidden_key_paths(child, forbidden_keys=forbidden_keys, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(
                find_forbidden_key_paths(
                    child,
                    forbidden_keys=forbidden_keys,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return paths


def project_runtime_case(record: RuntimeCaseRecord) -> dict[str, Any]:
    """Return the only fields a resolver may expose to runtime modules."""

    payload: dict[str, Any] = {
        "request": record.request,
        "audio_assets": [
            {
                "asset_ref": item.asset_ref,
                "role": item.role,
                "order": item.order,
            }
            for item in sorted(record.audio_assets, key=lambda item: item.order)
        ],
        "visible_context": record.visible_context,
        "session_history": record.session_history,
    }
    if record.source_text is not None:
        payload["source_text"] = record.source_text
    return payload


def rendered_runtime_hash(record: RuntimeCaseRecord) -> str:
    return content_sha256(project_runtime_case(record))


def audit_runtime_projection(record: RuntimeCaseRecord) -> list[str]:
    return find_forbidden_key_paths(
        project_runtime_case(record),
        forbidden_keys=FORBIDDEN_RUNTIME_KEYS,
    )
