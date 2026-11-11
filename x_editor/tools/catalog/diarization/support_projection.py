"""Project verified DiariZen annotations onto the bytes actually decoded.

This adapter contract covers input coordinates and right zero padding. It does
not assert speaker correctness, remove overlap, or repair other timestamp APIs.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SupportedDiarization:
    annotation: Any
    raw_annotation: list[dict[str, Any]]
    source_support_projection: dict[str, Any]


def verified_diarizen_contract(pipeline: Any) -> dict[str, Any]:
    """Require the reviewed implementation, not a matching model-name string."""
    from diarizen.pipelines.inference import DiariZenPipeline
    from pyannote.audio.core.inference import Inference

    if type(pipeline) is not DiariZenPipeline:
        raise ValueError("source support is unverified for this pipeline implementation")
    segmentation = getattr(pipeline, "_segmentation", None)
    if type(segmentation) is not Inference or segmentation.skip_aggregation is not True:
        raise ValueError("DiariZen segmentation does not match the reviewed raw-window contract")
    window = float(segmentation.duration)
    if not math.isfinite(window) or window <= 0:
        raise ValueError("DiariZen segmentation has no finite positive window duration")
    contract = json.loads(Path(__file__).with_name("source_support_contract.json").read_text())
    observed = {}
    for name, expected in contract["source_files"].items():
        path = Path(inspect.getfile(importlib.import_module(name)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"DiariZen input-coordinate source contract changed: {name}")
        observed[name] = {"path": str(path), "sha256": digest}
    return {**contract, "observed_sources": observed, "window_duration_sec": window}


def decoded_input(payload: bytes, audio: Path, expected_sha256: str | None) -> dict[str, Any]:
    import torchaudio

    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("diarization input bytes differ from the requested asset revision")
    waveform, sample_rate = torchaudio.load(BytesIO(payload))
    if waveform.ndim != 2 or waveform.shape[-1] <= 0 or sample_rate <= 0:
        raise ValueError("diarization input has no measurable decoded support")
    frames = int(waveform.shape[-1])
    return {
        "namespace": "x-editor.diarization.input-bytes/v1",
        "path": str(audio.resolve()),
        "content_sha256": digest,
        "content_revision": f"sha256:{digest}",
        "sample_rate": int(sample_rate),
        "frames": frames,
        "channels": int(waveform.shape[0]),
        "duration_sec": frames / int(sample_rate),
        "decoder": "torchaudio.load(BytesIO)",
    }


def annotation_rows(annotation: Any) -> list[dict[str, Any]]:
    return [
        {
            "track": str(track),
            "speaker_id": str(speaker),
            "start_sec": float(segment.start),
            "end_sec": float(segment.end),
        }
        for segment, track, speaker in annotation.itertracks(yield_label=True)
    ]


def project_annotation(
    annotation: Any, *, asset: dict[str, Any], contract: dict[str, Any]
) -> SupportedDiarization:
    from pyannote.core import Annotation, Segment

    if not isinstance(annotation, Annotation):
        raise TypeError("DiariZen did not return its declared Annotation output")
    if (
        contract.get("coordinate_system") != "input_audio_seconds"
        or contract.get("padding") != "right_zero_padding_only"
    ):
        raise ValueError("annotation has no verified source-coordinate padding contract")
    duration = asset["frames"] / asset["sample_rate"]
    if not math.isfinite(duration) or duration <= 0 or duration != asset["duration_sec"]:
        raise ValueError("decoded audio support is inconsistent")
    raw = annotation_rows(annotation)
    # A raw final window can extend by at most one complete window. Negative
    # time or a different coordinate scale is outside this adapter's authority.
    bound = duration + contract["window_duration_sec"]
    for row in raw:
        start, end = row["start_sec"], row["end_sec"]
        if (
            not all(math.isfinite(value) for value in (start, end))
            or start < 0
            or end <= start
            or end > bound
        ):
            raise ValueError("annotation exceeds the verified right-padding support")
    source_support = Segment(0.0, duration)
    bounded = annotation.crop(source_support, mode="intersection")
    projected = annotation_rows(bounded)
    # Carry raw row indices through the same native operation. Native crop can
    # rename colliding tracks, so matching only speaker text would lose lineage.
    indexed = annotation.copy()
    for raw_index, (segment, track, _) in enumerate(annotation.itertracks(yield_label=True)):
        indexed[segment, track] = raw_index
    indexed_projection = list(
        indexed.crop(source_support, mode="intersection").itertracks(yield_label=True)
    )
    projection_map = [{"raw_index": index, "projected_indices": []} for index in range(len(raw))]
    if len(indexed_projection) != len(projected):
        raise ValueError("native annotation projection lost index lineage")
    for projected_index, ((segment, track, raw_index), row) in enumerate(
        zip(indexed_projection, projected)
    ):
        if (float(segment.start), float(segment.end), str(track)) != (
            row["start_sec"],
            row["end_sec"],
            row["track"],
        ):
            raise ValueError("native annotation projection changed index lineage")
        projection_map[int(raw_index)]["projected_indices"].append(projected_index)
    for entry in projection_map:
        if entry["projected_indices"]:
            entry["status"] = "kept"
        elif raw[entry["raw_index"]]["start_sec"] >= duration:
            entry["status"] = "outside_decoded_support"
        else:
            entry["status"] = "native_intersection_empty"
    receipt = {
        "schema_version": "x-editor.diarization.source-support-projection/v1",
        "coordinate_system": "input_audio_seconds",
        "operation": "Annotation.crop(Segment(0, decoded_frames/sample_rate), intersection)",
        "input_asset": asset,
        "backend_contract": contract,
        "raw_annotation_namespace": "x-editor.diarization.raw-annotation/v1",
        "projected_annotation_namespace": "x-editor.diarization.source-annotation/v1",
        "projected_annotation": projected,
        "raw_to_projected": projection_map,
        "changed": raw != projected,
        "fully_outside_input": [row for row in raw if row["start_sec"] >= duration],
        "padding_authorizes_target": False,
        "overlap_policy": "preserved by native Annotation intersection",
    }
    return SupportedDiarization(bounded, raw, receipt)


def run_with_source_support(
    pipeline: Any, audio: Path, *, expected_sha256: str | None = None
) -> SupportedDiarization:
    contract = verified_diarizen_contract(pipeline)
    payload = audio.read_bytes()
    asset = decoded_input(payload, audio, expected_sha256)
    # The decoder and inference consume independent cursors on identical bytes;
    # a later filesystem replacement cannot change this call's input version.
    annotation = pipeline(BytesIO(payload), sess_name=audio.stem)
    return project_annotation(annotation, asset=asset, contract=contract)
