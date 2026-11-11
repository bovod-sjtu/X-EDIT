"""Evaluator-only materialization of bounded audio evidence clips."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from x_editor.evaluation.guidance_contracts import (
    ArtifactQualityEvaluationTask,
    MaterializedEvaluationAudioClip,
)


def materialize_artifact_evaluation_clips(
    task: ArtifactQualityEvaluationTask,
    *,
    output_dir: str | Path,
    minimum_crop_sec: float = 0.01,
) -> ArtifactQualityEvaluationTask:
    """Attach exact local clips without changing frozen rubrics or selection.

    Regions are already part of the task contract. This function only decodes
    their named artifacts, applies the frozen context margin, and writes bounded
    PCM clips for an audio-capable Evaluator. A global task with no regions keeps
    its full-artifact attachments.
    """

    if minimum_crop_sec <= 0.0:
        raise ValueError("minimum_crop_sec must be positive")
    if not task.regions:
        return task
    artifacts = {
        item.artifact_ref_id: item
        for item in (
            *task.input_artifacts,
            task.output_artifact,
            *task.reference_artifacts,
        )
    }
    destination_root = Path(output_dir).expanduser().resolve() / _safe_id(task.task_id)
    clips: list[MaterializedEvaluationAudioClip] = []
    for index, region in enumerate(task.regions):
        artifact = artifacts[region.artifact_ref_id]
        source = Path(artifact.path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"evaluation artifact is unavailable: {source}")
        observed_hash = _sha256_file(source)
        if observed_hash != artifact.content_sha256:
            raise ValueError(f"evaluation artifact hash mismatch for {artifact.artifact_ref_id}")
        info = sf.info(str(source))
        duration_sec = info.frames / info.samplerate
        if region.end_sec > duration_sec + 1e-3:
            raise ValueError(
                f"region {region.region_id} exceeds artifact duration {duration_sec:.6f}"
            )
        clip_start = max(0.0, region.start_sec - task.context_padding_sec)
        clip_end = min(duration_sec, region.end_sec + task.context_padding_sec)
        if clip_end - clip_start < minimum_crop_sec:
            raise ValueError(f"region {region.region_id} has no usable crop duration")
        start_frame = round(clip_start * info.samplerate)
        end_frame = round(clip_end * info.samplerate)
        audio, sample_rate = sf.read(
            str(source),
            start=start_frame,
            stop=end_frame,
            always_2d=True,
            dtype="float32",
        )
        if audio.size == 0:
            raise ValueError(f"region {region.region_id} decoded no samples")
        destination = destination_root / f"{index:03d}-{_safe_id(region.region_id)}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(destination),
            np.clip(audio, -1.0, 1.0),
            sample_rate,
            subtype="PCM_16",
        )
        actual_clip_end = clip_start + len(audio) / sample_rate
        target_start = region.start_sec - clip_start
        target_end = min(region.end_sec - clip_start, actual_clip_end - clip_start)
        actual_padding = max(
            region.start_sec - clip_start,
            actual_clip_end - region.end_sec,
            0.0,
        )
        clips.append(
            MaterializedEvaluationAudioClip(
                clip_id=f"clip:{task.task_id}:{region.region_id}",
                region_id=region.region_id,
                source_artifact_ref_id=region.artifact_ref_id,
                path=str(destination),
                content_sha256=_sha256_file(destination),
                original_start_sec=region.start_sec,
                original_end_sec=region.end_sec,
                clip_start_sec=clip_start,
                clip_end_sec=actual_clip_end,
                target_start_in_clip_sec=target_start,
                target_end_in_clip_sec=target_end,
                context_padding_sec=actual_padding,
            )
        )
    payload = task.model_dump(mode="json")
    payload["materialized_clips"] = [item.model_dump(mode="json") for item in clips]
    return ArtifactQualityEvaluationTask.model_validate(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return normalized or "evaluation"


__all__ = ["materialize_artifact_evaluation_clips"]
