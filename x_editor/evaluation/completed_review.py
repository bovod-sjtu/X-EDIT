"""Identity and role checks before an opt-in completed-output critique."""

from __future__ import annotations

import hashlib
from pathlib import Path

from x_editor.evaluation.guidance_contracts import (
    ArtifactQualityEvaluationTask,
    EvaluationTask,
)


def validate_completed_review(
    tasks: list[EvaluationTask], output_path: str | None
) -> None:
    if not output_path:
        raise ValueError("completed-output review requires a delivered artifact")
    path = Path(output_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for task in tasks:
        if (
            task.evaluator_role != "runtime_feedback_critic"
            or task.evaluation_use != "guided_attempt_diagnostic"
        ):
            raise ValueError("outcome judges cannot supply runtime feedback")
        if isinstance(task, ArtifactQualityEvaluationTask):
            if (
                task.focus != "final_delivery"
                or Path(task.output_artifact.path).resolve() != path
            ):
                raise ValueError("critique does not refer to the delivered output")
            observed = task.output_artifact.content_sha256
        elif task.focus == "artifact_measurements":
            payload = task.subject_output.payload
            observed = (
                payload.get("output_sha256") if isinstance(payload, dict) else None
            )
        else:
            raise ValueError(
                "completed-output review requires artifact quality or bound measurements"
            )
        if observed != digest:
            raise ValueError("critique output hash differs from the delivered artifact")
