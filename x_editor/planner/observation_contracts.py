"""Planner-visible audio cardinality contracts; no automatic job rewriting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from x_editor.perception.workflow_catalog import workflow_perception_capability


def observation_input_contracts(
    catalog: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose the declared input mode as an actionable observation job shape."""
    contracts = []
    for capability in catalog:
        operation = capability.get("operation")
        mode = capability.get("audio_input_mode")
        if not operation or mode not in {"single_audio", "paired_audio", "multi_audio"}:
            continue
        contracts.append(
            {
                "operation": operation,
                "audio_input_mode": mode,
                "primary_audio_count": 1,
                "supporting_audio_count": (
                    0 if mode == "single_audio" else 1 if mode == "paired_audio" else "at_least_one"
                ),
            }
        )
    return contracts


class PerceptionJobShapeError(ValueError):
    """A correctable structural failure with the original asset binding retained.

    Feedback explains a legal next plan, but never authorizes a retry or chooses
    which asset to observe. Controllers retain their configured call budgets.
    """

    def __init__(
        self,
        *,
        job_id: str,
        audio_refs: Sequence[str],
        route_errors: Mapping[str, str],
    ) -> None:
        self.feedback = {
            "code": "perception_job_audio_shape_invalid",
            "job_id": job_id,
            "audio_refs": list(audio_refs),
            "route_errors": dict(route_errors),
            "expected_input_contracts": observation_input_contracts(
                [
                    capability.to_dict()
                    for operation in route_errors
                    if (capability := workflow_perception_capability(operation)) is not None
                ]
            ),
            "automatic_retry_authorized": False,
            "suggested_action": (
                "For independent measurements, author one job per asset with its own "
                "primary_audio_ref and no supporting_audio_refs; compare the returned "
                "asset-bound evidence afterwards. A genuinely joint observation must "
                "use a declared paired/multi-audio capability. Do not drop an asset, "
                "change the selected editing base, or infer an unobserved value."
            ),
        }
        super().__init__(
            f"Perception job {job_id!r} has an impossible audio shape: "
            f"{dict(route_errors)}; audio_refs={list(audio_refs)!r}. "
            + self.feedback["suggested_action"]
        )
