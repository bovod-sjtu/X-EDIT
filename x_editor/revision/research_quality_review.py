"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import time


from pathlib import Path


from x_editor.revision.artifact_io import audio_ref, write_json


from x_editor.evaluation.audio_evidence_client import AudioEvidenceEvaluationClient


class RecordedJudgeClient(AudioEvidenceEvaluationClient):
    def __init__(self, *, directory: Path, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.sequence = 0

    def call_json(self, system_prompt, user_text, **kwargs):
        self.sequence += 1
        path = self.directory / f"{self.sequence:04d}"
        path.mkdir(parents=True, exist_ok=False)
        write_json(
            path / "request.json",
            {"system_prompt": system_prompt, "user_text": user_text, **kwargs,
             "sampling": {"temperature": self.temperature, "seed": self.seed}},
        )
        start = time.monotonic()
        try:
            response = super().call_json(system_prompt, user_text, **kwargs)
            write_json(path / "response.json", response)
            return response
        finally:
            write_json(
                path / "receipt.json",
                {
                    "elapsed_sec": time.monotonic() - start,
                    "http_attempts": self.last_transport_attempt_count,
                    "metrics": self.last_call_metrics,
                    "raw_text": self.last_raw_text,
                },
            )
