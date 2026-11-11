"""Text Agent: Gemini 3.1 Flash for planning and output generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from x_editor.agents.base import GeminiClient


class TextAgent:
    """
    Text-only agent using Gemini 3.1 Flash.
    Responsible for breaking down user request into structured edit sequence.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 8192,
        timeout: float = 30.0,
    ) -> None:
        self.client = GeminiClient(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._load_prompts()

    def _load_prompts(self) -> None:
        prompt_dir = Path(__file__).parent.parent / "prompts"
        self.structuring_system = (prompt_dir / "structuring_system.md").read_text(encoding="utf-8")
        self.structuring_user = (prompt_dir / "structuring_user.md").read_text(encoding="utf-8")

    def plan(
        self,
        user_prompt: str,
        intent_description: str,
        audio_perception: dict[str, Any],
        source_text: str,
    ) -> dict[str, Any]:
        """
        Generate structured edit sequence.
        Returns dict with planning_status, source_text, edits, notes.
        """
        user_text = self.structuring_user.format(
            user_prompt=user_prompt,
            intent_description=intent_description,
            audio_perception_json=json.dumps(audio_perception, indent=2, ensure_ascii=False),
            source_text=source_text,
        )

        response_schema = {
            "type": "object",
            "properties": {
                "planning_status": {
                    "type": "string",
                    "enum": ["READY", "NEED_CLARIFICATION", "FAIL"],
                },
                "source_text": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["text_edit", "emotion_edit", "prosody_edit", "vc_edit", "se_edit"],
                            },
                            "source_text": {"type": "string"},
                            "target_text": {"type": "string"},
                            "source_audio": {"type": "string"},
                            "instruction": {"type": "string"},
                            "reference_audio": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                        "required": ["type", "source_text", "target_text", "source_audio"],
                    },
                },
                "clarification_question": {"type": "string"},
                "clarification_focus": {
                    "type": "string",
                    "enum": ["pitch", "timbre", "dynamics", "effects", "reference_comparison", "other"],
                },
                "notes": {"type": "string"},
            },
            "required": ["planning_status", "source_text", "edits"],
        }

        return self.client.call_json(
            system_prompt=self.structuring_system,
            user_text=user_text,
        )
