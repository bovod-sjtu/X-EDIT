"""Omni Agent: Gemini 2.5 Pro with audio perception capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from x_editor.agents.base import GeminiClient
from x_editor.core.schemas import PerceptionOutput


class OmniAgent:
    """
    Omni-modal agent using Gemini 2.5 Pro.
    Responsible for audio perception and intent clarification.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 40960,
        timeout: float = 60.0,
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
        self.perception_system = (prompt_dir / "perception_system.md").read_text(encoding="utf-8")
        self.perception_user = (prompt_dir / "perception_user.md").read_text(encoding="utf-8")
        self.clarification_system = (prompt_dir / "clarification_system.md").read_text(encoding="utf-8")
        self.clarification_user = (prompt_dir / "clarification_user.md").read_text(encoding="utf-8")

    async def perceive(
        self,
        user_prompt: str,
        base_audio_path: str,
        reference_audio_paths: list[str] | None = None,
    ) -> PerceptionOutput:
        """
        Perform initial audio perception.
        Returns structured perception report + intent description + source_text.
        """
        refs = reference_audio_paths or []
        all_audios = [base_audio_path] + refs

        user_text = self.perception_user.format(
            user_prompt=user_prompt,
            base_audio_filename=Path(base_audio_path).name,
            reference_audio_list=str([Path(p).name for p in refs]),
        )

        response_schema = {
            "type": "object",
            "properties": {
                "source_text": {"type": "string"},
                "base_audio_analysis": {
                    "type": "object",
                    "properties": {
                        "pitch": {"type": "object", "properties": {
                            "level": {"type": "string"},
                            "description": {"type": "string"},
                        }},
                        "gender_perceived": {"type": "string"},
                        "register": {"type": "string"},
                        "timbre": {"type": "array", "items": {"type": "string"}},
                        "dynamics": {"type": "string"},
                        "effects_present": {"type": "array", "items": {"type": "string"}},
                        "notable_issues": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "reference_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref_id": {"type": "integer"},
                            "notable_qualities": {"type": "array", "items": {"type": "string"}},
                            "target_characteristics": {"type": "string"},
                        },
                    },
                },
                "intent_description": {"type": "string"},
            },
            "required": ["source_text", "base_audio_analysis", "intent_description"],
        }

        result = self.client.call_json(
            system_prompt=self.perception_system,
            user_text=user_text,
            audio_paths=all_audios,
        )

        return PerceptionOutput(
            base_audio_analysis=result.get("base_audio_analysis", {}),
            reference_analysis=result.get("reference_analysis", []),
            intent_description=result.get("intent_description", ""),
        ), result.get("source_text", "")

    async def clarify(
        self,
        user_prompt: str,
        base_audio_path: str,
        reference_audio_paths: list[str] | None,
        previous_perception: dict[str, Any],
        question: str,
        focus: str | None = None,
    ) -> dict[str, Any]:
        """
        Answer a clarification question by re-examining the audio.
        Returns structured answer with optional revised analysis.
        """
        refs = reference_audio_paths or []
        all_audios = [base_audio_path] + refs

        user_text = self.clarification_user.format(
            previous_perception_json=json.dumps(previous_perception, indent=2, ensure_ascii=False),
            clarification_question=question,
            clarification_focus=focus or "general",
            user_prompt=user_prompt,
        )

        response_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "revised_analysis": {"type": "object"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "suggestion_to_planner": {"type": "string"},
            },
            "required": ["answer", "confidence"],
        }

        return self.client.call_json(
            system_prompt=self.clarification_system,
            user_text=user_text,
            audio_paths=all_audios,
        )
