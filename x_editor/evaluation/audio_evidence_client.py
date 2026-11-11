"""Explicit attachment boundaries for independent audio evaluation requests.

An ordered list of audio parts alone does not reliably bind silent attachments
to their inventory entries. Labels travel beside the original parts; the audio
bytes, their order, and the provider's encoding remain unchanged.
"""

from typing import Any

from x_editor.agents.openai_compatible import OpenAICompatibleChatClient


class AudioEvidenceEvaluationClient(OpenAICompatibleChatClient):
    """Keep a single attachment namespace in the text and multimodal payload."""

    def _build_payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = super()._build_payload(*args, **kwargs)
        content = payload["messages"][1]["content"]
        if not isinstance(content, list):
            return payload
        audio_types = {"input_audio", "audio_url"}
        count = sum(part.get("type") in audio_types for part in content)
        if not count:
            return payload
        bound = [
            {
                "type": "text",
                "text": (
                    f"There are exactly {count} audio attachments. Attachment numbers "
                    "match the audio_inventory in the task. Silent attachments still "
                    "occupy their numbered position."
                ),
            }
        ]
        index = 0
        for part in content:
            if part.get("type") in audio_types:
                index += 1
                bound.extend(
                    [
                        {"type": "text", "text": f"Attachment {index} begins here:"},
                        part,
                        {"type": "text", "text": f"Attachment {index} ends here."},
                    ]
                )
            else:
                bound.append(part)
        payload["messages"][1]["content"] = bound
        return payload
