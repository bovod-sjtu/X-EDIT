"""HTTP client for the Qwen3-TTS local service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from x_editor.tools.catalog._http_retry import post_with_retries, retry_attempts


class Qwen3TTSClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8017",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def synthesize(self, **kwargs: Any) -> dict[str, Any]:
        mode = kwargs.get("mode", "custom_voice")
        timeout = float(kwargs.get("timeout_seconds") or self.timeout)
        _voice_design_backend(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "voice_clone":
            endpoint = "/synthesize_clone"
            payload = {
                "text": kwargs["text"],
                "language": _service_language(kwargs.get("language", "Auto")),
                "ref_audio": kwargs["ref_audio_path"],
                "ref_text": kwargs.get("ref_text"),
                "x_vector_only_mode": bool(kwargs.get("x_vector_only_mode", False)),
                "output_format": "wav",
            }
        elif mode == "custom_voice":
            endpoint = "/synthesize_custom"
            payload = {
                "text": kwargs["text"],
                "speaker": kwargs.get("speaker", "vivian"),
                "language": _service_language(kwargs.get("language", "Auto")),
                "instruct": kwargs.get("instruct"),
                "output_format": "wav",
            }
        else:
            raise ValueError(f"Unsupported qwen3_tts mode: {mode}")
        resp = post_with_retries(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=timeout,
            service_name="Qwen3-TTS",
            retry_env="X_EDITOR_QWEN3_TTS_RETRIES",
            backoff_env="X_EDITOR_QWEN3_TTS_RETRY_BACKOFF",
            default_retries=1,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Qwen3-TTS service error {resp.status_code}: {resp.text[:500]}")
        output_path.write_bytes(resp.content)
        result = {
            "output_audio_path": str(output_path),
            "sample_rate": _header_int(resp, "X-Sample-Rate"),
            "duration_seconds": _header_float(resp, "X-Duration-Seconds"),
            "voice_design_backend": "qwen3_tts" if mode == "custom_voice" else None,
        }
        attempts = retry_attempts(resp)
        if attempts:
            result["request_retries"] = attempts
        return result



def _voice_design_backend(kwargs: dict[str, Any]) -> str:
    value = (
        kwargs.get("voice_design_backend")
        or os.getenv("QWEN3_TTS_VOICE_DESIGN_BACKEND")
        or os.getenv("X_EDITOR_VOICE_DESIGN_BACKEND")
        or "qwen3_tts"
    )
    value = str(value).strip().lower().replace("-", "_")
    if value in {"qwen", "qwen3", "qwen3tts", "qwen3_tts", "auto"}:
        return "qwen3_tts"
    raise ValueError(f"Unsupported voice_design_backend: {value}")


def _service_language(value: Any) -> str:
    raw = str(value or "Auto").strip()
    if not raw:
        return "auto"
    normalized = raw.lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "zh": "chinese",
        "zho": "chinese",
        "cmn": "chinese",
        "cn": "chinese",
        "zh-cn": "chinese",
        "zh-tw": "chinese",
        "chinese": "chinese",
        "mandarin": "chinese",
        "putonghua": "chinese",
        "中文": "chinese",
        "汉语": "chinese",
        "普通话": "chinese",
        "en": "english",
        "eng": "english",
        "english": "english",
        "英语": "english",
        "英文": "english",
        "fr": "french",
        "fra": "french",
        "fre": "french",
        "french": "french",
        "法语": "french",
        "de": "german",
        "deu": "german",
        "ger": "german",
        "german": "german",
        "德语": "german",
        "it": "italian",
        "ita": "italian",
        "italian": "italian",
        "意大利语": "italian",
        "ja": "japanese",
        "jpn": "japanese",
        "japanese": "japanese",
        "日语": "japanese",
        "日本语": "japanese",
        "ko": "korean",
        "kor": "korean",
        "korean": "korean",
        "韩语": "korean",
        "韩国语": "korean",
        "pt": "portuguese",
        "por": "portuguese",
        "portuguese": "portuguese",
        "葡萄牙语": "portuguese",
        "ru": "russian",
        "rus": "russian",
        "russian": "russian",
        "俄语": "russian",
        "es": "spanish",
        "spa": "spanish",
        "spanish": "spanish",
        "西班牙语": "spanish",
    }
    return aliases.get(normalized, normalized)




def _header_int(resp: requests.Response, name: str) -> int | None:
    value = resp.headers.get(name)
    return int(value) if value else None


def _header_float(resp: requests.Response, name: str) -> float | None:
    value = resp.headers.get(name)
    return float(value) if value else None
