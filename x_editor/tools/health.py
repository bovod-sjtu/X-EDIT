"""Service health and capability checks for X-Editor tools."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ServiceCheck:
    name: str
    url: str
    required_paths: tuple[str, ...]
    expected_root_marker: str | None = None
    required_health_flags: tuple[str, ...] = ()
    required_availability_any: tuple[str, ...] = ()


SERVICE_CHECKS = [
    ServiceCheck("ace_step_music", "http://127.0.0.1:8044", ("/health", "/release_task", "/query_result"), None),
    ServiceCheck("edit_text", "http://127.0.0.1:8035", ("/health", "/execute"), None),
    ServiceCheck("edit_pitch_rate", "http://127.0.0.1:8036", ("/health", "/execute"), None),
    ServiceCheck("edit_emotion_style", "http://127.0.0.1:8037", ("/health", "/execute"), None),
    ServiceCheck("edit_pause", "http://127.0.0.1:8038", ("/health", "/execute"), None),
    ServiceCheck("edit_volume", "http://127.0.0.1:8006", ("/health", "/normalize"), None),
    ServiceCheck("edit_audio_assembly", "http://127.0.0.1:8021", ("/health", "/assemble"), None),
    ServiceCheck("seed_vc", "http://127.0.0.1:8003", ("/health", "/convert"), "SeedVC"),
    ServiceCheck("qwen3_tts_custom", "http://127.0.0.1:8017", ("/health", "/info", "/synthesize_custom"), "Qwen3-TTS", ("custom_voice_loaded",)),
    ServiceCheck("qwen3_tts_clone", "http://127.0.0.1:8019", ("/health", "/info", "/synthesize_clone"), "Qwen3-TTS", ("base_loaded", "tokenizer_loaded")),
    ServiceCheck("clearer_voice_studio", "http://127.0.0.1:8027", ("/health", "/enhance"), "ClearerVoice"),
    ServiceCheck("quarkaudio_unise", "http://127.0.0.1:8025", ("/health", "/tse/extract"), "QuarkAudio"),
    ServiceCheck("audio_separator_roformer", "http://127.0.0.1:8042", ("/health", "/separate"), "Audio Separator"),
    ServiceCheck("diarization", "http://127.0.0.1:8041", ("/health", "/diarize"), "Diarization"),
    ServiceCheck(
        "seamless_expressive_s2st",
        "http://127.0.0.1:8039",
        ("/health", "/translate"),
        "seamless_expressive_s2st",
        required_availability_any=("m4t_predict", "expressivity_predict"),
    ),
]


def check_service(spec: ServiceCheck, timeout: float = 3.0) -> dict[str, Any]:
    result: dict[str, Any] = {"name": spec.name, "url": spec.url, "healthy": False, "checks": {}}
    try:
        root = requests.get(f"{spec.url}/", timeout=timeout)
        result["root_status"] = root.status_code
        result["root_text"] = root.text[:300]
        if spec.expected_root_marker and spec.expected_root_marker.lower() not in root.text.lower():
            result["root_warning"] = f"root response does not look like {spec.expected_root_marker}"
    except Exception as exc:
        result["error"] = f"root check failed: {type(exc).__name__}: {exc}"
        return result

    for path in spec.required_paths:
        url = f"{spec.url}{path}"
        try:
            if path in {
                "/convert",
                "/synthesize_custom",
                "/synthesize_clone",
                "/tse/extract",
                "/enhance",
                "/execute",
                "/normalize",
                "/assemble",
                "/diarize",
                "/separate",
                "/stretch",
                "/smooth",
                "/release_task",
                "/query_result",
            }:
                response = requests.options(url, timeout=timeout)
                if response.status_code == 405:
                    ok = True
                elif response.status_code == 404:
                    ok = False
                else:
                    ok = response.status_code < 500
            else:
                response = requests.get(url, timeout=timeout)
                ok = response.status_code < 500 and response.status_code != 404
                health_flag_errors = []
                if path == "/health" and ok:
                    try:
                        health = response.json()
                        health_status = str(health.get("status", "")).lower()
                        if health_status and health_status not in {"healthy", "ok", "running"}:
                            ok = False
                        for flag in spec.required_health_flags:
                            if health.get(flag) is not True:
                                ok = False
                                health_flag_errors.append(flag)
                        availability = health.get("availability") or {}
                        if spec.required_availability_any and not any(
                            availability.get(flag) is True for flag in spec.required_availability_any
                        ):
                            ok = False
                            health_flag_errors.extend(f"availability.{flag}" for flag in spec.required_availability_any)
                    except Exception:
                        pass
            item = {"status_code": response.status_code, "ok": ok, "text": response.text[:200]}
            if path == "/health" and health_flag_errors:
                item["missing_health_flags"] = health_flag_errors
            result["checks"][path] = item
        except Exception as exc:
            result["checks"][path] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    result["healthy"] = all(item.get("ok") for item in result["checks"].values()) and "root_warning" not in result
    return result


def check_all_services() -> list[dict[str, Any]]:
    return [check_service(spec) for spec in SERVICE_CHECKS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check X-Editor model services")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = check_all_services()
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    for result in results:
        status = "OK" if result["healthy"] else "BAD"
        print(f"[{status}] {result['name']} {result['url']}")
        if result.get("root_warning"):
            print(f"  warning: {result['root_warning']}")
        if result.get("error"):
            print(f"  error: {result['error']}")
        for path, item in result.get("checks", {}).items():
            print(f"  {path}: {'OK' if item.get('ok') else 'BAD'} status={item.get('status_code')} {item.get('error', '')}")


if __name__ == "__main__":
    main()
