"""Local adapter for official MOSS-Transcribe-Diarize Python inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--model-id", default=os.getenv("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-Transcribe-Diarize"))
    parser.add_argument("--device", default=os.getenv("MOSS_DEVICE", "auto"))
    parser.add_argument("--dtype", default=os.getenv("MOSS_DTYPE", "auto"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("MOSS_MAX_NEW_TOKENS", "65536")))
    args = parser.parse_args()

    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    audio = Path(request["audio_path"]).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"audio not found: {audio}")
    _prepare_import_paths()
    with contextlib.redirect_stdout(sys.stderr):
        result = _run_official_python(
            audio,
            model_id=args.model_id,
            device_arg=args.device,
            dtype_arg=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
    print(json.dumps(result, ensure_ascii=False))


def _prepare_import_paths() -> None:
    repo = os.getenv("MOSS_REPO")
    if repo:
        sys.path.insert(0, repo)


def _run_official_python(
    audio: Path,
    *,
    model_id: str,
    device_arg: str,
    dtype_arg: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
        resolve_device,
    )

    device = resolve_device(device_arg)
    if dtype_arg == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, dtype_arg)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="auto",
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    messages = build_transcription_messages(str(audio))
    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    text = str(result.get("text") or "")
    segments = [
        {
            "speaker_id": str(segment.speaker),
            "start_sec": float(segment.start),
            "end_sec": float(segment.end),
            "text": str(segment.text),
        }
        for segment in parse_transcript(text)
    ]
    return {
        "backend": "moss_transcribe_diarize_official_python",
        "model": model_id,
        "text": text,
        "segments": segments,
        "raw": result,
    }


if __name__ == "__main__":
    main()
