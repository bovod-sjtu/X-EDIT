"""Owned AuK service with declared channel preparation and optional native PE."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time

from pydantic import BaseModel, ConfigDict, Field
import soundfile as sf


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_audio_path: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=8192)
    output_path: str = Field(min_length=1)
    gen_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry(path):
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("empty or invalid audio")
    return {"frames": info.frames, "sample_rate": info.samplerate,
            "channels": info.channels, "format": info.format,
            "duration_seconds": info.frames / info.samplerate}


def execute_edit(engine, request, model_identity, *, enhancer=None):
    from x_editor.tools.catalog.auk.audio_input import prepare_source

    source, target = Path(request.input_audio_path).resolve(), Path(request.output_path).resolve()
    if source == target or target.exists():
        raise ValueError("output must be a new path distinct from the source")
    facts = geometry(source)
    if not request.instruction.strip():
        raise ValueError("instruction must not be blank")
    source_sha = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Request-local temporary directory prevents partial writes becoming deliveries.
    with tempfile.TemporaryDirectory(prefix="auk-", dir=target.parent) as tmp:
        model_source, channel_preparation = prepare_source(source, Path(tmp))
        generated = Path(tmp) / "generated.wav"
        preparation = {"enabled": False, "original_instruction": request.instruction}
        prepared = None
        try:
            instruction, seconds = request.instruction, request.gen_seconds
            if enhancer is not None:
                prepared = enhancer.prepare(request.instruction, str(model_source),
                    target_duration=request.gen_seconds or facts["duration_seconds"])
                if not prepared.audio:
                    raise ValueError("AuK edit preparation must retain its bound source")
                pe_facts = geometry(prepared.audio)
                if pe_facts["frames"] != facts["frames"] or pe_facts["sample_rate"] != facts["sample_rate"]:
                    raise ValueError("AuK PE changed the bound source timeline")
                model_source = Path(prepared.audio)
                instruction = prepared.instruction
                seconds = request.gen_seconds if request.gen_seconds is not None else prepared.gen_seconds
                preparation = {"enabled": True, "original_instruction": request.instruction,
                    "effective_instruction": instruction,
                    "details": prepared.debug_dict(include_raw_responses=False)}
            preparation["model_input_sha256"] = sha(model_source)
            preparation["model_input_geometry"] = geometry(model_source)
            parameters = engine.edit(input_audio_path=str(model_source), instruction=instruction,
                                     output_path=str(generated), gen_seconds=seconds)
        finally:
            if prepared is not None:
                prepared.cleanup()
        output_facts = geometry(generated)
        if output_facts["format"] != "WAV":
            raise ValueError("AuK output is not a WAV container")
        import numpy as np
        audio, _ = sf.read(generated)
        if not np.isfinite(audio).all():
            raise ValueError("AuK output contains non-finite samples")
        if sha(source) != source_sha:
            raise ValueError("source changed during generation")
        output_sha = sha(generated)
        os.link(generated, target)  # Atomic publication without overwriting a racing request.
    return {"output_audio_path": str(target), "backend": "auk",
            "input_audio_path": str(source), "source_sha256": source_sha,
            "output_sha256": output_sha, "source_geometry": facts,
            **output_facts, "instruction": request.instruction,
            "model_identity": model_identity, "parameters": parameters,
            "channel_preparation": channel_preparation, "prompt_enhancement": preparation,
            "inference_and_io_seconds": time.monotonic() - started,
            "semantic_quality": "unknown"}


def create_app(engine, model_identity, *, enhancer=None):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="X-Editor AuK")
    # Bound waiting work while the engine serializes GPU inference and RNG use.
    slot = threading.BoundedSemaphore(8)

    @app.get("/health")
    def health():
        # The app is served only after model loading; use the shared bootstrap
        # health vocabulary so a loaded engine is admitted by the Executor.
        return {"status": "healthy", "model_identity": model_identity,
                "capabilities": ["x-editor/tool/auk/edit/v1"],
                "source_channel_policy": "native_channel_mean", "output_channels": 1,
                "prompt_enhancer": "native_bound" if enhancer is not None else "disabled"}

    @app.post("/edit")
    def edit(request: EditRequest):
        # Do not build an unbounded HTTP backlog behind a GPU/RNG lock.
        if not slot.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="AuK engine busy")
        try:
            return execute_edit(engine, request, model_identity, enhancer=enhancer)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            slot.release()

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "qwen-path", "upstream-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder-device", default="cuda:1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8065)
    parser.add_argument("--backend-url", help="Reuse a loaded AuK engine through its owned HTTP service")
    parser.add_argument("--prompt-enhancer", choices=("disabled", "native_bound"), default="disabled")
    parser.add_argument("--enhancer-url")
    parser.add_argument("--enhancer-model")
    parser.add_argument("--enhancer-asr-url")
    args = parser.parse_args()
    identity = {"upstream_revision": args.upstream_revision, "checkpoint_sha256": sha(args.checkpoint),
                "config_sha256": sha(args.config), "encoder_path": args.qwen_path,
                "generation_dtype": "bf16", "device": args.device, "encoder_device": args.encoder_device}
    if args.backend_url:
        from x_editor.tools.catalog.auk.proxy_engine import ProxyEngine
        engine = ProxyEngine(args.backend_url)
    else:
        from x_editor.tools.catalog.auk.engine import AukEngine
        engine = AukEngine(config=args.config, checkpoint=args.checkpoint, qwen_path=args.qwen_path,
                           device=args.device, encoder_device=args.encoder_device)
    enhancer = None
    if args.prompt_enhancer == "native_bound":
        from x_editor.tools.catalog.auk.prompt_enhancer import build_enhancer
        enhancer = build_enhancer(base_url=args.enhancer_url, model=args.enhancer_model,
                                  asr_url=args.enhancer_asr_url)
    uvicorn.run(create_app(engine, identity, enhancer=enhancer), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
