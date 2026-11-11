"""Single-operation AuK inference; no task classification or session ownership."""

from __future__ import annotations

import threading
import time


def place_engine(engine, *, device, encoder_device):
    """Move CPU-loaded components independently, avoiding a full-model GPU peak."""
    encoder = engine.model.text_encoder
    engine.model.text_encoder = None
    try:
        engine.model.to(device)
        engine.vae_model.to(device)
        encoder.to(encoder_device)
    finally:
        engine.model.text_encoder = encoder
    engine.device = device
    if device != encoder_device:
        import torch

        def move_inputs(module, args, kwargs):
            return args, {key: value.to(encoder_device) if torch.is_tensor(value) else value
                          for key, value in kwargs.items()}

        def move_hidden_states(module, args, output):
            output.hidden_states = tuple(value.to(device) for value in output.hidden_states)
            return output

        encoder.register_forward_pre_hook(move_inputs, with_kwargs=True)
        encoder.register_forward_hook(move_hidden_states)
    return engine


class AukEngine:
    def __init__(self, *, config, checkpoint, qwen_path, device, encoder_device,
                 seed=42, nfe=32, cfg_strength=2.0):
        from auk.infer.infer_auk import AukInfer

        self.engine = place_engine(
            AukInfer(config, checkpoint, qwen_path=qwen_path, device="cpu", dtype="bf16"),
            device=device, encoder_device=encoder_device,
        )
        self.seed, self.nfe, self.cfg_strength = seed, nfe, cfg_strength
        self.lock = threading.Lock()

    def edit(self, *, input_audio_path, instruction, output_path, gen_seconds=None):
        import torch
        from auk.infer.infer_auk import save_audio

        # Source is already bound by the framework. No old turns or other cases
        # enter conditioning. The service owns and records optional prompt preparation.
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio", "audio": input_audio_path},
        ]}]
        queued = time.monotonic()
        with self.lock:
            started = time.monotonic()
            torch.manual_seed(self.seed)  # Includes stochastic source VAE encoding.
            try:
                audio, sr = self.engine.generate(
                    messages, audio=input_audio_path, gen_seconds=gen_seconds,
                    seed=self.seed, nfe=self.nfe, cfg_strength=self.cfg_strength,
                )
                if audio.numel() == 0 or not torch.isfinite(audio).all():
                    raise ValueError("AuK returned empty or non-finite audio")
                save_audio(audio, sr, output_path)
            finally:
                self.engine.model.transformer.clear_cache()
        return {"seed": self.seed, "nfe": self.nfe, "cfg_strength": self.cfg_strength,
                "queue_seconds": started - queued,
                "inference_seconds": time.monotonic() - started,
                "duration_policy": "explicit" if gen_seconds is not None else "match_source"}
