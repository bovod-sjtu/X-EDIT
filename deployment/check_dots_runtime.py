"""Exercise the installed Edit runtime control flow without loading model weights."""
from __future__ import annotations

import json
from types import SimpleNamespace


def check_runtime():
    import torch
    from dots_tts.edit_runtime import DotsTtsEditRuntime

    checks = []
    for profiling in (False, True):
        runtime = DotsTtsEditRuntime.__new__(DotsTtsEditRuntime)
        runtime.device = torch.device("cpu")
        runtime.precision = "float32"
        runtime.max_generate_length = 32
        runtime.vocoder_merge_steps = 4
        runtime.sample_rate = 48000
        runtime.resolve_sampling_options = lambda **kwargs: ("euler", 10, 1.2)
        inputs = {
            "fid": "runtime-contract", "text": "Good morning.",
            "prompt_text": "Hello.", "template_name": "tts", "language": "en",
            "source_text": "Good evening.", "source_text_source": "explicit",
            "target_text_source": "instruction",
        }
        runtime._prepare_audio_fill_tts_inputs = lambda **kwargs: inputs
        runtime._prepare_edit_inputs = lambda **kwargs: inputs
        chunks = [torch.full((1, 480), 0.1), torch.full((1, 240), 0.2)]

        def stream(data, **kwargs):
            if data is not inputs:
                raise AssertionError("Streaming input identity changed")
            yield from chunks

        def generate(data, **kwargs):
            if data is not inputs:
                raise AssertionError("Edit input identity changed")
            return torch.cat(chunks, dim=-1)

        runtime.model = SimpleNamespace(generate_audio_stream=stream, generate_audio=generate)
        actual = list(runtime.generate_stream(
            text="Good morning.", prompt_audio_path="not-read.wav", prompt_text="Hello.",
            use_xvector=False, profile_inference=profiling,
        ))
        if len(actual) != len(chunks):
            raise AssertionError("Streaming chunk count changed")
        for output, expected in zip(actual, chunks):
            torch.testing.assert_close(output, expected)
        checks.append({"method": "generate_stream", "profiling": profiling, "passed": True})
        result = runtime.generate_edit(
            source_audio_path="not-read.wav",
            instruction='Good <sub targ="morning">evening</sub>.',
            profile_inference=profiling,
        )
        torch.testing.assert_close(result["audio"], torch.cat(chunks, dim=-1))
        if result["target_text"] != inputs["text"] or result["sample_rate"] != runtime.sample_rate:
            raise AssertionError("Edit response contract changed")
        checks.append({"method": "generate_edit", "profiling": profiling, "passed": True})
    return {"passed": True, "checks": checks, "model_inference_verified": False,
            "scope": "Native control flow with synthetic tensors and stubbed model; no weights or GPU"}


if __name__ == "__main__":
    print(json.dumps(check_runtime(), indent=2))
