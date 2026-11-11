# MOSS-Transcribe-Diarize Speaker Evidence Backend

This service isolates MOSS-Transcribe-Diarize from the main X-Editor runtime.
It is used as timestamped speaker-initialization evidence for Chinese
speaker-aware perception.

## Capability Boundary

- Use this backend when Chinese multi-speaker audio needs speaker-aware
  perception and DiariZen / SE-DiCoW are unreliable.
- MOSS provides initial speaker labels, timestamps, and a transcript prior.
- Qwen3-ASR remains the wording prior in `perception.speaker_attributed_review`.
- StrongOmni/Gemini targeted-slice review remains downstream and unchanged.
- The backend is evidence, not ground truth.

Example response structure (transcript shown in English for illustration):

```json
{
  "segments": [
    {"speaker_id": "1", "start_sec": 0.48, "end_sec": 1.66, "text": "Welcome, everyone."}
  ],
  "text": "[0.48][S01]Welcome, everyone.[1.66]"
}
```

## Recommended Serving Path

Run the official model with SGLang Omni or vLLM using the OpenAI-compatible
`/v1/audio/transcriptions` endpoint, then point the wrapper at it:

```bash
export MOSS_OPENAI_BASE_URL=http://127.0.0.1:8000
export MOSS_OPENAI_MODEL=OpenMOSS-Team/MOSS-Transcribe-Diarize
cd x_editor/tools/catalog/moss_transcribe_diarize
uv run --project . python service.py --host 127.0.0.1 --port 8048
```

The wrapper posts audio with `response_format=verbose_json` and normalizes
either `segments` or compact `[start][Sxx]text[end]` transcript text.

## Local Python Adapter

For small experiments, install the official repo in a clean environment and use
the adapter:

```bash
export MOSS_REPO=/path/to/MOSS-Transcribe-Diarize
export MOSS_TRANSCRIBE_COMMAND='python x_editor/tools/catalog/moss_transcribe_diarize/moss_infer.py --request-json {request_json}'
```

Large model weights and cloned repos must stay outside git.

For the reproducible release use `deployment/stack.py` from the repository
root, or set `MOSS_REPO` and `MOSS_TRANSCRIBE_INFERENCE_PYTHON` explicitly.
No shared development checkout is auto-discovered.


## X-Editor Usage

For Chinese speaker-aware perception:

```python
run_speaker_attributed_review(
    audio_path=audio,
    instruction="Review the speaker transcript.",
    output_dir=out,
    language="zh",
    speaker_evidence_backend="moss",
)
```

The perception module uses MOSS for `speaker_evidence_backend="moss"` and for
`"auto"` or omitted/default backend selection when the speech language is
Chinese. English/non-Chinese `auto` routes to DiariZen plus optional SE-DiCoW
hints. Set `"diarizen"` or `"sedicow"` explicitly for those baselines.
