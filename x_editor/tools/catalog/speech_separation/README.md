# Speech Separation Tool

Planner-visible wrapper for vocal/accompaniment separation.

Default backend:

```text
audio_separator_roformer at http://127.0.0.1:8042
```

This wrapper is intentionally light. The heavy `audio-separator` dependency and
RoFormer model cache live in `../audio_separator_roformer`.

The configured models separate vocals from accompaniment, not individual
overlapping speakers. `stem_policy` defaults to `vocals`; use `instrumental`
for accompaniment. Semantic selection requires a unique provider-labeled stem;
file order does not establish its content or speaker identity.

Target-speaker extraction needs a separately available, qualified tool; a
reference clip alone does not enable that capability. Use `speech_enhancement`
with ClearerVoice for single-speaker cleanup. See the
[tool and evidence contracts](../../../../docs/TOOL_CONTRACTS.md).
