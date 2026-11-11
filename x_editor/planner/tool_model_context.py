"""Planner/executor context comparing X-Editor tool and model capabilities."""

from __future__ import annotations


TOOL_MODEL_CAPABILITY_CONTEXT = """\
Tool/model capability boundaries:
- artifact_ops.extract_spans: timeline localization/export only. It selects
  source regions and emits a local clip plus selected_spans metadata; it does
  not edit content, style, speaker identity, volume, or acoustic scene.
  For unknown reference clips/enrollment audio, do not use it as the first
  locator. Ground speech/speaker/voice references with
  perception.speaker_attributed_review, or sound-event/acoustic references with
  perception.timestamped_caption/FlexSED, then pass selected_spans to
  artifact_ops.extract_spans only as the cutter/export step.
- audio_effects: deterministic DSP acoustic/channel rendering. Use for telephone,
  old radio, underwater, strong wind, muffled/heavy-wall, and reference channel
  matching. It preserves timeline/content better than text-to-audio generation,
  but the effect realism is approximate.
- speech_enhancement: ClearerVoice cleanup/restoration. Use to remove noise,
  dereverb, or make speech clearer/natural. Do not use it to add telephone,
  radio, underwater, wind, or muffled degradation.
- spatial_reverb: room impulse response only. Use for halls, rooms, bathrooms,
  cathedrals, and echo/reverb spaces. Do not use it for phone/radio/underwater
  channel effects or reference channel matching. For local reverb, first extract
  the target span, apply reverb to the local clip, then reconstruct.
- seed_vc/intra_audio_voice_transfer: speaker timbre conversion. They preserve
  spoken content but do not transfer microphone, radio, telephone, underwater,
  wall, or recording-channel conditions.
- edit_data_pipeline: text/prosody/pitch/rate/pause/volume edits grounded in
  source_text and tags. text_edit is only for spoken-word changes with
  <sub>/<ins>/<del>. emotion_style_edit is only for supported emotion tags
  (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm, none);
  map hushed/reflective/internal-monologue delivery to melancholic or calm. It
  is not an acoustic effects renderer and not a whisper/rap/singing/accent tool.
- qwen3_tts/accent_conversion: speech synthesis/re-synthesis. Use for generated
  speech, voice design, accent/dialect conversion, or long text-edit fallback;
  avoid using them just to add channel/environment effects when audio_effects
  can preserve the original speech.
  Accent/dialect translation is a composite edit: first normalize localized or
  regional wording with edit_data_pipeline.text_edit, then convert the
  pronunciation/accent/dialect with accent_conversion.
  For "accent/dialect of audio2/reference audio" requests, do not invent a
  guessed accent label. Pass the reference audio to accent_conversion; if the
  reference cannot be named, the runtime can use the provided reference clip as
  the accent reference.
- seamless_expressive_s2st: speech-to-speech translation when the backend is
  available. If unavailable, use transcript-based text_edit/qwen3 fallback only
  when source_text and target_text are explicit.
- artifact_ops.concat: pure audio concatenation/merge/join. It does not need
  source_text and should not be replaced by edit_data_pipeline/audio_assembly.
- intra_audio_voice_transfer/seed_vc: use for timbre replacement. Do not add a
  no-op text_edit for voice/timbre-only requests.
- artifact_ops.speaker_scoped_edit + speech_enhancement: use this single scoped
  workflow for semantic local cleanup such as "internal monologue(s) into normal
  speech" or "remove reverb/echo only from those semantic segments". Avoid
  splitting semantic cleanup into many extract/reconstruct steps unless exact
  spans are supplied.
"""


def tool_model_capability_context() -> str:
    """Return compact planner-facing capability comparison text."""

    return TOOL_MODEL_CAPABILITY_CONTEXT
