"""MCP wrapper for the local edit-data-pipeline HTTP services."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from client import EditDataPipelineClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["operation", "output_path"],
    "properties": {
        "operation": {
            "type": "string",
            "enum": [
                "text_edit",
                "pitch_rate_edit",
                "emotion_style_edit",
                "pause_edit",
                "speech_rate",
                "volume_normalize",
                "pitch_shift",
                "boundary_smooth",
                "audio_assembly",
            ],
        },
        "input_audio_path": {"type": "string"},
        "output_path": {"type": "string"},
        "instruction": {"type": "string"},
        "service_url": {"type": "string"},
        "source_text": {"type": "string"},
        "target_text": {"type": "string"},
        "num_steps": {"type": "integer", "minimum": 1, "maximum": 128},
        "guidance_scale": {"type": "number", "minimum": 0, "maximum": 10},
        "use_xvector": {"anyOf": [{"type": "boolean"}, {"const": "auto"}]},
        "speaker_scale": {"type": "number", "minimum": 0, "maximum": 10},
        "instruction_st": {"type": "string"},
        "generation_id": {"type": "string"},
        "profile_id": {"type": "integer"},
        "sample_id": {"type": "string"},
        "clip_id": {"type": "string"},
        "evaluate": {"type": "boolean"},
        "persist": {"type": "boolean"},
        "output_dir": {"type": "string"},
        "instructions_nl": {"type": "object"},
        "reverse_instructions_nl": {"type": "object"},
        "instruction_nl": {"type": "string"},
        "rate": {"type": "number"},
        "rate_method": {"type": "string"},
        "method": {
            "type": "string",
            "default": "wsola+indextts2",
            "description": "Speech-rate default is wsola+indextts2; pure wsola is an explicit fallback.",
        },
        "target_db": {"type": "number"},
        "headroom_db": {"type": "number"},
        "semitones": {"type": "number"},
        "text": {"type": "string"},
        "speaker_audio_path": {"type": "string"},
        "points": {"type": "array", "items": {"type": "object"}},
        "source_audio_path": {"type": "string"},
        "target_audio_path": {"type": "string"},
        "edit_tagged_text": {"type": "string"},
        "time_spans_from_source": {"type": "array"},
        "time_spans_from_target": {"type": "array"},
        "preserve_outer_boundaries": {"type": "boolean"},
        "language": {"type": "string"},
        "aligner_backend": {"type": "string"},
        "synth_method": {"type": "string", "enum": ["insert_only", "single_region", "dual_boundary"]},
        "pitch_method": {"type": "string"},
        "psola_indextts2_cco_pitch_boost": {"type": "string"},
        "smooth_policy": {"type": "string"},
        "crossfade_ms": {"type": "number"},
        "smooth_margin_sec": {"type": "number"},
        "duration_mode": {"type": "string"},
        "synthesis_mode": {"type": "string"},
        "emotion_condition_mode": {"type": "string"},
        "expand_radius": {"type": "number"},
        "center_regen_second_pass": {"type": "boolean"},
        "center_regen_margin_sec": {"type": "number"},
        "merge_gap": {"type": "number"},
        "nfe_step": {"type": "integer"},
        "cfg_strength": {"type": "number"},
        "sway_sampling_coef": {"type": "number"},
        "source_crossfade_sec": {"type": "number"},
        "regen_crossfade_sec": {"type": "number"},
        "seed": {"type": "integer"},
        "disable_prepend_context": {"type": "boolean"},
        "fail_if_cannot_prepend_context": {"type": "boolean"},
        "min_unmasked_context_sec": {"type": "number"},
        "min_unmasked_context_ratio": {"type": "number"},
        "smoothing": {"type": "object"},
        "debug": {"type": "boolean"},
        "debug_dir": {"type": "string"},
        "debug_output_dir": {"type": "string"},
        "trace_id": {"type": "string"},
    },
}


def apply_edit(arguments: dict) -> dict:
    client = EditDataPipelineClient()
    return client.apply(arguments)


if __name__ == "__main__":
    run_stdio_server(
        "edit_data_pipeline",
        [
            LocalTool(
                name="edit_data_pipeline.apply",
                description=(
                    "Apply tagged edits through the selected dots.tts.edit model (default) or "
                    "edit-data-pipeline HTTP backend; speech_rate defaults to "
                    "wsola+indextts2, pause_edit, volume_normalize, pitch_shift, boundary_smooth, or audio_assembly."
                ),
                input_schema=INPUT_SCHEMA,
                handler=apply_edit,
            )
        ],
    )
