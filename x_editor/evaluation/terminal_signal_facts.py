"""Descriptive full-file coordinate comparisons for post-delivery diagnosis.

Matching geometry allows measurement, not a claim of lineage or correct intent.
No expected gain, semantic labels, benchmark targets or verdicts enter this view.
"""

from x_editor.evaluation.guidance_contracts import EvaluationArtifactRef
from x_editor.evaluation.paired_signal_facts import _read_bound, measure_pcm_comparison


def terminal_signal_facts(sources, candidate: EvaluationArtifactRef) -> dict:
    output, output_facts = _read_bound(candidate)
    rows = []
    for source_ref in sources:
        source, source_facts = _read_bound(source_ref)
        row = {"source": {**source_facts, "role": source_ref.role},
               "output": output_facts, "comparison_status": "unknown"}
        rows.append(row)
        if source is None or output is None:
            row["reason"] = "bound_audio_unavailable"
        elif source.shape != output.shape or source_facts["sample_rate"] != output_facts["sample_rate"]:
            row["reason"] = "incompatible_sample_rate_frame_count_or_channels"
        elif not source.size:
            row["reason"] = "empty_audio"
        else:
            row.update(measure_pcm_comparison(source, output))
    return {
        "protocol_id": "x-editor.terminal-coordinate-facts/v1",
        "scope": "Compare entire files at identical frame coordinates; no inferred source mapping.",
        "semantic_grounding": "unverified",
        "interpretation": "These are measurements, not an edit-success verdict. A linear gain with "
        "negligible fit residual describes the entire measured mixture, including every component "
        "present. It does not establish which components exist or whether uniform modification was "
        "requested. Inspect public scope and preservation conditions separately. A nonzero residual "
        "does not prove an error. No resampling, alignment search, cropping or channel mixing is used.",
        "pairs": rows,
    }


def terminal_prefix_facts(sources, candidate: EvaluationArtifactRef) -> dict:
    """Describe full source units at output onset, without an expected edit map.

    A matching prefix cannot by itself prove duplication or wrong scope. It only
    establishes what is present at these coordinates; a judge must interpret the
    public request. Incompatible geometry is retained as unknown.
    """
    output, output_facts = _read_bound(candidate)
    rows = []
    for ref in sources:
        source, source_facts = _read_bound(ref)
        row = {"source": source_facts, "output": output_facts, "comparison_status": "unknown"}
        rows.append(row)
        if source is None or output is None:
            row["reason"] = "bound_audio_unavailable"
        elif (not source.size or len(output) < len(source)
              or source.shape[1:] != output.shape[1:]
              or source_facts["sample_rate"] != output_facts["sample_rate"]):
            row["reason"] = "source_does_not_fit_compatible_output_prefix"
        else:
            row.update(measure_pcm_comparison(source, output[:len(source)]))
            row["compared_output_frames"] = [0, len(source)]
            row["unmeasured_suffix_frames"] = len(output) - len(source)
    return {"protocol_id": "x-editor.terminal-prefix-facts/v1", "pairs": rows,
            "scope": "Full source versus output prefix only. No inferred repetition, authorization or suffix contents."}
