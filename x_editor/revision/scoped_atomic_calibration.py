"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import hashlib


from pathlib import Path


from x_editor.revision.atomic_module_review import checkpoint_sources, invoke, read, write


from x_editor.evaluation.atomic_module_rubrics import (
    AtomicModuleAnswer, PublicModuleReviewContext, bind_design, module_questions, validate_answer,
)


from x_editor.evaluation.scoped_atomic_review import (
    PROTOCOL_ID, ATOMIC_REFERENCE_PROTOCOL, ATOMIC_REFERENCE_PROMPT, PointerAtomicAnswer, validate_pointer_answer, materialize_atomic_citations, CONSISTENCY_REFERENCE_PROTOCOL, SCOPE_PROMPT, CONSISTENCY_PROMPT, scoped_sources, scoped_judge_prompt,
    materialize_consistency_citations,
    consistency_payload, consistency_prompt, validate_consistency, consistency_summary, feedback_candidates,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_row(row):
    if row["id"] in {".", ".."} or Path(row["id"]).name != row["id"]:
        raise ValueError("unsafe row identity")
    for key in ("sources", "context", "design"):
        if digest(row["paths"][key]) != row["sha256"][key]:
            raise ValueError(f"frozen calibration source changed: {row['id']}/{key}")
    context = PublicModuleReviewContext.model_validate(read(row["paths"]["context"]))
    design = bind_design(context, read(row["paths"]["design"]))
    sources = read(row["paths"]["sources"])
    if sources["public_context"] != context.model_dump(mode="json"):
        raise ValueError("checkpoint and question context differ")
    return design, scoped_sources(sources)
