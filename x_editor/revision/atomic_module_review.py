"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import hashlib


import json


import os


import tempfile


import time


from pathlib import Path


from x_editor.revision.research_quality_review import RecordedJudgeClient


from x_editor.agents.openai_compatible import OpenAICompatibleServiceUnavailable


from x_editor.evaluation.atomic_module_contract_audit import audit_module_contract


from x_editor.evaluation.atomic_module_prompts import (
    AUTHOR_PROMPT,
    JUDGE_PROMPT,
    author_prompt,
    judge_prompt,
)


from x_editor.evaluation.atomic_module_rubrics import (
    PROTOCOL_ID,
    AtomicModuleAnswer,
    PublicModuleReviewContext,
    bind_authored_design,
    bind_design,
    module_questions,
    normalized_atomic_summary,
    validate_answer,
)


from x_editor.evaluation.atomic_rubric_premises import (
    PREMISE_PROMPT,
    PremiseReview,
    resolve_premise_review,
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Completed records are immutable; exact resumes are allowed.
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f"refusing to overwrite prior evidence: {path}")
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != data:
                raise ValueError(f"refusing to overwrite concurrent evidence: {path}")
    finally:
        temporary.unlink()


def public_context(scenario, turn_index):
    turn_keys = {
        "turn_index",
        "user_request",
        "source_text",
        "source_asset_refs",
        "reference_asset_refs",
        "use_previous_output",
    }
    turns = [
        {key: value for key, value in turn.items() if key in turn_keys}
        for turn in scenario["turns"][: turn_index + 1]
    ]
    visible_refs = {
        ref
        for turn in turns
        for key in ("source_asset_refs", "reference_asset_refs")
        for ref in turn.get(key, [])
    }
    assets = [
        {
            key: value
            for key, value in asset.items()
            if key in {"asset_ref", "role", "content_sha256", "order"}
        }
        for asset in scenario["audio_assets"]
        if asset["asset_ref"] in visible_refs
    ]
    return PublicModuleReviewContext(
        turns=tuple(turns), target_turn_index=turn_index, audio_inventory=tuple(assets)
    )


def configured_client(path, args, *, author=False):
    return RecordedJudgeClient(
        directory=path,
        api_key=None,
        base_url=args.base_url,
        model_name=args.model,
        provider_name=args.provider,
        temperature=0,
        seed=getattr(args, "seed", 0),
        max_tokens=args.author_tokens if author else args.judge_tokens,
        timeout=args.timeout,
        max_retries=0,
        retry_backoff=0,
        allow_json_repair=False,
        allow_response_format_fallback=False,
        require_api_key=False,
        chat_template_kwargs={"enable_thinking": False},
        audio_content_part=None,
    )


def invoke(path, prompt, args, *, author=False):
    if getattr(args, 'factor_review_evidence', False):
        from x_editor.evaluation.efficient_review_transport import encode_prompt
        prompt, transport = encode_prompt(prompt)
        write(path / 'evidence_transport.json', transport)
    client = configured_client(path / "calls", args, author=author)
    return client.call_json(
        prompt["system_prompt"],
        prompt["user_prompt"],
        response_format={"type": "json_object"},
        repair_schema_name="atomic_module_review",
        request_id=hashlib.sha256(str(path).encode()).hexdigest()[:24],
    )


def author_one(identity, index, context, args):
    path = args.output / "cases" / identity / "turns" / f"{index:04d}"
    write(path / "public_context.json", context.model_dump(mode="json"))
    prompt = author_prompt(context)
    write(path / "author_prompt.json", prompt)
    if (path / "design.json").exists():
        bind_design(context, read(path / "design.json"))
        return {"case_id": identity, "turn_index": index, "status": "existing"}
    if (path / "author_receipt.json").exists():
        return {"case_id": identity, "turn_index": index, **read(path / "author_receipt.json")}
    if args.dry_run:
        return {"case_id": identity, "turn_index": index, "status": "dry_run"}
    started = time.monotonic()
    try:
        payload = invoke(path, prompt, args, author=True)
        design = bind_authored_design(context, payload)
        write(path / "design.json", design.model_dump(mode="json"))
        receipt = {
            "status": "completed",
            "necessary_evidence": len(design.necessary_evidence),
            "reasoner_questions": len(module_questions(design, "reasoner")),
            "planner_questions": len(module_questions(design, "planner")),
        }
    except (ValueError, TypeError, OSError, OpenAICompatibleServiceUnavailable) as error:
        receipt = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    receipt["elapsed_sec"] = time.monotonic() - started
    write(path / "author_receipt.json", receipt)
    return {"case_id": identity, "turn_index": index, **receipt}


def review_one(identity, index, context, args):
    """Revise only public-intent premises; never load a module checkpoint here."""
    path = args.output / "cases" / identity / "turns" / f"{index:04d}"
    parent = args.draft_root / "cases" / identity / "turns" / f"{index:04d}"
    write(path / "public_context.json", context.model_dump(mode="json"))
    if (path / "author_receipt.json").exists():
        return {"case_id": identity, "turn_index": index, **read(path / "author_receipt.json")}
    deadline = time.monotonic() + args.wait_seconds
    parent_path = parent / "design.json"
    parent_validation_error = None
    while not parent_path.is_file():
        parent_receipt = (
            read(parent / "author_receipt.json")
            if (parent / "author_receipt.json").is_file()
            else {}
        )
        raw_draft = parent / "calls/0001/response.json"
        if parent_receipt.get("status") == "failed" and raw_draft.is_file():
            parent_path = raw_draft
            parent_validation_error = parent_receipt.get("error")
            break
        if parent_receipt.get("status") == "failed" or time.monotonic() >= deadline:
            receipt = {"status": "parent_design_unavailable", "parent_receipt": parent_receipt}
            if not args.dry_run:
                write(path / "author_receipt.json", receipt)
            return {"case_id": identity, "turn_index": index, **receipt}
        if args.dry_run:
            return {"case_id": identity, "turn_index": index, "status": "parent_pending"}
        time.sleep(2)
    parent_context = PublicModuleReviewContext.model_validate(read(parent / "public_context.json"))
    if parent_context.fingerprint != context.fingerprint:
        raise ValueError("draft comes from another public conversation prefix")
    parent_design = read(parent_path)
    # Revalidate with the current contract so a revision receives actionable
    # missing-field details, rather than only an older generic error message.
    try:
        bind_authored_design(context, parent_design)
    except (ValueError, TypeError) as error:
        parent_validation_error = str(error)
    write(path / "parent_design.json", parent_design)
    write(
        path / "parent_reference.json",
        {
            "path": str(parent_path),
            "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            "parent_validation_error": parent_validation_error,
            "actor_outputs_visible": False,
        },
    )
    prompt = {
        "protocol_id": PROTOCOL_ID,
        "system_prompt": PREMISE_PROMPT,
        "user_prompt": json.dumps(
            {
                "public_context": context.model_dump(mode="json"),
                "draft_design": parent_design,
                "draft_validation_error": parent_validation_error,
                "authoring_specification": AUTHOR_PROMPT,
                "response_schema": PremiseReview.model_json_schema(),
                "revised_design_schema": json.loads(author_prompt(context)["user_prompt"])[
                    "response_schema"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    write(path / "author_prompt.json", prompt)
    if args.dry_run:
        return {"case_id": identity, "turn_index": index, "status": "dry_run"}
    started = time.monotonic()
    try:
        payload = invoke(path / "premise_review", prompt, args, author=True)
        review, design = resolve_premise_review(context, parent_design, payload)
        write(path / "premise_review.json", review.model_dump(mode="json"))
        receipt = {
            "status": "completed" if design else "premise_unknown",
            "premise_decision": review.decision,
            "issues": list(review.issues),
        }
        if design:
            write(path / "design.json", design.model_dump(mode="json"))
            receipt.update(
                necessary_evidence=len(design.necessary_evidence),
                reasoner_questions=len(module_questions(design, "reasoner")),
                planner_questions=len(module_questions(design, "planner")),
            )
    except (ValueError, TypeError, OSError, OpenAICompatibleServiceUnavailable) as error:
        receipt = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    receipt["elapsed_sec"] = time.monotonic() - started
    write(path / "author_receipt.json", receipt)
    return {"case_id": identity, "turn_index": index, **receipt}


def checkpoint_sources(run_root, identity, module, context):
    root = run_root / "modules/runs" / identity / module
    call = root / "module_invocations" / module / "0001"
    if not (call / "input.json").is_file():
        raise FileNotFoundError(f"no standard-input checkpoint: {identity}/{module}")
    module_input = read(call / "input.json")["args"][0]
    receipt = read(call / "receipt.json")
    accepted = (call / "output.json").is_file()
    output = read(call / "output.json") if accepted else None
    wire_error = None
    # Standard module probes record HTTP beside module_invocations. Some
    # invocation-scoped recorders nest it inside the call instead. Prefer that
    # narrower scope, and never reconstruct output by parsing an error message.
    response = call / "http/0001/response.txt"
    if not response.is_file():
        response = root / "http/0001/response.txt"
    if output is None and response.is_file():
        try:
            content = read(response)["choices"][0]["message"]["content"]
            output = json.loads(content)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            wire_error = str(error)
    audit = audit_module_contract(module, module_input, output, accepted=accepted, receipt=receipt)
    audit["output_representation"] = (
        "accepted_canonical" if accepted else "unaccepted_provider_candidate"
    )
    audit["wire_parse_error"] = wire_error
    if not accepted and response.is_file():
        audit["raw_provider_response_source"] = {
            "path": str(response),
            "sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
        }
    # A content match proves identity; a shared basename or array index does not.
    audit["asset_identity_correspondence"] = [
        {
            "public_asset_ref": asset["asset_ref"],
            "module_audio_ref": fact.get("audio_ref"),
            "content_sha256": asset["content_sha256"],
        }
        for asset in context.audio_inventory
        if asset.get("content_sha256")
        for fact in module_input.get("audio_asset_facts", [])
        if fact.get("observed_content_sha256") == asset["content_sha256"]
    ]
    return {
        "public_context": context.model_dump(mode="json"),
        "module_input": module_input,
        "module_output": output,
        "contract_audit": audit,
    }
