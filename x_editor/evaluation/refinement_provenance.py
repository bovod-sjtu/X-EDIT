"""Bind declared repair reasons to observed reviews, without another judge call.

This records which evidence a guide cites. It does not estimate a reviewer's
causal effect or certify the cited model judgment as acoustic ground truth.
"""

from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint


def diagnostic_catalog(atomic, utility=None, observations=(), *, purpose_observations=()):
    entries = {}

    def add(channel, kind, value):
        entry = {"channel": channel, "kind": kind, **value}
        entries["diagnostic:" + canonical_evaluation_fingerprint(entry)] = entry

    for position in atomic:
        withheld = set(position.get("withheld_as_facts", ()))
        for answer in position["answers"]:
            if answer["question_id"] not in withheld and answer["grade"] in (0, 1, 2):
                add("semantic", "module_question", {
                    "turn_index": position["turn_index"], "module": position["module"],
                    "answer": answer,
                })
    if utility:
        for answer in utility.answers:
            if answer.grade is not None and answer.grade < 3:
                add("multimodal", "terminal_criterion", {
                    "artifact_sha256": utility.artifact_sha256, "answer": answer.model_dump(mode="json"),
                })
        for row in (utility.scope_review or {}).get('execution_purposes',{}).get('observations',[]):
            if row.get('grade') is not None and row['grade'] < 3:
                add('multimodal', 'execution_purpose', {'observation':row})
    for row in purpose_observations:
        if row.get('grade') is not None and row['grade'] < 3:
            add('multimodal', 'execution_purpose', {'observation':row})
    for observation in observations:
        if observation.get("claim_check", {}).get("presence") in {"present", "absent"}:
            add("multimodal", "targeted_observation", {"observation": observation})
    return entries


def bind_provenance(draft, catalog, guidance):
    rows = []
    for turn in draft.turns:
        packet = guidance.get(turn.turn_index)
        if packet is None:
            continue
        citations = {identity: catalog[identity] for identity in turn.repair_evidence_ids if identity in catalog}
        missing = sorted(set(turn.repair_evidence_ids) - set(citations))
        channels = sorted({entry["channel"] for entry in citations.values()})
        category = ("joint" if len(channels) == 2 else channels[0]) if channels and not missing else "unattributed"
        rows.append({
            "turn_index": turn.turn_index, "guidance_id": packet.guidance_id,
            "target_modules": list(packet.target_modules), "category": category,
            "cited_diagnostics": citations, "unbound_citations": missing,
            "interpretation": "Declared evidence used to form guidance; not a causal ablation.",
        })
    return rows
