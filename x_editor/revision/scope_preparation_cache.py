"""Production review support extracted without changing function bodies."""

from copy import deepcopy


from x_editor.revision.refinement_cache import semantic_identity


from x_editor.evaluation.iterative_runtime import write_record


from x_editor.evaluation.runtime_scope_contracts import (
    SCOPE_COVERAGE_PROMPT, SCOPE_PLAN_PROMPT,
)


def prepare_once(*, cache, prompt, payload, args, assets, destination, call):
    """Keep full public evidence/version identity and rerun all downstream checks.

    Only deterministic text preparation is eligible. A new observation, source,
    candidate, model setting or prompt makes a new request. Scores, independent
    repetitions and other semantic calls cannot enter this cache.
    """
    if cache is None or prompt not in (SCOPE_PLAN_PROMPT, SCOPE_COVERAGE_PROMPT):
        return call()
    identity = {
        "kind": "exact_scope_preparation/v1",
        "prompt": prompt, "payload": payload, "assets": assets,
        "provider": {**semantic_identity(args), "seed": getattr(args, "seed", 0),
                     "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
    }
    (response, source), reused, key = cache.once(
        identity, lambda: (call(), str(destination)),
    )
    write_record(destination / "preparation_reuse.json", {
        "key": key, "source": source, "reused": reused,
        "new_provider_calls": 0 if reused else 1,
        "scope": "episode; identical full input; preparation only",
        "acoustic_votes_reused": False,
        "downstream_binding_and_coverage_validation_required": True,
    })
    return deepcopy(response)
