"""An audio hash identifies an asset, not all evidence used to assess it."""


def scope_revision(previous, current):
    """Only explicitly bound, changed observation contexts permit reassessment."""
    old = (previous.scope_review or {}).get('configuration_fingerprint')
    new = (current.scope_review or {}).get('configuration_fingerprint')
    same_target = (previous.artifact_sha256 == current.artifact_sha256
                   and previous.design_fingerprint == current.design_fingerprint
                   and previous.assessor_role == current.assessor_role)
    return bool(same_target and old and new and old != new)


def validate_cached_assessment(previous, current):
    def terminal(value):
        if value is None:
            return None
        scope = dict(value.scope_review or {})
        # Step diagnostics do not change a fixed final-artifact utility. They
        # keep their own version-bound record and may differ after replanning.
        scope.pop('execution_purposes', None)
        return value.model_copy(update={'scope_review': scope or None})
    if previous is not None and terminal(previous) != terminal(current) and not scope_revision(previous, current):
        raise ValueError('identical artifact received contradictory cached utility in the same assessment context')
