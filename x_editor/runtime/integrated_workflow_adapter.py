"""Single-model planning with the production session and execution boundary."""

from x_editor.runtime.separated_semantic_workflow_adapter import (
    SeparatedSemanticWorkflowPlannerAdapter,
)


class IntegratedWorkflowPlannerAdapter(SeparatedSemanticWorkflowPlannerAdapter):
    """Reuse product validation, without an independent Reasoner inference."""

    def _plan_locked(self, user_request, context):
        # This feedback is otherwise delivered only to the second model call.
        # The integrated model needs the same current capability observations.
        self.reasoner.bind_capability_feedback(
            context.get("workflow_perception_capability_feedback")
        )
        return super()._plan_locked(user_request, context)
