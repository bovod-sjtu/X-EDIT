"""Optional action-oriented interpretation of already scoped retry guidance."""

from typing import Literal

RetryGuidanceMode = Literal["diagnostic", "actionable/v1"]

ACTIONABLE_RETRY_POLICY = (
    "This is a fresh planning decision. The prior failure pattern is summarized in "
    "source_failure_codes and iteration_feedback.unresolved_work; the current repair "
    "guidance is in next_checks. Do not reconstruct the failed attempt's reasoning or "
    "copy its plan. Use current source evidence, the current Reasoner conclusions, "
    "public session history, and all available tool contracts. Supported conclusions "
    "remain valid only when they still follow from these inputs. A semantic target in "
    "a prose memo specifies the intended audible meaning, not necessarily a literal "
    "synthesis string. Preserve explicit user wording and typed contract requirements. "
    "Use a supported, explicit target wording draft from the current Reasoner memo "
    "when compiling synthesis arguments; do not replace that draft with a shorthand "
    "semantic label merely because it appeared in an earlier memo. Preserve its "
    "meaning and update dependent target fields consistently under the tool contract. "
    "A missing target wording draft is an upstream handoff gap, not evidence that "
    "the semantic calculation is wrong. Never rewrite admitted source text, "
    "source-side tags, coordinates, or other observed facts to fit a hypothesis. "
    "In the existing decision_summary, briefly state which relevant guidance you "
    "applied, or why you retained the previous choice or need another observation. "
    "A legal plan is not evidence that its audio will satisfy the request. Do not "
    "claim that an unchanged call has implemented a proposed change. Outcome "
    "verification belongs after execution, not in this pre-execution decision. "
    "Guidance is a diagnostic hypothesis, not evidence or authority to bypass validation."
)
