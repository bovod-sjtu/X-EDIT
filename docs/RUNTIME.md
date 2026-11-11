# Runtime contract

`RuntimeConfig` resolves paths relative to its configuration file and validates
the unchanged `OpenProviderWorkflowProfile`. CLI and Python both delegate to
`XEdit`, then `OpenProviderWorkflowRuntime`. No duplicate Planner or Executor is
introduced. `forward` calls `arun_scenario(..., arm="W1")`; `revision` wraps the
same factory in `IterativeScenarioRuntime`.

Inputs use `WorkflowRuntimeScenario`: ordered source/reference assets, ordered
public turns, and optional previous-output selection. File paths are resolved
privately; model-facing assets retain opaque references. The public adapter
checks audio readability and SHA-256 before starting. Wrong roles, duplicate
references, missing files and mismatched hashes are rejected, not repaired by
guessing. `configs/request.example.json` shows a two-turn session.

`configs/forward.json` contains the complete frozen actor preset. The name
`evaluator_mode=disabled` describes the **inner Forward**; outer revision is
configured independently in `configs/revision.json`. Do not turn on a second
inner repair loop. Model outputs retain the existing Graph, high-level edit
program, Functions and Command contracts; changing the representation is an
explicit profile change, not an automatic fallback.

Only endpoint/model relocation is supported through `XEDIT_*_URL/MODEL`.
Temperature, seeds, token limits, capability set, timeouts, and perception budget
come from the profile. Core tools continue to use their documented environment
variables. Authentication is never read from sibling projects. The default
review services are local, unauthenticated OpenAI-compatible endpoints; put
remote services behind a private/authenticated gateway rather than exposing
these defaults publicly.

Each run owns a new directory:

| File/directory | Meaning |
| --- | --- |
| `input.json` | Private resolved scenario, including asset hashes |
| `runtime.json` | Effective actor, scheduling and revision settings |
| `forward/` | Forward module calls, source evidence, execution and committed result |
| `revision/` | Attempts, semantic/audio checks, feedback, and selection receipts |
| `public_designs/` | Candidate-blind questions derived from public task information |
| `result.json` | Final selected artifact and result, or explicit nondelivery |
| `error.json` | Adapter/runtime exception; never disguised as delivery |

Review support formerly imported from `examples/evaluation` is packaged under
`x_editor.revision`. Function/class source was extracted mechanically and
imports relocated. Question prompts, scoring, raw-error routing and selection
rules are unchanged. Scope-plan reuse requires matching inputs; independent
benchmark judging is not run by this API.

The existing failed-prefix continuation can be selected with
`execution_reuse="failed_prefix"` in revision settings. It revalidates a prefix
against the new admitted plan; it does not assert that the prefix sounds good,
refund budget, or guarantee a skipped call. It remains disabled in the supplied
preset. Full quality-based automatic rollback is not promised.

For simultaneous jobs, use separate processes/output directories. The inherited
HTTP recording context patches transport methods within a process; concurrent
`XEdit.arun` invocations in one process are not yet qualified.
