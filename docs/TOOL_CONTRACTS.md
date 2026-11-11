# Tool and evidence contracts

These rules apply to both tagged-edit backends and are independent of any
benchmark. They do not add model calls, retries, tool availability, or licenses.

## Confirm what to select, not only where it occurs

Candidate lists, transcripts and word timestamps locate possible targets. They
do not automatically establish a requested speaker identity, contextual role,
or acoustic property. Reasoner and Planner must distinguish the selection
criterion from its location. Reuse an existing scoped finding when it resolves
that distinction; otherwise request the specific missing observation within the
configured budget. Do not replace an unresolved target with the first candidate.
If observation cannot resolve it, keep the uncertainty explicit rather than
claiming the edit is grounded. Prompt guidance is not a semantic correctness
guarantee and must be validated with real inference.

## Separate vocals from accompaniment

The configured RoFormer checkpoints provide vocal/accompaniment separation,
not isolation of individual overlapping speakers. A vocal stem may contain
several speakers. Target-speaker extraction requires a separately supported,
enabled and qualified tool; neither this wrapper nor a reference clip enables it.

The typed capability card offers `stem_policy: vocals` (default) and
`stem_policy: instrumental`. The service, remix handler and artifact verifier
share the same selection rule:

- Semantic selection requires exactly one matching provider-labeled stem.
- `speech`, `voice` and `foreground` alias vocals; background/music alias
  accompaniment. They never identify a particular speaker.
- Missing or ambiguous labels fail explicitly, with no fallback to list order.
- Explicit legacy `first`, `last` and `residual` requests remain positional
  compatibility options at the service boundary, not semantic Planner choices.
  Unknown custom policies must uniquely match a filename substring or fail.
- The verifier checks that the delivered file is bound to the requested stem,
  as well as checking readability. Provider labels are declarations, not proof
  of acoustic quality or speaker identity; listening/perception is separate.

Custom checkpoints with different output conventions need an explicit adapter
and capability review. Do not relabel arbitrary outputs just to pass a check.
When remixing, failure to identify the required background must not silently
discard it. Existing plans with positional policies are not rewritten as if
they had originally requested a vocal stem.

## Bind edited text to the actual crop

Transcript-conditioned editing must use the accepted crop, not unqualified
source-wide text or timestamps. The runtime admits two evidence paths:

1. **Fresh crop observation:** the observation's artifact ID, revision and path
   must identify the accepted crop. Word timestamps must be crop-local, complete,
   ordered, non-overlapping and within its duration. The authored source text
   and rendered edit label must match those words. No recognition or label
   repair is performed by the binder.
2. **Safe inheritance:** an accepted single-span, unit-time mapping may project
   complete source words into crop coordinates. It must reference the accepted
   source artifact revision. Overlaps wholly outside the retained interval do
   not invalidate that interval; overlaps or partial words inside it still do.
   Only the already-supported, semantics-preserving label projection is allowed.

Multi-span extraction, crossfades, cut words or ambiguous mappings do not permit
inventing inherited alignment. Planner should request observation of the actual
crop and author crop-local labels. Without that evidence, execution still stops.
Missing evidence does not authorize dropping a requested edit or copying the
input as a successful result. Binding records preserve the source evidence and
parameters before/after; they do not overwrite global observations.

## Connect semantic advice to executable evidence

Reasoner advice helps identify the requested target and unresolved facts; it is
not proof that the selected tool's inputs are ready. Planner remains responsible
for the selected operation and its input contract. Emotion and prosody tools can
require an exact transcript even when the user does not request a word change.
Conflicting transcripts and incomplete word coverage must remain explicit until
resolved. When source text and the selected words are already grounded, local
tags on the full source can avoid unnecessary cropping without broadening the
edit. When a crop is necessary, request its observation if safe inheritance is
unavailable. The runtime does not invent that observation or add retries.

A semantic-property observation can use speaker-attributed review when it also
requests speaker segments, speaker profiles or speaker-attributed transcription.
The semantic condition is retained; a bare property request does not imply a
speaker target. The route still requires the corresponding enabled service.

## Keep speaker edits distinct from timeline extraction

`artifact_ops.speaker_scoped_edit` edits selected spans and reconstructs the
original timeline. Its capability card enumerates the supported nested edits;
the selected tool must also be enabled and satisfy its own input contract.
Extraction, concatenation, reordering and reconstruction are standalone timeline
operations, not nested speaker edits. Unsupported nesting fails before invoking
an editing service; it is not silently rewritten into a different user objective.

## Emotion labels at the pipeline transport boundary

For `edit_data_pipeline` emotion editing, whitespace-only separators between
emotion tags are attached to an adjacent tagged span before HTTP transmission.
This prevents a separator being synthesized as an empty neutral segment by the
pipeline parser. The rendered source, spoken words and emotion controls are
unchanged. The adapter records authored and transmitted labels in
`emotion_label_transport`; it does not merge distinct emotion spans, drop text
or alter retry budgets. An explicitly empty emotion target is rejected.
This adaptation does not apply to `dots_tts_edit` requests and is not a guarantee
of identical model-generated audio.

## Preserve explicit configuration through tool selection

Tool selection retains an explicitly bound, frozen development-prior snapshot
when every operation in its comparisons remains enabled. It does not recalculate
rankings or use evaluation outcomes. If selection removes a compared operation,
the complete snapshot and binding are disabled together with a warning; a
partial comparison is not silently treated as the original frozen evidence.
Supply a separately validated snapshot or explicitly omit both prior fields for
that selection. Changing backend implementations also requires checking whether
the configured development evidence applies to those implementations.

Snapshot presence in configuration is not proof that a planning mode consumes
it: the traceable planner adapter projects this guidance, whereas the current
integrated/advisory planner path does not. Check actual module inputs when
comparing runs. Reasoner remains an adviser, not an automatic post-failure retry
loop; forward-mode execution and its observation budgets remain unchanged.

## Validation boundary

Schema checks, deterministic evidence replay and mocked-tool integration can
verify these contracts. They do not establish final audio quality or a benchmark
score improvement. New inference must use a separate run directory with fixed
source, configuration, model versions and recorded inputs; retain failures and
old results. Compare old and new audio with the same scoring setup before
attributing a metric change to the framework.
