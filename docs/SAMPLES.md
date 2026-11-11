# Run representative samples

> Historical snapshot (2026-10-08 and earlier). The current pipeline layout and
> verification are in [PIPELINE_MIGRATION.md](PIPELINE_MIGRATION.md). These earlier
> results do not certify the new backend revision.

Follow [DEPLOYMENT.md](DEPLOYMENT.md), then load the generated bindings:

```bash
source /data/xedit/client.env
uv run --locked x-edit doctor --config configs/runtime.yaml
uv run --locked x-edit doctor --config configs/runtime.yaml --live
```

`doctor --live` checks the complete optional catalog. A missing optional service
can fail that check even when a selected edit works. Health is not inference.
Do not advertise unavailable tools in a reduced deployment profile.

## Generate diagnostic inputs and check actual services

```bash
/local-ssd/xedit-envs/edit_cpu/bin/python deployment/qualify_services.py \
  --data /data/xedit --output /data/checks/services-001
```

The command generates `source.wav` saying “The next train will leave at six
fifteen.” and `tts.wav`, a Chinese voice reference, with deployed Qwen TTS.
It then makes real service calls. This fixed sentence is only in the
qualification harness, not in controller routing or planning rules. No private
dataset or sibling project is needed. Use authorized input recordings; generated
audio remains in your run directory and is not part of this source release.

## Full Forward

```bash
uv run --locked x-edit edit --config configs/runtime.yaml \
  --audio /data/checks/services-001/source.wav \
  --instruction 'Replace train with bus, preserving all other words and the voice.' \
  --mode forward --output /data/runs/text-001

uv run --locked x-edit edit --config configs/runtime.yaml \
  --audio /data/checks/services-001/source.wav \
  --reference /data/checks/services-001/tts.wav \
  --instruction 'Match the reference voice while preserving the source words.' \
  --mode forward --output /data/runs/voice-001
```

For multiple turns/assets, edit the absolute input paths in
`configs/request.example.json`, then use `x-edit run --request ...` as in the
README. The second turn consumes the preceding output, not the original audio.

## Sequential policy revision

```bash
uv run --locked x-edit edit --config configs/runtime.yaml \
  --audio /data/checks/services-001/source.wav \
  --instruction 'Lower the recording by 6 dB while preserving its words and duration.' \
  --mode revision --output /data/runs/revision-001
```

No delivery leads to actual-error feedback; a delivered attempt receives
semantic review before audio review. A passing attempt need not retry. Judge
scores do not certify true quality. Historical fault injection was an external
harness, deliberately not a runtime feature.

Use a new output directory each time. Retain `result.json`, `error.json` if
present, module/tool records and failures. Check text, requested effect,
preservation and artifacts separately. CLI exit zero means delivery.

## Verified scope

The historical 24-task cold-deployment panel completed all tasks and delivered
23. Its [sanitized evidence](sample_validation.json) includes a speech-only
attenuation task with background preservation that stopped before execution:
the proposed graph did not specify a unique primary output. A pause edit
delivered audio but realized 0.2 seconds instead of 0.3 seconds. These are not
relabeled as successful edits. Broad human listening is still needed.
