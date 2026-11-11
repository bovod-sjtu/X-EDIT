# Documentation-only extension candidates

**SAM-Audio and VoiceSculptor-VD are not included integrations.** They are not
in the shipped tool catalog, planner choices, executor registry, default profile,
service stack or provisioning manifest. This release contains no corresponding
client, MCP wrapper, inference adapter, launcher or backend environment. Setting
their former environment variables does not enable them. Qwen3-TTS has no route
to either tool. Requests selecting an unavailable operation/backend must fail,
not silently select another model.

The links below are background for users who choose to develop their own
extensions. They are not runtime dependencies, supported extension packages,
automatic download instructions, endorsement or evidence of commercial rights.

| Candidate | Upstream references | Review before developing an extension |
| --- | --- | --- |
| SAM-Audio | [Source](https://github.com/facebookresearch/sam-audio), [official model](https://huggingface.co/facebook/sam-audio-large) | Review the exact [SAM agreement](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/LICENSE) and auxiliary components, including [ImageBind's license](https://github.com/facebookresearch/ImageBind/blob/53680b02d7e37b19b124fa37bae4b6c98c38f5be/LICENSE). Do not infer an MIT/Apache grant from the surrounding framework. |
| VoiceSculptor-VD | [Source](https://github.com/ASLP-lab/VoiceSculptor), [model](https://huggingface.co/ASLP-lab/VoiceSculptor-VD) | The model card's Apache declaration does not cover every dependency. The historically used [XCodec2 weights](https://huggingface.co/HKUSTAudio/xcodec2/blob/e412427ed30f0cf9d5e3c95562113deb10a32d03/README.md) declare CC-BY-NC-4.0. Review your complete selected stack separately. |

Historical model/source identifiers are retained in
[extension_references.json](extension_references.json) solely for provenance.
They are not inputs to `deployment/dependencies.py`. The historical SAM mirror
is not asserted to be identical to official weights. Users must resolve their
own authorized source, version, checkpoint and auxiliary dependencies.

## Developing a separate extension

1. Review the intended research/commercial use and all selected licenses. Obtain
   any additional required permission from rights holders before deployment.
   An X·Edit enable flag, installation or successful request grants no license.
   The extension's service operator remains responsible for its own model use;
   delegating installation or asking clients to follow upstream terms does not
   transfer that responsibility.
2. Keep implementation, model acquisition, environments and generated media in
   a separate user-controlled repository and deployment directory. Do not copy
   them into X·Edit's source, release archive, wheel or image.
3. Define your own HTTP contract, input/output identity, failures, timeouts and
   service ownership. A SAM-style separator needs explicit target and residual
   outputs; separation does not establish that the requested target existed.
   A voice-design extension must distinguish new synthesis from editing existing
   speech and must not silently discard words, speaker identity or timing.
4. In your extension repository, create your own capability card, manual, MCP
   wrapper and validation. Generic loaders accept an explicit `catalog_dir`;
   `ToolExecutor(catalog_dir=...)` can use a separately maintained catalog.
5. Use a custom application to wire the catalog into planning and explicitly
   register a handler with `AudioOperationRegistry.register`. Pass that registry
   to `AudioEditExecutor(registry=...)` and expose only authorized, verified
   operations in your application profile. An explicit registry replaces default
   registration; maintain every operation your application needs. This is an
   integration task, not an automatic plugin install or a stock CLI option.
6. Test missing setup, unauthorized backends, output-path integrity, source
   preservation and real model inference independently. Keep records and media
   outside X·Edit. Do not present old benchmark results as extension validation.
7. Review the complete extension deliverable before publishing or using it
   commercially. Its original code and third-party components need their own
   license notices; X·Edit does not relicense or certify them.

No implementation or ready-to-run extension for these candidates is supplied.
The existing native Qwen3-TTS path remains available; it is not a claim of
equivalence to all VoiceSculptor capabilities or a newly validated replacement.
