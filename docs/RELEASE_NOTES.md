# Release notes

## 2026-11-17 — public upstream integration

- Point the pipeline gitlink and fetch manifest to the public GitHub repository.
- Adapt service startup to the required pipeline configuration arguments and
  remove the obsolete artifact-path monkeypatch.
- Pin the current dots.tts runtime and dots.tts.edit V1.1 weights separately.
  The default backend remains the model; the pipeline is explicitly selectable.
- Add the manuscript framework figure and a direct English research README.
- Remove superseded preparation reports, copied machine inventories, and an
  unused model launcher. Git revisions and deployment receipts replace the
  obsolete per-file source export inventory.
- Preserve isolated environments and the external-installation/license boundary.
  No planning prompts, core algorithms, benchmark records, or score rules change.
- See [VALIDATION.md](VALIDATION.md) for the checks actually completed on this version.

## Earlier releases

Earlier releases added tool-selected capabilities, the model/pipeline switch,
separate service and MCP environments, and explicit external-tool consent.
Their operation tests and benchmark results remain tied to their original
versions. Consult Git history rather than applying historical acceptance
reports to updated dependencies.
