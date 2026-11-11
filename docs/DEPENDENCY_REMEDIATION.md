# External dependencies and distribution

Updated 2026-10-26. The release boundary is defined in [RELEASE_SCOPE.md](RELEASE_SCOPE.md).
Original framework code is distributed under Apache-2.0; external model services,
their implementation checkouts, weights, datasets and generated audio are
obtained or produced separately.

## Reference project

The reference separates original orchestration and service interfaces from
independently acquired SDKs and weights. X·Edit now pins its provider-neutral
revision `c36f753a91896e901ba9cf621ac02d6ef507a21d`, inspected through authorized
Git access. Its root LICENSE is Apache-2.0. Its six SDK gitlinks remain disclosed;
that does not make all of their weights commercially usable.

X·Edit does not fetch, install or launch F5, and does not call its editing
interfaces before explicit runtime opt-in. Missing setup returns
“请自行下载并安装F5-TTS”. F5 remains a disclosed functional prerequisite.

SAM and VoiceSculptor are removed from the shipped toolset and code integrations.
No corresponding clients, MCP wrappers, capability cards or runtime registrations
remain, and Qwen does not redirect to VoiceSculptor. They appear only in
[independent extension guidance](EXTENSION_GUIDE.md) and historical provenance.
No replacement model or commercial authorization is inferred.

## Source and weight declarations

| Integration | Selected declarations | Distribution boundary |
| --- | --- | --- |
| F5-TTS | [SDK](https://github.com/SWivid/F5-TTS/tree/27e20fcf39fbce3f187a61d4a3e92065d2d9b947): MIT; [weights](https://huggingface.co/SWivid/F5-TTS/blob/84e5a410d9cead4de2f847e7c9369a6440bdfaca/README.md): CC-BY-NC-4.0. | External synthesis/editing service; SDK, weights and resulting audio are not bundled. |
| SAM-Audio | [Source agreement](https://github.com/facebookresearch/sam-audio/blob/bb4c6999d2677c7402360e426afc01ddfad6dce0/LICENSE): SAM License; [ImageBind](https://github.com/facebookresearch/ImageBind/blob/53680b02d7e37b19b124fa37bae4b6c98c38f5be/LICENSE): CC-BY-NC-SA-4.0. | Documentation-only extension candidate, not a shipped tool or dependency; review any user-selected implementation separately. |
| VoiceSculptor-VD | [Model card](https://huggingface.co/ASLP-lab/VoiceSculptor-VD/blob/6d7a5573960727b00a5b8f3de032a22dfee7f16e/README.md): Apache-2.0; [xcodec2 weights](https://huggingface.co/HKUSTAudio/xcodec2/blob/e412427ed30f0cf9d5e3c95562113deb10a32d03/README.md): CC-BY-NC-4.0. | Documentation-only extension candidate, not a shipped adapter; the model card does not remove auxiliary restrictions. |

Other declarations include noncommercial DiariZen and MaskGCT weights,
Qwen2.5-Omni-3B research terms, IndexTTS custom terms and checkpoint-specific
RoFormer rights requiring confirmation. Seed-VC's GPL terms are copyleft, not
noncommercial restrictions. CC-BY attribution and mixed per-file licenses are
recorded separately. See [the full inventory](LICENSES.md).

SAM and VoiceSculptor integrations are removed, not automatically replaced. Previously researched replacement candidates are
recorded as inactive in `third_party/release_review.json`; none has been substituted
without license and capability verification.

## FFmpeg distribution

The old Dockerfiles installed the OS `ffmpeg` package. The inspected development
host reports `--enable-gpl` and GPLv2-or-later; this observation is about the host,
not a measurement of any previously published container image.

Both Docker build paths now use `deployment/build_ffmpeg.sh`. It builds the
official FFmpeg 8.1.2 archive at a fixed SHA256, disables GPL, nonfree, version3 and
dependency autodetection, and builds shared libraries. The controller image has
a separate build stage; the GPU OS installer uses the same build script.
The decord installation probe generates its diagnostic video with FFmpeg's
native MPEG-4 encoder, so that probe no longer requires GPL libx264.
The complete prefix retains corresponding source, checksum, source URL, LGPL
notice, license explanation and build configuration.

```bash
# Use a fresh dedicated image. Default prefix: /opt/xedit-ffmpeg.
bash deployment/build_ffmpeg.sh
python3 deployment/check_ffmpeg.py --prefix /opt/xedit-ffmpeg
# Exit 2 means the binary fails this release policy.
python3 deployment/check_ffmpeg.py --binary /usr/bin/ffmpeg
```

`XEDIT_FFMPEG_PREFIX` selects a new absolute destination; `XEDIT_FFMPEG_JOBS`
defaults to four workers. Existing prefixes are not replaced. Manual host installs
must put the prefix's `bin` first on `PATH`; Dockerfiles configure it themselves.
No shared host package or running model service was changed in this iteration.

Follow [FFmpeg's distribution guidance](https://www.ffmpeg.org/legal.html).
Keep the complete source/notice directory in published images. LGPL FFmpeg is
third-party software and is not relicensed under Apache. The prefix checker does
not audit extra media binaries or native libraries in a custom base or Python
wheels; the final images require a separate complete distribution inspection.

## Repeatable checks

```bash
python3 deployment/license_audit.py --output /path/to/review/license-audit.json
python3 deployment/audit.py
python3 deployment/release_audit.py
```

The first command walks direct and transitive declared service dependencies and
compares source/model pins with their license evidence. It separately reports
noncommercial restrictions, copyleft, components requiring separate review, missing
evidence and unresolved scope. Exit 2 is expected until the recorded blockers and
unreviewed scope are resolved. It does not remove services, download models or
turn external URLs into approval. The source audit cannot certify commercial
readiness.

The 2026-10-21 validation compiled FFmpeg in a separate temporary prefix, checked both
executables and source/notices, and exercised WAV-to-FLAC conversion and ffprobe.
The same checker rejected the host GPL binary. Replacement-model inference and
full Docker image builds have not run. No release was published or submitted.
