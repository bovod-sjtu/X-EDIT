# ACE-Step 1.5 Music Vocal Tool

This catalog entry integrates the official ACE-Step 1.5 API as an X-Editor
tool for requests such as "make this speaker sing", "turn this line into rap",
"make a cover", or "repaint this section as musical vocals".

## Capability Boundary

- Primary task: music vocal generation/editing, including singing, rap,
  text-to-music, cover, and repaint.
- ACE-Step can use lyrics, music captions, source audio, and reference audio.
- This is not a strict speech TTS, voice conversion, lip-sync, or
  timing-preserving dialogue editor. Speaker identity, exact pauses, exact word
  timing, and original background reconstruction are best-effort or out of
  scope.
- X-Editor does not ship ACE-Step model weights and does not fake successful
  output. If the official API is missing, the MCP call fails with a clear
  availability message.

## Research Notes

Sources checked during integration:

- https://github.com/ace-step/ACE-Step-1.5
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md
- https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
- https://huggingface.co/ACE-Step/Ace-Step1.5

Public docs/model card state:

- License: MIT.
- Python: 3.11-3.12 for the official runtime.
- Default official API URL: `http://localhost:8001`; X-Editor defaults to
  `http://127.0.0.1:8044` to avoid the common `8001` collision with local
  vLLM/OpenAI-compatible services.
- API flow: `POST /release_task`, poll `POST /query_result`, download with
  `GET /v1/audio?path=...`.
- Main task types used by this wrapper: `text2music`, `cover`, `repaint`.
- Audio duration is documented as 10-600 seconds.
- Core models are auto-downloaded on first run; docs mention about 10GB disk and
  GPU/CPU/MPS/ROCm/Intel paths, with CUDA GPU recommended.

## Install Wrapper Environment

```bash
cd x_editor/tools/catalog/ace_step_music
./setup.sh
./check_environment.sh
```

This installs only the lightweight X-Editor wrapper.

## Install and Run Official ACE-Step

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git /path/to/ACE-Step-1.5
cd /path/to/ACE-Step-1.5
uv sync
PORT=8044 uv run acestep-api
```

The first run downloads ACE-Step model artifacts. Do not commit downloaded
weights or private tokens.

Point X-Editor at the service:

```bash
export ACE_STEP_API_URL=http://127.0.0.1:8044
```

In this worktree the executor can also bootstrap a local official checkout when
`ACE_STEP_API_URL` is offline. It uses `ACE_STEP_REPO_DIR` when set, otherwise
`/path/to/workspace/ACE-Step-1.5`, and starts a conservative
low-memory API server with `ACESTEP_INIT_LLM=false` by default. Timeline
cover/repaint calls default `use_format=false` unless explicitly overridden, so
they do not require ACE-Step's 5Hz LM on crowded shared GPUs.

If you intentionally run the official ACE-Step API on its documented default
port, set `ACE_STEP_API_URL=http://127.0.0.1:8001`. In this workspace, port
`8001` is often occupied by Qwen/vLLM, whose `/health` endpoint returns 200 but
does not expose ACE-Step's `/release_task` or `/query_result` routes. The client
checks those routes before generation and fails early if the URL points at the
wrong service.

Optional API key, if the upstream server enables auth:

```bash
export ACE_STEP_API_KEY=...
```

## Optional X-Editor Adapter Service

The MCP wrapper can call the official API directly. If a stable X-Editor-local
adapter is useful:

```bash
cd x_editor/tools/catalog/ace_step_music
export ACE_STEP_UPSTREAM_URL=http://127.0.0.1:8044
.venv/bin/python service.py --host 127.0.0.1 --port 8043
```

Health:

```bash
curl http://127.0.0.1:8043/health
```

## MCP Tool Call Shape

```json
{
  "instruction": "Make this sentence sound like confident rap vocals.",
  "lyrics": "We ship the editor today",
  "music_prompt": "hip hop beat with confident rap vocals, clear rhythm",
  "output_path": "/data/xedit/runs/s1_ace_step_music.wav",
  "mode": "text2music",
  "vocal_style": "rap",
  "vocal_language": "en",
  "duration_seconds": 10,
  "audio_format": "wav"
}
```

Cover/repaint calls can add `source_audio_path` and `reference_audio_path`.
Because the official API accepts local server paths and multipart uploads, this
wrapper uploads local source/reference files when present.
