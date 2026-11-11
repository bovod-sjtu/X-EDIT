# ClearerVoice-Studio Backend

This directory contains the ClearerVoice backend used by the planner-visible
`speech_enhancement` tool.

The backend owns the ClearVoice runtime, isolated environment setup, local
caches, and the shared service process. The planner should not route directly
to this backend; it should call `speech_enhancement` through MCP.

Run:

```bash
cd x_editor/tools/catalog/clearer_voice_studio
./setup.sh
./check_environment.sh
.venv/bin/python service.py --host 127.0.0.1 --port 8027
```

`setup.sh` installs the FastAPI backend dependencies into `.venv`. It does not
install the full `clearvoice` runtime by default because that package pulls
large model/runtime dependencies and is slow on shared storage. Use one of these runtime
paths:

- run `X_EDITOR_CLEARERVOICE_INSTALL_FULL_DEPS=1 ./setup.sh` to install
  `clearvoice==0.1.2` into this `.venv`;
- or start `service.py` with an existing isolated ClearVoice Python, for example
  `<clearvoice-venv>/bin/python service.py --host 127.0.0.1 --port 8027`.

Current HTTP contract:

- `GET /health`
- `POST /enhance`

By default, `/enhance` fails on ClearerVoice inference errors instead of
copying the input audio. Set `X_EDITOR_CLEARVOICE_ALLOW_COPY_FALLBACK=1` only
when explicitly debugging pipeline continuity rather than measuring real
enhancement success.

Vocal/accompaniment separation is served by `audio_separator_roformer` on port
`8042`. It cannot isolate individual overlapping speakers. Target-speaker
extraction requires a separately available, qualified tool; this enhancement
service does not provide it.
