# Qwen3-TTS Tool

MCP wrapper for a running Qwen3-TTS HTTP service.

Default custom-voice service URL: `http://127.0.0.1:8017`.
Default voice-clone service URL: `http://127.0.0.1:8019`.

Override with:

```bash
export QWEN3_TTS_SERVICE_URL=http://127.0.0.1:8017
export QWEN3_TTS_CLONE_SERVICE_URL=http://127.0.0.1:8019
```

For managed startup, set `QWEN3_TTS_SERVICE_COMMAND` and, for clone mode,
`QWEN3_TTS_CLONE_SERVICE_COMMAND`.  Accent-conversion callers may pass
`timeout_seconds` so each clone chunk fails clearly instead of blocking the
whole edit indefinitely.

## Native voice design

Only the native Qwen3-TTS backend is included. `voice_design_backend` accepts
`qwen3_tts` or `auto`; other values fail instead of invoking another model.
Use `instruct` for supported style requests. No alternate voice-design adapter
is installed or registered.

The pinned pipeline provides the `qwen3_tts_service.api_service` module.
Use the corresponding entries in `deployment/stack.json`; see
[deployment instructions](../../../../docs/DEPLOYMENT.md) for model downloads,
isolated environments and configured endpoints.
