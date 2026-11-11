# Paralinguistic Event Bank Tool

Local MCP wrapper that materializes short breath, laugh, cough, or sigh event
assets for `paralinguistic_event_add`.

Run:

```bash
cd x_editor/tools/catalog/paralinguistic_event_bank
./setup.sh
uv run --project . python mcp_server.py
```

The tool supports explicit `event_asset_path` or recursive `event_bank_dir`
lookup. If neither is available, it writes a deterministic synthetic fallback
asset and reports that fallback in `event_asset_debug`.
