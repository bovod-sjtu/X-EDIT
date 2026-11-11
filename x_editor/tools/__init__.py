"""X-Editor tool metadata and execution package.

The new tool layer is intentionally local to X-Editor.  It does not import
or mutate AUDIO_AGENT paths; concrete MCP execution will be implemented under
``x_editor.tools`` and tool metadata is loaded from ``tools/catalog``.
"""

__all__: list[str] = []
