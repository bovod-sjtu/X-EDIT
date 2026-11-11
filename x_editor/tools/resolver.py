"""Audio path resolver and version manager for X-Editor.

Handles:
- Resolving 'current' audio_id to actual file path
- Auto-generating output paths for tool executions
- Tracking audio version history
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from x_editor.core.schemas import AudioVersion


class AudioResolver:
    """Resolves audio references and manages versioned outputs."""

    def __init__(self, temp_dir: Path) -> None:
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._version_counter = 0
        self._base_copied = False

    def setup_base(self, base_audio_path: str) -> str:
        """Copy base audio into temp directory and return its path."""
        src = Path(base_audio_path)
        dst = self.temp_dir / f"base_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        self._base_copied = True
        return str(dst)

    def resolve_audio_path(self, audio_ref: str, current_path: str) -> str:
        """
        Resolve an audio reference to an actual path.
        Supports:
        - 'current' -> current processing audio
        - 'base' -> original base audio
        - absolute/relative path -> as-is
        """
        if audio_ref == "current":
            return current_path
        if audio_ref == "base":
            base = self.temp_dir / "base_*"
            matches = list(self.temp_dir.glob("base_*"))
            if matches:
                return str(matches[0])
            return current_path
        return audio_ref

    def generate_output_path(self, tool_name: str, step_id: str) -> str:
        """Generate a versioned output path for a tool execution."""
        self._version_counter += 1
        filename = f"v{self._version_counter:03d}_{step_id}_{tool_name}.wav"
        return str(self.temp_dir / filename)

    def add_version(
        self,
        versions: list[AudioVersion],
        path: str,
        step_id: str,
        tool_name: str,
        description: str,
    ) -> list[AudioVersion]:
        """Append a new version record."""
        version = AudioVersion(
            version_id=len(versions) + 1,
            path=path,
            step_id=step_id,
            tool_name=tool_name,
            description=description,
        )
        versions.append(version)
        return versions

    def prepare_tool_parameters(
        self,
        parameters: dict[str, Any],
        current_audio_path: str,
        step_id: str,
        tool_name: str,
    ) -> dict[str, Any]:
        """
        Prepare tool parameters by resolving audio references and generating output path.
        Returns a copy of parameters with resolved paths.
        """
        params = dict(parameters)

        # Resolve audio_path or audio_id
        for key in list(params.keys()):
            if key in ("audio_path", "audio_id", "input_path"):
                val = params.pop(key)
                params["audio_path"] = self.resolve_audio_path(str(val), current_audio_path)

        # Auto-generate output_path if 'auto' or missing
        if "output_path" in params:
            if params["output_path"] in ("auto", "", None):
                params["output_path"] = self.generate_output_path(tool_name, step_id)
        else:
            # Only add if tool typically produces audio output
            # Analysis tools don't need output_path
            pass

        return params
