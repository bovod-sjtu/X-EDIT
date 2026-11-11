"""Client for the local paralinguistic event asset bank."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from x_editor.runtime.paralinguistic_events import materialize_paralinguistic_event_asset


class ParalinguisticEventBankClient:
    def materialize(self, **kwargs: Any) -> dict[str, Any]:
        return materialize_paralinguistic_event_asset(
            event=str(kwargs["event"]),
            output_path=str(kwargs["output_path"]),
            source_audio_path=kwargs.get("source_audio_path"),
            event_asset_path=kwargs.get("event_asset_path"),
            event_bank_dir=kwargs.get("event_bank_dir"),
        )
