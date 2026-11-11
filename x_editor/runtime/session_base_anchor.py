"""Keep history selection fixed without freezing a provisional multi-asset role."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionBaseAnchor:
    node_id: str | None
    audio_ref: str | None
    asset_is_explicit: bool

    @classmethod
    def from_turn(cls, turn: Any, memo: Any) -> "SessionBaseAnchor":
        return cls(
            turn.selected_base_state_node_id,
            turn.selected_base_audio_ref,
            memo.selected_base_state_node_id is not None
            or memo.selected_base_audio_ref is not None
            or memo.requested_change_ledger_status != "provider_omitted",
        )

    def permits(self, node_id: str | None, audio_ref: str | None, *, state: Any,
                bound_audio_refs: set[str], allow_asset_selection: bool) -> bool:
        if (node_id, audio_ref) == (self.node_id, self.audio_ref):
            return True
        if (not allow_asset_selection or self.asset_is_explicit or node_id != self.node_id
                or state is None or audio_ref not in bound_audio_refs
                or self.audio_ref not in bound_audio_refs):
            return False
        nodes = [node for node in state.nodes if node.node_id == self.node_id]
        return (len(nodes) == 1 and nodes[0].session_eligible
                and nodes[0].contains_audio_ref(self.audio_ref)
                and nodes[0].contains_audio_ref(audio_ref))
