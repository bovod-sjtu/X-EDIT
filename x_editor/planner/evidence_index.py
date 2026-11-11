"""Lossless prompt representation of the admitted evidence reverse index.

Runtime validation retains the original map. This representation factors repeated
source IDs and path prefixes without selecting, summarizing, or dropping evidence.
"""

from typing import Any, Literal

EvidenceIndexFormat = Literal["expanded", "prefix-plus-suffix/v1"]


def factor_evidence_index(index: dict[str, list[str]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], dict[str, list[str]]] = {}
    for path, sources in index.items():
        prefix, separator, field = path.rpartition(".")
        # Keep the separator in the prefix: concatenation is exact even for keys
        # containing quoted selectors, dots, empty segments, or no separator.
        prefix = prefix + separator
        groups.setdefault(tuple(sources), {}).setdefault(prefix, []).append(field)
    return {
        "encoding": "prefix-plus-suffix/v1",
        "decode": (
            "Each prefix + suffix is one exact canonical evidence_path. All paths "
            "in a group have that group's evidence_ids. Concatenate verbatim; do "
            "not cite a prefix alone or use an evidence ID as a path."
        ),
        "groups": [
            {"evidence_ids": list(sources), "paths": paths}
            for sources, paths in groups.items()
        ],
    }


def expand_evidence_index(index: dict[str, Any]) -> dict[str, list[str]]:
    """Reconstruct the exact map for round-trip audits of the prompt boundary."""
    if index["encoding"] != "prefix-plus-suffix/v1":
        raise ValueError("Unsupported evidence index encoding")
    expanded: dict[str, list[str]] = {}
    for group in index["groups"]:
        for prefix, suffixes in group["paths"].items():
            for suffix in suffixes:
                path = prefix + suffix
                if path in expanded:
                    raise ValueError("Duplicate evidence path")
                expanded[path] = list(group["evidence_ids"])
    return expanded
