"""Accent and dialect reference library.

This package contains code and label definitions only. Reference recordings and
manifests are user-provided external resources; see README.md for construction.
"""

from .labels import (
    WU_CHINESE_FAMILY,
    WU_SUBDIALECTS,
    AccentDefinition,
    accents_compatible,
    canonical_label,
    list_accent_definitions,
    list_canonical_labels,
)
from .schemas import AccentLibraryEntry, AccentMatch
from .store import AccentDialectLibrary, default_library_dir, default_library_path


__all__ = [
    "AccentDefinition",
    "AccentDialectLibrary",
    "AccentLibraryEntry",
    "AccentMatch",
    "WU_CHINESE_FAMILY",
    "WU_SUBDIALECTS",
    "accents_compatible",
    "canonical_label",
    "default_library_dir",
    "default_library_path",
    "list_accent_definitions",
    "list_canonical_labels",
]
