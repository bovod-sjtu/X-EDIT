"""Serializable data structures for the accent/dialect library."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .labels import accent_definition, canonical_label


@dataclass
class AccentLibraryEntry:
    id: str
    canonical_accent: str
    reference_audio_path: str
    transcript: str | None = None
    gender: str = "unknown"
    age_group: str = "unknown"
    speaker_id: str | None = None
    duration: float | None = None
    sample_rate: int | None = None
    language: str | None = None
    locale: str | None = None
    region: str | None = None
    family: str | None = None
    label_type: str = "dialect"
    aliases: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = canonical_label(self.canonical_accent) or self.canonical_accent
        if not self.id.strip():
            raise ValueError("AccentLibraryEntry requires a non-empty id")
        if not canonical:
            raise ValueError("AccentLibraryEntry requires canonical_accent")
        self.canonical_accent = canonical
        definition = accent_definition(canonical)
        if definition:
            self.label_type = definition.label_type
            self.language = self.language or definition.language
            self.locale = self.locale or definition.locale
            self.region = self.region or definition.region
            self.family = self.family or definition.family
            if not self.aliases:
                self.aliases = list(definition.aliases)
        if not self.reference_audio_path:
            raise ValueError(f"Entry {self.id} requires reference_audio_path")

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "AccentLibraryEntry":
        def json_field(name: str) -> dict[str, Any]:
            value = row.get(name)
            if isinstance(value, dict):
                return value
            if value in (None, ""):
                return {}
            try:
                parsed = json.loads(str(value))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

        def list_field(name: str) -> list[str]:
            value = row.get(name)
            if isinstance(value, list):
                return [str(item) for item in value]
            if value in (None, ""):
                return []
            try:
                parsed = json.loads(str(value))
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
            return [item.strip() for item in str(value).split(",") if item.strip()]

        return cls(
            id=str(row.get("id") or ""),
            canonical_accent=str(row.get("canonical_accent") or row.get("accent") or ""),
            reference_audio_path=str(row.get("reference_audio_path") or row.get("audio_path") or ""),
            transcript=_none_if_empty(row.get("transcript") or row.get("text")),
            gender=str(row.get("gender") or "unknown"),
            age_group=str(row.get("age_group") or "unknown"),
            speaker_id=_none_if_empty(row.get("speaker_id")),
            duration=_float_or_none(row.get("duration") or row.get("duration_sec")),
            sample_rate=_int_or_none(row.get("sample_rate")),
            language=_none_if_empty(row.get("language")),
            locale=_none_if_empty(row.get("locale")),
            region=_none_if_empty(row.get("region")),
            family=_none_if_empty(row.get("family")),
            label_type=str(row.get("label_type") or "dialect"),
            aliases=list_field("aliases_json") or list_field("aliases"),
            quality=json_field("quality_json") or json_field("quality"),
            provenance=json_field("provenance_json") or json_field("provenance"),
            metadata=json_field("metadata_json") or json_field("metadata"),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "canonical_accent": self.canonical_accent,
            "label_type": self.label_type,
            "language": self.language,
            "locale": self.locale,
            "region": self.region,
            "family": self.family,
            "aliases_json": json.dumps(self.aliases, ensure_ascii=False),
            "reference_audio_path": self.reference_audio_path,
            "transcript": self.transcript,
            "gender": self.gender,
            "age_group": self.age_group,
            "speaker_id": self.speaker_id,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "quality_json": json.dumps(self.quality, ensure_ascii=False),
            "provenance_json": json.dumps(self.provenance, ensure_ascii=False),
            "metadata_json": json.dumps(self.metadata, ensure_ascii=False),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def audio_exists(self) -> bool:
        return Path(self.reference_audio_path).expanduser().is_file()


@dataclass
class AccentMatch:
    entry: AccentLibraryEntry
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "reasons": self.reasons,
            "entry": self.entry.to_dict(),
        }


def _none_if_empty(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    return text if text else None


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
