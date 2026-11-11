"""Persistent accent/dialect library storage and retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .labels import accents_compatible, canonical_label, is_wu_subdialect, requires_fine_label
from .schemas import AccentLibraryEntry, AccentMatch


def default_library_dir() -> Path:
    override = os.getenv("X_EDITOR_ACCENT_LIBRARY_DATA_DIR")
    if override:
        return Path(override).expanduser()
    data_home = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))).expanduser()
    return data_home / "x-edit" / "accent_library"


def default_library_path() -> Path:
    override = os.getenv("X_EDITOR_ACCENT_LIBRARY_PATH")
    if override:
        return Path(override).expanduser()
    return default_library_dir() / "accent_library.jsonl"


class AccentDialectLibrary:
    """Local JSONL or optional Parquet reference library for accent and dialect matching."""

    def __init__(
        self,
        entries: Iterable[AccentLibraryEntry] | None = None,
        *,
        path: str | Path | None = None,
    ) -> None:
        self.path = (Path(path) if path else default_library_path()).expanduser().resolve()
        self.entries = list(entries or [])

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AccentDialectLibrary":
        library_path = (Path(path) if path else default_library_path()).expanduser().resolve()
        _check_format(library_path)
        if not library_path.exists():
            return cls(path=library_path)
        if library_path.suffix.lower() == ".jsonl":
            entries = []
            for number, line in enumerate(library_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                    entries.append(AccentLibraryEntry.from_row(row))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{library_path}:{number}: {exc}") from exc
            return cls(_resolve_reference_audio_paths(entries, library_path), path=library_path)
        import pandas as pd

        try:
            df = pd.read_parquet(library_path)
        except ImportError as exc:
            raise RuntimeError("Parquet libraries require pyarrow or fastparquet; prefer the JSONL builder.") from exc
        entries = [AccentLibraryEntry.from_row(row) for row in df.to_dict("records")]
        return cls(_resolve_reference_audio_paths(entries, library_path), path=library_path)

    def save(self, path: str | Path | None = None) -> Path:
        output_path = (Path(path) if path else self.path).expanduser().resolve()
        _check_format(output_path)
        rows = [_portable_row(entry, output_path, self.path) for entry in self.entries]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".jsonl":
            with output_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return output_path
        import pandas as pd

        try:
            pd.DataFrame(rows).to_parquet(output_path, index=False)
        except ImportError as exc:
            raise RuntimeError("Parquet libraries require pyarrow or fastparquet; prefer JSONL.") from exc
        return output_path

    def add(self, entry: AccentLibraryEntry, *, replace: bool = True) -> None:
        if replace:
            self.entries = [item for item in self.entries if item.id != entry.id]
        elif any(item.id == entry.id for item in self.entries):
            raise ValueError(f"Duplicate accent library entry id: {entry.id}")
        self.entries.append(entry)

    def extend(self, entries: Iterable[AccentLibraryEntry], *, replace: bool = True) -> None:
        for entry in entries:
            self.add(entry, replace=replace)

    def labels(self) -> list[str]:
        seen: list[str] = []
        for entry in self.entries:
            if entry.canonical_accent not in seen:
                seen.append(entry.canonical_accent)
        return seen

    def by_label(self, label: str, *, allow_compatible: bool = False) -> list[AccentLibraryEntry]:
        canonical = canonical_label(label) or label
        if requires_fine_label(canonical) and not allow_compatible:
            return []
        return [
            entry
            for entry in self.entries
            if entry.canonical_accent == canonical
            or (allow_compatible and accents_compatible(entry.canonical_accent, canonical))
        ]

    def find(
        self,
        *,
        accent: str,
        gender: str | None = None,
        age_group: str | None = None,
        language: str | None = None,
        require_audio_exists: bool = True,
        top_k: int = 5,
    ) -> list[AccentMatch]:
        canonical = canonical_label(accent) or accent
        if requires_fine_label(canonical):
            raise ValueError(
                "Wu Chinese is a family fallback, not a retrievable edit target. "
                "Use a fine label such as Shanghai Dialect, Suzhou Dialect, or Ningbo Dialect."
            )

        matches: list[AccentMatch] = []
        for entry in self.entries:
            if require_audio_exists and not entry.audio_exists():
                continue
            score, reasons = self._score(entry, canonical, gender=gender, age_group=age_group, language=language)
            if score > 0:
                matches.append(AccentMatch(entry=entry, score=score, reasons=reasons))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:top_k]

    def summary(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.canonical_accent] = counts.get(entry.canonical_accent, 0) + 1
        return {
            "path": str(self.path),
            "entry_count": len(self.entries),
            "labels": counts,
        }

    def _score(
        self,
        entry: AccentLibraryEntry,
        target: str,
        *,
        gender: str | None,
        age_group: str | None,
        language: str | None,
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        if entry.canonical_accent == target:
            score += 100.0
            reasons.append("exact_accent")
        elif accents_compatible(entry.canonical_accent, target):
            if is_wu_subdialect(target) and entry.canonical_accent != target:
                return 0.0, []
            score += 35.0
            reasons.append("compatible_accent")
        else:
            return 0.0, []

        if gender and entry.gender == gender:
            score += 8.0
            reasons.append("gender")
        if age_group and entry.age_group == age_group:
            score += 5.0
            reasons.append("age_group")
        if language and entry.language == language:
            score += 4.0
            reasons.append("language")
        if entry.quality.get("usable_for_vc_reference", True):
            score += 2.0
            reasons.append("vc_usable")
        if entry.quality.get("usable_for_zeroshot_tts_reference", True):
            score += 2.0
            reasons.append("tts_usable")
        return score, reasons



def _check_format(path: Path) -> None:
    if path.suffix.lower() not in {".jsonl", ".parquet"}:
        raise ValueError(f"Accent library must be .jsonl or .parquet: {path}")


def _resolve_reference_audio_paths(
    entries: Iterable[AccentLibraryEntry], library_path: Path,
) -> list[AccentLibraryEntry]:
    """Resolve relative paths only against their manifest, never by basename."""
    result = list(entries)
    for entry in result:
        audio = Path(entry.reference_audio_path).expanduser()
        if not audio.is_absolute():
            audio = library_path.parent / audio
        entry.reference_audio_path = str(audio.resolve())
    return result


def _portable_row(entry: AccentLibraryEntry, output: Path, source: Path) -> dict:
    row = entry.to_row()
    audio = Path(entry.reference_audio_path).expanduser()
    if not audio.is_absolute():
        audio = source.parent / audio
    audio = audio.resolve()
    # Keep a self-contained library relocatable. Explicit external paths stay exact.
    row["reference_audio_path"] = (
        audio.relative_to(output.parent).as_posix()
        if audio.is_relative_to(output.parent) else str(audio)
    )
    return row
