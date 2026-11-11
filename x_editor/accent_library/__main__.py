"""Offline construction of a portable, user-supplied accent reference library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

from .labels import accent_definition, list_canonical_labels, requires_fine_label
from .schemas import AccentLibraryEntry
from .store import AccentDialectLibrary


def validate_library(path: Path) -> tuple[AccentDialectLibrary, list[str]]:
    """Check import eligibility, not accent identity or perceptual quality."""
    if not path.expanduser().is_file():
        raise ValueError(f"Manifest does not exist: {path}")
    library = AccentDialectLibrary.load(path)
    if not library.entries:
        raise ValueError("Manifest is empty; add reviewed reference recordings first")
    seen: set[str] = set()
    warnings: list[str] = []
    for entry in library.entries:
        if entry.id in seen:
            raise ValueError(f"Duplicate reference id: {entry.id}")
        seen.add(entry.id)
        if not accent_definition(entry.canonical_accent) or requires_fine_label(entry.canonical_accent):
            raise ValueError(f"{entry.id}: choose a specific supported label from the labels command")
        # The current retrieval route can use either VC or zero-shot TTS.
        if not entry.transcript or not entry.transcript.strip():
            raise ValueError(f"{entry.id}: a verbatim transcript is required for the TTS reference route")
        for flag in ("usable_for_vc_reference", "usable_for_zeroshot_tts_reference"):
            if entry.quality.get(flag, True) is not True:
                raise ValueError(f"{entry.id}: {flag} is not true; exclude unapproved references")
        for field in ("source_uri", "license"):
            if not str(entry.provenance.get(field) or "").strip():
                raise ValueError(f"{entry.id}: provenance.{field} must record the source and usage terms")
        audio_path = Path(entry.reference_audio_path)
        if not audio_path.is_file():
            raise ValueError(f"{entry.id}: missing reference audio: {audio_path}")
        try:
            with sf.SoundFile(audio_path) as audio:
                if audio.channels != 1 or audio.frames <= 0 or audio.samplerate <= 0:
                    raise ValueError(f"{entry.id}: provide a non-empty mono recording")
                duration, sample_rate = audio.frames / audio.samplerate, audio.samplerate
                peak = 0.0
                for block in audio.blocks(blocksize=65536, dtype="float32"):
                    if not np.isfinite(block).all():
                        raise ValueError(f"{entry.id}: audio contains non-finite samples")
                    peak = max(peak, float(np.abs(block).max()))
        except (sf.LibsndfileError, OSError) as exc:
            raise ValueError(f"{entry.id}: audio cannot be decoded: {exc}") from exc
        if peak == 0 or peak > 1.0:
            raise ValueError(f"{entry.id}: silent or out-of-range waveform; inspect it before importing")
        if entry.duration is not None and (
            not np.isfinite(entry.duration) or abs(entry.duration - duration) > max(.001, 1 / sample_rate)
        ):
            raise ValueError(f"{entry.id}: duration metadata does not match the file")
        if entry.sample_rate is not None and entry.sample_rate != sample_rate:
            raise ValueError(f"{entry.id}: sample_rate metadata does not match the file")
        entry.duration, entry.sample_rate = duration, sample_rate
        if not 5 <= duration <= 10:
            warnings.append(f"{entry.id}: {duration:.2f}s is outside the recommended 5–10s range")
        if peak >= .999:
            warnings.append(f"{entry.id}: near full-scale amplitude; inspect for clipping")
    return library, warnings


def build_library(manifest: Path, output: Path) -> dict:
    """Copy into a new directory, write relative paths, publish only on success."""
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output must be a new directory (existing libraries are never overwritten): {output}")
    library, warnings = validate_library(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".accent-build-", dir=output.parent))
    try:
        (staging / "audio").mkdir()
        rows = []
        for index, entry in enumerate(library.entries):
            # IDs are metadata, never user-controlled filesystem path components.
            name = f"{index:05d}-{hashlib.sha256(entry.id.encode()).hexdigest()[:16]}.wav"
            destination = staging / "audio" / name
            samples, rate = sf.read(entry.reference_audio_path, dtype="float32")
            sf.write(destination, samples, rate, subtype="PCM_16")
            entry.metadata["library_audio_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            entry.reference_audio_path = f"audio/{name}"
            rows.append(entry.to_dict())
        (staging / "accent_library.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        # Detect conversion problems before the library becomes visible to the runtime.
        validate_library(staging / "accent_library.jsonl")
        if output.exists():
            raise ValueError(f"Output appeared during construction: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"manifest": str(output / "accent_library.jsonl"), "entries": len(rows),
            "warnings": warnings, "accent_and_transcript_verified": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("labels", help="List supported canonical labels")
    build = commands.add_parser("build", help="Build from reviewed JSONL/Parquet metadata and local audio")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Check reference files and the runtime metadata contract")
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "labels":
            result = {"labels": [label for label in list_canonical_labels() if not requires_fine_label(label)]}
        elif args.command == "build":
            result = build_library(args.manifest, args.output)
        else:
            library, warnings = validate_library(args.manifest)
            result = {**library.summary(), "warnings": warnings,
                      "accent_and_transcript_verified": False}
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"accent library: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
