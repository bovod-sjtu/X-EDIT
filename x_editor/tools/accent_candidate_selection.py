"""Select high-quality accent reference candidates from configured speech data.

The selector keeps the expensive checks late in the pipeline:
annotation/duration -> Gemini accent consistency -> ASR WER -> UTMOS.
It supports FCaps JSONL, captioned InstructTTS/Alalm parquet shards, parquet
lists, and a few directory datasets with explicit speaker/accent metadata.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import requests

from x_editor.accent_library import default_library_dir
from x_editor.agents.base import GeminiClient
from x_editor.perception.speech import SpeechPerceptionClient


DEFAULT_LIBRARY_DATA_DIR = default_library_dir()
DEFAULT_OUTPUT = DEFAULT_LIBRARY_DATA_DIR / "candidates" / "accent_candidates.parquet"
DEFAULT_CACHE_DIR = DEFAULT_LIBRARY_DATA_DIR / "cache_candidate_selection"
DEFAULT_AUDIO_DIR = DEFAULT_LIBRARY_DATA_DIR / "candidates" / "audio"
DEFAULT_ASR_URL = "http://127.0.0.1:8010"
DEFAULT_UTMOS_URL = ""  # Must be supplied; no UTMOS service is assumed.

OUTPUT_COLUMNS = [
    "id",
    "dataset_name",
    "audio_path",
    "duration",
    "text",
    "speaker_id",
    "gender",
    "age_group",
    "annotated_accent",
    "canonical_accent",
    "source_type",
    "source_path",
    "source_row",
    "segment_index",
    "instruction",
    "sample_rate",
    "gemini_detected_accent",
    "gemini_canonical_accent",
    "gemini_confidence",
    "gemini_matches_annotation",
    "gemini_reason",
    "asr_text",
    "asr_error",
    "wer",
    "utmos",
    "utmos_error",
    "selected",
    "reject_reason",
    "metadata_json",
]


ACCENT_ALIASES: dict[str, tuple[str, ...]] = {
    "General American English": (
        "american",
        "american accent",
        "american english",
        "general american",
        "general american english",
        "us english",
        "u.s. english",
        "usa english",
        "united states accent",
    ),
    "Southern American English": (
        "southern american",
        "southern american accent",
        "southern american english",
        "southern accent",
    ),
    "British English": (
        "british",
        "british accent",
        "british english",
        "english accent",
        "vctk english",
        "uk accent",
        "received pronunciation",
        "received pronunciation accent",
        "rp accent",
    ),
    "Australian English": (
        "australian",
        "australian accent",
        "australian english",
    ),
    "Indian English": (
        "indian",
        "indian accent",
        "indian english",
    ),
    "French-accented English": (
        "french accented english",
        "french-accented english",
        "french accent",
    ),
    "Chinese-accented English": (
        "chinese accented english",
        "chinese-accented english",
        "chinese accent",
        "mandarin accented english",
        "mandarin-accented english",
    ),
    "German-accented English": (
        "german accented english",
        "german-accented english",
        "german accent",
    ),
    "Spanish-accented English": (
        "spanish accented english",
        "spanish-accented english",
        "spanish accent",
    ),
    "Russian-accented English": (
        "russian accented english",
        "russian-accented english",
        "russian accent",
    ),
    "Ukrainian-accented English": (
        "ukrainian accented english",
        "ukrainian-accented english",
        "ukrainian accent",
    ),
    "Dari-accented English": (
        "dari accented english",
        "dari-accented english",
        "dari accent",
    ),
    "Japanese-accented English": (
        "japanese accented english",
        "japanese-accented english",
        "japanese accent",
    ),
    "Korean-accented English": (
        "korean accented english",
        "korean-accented english",
        "korean accent",
    ),
    "Thai-accented English": (
        "thai accented english",
        "thai-accented english",
        "thai accent",
    ),
    "Scottish English": (
        "scottish",
        "scottish accent",
        "scottish english",
    ),
    "Irish English": (
        "irish",
        "irish accent",
        "irish english",
    ),
    "Northern Irish English": (
        "northernirish",
        "northern irish",
        "northern irish accent",
        "northern irish english",
    ),
    "Welsh English": (
        "welsh",
        "welsh accent",
        "welsh english",
    ),
    "Canadian English": (
        "canadian",
        "canadian accent",
        "canadian english",
    ),
    "New Zealand English": (
        "newzealand",
        "new zealand",
        "new zealand accent",
        "new zealand english",
    ),
    "South African English": (
        "southafrican",
        "south african",
        "south african accent",
        "south african english",
    ),
    "Standard Mandarin": (
        "standard mandarin",
        "mandarin chinese",
        "putonghua",
        "standard chinese",
        "普通话",
        "标准普通话",
    ),
    "Cantonese": (
        "cantonese",
        "cantonese chinese",
        "yue chinese",
        "yue dialect",
        "粤语",
        "廣東話",
        "广东话",
    ),
    "Dongbei Dialect": (
        "dongbei",
        "dongbei dialect",
        "northeastern mandarin",
        "northeastern chinese",
        "东北话",
        "东北方言",
        "东北官话",
    ),
    "Wu Chinese": (
        "wu chinese",
        "wu dialect",
        "wuyu",
        "wu yu",
        "wu language",
        "吴语",
        "吴方言",
    ),
    "Shanghai Dialect": (
        "shanghai dialect",
        "shanghainese",
        "shanghai wu",
        "上海话",
        "上海方言",
    ),
    "Suzhou Dialect": (
        "suzhou dialect",
        "suzhou accent",
        "suzhou wu",
        "suzhou wuyu",
        "苏州话",
        "苏州方言",
    ),
    "Taiwanese Mandarin": (
        "taiwanese mandarin",
        "taiwan accent",
        "taiwanese accent",
    ),
    "Ningbo Dialect": (
        "ningbo dialect",
        "ningbo accent",
        "ningbo wu",
        "ningbo wuyu",
        "宁波话",
        "宁波方言",
    ),
    "Minnan Dialect": (
        "minnan",
        "minnan dialect",
        "min nan",
        "min-nan-yu",
        "southern min",
        "hokkien",
        "闽南语",
        "闽南话",
        "闽南方言",
    ),
    "Sichuan Dialect": (
        "sichuan",
        "sichuan dialect",
        "sichuanese",
        "sichuan hua",
        "southwestern mandarin",
        "四川话",
        "四川方言",
        "西南官话",
    ),
    "Beijing Dialect": (
        "beijing dialect",
        "beijing accent",
        "北京话",
        "北京方言",
    ),
    "Beijing Guanhua": (
        "beijing guanhua",
        "beijing mandarin",
        "北京官话",
        "北京官話",
    ),
    "Kansai Japanese": (
        "kansai japanese",
        "kansai dialect",
        "kansai accent",
    ),
    "Native Korean": (
        "native korean",
        "korean",
        "korean language",
    ),
}

GENERIC_ACCENT_VALUES = {
    "",
    "english",
    "native english",
    "native",
    "unknown",
    "none",
    "other",
    "n/a",
}

COMPATIBLE_ACCENT_GROUPS = [
    {"General American English", "Southern American English"},
    {"British English", "Scottish English", "Irish English", "Northern Irish English", "Welsh English"},
    {"Australian English"},
    {"Canadian English", "General American English"},
    {"New Zealand English", "Australian English"},
    {"South African English", "British English"},
    {"Indian English"},
    {"French-accented English"},
    {"Chinese-accented English", "Standard Mandarin"},
    {"German-accented English"},
    {"Spanish-accented English"},
    {"Russian-accented English"},
    {"Ukrainian-accented English"},
    {"Dari-accented English"},
    {"Japanese-accented English"},
    {"Korean-accented English", "Native Korean"},
    {"Thai-accented English"},
    {"Cantonese"},
    {"Dongbei Dialect", "Standard Mandarin"},
    {"Wu Chinese", "Shanghai Dialect", "Suzhou Dialect", "Ningbo Dialect"},
    {"Shanghai Dialect", "Standard Mandarin", "Chinese-accented English"},
    {"Suzhou Dialect", "Standard Mandarin", "Chinese-accented English"},
    {"Taiwanese Mandarin", "Standard Mandarin", "Chinese-accented English"},
    {"Ningbo Dialect", "Standard Mandarin", "Chinese-accented English"},
    {"Minnan Dialect"},
    {"Sichuan Dialect", "Standard Mandarin"},
    {"Beijing Dialect", "Standard Mandarin", "Chinese-accented English"},
    {"Beijing Guanhua", "Beijing Dialect", "Standard Mandarin"},
    {"Kansai Japanese"},
    {"Native Korean"},
]


def _label_key(value: str) -> str:
    value = value.lower().replace("u.s.", "us").replace("u.k.", "uk")
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains_label_key(haystack: str, needle: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", needle):
        return needle in haystack
    return f" {needle} " in f" {haystack} "


_ALIAS_TO_CANONICAL = {
    _label_key(alias): canonical
    for canonical, aliases in ACCENT_ALIASES.items()
    for alias in (canonical, *aliases)
}
_SORTED_ALIAS_KEYS = sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u4e00-\u9fff]", re.IGNORECASE)
WU_FAMILY_ACCENT = "Wu Chinese"
WU_FINE_ACCENTS = {
    "Shanghai Dialect",
    "Suzhou Dialect",
    "Ningbo Dialect",
}


@dataclass
class CandidateRecord:
    """A normalized candidate segment before external quality checks."""

    id: str
    text: str
    duration: float
    annotated_accent: str
    canonical_accent: str
    source_type: str
    source_path: str
    source_row: int
    dataset_name: str | None = None
    segment_index: int = 0
    gender: str | None = None
    speaker_id: str | None = None
    instruction: str | None = None
    source_audio_path: str | None = None
    embedded_audio_b64: str | None = field(default=None, repr=False)
    embedded_audio_format: str = "wav"
    sample_rate: int | None = None
    age_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def materialize_audio(self, audio_dir: Path) -> Path:
        """Return a local audio path, extracting embedded parquet audio if needed."""

        if self.source_audio_path:
            path = Path(self.source_audio_path).expanduser()
            if path.exists():
                return path.resolve()

        if not self.embedded_audio_b64:
            raise FileNotFoundError(f"No available audio payload for candidate {self.id}")

        audio_dir.mkdir(parents=True, exist_ok=True)
        suffix = self.embedded_audio_format.lstrip(".") or "wav"
        path = audio_dir / f"{safe_filename(self.id)}.{suffix}"
        if not path.exists():
            path.write_bytes(base64.b64decode(self.embedded_audio_b64))
        return path.resolve()

    def output_row(self, audio_path: str | None, checks: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": self.id,
            "dataset_name": self.dataset_name,
            "audio_path": audio_path,
            "duration": self.duration,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "gender": self.gender,
            "age_group": self.age_group,
            "annotated_accent": self.annotated_accent,
            "canonical_accent": self.canonical_accent,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "source_row": self.source_row,
            "segment_index": self.segment_index,
            "instruction": self.instruction,
            "sample_rate": self.sample_rate,
            "gemini_detected_accent": None,
            "gemini_canonical_accent": None,
            "gemini_confidence": None,
            "gemini_matches_annotation": None,
            "gemini_reason": None,
            "asr_text": None,
            "asr_error": None,
            "wer": None,
            "utmos": None,
            "utmos_error": None,
            "selected": False,
            "reject_reason": None,
            "metadata_json": json.dumps(self.metadata, ensure_ascii=False),
        }
        row.update(checks)
        return row


class JsonCache:
    """Small file cache for expensive Gemini/ASR/UTMOS calls."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, stage: str, key: str) -> dict[str, Any] | None:
        path = self._path(stage, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, stage: str, key: str, value: dict[str, Any]) -> None:
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _path(self, stage: str, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.root / stage / f"{digest}.json"


@dataclass
class DatasetSourceConfig:
    """One data source in an accent-selection config file."""

    name: str
    path: Path
    enabled: bool = True
    dataset_root: Path | None = None
    target_accents: list[str] = field(default_factory=list)
    language: str | None = None
    max_scan: int | None = None
    max_external_evals: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionConfig:
    input_path: Path | None = None
    sources: list[DatasetSourceConfig] = field(default_factory=list)
    output_path: Path = DEFAULT_OUTPUT
    dataset_root: Path | None = None
    target_accents: list[str] = field(default_factory=list)
    min_duration: float = 5.0
    max_duration: float = 10.0
    target_count: int = 50
    target_per_accent: int | None = None
    target_count_by_accent: dict[str, int] = field(default_factory=dict)
    max_scan: int = 5000
    max_external_evals: int = 250
    max_wer: float = 0.03
    min_utmos: float = 3.5
    min_gemini_confidence: float = 0.65
    cache_dir: Path = DEFAULT_CACHE_DIR
    audio_dir: Path = DEFAULT_AUDIO_DIR
    asr_url: str = DEFAULT_ASR_URL
    utmos_url: str = DEFAULT_UTMOS_URL
    language: str | None = "English"
    skip_gemini: bool = False
    skip_asr: bool = False
    skip_utmos: bool = False
    reject_report: Path | None = None
    gemini_timeout: float = 60.0
    service_timeout: float = 600.0
    parquet_batch_size: int = 32


def select_accent_candidates(config: SelectionConfig) -> dict[str, Any]:
    """Run the selector and write selected rows to parquet."""

    sources = _active_sources(config)
    if not sources:
        raise ValueError("Provide --input or a config with at least one enabled source")
    for source in sources:
        if not source.path.exists():
            raise ValueError(f"Dataset source does not exist: {source.path}")
    if not config.skip_gemini and not os.getenv("X_EDITOR_GEMINI_BASE_URL", "").strip():
        raise ValueError("Set X_EDITOR_GEMINI_BASE_URL for candidate mining, or use the offline library builder")
    if not config.skip_utmos and not config.utmos_url:
        raise ValueError("Set --utmos-url or X_EDITOR_UTMOS_URL for the UTMOS /predict service")
    try:
        import pyarrow  # Optional candidate-mining / Parquet dependency.
    except ImportError as exc:
        raise RuntimeError("Candidate mining requires pyarrow; the JSONL library builder does not") from exc

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.audio_dir.mkdir(parents=True, exist_ok=True)
    cache = JsonCache(config.cache_dir)

    counters: dict[str, int] = {
        "scanned": 0,
        "annotation_pass": 0,
        "duration_pass": 0,
        "target_accent_pass": 0,
        "audio_available": 0,
        "external_evaluated": 0,
        "gemini_pass": 0,
        "wer_pass": 0,
        "utmos_pass": 0,
        "selected": 0,
    }
    selected_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []

    gemini_client = None
    asr_client = None
    if not config.skip_gemini:
        gemini_client = GeminiClient(max_tokens=2048, timeout=config.gemini_timeout)
    if not config.skip_asr:
        asr_client = SpeechPerceptionClient(
            asr_url=config.asr_url,
            timeout=config.service_timeout,
            cache_dir=config.cache_dir / "speech_perception",
        )

    per_source: dict[str, dict[str, int]] = {}
    selected_by_accent: dict[str, int] = {}
    global_accent_quotas = _accent_quota_limits(config, _all_quota_targets(config))
    stop_all = False
    for source in _active_sources(config):
        source_counters = {
            "scanned": 0,
            "annotation_pass": 0,
            "duration_pass": 0,
            "target_accent_pass": 0,
            "audio_available": 0,
            "external_evaluated": 0,
            "gemini_pass": 0,
            "wer_pass": 0,
            "utmos_pass": 0,
            "selected": 0,
        }
        per_source[source.name] = source_counters
        source_targets = [
            normalize_accent_label(item) or item.strip()
            for item in (source.target_accents or config.target_accents)
            if item.strip()
        ]
        accent_quotas = _accent_quota_limits(config, source_targets)
        source_max_scan = source.max_scan if source.max_scan is not None else config.max_scan
        source_max_external = (
            source.max_external_evals
            if source.max_external_evals is not None
            else config.max_external_evals
        )
        source_language = source.language if source.language is not None else config.language
        for candidate in iter_candidates(
            source.path,
            dataset_root=source.dataset_root or config.dataset_root,
            batch_size=config.parquet_batch_size,
            dataset_name=source.name,
        ):
            counters["scanned"] += 1
            source_counters["scanned"] += 1
            if source.metadata:
                candidate.metadata.setdefault("dataset_source", source.metadata)
            if counters["scanned"] > config.max_scan or source_counters["scanned"] > source_max_scan:
                break

            if not candidate.canonical_accent:
                _append_reject(reject_rows, candidate, "missing_or_ambiguous_accent_annotation")
                continue
            counters["annotation_pass"] += 1
            source_counters["annotation_pass"] += 1
            if candidate.duration < config.min_duration or candidate.duration > config.max_duration:
                _append_reject(reject_rows, candidate, "duration_out_of_range")
                continue
            counters["duration_pass"] += 1
            source_counters["duration_pass"] += 1
            if source_targets and not any(
                accents_compatible(candidate.canonical_accent, target) for target in source_targets
            ):
                _append_reject(reject_rows, candidate, "target_accent_mismatch")
                continue
            counters["target_accent_pass"] += 1
            source_counters["target_accent_pass"] += 1
            quota_label = _quota_label_for(candidate.canonical_accent, accent_quotas)
            needs_fine_resolution = _needs_wu_fine_resolution(
                candidate.canonical_accent,
                source_targets,
                accent_quotas,
            )
            if accent_quotas and not quota_label and not needs_fine_resolution:
                _append_reject(reject_rows, candidate, "target_accent_mismatch")
                continue
            if quota_label and selected_by_accent.get(quota_label, 0) >= accent_quotas[quota_label]:
                _append_reject(reject_rows, candidate, "accent_quota_filled")
                continue

            if (
                counters["external_evaluated"] >= config.max_external_evals
            ):
                stop_all = True
                break
            if source_counters["external_evaluated"] >= source_max_external:
                break

            checks: dict[str, Any] = {"selected": False, "reject_reason": None}
            try:
                audio_path = candidate.materialize_audio(config.audio_dir / safe_filename(source.name))
            except Exception as exc:
                _append_reject(reject_rows, candidate, f"audio_unavailable: {exc}")
                continue
            _fill_audio_metadata(candidate, audio_path)
            counters["audio_available"] += 1
            counters["external_evaluated"] += 1
            source_counters["audio_available"] += 1
            source_counters["external_evaluated"] += 1

            gemini_ok, gemini_checks = _check_gemini(
                candidate,
                audio_path,
                client=gemini_client,
                cache=cache,
                min_confidence=config.min_gemini_confidence,
                skip=config.skip_gemini,
                target_accents=source_targets or list(accent_quotas),
            )
            checks.update(gemini_checks)
            if not gemini_ok:
                checks["reject_reason"] = "gemini_accent_mismatch"
                _append_reject(reject_rows, candidate, checks["reject_reason"], checks, str(audio_path))
                continue
            _maybe_refine_candidate_accent_from_gemini(
                candidate,
                checks,
                source_targets=source_targets,
                accent_quotas=accent_quotas,
            )
            quota_label = _quota_label_for(candidate.canonical_accent, accent_quotas)
            if _needs_wu_fine_resolution(candidate.canonical_accent, source_targets, accent_quotas):
                checks["reject_reason"] = "gemini_fine_accent_unresolved"
                _append_reject(reject_rows, candidate, checks["reject_reason"], checks, str(audio_path))
                continue
            if accent_quotas and not quota_label:
                checks["reject_reason"] = "target_accent_mismatch_after_gemini"
                _append_reject(reject_rows, candidate, checks["reject_reason"], checks, str(audio_path))
                continue
            if quota_label and selected_by_accent.get(quota_label, 0) >= accent_quotas[quota_label]:
                checks["reject_reason"] = "accent_quota_filled"
                _append_reject(reject_rows, candidate, checks["reject_reason"], checks, str(audio_path))
                continue
            counters["gemini_pass"] += 1
            source_counters["gemini_pass"] += 1

            wer_ok, wer_checks = _check_wer(
                candidate,
                audio_path,
                asr_client=asr_client,
                cache=cache,
                max_wer=config.max_wer,
                language=source_language,
                skip=config.skip_asr,
            )
            checks.update(wer_checks)
            if not wer_ok:
                checks["reject_reason"] = "asr_service_error" if checks.get("asr_error") else "wer_above_threshold"
                _append_reject(reject_rows, candidate, checks["reject_reason"], checks, str(audio_path))
                continue
            counters["wer_pass"] += 1
            source_counters["wer_pass"] += 1

            utmos_ok, utmos_checks = _check_utmos(
                audio_path,
                cache=cache,
                utmos_url=config.utmos_url,
                min_utmos=config.min_utmos,
                timeout=config.service_timeout,
                skip=config.skip_utmos,
            )
            checks.update(utmos_checks)
            if not utmos_ok:
                checks["reject_reason"] = (
                    "utmos_service_error" if checks.get("utmos_error") else "utmos_below_threshold"
                )
                _append_reject(reject_rows, candidate, checks["reject_reason"], checks, str(audio_path))
                continue
            counters["utmos_pass"] += 1
            source_counters["utmos_pass"] += 1

            checks["selected"] = True
            checks["reject_reason"] = None
            selected_rows.append(candidate.output_row(str(audio_path), checks))
            counters["selected"] += 1
            source_counters["selected"] += 1
            if quota_label:
                selected_by_accent[quota_label] = selected_by_accent.get(quota_label, 0) + 1
            if counters["selected"] >= config.target_count:
                stop_all = True
                break
            if accent_quotas and _accent_quotas_filled(selected_by_accent, accent_quotas):
                break
            if global_accent_quotas and _accent_quotas_filled(selected_by_accent, global_accent_quotas):
                stop_all = True
                break
        if stop_all or counters["scanned"] > config.max_scan:
            break

    selected_df = pd.DataFrame(selected_rows, columns=OUTPUT_COLUMNS)
    selected_df.to_parquet(config.output_path, index=False)

    if config.reject_report:
        config.reject_report.parent.mkdir(parents=True, exist_ok=True)
        with config.reject_report.open("w", encoding="utf-8") as f:
            for row in reject_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "output_path": str(config.output_path),
        "reject_report": str(config.reject_report) if config.reject_report else None,
        "selected_count": len(selected_rows),
        "counters": counters,
        "per_source": per_source,
        "selected_by_accent": selected_by_accent,
        "accent_quotas": global_accent_quotas,
    }


def iter_candidates(
    input_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    batch_size: int = 32,
    dataset_name: str | None = None,
) -> Iterator[CandidateRecord]:
    """Yield normalized candidates from JSONL, parquet, or parquet-list input."""

    path = Path(input_path)
    if path.is_dir():
        if (path / "speaker-info.txt").exists() and (path / "wav48_silence_trimmed").exists():
            yield from _iter_vctk_candidates(path, dataset_name=dataset_name)
            return
        if (path / "speaker_statistics.json").exists() and (path / "transcripts.json").exists():
            yield from _iter_ears_candidates(path, dataset_name=dataset_name)
            return
        raise ValueError(f"Unsupported directory dataset for accent selection: {path}")
    if path.suffix.lower() == ".list":
        for shard in _read_list_file(path):
            if not shard.exists():
                continue
            yield from _iter_parquet_candidates(shard, batch_size=batch_size, dataset_name=dataset_name)
        return
    if path.suffix.lower() == ".parquet":
        yield from _iter_parquet_candidates(path, batch_size=batch_size, dataset_name=dataset_name)
        return
    if path.suffix.lower() == ".jsonl":
        root = Path(dataset_root) if dataset_root else path.parent
        yield from _iter_fcaps_jsonl_candidates(path, dataset_root=root, dataset_name=dataset_name)
        return
    raise ValueError(f"Unsupported input type: {path}")


def _active_sources(config: SelectionConfig) -> list[DatasetSourceConfig]:
    if config.sources:
        return [source for source in config.sources if source.enabled]
    if config.input_path is None:
        raise ValueError("SelectionConfig requires input_path or at least one enabled source")
    return [
        DatasetSourceConfig(
            name=_default_source_name(config.input_path),
            path=config.input_path,
            dataset_root=config.dataset_root,
            target_accents=config.target_accents,
            language=config.language,
            max_scan=config.max_scan,
            max_external_evals=config.max_external_evals,
        )
    ]


def _all_quota_targets(config: SelectionConfig) -> list[str]:
    targets = list(config.target_accents)
    for source in _active_sources(config):
        targets.extend(source.target_accents)
    deduped: list[str] = []
    for target in targets:
        canonical = normalize_accent_label(target) or str(target).strip()
        if canonical and canonical not in deduped:
            deduped.append(canonical)
    return deduped


def _iter_fcaps_jsonl_candidates(
    path: Path,
    *,
    dataset_root: Path,
    dataset_name: str | None = None,
) -> Iterator[CandidateRecord]:
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate = candidate_from_fcaps_row(
                row,
                source_path=str(path),
                row_index=row_index,
                dataset_root=dataset_root,
                dataset_name=dataset_name,
            )
            if candidate:
                yield candidate


def candidate_from_fcaps_row(
    row: dict[str, Any],
    *,
    source_path: str,
    row_index: int,
    dataset_root: str | Path,
    dataset_name: str | None = None,
) -> CandidateRecord | None:
    supervisions = row.get("supervisions") or []
    supervision = supervisions[0] if supervisions else {}
    custom = supervision.get("custom") or {}
    text = str(supervision.get("text") or "").strip()
    duration = _float_or_none(row.get("duration"))
    if not text or duration is None:
        return None

    captions = _flatten_texts(
        custom.get("finegrained_captions"),
        custom.get("global_captions"),
        custom.get("intrinsic_tags"),
    )
    annotated_raw, canonical, reason = infer_annotated_accent(
        custom.get("accent"),
        captions,
    )
    if not canonical:
        return CandidateRecord(
            id=str(row.get("id") or f"row_{row_index}"),
            text=text,
            duration=duration,
            annotated_accent=str(custom.get("accent") or reason or ""),
            canonical_accent="",
            source_type="fcaps_jsonl",
            source_path=source_path,
            source_row=row_index,
            dataset_name=dataset_name,
            gender=supervision.get("gender"),
            speaker_id=supervision.get("speaker"),
            instruction=None,
            source_audio_path=_resolve_fcaps_audio_path(row, dataset_root),
            metadata={"annotation_reject_reason": reason, "custom": custom},
        )

    return CandidateRecord(
        id=str(row.get("id") or f"row_{row_index}"),
        text=text,
        duration=duration,
        annotated_accent=annotated_raw or canonical,
        canonical_accent=canonical,
        source_type="fcaps_jsonl",
        source_path=source_path,
        source_row=row_index,
        dataset_name=dataset_name,
        gender=supervision.get("gender"),
        speaker_id=supervision.get("speaker"),
        instruction=None,
        source_audio_path=_resolve_fcaps_audio_path(row, dataset_root),
        age_group=_infer_age_group(" ".join(captions)),
        metadata={"custom": custom},
    )


def _iter_vctk_candidates(path: Path, *, dataset_name: str | None = None) -> Iterator[CandidateRecord]:
    speaker_info = _load_vctk_speaker_info(path / "speaker-info.txt")
    txt_root = path / "txt"
    for row_index, text_path in enumerate(sorted(txt_root.glob("p*/p*.txt"))):
        speaker_id = text_path.parent.name
        info = speaker_info.get(speaker_id, {})
        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        audio_path = _vctk_audio_path(path, speaker_id, text_path.stem)
        if audio_path is None:
            continue
        duration, sample_rate = _audio_info(audio_path)
        if duration is None:
            continue

        raw_accent = str(info.get("accent") or "").strip()
        canonical = _canonical_vctk_accent(raw_accent)
        region = str(info.get("region") or "").strip()
        annotated = raw_accent if not region else f"{raw_accent} ({region})"
        yield CandidateRecord(
            id=text_path.stem,
            text=text,
            duration=duration,
            annotated_accent=annotated,
            canonical_accent=canonical or "",
            source_type="vctk_directory",
            source_path=str(path),
            source_row=row_index,
            dataset_name=dataset_name,
            gender=_normalize_gender(info.get("gender")),
            speaker_id=speaker_id,
            instruction=f"VCTK speaker accent={annotated}",
            source_audio_path=str(audio_path),
            sample_rate=sample_rate,
            age_group=_age_to_group(info.get("age")),
            metadata={"speaker_info": info},
        )


def _load_vctk_speaker_info(path: Path) -> dict[str, dict[str, Any]]:
    speakers: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        parts = line.split(maxsplit=5)
        if len(parts) < 4:
            continue
        speaker_id, age, gender, accent = parts[:4]
        region = parts[4] if len(parts) >= 5 else ""
        comments = parts[5] if len(parts) >= 6 else ""
        speakers[speaker_id] = {
            "age": age,
            "gender": gender,
            "accent": accent,
            "region": region,
            "comments": comments,
        }
    return speakers


def _canonical_vctk_accent(accent: str) -> str | None:
    key = _label_key(accent)
    if key in {"", "unknown"}:
        return None
    if key == "english":
        return "British English"
    return normalize_accent_label(accent)


def _vctk_audio_path(root: Path, speaker_id: str, utterance_id: str) -> Path | None:
    speaker_dir = root / "wav48_silence_trimmed" / speaker_id
    for suffix in ("mic1.flac", "mic1.wav", "mic2.flac", "mic2.wav"):
        candidate = speaker_dir / f"{utterance_id}_{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(speaker_dir.glob(f"{utterance_id}*"))
    return matches[0] if matches else None


def _iter_ears_candidates(path: Path, *, dataset_name: str | None = None) -> Iterator[CandidateRecord]:
    speaker_stats = json.loads((path / "speaker_statistics.json").read_text(encoding="utf-8"))
    transcripts = json.loads((path / "transcripts.json").read_text(encoding="utf-8"))
    row_index = 0
    for speaker_dir in sorted(item for item in path.glob("p*") if item.is_dir()):
        speaker_id = speaker_dir.name
        info = speaker_stats.get(speaker_id, {}) if isinstance(speaker_stats, dict) else {}
        native_language = str(info.get("native language") or "").strip()
        canonical = _canonical_native_language_accent(native_language)
        for audio_path in sorted(speaker_dir.glob("*.wav")):
            stem = audio_path.stem
            if not (stem.startswith("sentences_") or stem.startswith("rainbow_")):
                continue
            if not stem.endswith("_regular"):
                continue
            text = str(transcripts.get(stem) or "").strip()
            if not text:
                continue
            duration, sample_rate = _audio_info(audio_path)
            if duration is None:
                continue
            yield CandidateRecord(
                id=f"{speaker_id}_{stem}",
                text=text,
                duration=duration,
                annotated_accent=native_language,
                canonical_accent=canonical or "",
                source_type="ears_directory",
                source_path=str(path),
                source_row=row_index,
                dataset_name=dataset_name,
                gender=_normalize_gender(info.get("gender")),
                speaker_id=speaker_id,
                instruction=f"EARS native language={native_language}",
                source_audio_path=str(audio_path),
                sample_rate=sample_rate,
                age_group=_age_to_group(info.get("age")),
                metadata={"speaker_info": info},
            )
            row_index += 1


def _canonical_native_language_accent(native_language: str) -> str | None:
    key = _label_key(native_language)
    if key in {"", "prefer not to answer", "unknown"}:
        return None
    native_language_to_accent = {
        "american english": "General American English",
        "british english": "British English",
        "german": "German-accented English",
        "mandarin": "Chinese-accented English",
        "spanish": "Spanish-accented English",
        "russian": "Russian-accented English",
        "ukrainian": "Ukrainian-accented English",
        "dari": "Dari-accented English",
    }
    return native_language_to_accent.get(key) or normalize_accent_label(native_language)


def _iter_parquet_candidates(
    path: Path,
    *,
    batch_size: int,
    dataset_name: str | None = None,
) -> Iterator[CandidateRecord]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    schema_names = set(parquet_file.schema.names)
    instruct_columns = [name for name in ["audio_id", "audios", "segments"] if name in schema_names]
    ditar_columns = [
        name
        for name in [
            "audio_id",
            "duration",
            "output_result",
            "vocal_audios",
            "orig_audios",
            "bgm_audios",
            "create_time",
            "dt",
        ]
        if name in schema_names
    ]
    if {"audio_id", "audios", "segments"}.issubset(schema_names):
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=instruct_columns):
            for row in batch.to_pylist():
                yield from candidates_from_instruct_parquet_row(
                    row,
                    source_path=str(path),
                    row_index=row_offset,
                    dataset_name=dataset_name,
                )
                row_offset += 1
        return
    if {"audio_id", "output_result"}.issubset(schema_names) and (
        "vocal_audios" in schema_names or "orig_audios" in schema_names
    ):
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=ditar_columns):
            for row in batch.to_pylist():
                yield from candidates_from_ditar_dialect_parquet_row(
                    row,
                    source_path=str(path),
                    row_index=row_offset,
                    dataset_name=dataset_name,
                )
                row_offset += 1
        return
    raise ValueError(f"Unsupported parquet schema for accent selection: {path}")


def candidates_from_instruct_parquet_row(
    row: dict[str, Any],
    *,
    source_path: str,
    row_index: int,
    dataset_name: str | None = None,
) -> list[CandidateRecord]:
    audio_id = str(row.get("audio_id") or f"row_{row_index}")
    try:
        audios = json.loads(row.get("audios") or "{}")
        segments = json.loads(row.get("segments") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(audios, dict) or not isinstance(segments, list):
        return []

    candidates: list[CandidateRecord] = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        start = _float_or_none(segment.get("start")) or 0.0
        end = _float_or_none(segment.get("end"))
        duration = _float_or_none(segment.get("duration"))
        if duration is None and end is not None:
            duration = max(0.0, end - start)
        if not text or duration is None:
            continue
        annotation_texts = _segment_annotation_texts(segment)
        instruction = " ".join(annotation_texts).strip()
        annotated_raw, canonical, reason = infer_annotated_accent(None, annotation_texts)
        chunk_id = str(segment.get("chunk_id") or "")
        audio_b64 = audios.get(chunk_id)
        if audio_b64 is None and len(audios) == 1:
            audio_b64 = next(iter(audios.values()))
        speaker = segment.get("speaker") if isinstance(segment.get("speaker"), dict) else {}
        sample_id = f"{audio_id}_{chunk_id or segment_index}"
        candidates.append(
            CandidateRecord(
                id=sample_id,
                text=text,
                duration=duration,
                annotated_accent=annotated_raw or reason or "",
                canonical_accent=canonical or "",
                source_type="captioned_parquet",
                source_path=source_path,
                source_row=row_index,
                dataset_name=dataset_name,
                segment_index=segment_index,
                gender=_normalize_gender(segment.get("gender") or speaker.get("gender")),
                speaker_id=speaker.get("speaker_id") if isinstance(speaker, dict) else None,
                instruction=instruction,
                embedded_audio_b64=audio_b64,
                embedded_audio_format="wav",
                age_group=_infer_age_group(instruction),
                metadata={"chunk_id": chunk_id, "annotation_reject_reason": reason if not canonical else None},
            )
        )
    return candidates


def candidates_from_ditar_dialect_parquet_row(
    row: dict[str, Any],
    *,
    source_path: str,
    row_index: int,
    dataset_name: str | None = None,
) -> list[CandidateRecord]:
    """Extract candidate chunks from DiTAR dialect_v0.1 parquet rows."""

    try:
        output_result = json.loads(row.get("output_result") or "{}")
        vocal_audios = json.loads(row.get("vocal_audios") or "{}")
        orig_audios = json.loads(row.get("orig_audios") or "{}")
    except json.JSONDecodeError:
        return []
    if not isinstance(output_result, dict):
        return []
    if not isinstance(vocal_audios, dict):
        vocal_audios = {}
    if not isinstance(orig_audios, dict):
        orig_audios = {}
    segments = output_result.get("segments") or []
    if not isinstance(segments, list):
        return []

    audio_id = str(row.get("audio_id") or output_result.get("audio_path") or f"row_{row_index}")
    annotated_accent = _infer_ditar_dialect_accent(
        source_path=source_path,
        dataset_name=dataset_name,
        audio_id=audio_id,
        output_result=output_result,
    )
    canonical_accent = normalize_accent_label(annotated_accent) or ""
    candidates: list[CandidateRecord] = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        start = _float_or_none(segment.get("start"))
        end = _float_or_none(segment.get("end"))
        start_ori = _float_or_none(segment.get("start_ori"))
        end_ori = _float_or_none(segment.get("end_ori"))
        duration = None
        if start is not None and end is not None:
            duration = max(0.0, end - start)
        if duration is None and start_ori is not None and end_ori is not None:
            duration = max(0.0, end_ori - start_ori)
        if not text or duration is None:
            continue

        chunk_id = f"chunk_{segment_index}"
        audio_b64 = _ditar_chunk_audio(vocal_audios, orig_audios, chunk_id)
        sample_id = f"{safe_filename(Path(audio_id).stem)}_{row_index:06d}_{chunk_id}"
        candidates.append(
            CandidateRecord(
                id=sample_id,
                text=text,
                duration=duration,
                annotated_accent=annotated_accent,
                canonical_accent=canonical_accent,
                source_type="ditar_dialect_parquet",
                source_path=source_path,
                source_row=row_index,
                dataset_name=dataset_name,
                segment_index=segment_index,
                instruction=(
                    f"DiTAR dialect_v0.1 source={annotated_accent}; "
                    f"qwen_language={output_result.get('qwen_language')}"
                ),
                source_audio_path=None,
                embedded_audio_b64=audio_b64,
                embedded_audio_format="flac",
                metadata={
                    "audio_id": audio_id,
                    "source_audio_path": output_result.get("audio_path"),
                    "qwen_language": output_result.get("qwen_language"),
                    "row_duration": row.get("duration") or output_result.get("duration"),
                    "segment_start": start,
                    "segment_end": end,
                    "segment_start_ori": start_ori,
                    "segment_end_ori": end_ori,
                    "create_time": row.get("create_time"),
                    "dt": row.get("dt"),
                    "umap_center_similarity": segment.get("umap_center_similarity"),
                    "spectral_center_similarity": segment.get("spectral_center_similarity"),
                    "firered_asr_wer_info": segment.get("firered_asr_wer_info"),
                    "asr_cross_asr_audio_wer": segment.get("asr_cross_asr_audio_wer"),
                    "audio_payload": "vocal_audios" if f"{chunk_id}_vocal.wav" in vocal_audios else "orig_audios",
                },
            )
        )
    return candidates


def _ditar_chunk_audio(
    vocal_audios: dict[str, Any],
    orig_audios: dict[str, Any],
    chunk_id: str,
) -> str | None:
    vocal_key = f"{chunk_id}_vocal.wav"
    orig_key = f"{chunk_id}.wav"
    audio_b64 = vocal_audios.get(vocal_key) or orig_audios.get(orig_key)
    if isinstance(audio_b64, str):
        return audio_b64
    return None


def _infer_ditar_dialect_accent(
    *,
    source_path: str,
    dataset_name: str | None,
    audio_id: str,
    output_result: dict[str, Any],
) -> str:
    text = " ".join(
        str(item or "")
        for item in [
            dataset_name,
            source_path,
            audio_id,
            output_result.get("audio_path"),
        ]
    ).lower()
    path_to_accent = [
        ("beijing_guanhua", "Beijing Guanhua"),
        ("cantonese", "Cantonese"),
        ("dongbei", "Dongbei Dialect"),
        ("min-nan-yu", "Minnan Dialect"),
        ("minnan", "Minnan Dialect"),
        ("sichuan", "Sichuan Dialect"),
        ("wuyu", "Wu Chinese"),
        ("wu", "Wu Chinese"),
    ]
    for marker, accent in path_to_accent:
        if marker in text:
            return accent
    qwen_language = str(output_result.get("qwen_language") or "")
    return normalize_accent_label(qwen_language) or qwen_language or ""


def _segment_annotation_texts(segment: dict[str, Any]) -> list[str]:
    return _flatten_texts(
        segment.get("instruct"),
        segment.get("acoustic_caption"),
        segment.get("caption"),
        segment.get("finegrained_captions"),
        segment.get("global_captions"),
        segment.get("intrinsic_tags"),
    )


def infer_annotated_accent(
    raw_accent: Any,
    annotation_texts: Iterable[str],
) -> tuple[str | None, str | None, str | None]:
    """Infer a canonical accent label from dataset annotation fields."""

    raw = str(raw_accent or "").strip()
    raw_key = _label_key(raw)
    raw_canonical = None if raw_key in GENERIC_ACCENT_VALUES else normalize_accent_label(raw)

    text_mentions: list[str] = []
    for text in annotation_texts:
        text_mentions.extend(extract_accent_mentions(str(text or "")))
    text_canonical = _reduce_compatible_accents(text_mentions)

    if raw_canonical and text_canonical:
        if accents_compatible(raw_canonical, text_canonical):
            chosen = _more_specific_accent(raw_canonical, text_canonical)
            return raw or chosen, chosen, None
        return raw, None, f"conflicting_annotation:{raw_canonical}|{text_canonical}"
    if raw_canonical:
        return raw or raw_canonical, raw_canonical, None
    if text_canonical:
        return text_canonical, text_canonical, None
    if text_mentions:
        return raw or ",".join(sorted(set(text_mentions))), None, "ambiguous_annotation"
    return raw or None, None, "missing_accent_annotation"


def extract_accent_mentions(text: str) -> list[str]:
    key = _label_key(text)
    if not key:
        return []
    mentions: list[str] = []
    for alias_key in _SORTED_ALIAS_KEYS:
        if _contains_label_key(key, alias_key):
            canonical = _ALIAS_TO_CANONICAL[alias_key]
            if canonical not in mentions:
                mentions.append(canonical)
    return mentions


def normalize_accent_label(value: Any) -> str | None:
    key = _label_key(str(value or ""))
    if not key or key in GENERIC_ACCENT_VALUES:
        return None
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    for alias_key in _SORTED_ALIAS_KEYS:
        if _contains_label_key(key, alias_key):
            return _ALIAS_TO_CANONICAL[alias_key]
    return None


def accents_compatible(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    left_norm = normalize_accent_label(left) or left
    right_norm = normalize_accent_label(right) or right
    if left_norm == right_norm:
        return True
    return any(left_norm in group and right_norm in group for group in COMPATIBLE_ACCENT_GROUPS)


def _is_wu_fine_accent(value: str | None) -> bool:
    normalized = normalize_accent_label(value) or value
    return normalized in WU_FINE_ACCENTS


def _needs_wu_fine_resolution(
    candidate_accent: str | None,
    target_accents: Iterable[str],
    accent_quotas: dict[str, int],
) -> bool:
    """Return true when a broad Wu candidate must be resolved to a subtype."""

    candidate = normalize_accent_label(candidate_accent) or candidate_accent
    if candidate != WU_FAMILY_ACCENT:
        return False
    targets = [normalize_accent_label(item) or item for item in target_accents]
    quota_targets = [normalize_accent_label(item) or item for item in accent_quotas]
    return any(_is_wu_fine_accent(item) for item in [*targets, *quota_targets])


def _maybe_refine_candidate_accent_from_gemini(
    candidate: CandidateRecord,
    checks: dict[str, Any],
    *,
    source_targets: list[str],
    accent_quotas: dict[str, int],
) -> None:
    gemini_accent = normalize_accent_label(checks.get("gemini_canonical_accent")) or normalize_accent_label(
        checks.get("gemini_detected_accent")
    )
    if not gemini_accent:
        return
    should_refine_wu = (
        candidate.canonical_accent == WU_FAMILY_ACCENT
        and _is_wu_fine_accent(gemini_accent)
        and (
            _needs_wu_fine_resolution(candidate.canonical_accent, source_targets, accent_quotas)
            or accents_compatible(candidate.canonical_accent, gemini_accent)
        )
    )
    if not should_refine_wu:
        return
    candidate.metadata["accent_family"] = WU_FAMILY_ACCENT
    candidate.metadata["pre_gemini_canonical_accent"] = candidate.canonical_accent
    candidate.metadata["pre_gemini_annotated_accent"] = candidate.annotated_accent
    candidate.canonical_accent = gemini_accent
    candidate.annotated_accent = gemini_accent


def _accent_quota_limits(
    config: SelectionConfig,
    target_accents: Iterable[str],
) -> dict[str, int]:
    quotas = dict(config.target_count_by_accent)
    if config.target_per_accent is not None:
        for accent in target_accents:
            canonical = normalize_accent_label(accent) or str(accent).strip()
            if canonical:
                quotas.setdefault(canonical, config.target_per_accent)
    return {accent: int(limit) for accent, limit in quotas.items() if int(limit) > 0}


def _quota_label_for(accent: str, quotas: dict[str, int]) -> str | None:
    canonical = normalize_accent_label(accent) or accent
    if canonical in quotas:
        return canonical
    for quota_accent in quotas:
        if canonical == WU_FAMILY_ACCENT and _is_wu_fine_accent(quota_accent):
            continue
        if accents_compatible(canonical, quota_accent):
            return quota_accent
    return None


def _accent_quotas_filled(
    selected_by_accent: dict[str, int],
    quotas: dict[str, int],
) -> bool:
    return bool(quotas) and all(selected_by_accent.get(accent, 0) >= limit for accent, limit in quotas.items())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text_for_wer(reference)
    hyp_tokens = normalize_text_for_wer(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _levenshtein_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def normalize_text_for_wer(text: str) -> list[str]:
    normalized = text.lower().translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    return [match.group(0).lower() for match in _WORD_RE.finditer(normalized)]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:180] or "audio"


def load_selection_config(config_path: str | Path) -> SelectionConfig:
    """Load a YAML/JSON selector config with one or more dataset sources."""

    path = Path(config_path).expanduser().resolve()
    data = _load_config_mapping(path)
    sources_data = data.get("datasets") or data.get("sources") or data.get("data_sources")
    if not isinstance(sources_data, list) or not sources_data:
        raise ValueError(f"Config must contain a non-empty datasets/sources list: {path}")

    base_dir = path.parent
    sources: list[DatasetSourceConfig] = []
    for index, item in enumerate(sources_data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset source #{index} must be an object")
        for source_index, source_path in enumerate(_source_paths_from_item(item)):
            source_name = str(item.get("name") or item.get("id") or _default_source_name(source_path))
            if source_index:
                source_name = f"{source_name}_{source_index + 1}"
            sources.append(
                DatasetSourceConfig(
                    name=source_name,
                    path=_resolve_config_path(source_path, base_dir),
                    enabled=_bool_config(item.get("enabled", item.get("use", True))),
                    dataset_root=_optional_config_path(item.get("dataset_root"), base_dir),
                    target_accents=_string_list(item.get("target_accents") or item.get("accents")),
                    language=item.get("language"),
                    max_scan=_optional_int(item.get("max_scan")),
                    max_external_evals=_optional_int(item.get("max_external_evals")),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "name",
                            "id",
                            "path",
                            "paths",
                            "data_path",
                            "data_paths",
                            "storage_path",
                            "storage_paths",
                            "list_path",
                            "enabled",
                            "use",
                            "dataset_root",
                            "target_accents",
                            "accents",
                            "language",
                            "max_scan",
                            "max_external_evals",
                        }
                    },
                )
            )

    return SelectionConfig(
        sources=sources,
        output_path=_optional_config_path(data.get("output_path") or data.get("output"), base_dir) or DEFAULT_OUTPUT,
        dataset_root=_optional_config_path(data.get("dataset_root"), base_dir),
        target_accents=_string_list(data.get("target_accents") or data.get("accents")),
        min_duration=float(data.get("min_duration", 5.0)),
        max_duration=float(data.get("max_duration", 10.0)),
        target_count=int(data.get("target_count", 50)),
        target_per_accent=_optional_int(data.get("target_per_accent")),
        target_count_by_accent=_accent_quota_mapping(data.get("target_count_by_accent")),
        max_scan=int(data.get("max_scan", 5000)),
        max_external_evals=int(data.get("max_external_evals", 250)),
        max_wer=float(data.get("max_wer", 0.03)),
        min_utmos=float(data.get("min_utmos", 3.5)),
        min_gemini_confidence=float(data.get("min_gemini_confidence", 0.65)),
        cache_dir=_optional_config_path(data.get("cache_dir"), base_dir) or DEFAULT_CACHE_DIR,
        audio_dir=_optional_config_path(data.get("audio_dir"), base_dir) or DEFAULT_AUDIO_DIR,
        asr_url=str(data.get("asr_url") or os.getenv("EDIT_PIPELINE_ASR_URL", DEFAULT_ASR_URL)),
        utmos_url=str(
            data.get("utmos_url")
            or os.getenv("X_EDITOR_UTMOS_URL")
            or os.getenv("UTMOS_URL")
            or DEFAULT_UTMOS_URL
        ).rstrip("/"),
        language=data.get("language", "English"),
        skip_gemini=bool(data.get("skip_gemini", False)),
        skip_asr=bool(data.get("skip_asr", False)),
        skip_utmos=bool(data.get("skip_utmos", False)),
        reject_report=_optional_config_path(data.get("reject_report"), base_dir),
        gemini_timeout=float(data.get("gemini_timeout", 60.0)),
        service_timeout=float(data.get("service_timeout", 600.0)),
        parquet_batch_size=int(data.get("parquet_batch_size", 32)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select accent-library candidates with Gemini/ASR/UTMOS checks.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML/JSON config containing datasets/sources. CLI --input is ignored unless no config is set.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="FCaps JSONL, one parquet file, or a .list file containing parquet shards.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output parquet path.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Dataset root for resolving relative FCaps audio paths. Defaults to JSONL parent.",
    )
    parser.add_argument(
        "--target-accent",
        action="append",
        default=[],
        help="Optional accent filter. Can be repeated, e.g. --target-accent british.",
    )
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument(
        "--target-per-accent",
        type=int,
        default=None,
        help="Optional balanced quota per target accent before stopping.",
    )
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--max-external-evals", type=int, default=250)
    parser.add_argument("--max-wer", type=float, default=0.03)
    parser.add_argument("--min-utmos", type=float, default=3.5)
    parser.add_argument("--min-gemini-confidence", type=float, default=0.65)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--asr-url", default=os.getenv("EDIT_PIPELINE_ASR_URL", DEFAULT_ASR_URL))
    parser.add_argument(
        "--utmos-url",
        default=os.getenv("X_EDITOR_UTMOS_URL") or os.getenv("UTMOS_URL") or DEFAULT_UTMOS_URL,
    )
    parser.add_argument("--language", default="English")
    parser.add_argument("--reject-report", default=None)
    parser.add_argument("--gemini-timeout", type=float, default=60.0)
    parser.add_argument("--service-timeout", type=float, default=600.0)
    parser.add_argument("--parquet-batch-size", type=int, default=32)
    parser.add_argument("--skip-gemini", action="store_true", help="Debug only: accept Gemini check.")
    parser.add_argument("--skip-asr", action="store_true", help="Debug only: accept WER check.")
    parser.add_argument("--skip-utmos", action="store_true", help="Debug only: accept UTMOS check.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Equivalent to --skip-gemini --skip-asr --skip-utmos for parser smoke tests.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SelectionConfig:
    skip_gemini = bool(args.skip_gemini or args.dry_run)
    skip_asr = bool(args.skip_asr or args.dry_run)
    skip_utmos = bool(args.skip_utmos or args.dry_run)
    if args.config:
        config = load_selection_config(args.config)
        config.skip_gemini = config.skip_gemini or skip_gemini
        config.skip_asr = config.skip_asr or skip_asr
        config.skip_utmos = config.skip_utmos or skip_utmos
        if args.output != str(DEFAULT_OUTPUT):
            config.output_path = Path(args.output)
        if args.cache_dir != str(DEFAULT_CACHE_DIR):
            config.cache_dir = Path(args.cache_dir)
        if args.audio_dir != str(DEFAULT_AUDIO_DIR):
            config.audio_dir = Path(args.audio_dir)
        if args.reject_report:
            config.reject_report = Path(args.reject_report)
        if args.utmos_url != (os.getenv("X_EDITOR_UTMOS_URL") or os.getenv("UTMOS_URL") or DEFAULT_UTMOS_URL):
            config.utmos_url = args.utmos_url.rstrip("/")
        if args.target_accent:
            config.target_accents = list(args.target_accent)
        if args.min_duration != 5.0:
            config.min_duration = args.min_duration
        if args.max_duration != 10.0:
            config.max_duration = args.max_duration
        if args.target_count != 50:
            config.target_count = args.target_count
        if args.target_per_accent is not None:
            config.target_per_accent = args.target_per_accent
        if args.max_scan != 5000:
            config.max_scan = args.max_scan
        if args.max_external_evals != 250:
            config.max_external_evals = args.max_external_evals
        if args.max_wer != 0.03:
            config.max_wer = args.max_wer
        if args.min_utmos != 3.5:
            config.min_utmos = args.min_utmos
        if args.min_gemini_confidence != 0.65:
            config.min_gemini_confidence = args.min_gemini_confidence
        if args.language != "English":
            config.language = args.language or None
        if args.gemini_timeout != 60.0:
            config.gemini_timeout = args.gemini_timeout
        if args.service_timeout != 600.0:
            config.service_timeout = args.service_timeout
        if args.parquet_batch_size != 32:
            config.parquet_batch_size = args.parquet_batch_size
        return config
    if not args.input:
        raise ValueError("Provide an explicit --input or --config; no dataset is bundled")
    return SelectionConfig(
        input_path=Path(args.input).expanduser().resolve(),
        output_path=Path(args.output),
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        target_accents=list(args.target_accent or []),
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        target_count=args.target_count,
        target_per_accent=args.target_per_accent,
        max_scan=args.max_scan,
        max_external_evals=args.max_external_evals,
        max_wer=args.max_wer,
        min_utmos=args.min_utmos,
        min_gemini_confidence=args.min_gemini_confidence,
        cache_dir=Path(args.cache_dir),
        audio_dir=Path(args.audio_dir),
        asr_url=args.asr_url,
        utmos_url=args.utmos_url.rstrip("/"),
        language=args.language or None,
        skip_gemini=skip_gemini,
        skip_asr=skip_asr,
        skip_utmos=skip_utmos,
        reject_report=Path(args.reject_report) if args.reject_report else None,
        gemini_timeout=args.gemini_timeout,
        service_timeout=args.service_timeout,
        parquet_batch_size=args.parquet_batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = select_accent_candidates(config_from_args(args))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_config_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config files; use JSON instead") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return data


def _source_paths_from_item(item: dict[str, Any]) -> list[Any]:
    multi = (
        item.get("paths")
        or item.get("data_paths")
        or item.get("storage_paths")
    )
    if multi is not None:
        values = _string_list(multi)
        if not values:
            raise ValueError(f"Dataset source has empty paths: {item}")
        return values
    single = item.get("path") or item.get("data_path") or item.get("storage_path") or item.get("list_path")
    if not single:
        raise ValueError(f"Dataset source missing path/data_path/storage_path/list_path: {item}")
    return [single]


def _resolve_config_path(value: Any, base_dir: Path) -> Path:
    raw = os.path.expandvars(str(value))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _optional_config_path(value: Any, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_config_path(value, base_dir)


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _accent_quota_mapping(value: Any) -> dict[str, int]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("target_count_by_accent must be a mapping of accent label to integer quota")
    quotas: dict[str, int] = {}
    for accent, limit in value.items():
        canonical = normalize_accent_label(accent) or str(accent).strip()
        if canonical:
            quotas[canonical] = int(limit)
    return quotas


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _bool_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _default_source_name(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".list":
        return path.stem
    if path.suffix.lower() == ".parquet":
        return path.parent.name or path.stem
    return path.stem


def _candidate_precheck(
    candidate: CandidateRecord,
    config: SelectionConfig,
    target_accents: list[str],
) -> str | None:
    if not candidate.canonical_accent:
        return "missing_or_ambiguous_accent_annotation"
    if candidate.duration < config.min_duration or candidate.duration > config.max_duration:
        return "duration_out_of_range"
    if target_accents and not any(
        accents_compatible(candidate.canonical_accent, target) for target in target_accents
    ):
        return "target_accent_mismatch"
    return None


def _check_gemini(
    candidate: CandidateRecord,
    audio_path: Path,
    *,
    client: GeminiClient | None,
    cache: JsonCache,
    min_confidence: float,
    skip: bool,
    target_accents: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if skip:
        return True, {
            "gemini_detected_accent": "skipped",
            "gemini_canonical_accent": candidate.canonical_accent,
            "gemini_confidence": None,
            "gemini_matches_annotation": True,
            "gemini_reason": "skipped",
        }
    if client is None:
        return False, {"gemini_reason": "gemini_client_not_configured"}

    target_key = json.dumps(sorted(target_accents or []), ensure_ascii=False)
    cache_key = f"{_cache_key(candidate, 'gemini')}:{target_key}"
    cached = cache.get("gemini", cache_key)
    if cached and "Failed to parse JSON" in str(cached.get("error", "")):
        cached = None
    if cached is None:
        try:
            cached = _call_gemini_accent_judge(
                candidate,
                audio_path,
                client,
                target_accents=target_accents,
            )
        except Exception as exc:
            cached = {"error": str(exc)}
        cache.set("gemini", cache_key, cached)

    if cached.get("error"):
        return False, {"gemini_reason": cached["error"], "gemini_matches_annotation": False}

    detected = str(cached.get("detected_accent") or "")
    gemini_canonical = str(cached.get("canonical_accent") or detected)
    confidence = _float_or_none(cached.get("confidence"))
    model_match = bool(cached.get("matches_annotation"))
    local_match = accents_compatible(candidate.canonical_accent, gemini_canonical) or accents_compatible(
        candidate.canonical_accent,
        detected,
    )
    ok = model_match and local_match and (confidence is None or confidence >= min_confidence)
    return ok, {
        "gemini_detected_accent": detected or None,
        "gemini_canonical_accent": normalize_accent_label(gemini_canonical) or gemini_canonical or None,
        "gemini_confidence": confidence,
        "gemini_matches_annotation": ok,
        "gemini_reason": cached.get("reason") or cached.get("explanation"),
    }


def _call_gemini_accent_judge(
    candidate: CandidateRecord,
    audio_path: Path,
    client: GeminiClient,
    *,
    target_accents: list[str] | None = None,
) -> dict[str, Any]:
    system_prompt = """You are an accent verification judge for a speech dataset.
Output one compact JSON object only. Do not explain outside JSON.
Judge whether the spoken accent in the audio is consistent with the annotation.
Use the provided canonical accent names when possible.
If the annotation is a broad dialect family such as Wu Chinese, return the most
specific subtype you can identify, such as Shanghai Dialect, Suzhou Dialect, or
Ningbo Dialect. Treat a subtype as matching its broad family."""
    user_payload = {
        "task": "accent_consistency_check",
        "annotated_accent": candidate.annotated_accent,
        "canonical_annotation": candidate.canonical_accent,
        "requested_target_accents": target_accents or [],
        "allowed_canonical_accent_labels": sorted(ACCENT_ALIASES),
        "fine_grained_dialect_families": {
            "Wu Chinese": sorted(WU_FINE_ACCENTS),
        },
        "gender_annotation": candidate.gender,
        "transcript": candidate.text,
        "instruction_or_caption": candidate.instruction,
        "allowed_output_schema": {
            "detected_accent": "short label",
            "canonical_accent": "one canonical accent label",
            "matches_annotation": True,
            "confidence": 0.0,
            "reason": "short reason",
        },
    }
    return client.call_json(system_prompt, json.dumps(user_payload, ensure_ascii=False), audio_paths=[str(audio_path)])


def _check_wer(
    candidate: CandidateRecord,
    audio_path: Path,
    *,
    asr_client: SpeechPerceptionClient | None,
    cache: JsonCache,
    max_wer: float,
    language: str | None,
    skip: bool,
) -> tuple[bool, dict[str, Any]]:
    if skip:
        return True, {"asr_text": "skipped", "asr_error": None, "wer": None}
    if asr_client is None:
        return False, {"asr_text": None, "asr_error": "asr_client_not_configured", "wer": None}

    cache_key = _cache_key(candidate, "asr")
    cached = cache.get("asr", cache_key)
    if cached is None:
        try:
            cached = asr_client.transcribe(audio_path, language=language)
        except Exception as exc:
            cached = {"error": str(exc)}
        cache.set("asr", cache_key, cached)
    if cached.get("error"):
        return False, {"asr_text": None, "asr_error": cached["error"], "wer": None}

    asr_text = str(cached.get("text") or "").strip()
    wer = word_error_rate(candidate.text, asr_text)
    return wer <= max_wer, {"asr_text": asr_text, "asr_error": None, "wer": wer}


def _check_utmos(
    audio_path: Path,
    *,
    cache: JsonCache,
    utmos_url: str,
    min_utmos: float,
    timeout: float,
    skip: bool,
) -> tuple[bool, dict[str, Any]]:
    if skip:
        return True, {"utmos": None, "utmos_error": None}
    cache_key = f"{audio_path}:{audio_path.stat().st_size}:{audio_path.stat().st_mtime_ns}"
    cached = cache.get("utmos", cache_key)
    if cached is None:
        try:
            cached = _call_utmos(audio_path, utmos_url=utmos_url, timeout=timeout)
        except Exception as exc:
            cached = {"error": str(exc)}
        cache.set("utmos", cache_key, cached)
    if cached.get("error"):
        return False, {"utmos": None, "utmos_error": cached["error"]}
    score = _float_or_none(cached.get("score") or cached.get("utmos") or cached.get("mos"))
    return score is not None and score > min_utmos, {"utmos": score, "utmos_error": None}


def _call_utmos(audio_path: Path, *, utmos_url: str, timeout: float) -> dict[str, Any]:
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    response = requests.post(
        f"{utmos_url.rstrip('/')}/predict",
        json={"audio_data": audio_b64},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if "score" not in data:
        raise RuntimeError(f"UTMOS response missing score: {data}")
    return data


def _fill_audio_metadata(candidate: CandidateRecord, audio_path: Path) -> None:
    if candidate.sample_rate is not None:
        return
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        candidate.sample_rate = int(info.samplerate)
        if not candidate.duration and info.frames and info.samplerate:
            candidate.duration = float(info.frames / info.samplerate)
    except Exception:
        return


def _append_reject(
    reject_rows: list[dict[str, Any]],
    candidate: CandidateRecord,
    reason: str,
    checks: dict[str, Any] | None = None,
    audio_path: str | None = None,
) -> None:
    row = candidate.output_row(audio_path, checks or {})
    row["selected"] = False
    row["reject_reason"] = reason
    reject_rows.append(row)


def _resolve_fcaps_audio_path(row: dict[str, Any], dataset_root: str | Path) -> str | None:
    sources = ((row.get("recording") or {}).get("sources") or [])
    source = sources[0].get("source") if sources else None
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = Path(dataset_root) / source_path
    return str(source_path)


def _read_list_file(path: Path) -> list[Path]:
    shards: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        shard = Path(item)
        if not shard.is_absolute():
            shard = path.parent / shard
        shards.append(shard)
    return shards


def _flatten_texts(*values: Any) -> list[str]:
    texts: list[str] = []
    for value in values:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, Iterable):
            texts.extend(str(item) for item in value if item is not None)
    return texts


def _reduce_compatible_accents(accents: list[str]) -> str | None:
    unique: list[str] = []
    for accent in accents:
        if accent not in unique:
            unique.append(accent)
    if not unique:
        return None
    for accent in unique:
        if all(accents_compatible(accent, other) for other in unique):
            return accent
    return None


def _more_specific_accent(left: str, right: str) -> str:
    if left == WU_FAMILY_ACCENT and _is_wu_fine_accent(right):
        return right
    if right == WU_FAMILY_ACCENT and _is_wu_fine_accent(left):
        return left
    if left == "General American English" and right == "Southern American English":
        return right
    if right == "General American English" and left == "Southern American English":
        return left
    return right if right != left else left


def _infer_age_group(text: str | None) -> str | None:
    key = _label_key(text or "")
    if "middle aged" in key:
        return "middle_aged"
    if "young adult" in key or "young" in key:
        return "young_adult"
    if "child" in key:
        return "child"
    if "teen" in key:
        return "teen"
    if "elderly" in key or "senior" in key or "old man" in key or "old woman" in key:
        return "senior"
    if "mature" in key:
        return "adult"
    return None


def _age_to_group(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    number = None
    match = re.search(r"\d+", text)
    if match:
        number = int(match.group(0))
    if number is not None:
        if number < 13:
            return "child"
        if number < 20:
            return "teen"
        if number < 36:
            return "young_adult"
        if number < 56:
            return "adult"
        return "senior"
    return _infer_age_group(text)


def _normalize_gender(value: Any) -> str | None:
    key = str(value or "").strip().lower()
    if key in {"m", "male", "man"}:
        return "male"
    if key in {"f", "female", "woman"}:
        return "female"
    if key:
        return key
    return None


def _audio_info(path: Path) -> tuple[float | None, int | None]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return float(info.frames / info.samplerate), int(info.samplerate)
        return None, int(info.samplerate) if info.samplerate else None
    except Exception:
        return None, None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (left_item != right_item)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _cache_key(candidate: CandidateRecord, stage: str) -> str:
    payload = {
        "stage": stage,
        "id": candidate.id,
        "text": candidate.text,
        "duration": candidate.duration,
        "accent": candidate.canonical_accent,
        "source_path": candidate.source_path,
        "source_row": candidate.source_row,
        "segment_index": candidate.segment_index,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
