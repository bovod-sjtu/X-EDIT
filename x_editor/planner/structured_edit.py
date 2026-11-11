"""Natural-language speech edit parsing into edit-data-pipeline calls."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructuredEdit:
    operation: str
    instruction_st: str | None = None
    source_text: str | None = None
    target_text: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    supported: bool = True
    reason: str | None = None


def build_structured_edit_sequence(
    prompt: str,
    source_text: str | None,
    *,
    language: str | None = None,
) -> list[StructuredEdit]:
    """Convert one user instruction into ordered edit-data-pipeline operations."""

    current_text = source_text or "<source_text_from_perception>"
    operations: list[StructuredEdit] = []

    text_edit = build_text_edit(prompt, current_text, language=language)
    if text_edit:
        operations.append(text_edit)
        if text_edit.target_text:
            current_text = text_edit.target_text

    emotion_edit = build_emotion_edit(prompt, current_text, language=language)
    if emotion_edit:
        operations.append(emotion_edit)

    pause_edit = build_pause_edit(prompt, current_text, language=language)
    if pause_edit:
        operations.append(pause_edit)

    speaker_pitch_edit, speaker_volume_edit = build_speaker_profile_edits(prompt, current_text, language=language)

    pitch_rate_edit = build_pitch_rate_edit(prompt, current_text, language=language)
    if pitch_rate_edit:
        operations.append(pitch_rate_edit)
    elif speaker_pitch_edit:
        operations.append(speaker_pitch_edit)

    volume_edit = build_volume_edit(prompt, current_text, language=language)
    if volume_edit:
        operations.append(volume_edit)
    elif speaker_volume_edit:
        operations.append(speaker_volume_edit)

    return operations


def build_edit_data_pipeline_arguments(
    operation: str,
    prompt: str,
    source_text: str | None,
    *,
    input_audio_path: str,
    output_path: str,
    output_dir: str | None = None,
    language: str | None = None,
) -> dict[str, Any] | None:
    """Build concrete arguments for one edit-data-pipeline operation."""

    source = source_text or "<source_text_from_perception>"
    sequence = build_structured_edit_sequence(prompt, source, language=language)
    edit = next((item for item in sequence if item.operation == operation), None)
    if edit is None:
        edit = _fallback_edit(operation, prompt, source, language=language)
    if edit is None or not edit.supported:
        return None

    args: dict[str, Any] = {
        "operation": edit.operation,
        "input_audio_path": input_audio_path,
        "output_path": output_path,
        "instruction": prompt,
    }
    args.update(edit.arguments)
    if edit.source_text is not None:
        args["source_text"] = edit.source_text
    if edit.instruction_st is not None:
        args["instruction_st"] = edit.instruction_st
    if edit.target_text is not None:
        args["target_text"] = edit.target_text
    if edit.operation in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit"}:
        args.setdefault("language", language or _language_hint(edit.source_text or source))
        args.setdefault("evaluate", False)
    if edit.operation == "text_edit" and _looks_long_for_f5(edit.source_text or source):
        args.setdefault("disable_prepend_context", True)
        args.setdefault("fail_if_cannot_prepend_context", False)
        args.setdefault("synthesis_mode", "per_op")
        args.setdefault("duration_mode", "estimate")
        args.setdefault("expand_radius", 0.12)
        args.setdefault("merge_gap", 0.0)
    if edit.operation in {"emotion_style_edit", "pause_edit"}:
        args.setdefault("persist", True)
        if output_dir:
            args.setdefault("output_dir", f"{output_dir}/{edit.operation}_service_artifacts")
    return args


def build_local_text_edit_repair_arguments(
    prompt: str,
    source_text: str | None,
    *,
    input_audio_path: str,
    output_path: str,
    output_dir: str | None = None,
    language: str | None = None,
) -> dict[str, Any] | None:
    """Build text_edit args for a local source-text window plus later audio splice."""

    if not source_text:
        return None
    window = _local_repair_window(prompt, source_text)
    if window is None:
        return None
    start, end = window
    local_source = source_text[start:end]
    args = build_edit_data_pipeline_arguments(
        "text_edit",
        prompt,
        local_source,
        input_audio_path=input_audio_path,
        output_path=output_path,
        output_dir=output_dir,
        language=language,
    )
    if args is None:
        return None
    args["_x_editor_repair_splice"] = {
        "strategy": "estimated_local_text_edit_splice",
        "source_char_start": start,
        "source_char_end": end,
        "source_text_length": len(source_text),
        "full_input_audio_path": input_audio_path,
        "final_output_path": output_path,
    }
    return args


def build_text_edit(prompt: str, source_text: str, *, language: str | None = None) -> StructuredEdit | None:
    text = prompt.lower()
    quoted = _quoted_strings(prompt)

    if _contains(text, ["translate", "convert"]) and _contains(text, ["chinese", "english", "japanese", "spanish", "french"]):
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            supported=False,
            reason="speech-to-speech translation needs a target transcript generator before text_edit",
        )

    old_new = _replace_pair(prompt)
    if old_new:
        old, new = old_new
        all_occurrences = _contains(text, ["every instance", "all instance", "each instance", "每个", "所有"])
        instruction_st, target_text = _sub_instruction(source_text, old, new, all_occurrences=all_occurrences)
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=target_text,
        )

    swap = _swap_pair(prompt)
    if swap:
        first, second = swap
        instruction_st, target_text = _swap_instruction(source_text, first, second)
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=target_text,
        )

    insert = _insert_request(prompt)
    if insert:
        insertion, anchor, placement = insert
        result = _insert_instruction(source_text, insertion, anchor, placement, prompt=prompt)
        if result is None:
            return StructuredEdit(
                operation="text_edit",
                source_text=source_text,
                supported=False,
                reason="could not locate insertion anchor in source_text",
            )
        instruction_st, target_text = result
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=target_text,
        )

    if _contains(text, ["replace the first sentence", "replace first sentence"]) and quoted:
        old = _first_sentence(source_text)
        new = quoted[-1]
        instruction_st, target_text = _sub_instruction(source_text, old, new)
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=target_text,
        )

    if _contains(text, ["delete the first sentence", "remove the first sentence", "删除第一句", "移除第一句"]):
        old = _first_sentence(source_text)
        result = _delete_instruction(source_text, old)
        if result is None:
            return StructuredEdit(
                operation="text_edit",
                source_text=source_text,
                supported=False,
                reason="could not locate deletion span in source_text",
            )
        instruction_st, target_text = result
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=target_text,
        )

    if _contains(text, ["delete the last sentence", "remove the last sentence", "删除最后一句", "移除最后一句"]):
        old = _last_sentence(source_text)
        result = _delete_instruction(source_text, old)
        if result is None:
            return StructuredEdit(
                operation="text_edit",
                source_text=source_text,
                supported=False,
                reason="could not locate deletion span in source_text",
            )
        instruction_st, target_text = result
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=target_text,
        )

    if _contains(text, ["remove", "delete", "删除", "移除", "去掉", "删掉"]) and quoted:
        old = quoted[0]
        all_occurrences = _contains(text, ["all", "every", "each", "所有", "全部", "每个", "每一次"])
        result = _delete_instruction(source_text, old, all_occurrences=all_occurrences)
        if result is None:
            return StructuredEdit(
                operation="text_edit",
                source_text=source_text,
                supported=False,
                reason="could not locate deletion phrase in source_text",
            )
        instruction_st, target_text = result
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=target_text,
        )

    if _contains(text, ["改字", "改文字", "改文本", "改内容", "改文案", "改台词", "rewrite text", "replace text"]) and quoted:
        target = quoted[-1]
        return StructuredEdit(
            operation="text_edit",
            source_text=source_text,
            instruction_st=f'<sub targ="{_xml_attr(target)}">{_xml_text(source_text)}</sub>',
            target_text=target,
        )

    return None


def build_pitch_rate_edit(prompt: str, source_text: str, *, language: str | None = None) -> StructuredEdit | None:
    if _wants_question_to_declarative_tone(prompt):
        return None
    if _wants_questioning_tone_target(prompt):
        quoted = _quoted_strings(prompt)
        target = quoted[0] if quoted else _last_word(source_text)
        all_occurrences = _contains(prompt, ["all", "every", "each", "instances", "所有", "全部"])
        instruction_st = _wrap_phrase_occurrences_with_trailing_marks(
            source_text,
            target,
            lambda text: f'<pitch, semitones=2>{_xml_text(text)}</pitch>',
            all_occurrences=all_occurrences,
            marks={".", "!", "。", "！"},
        )
        return StructuredEdit(
            operation="pitch_rate_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            arguments={
                "pitch_method": "psola+indextts2_cco",
                "smooth_policy": "never",
            },
        )
    if _wants_emphasis(prompt):
        quoted = _quoted_strings(prompt)
        target = quoted[0] if quoted else _last_word(source_text)
        instruction_st = _wrap_first_occurrence(
            source_text,
            target,
            lambda text: f'<pitch, semitones=4>{_xml_text(text)}</pitch>',
        )
        return StructuredEdit(
            operation="pitch_rate_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            arguments={
                "pitch_method": "psola+indextts2_cco",
                "smooth_policy": "never",
            },
        )
    if _wants_pitch(prompt):
        semitones = _pitch_semitones(prompt)
        span = _marked_span_from_prompt(prompt, source_text)
        instruction_st = (
            _wrap_source_span(source_text, span, lambda text: f'<pitch, semitones={semitones:g}>{_xml_text(text)}</pitch>')
            if span
            else f'<pitch, semitones={semitones:g}>{_xml_text(source_text)}</pitch>'
        )
        return StructuredEdit(
            operation="pitch_rate_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            arguments={
                "pitch_method": "psola+indextts2_cco",
                "smooth_policy": "never",
            },
        )
    if _wants_rate(prompt):
        factor = _rate_factor(prompt)
        span = _marked_span_from_prompt(prompt, source_text, allow_to_end=True)
        instruction_st = (
            _wrap_source_span(source_text, span, lambda text: f'<rate, factor={factor:g}>{_xml_text(text)}</rate>')
            if span
            else f'<rate, factor={factor:g}>{_xml_text(source_text)}</rate>'
        )
        return StructuredEdit(
            operation="pitch_rate_edit",
            source_text=source_text,
            instruction_st=instruction_st,
            arguments={
                "pitch_method": "psola+indextts2_cco",
                "smooth_policy": "auto",
            },
        )
    return None


def build_emotion_edit(prompt: str, source_text: str, *, language: str | None = None) -> StructuredEdit | None:
    if not _wants_emotion(prompt):
        return None
    if _wants_question_to_declarative_tone(prompt):
        return StructuredEdit(
            operation="emotion_style_edit",
            source_text=source_text,
            instruction_st=f'<emo type="calm" level="1">{_xml_text(source_text)}</emo>',
            arguments={
                "instruction_nl": prompt,
                "return_synthetic_target_audio": True,
                "max_candidates": 3,
            },
        )
    vocal_effort_style = _target_vocal_effort_style(prompt)
    if vocal_effort_style:
        return StructuredEdit(
            operation="emotion_style_edit",
            source_text=source_text,
            instruction_st=f'<emo desc="{_xml_attr(vocal_effort_style)}">{_xml_text(source_text)}</emo>',
            arguments={
                "emotion_condition_mode": "desc",
                "synthesis_mode": "single",
                "instruction_nl": prompt,
                "return_synthetic_target_audio": True,
                "max_candidates": 3,
            },
        )
    emotion = _emotion_tag(prompt)
    level = 3 if emotion in {"angry", "surprised"} else _emotion_level(prompt)
    return StructuredEdit(
        operation="emotion_style_edit",
        source_text=source_text,
        instruction_st=f'<emo type="{_xml_attr(emotion)}" level="{level}">{_xml_text(source_text)}</emo>',
        arguments={
            "return_synthetic_target_audio": True,
            "max_candidates": 3,
        },
    )


def build_pause_edit(prompt: str, source_text: str, *, language: str | None = None) -> StructuredEdit | None:
    if not _wants_pause(prompt):
        return None
    return StructuredEdit(
        operation="pause_edit",
        source_text=source_text,
        instruction_st=_pause_instruction_st(prompt, source_text),
        arguments={
            "profile_id": 1,
            "stage2_method": "method_one",
            "expand_radius": 0.16,
        },
    )


def build_volume_edit(prompt: str, source_text: str, *, language: str | None = None) -> StructuredEdit | None:
    if not _wants_volume(prompt):
        return None
    return StructuredEdit(
        operation="volume_normalize",
        source_text=source_text,
        arguments={
            "target_db": _target_db(prompt),
            "method": "rms",
        },
    )


def build_speaker_profile_edits(
    prompt: str,
    source_text: str,
    *,
    language: str | None = None,
) -> tuple[StructuredEdit | None, StructuredEdit | None]:
    params = _speaker_profile_params(prompt)
    if not params:
        return None, None

    pitch_edit = StructuredEdit(
        operation="pitch_rate_edit",
        source_text=source_text,
        instruction_st=f'<pitch, semitones={params["semitones"]:g}>{_xml_text(source_text)}</pitch>',
        arguments={
            "pitch_method": "psola+indextts2_cco",
            "smooth_policy": "never",
            "inferred_speaker_control": True,
            "speaker_control_profile": params["profile"],
        },
    )
    volume_edit = StructuredEdit(
        operation="volume_normalize",
        source_text=source_text,
        arguments={
            "target_db": params["target_db"],
            "method": "rms",
            "inferred_speaker_control": True,
            "speaker_control_profile": params["profile"],
        },
    )
    return pitch_edit, volume_edit


def _fallback_edit(operation: str, prompt: str, source_text: str, *, language: str | None = None) -> StructuredEdit | None:
    if operation == "text_edit":
        return build_text_edit(prompt, source_text, language=language)
    if operation == "pitch_rate_edit":
        return build_pitch_rate_edit(prompt, source_text, language=language) or build_speaker_profile_edits(
            prompt,
            source_text,
            language=language,
        )[0]
    if operation == "emotion_style_edit":
        return build_emotion_edit(prompt, source_text, language=language)
    if operation == "pause_edit":
        return build_pause_edit(prompt, source_text, language=language)
    if operation == "volume_normalize":
        return build_volume_edit(prompt, source_text, language=language) or build_speaker_profile_edits(
            prompt,
            source_text,
            language=language,
        )[1]
    return None


def _replace_pair(prompt: str) -> tuple[str, str] | None:
    quoted = _quoted_strings(prompt)
    if len(quoted) >= 2 and _contains(prompt.lower(), ["replace", "change", "modify", "relace", "改成", "替换成", "换成", "改为", "替换为"]):
        return quoted[0], quoted[1]
    lowered = prompt.lower()
    if len(quoted) == 1 and _contains(lowered, ["period of silence", "silence", "silent gap"]):
        return quoted[0], ""
    patterns = [
        r"\breplace\s+(?:all\s+)?(?:instances?\s+of\s+)?(?:the\s+word\s+)?(?P<old>[A-Za-z0-9_.+-]+)\s+with\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
        r"\bchange\s+(?:the\s+)?word\s+(?P<old>[A-Za-z0-9_.+-]+)\s+to\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
        r"\bchange\s+(?P<old>[A-Za-z0-9_.+-]+)\s+to\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            old = match.group("old").strip(".,;:!?")
            new = match.group("new").strip(".,;:!?")
            if old and new:
                return old, new
    return None


def _swap_pair(prompt: str) -> tuple[str, str] | None:
    quoted = _quoted_strings(prompt)
    lowered = prompt.lower()
    if len(quoted) >= 2 and _contains(
        lowered,
        [
            "swap",
            "switch",
            "exchange",
            "switch order",
            "swap order",
            "reverse order",
            "交换",
            "互换",
            "调换",
            "对调",
            "交换顺序",
            "调整顺序",
        ],
    ):
        return quoted[0], quoted[1]

    match = re.search(
        r"\b(?:swap|switch|exchange)\s+(.+?)\s+(?:and|with)\s+(.+?)(?:[.!?。！？]|$)",
        prompt,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = re.search(r"(?:交换|互换|调换)(.+?)(?:和|与)(.+?)(?:的?顺序)?(?:[。！？.!?]|$)", prompt)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return None


def _insert_request(prompt: str) -> tuple[str, str | None, str] | None:
    quoted = _quoted_string_spans(prompt)
    lowered = prompt.lower()
    if not _contains(lowered, ["insert", "add", "append", "prepend", "插入", "加入", "添加"]):
        return None

    if len(quoted) >= 2:
        first_text, first_start, first_end = quoted[0]
        second_text, second_start, _second_end = quoted[1]
        between = lowered[first_end:second_start]
        before_first = lowered[:first_start]
        if _contains(between, ["insert", "add", "插入", "加入", "添加"]):
            if _contains(between, ["after", "following", "之后", "后面", "后"]):
                return second_text, first_text, "after"
            if _contains(between, ["before", "prior to", "之前", "前面", "前"]):
                return second_text, first_text, "before"
        if _contains(between, ["after", "following", "之后", "后面", "后"]):
            return first_text, second_text, "after"
        if _contains(between, ["before", "prior to", "之前", "前面", "前"]):
            return first_text, second_text, "before"
        if _contains(before_first, ["after", "following", "之后", "后面", "后"]) and _contains(
            between,
            ["insert", "add", "插入", "加入", "添加"],
        ):
            return second_text, first_text, "after"
        if _contains(before_first, ["before", "prior to", "之前", "前面", "前"]) and _contains(
            between,
            ["insert", "add", "插入", "加入", "添加"],
        ):
            return second_text, first_text, "before"

    if len(quoted) >= 1:
        insertion = quoted[0][0]
        if _contains(lowered, ["prepend", "at the beginning", "at the start", "beginning of", "start of", "开头", "开始"]):
            return insertion, None, "start"
        if _contains(lowered, ["append", "at the end", "end of", "ending", "末尾", "结尾", "最后"]):
            return insertion, None, "end"

    patterns = [
        (r"\b(?:insert|add)\s+(?P<insert>[A-Za-z0-9_.+-]+)\s+(?:after|following)\s+(?P<anchor>[A-Za-z0-9_.+-]+)\b", "after"),
        (r"\b(?:insert|add)\s+(?P<insert>[A-Za-z0-9_.+-]+)\s+before\s+(?P<anchor>[A-Za-z0-9_.+-]+)\b", "before"),
    ]
    for pattern, placement in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            insertion = match.group("insert").strip(".,;:!?")
            anchor = match.group("anchor").strip(".,;:!?")
            if insertion and anchor:
                return insertion, anchor, placement
    return None


def _insert_instruction(
    source_text: str,
    insertion: str,
    anchor: str | None,
    placement: str,
    *,
    prompt: str,
) -> tuple[str, str] | None:
    if not insertion:
        return None
    all_occurrences = _contains(prompt.lower(), ["every instance", "all instances", "each instance", "每个", "所有", "全部"])
    if placement == "start":
        positions = [0]
    elif placement == "end":
        positions = [len(source_text)]
    else:
        if not anchor:
            return None
        spans = _source_spans_for_needle(source_text, anchor, all_occurrences=all_occurrences)
        if not spans:
            return None
        if not all_occurrences and _contains(prompt.lower(), ["last", "final", "最后", "末尾", "结尾"]):
            spans = spans[-1:]
        positions = [start if placement == "before" else end for start, end in spans]

    instruction = source_text
    target = source_text
    for position in sorted(positions, reverse=True):
        insert_text = _spaced_insertion(source_text, position, insertion)
        instruction = instruction[:position] + f"<ins>{_xml_text(insert_text)}</ins>" + instruction[position:]
        target = target[:position] + insert_text + target[position:]
    return instruction, target


def _spaced_insertion(source_text: str, position: int, insertion: str) -> str:
    text = insertion.strip()
    if not text:
        return insertion
    prev_char = source_text[position - 1] if position > 0 else ""
    next_char = source_text[position] if position < len(source_text) else ""
    leading = ""
    trailing = ""
    if (
        prev_char
        and not prev_char.isspace()
        and not (_is_cjk_char(prev_char) or _is_cjk_char(text[0]))
        and _wordish_boundary_char(prev_char)
        and not text[0].isspace()
        and _wordish_boundary_char(text[0])
    ):
        leading = " "
    if (
        next_char
        and not next_char.isspace()
        and not (_is_cjk_char(next_char) or _is_cjk_char(text[-1]))
        and _wordish_boundary_char(next_char)
        and not text[-1].isspace()
        and _wordish_boundary_char(text[-1])
    ):
        trailing = " "
    return f"{leading}{text}{trailing}"


def _wordish_boundary_char(value: str) -> bool:
    return value.isalnum() or "\u4e00" <= value <= "\u9fff"


def _is_cjk_char(value: str) -> bool:
    return bool(value) and "\u4e00" <= value <= "\u9fff"


def _sub_instruction(
    source_text: str,
    old: str,
    new: str,
    *,
    all_occurrences: bool = False,
) -> tuple[str, str]:
    new = _target_alias_for_source(source_text, new)
    if not old:
        return f'<sub targ="{_xml_attr(new)}">{_xml_text(source_text)}</sub>', new
    spans = _source_spans_for_needle(source_text, old, all_occurrences=all_occurrences)
    if not spans:
        return f'<sub targ="{_xml_attr(new)}">{_xml_text(source_text)}</sub>', new
    parts = []
    last = 0
    target_parts = []
    target_last = 0
    for start, end in spans:
        parts.append(_xml_text(source_text[last:start]))
        parts.append(f'<sub targ="{_xml_attr(new)}">{_xml_text(source_text[start:end])}</sub>')
        last = end
        target_parts.append(source_text[target_last:start])
        target_parts.append(new)
        target_last = end
    parts.append(_xml_text(source_text[last:]))
    target_parts.append(source_text[target_last:])
    target_text = "".join(target_parts)
    return "".join(parts), target_text


def _delete_instruction(
    source_text: str,
    old: str,
    *,
    all_occurrences: bool = False,
) -> tuple[str, str] | None:
    if not old:
        return None
    spans = _source_spans_for_needle(source_text, old, all_occurrences=all_occurrences)
    if not spans:
        return None
    instruction_parts = []
    target_parts = []
    last = 0
    for start, end in spans:
        instruction_parts.append(_xml_text(source_text[last:start]))
        instruction_parts.append(f"<del>{_xml_text(source_text[start:end])}</del>")
        target_parts.append(source_text[last:start])
        last = end
    instruction_parts.append(_xml_text(source_text[last:]))
    target_parts.append(source_text[last:])
    return "".join(instruction_parts), "".join(target_parts)


def _swap_instruction(source_text: str, first: str, second: str) -> tuple[str, str]:
    first_spans = _source_spans_for_needle(source_text, first)
    second_spans = _source_spans_for_needle(source_text, second)
    if not first_spans or not second_spans:
        return f'<sub targ="{_xml_attr(source_text)}">{_xml_text(source_text)}</sub>', source_text

    first_span = first_spans[0]
    second_span = second_spans[0]
    if first_span[0] == second_span[0]:
        return f'<sub targ="{_xml_attr(source_text)}">{_xml_text(source_text)}</sub>', source_text

    replacements = [
        (first_span[0], first_span[1], second),
        (second_span[0], second_span[1], first),
    ]
    replacements.sort(key=lambda item: item[0])

    instruction_parts = []
    target_parts = []
    last = 0
    for start, end, target in replacements:
        instruction_parts.append(_xml_text(source_text[last:start]))
        instruction_parts.append(f'<sub targ="{_xml_attr(target)}">{_xml_text(source_text[start:end])}</sub>')
        target_parts.append(source_text[last:start])
        target_parts.append(target)
        last = end
    instruction_parts.append(_xml_text(source_text[last:]))
    target_parts.append(source_text[last:])
    return "".join(instruction_parts), "".join(target_parts)


def _local_repair_window(prompt: str, source_text: str) -> tuple[int, int] | None:
    text = prompt.lower()
    spans: list[tuple[int, int]] = []
    old_new = _replace_pair(prompt)
    if old_new:
        old, _ = old_new
        all_occurrences = _contains(text, ["every instance", "all instance", "each instance", "每个", "所有"])
        spans = _source_spans_for_needle(source_text, old, all_occurrences=all_occurrences)
    else:
        insert = _insert_request(prompt)
        if insert:
            _insertion, anchor, placement = insert
            if anchor:
                spans.extend(_source_spans_for_needle(source_text, anchor))
            elif placement == "start":
                spans.append((0, min(len(source_text), 1)))
            elif placement == "end":
                spans.append((max(0, len(source_text) - 1), len(source_text)))
        if not spans and _contains(text, ["remove", "delete", "删除", "移除", "去掉", "删掉"]):
            quoted = _quoted_strings(prompt)
            if quoted:
                all_occurrences = _contains(text, ["all", "every", "each", "所有", "全部", "每个", "每一次"])
                spans.extend(_source_spans_for_needle(source_text, quoted[0], all_occurrences=all_occurrences))
        if not spans:
            swap = _swap_pair(prompt)
            if swap:
                first, second = swap
                spans.extend(_source_spans_for_needle(source_text, first))
                spans.extend(_source_spans_for_needle(source_text, second))
    if not spans:
        return None

    min_start = min(start for start, _ in spans)
    max_end = max(end for _, end in spans)
    cjk = bool(re.search(r"[\u4e00-\u9fff]", source_text))
    context = 24 if cjk else 80
    max_chars = 180 if cjk else 360
    start = max(0, min_start - context)
    end = min(len(source_text), max_end + context)
    if end - start > max_chars:
        start = max(0, min_start - context)
        end = min(len(source_text), start + max_chars)
        if end < max_end:
            end = min(len(source_text), max_end + context)
            start = max(0, end - max_chars)
    return start, end


def _quoted_strings(text: str) -> list[str]:
    return [value for value, _start, _end in _quoted_string_spans(text)]


def _quoted_string_spans(text: str) -> list[tuple[str, int, int]]:
    values = []
    for match in re.finditer(r"“([^”]+)”|'([^']+)'|\"([^\"]+)\"", text):
        values.append((next(group for group in match.groups() if group is not None), match.start(), match.end()))
    return values


def _first_sentence(text: str) -> str:
    match = re.match(r".+?[.!?。！？](?:\s+|$)", text)
    if match:
        return match.group(0).strip()
    return text.strip()


def _last_sentence(text: str) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    return parts[-1] if parts else text.strip()


def _last_word(text: str) -> str:
    words = re.findall(r"[\w'-]+|[\u4e00-\u9fff]", text)
    return words[-1] if words else text


def _wrap_first_occurrence(source_text: str, needle: str, render) -> str:
    if needle:
        spans = _source_spans_for_needle(source_text, needle)
        if spans:
            start, end = spans[0]
            matched = source_text[start:end]
            return source_text[:start] + render(matched) + source_text[end:]
    return render(source_text)


def _wrap_occurrences(source_text: str, needle: str, render, *, all_occurrences: bool = False) -> str:
    if not needle:
        return render(source_text)
    spans = _source_spans_for_needle(source_text, needle, all_occurrences=all_occurrences)
    if not spans:
        return render(source_text)
    output = source_text
    for start, end in reversed(spans):
        output = output[:start] + render(output[start:end]) + output[end:]
    return output


def _wrap_phrase_occurrences_with_trailing_marks(
    source_text: str,
    needle: str,
    render,
    *,
    all_occurrences: bool = False,
    marks: set[str] | None = None,
) -> str:
    if not needle:
        return render(source_text)
    marks = marks or set()
    spans = _source_spans_for_needle(source_text, needle, all_occurrences=all_occurrences)
    if not spans:
        return render(source_text)
    expanded: list[tuple[int, int]] = []
    for start, end in spans:
        index = end
        while index < len(source_text) and source_text[index].isspace():
            index += 1
        if index < len(source_text) and source_text[index] in marks:
            end = index + 1
        expanded.append((start, end))
    output = source_text
    for start, end in reversed(expanded):
        output = output[:start] + render(output[start:end]) + output[end:]
    return output


def _wrap_source_span(source_text: str, span: tuple[int, int], render) -> str:
    start, end = span
    return source_text[:start] + render(source_text[start:end]) + source_text[end:]


def _marked_span_from_prompt(
    prompt: str,
    source_text: str,
    *,
    allow_to_end: bool = False,
) -> tuple[int, int] | None:
    quoted = _quoted_strings(prompt)
    if not quoted:
        return None

    text = prompt.lower()
    marker = quoted[0]
    spans = _source_spans_for_needle(source_text, marker)
    if not spans:
        return None
    start, end = spans[0]
    if allow_to_end and _contains(
        text,
        [
            "to the end",
            "until the end",
            "from the word",
            "starting from",
            "start from",
            "从",
            "到结尾",
            "至结尾",
        ],
    ):
        return start, len(source_text)
    return start, end


TEXT_ALIASES: dict[str, list[str]] = {
    "he jiong": ["何炅"],
    "xie yilin": ["谢依霖"],
}


def _source_spans_for_needle(
    source_text: str,
    needle: str,
    *,
    all_occurrences: bool = False,
) -> list[tuple[int, int]]:
    for candidate in _needle_candidates(needle):
        spans = _exact_spans(source_text, candidate, all_occurrences=all_occurrences)
        if spans:
            return spans
        spans = _normalized_spans(source_text, candidate, all_occurrences=all_occurrences)
        if spans:
            return spans
        spans = _cjk_fuzzy_spans(source_text, candidate, all_occurrences=all_occurrences)
        if spans:
            return spans
        spans = _fuzzy_token_spans(source_text, candidate, all_occurrences=all_occurrences)
        if spans:
            return spans
    return []


def _needle_candidates(needle: str) -> list[str]:
    values = [needle]
    lowered = needle.strip().lower()
    values.extend(TEXT_ALIASES.get(lowered, []))
    values.extend(_number_aliases(needle))
    deduped = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _target_alias_for_source(source_text: str, value: str) -> str:
    """Use source-script entity aliases when the source transcript is not English."""

    lowered = value.strip().lower()
    aliases = TEXT_ALIASES.get(lowered, [])
    if not aliases or not re.search(r"[\u4e00-\u9fff]", source_text):
        return value
    for alias in aliases:
        if re.search(r"[\u4e00-\u9fff]", alias):
            return alias
    return value


def _exact_spans(source_text: str, needle: str, *, all_occurrences: bool) -> list[tuple[int, int]]:
    matches = list(re.finditer(re.escape(needle), source_text, flags=re.IGNORECASE))
    if not all_occurrences:
        matches = matches[:1]
    return [(match.start(), match.end()) for match in matches]


def _normalized_spans(source_text: str, needle: str, *, all_occurrences: bool) -> list[tuple[int, int]]:
    norm_source, mapping = _normalize_for_match(source_text)
    norm_needle, _ = _normalize_for_match(needle)
    if not norm_source or not norm_needle:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = norm_source.find(norm_needle, start)
        if idx < 0:
            break
        source_start = mapping[idx]
        source_end = mapping[idx + len(norm_needle) - 1] + 1
        spans.append((source_start, source_end))
        if not all_occurrences:
            break
        start = idx + len(norm_needle)
    return spans


_CJK_MATCH_CHAR_ALIASES = {
    "地": "的",
    "得": "的",
    "唉": "哎",
    "欸": "哎",
    "诶": "哎",
    "嗳": "哎",
    "啦": "了",
    "喽": "了",
    "囖": "了",
    "囉": "了",
    "即": "及",
    "姓": "性",
    "命": "名",
    "欲": "狱",
}

_CJK_FUZZY_STOP_CHARS = set("的一是在不了和有就都人中上个们")


def _cjk_fuzzy_spans(source_text: str, needle: str, *, all_occurrences: bool) -> list[tuple[int, int]]:
    """Recover short Chinese anchors when ASR uses a near homophone or variant."""

    norm_needle, _ = _normalize_for_match(needle)
    if not norm_needle or not re.search(r"[\u4e00-\u9fff]", norm_needle):
        return []
    if len(norm_needle) < 2:
        return []
    norm_source, mapping = _normalize_for_match(source_text)
    width = len(norm_needle)
    if width > len(norm_source):
        return []

    needle_chars = set(norm_needle)
    max_distance = 1 if width <= 4 else max(1, round(width * 0.25))
    min_ratio = 0.66 if width >= 3 else 0.5
    candidates: list[tuple[float, int, int]] = []
    for index in range(0, len(norm_source) - width + 1):
        window = norm_source[index : index + width]
        shared = needle_chars.intersection(window) - _CJK_FUZZY_STOP_CHARS
        if not shared:
            continue
        distance = _bounded_edit_distance(norm_needle, window, max_distance=max_distance)
        ratio = difflib.SequenceMatcher(None, norm_needle, window).ratio()
        if distance > max_distance and ratio < min_ratio:
            continue
        start = mapping[index]
        end = mapping[index + width - 1] + 1
        candidates.append((ratio - 0.04 * distance, start, end))

    if not candidates:
        return []
    if all_occurrences:
        candidates.sort(key=lambda item: (item[1], item[2]))
        spans: list[tuple[int, int]] = []
        last_end = -1
        for _score, start, end in candidates:
            if start < last_end:
                continue
            spans.append((start, end))
            last_end = end
        return spans

    best_score = max(score for score, _start, _end in candidates)
    best = [(start, end) for score, start, end in candidates if score == best_score]
    best.sort(key=lambda item: (item[0], item[1]))
    return best[:1]


def _fuzzy_token_spans(source_text: str, needle: str, *, all_occurrences: bool) -> list[tuple[int, int]]:
    norm_needle, _ = _normalize_for_match(needle)
    if not norm_needle or len(norm_needle) < 5 or re.search(r"[\u4e00-\u9fff]", norm_needle):
        return []
    needle_tokens = _token_spans(needle)
    source_tokens = _token_spans(source_text)
    if not needle_tokens or not source_tokens:
        return []
    token_count = len(needle_tokens)
    if token_count > len(source_tokens):
        return []

    spans: list[tuple[int, int]] = []
    for index in range(0, len(source_tokens) - token_count + 1):
        window = source_tokens[index : index + token_count]
        window_text = "".join(item[0] for item in window)
        norm_window, _ = _normalize_for_match(window_text)
        if not norm_window:
            continue
        ratio = difflib.SequenceMatcher(None, norm_needle, norm_window).ratio()
        distance = _bounded_edit_distance(norm_needle, norm_window, max_distance=1)
        if ratio >= 0.86 or distance <= 1:
            spans.append((window[0][1], window[-1][2]))
            if not all_occurrences:
                break
    return spans


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"[\w'-]+", text)]


def _bounded_edit_distance(left: str, right: str, *, max_distance: int) -> int:
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _normalize_for_match(text: str) -> tuple[str, list[int]]:
    chars = []
    mapping = []
    for idx, char in enumerate(text.lower()):
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            chars.append(_CJK_MATCH_CHAR_ALIASES.get(char, char))
            mapping.append(idx)
    return "".join(chars), mapping


ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _number_aliases(text: str) -> list[str]:
    aliases: list[str] = []
    for match in re.finditer(r"\b\d{1,6}\b", text.replace(",", "")):
        value = int(match.group(0))
        spoken = _english_number(value)
        if spoken:
            aliases.append(spoken)
            aliases.append(f"{spoken} dollars")
    return aliases


def _english_number(value: int) -> str | None:
    if value < 20:
        return ONES.get(value)
    if value < 100:
        tens = value // 10 * 10
        rest = value % 10
        return TENS[tens] if rest == 0 else f"{TENS[tens]} {ONES[rest]}"
    if value < 1000:
        hundreds = value // 100
        rest = value % 100
        prefix = f"{ONES[hundreds]} hundred"
        return prefix if rest == 0 else f"{prefix} {_english_number(rest)}"
    if value < 1000000:
        thousands = value // 1000
        rest = value % 1000
        prefix = f"{_english_number(thousands)} thousand"
        return prefix if rest == 0 else f"{prefix} {_english_number(rest)}"
    return None


def _pause_instruction_st(prompt: str, source_text: str) -> str:
    tag = f'<pause act="ins" level="{_pause_level(prompt)}"/>'
    for marker in _quoted_strings(prompt):
        idx = source_text.find(marker)
        if idx >= 0:
            end = idx + len(marker)
            return source_text[:end] + tag + source_text[end:]
    after_match = re.search(
        r"(?i)\bafter\s+(?:the\s+)?(?P<anchor>[^,;.!?]+)",
        prompt,
    )
    if after_match:
        stopwords = {
            "a",
            "an",
            "and",
            "be",
            "declared",
            "first",
            "is",
            "moment",
            "sentence",
            "the",
            "thought",
        }
        anchor_words = re.findall(r"[A-Za-z0-9']+", after_match.group("anchor"))
        for word in reversed(anchor_words):
            if len(word) < 3 or word.casefold() in stopwords:
                continue
            match = re.search(rf"(?i)\b{re.escape(word)}\b", source_text)
            if match and match.end() < len(source_text):
                return source_text[: match.end()] + tag + source_text[match.end() :]
    if _contains(
        prompt,
        [
            "after the first thought",
            "after first thought",
            "after the first sentence",
            "after first sentence",
            "第一句话后",
            "第一句后",
        ],
    ):
        match = re.search(r"[.!?。！？]", source_text)
        if match and match.end() < len(source_text):
            return source_text[: match.end()] + tag + source_text[match.end() :]
    for punct in [",", "，", ";", "；"]:
        idx = source_text.find(punct)
        if 0 <= idx < len(source_text) - 1:
            return source_text[: idx + 1] + tag + source_text[idx + 1 :]
    parts = source_text.split(" ", 1)
    if len(parts) == 2:
        return parts[0] + tag + " " + parts[1]
    idx = max(1, len(source_text) // 2)
    return source_text[:idx] + tag + source_text[idx:]


def _contains(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def _contains_word(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(word.lower())}\b", lower) for word in words)


def _wants_pitch(prompt: str) -> bool:
    return _contains(
        prompt,
        [
            "音高",
            "升调",
            "降调",
            "半音",
            "pitch",
            "semitone",
            "higher pitch",
            "lower pitch",
            "intonation",
            "rising intonation",
            "falling intonation",
        ],
    )


def _wants_question_to_declarative_tone(prompt: str) -> bool:
    text = prompt.lower()
    return _contains(
        text,
        [
            "questioning tone",
            "questioning intonation",
            "question tone",
            "questioning",
            "interrogative intonation",
            "interrogative tone",
            "rising intonation",
            "question-like",
        ],
    ) and _contains(
        text,
        ["declarative tone", "declarative", "falling intonation", "statement tone"],
    )


def _wants_questioning_tone_target(prompt: str) -> bool:
    text = prompt.lower()
    if _wants_question_to_declarative_tone(prompt):
        return False
    has_question_target = _contains(
        text,
        [
            "questioning tone",
            "questioning intonation",
            "question tone",
            "interrogative intonation",
            "interrogative tone",
            "rising intonation",
            "rising tone",
            "question-like",
            "疑问语气",
            "疑问语调",
            "升调",
        ],
    )
    if not has_question_target:
        return False
    return _contains(text, ["change", "make", "turn", "convert", "to", "into", "改成", "变成", "转换"])


def _wants_emphasis(prompt: str) -> bool:
    return _contains(prompt, ["emphasis", "stress", "stressed", "emphasize", "emphasise", "重音", "重读", "强调"])


def _wants_rate(prompt: str) -> bool:
    return _contains(
        prompt,
        [
            "语速",
            "加速",
            "减速",
            "放慢",
            "变快",
            "变慢",
            "tempo",
            "speech rate",
            "speaking rate",
            "pace",
            "speed up",
            "slow down",
            "faster",
            "slower",
        ],
    )


def _wants_emotion(prompt: str) -> bool:
    if _wants_question_to_declarative_tone(prompt):
        return True
    if _wants_questioning_tone_target(prompt):
        return False
    if _contains(
        prompt,
        [
            "情绪",
            "语气",
            "风格",
            "开心",
            "高兴",
            "难过",
            "悲伤",
            "生气",
            "愤怒",
            "害怕",
            "恐惧",
            "厌恶",
            "忧郁",
            "惊讶",
            "平静",
            "玩闹",
            "玩闹感",
            "俏皮",
            "调皮",
            "嬉戏",
            "calm",
            "neutral",
            "urgent",
            "cheerful",
            "playful",
            "teasing",
            "mischievous",
            "lively",
            "vibrant",
            "energetic",
            "soothing",
            "doting",
            "affectionate",
            "loving",
            "tender",
            "caring",
            "fond",
            "hushed",
            "reflective",
            "introspective",
            "subdued",
            "internal monologue",
            "internal thought",
            "expressive",
            "emotional",
            "emotion",
            "happy",
            "sad",
            "angry",
            "melancholic",
            "melancholy",
            "surprised",
            "excited",
            "depressed",
            "reluctant",
            "frightened",
            "afraid",
            "fearful",
            "panic",
            "panicked",
            "tremor",
            "tremors",
            "breathy",
            "breathiness",
            "shout",
            "shouting",
            "yell",
            "yelling",
            "whisper",
            "whispering",
            "whispered",
            "大喊",
            "喊叫",
            "耳语",
            "低语",
        ],
    ):
        return True
    if re.search(r"\banger\b", prompt.lower()):
        return True
    if _tone_or_style_is_only_preserved(prompt):
        return False
    return _contains_word(prompt, ["mood", "tone", "style"])


def _tone_or_style_is_only_preserved(text: str) -> bool:
    return _contains(
        text,
        [
            "same voice and tone",
            "same tone",
            "original tone",
            "matching the same voice and tone",
            "match the same voice and tone",
            "tone unchanged",
            "same style",
            "original style",
            "style unchanged",
        ],
    )


def _wants_pause(prompt: str) -> bool:
    return _contains(
        prompt,
        [
            "停顿",
            "暂停",
            "停一下",
            "pause",
            "silence",
            "short break",
            "brief break",
            "add a break",
            "a moment after",
        ],
    )


def _wants_volume(prompt: str) -> bool:
    return _contains(
        prompt,
        [
            "音量",
            "响度",
            "volume",
            "loudness",
            "normalize",
            "normalise",
            "enhance vocal",
            "increase the volume",
            "boost the volume",
            "boost voice",
            "amplify",
            "strengthen",
            "weaken",
            "attenuate",
            "suppress",
            "turn down",
            "reduce volume",
            "reduce the volume",
            "reduce voice",
            "lower volume",
            "louder",
            "quieter",
            "softer",
            "softly pronounce",
            "speak softly",
        ],
    )


def _speaker_profile_params(prompt: str) -> dict[str, Any] | None:
    text = prompt.lower()
    if _contains(text, ["reference", "参考音频", "参考声音", "参考说话人"]):
        return None
    if _contains(text, ["female voice", "女声", "女人声", "女性声音"]):
        return {"profile": "female_voice", "semitones": 5, "target_db": -19.0}
    if _contains(text, ["child voice", "kid voice", "little boy", "little girl", "童声", "儿童声", "小孩声"]):
        return {"profile": "child_voice", "semitones": 8, "target_db": -18.0}
    if _contains(text, ["adult voice", "sound like an adult", "like an adult", "adult speaker"]):
        return {"profile": "adult_voice", "semitones": -5, "target_db": -20.5}
    if _contains(text, ["deep male", "male voice", "boy's", "男声", "男人声", "男性声音"]):
        semitones = -6 if _contains(text, ["deep", "低沉", "浑厚"]) else -4
        return {"profile": "male_voice", "semitones": semitones, "target_db": -20.5}
    if _contains(text, ["elderly", "old voice", "older", "老年", "年老", "苍老"]):
        return {"profile": "elderly_voice", "semitones": -3, "target_db": -22.0}
    if _contains(text, ["younger", "young voice", "更年轻", "年轻"]):
        return {"profile": "younger_voice", "semitones": 4, "target_db": -19.0}
    if _contains(text, ["deeper", "lower voice", "voice lower", "低一点", "更低"]):
        return {"profile": "lower_voice", "semitones": -5, "target_db": -20.5}
    if _contains(text, ["higher voice", "voice higher", "更高", "尖一点"]):
        return {"profile": "higher_voice", "semitones": 5, "target_db": -19.0}
    return None


def _pitch_semitones(prompt: str) -> int:
    text = prompt.lower()
    match = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:半音|semitones?|st)", text)
    if match:
        value = int(round(float(match.group(1))))
        return max(-12, min(12, value))
    if _contains(text, ["半个八度", "half octave"]):
        return -6 if _contains(text, ["降低", "低", "降", "lower", "down"]) else 6
    if _contains(text, ["降低", "低", "降调", "lower", "down"]):
        return -8
    if _contains(text, ["升高", "高", "升调", "higher", "up", "提升"]):
        return 8
    return 8


def _rate_factor(prompt: str) -> float:
    text = prompt.lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:x|倍)", text)
    if match:
        value = float(match.group(1))
        return min(3.0, max(0.3, value))
    if _contains(text, ["slow", "慢", "减速", "放慢", "slower"]):
        return 0.8
    if _contains(text, ["fast", "快", "加速", "faster", "speed up"]):
        return 1.2
    return 1.0


def _emotion_tag(prompt: str) -> str:
    text = prompt.lower()
    if _contains(text, ["declarative", "statement tone", "neutral", "flat statement", "matter-of-fact"]):
        return "calm"
    if _contains(text, ["生气", "愤怒", "angry"]) or re.search(r"\banger\b", text):
        return "angry"
    if _contains(text, ["惊讶", "surprised", "shocked"]):
        return "surprised"
    if _contains(text, ["悲伤", "难过", "sad"]):
        return "sad"
    if _contains(text, ["害怕", "恐惧", "afraid", "fear", "frightened", "fearful", "panic", "panicked"]):
        return "afraid"
    if _contains(text, ["depressed", "reluctant", "melancholic"]):
        return "melancholic"
    if _contains(
        text,
        [
            "hushed",
            "reflective",
            "introspective",
            "subdued",
            "internal monologue",
            "inner monologue",
            "internal thought",
            "内心独白",
        ],
    ):
        return "melancholic"
    if _contains(text, ["平静", "舒缓", "calm", "neutral", "soothing"]):
        return "calm"
    if _contains(
        text,
        [
            "玩闹",
            "玩闹感",
            "俏皮",
            "调皮",
            "嬉戏",
            "playful",
            "teasing",
            "mischievous",
            "doting",
            "affectionate",
            "loving",
            "tender",
            "caring",
            "fond",
            "lively",
            "vibrant",
            "energetic",
        ],
    ):
        return "happy"
    return "happy"


def _target_vocal_effort_style(prompt: str) -> str | None:
    text = prompt.lower()
    target_text = text
    for marker in [" into ", " to ", " as ", " in a ", " with a "]:
        if marker in text:
            target_text = text.rsplit(marker, 1)[-1]
            break
    if _contains(target_text, ["whisper", "whispering", "whispered", "耳语", "低语", "悄声"]):
        return "whispering voice"
    if _contains(target_text, ["shout", "shouting", "yell", "yelling", "大喊", "喊叫"]):
        return "shouting voice"
    if _contains(text, ["whisper style", "whispering style", "whispered style", "耳语风格", "低语风格"]):
        return "whispering voice"
    if _contains(text, ["shout style", "shouting style", "yelling style", "大喊风格", "喊叫风格"]):
        return "shouting voice"
    return None


def _emotion_level(prompt: str) -> int:
    if _contains(prompt, ["declarative", "statement tone", "flat statement", "matter-of-fact"]):
        return 1
    match = re.search(r"level\s*=?\s*([1-3])|([1-3])\s*级", prompt.lower())
    if match:
        return int(next(group for group in match.groups() if group))
    return 2


def _pause_level(prompt: str) -> int:
    text = prompt.lower()
    if _contains(text, ["长", "long"]):
        return 3
    if _contains(text, ["短", "short"]):
        return 1
    return 2


def _target_db(prompt: str) -> float:
    match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:db|dB|分贝)", prompt)
    if match:
        return min(0.0, max(-60.0, float(match.group(1))))
    lowered = prompt.lower()
    if _contains(lowered, ["mute", "silence", "静音"]):
        return -60.0
    if _contains(lowered, ["decrease", "lower", "reduce", "weaken", "attenuate", "suppress", "turn down", "降低", "减小", "减弱"]):
        return -25.0
    return -18.0


def _language_hint(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _looks_long_for_f5(text: str) -> bool:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    return len(text) > 220 or cjk_chars > 120


def _xml_attr(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
