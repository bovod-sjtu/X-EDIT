"""Version-bound two-question audio checks outside the Forward execution path."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel


class HeardDescription(FrozenEvaluationModel):
    heard: str = Field(min_length=1, max_length=1600)
    uncertain: tuple[str, ...] = ()


class LocalEventCheck(FrozenEvaluationModel):
    presence: Literal["present", "absent", "unknown"]
    observed_description: str = Field(min_length=1, max_length=1600)
    local_intervals_sec: tuple[tuple[float, float], ...] = ()
    precision_limit: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def valid_intervals(self):
        if self.presence != "present" and self.local_intervals_sec:
            raise ValueError("absent/unknown cannot assert event intervals")
        if any(
            not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a
            for a, b in self.local_intervals_sec
        ):
            raise ValueError("invalid local interval")
        return self


DESCRIBE_PROMPT = """独立描述附件片段中实际听到的声音或语音，不猜测片段外内容。
不要推测任务目的。听不清的部分明确说明。简洁中文回答，严格返回schema。"""
CHECK_PROMPT = """核查附件片段中的一个待验证声称。声称可能正确、错误或部分正确。
先前回答只是另一份观察，不是答案。判断该事件是否实际出现；如果能更精确定位，给出
相对本附件起点的local_intervals_sec。时间边界不能超过片段时长；不确定不要编造精度。
absent只适用于所听片段，不能据此断言整条音频没有事件。严格返回schema。"""

WINDOW_COORDINATE_PROMPT = """attached_window区分原文件时间与附件局部时间。
声称中提到的区间可能使用原文件坐标；附件已裁出该区间并从局部0秒播放。
不能将原文件秒数直接与附件时长比较并声称越界。原文件坐标减去parent_start_sec
才是本附件局部坐标；返回的local_intervals_sec只能用附件局部坐标。
裁片合法性已由程序核对，你负责听到什么；无法从声音区分合成与保留的技术来源时，
不要猜生成过程，描述可听的词/声音及无法确认的接点，保留unknown。"""


def recheck_observation(check, *, source: Path, output: Path, call):
    """call(prompt, payload, audio_paths, call_dir) records exactly one request."""
    import soundfile as sf

    source = Path(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    if before != check.audio_sha256:
        raise ValueError("observation source/version changed")
    info = sf.info(source)
    if check.end_sec > info.frames / info.samplerate:
        raise ValueError("observation interval exceeds actual audio")
    start = math.floor(check.start_sec * info.samplerate)
    stop = math.ceil(check.end_sec * info.samplerate)
    output.mkdir(parents=True, exist_ok=False)
    data, rate = sf.read(source, start=start, stop=stop, always_2d=True, dtype="float32")
    clip = output / "observed_clip.wav"
    sf.write(clip, data, rate, subtype="FLOAT")
    actual_start, duration = start / rate, len(data) / rate
    description = HeardDescription.model_validate(
        call(
            DESCRIBE_PROMPT,
            {
                "schema": HeardDescription.model_json_schema(),
                "duration_sec": duration,
            },
            [str(clip)],
            output / "blind_description",
        )
    )
    finding = LocalEventCheck.model_validate(
        call(
            CHECK_PROMPT + "\n" + WINDOW_COORDINATE_PROMPT,
            {
                "schema": LocalEventCheck.model_json_schema(),
                "duration_sec": duration,
                "attached_window": {
                    "parent_audio_ref": check.audio_ref,
                    "parent_start_sec": actual_start,
                    "parent_end_sec": stop / rate,
                    "local_start_sec": 0.0,
                    "local_end_sec": duration,
                },
                "claim_to_test": check.event_claim,
                "prior_independent_description": description.model_dump(mode="json"),
            },
            [str(clip)],
            output / "claim_check",
        )
    )
    if any(end > duration for _, end in finding.local_intervals_sec):
        raise ValueError("judge interval exceeds the actual attached clip")
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError("source changed during observation")
    receipt = {
        "check": check.model_dump(mode="json"),
        "source_sha256": before,
        "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "actual_source_interval_sec": [actual_start, stop / rate],
        "blind_description": description.model_dump(mode="json"),
        "claim_check": finding.model_dump(mode="json"),
        "reported_source_intervals_sec": [
            [a + actual_start, b + actual_start] for a, b in finding.local_intervals_sec
        ],
        "scope": "this version and clip only",
        "acoustic_truth_certified": False,
        "direct_runtime_evidence": False,
    }
    with (output / "observation.json").open("x") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    return receipt
