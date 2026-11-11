"""Public edit scopes and fallible, version-bound regional listening evidence.

Scopes express requested changes, not a tool's claimed edit mask. Reported
timestamps are observations; interval validity does not certify grounding.
"""

from typing import Literal

from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel

RegionFocus = Literal[
    "edited_region", "preserved_region", "boundary_transition", "audio_quality"
]
PRIMARY_FOCI = {"edited_region", "preserved_region", "boundary_transition"}
DIMENSIONS = {
    "edited_region": "goal", "preserved_region": "preservation",
    "boundary_transition": "perceptual_quality", "audio_quality": "perceptual_quality",
}


class RegionScope(FrozenEvaluationModel):
    focus: RegionFocus
    source_audio_refs: tuple[str, ...] = Field(min_length=1)
    authorized_change: str = Field(min_length=4, max_length=800)
    region_description: str = Field(min_length=4, max_length=1000)
    comparison_basis: str = Field(min_length=4, max_length=1000)


class RegionCheck(RegionScope):
    """Internal projection of a criterion and its scope, not a second authored list."""

    criterion_id: str = Field(min_length=1)


class RegionExemption(FrozenEvaluationModel):
    focus: Literal["preserved_region", "boundary_transition"]
    reason: str = Field(min_length=12, max_length=800)


class CheckedAudioRegion(FrozenEvaluationModel):
    audio_ref: str = Field(min_length=1)
    start_sec: float = Field(ge=0, allow_inf_nan=False)
    end_sec: float = Field(gt=0, allow_inf_nan=False)
    observation: str = Field(min_length=4, max_length=1000)
    # Only a boundary question uses this field; its window must include both sides.
    boundary_sec: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("checked region must have positive extent")
        if self.boundary_sec is not None and not self.start_sec < self.boundary_sec < self.end_sec:
            raise ValueError("boundary listening must include both sides of the transition")
        return self


class RegionalObservation(FrozenEvaluationModel):
    criterion_id: str = Field(min_length=1)
    focus: RegionFocus
    checked_regions: tuple[CheckedAudioRegion, ...] = Field(default=(), max_length=16)
    unresolved_region_mapping: tuple[str, ...] = Field(default=(), max_length=8)


def validate_region_design(design):
    criteria = {c.criterion_id: c for c in design.criteria}
    ids = [c.criterion_id for c in design.region_checks]
    if len(set(ids)) != len(ids) or set(ids) != set(criteria):
        raise ValueError("regional design must bind every criterion exactly once")
    for check in design.region_checks:
        if criteria[check.criterion_id].dimension != DIMENSIONS[check.focus]:
            raise ValueError("regional focus and utility dimension disagree")
        if len(set(check.source_audio_refs)) != len(check.source_audio_refs):
            raise ValueError("regional source roles must not repeat")
    present = {c.focus for c in design.region_checks}
    exemptions = [e.focus for e in design.region_exemptions]
    if len(set(exemptions)) != len(exemptions) or present.intersection(exemptions):
        raise ValueError("a regional focus cannot be checked and exempted")
    if PRIMARY_FOCI - present - set(exemptions):
        raise ValueError("edited region, preserved region and boundaries need coverage or an exemption")
    return design


def bind_region_design(design, scenario):
    available = {a.asset_ref for a in scenario.audio_assets}
    if any(set(c.source_audio_refs) - available for c in design.region_checks):
        raise ValueError("regional design cites an unavailable public source")
    return design


def validate_regional_observation(observation, answer, check, inventory):
    assets = {a["audio_ref"]: a for a in inventory}
    candidate = {a["audio_ref"] for a in inventory if a["role"] == "candidate"}
    for region in observation.checked_regions:
        asset = assets.get(region.audio_ref)
        if asset is None or region.end_sec > asset["duration_sec"] + 1e-6:
            raise ValueError("regional evidence lies outside its bound audio")
        if region.audio_ref not in answer.evidence_ids:
            raise ValueError("regional observation lacks its audio citation")
    if answer.grade is None:
        return observation
    if observation.unresolved_region_mapping:
        raise ValueError("unresolved region correspondence cannot support a determinate grade")
    heard = {r.audio_ref for r in observation.checked_regions}
    if not candidate.intersection(heard):
        raise ValueError("regional grade needs a heard candidate interval")
    if check.focus == "preserved_region" and not set(check.source_audio_refs).issubset(heard):
        raise ValueError("preservation needs corresponding baseline and candidate intervals")
    if check.focus == "boundary_transition" and not any(
        r.audio_ref in candidate and r.boundary_sec is not None for r in observation.checked_regions
    ):
        raise ValueError("boundary grade needs a candidate window across the transition")
    return observation


REGION_REQUIREMENTS_PROMPT = """1 edited_region：用户授权的每处修改是否完成，修改对象/内容/属性和范围是否正确；
2 preserved_region：授权范围以外的内容及同一时段未授权改变的说话人/背景成分是否保留；
3 boundary_transition：每个局部编辑与保留声音的接点是否有截断音素、爆音、异常停顿、
突变的音量/音色或重复衔接。不能用整段自然度题代替接点检查。
audio_quality 可以另作保障题，按指令相关的清晰度和可用性提问，不重复前三类。
保留说话人身份不能替代保留未编辑的词/声音；局部改词至少要单独检查其余内容。
根据实际任务分别列出未编辑内容、声音身份或背景成分的必要保留条件，不能只选一个容易的问题。
每道 criteria 题目直接包含一个 region 对象；不要另列 region_checks 或重复题号。
region 给出 focus、公开 source_audio_refs、authorized_change、
region_description 和 comparison_basis。用可听词、事件、说话人、先后及版本关系定位；
没有可靠时间证据不得编造秒数。题目依公开要求划定授权范围，不能依赖未来工具声称的修改范围。
局部替换、插入、删除允许合理时长变化，比较对应内容而非机械比较相同秒数。
删除后核查新相邻内容接点；多音频逐个绑定，重复目标逐处覆盖；恢复旧版本也要满足
仍有效的保留要求。最终音频和源文件不是同一时间轴；不要指定未获得的中间版本为附件。
全局处理没有局部接点时，可以为 boundary_transition 写 region_exemptions 并说明依据；
不存在保留区域/属性时才允许 preserved_region 豁免。不确定边界在哪里不是不适用。
豁免不得计作通过；不得将源音频已有缺陷自动归罪于编辑。三个主要方面的失败不能
被其他问题高分抵消。每题只检验一个条件，保持题量精简；所有题仍需公开指令蕴含审查。
"""
REGION_DESIGN_PROMPT = "本次使用 edit-regions/v2 协议，三个主要问题必须分别覆盖：\n" + REGION_REQUIREMENTS_PROMPT

REGION_JUDGE_PROMPT = """本题的 region_check 指明授权变化、评测区域和比较基准。
先按实际声音识别对应片段，再判断 criterion.question；计划或工具声称不能定义合法范围。
edited_region 核查每个目标及正确范围，不只确认产物中出现目标词。
preserved_region 比较授权范围之外及重叠的非目标成分，授权词汇/风格变化本身不扣保留分。
时长发生合理变化时要对照对应内容，不把后续整体时间偏移误判为内容被改。
boundary_transition 核听接点两侧的上下文，分别关注进出接点；删除后检查新邻接处，
不得仅因整句清晰就给接点满分。用户明确要求的停顿、静音或突变不是编辑伪影。
不要把实际音频不存在的接点当真，无法定位或无法核实对应关系就 unknown。
在 checked_regions 中记录实际核听的附件audio_ref、该附件自己的start_sec/end_sec和具体观察。
边界题的每个 checked_regions 元素内部还要填写 boundary_sec，顶层不得填写它。
必须满足 start_sec < boundary_sec < end_sec；窗口必须跨过接点，而不是仅记录接点瞬间。
禁止 start_sec=end_sec，也不要把只有静音内部的窗口当作接点两侧。每个源/候选各自记录
有上下文的实际窗口；无法支持这样一个窗口就 unknown，而不是无证据填写satisfied。
保留题记录全部比较基准与候选的对应窗口。不得把源坐标直接抄到变长后的候选。
秒数是可错的观察，不是工具真值；不要编造精度。无法确定映射时填
unresolved_region_mapping，grade=null。确定分数必须有对应候选声音证据。
这些区域只支撑本题，不能据局部核听宣称整份音频所有其他要求已通过。
"""


class RegionCoverageJudgment(FrozenEvaluationModel):
    focus: Literal["edited_region", "preserved_region", "boundary_transition"]
    status: Literal["covered", "missing", "not_applicable"]
    criterion_ids: tuple[str, ...] = ()
    required_conditions: tuple[str, ...] = Field(min_length=1, max_length=8)
    uncovered_conditions: tuple[str, ...] = Field(default=(), max_length=8)
    explanation: str = Field(min_length=8, max_length=1200)


REGION_COVERAGE_PROMPT = """为三个主要方面各填一项 coverage_checks。先独立从公开指令推断
本方面必须满足哪些条件，再列实际覆盖它们的criterion_ids；缺项记missing并写明缺什么。
preserved_region 必须考虑未编辑内容，不能只因有说话人身份题就认为保留要求齐全。
边界检查要覆盖局部处理的进入/退出或删除后的新邻接处；整体清晰度题不代替接点题。
若无局部接点，核实公开任务确实支持对应region_exemptions后才记not_applicable。
不适用不能用来处理未知位置。逐项审查region描述和比较版本是否正确；不需要猜秒数。
不能因为都有focus标签就认证完整覆盖。对看不到或尚未产生的音频不作实际质量认证。
"""


def regional_coverage_problems(design, checks):
    foci = [c.focus for c in checks]
    if len(foci) != 3 or set(foci) != PRIMARY_FOCI:
        return ["coverage audit must account for each primary aspect exactly once"]
    exemptions = {e.focus for e in design.region_exemptions}
    known_ids = {c.criterion_id for c in design.criteria}
    problems = []
    for check in checks:
        dedicated = {c.criterion_id for c in design.region_checks if c.focus == check.focus}
        cited = set(check.criterion_ids)
        # A complementary quality question may support preservation too. Require
        # a dedicated check without forcing the auditor to hide valid cross-support.
        if (cited - known_ids or check.status == "missing" or check.uncovered_conditions
                or (check.status == "covered" and (not cited.intersection(dedicated) or check.focus in exemptions))
                or (check.status == "not_applicable" and (check.focus not in exemptions or check.criterion_ids))):
            problems.append(check.model_dump(mode="json"))
    return problems

REGION_GUIDANCE_PROMPT = """本轮附有三类区域核查。分别记录：目标修改缺什么、非目标内容
损伤在哪里、哪个接点过渡异常；不要把三类错误合成一句音质差。
regional_observations 中的时间是评委可错的定位观察，不能作为认证真值直接代填工具参数。
定位/对应关系未明时先核听；有依据的产物错误再回查相关参数、范围、依赖和可用工具能力。
推理正确就保留，不要求执行前的Reasoner听评后来才生成的音频；Planner根据证据自行
重审局部处理与衔接方式，不指定固定工具、固定淡化毫秒数或代写计划。
正确的其他目标、非目标内容和已平滑接点作为保留条件；一般自然度高分不能抵消这些损伤。
"""
