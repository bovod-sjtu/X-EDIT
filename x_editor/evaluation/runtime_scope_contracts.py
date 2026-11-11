"""Fallible scope proposals for the opt-in, post-execution audio reviewer.

The model proposes seconds. The caller binds actual versions and sample frames;
neither a tool's edit mask nor a valid coordinate is semantic ground truth.
"""

from typing import Literal

from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel


class ScopeWindow(FrozenEvaluationModel):
    audio_ref: str = Field(min_length=1)
    start_sec: float = Field(ge=0, allow_inf_nan=False)
    end_sec: float = Field(gt=0, allow_inf_nan=False)
    boundary_sec: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("empty scope window")
        if self.boundary_sec is not None and not self.start_sec < self.boundary_sec < self.end_sec:
            raise ValueError("boundary context must include both sides")
        return self


class ScopeProbe(FrozenEvaluationModel):
    signal_measurement: Literal['none', 'aligned_pcm_comparison'] = 'none'
    question: str = Field(min_length=8, max_length=1400)
    explanation: str = Field(min_length=8, max_length=1200)
    candidate: ScopeWindow
    references: tuple[ScopeWindow, ...] = Field(default=(), max_length=8)


class CriterionScope(FrozenEvaluationModel):
    criterion_id: str = Field(min_length=1)
    probes: tuple[ScopeProbe, ...] = Field(default=(), max_length=12)
    missing_evidence: tuple[str, ...] = ()


class ScopeObservationRequest(FrozenEvaluationModel):
    method: Literal["listen", "speech_grounding"] = "listen"
    criterion_ids: tuple[str, ...] = Field(min_length=1)
    audio_ref: str = Field(min_length=1)
    event_claim: str = Field(min_length=1, max_length=800)
    start_sec: float = Field(default=0, ge=0, allow_inf_nan=False)
    end_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class RuntimeScopePlan(FrozenEvaluationModel):
    criteria: tuple[CriterionScope, ...] = Field(min_length=1, max_length=24)
    observation_requests: tuple[ScopeObservationRequest, ...] = Field(default=(), max_length=6)


class CriterionScopeCoverage(FrozenEvaluationModel):
    criterion_id: str = Field(min_length=1)
    supported_probe_indices: tuple[int, ...] = ()
    coverage: Literal["complete", "incomplete", "unknown"]
    uncovered_conditions: tuple[str, ...] = ()
    # The model response budget bounds size. A verbose rationale must not
    # invalidate otherwise well-formed decisions for every other criterion.
    # Keep its full text for audit; probe validity and coverage stay separate.
    explanation: str = Field(min_length=8)


class RuntimeScopeCoverage(FrozenEvaluationModel):
    criteria: tuple[CriterionScopeCoverage, ...] = Field(min_length=1, max_length=24)
    missing_design_requirements: tuple[str, ...] = ()
    observation_requests: tuple[ScopeObservationRequest, ...] = Field(default=(), max_length=6)


SCOPE_PLAN_PROMPT = """你是多模态评测的范围准备端，不是执行Planner，也不替音频评委评分。
根据公开指令、已冻结的检查题、本次真实感知/工具记录和新增核听，准备实际裁片计划。
语义评委检查推理、参数和合同；这里为多模评委准备可听的局部事实问题。
逐题列出全部必要窗口，分别检查授权编辑、未编辑内容与应保留属性、每个编辑接点。
保留区的词/背景事件是否遗漏和说话人/响度是否保留是不同问题；不能只问身份就认证保留。
preserved_region的probe.question应明确要求对照相应源/参考与候选，不只问候选里
某个期望词是否存在；参考版本选择依据仍须符合公开要求。
左右两侧、多处目标、多音频对应、不同版本应逐处考虑。每个问题只问一个可听条件，
但同一条件有多个区段就列多个probe，不能只抽一个代表就称完整覆盖。
boundary_transition是实际编辑与保留内容的交界，需要两侧声音。文件开头/结尾没有
外侧声音，不自动变成内部接点；起止处伪影可作为适用的音质题，不虚构越界窗口。
此类probe必须填写candidate.boundary_sec，且位于start_sec与end_sec之间；
尚不知道接点就将该题probes留空、说明missing_evidence并申请定位，不猜测接点或省略字段。
source与candidate使用各自时间轴。删除/插入/变速后不能复制源时间戳到候选。
执行计划和工具声称的范围是待核实假设，不是用户授权或真实产物；转写终点不是文件终点。
仅使用audio_inventory中的audio_ref。seconds与boundary_sec由你提出，版本及帧号由程序绑定。
每个probe的candidate必须引用role=candidate的终态音频；references只可引用源/参考音频，
绝不能用候选与自己比较。二者即使文件内容相同，也保持各自语义角色与时间轴。
evidence_ids必须来自evidence_catalog；只引用几何信息不足以定位语义事件。
未知目标位置、版本映射或接点时，保留missing_evidence并提出observation_requests，
由核听端先盲听再定向定位。位置完全未知可以请求整条该版本音频；absent仅适用所听范围。
本轮新增观察要明确采用或解释冲突，不能重复旧ASR/旧猜测。不要凭新旧先后无条件相信任一模型。
核听也可能不准确，不得编造精度。缺证据的题可暂不列probe；下轮拿到观察再规划。
每个criterion_id恰好出现一次，保留原题全部条件，不可因候选失败而降低标准。
遵守提供的总probe预算和剩余观察预算；不足就明确缺口，不能以抽样冒充全覆盖。
返回实际schema对象；不输出质量等级。"""

SCOPE_PLAN_PROMPT += """
每题只要求判断它所必需的定位精度。检查整句是否保留某项信息，可提供完整句子并逐项
比较；不因缺少逐词边界就拒绝内容判断。多个目标的覆盖仍必须逐项明确，不能抽样。
检查具体接点则须定位真实交界并提供两侧上下文，不能由内容正确推断接点平滑。
available_signal_measurements列出可选的只读测量。若本题需核对同一段声音的幅度或
波形保留，且提出了对应源/候选窗口，可请求signal_measurement=aligned_pcm_comparison。
只有相同帧数、采样率和声道才计算拟合增益及残差；这是事实，不是评分或对应关系真值。
时间伸缩、不同词或错位片段不能仅凭拟合增益判音量；默认none，不增加用户未要求的
逐样本一致约束。其他质量题仍依赖听评；测量不能证明语义、自然度或事件不存在。
available_observation_methods列出本次确实可调用的补感知能力，按缺口选择method。
speech_grounding先独立转写所请求音频，再仅对该转写对齐；不会把目标文本送入ASR。
它可帮助定位语音，但ASR可能错，强制对齐不证明词确实存在，也不证明工具在此拼接。
非语音事件仍可请求listen；只调用可用方法。新增观察足以支持本题时，应消费结果并
撤去已解决的请求；仍冲突则明确记录冲突，不能让旧的精度要求替代本题实际需要。"""

SCOPE_COVERAGE_PROMPT = """独立审查这份多模态听评范围计划；你不听候选，不预测分数。
逐个原始criterion检查：probe问题是否等价地覆盖原题；依据是否支持该版本/区段；
不同时间轴是否错误等同；是否把执行器实际修改区误当成用户授权区。
supported_probe_indices只列范围和问题都有足够依据的probe的0起始序号。
ineligible_probes是程序发现的绑定错误，不能纳入supported_probe_indices；对应条件保持未决，
其他合法题目仍独立检查。不得因为一个接点题缺字段就否定全部内容题。
完整覆盖必须包括本题所有目标/保留区段/接点。身份相同不能替代未遗漏内容与属性保留；
听一个边界不能替代另一个边界，整句清晰不能证明局部接点平滑。
只计算实际编辑内容与保留内容的交界。文件开头/结尾本身不是新的内部接点，不能仅
因没检查文件两个端点就否定内部过渡覆盖；适用的起止伪影属于另一个音质条件。
用公开指令额外检查整份冻结题目是否漏掉必要要求；若漏了记入missing_design_requirements，
不能在看到候选后擅自重写题目。不能用默认逐样本一致限制合法的合成或时长适配。
输入里的时间是待审查观察，不是gold；范围依据不足就unknown并请求针对性补感知。
覆盖不全用incomplete和uncovered_conditions，不能因填满表格就complete。
必须每个criterion_id恰好一次；这只是可错的语义审查，不是覆盖真值。返回实际schema对象。"""

SCOPE_COVERAGE_PROMPT += """
按原题需要的精度审查：完整句子可以覆盖该句内容是否遗漏的检查，不必一律获得逐词
起止；边界平滑的检查必须有对应交界两侧的定位依据，内容正确不能代替边界证据。
完整句子不足以认证长音频其他位置或其他说话人。补感知方法仅从提供的能力中选择。"""

RUNTIME_GOAL_PROMPT = """后续会自动定位并裁片，所以本阶段只冻结公开指令要求，不编造候选坐标。
多模题区别于语义题：不要问推理是否正确或参数是否合法，而要问实际可听的终态条件。
保留条件至少分别考虑：未授权修改的语音/事件内容是否遗漏或重复；应保留的身份、响度、
背景等属性是否损伤。适用时分别成题，不能用身份相同这一个问题代替其他保留检查。
局部任务的两侧和多个接点要在范围描述中完整说明。全局变更也要保留不受授权影响的内容。
每题只表达一个可听条件，可在范围准备时展开多个窗口；不要把内容、音色、情绪混为一问。
既不要遗漏保留，也不要增加逐样本一致或固定时长等用户未指定的约束。
审查设计时同样检查这些区别；若缺少内容保留或只用一个接点代替全部接点，不得认证覆盖。
范围无法预先知道不是题意错误：明确说明需要在运行时补感知，不将猜测写成事实。"""

RUNTIME_GOAL_PROMPT += """
retention_coverage必须分别回答content和attributes两项是否适用；required时必须引用
不同的preserved_region题目。content检查原先仍有效的词义、事件、顺序是否丢失或重复；
attributes检查指令未授权改变的身份/响度/背景等。不能用音质题或身份题代替内容题。
例如添加一个状态描述通常不授权删除原先仍有效的其余信息；合法措辞可变，信息不能擅自丢失。
只有公开指令确实没有相应保留义务才能not_applicable，并给出具体理由。复核端必须审查
豁免理由以及引用的问题实际问了什么，不能只因字段齐全而通过。"""

RUNTIME_SCOPE_GUIDANCE_PROMPT = """terminal_audio_review.scope_review描述评估端的检查范围，
不是执行模块的能力成绩。missing_design_requirements、缺覆盖或核听未知首先是评估侧
未决问题，放入unresolved_review_conflicts；不能据此宣称Actor做错或要求重做正确音频。
new_observations是绑定指定版本的实际追加观察，仍可能有误。需要执行任务本身所缺的
信息时，才指导Reasoner补感知；不要因为Eval没定位好，就批评执行前Reasoner没有核听
未来产物。只有本轮有依据的音频违反或可执行性故障，才形成修改调用的指导。
对已正确的推理/计划保留已有支持，说明哪个具体观察确实改变了下一步决策。"""
