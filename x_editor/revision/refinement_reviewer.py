"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import asyncio


import json


import traceback


from copy import deepcopy


from concurrent.futures import ThreadPoolExecutor


from pathlib import Path


from x_editor.revision.atomic_module_review import invoke as raw_invoke


from x_editor.revision.atomic_module_review import public_context, read


from x_editor.revision.minimal_atomic_calibration import judge, review


from x_editor.revision.refinement_branch_ablation import project_feedback_payload, scope_history


from x_editor.revision.refinement_cache import EpisodeReviewCache, semantic_identity


from x_editor.revision.refinement_checkpoints import (
    latest_module_sources,
    recorded_tool_observations,
)


from x_editor.revision.refinement_goals import call_audio


from x_editor.revision.refinement_scheduling import limited


from x_editor.evaluation.atomic_module_rubrics import bind_design, module_questions


from x_editor.evaluation.guidance_contracts import (
    EvaluationArtifactRef,
    canonical_evaluation_fingerprint,
)


from x_editor.evaluation.iterative_runtime import IterationReview, sha256, write_record


from x_editor.evaluation.guidance_state import (
    PROTOCOL as GUIDANCE_STATE_PROTOCOL, STATE_PROMPT, deferred_guidance_items,
    guidance_state_context, latest_deferred_guidance, refresh_guidance_state, validate_guidance_state,
)


from x_editor.evaluation.minimal_atomic_review import (
    ATOMIC_PROMPT,
    CONSISTENCY_PROMPT,
    MinimalAtomicAnswer,
    build_catalog,
)


from x_editor.evaluation.refinement_contracts import (
    FeedbackScopeReview,
    RegionUtilityDesign,
    UtilityAnswer,
    UtilityAssessment,
    utility_bounds,
)


from x_editor.evaluation.refinement_diagnostics import (
    diagnostic_evidence,
    executable_contract_question,
    is_prior_guidance,
)


from x_editor.evaluation.refinement_feedback import (
    FEEDBACK_SCOPE_PROMPT,
    UTILITY_JUDGE_PROMPT,
    compile_feedback,
    feedback_prompt,
    screen_feedback,
)


from x_editor.evaluation.refinement_focus import (
    FOCUS_PROMPT, OutcomeReviewFocus, compact_focus_positions, validate_focus,
)


from x_editor.evaluation.refinement_observation import recheck_observation


from x_editor.evaluation.scoped_atomic_review import consistency_summary


from x_editor.evaluation.terminal_observation_review import (
    OBSERVATION_PROMPT,
    OBSERVED_UTILITY_PROMPT,
    ObservedUtilityAnswer,
    observe_attachments,
    observed_criterion_payload,
    validate_observed_answer,
    validate_terminal_references,
)


from x_editor.evaluation.terminal_signal_facts import terminal_signal_facts


from x_editor.evaluation.edit_region_scope import (
    REGION_GUIDANCE_PROMPT, REGION_JUDGE_PROMPT, bind_region_design,
)


from x_editor.revision.region_audio_review import regional_answer, region_answer_schema


def invoke(path, prompt, args, *, author=False):
    draw = getattr(args, 'mc_review_draw', None)
    if draw is not None:
        return draw.semantic(prompt, lambda: limited(raw_invoke, getattr(args, 'semantic_limiter', None),
            path, prompt, args, author=author), path)
    return limited(raw_invoke, getattr(args, "semantic_limiter", None),
                   path, prompt, args, author=author)


class PublicRefinementReviewer:
    def __init__(self, *, module_designs, utility_design, args, audio_config, cache=None):
        self.designs, self.utility_design = module_designs, utility_design
        self.args, self.audio_config = args, audio_config
        self.cache = cache or EpisodeReviewCache()
        self.observation_packets = []

    def selection_proxy_configuration(self):
        """Declare the terminal proxy shared with a deferred selector.

        Step diagnostics are intentionally excluded: the final-audio utility is
        unchanged by attaching intermediate observations. Runtime evidence,
        rubric and artifact identities are checked for each actual reuse.
        """
        return {
            'reviewer': type(self).__module__ + '.' + type(self).__qualname__,
            'semantic': semantic_identity(self.args),
            'audio': {k: v for k, v in self.audio_config.items() if k != 'limiter'},
            'protocol': {key: getattr(self.args, key, None) for key in (
                'terminal_rubric_protocol', 'audio_judgment_protocol',
                'scope_max_rounds', 'scope_max_observations', 'scope_max_probes',
                'scope_speech_observer', 'scope_answer_revision_attempts', 'scope_signal_measurements')},
        }

    def online_audio(self, scenario, result, artifact, output, cached_utility):
        """Reuse only online judgments inside this episode; final assessment bypasses this."""
        if self.utility_design is None:
            return None
        if not artifact or cached_utility:
            return self.assess_audio(scenario, result, artifact, output, cached_utility)
        identity = {
            "role": "online_audio", "scenario": scenario.fingerprint,
            "sources": [(a.asset_ref, a.role, sha256(a.path)) for a in scenario.audio_assets],
            "artifact": artifact["sha256"], "design": self.utility_design.model_dump(mode="json"),
            "prompt": UTILITY_JUDGE_PROMPT,
            "audio_judgment_protocol": getattr(self.args, 'audio_judgment_protocol', 'legacy'),
            "observation_protocol": (
                [OBSERVATION_PROMPT, OBSERVED_UTILITY_PROMPT]
                if self.audio_config.get("terminal_review_mode", "direct") == "observed" else None
            ),
            "provider": {k: v for k, v in self.audio_config.items() if k != "limiter"},
        }
        if getattr(self.args, 'execution_purpose_review', False):
            from x_editor.evaluation.execution_purpose_review import sequence_identity
            identity['execution_purposes'] = sequence_identity(scenario,result)
        if isinstance(self.utility_design, RegionUtilityDesign):
            identity["regional_prompt"] = REGION_JUDGE_PROMPT
        from x_editor.evaluation.base_terminal_design import BaseTerminalDesign
        if (getattr(self.args, "terminal_rubric_protocol", "legacy") == "runtime_regions"
                and not isinstance(self.utility_design, BaseTerminalDesign)):
            from x_editor.revision.runtime_region_review import scope_identity
            identity["runtime_scope"] = scope_identity(self.args, result)
        def assess():
            if self.audio_config.get("terminal_review_mode", "direct") == "observed":
                return self.assess_audio(scenario, result, artifact, output, None,
                                         observation_cache=self.cache)
            return self.assess_audio(scenario, result, artifact, output, None)
        pair, reused, key = self.cache.once(identity, lambda: (assess(), str(output)))
        utility, source = pair
        if reused:
            write_record(output / "audio_cache_receipt.json", {
                "key": key, "source": source, "artifact_sha256": artifact["sha256"],
                "review": utility.model_dump(mode="json"), "new_audio_judge_calls": 0,
                "reason": "identical episode, sources, artifact, public goals and judge protocol",
            })
        return utility

    def focus(self, scenario, result, prepared, utility, output, artifact=None):
        self.current_evidence_state = None
        positions = []
        for (turn, module), (design, questions, catalog, sources) in prepared.items():
            selected = {identity for identity, entry in catalog["entries"].items()
                        if entry["source"] in {"module_output", "public_context"}
                        or (entry["source"] == "module_input"
                            and not entry["pointer"].startswith("/candidate_operation_contracts")
                            and not is_prior_guidance(entry))}
            positions.append({
                "turn_index": turn, "module": module,
                "questions": [q.model_dump(mode="json") for q in questions],
                "evidence": diagnostic_evidence(catalog, selected),
            })
        measured = None
        if artifact:
            measured = terminal_signal_facts(
                [EvaluationArtifactRef(artifact_ref_id=a.asset_ref, role=a.role,
                    path=a.path, content_sha256=a.content_sha256) for a in scenario.audio_assets],
                EvaluationArtifactRef(artifact_ref_id="terminal_candidate", role="output",
                    path=artifact["path"], content_sha256=artifact["sha256"]),
            )
            write_record(output / "focus" / "signal_facts.json", measured)
        payload = {
            "schema": OutcomeReviewFocus.model_json_schema(),
            "public_turns": [t.user_request for t in scenario.turns],
            "utility_design": self.utility_design.model_dump(mode="json") if self.utility_design else {
                "criteria": [], "status": "pending_not_passed"},
            "terminal_audio_review": utility.model_dump(mode="json") if utility else None,
            "terminal_coordinate_measurements": measured,
            "terminal_status": result.status,
            "actual_turns": [{"turn_index": t.turn_index, "status": t.result.status,
                              "failure_stage": t.result.failure_stage,
                              "failure_codes": t.result.failure_codes,
                              "error": t.result.terminal_error_message,
                              "tool_observations": recorded_tool_observations(t.result)}
                             for t in result.turn_results],
            **compact_focus_positions(positions),
        }
        focus_prompt = FOCUS_PROMPT
        if getattr(self.args, 'stage_aware_recovery', False):
            from x_editor.evaluation.diagnostic_index_projection import factor_diagnostic_indexes
            payload['shared_evidence_catalog'] = factor_diagnostic_indexes(payload['shared_evidence_catalog'])
            focus_prompt += ('\n证据路径映射可使用 prefix-plus-suffix/v1 无损表示。依各 value 中的 decode '
                             '将 prefix 与 suffix 原样拼接，恢复同组的 evidence_ids 或单个 audio_ref。'
                             '这只改变存储格式，不改变证据内容、命名空间或有效性。')
        branch = getattr(self.args, "feedback_branch", None)
        if branch:
            from x_editor.revision.refinement_branch_ablation import project_focus_payload
            payload = project_focus_payload(payload, branch)
        context_mode = getattr(self.args, "evidence_context_mode", "disabled")
        context = None
        if context_mode != "disabled":
            from x_editor.evaluation.evidence_context import observation_context, EvidenceContextDecision, CONTEXT_PROMPT
            context, current_packet = observation_context(payload, self.observation_packets)
            # Both experimental conditions retain the identical observations.
            # Model-authored summaries are never promoted into new evidence.
            self.observation_packets.append(current_packet)
            payload["evidence_context"] = context
            if context_mode == "explicit_state":
                payload["schema"] = EvidenceContextDecision.model_json_schema()
                focus_prompt += "\n" + CONTEXT_PROMPT
        if (getattr(self.args, "terminal_rubric_protocol", "legacy") == "runtime_regions"
                and branch != "semantic"):
            from x_editor.evaluation.runtime_scope_contracts import RUNTIME_SCOPE_GUIDANCE_PROMPT
            focus_prompt += "\n" + RUNTIME_SCOPE_GUIDANCE_PROMPT
        prompt = {"system_prompt": focus_prompt,
                  "user_prompt": json.dumps(payload, ensure_ascii=False)}
        criteria = [] if branch == "semantic" or self.utility_design is None else [c.criterion_id for c in self.utility_design.criteria]
        if branch == "semantic" or self.utility_design is None:
            # Keep public goals, but do not let withheld terminal grades select
            # a repair or force a fictitious terminal-quality state.
            payload["utility_design"]["criteria"] = []
            prompt["system_prompt"] += (
                "\n本条件只检查推理、计划及真实阻断。终态听音要求没有分配给本分支；"
                "requirement_states（若使用该结构）与 decision.requirement_checks 必须均为空数组。"
                "模块问题中的保留意图可检查是否被正确表达，但不能把它转换为终态音频检查题。"
                "根据可见模块证据选择具体审查位置；无产物或无法听音不妨碍审查已经记录的参数错误。"
            )
            focus_prompt = prompt["system_prompt"]
            prompt["user_prompt"] = json.dumps(payload, ensure_ascii=False)
        write_record(output / "focus" / "prompt.json", prompt)
        for revision in range(getattr(self.args, "focus_revision_attempts", 0) + 1):
            destination = output / "focus" / ("calls" if revision == 0 else "revision_calls")
            raw = invoke(destination, prompt, self.args)
            try:
                if context_mode == "explicit_state":
                    from x_editor.evaluation.evidence_context import validate_context_decision
                    focus, state = validate_context_decision(raw, context, positions, criteria)
                    write_record(output / "focus" / "evidence_state.json", state.model_dump(mode="json"))
                    from x_editor.evaluation.evidence_context import guidance_evidence_context
                    self.current_evidence_state = guidance_evidence_context(state, payload)
                    write_record(output / "focus" / "guidance_evidence_context.json", self.current_evidence_state)
                else:
                    focus = validate_focus(raw, positions, criteria)
                break
            except ValueError as exc:
                write_record(output / "focus" / f"rejected_{revision}.json", {"response": raw, "error": str(exc)})
                if revision == getattr(self.args, "focus_revision_attempts", 0):
                    raise
                # Preserve the rejected judgment; request a new, fully validated
                # decision from the original observations. Never coerce to pass.
                prompt = {"system_prompt": focus_prompt + "\n上份回答违反输出合同。请根据原始观察重新判断，修正矛盾；不必保留旧结论，也不能编造证据。",
                          "user_prompt": json.dumps({**payload, "rejected_response": raw,
                                                    "validation_error": str(exc)}, ensure_ascii=False)}
                write_record(output / "focus" / "revision_prompt.json", prompt)
        write_record(output / "focus" / "decision.json", focus.model_dump(mode="json"))
        return focus

    def review_position(self, key, prepared, output):
        turn, module = key
        design, questions, catalog, sources = prepared
        position = output / "modules" / str(turn) / module
        cache = getattr(self, 'decision_cache', self.cache)
        write_record(position / "sources.json", sources)
        write_record(position / "catalog.json", catalog)
        def ask(question):
            destination = position / "questions" / question.question_id
            call = lambda: limited(judge, getattr(self.args, "semantic_limiter", None),
                                   destination, question, design, catalog, self.args)
            if getattr(self.args, "review_strategy", "full") == "full":
                return call()
            return cache.directory(
                {"kind": "module_question", "prompt": ATOMIC_PROMPT,
                 "question": question.model_dump(mode="json"),
                 "design": design.model_dump(mode="json"), "catalog": catalog,
                 "provider": semantic_identity(self.args)}, destination, call,
                ("answer.json", "receipt.json", "question.json", "resolved_citations.json"),
            )
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
            list(pool.map(ask, questions))
        answers = [MinimalAtomicAnswer.model_validate(
            read(position / "questions" / q.question_id / "answer.json")) for q in questions
            if (position / "questions" / q.question_id / "answer.json").is_file()]
        # Consistency stays criterion-local and includes the original full evidence.
        def check_consistency():
            return limited(review, getattr(self.args, "semantic_limiter", None),
                           position / "consistency", questions, answers, catalog, self.args)
        if getattr(self.args, "review_strategy", "full") == "full":
            consistency = check_consistency()
        else:
            pair, reused, key = cache.once(
                {"kind": "consistency", "prompt": CONSISTENCY_PROMPT,
                 "questions": [q.model_dump(mode="json") for q in questions],
                 "answers": [a.model_dump(mode="json") for a in answers],
                 "catalog": catalog, "provider": semantic_identity(self.args)},
                lambda: (check_consistency(), str(position / "consistency")),
            )
            consistency, source = pair
            if reused:
                write_record(position / "consistency" / "reuse.json", {
                    "key": key, "source": source, "new_provider_calls": 0,
                    "review": consistency.model_dump(mode="json") if consistency else None,
                })
        summary = consistency_summary(questions, answers, consistency)
        write_record(position / "summary.json", summary)
        return {
            "turn_index": turn, "module": module,
            "questions": [q.model_dump(mode="json") for q in questions],
            "answers": [a.model_dump(mode="json") for a in answers],
            "evidence": diagnostic_evidence(catalog, {i for a in answers for i in a.evidence_ids}),
            "withheld_as_facts": summary["withheld_question_ids"],
            "consistency": consistency.model_dump(mode="json") if consistency else None,
        }

    def semantic_audio_design(self, prompt, payload, directory):
        args = self.args
        if set(payload) == {'questions'}:
            from copy import copy
            args = copy(args)
            args.author_tokens = 512
        return invoke(directory, {'system_prompt':prompt,
            'user_prompt':json.dumps(payload,ensure_ascii=False)}, args, author=True)

    def audio_call(self, prompt, payload, paths, directory):
        config = self.audio_config
        if getattr(self.args, 'audio_judgment_protocol', 'legacy') == 'numeric_polarity':
            # Neutral observation needs more words; numeric decisions need only two fields.
            config = {**config, 'max_tokens': 128 if 'levels' in payload else 512}
        draw = getattr(self.args, 'mc_review_draw', None)
        if draw is not None:
            return draw.audio(config, prompt, payload, paths, directory)
        return call_audio(config, prompt, payload, paths, directory)

    async def review_mc(self, **kwargs):
        from x_editor.revision.mc_review_sampling import review_mc
        return await review_mc(self, **kwargs)

    def assess_audio(self, scenario, result, artifact, output, cached_utility, *, observation_cache=None):
        if not artifact or self.utility_design is None:
            return None
        purpose = None
        if getattr(self.args, 'execution_purpose_review', False):
            from x_editor.evaluation.execution_purpose_review import review_sequence, sequence_identity
            identity = canonical_evaluation_fingerprint(sequence_identity(scenario,result))
            old = (cached_utility.scope_review or {}).get('execution_purposes') if cached_utility else None
            if old and old.get('context_fingerprint') == identity:
                purpose = old
            else:
                # An unchanged final WAV cannot stand in for changed intermediate outputs.
                purpose = review_sequence(scenario,result,self.semantic_audio_design,self.audio_call,
                    output/'execution_purposes',observation_cache=observation_cache,
                    module_sources=getattr(self,'current_decision_sources',None))
                purpose['context_fingerprint'] = identity
        # Preserve the terminal judgment before attaching diagnostic step observations.
        # Both records are immutable; they must not compete for utility.json.
        terminal_output = output / 'terminal_quality' if purpose is not None else output
        assessed = self._assess_terminal_audio(scenario,result,artifact,terminal_output,cached_utility,
                                             observation_cache=observation_cache)
        if assessed is not None and purpose is not None:
            assessed = assessed.model_copy(update={'scope_review':{
                **(assessed.scope_review or {}),'execution_purposes':purpose}})
            write_record(output/'utility.json',assessed.model_dump(mode='json'))
        return assessed

    def _assess_terminal_audio(self, scenario, result, artifact, output, cached_utility, *, observation_cache=None):
        from x_editor.evaluation.base_terminal_design import BaseTerminalDesign, observation_confidence
        if not artifact or self.utility_design is None:
            return None
        if cached_utility:
            if (cached_utility.artifact_sha256 != artifact["sha256"]
                    or cached_utility.design_fingerprint != canonical_evaluation_fingerprint(self.utility_design)):
                raise ValueError("cached audio assessment belongs to another artifact or rubric design")
            if (getattr(self.args, "terminal_rubric_protocol", "legacy") == "runtime_regions"
                    and not isinstance(self.utility_design, BaseTerminalDesign)):
                from x_editor.revision.runtime_region_review import scope_identity
                if (not cached_utility.scope_review or cached_utility.scope_review.get("configuration_fingerprint")
                        != canonical_evaluation_fingerprint(scope_identity(self.args, result))):
                    write_record(output / "audio_cache_invalidation.json", {
                        "reason": "runtime scope evidence or configuration changed",
                        "previous_configuration_fingerprint": (cached_utility.scope_review or {}).get("configuration_fingerprint"),
                        "current_configuration_fingerprint": canonical_evaluation_fingerprint(scope_identity(self.args, result)),
                        "previous_assessment_retained": True,
                    })
                    # Identical audio does not make an earlier runtime context
                    # current. Re-assess under this attempt; never rebind old grades.
                    return self.assess_audio(scenario, result, artifact, output, None,
                                             observation_cache=observation_cache)
            write_record(
                output / "audio_cache_receipt.json",
                {
                    "artifact_sha256": artifact["sha256"],
                    "review": cached_utility.model_dump(mode="json"),
                    "new_audio_judge_calls": 0,
                    "reason": "same bytes, public goals and review protocol",
                },
            )
            return cached_utility
        import soundfile as sf

        regional = isinstance(self.utility_design, RegionUtilityDesign)
        checks = {}
        if regional:
            bind_region_design(self.utility_design, scenario)
            checks = {c.criterion_id: c for c in self.utility_design.region_checks}
        if (getattr(self.args, "terminal_rubric_protocol", "legacy") == "runtime_regions"
                and not isinstance(self.utility_design, BaseTerminalDesign)):
            if not regional:
                raise ValueError("automatic scope review requires a regional utility design")
            from x_editor.revision.runtime_region_review import assess_runtime_regions
            assessed = assess_runtime_regions(self, scenario, result, artifact, output, invoke)
            write_record(output / "observation_confidence.json",
                         observation_confidence(self.utility_design, assessed.answers))
            return assessed
        attachments = [
            {"audio_ref": a.asset_ref, "role": a.role, "path": a.path}
            for a in scenario.audio_assets
        ]
        attachments.append(
            {"audio_ref": "terminal_candidate", "role": "candidate", "path": artifact["path"]}
        )
        inventory = [
            {
                "audio_ref": a["audio_ref"],
                "role": a["role"],
                "attachment_number": index + 1,
                "sha256": sha256(a["path"]),
                "duration_sec": sf.info(a["path"]).duration,
            }
            for index, a in enumerate(attachments)
        ]
        if getattr(self.args, 'audio_judgment_protocol', 'legacy') in {'evidence_ordinal', 'numeric_polarity'}:
            from x_editor.evaluation.evidence_ordinal_review import observe_audio, judge_evidence
            kwargs = {}
            if self.args.audio_judgment_protocol == 'numeric_polarity':
                from x_editor.evaluation.numeric_polarity_review import judge_evidence
                kwargs['semantic_call'] = self.semantic_audio_design
            paths = [a['path'] for a in attachments]
            observations = observe_audio(inventory, paths, self.audio_call, output/'neutral_observations')
            answers = []
            for criterion in self.utility_design.criteria:
                decision = judge_evidence(criterion.question, inventory, paths, observations,
                    self.audio_call, output/'audio'/criterion.criterion_id,
                    public_turns=[t.user_request for t in scenario.turns], **kwargs)
                answers.append(UtilityAnswer(question_id=criterion.criterion_id, grade=decision['grade'],
                    finding=decision['finding'], evidence_ids=tuple(i['audio_ref'] for i in inventory)))
            utility = UtilityAssessment(design_fingerprint=canonical_evaluation_fingerprint(self.utility_design),
                artifact_sha256=artifact['sha256'], answers=tuple(answers),
                assessor_role='online_selection_proxy')
            write_record(output/'utility.json', utility.model_dump(mode='json'))
            write_record(output/'observation_confidence.json', observation_confidence(self.utility_design, answers))
            return utility
        observed = self.audio_config.get("terminal_review_mode", "direct") == "observed"
        observations = None
        if observed:
            observations = observe_attachments(
                [{**a, "sha256": info["sha256"]} for a, info in zip(attachments, inventory)],
                call=self.audio_call, output=output / "neutral_observations",
                provider={k: v for k, v in self.audio_config.items() if k != "limiter"},
                cache=observation_cache, concurrency=self.audio_config.get("concurrency", 1),
            )
        evidence_ids = {a["audio_ref"] for a in inventory} | {"public_turns"}
        answers, regional_observations = [], []
        errors = [f"neutral observation {note['audio_ref']}: {note['error']}"
                  for note in (observations or []) if note.get("error")]
        def assess(criterion):
            directory = output / "audio" / criterion.criterion_id
            try:
                schema = (region_answer_schema(checks[criterion.criterion_id]) if regional else
                          (ObservedUtilityAnswer if observed else UtilityAnswer).model_json_schema())
                schema["properties"]["question_id"]["const"] = criterion.criterion_id
                schema["properties"]["evidence_ids"]["items"]["enum"] = sorted(evidence_ids)
                payload = self.audio_call(
                    (OBSERVED_UTILITY_PROMPT + "\n" + REGION_JUDGE_PROMPT if regional else
                     OBSERVED_UTILITY_PROMPT if observed else UTILITY_JUDGE_PROMPT),
                    {
                        "schema": schema,
                        "allowed_evidence_ids": sorted(evidence_ids),
                        "question_id": criterion.criterion_id,
                        **({"region_check": checks[criterion.criterion_id].model_dump(mode="json")}
                           if regional else {}),
                        **(observed_criterion_payload(
                            criterion, [t.user_request for t in scenario.turns], inventory, observations
                        ) if observed else {
                            "criterion": criterion.model_dump(mode="json"),
                            "final_state": self.utility_design.final_state,
                            "public_turns": [t.user_request for t in scenario.turns],
                            "audio_inventory": inventory,
                        }),
                    },
                    [a["path"] for a in attachments],
                    directory / "calls",
                )
                write_record(directory / "original_answer.json", payload)
                observation = None
                if regional:
                    answer, observation = regional_answer(
                        payload, criterion.criterion_id, checks[criterion.criterion_id], inventory,
                    )
                else:
                    answer = (validate_observed_answer(payload, criterion.criterion_id, inventory)
                              if observed else validate_terminal_references(
                                  UtilityAnswer.model_validate(payload), criterion.criterion_id, inventory))
                if (
                    answer.question_id != criterion.criterion_id
                    or set(answer.evidence_ids) - evidence_ids
                ):
                    raise ValueError("audio answer uses the wrong criterion or reference")
                write_record(directory / "answer.json", answer.model_dump(mode="json"))
                return answer, None, observation
            except Exception as error:  # noqa: BLE001 -- retain every external provider failure
                message = f"{criterion.criterion_id}: {type(error).__name__}: {error}"
                write_record(directory / "error.json", {"error": message})
                return None, message, None
        with ThreadPoolExecutor(max_workers=self.audio_config.get("concurrency", 1)) as pool:
            for answer, error, observation in pool.map(assess, self.utility_design.criteria):
                if answer is not None:
                    answers.append(answer)
                if error is not None:
                    errors.append(error)
                if observation is not None:
                    regional_observations.append(observation)
        if observed and answers and self.audio_config.get("terminal_scope_audit", False):
            from x_editor.evaluation.terminal_scope_audit import (
                SCOPE_AUDIT_PROMPT, apply_scope_audit, scope_audit_payload,
            )
            try:
                raw = {a.question_id: read(output / "audio" / a.question_id / "original_answer.json")
                       for a in answers}
                ids = set(raw)
                payload = scope_audit_payload(
                    [c for c in self.utility_design.criteria if c.criterion_id in ids], raw,
                    [t.user_request for t in scenario.turns], inventory,
                )
                audit = invoke(output / "scope_audit", {
                    "system_prompt": SCOPE_AUDIT_PROMPT,
                    "user_prompt": json.dumps(payload, ensure_ascii=False),
                }, self.args)
                answers, issues = apply_scope_audit(audit, answers)
                errors.extend(issues)
                write_record(output / "scope_audit" / "audit.json", audit)
            except Exception as error:  # noqa: BLE001 -- absent qualification cannot certify a grade
                errors.append(f"terminal scope audit failed: {type(error).__name__}: {error}")
                answers = [a.model_copy(update={"grade": None}) for a in answers]
        utility = UtilityAssessment(
            design_fingerprint=canonical_evaluation_fingerprint(self.utility_design),
            artifact_sha256=artifact["sha256"],
            answers=tuple(answers),
            errors=tuple(errors),
            assessor_role="online_selection_proxy",
            regional_observations=tuple(regional_observations),
        )
        write_record(output / "utility.json", utility.model_dump(mode="json"))
        write_record(output / "observation_confidence.json",
                     observation_confidence(self.utility_design, answers))
        return utility

    def review_sync(self, **kwargs):
        """A diagnostic failure must not erase an already bound audio assessment."""
        if ((getattr(self.args, 'error_first_guidance', False) or self.decision_first)
                and kwargs.get('artifact') is None and kwargs.get('build_feedback', True)):
            if getattr(self.args, 'feedback_branch', None) == 'multimodal':
                # Removing semantic diagnosis must not reintroduce it through
                # a differently named Guider. The common raw-error path remains.
                return IterationReview()
            from x_editor.revision.error_first_review import review_failure
            return review_failure(reviewer=self, invoke=invoke, **{key: kwargs[key] for key in
                ('scenario', 'result', 'directory', 'output', 'attempt_id')})
        audio_state = {}
        try:
            return self._review_sync(**kwargs, _audio_state=audio_state)
        except Exception as error:
            if getattr(self.args, "review_strategy", "full") == "full" or not audio_state:
                raise
            message = f"{type(error).__name__}: {error}"
            write_record(kwargs["output"] / "diagnostic_error.json",
                         {"error": message, "traceback": traceback.format_exc()})
            return IterationReview(utility=audio_state["utility"], errors=[message])

    def _review_sync(
        self,
        *,
        scenario,
        result,
        directory,
        output,
        attempt_id,
        history,
        utility_design,
        artifact,
        cached_utility,
        build_feedback=True,
        _audio_state=None,
        _decision_review=None,
    ):
        if utility_design != self.utility_design:
            raise ValueError("reviewer utility design changed")
        output.mkdir(parents=True, exist_ok=False)
        strategy = getattr(self.args, "review_strategy", "full")
        branch = getattr(self.args, "feedback_branch", None)
        sequential = self.decision_first
        optimized, outcome_first = strategy != "full", strategy in {"outcome_first", "decision_first"}
        semantic_isolated = branch == "semantic" and getattr(self.args, "strict_feedback_isolation", False)
        if semantic_isolated and not outcome_first:
            raise ValueError("strict source isolation requires outcome-first review")
        utility = self.online_audio(scenario, result, artifact, output, cached_utility) if optimized and not semantic_isolated and not sequential else None
        if optimized and _audio_state is not None:
            _audio_state["utility"] = utility
        write_record(output / "review_policy.json", {
            "strategy": strategy, "feedback_branch": branch,
            "strict_feedback_isolation": getattr(self.args, "strict_feedback_isolation", False),
            "build_feedback": build_feedback, "module_audit_can_certify_quality": False,
            "guidance_state_protocol": GUIDANCE_STATE_PROTOCOL,
        })
        if optimized and not build_feedback and not sequential:
            write_record(output / "deferred_module_audit.json", {
                "source_directory": str(directory), "source_attempt_id": attempt_id,
                "module_designs": self.designs,
                "reason": "No next guidance consumes module scores; final audio still assessed.",
                "status": "deferred_not_passed", "new_module_judge_calls": 0,
            })
            return IterationReview(utility=utility)
        if sequential:
            if _decision_review is None:
                from x_editor.revision.decision_first_review import review_decisions
                _decision_review = review_decisions(self, scenario=scenario, result=result,
                    directory=directory, output=output / 'decisions')
            sources_by_module, extraction_errors = _decision_review.sources, _decision_review.errors
            self.current_decision_sources = sources_by_module
        else:
            sources_by_module, extraction_errors = latest_module_sources(scenario, result, directory)
        if (not sequential and artifact is not None and build_feedback and not self.designs
                and hasattr(self, 'module_design_provider')):
            self.designs = self.module_design_provider()
        atomic, allowed, catalogs, prepared = [], {}, {}, {}
        for (turn, module), sources in (sorted(sources_by_module.items()) if not sequential else ()):
            context = public_context(scenario.model_dump(mode="json"), turn)
            if turn not in self.designs:
                extraction_errors.append(f"turn {turn}: module questions unavailable; runtime errors remain eligible")
                continue
            design = bind_design(context, self.designs[turn])
            questions = module_questions(design, module)
            if module == "planner":
                questions = (*questions, executable_contract_question())
            catalog = build_catalog(sources)
            catalogs[(turn, module)] = catalog
            prepared[(turn, module)] = design, questions, catalog, sources
        decision_passed = False
        if sequential:
            prepared, catalogs, atomic = (_decision_review.prepared,
                                         _decision_review.catalogs, _decision_review.atomic)
            decision_passed = _decision_review.gate['quality_review_allowed']
            write_record(output / 'decision_gate.json', _decision_review.gate)
            if not decision_passed:
                # Only semantic findings can guide this phase. Even targeted
                # audio checks in the Guider must wait for decision eligibility.
                branch, outcome_first = 'semantic', False
            elif not semantic_isolated:
                utility = self.online_audio(scenario, result, artifact, output, cached_utility)
                if _audio_state is not None:
                    _audio_state['utility'] = utility
                if utility is None:
                    return IterationReview(errors=['Decision qualified; output quality assessment unavailable.'],
                                           stop_reason='quality_review_unknown')
            if not build_feedback or (decision_passed and semantic_isolated):
                return IterationReview(utility=utility, errors=list(map(str, extraction_errors)),
                    stop_reason='decision_qualified' if decision_passed and semantic_isolated else None)
            if not atomic:
                return IterationReview(errors=list(map(str, extraction_errors)),
                                       stop_reason='decision_review_unknown')
        focus = None
        if outcome_first:
            try:
                focus = self.focus(scenario, result, prepared, utility, output, None if semantic_isolated else artifact)
            except Exception as exc:  # noqa: BLE001 -- preserve audio when the diagnostic judge fails
                error = f"{type(exc).__name__}: {exc}"
                write_record(output / "focus" / "error.json", {"error": error})
                return IterationReview(utility=utility, errors=[error, *map(str, extraction_errors)])
            if (focus.disposition == "no_supported_issue" and utility is not None and utility.errors):
                # A diagnostic model cannot turn a rejected audio answer into
                # qualified evidence by paraphrasing it as a satisfied goal.
                focus = focus.model_copy(update={"disposition": "evaluator_uncertain"})
            unassigned_uncertainty = (
                getattr(self.args, "evidence_context_mode", "disabled") != "disabled"
                and focus.disposition == "supported_error_unassigned"
                and (any(c.status == "unknown" for c in focus.requirement_checks) or bool(focus.evaluator_disagreements)))
            if (not semantic_isolated and (focus.disposition == "evaluator_uncertain" or unassigned_uncertainty) and artifact is not None
                    and getattr(self.args, "bounded_uncertainty_recheck", False)):
                # A judge uncertainty consumes one fresh observation pass, not
                # an actor attempt. Keep both judgments and do not vote on them.
                recheck = output / "uncertainty_recheck"
                recheck.mkdir(parents=True, exist_ok=False)
                utility = self.assess_audio(scenario, result, artifact, recheck, None)
                if _audio_state is not None:
                    _audio_state["utility"] = utility
                focus = self.focus(scenario, result, prepared, utility, recheck, artifact)
                if focus.disposition == "no_supported_issue" and utility is not None and utility.errors:
                    focus = focus.model_copy(update={"disposition": "evaluator_uncertain"})
            if focus.disposition != "review_modules":
                supported = (focus.disposition == "no_supported_issue"
                             and bool(focus.requirement_checks)
                             and all(c.status == "satisfied" for c in focus.requirement_checks)
                             and utility_bounds(utility_design, utility)["lower"] == 1.0)
                reason = "supported_stop" if supported else focus.disposition
                write_record(output / "focused_recovery_decision.json", {
                    "action": "retain_current_candidate", "reason": reason,
                    "independent_quality_certified": False,
                    "new_module_audits": 0,
                })
                return IterationReview(utility=utility, stop_reason=reason,
                                       errors=list(map(str, extraction_errors)))
            selected = {(p.turn_index, p.module): set(p.question_ids) for p in focus.positions}
            deferred = []
            for key, (design, questions, catalog, sources) in list(prepared.items()):
                keep = selected.get(key, set())
                deferred.extend({"turn_index": key[0], "module": key[1],
                                 "question": q.model_dump(mode="json")} for q in questions
                                if q.question_id not in keep)
                if keep:
                    prepared[key] = design, tuple(q for q in questions if q.question_id in keep), catalog, sources
                else:
                    del prepared[key]
            write_record(output / "deferred_module_audit.json", {
                "source_directory": str(directory), "source_attempt_id": attempt_id,
                "questions": deferred, "status": "deferred_not_passed",
            })
        # Independent positions may overlap. The shared provider limiter still bounds calls.
        if sequential:
            # These exact checkpoints were already judged before any listening.
            # Outcome focus may narrow guidance but cannot re-grade the gate.
            pass
        elif branch == "multimodal":
            atomic = []
        elif outcome_first or branch:
            with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
                atomic = list(pool.map(lambda item: self.review_position(*item, output), prepared.items()))
        else:
            atomic = [self.review_position(key, value, output) for key, value in prepared.items()]
            if not optimized:
                utility = self.assess_audio(scenario, result, artifact, output, cached_utility)
        if branch:
            for (turn, _), (_, questions, _, _) in prepared.items():
                allowed.setdefault(turn, []).extend(q.question_id for q in questions)
        for item in atomic:
            allowed.setdefault(item["turn_index"], []).extend(q["question_id"] for q in item["questions"])
        purpose_rows = ((utility.scope_review or {}).get('execution_purposes',{}).get('observations',[])
                        if utility is not None and branch != 'semantic' else [])
        for item in purpose_rows:
            allowed.setdefault(item['turn_index'],[]).append(item['question_id'])
        if not build_feedback:
            write_record(output / "feedback_not_requested.json", {
                "reason": "No subsequent attempt consumes guidance; module and audio judgments retained.",
                "provider_calls": 0,
            })
            return IterationReview(utility=utility, errors=list(map(str, extraction_errors)))
        # Maps come from actual asset facts and accepted prior outputs, not filenames.
        inventory_options = ({'include_intermediates': True}
            if getattr(self.args, 'execution_purpose_review', False) and branch != 'semantic' else {})
        audio_inventory, paths = observation_inventory(scenario, result, catalogs, **inventory_options)
        guidance, memories, feedback_errors, deferred_items = {}, [], [], {}
        deferred_items = latest_deferred_guidance(history)
        for turn in sorted(allowed):
            own_inventory = [a for a in audio_inventory if a["turn_index"] == turn]
            if branch == "semantic":
                own_inventory = []
            own_history = [
                item
                for h in history
                for item in h["review"].get("turns", [])
                if item["turn_index"] == turn
            ]
            prefix = public_context(scenario.model_dump(mode="json"), turn).model_dump(mode="json")
            prompt = feedback_prompt(
                {str(turn): prefix},
                [item for item in atomic if item["turn_index"] == turn],
                audio_inventory=own_inventory,
                episode_history=own_history,
                focused=outcome_first or sequential,
            )
            prior_entries = {
                identity: entry
                for (index, _), catalog in catalogs.items()
                if index == turn
                for identity, entry in catalog["entries"].items()
                if is_prior_guidance(entry)
            }
            prompt["history_for_scope_review"] = {
                "earlier_episode_reviews": own_history,
                "prior_guidance_input_entries": prior_entries,
            }
            payload = json.loads(prompt["user_prompt"])
            payload['target_turn_index'] = turn
            payload['available_question_ids'] = allowed[turn]
            prompt['system_prompt'] += (
                '\nOnly target_turn_index is being reviewed in this call. Copy that actual '
                'conversation index into turns[].turn_index; do not reset it to zero. '
                'The /turns/0 JSON pointer is a position within the response, not a conversation '
                'turn number. Cite only available_question_ids. Existing observations and '
                'guidance-state update references have separate namespaces.'
            )
            if sequential:
                payload['decision_review_gate'] = _decision_review.gate
                prompt['system_prompt'] += (
                    '\nDecision-first policy: scores at or above the declared minimum are '
                    'qualified decision work. Do not request repairs merely to raise a '
                    'qualified decision score. Failed checks need semantic correction; '
                    'unknown checks need clarification, not an invented failure. After '
                    'qualification, actual audio failures may justify upstream revisions. '
                    'An earlier correct decision is not proof of correct output sound.'
                )
            from x_editor.evaluation.refinement_diagnostics import (
                CAPABILITY_CONTEXT_PROMPT, available_capability_briefs,
            )
            payload["available_tool_capabilities"] = available_capability_briefs(
                catalogs.get((turn, "planner"), {"entries": {}}))
            prompt["system_prompt"] += "\n" + CAPABILITY_CONTEXT_PROMPT
            actual_turn = next(t for t in result.turn_results if t.turn_index == turn)
            payload["recorded_tool_observations"] = recorded_tool_observations(actual_turn.result)
            from x_editor.evaluation.plan_dependency_observation import observe_plan_dependencies
            payload["recorded_plan_dependencies"] = observe_plan_dependencies(
                sources_by_module.get((turn, "planner"), {}).get("module_output"))
            if focus:
                if getattr(self, "current_evidence_state", None) is not None:
                    from x_editor.evaluation.evidence_context import GUIDANCE_CONTEXT_PROMPT
                    payload["current_evidence_state"] = self.current_evidence_state
                    prompt["system_prompt"] += "\n" + GUIDANCE_CONTEXT_PROMPT
                payload["diagnostic_scope"] = {
                    "selected_positions": [p.model_dump(mode="json") for p in focus.positions
                                           if p.turn_index == turn],
                    "unreviewed_modules": "deferred, not passed or failed",
                    "guidance_rule": "只给selected_positions中的模块填写unresolved_work或next_checks。未审模块不能判错或判为通过；依赖通知只放upstream_changes，仍有效的保留要求放preservation_requirements，不能把等待上游的提醒写成新的修复动作。若发现审查范围以外的必要缺口，列入unresolved_review_conflicts，不能假装该模块已审查。",
                }
            if turn == len(scenario.turns) - 1:
                if utility and (utility.scope_review or {}).get('execution_purposes'):
                    prompt['system_prompt'] += (
                        '\nIntermediate purpose observations bind actual producing steps and audio versions. '
                        'A low intermediate score diagnoses a possible cause; it is not a new final requirement. '
                        'Do not repair an earlier stem merely because it lacks background that a later mix restores. '
                        'Use low terminal grades together with relevant step observations to select upstream guidance. '
                        'Opposite-question disagreement is observation uncertainty; do not invent an acoustic cause. '
                        'Already-correct reasoning stays valid; change only the necessary parameter or workflow.'
                    )
                payload["terminal_audio_review"] = (
                    utility.model_dump(mode="json") if utility else None
                )
                if isinstance(self.utility_design, RegionUtilityDesign):
                    payload["terminal_region_requirements"] = [
                        c.model_dump(mode="json") for c in self.utility_design.region_checks
                    ]
                    prompt["system_prompt"] += "\n" + REGION_GUIDANCE_PROMPT
                    if getattr(self.args, "terminal_rubric_protocol", "legacy") == "runtime_regions":
                        from x_editor.evaluation.runtime_scope_contracts import RUNTIME_SCOPE_GUIDANCE_PROMPT
                        prompt["system_prompt"] += "\n" + RUNTIME_SCOPE_GUIDANCE_PROMPT
            payload['execution_purpose_observations'] = [
                item for item in purpose_rows if item['turn_index'] == turn]
            if branch:
                payload["available_question_ids"] = allowed[turn]
                payload = project_feedback_payload(payload, branch)
                prompt["history_for_scope_review"] = scope_history(branch, own_history)
                prompt["system_prompt"] += (
                    "\nThis is a controlled feedback-source experiment. Respect feedback_information_policy. "
                    "Do not reconstruct unavailable reviews. Guidance targets must be reasoner or planner; "
                    "request missing perception through Reasoner. Semantic-only must leave observation_checks empty."
                )
            from x_editor.evaluation.refinement_provenance import (
                bind_provenance,
                diagnostic_catalog,
            )

            provenance = diagnostic_catalog(
                payload.get("atomic_reviews", []),
                utility if turn == len(scenario.turns) - 1 and branch != "semantic" else None,
                purpose_observations=payload.get("execution_purpose_observations",[]),
            )
            payload["repair_evidence_catalog"] = provenance
            prior_deferred = deferred_items.get(str(turn), {})
            prompt["history_for_scope_review"]["deferred_prior_items"] = prior_deferred
            payload["active_guidance_state"] = guidance_state_context(
                payload, prompt["history_for_scope_review"], turn_index=turn,
            )
            prompt["system_prompt"] += "\n" + STATE_PROMPT
            feedback_dir = output / "feedback" / f"{turn:04d}"
            try:
                prompt["user_prompt"] = json.dumps(payload, ensure_ascii=False)
                write_record(feedback_dir / "prompt.json", prompt)
                raw_draft = invoke(feedback_dir / "compilation", prompt, self.args, author=True)
                write_record(feedback_dir / "original_draft.json", raw_draft)
                draft, rejected = screen_feedback_with_revision(
                    raw_draft, prompt=prompt, directory=feedback_dir, args=self.args,
                    allowed_questions={turn: allowed[turn]},
                    audio_inventory=own_inventory,
                )
                write_record(feedback_dir / "observation_rejections.json", rejected)
                feedback_errors.extend(f"observation {r['index']}: {r['error']}" for r in rejected)
                write_record(feedback_dir / "draft.json", draft.model_dump(mode="json"))
                observations = []
                for i, check in enumerate(draft.observation_checks):
                    path = paths[(check.turn_index, check.audio_ref, check.audio_sha256)]
                    try:
                        observations.append(
                            recheck_observation(
                                check,
                                source=path,
                                output=feedback_dir / "observations" / f"{i:04d}",
                                call=self.audio_call,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 -- preserve failed attempt without fallback delivery
                        observations.append(
                            {"status": "unknown", "error": f"{type(exc).__name__}: {exc}"}
                        )
                if observations:
                    payload["new_targeted_observations"] = observations
                    provenance.update(diagnostic_catalog([], observations=observations))
                    payload["repair_evidence_catalog"] = provenance
                    payload = refresh_guidance_state(payload)
                    prompt["history_for_scope_review"]["pre_observation_draft"] = draft.model_dump(
                        mode="json"
                    )
                    second = {
                        "system_prompt": prompt["system_prompt"],
                        "user_prompt": json.dumps(payload, ensure_ascii=False),
                    }
                    revised = invoke(
                        feedback_dir / "after_observation", second, self.args, author=True
                    )
                    write_record(feedback_dir / "original_after_observation.json", revised)
                    draft, rejected = screen_feedback(
                        revised,
                        allowed_questions={turn: allowed[turn]},
                        audio_inventory=own_inventory,
                    )
                    write_record(feedback_dir / "revised_observation_rejections.json", rejected)
                    write_record(
                        feedback_dir / "after_observation.json", draft.model_dump(mode="json")
                    )
                draft = qualify_feedback(
                    draft,
                    payload=payload,
                    prompt=prompt,
                    directory=feedback_dir / "scope",
                    turn=turn,
                    args=self.args,
                    allowed_questions={turn: allowed[turn]},
                    audio_inventory=own_inventory,
                )
                if branch and any(m.module not in {"reasoner", "planner"} for t in draft.turns for m in t.modules):
                    raise ValueError("Branch trial guidance must target a live Reasoner/Planner consumer")
                write_record(feedback_dir / "eligible_guidance.json", draft.model_dump(mode="json"))
                write_record(feedback_dir / "guidance_state.json", validate_guidance_state(draft, payload))
                deferred_items[str(turn)] = deferred_guidance_items(draft, payload)
                memory = draft.model_dump(mode="json")
                memories.extend(memory["turns"])
                compiled = compile_feedback(
                        draft,
                        source_attempt_id=attempt_id,
                        review_id=canonical_evaluation_fingerprint(memory),
                        module_scoped=True,
                        actionable_only=outcome_first or sequential,
                    )
                guidance.update(compiled)
                write_record(feedback_dir / "provenance.json", bind_provenance(draft, provenance, compiled))
            except Exception as exc:  # noqa: BLE001 -- preserve failed attempt without fallback delivery
                error = f"{type(exc).__name__}: {exc}"
                write_record(feedback_dir / "error.json", {"error": error})
                deferred_items[str(turn)] = deepcopy(payload["active_guidance_state"]["prior_items"])
                feedback_errors.append(error)
        return IterationReview(
            guidance_by_turn=guidance,
            utility=utility,
            memory={"turns": memories, "deferred_guidance_items": deferred_items},
            errors=[*feedback_errors, *map(str, extraction_errors)],
        )

    @property
    def decision_first(self):
        return getattr(self.args, 'review_strategy', 'full') == 'decision_first'

    async def __call__(self, **kwargs):
        resolve_design = kwargs.pop('resolve_utility_design', None)
        if self.decision_first and kwargs.get('artifact') is not None:
            from x_editor.revision.decision_first_review import review_decisions
            decision = await asyncio.to_thread(review_decisions, self,
                **{k: kwargs[k] for k in ('scenario', 'result', 'directory')},
                output=kwargs['output'].parent.parent / 'decision_reviews' / kwargs['output'].parent.name / kwargs['output'].name)
            semantic_only = getattr(self.args, 'feedback_branch', None) == 'semantic'
            if decision.gate['quality_review_allowed'] and not semantic_only and resolve_design:
                await resolve_design()
                kwargs['utility_design'] = self.utility_design
            kwargs['_decision_review'] = decision
        return await asyncio.to_thread(self.review_sync, **kwargs)


def screen_feedback_with_revision(raw, *, prompt, directory, args,
                                  allowed_questions, audio_inventory):
    """One explicitly budgeted correction of an invalid critic envelope.

    Never fix turn numbers or citations in Python: the critic must author a
    valid response from the same evidence, and it is subsequently scope-audited.
    """
    try:
        return screen_feedback(raw, allowed_questions=allowed_questions,
                               audio_inventory=audio_inventory)
    except ValueError as exc:
        if not getattr(args, 'evaluation_contract_revisions', 0):
            raise
        payload = json.loads(prompt['user_prompt'])
        payload.update(rejected_response=raw, validation_error=str(exc),
            allowed_turn_question_ids=allowed_questions,
            revision_request='修正当前评审返回的轮次/题目引用/结构合同。严格使用当前实际轮次，'
                             '不能将JSON数组位置当作会话轮次；不能发明题号或证据。'
                             '重新依据相同事实回答，不保留无法支持的指导。')
        revised = invoke(directory/'contract_revision', dict(system_prompt=prompt['system_prompt'],
            user_prompt=json.dumps(payload, ensure_ascii=False)), args, author=True)
        write_record(directory/'contract_revision.json', dict(original_error=str(exc), response=revised))
        return screen_feedback(revised, allowed_questions=allowed_questions,
                               audio_inventory=audio_inventory)


def qualify_feedback(
    draft, *, payload, prompt, directory, turn, args, allowed_questions, audio_inventory
):
    """One scope review and at most one recorded revision of a critic draft.

    This budget belongs to the critic, not actor retries. Earlier rejected
    drafts and their judgments remain separate records.
    """
    for index in range(2):
        position = directory / f"{index:04d}"
        state_errors = []
        try:
            validate_guidance_state(draft, payload)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            state_errors.append(str(exc))
        scope = payload.get("diagnostic_scope")
        if scope is not None:
            owners = {(p["turn_index"], p["module"]) for p in scope["selected_positions"]}
            for item in draft.turns:
                for module in item.modules:
                    if (module.unresolved_work or module.next_checks) and (item.turn_index, module.module) not in owners:
                        state_errors.append(
                            f"actionable guidance targets an unaudited module: turn={item.turn_index}, "
                            f"module={module.module}. Keep dependency notifications in upstream_changes "
                            "and preservation requirements in preservation_requirements; only reviewed "
                            "positions may receive unresolved_work or next_checks. Preserve supported "
                            "guidance for reviewed modules."
                        )
        audit = FeedbackScopeReview.model_validate(
            invoke(
                position / "audit",
                {
                    "system_prompt": FEEDBACK_SCOPE_PROMPT,
                    "user_prompt": json.dumps(
                        {
                            "schema": FeedbackScopeReview.model_json_schema(),
                            "turn_index": turn,
                            "visible_context": payload,
                            "historical_review_material": prompt.get(
                                "history_for_scope_review", []
                            ),
                            "proposed_guidance": draft.model_dump(mode="json"),
                            "guidance_state_contract_errors": state_errors,
                        },
                        ensure_ascii=False,
                    ),
                },
                args,
            )
        )
        write_record(position / "audit.json", audit.model_dump(mode="json"))
        if audit.turn_index != turn:
            raise ValueError("feedback scope audit belongs to another turn")
        observations_accounted = (
            not payload.get("new_targeted_observations")
            or audit.new_observations_accounted_for is True
        )
        phase_qualified = ((getattr(args, "review_strategy", None) not in {"outcome_first", "decision_first"}
                            and "active_guidance_state" not in payload)
                           or audit.respects_module_phase is True)
        # The first recovery has no old issues to reconcile. A null judgment
        # means inapplicable in that case, not an unreviewed failure. Explicit
        # rejection still blocks; structural state validation remains required.
        prior_items = (payload.get("active_guidance_state") or {}).get("prior_items", {})
        state_qualified = (
            "active_guidance_state" not in payload
            or audit.guidance_state_accounted_for is True
            or (not prior_items and audit.guidance_state_accounted_for is None)
        )
        if audit.usable and observations_accounted and phase_qualified and state_qualified and not state_errors:
            return draft
        if index == 1:
            break
        revised = invoke(
            position / "revision",
            {
                "system_prompt": prompt["system_prompt"],
                "user_prompt": json.dumps(
                    {
                        **payload,
                        "scope_review": audit.model_dump(mode="json"),
                        "guidance_state_contract_errors": state_errors,
                        "revision_request": "根据本次事实和新增观察重新形成指导，回应复核指出的问题，保留有依据的正确部分；不复制旧指导，不重写执行计划。",
                    },
                    ensure_ascii=False,
                ),
            },
            args,
            author=True,
        )
        write_record(position / "original_revision.json", revised)
        draft, rejected = screen_feedback(
            revised, allowed_questions=allowed_questions, audio_inventory=audio_inventory
        )
        write_record(position / "rejected_observations.json", rejected)
    raise ValueError("feedback scope remains unqualified after one bounded critic revision")


def observation_inventory(scenario, result, catalogs, *, include_intermediates=False):
    import soundfile as sf

    candidates = {sha256(a.path): Path(a.path) for a in scenario.audio_assets}
    prior_outputs = []
    for turn in result.turn_results:
        forward = turn.result.forward_result
        if forward is not None and forward.status == "completed" and forward.final_output_path:
            path = Path(forward.final_output_path)
            if path.is_file():
                prior_outputs.append((sha256(path), turn.turn_index, path))
    inventory, paths = {}, {}
    for (turn, module), catalog in catalogs.items():
        for identity, entry in catalog["entries"].items():
            if entry["source"] != "module_input" or not entry["pointer"].startswith(
                "/audio_asset_facts/"
            ):
                continue
            fact = entry["value"]
            digest, ref = fact.get("observed_content_sha256"), fact.get("audio_ref")
            if not ref or not digest:
                continue
            visible = {a.content_sha256 for a in scenario.visible_audio_assets(turn)}
            path = candidates.get(digest) if digest in visible else None
            eligible_prior = [
                (index, p) for d, index, p in prior_outputs if d == digest and index < turn
            ]
            if path is None and eligible_prior:
                path = max(eligible_prior, key=lambda item: item[0])[1]
            if path is None:
                continue
            key = turn, ref, digest
            inventory.setdefault(
                key,
                {
                    "turn_index": turn,
                    "audio_ref": ref,
                    "sha256": digest,
                    "duration_sec": sf.info(path).duration,
                    "evidence_ids": [],
                },
            )["evidence_ids"].append(identity)
            paths[key] = path
    # A completed current-turn artifact is visible to its post-execution critic,
    # but never exposed as an earlier module input or a future-turn observation.
    for digest, turn, path in prior_outputs:
        ref = f"review:turn:{turn}:output"
        key = turn, ref, digest
        inventory[key] = {
            "turn_index": turn,
            "audio_ref": ref,
            "sha256": digest,
            "duration_sec": sf.info(path).duration,
            "evidence_ids": [],
            "role": "completed_current_turn_output",
        }
        paths[key] = path
    if include_intermediates:
        from x_editor.evaluation.execution_purpose_review import execution_inventory
        assets, steps = execution_inventory(scenario,result)
        for step in steps:
            for ref in step['output_refs']:
                asset=assets[ref]
                key=step['turn_index'],ref,asset['sha256']
                inventory[key]={'turn_index':step['turn_index'],'audio_ref':ref,
                    'sha256':asset['sha256'],'duration_sec':sf.info(asset['path']).duration,
                    'evidence_ids':[],'role':'recorded_intermediate_output',
                    'producer_step_id':step['step_id']}
                paths[key]=Path(asset['path'])
    # These IDs locate same-turn source claims, not certified event timestamps.
    # Public asset names and runtime refs are both supplied to the critic. Admit
    # their declared identity aliases only for a visible, hash-matched asset.
    for key, item in list(inventory.items()):
        turn, _ref, digest = key
        if digest not in candidates:
            continue
        for asset in scenario.visible_audio_assets(turn):
            if asset.content_sha256 != digest:
                continue
            alias_key = turn, asset.asset_ref, digest
            if alias_key not in inventory:
                inventory[alias_key] = {**item, 'audio_ref': asset.asset_ref,
                    'evidence_ids': list(item['evidence_ids'])}
                paths[alias_key] = paths[key]
    for (turn, _), catalog in catalogs.items():
        known = set(catalog["entries"])
        for key, item in inventory.items():
            if key[0] == turn:
                item["evidence_ids"] = sorted(set(item["evidence_ids"]) | known)
    return list(inventory.values()), paths
