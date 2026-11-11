"""Production review support extracted without changing function bodies."""

import asyncio


from copy import copy, deepcopy


from threading import RLock


from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint as fingerprint


from x_editor.evaluation.iterative_runtime import IterationReview, sha256, write_record


class FrozenPreparation:
    def __init__(self):
        self.responses = {}
        self.lock = RLock()

    def get(self, key, request, directory, *, first):
        with self.lock:
            if key not in self.responses:
                if not first:
                    raise ValueError('MC draw requested different preparation evidence')
                try:
                    self.responses[key] = (True, deepcopy(request()))
                except Exception as exc:
                    self.responses[key] = (False, f'{type(exc).__name__}: {exc}')
                    raise
            else:
                write_record(directory / 'mc_preparation_reuse.json', {
                    'request_fingerprint': key, 'new_model_calls': 0,
                    'cached_success': self.responses[key][0]})
            ok, result = self.responses[key]
            if not ok:
                raise ValueError('Frozen preparation failed: ' + result)
            return deepcopy(result)


class ReviewDraw:
    def __init__(self, preparation, index, seed, context):
        self.preparation, self.index, self.seed, self.context = preparation, index, seed, context
        self.numeric_calls = []

    def semantic(self, prompt, request, directory):
        return self.preparation.get(fingerprint(['semantic', prompt]), request, directory,
                                    first=self.index == 0)

    def audio(self, config, prompt, payload, paths, directory):
        from x_editor.revision.refinement_goals import call_audio
        hashes = [sha256(p) for p in paths]
        key = fingerprint(['audio', prompt, payload, hashes])
        if 'levels' not in payload:
            return self.preparation.get(key, lambda: call_audio(config, prompt, payload, paths, directory),
                                        directory, first=self.index == 0)
        seed = int(fingerprint([self.seed, self.context, self.index, key])[:8], 16) % (2**31)
        write_record(directory / 'mc_draw.json', {'draw': self.index, 'seed': seed,
            'request_fingerprint': key, 'audio_hashes': hashes,
            'backend_independence_certified': False})
        self.numeric_calls.append({'seed': seed, 'request_fingerprint': key})
        return call_audio({**config, 'seed': seed, 'seed_namespace': None}, prompt, payload, paths, directory)


async def review_mc(reviewer, *, repeats, seed, **kwargs):
    if (not reviewer.decision_first or
            getattr(reviewer.args, 'audio_judgment_protocol', None) != 'numeric_polarity'):
        raise ValueError('MC requires decision-first numeric positive/reverse review')
    if reviewer.audio_config.get('temperature', 0) <= 0:
        raise ValueError('MC numeric sampling requires an explicit positive temperature')
    from x_editor.revision.decision_first_review import review_decisions
    decision = await asyncio.to_thread(review_decisions, reviewer,
        **{k: kwargs[k] for k in ('scenario', 'result', 'directory')},
        output=kwargs['output'] / 'decision')
    write_record(kwargs['output'] / 'decision_gate.json', decision.gate)
    if not decision.gate['quality_review_allowed']:
        return [IterationReview(errors=['Semantic gate did not qualify audio review.']) for _ in range(repeats)]
    if kwargs.get('resolve_utility_design'):
        await kwargs['resolve_utility_design']()
    preparation = FrozenPreparation()
    results = []
    expected_questions = None
    context = [kwargs['scenario'].fingerprint, kwargs['artifact']['sha256']]
    for index in range(repeats):
        worker = copy(reviewer)
        worker.args = copy(reviewer.args)
        worker.args.execution_purpose_review = False  # final artifact selection only
        worker.args.reuse_scope_preparation = False  # the frozen tape owns preparation reuse
        draw = ReviewDraw(preparation, index, seed, context)
        worker.args.mc_review_draw = draw
        output = kwargs['output'] / f'draw_{index:02d}'
        try:
            assessed = await asyncio.to_thread(worker.assess_audio, kwargs['scenario'], kwargs['result'],
                kwargs['artifact'], output, None)
            questions = sorted(c['request_fingerprint'] for c in draw.numeric_calls)
            if expected_questions is None:
                expected_questions = questions
            elif questions != expected_questions:
                raise ValueError('MC replicas did not judge the same frozen questions')
            results.append(IterationReview(utility=assessed))
        except Exception as exc:
            results.append(IterationReview(errors=[f'{type(exc).__name__}: {exc}']))
        write_record(output / 'mc_receipt.json', {'draw': index, 'numeric_calls': draw.numeric_calls,
            'errors': results[-1].errors, 'independent_final_assessment_used': False})
    return results
