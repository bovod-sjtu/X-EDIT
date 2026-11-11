"""Production review support extracted without changing function bodies."""

from concurrent.futures import Future


from threading import Lock


from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint


from x_editor.evaluation.iterative_runtime import sha256, write_record


def semantic_identity(args):
    return {name: getattr(args, name, None) for name in (
        "provider", "model", "base_url", "judge_tokens", "author_tokens", "timeout",
        "factor_review_evidence", "compact_scope_preparation"
    )}


class EpisodeReviewCache:
    def __init__(self):
        self._lock = Lock()
        self._values = {}

    def once(self, identity, call):
        key = canonical_evaluation_fingerprint(identity)
        with self._lock:
            future = self._values.get(key)
            owner = future is None
            if owner:
                future = self._values[key] = Future()
        if owner:
            try:
                future.set_result(call())
            except BaseException as error:
                future.set_exception(error)
                raise
        return future.result(), not owner, key

    def directory(self, identity, destination, call, files):
        def produce():
            call()
            return destination

        source, reused, key = self.once(identity, produce)
        if reused:
            receipts = {}
            for name in files:
                path = source / name
                if path.is_file():
                    import json
                    write_record(destination / name, json.loads(path.read_text()))
                    receipts[name] = sha256(path)
            write_record(destination / "reuse.json", {
                "key": key, "source": str(source), "source_files": receipts,
                "new_provider_calls": 0, "exact_input_reuse": True,
            })
        return source
