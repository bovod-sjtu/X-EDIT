"""Production review support extracted without changing function bodies."""

from __future__ import annotations


from x_editor.evaluation.probe_invocation import RecordedModule, record_runtime_modules


class RecordedForward(RecordedModule):
    async def arun(self, request, *, observer=None):
        actual = {
            **request.model_dump(mode="json"),
            "audio_bindings": [
                binding.model_dump(mode="json") for binding in request.audio_bindings
            ],
            "runtime_output_dir": request.runtime_output_dir,
            "event_log_path": request.event_log_path,
            "session_state": request.session_state.model_dump(mode="json")
            if request.session_state
            else None,
        }
        destination, started = self._start(
            "arun", (actual,), {"observer_present": observer is not None}
        )
        try:
            result = await self.component.arun(request, observer=observer)
        except Exception as exc:
            self._finish(destination, started, error=exc)
            raise
        self._finish(destination, started, output=result)
        return result
