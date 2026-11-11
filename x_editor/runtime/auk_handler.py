"""Minimal AuK operation registration into the existing execution registry."""


class AukHandler:
    def __init__(self, executor):
        self.executor = executor

    async def handle(self, step, state):
        from x_editor.runtime.audio_plan_executor import _output_path, _resolve_input_audio

        params = dict(step.params)
        instruction = params.pop("instruction")
        if set(params) - {"gen_seconds"}:
            raise ValueError("AuK received undeclared planner parameters")
        return await self.executor._execute_tool("auk", {
            "input_audio_path": _resolve_input_audio(step, state),
            "output_path": _output_path(step, state), "instruction": instruction, **params,
        }, state)



def register_auk(executor):
    executor.registry.register(
        "auk", AukHandler(executor).handle, tool_id="auk", required_inputs=["input_audio"],
        required_params=["instruction"],
        description="Regenerate one bound source for explicit speech-editing intent; declared channel averaging yields mono output and preservation requires validation.",
    )
