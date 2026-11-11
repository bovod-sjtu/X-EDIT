"""Production review support extracted without changing function bodies."""

from x_editor.revision.module_chain_transport import (
    AssignmentTransport,
    current_assignment,
    recorded_http,
)


class RecordedScenario:
    def __init__(self, runtime, directory):
        self.runtime, self.directory = runtime, directory

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    async def arun_scenario(self, *args, **kwargs):
        # No new call budget: the existing actor's frozen limits remain in force.
        transport = AssignmentTransport(self.directory / "http", 2**63 - 1)
        token = current_assignment.set(transport)
        try:
            return await self.runtime.arun_scenario(*args, **kwargs)
        finally:
            current_assignment.reset(token)
