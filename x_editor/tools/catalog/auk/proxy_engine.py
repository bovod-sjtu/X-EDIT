"""Reuse an already loaded model for isolated adapter validation, without reloads."""

from x_editor.tools.catalog.auk.client import edit_audio


class ProxyEngine:
    def __init__(self, base_url):
        self.base_url = base_url

    def edit(self, **arguments):
        result = edit_audio(arguments, base_url=self.base_url)
        return {**result["parameters"], "backend_model_identity": result["model_identity"]}
