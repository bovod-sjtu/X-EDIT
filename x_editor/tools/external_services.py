"""Require operator installation and explicit opt-in before external model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from x_editor.tools.edit_backend import uses_dots_model


POLICIES = json.loads(Path(__file__).with_suffix(".json").read_text())


class ExternalToolSetupRequired(RuntimeError):
    def __init__(self, service: str) -> None:
        self.service = service
        self.policy = POLICIES[service]
        super().__init__(
            f"{self.policy['setup_message']} After reviewing the upstream terms and "
            f"starting your own service, explicitly set {self.policy['url_env']} and "
            f"{self.policy['enable_env']}=1. X·Edit does not download, install or start "
            f"this implementation. {self.policy['terms']}"
        )

    def to_dict(self) -> dict:
        return {"status": "external_tool_setup_required", "service": self.service,
                "message": str(self), "source_url": self.policy["source_url"],
                "model_url": self.policy["model_url"], "retryable": False}


def require_external_service(service: str) -> str:
    policy = POLICIES[service]
    endpoint = os.getenv(policy["url_env"], "").strip()
    try:
        parsed = urlsplit(endpoint)
        parsed.port
    except ValueError:
        raise ExternalToolSetupRequired(service) from None
    if (os.getenv(policy["enable_env"]) != "1" or parsed.scheme not in {"http", "https"}
            or not parsed.hostname or parsed.username or parsed.password):
        raise ExternalToolSetupRequired(service)
    return endpoint.rstrip("/")


def require_pipeline_service(operation: str) -> None:
    if uses_dots_model(operation):
        return
    if operation in POLICIES["f5"]["pipeline_operations"]:
        require_external_service("f5")
