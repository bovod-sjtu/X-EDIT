"""Read-only configuration and deployment diagnostics; never start services."""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

from x_editor.tools.loader import get_catalog_dir, load_all_capabilities
from x_editor.tools.edit_backend import selected_edit_backend
from x_editor.tools.service_bootstrap import matches_environment
from x_editor.tools.external_services import ExternalToolSetupRequired, require_external_service
from x_editor.tools.selection import tool_selection_context
from x_editor.tools.asr_contract import check_context_contract
from x_editor.tools.environments import configure_client_environment
from x_editor.capabilities.catalog import load_operation_capability_catalog


def check(app, *, live=False):
    with tool_selection_context(app.config.tools):
        return _check(app, live=live)


def _check(app, *, live=False):
    findings = []
    findings.append({"component": "edit_backend", "ok": True, "backend": selected_edit_backend()})
    selection = app.config.tools
    profile = app.config.profile
    catalogs = selection.catalog_ids() if selection is not None else None
    executables = ["ffmpeg", "ffprobe"]
    if catalogs is None or catalogs & {"edit_data_pipeline", "qwen3_tts", "seed_vc", "audio_effects",
                                       "spatial_reverb", "speech_enhancement", "speech_separation"}:
        executables.append("uv")
    for executable in executables:
        findings.append({"component": executable, "ok": bool(shutil.which(executable))})
    cards = load_all_capabilities()
    catalog = load_operation_capability_catalog()
    missing_cards = [operation for operation in profile.allowed_edit_operations if catalog.get(operation) is None]
    findings.append({"component": "capability_cards", "ok": not missing_cards,
                     "count": len(profile.allowed_edit_operations), "tool_cards": len(cards), "missing": missing_cards})
    endpoints = {}
    providers = [profile.planner, profile.perception]
    if profile.semantic_planning.reasoner:
        providers.append(profile.semantic_planning.reasoner)
    for p in providers:
        endpoints[p.component] = (p.base_url.rstrip("/") + "/models", p.model)
    if app.config.revision:
        for field in ("semantic_judge", "audio_judge"):
            p = app.config.revision[field]
            endpoints[field] = (p["base_url"].rstrip("/") + "/models", p["model"])
    for name, env, default in (
        ("asr", "EDIT_PIPELINE_ASR_URL", "http://127.0.0.1:8010"),
        ("aligner", "EDIT_PIPELINE_ALIGNER_URL", "http://127.0.0.1:8029"),
        ("moss", "X_EDITOR_MOSS_TRANSCRIBE_URL", "http://127.0.0.1:8048"),
    ):
        tool = "moss_transcribe_diarize" if name == "moss" else name
        if selection is not None and tool not in selection.enabled_tools:
            continue
        endpoints[name] = (os.getenv(env, default).rstrip("/") + "/health", None)
    for config in sorted(get_catalog_dir().glob("*/config.yaml")):
        if catalogs is not None and config.parent.name not in catalogs:
            continue
        data = yaml.safe_load(config.read_text()) or {}
        if os.environ.get("X_EDITOR_MCP_ENV_ROOT"):
            try:
                configure_client_environment(data.get("server", {}), config.parent)
            except FileNotFoundError as error:
                findings.append({"component": f"mcp:{config.parent.name}", "ok": False, "error": str(error)})
        services = data.get("managed_services", [])
        if isinstance(services, dict):
            services = [services]
        for s in services:
            if not matches_environment(s.get("when")):
                continue
            condition = s.get("when") or {}
            if selection is not None and config.parent.name == "edit_data_pipeline":
                operations = {f"edit_data_pipeline.{operation}" for operation in condition.get("values", [])}
                if not operations & selection.operations():
                    continue
            if s.get("external_tool"):
                try:
                    require_external_service(s["external_tool"])
                except ExternalToolSetupRequired as error:
                    if live and not any(row['component'] == s.get("name", config.parent.name) for row in findings):
                        findings.append({"component": s.get("name", config.parent.name),
                                         "ok": False, **error.to_dict()})
                    continue
            if not s.get("url_env"):
                continue
            url = os.getenv(s["url_env"], s.get("default_url", ""))
            endpoints.setdefault(s.get("name", config.parent.name), (url.rstrip("/") + s.get("health_path", "/health"), None))
    if live:
        if selection is not None and set(selection.enabled_tools) & {"edit_pitch_shift", "edit_speech_rate"}:
            endpoints["indextts"] = (os.getenv("EDIT_PIPELINE_INDEXTTS_URL", "http://127.0.0.1:9002").rstrip("/") + "/health", None)
            endpoints["qwen3_tts_tokens"] = (os.getenv("EDIT_PIPELINE_QWEN3TTS_TOKENIZER_URL", "http://127.0.0.1:8019").rstrip("/") + "/health", None)
        def probe(item):
            name, (url, model) = item
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                if model:
                    available = [m["id"] for m in data.get("data", [])]
                    ok = model in available
                    detail = {"expected_model": model, "available_models": available}
                else:
                    status = data.get("status")
                    ok = (status in ("ok", "healthy", "ready", "running", "loaded")
                          or (status is None and data.get("model_loaded") is True))
                    ok = ok and all(data.get(k) is not False for k in ("ready", "model_loaded", "healthy"))
                    detail = {"health": data}
                return {"component": name, "ok": ok, "url": url, **detail}
            except (requests.RequestException, ValueError) as error:
                return {"component": name, "ok": False, "url": url, "error_type": type(error).__name__}
        with ThreadPoolExecutor(max_workers=8) as pool:
            findings.extend(pool.map(probe, endpoints.items()))
        if "asr" in endpoints:
            findings.append(check_context_contract(endpoints["asr"][0].removesuffix("/health"), timeout=5))
    return {"ready": all(f["ok"] for f in findings),
            "scope": ("selected_tools" if selection is not None else "full_catalog") if live else "configuration_only",
            "inference_verified": False, "findings": findings,
            "enabled_tools": list(selection.enabled_tools) if selection is not None else None,
            "edit_operations": profile.allowed_edit_operations,
            "perception_operations": profile.allowed_perception_operations,
            "note": "Selection declares operator-provided capabilities; health is not inference verification. Disabled tools are not probed when tools configuration is supplied."}
