"""Portable command-line entry point; all execution delegates to x_editor.api."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="x-edit")
    parser.add_argument("--env-file", type=Path, help="Load deployment variables without overriding the shell")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Copy packaged default configuration into a new directory")
    init.add_argument("directory", type=Path)
    for name in ("edit", "run", "doctor", "profile"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name in ("edit", "run"):
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--mode", choices=("forward", "revision"), default="revision")
        if name == "edit":
            sub.add_argument("--audio", type=Path, required=True)
            sub.add_argument("--instruction", required=True)
            sub.add_argument("--reference", action="append", type=Path, default=[])
        if name == "run":
            sub.add_argument("--request", type=Path, required=True)
        if name == "doctor":
            sub.add_argument("--live", action="store_true", help="Also check model endpoints and tool health")
    args = parser.parse_args()
    if args.command == "init":
        import shutil
        source = Path(__file__).parent / "presets"
        shutil.copytree(source, args.directory, ignore=shutil.ignore_patterns("__pycache__"))
        print(str(args.directory.resolve() / "runtime.yaml"))
        return
    if args.env_file:
        from dotenv import load_dotenv
        if not args.env_file.is_file():
            parser.error("environment file does not exist")
        load_dotenv(args.env_file, override=False)
    from x_editor.api import XEdit, bind_assets
    from x_editor.runtime.workflow_case import WorkflowRuntimeScenario
    app = XEdit.from_config(args.config)
    if args.command == "profile":
        print(json.dumps(app.config.model_dump(mode="json"), indent=2))
    elif args.command == "doctor":
        from x_editor.readiness import check
        report = check(app, live=args.live)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ready"] else 2)
    else:
        if args.command == "edit":
            result = app.edit(args.audio, args.instruction, references=args.reference,
                              output=args.output, mode=args.mode)
        else:
            scenario = WorkflowRuntimeScenario.model_validate_json(args.request.read_text())
            scenario = bind_assets(scenario, args.request.resolve().parent)
            result = app.run(scenario, output=args.output, mode=args.mode)
        print(json.dumps({"delivered": result["delivered"], "artifact": result.get("selected_artifact"),
                          "result": str(args.output.resolve() / "result.json")}, indent=2))
        raise SystemExit(0 if result["delivered"] else 2)


if __name__ == "__main__":
    main()
