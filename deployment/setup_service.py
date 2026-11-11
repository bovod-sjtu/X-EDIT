"""Install one isolated backend with its recorded lock; no implicit upgrades."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from x_editor.tools.environments import clean_python_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("service", help="catalog directory, e.g. flexsed, sedicow_asr or moss_transcribe_diarize")
    p.add_argument("--python", default="3.11")
    p.add_argument("--env-path", type=Path, help="Dedicated backend environment outside the source checkout")
    p.add_argument("--source", action="append", type=Path, default=[], help="Pinned source package installed editable with --no-deps")
    args = p.parse_args()
    policies = json.loads((ROOT / 'x_editor/tools/external_services.json').read_text())
    if args.service in policies:
        p.error(policies[args.service]['setup_message'] + '; this installer does not provision external tools')
    project = ROOT / "x_editor/tools/catalog" / args.service
    if project.parent != ROOT / "x_editor/tools/catalog" or not project.is_dir():
        p.error("unknown catalog service")
    uv = shutil.which("uv")
    if not uv:
        p.error("install uv first")
    environment = (args.env_path or project / ".venv").expanduser().resolve()
    install_env = clean_python_environment()
    install_env['UV_PROJECT_ENVIRONMENT'] = str(environment)
    lock = ROOT / "deployment/requirements" / (args.service + ".lock")
    if not lock.exists():
        lock = project / "requirements.lock"
    if lock.exists():
        subprocess.run([uv, "venv", "--allow-existing", "--python", args.python, str(environment)], env=install_env, check=True)
        python = environment / "bin/python"
        subprocess.run([uv, "pip", "sync", "--python", str(python), str(lock)], env=install_env, check=True)
    elif (project / "uv.lock").exists():
        subprocess.run([uv, "sync", "--locked", "--python", args.python], cwd=project, env=install_env, check=True)
        python = environment / "bin/python"
    else:
        raise RuntimeError("No frozen environment for this service; do not silently resolve a new one")
    for source in args.source:
        subprocess.run([uv, "pip", "install", "--python", str(python), "--no-deps", "-e", str(source.resolve(strict=True))], env=install_env, check=True)
    subprocess.run([uv, "pip", "check", "--python", str(python)], env=install_env, check=True)
    installed = subprocess.check_output([uv, "pip", "freeze", "--python", str(python)], env=install_env, text=True)
    (environment / "install_receipt.json").write_text(json.dumps({"service": args.service,
        "python": str(python), "lock": str(lock) if lock.exists() else "uv.lock", "installed": installed}, indent=2))
    print(f"Environment ready: {python}; model download and inference are separate checks")


if __name__ == "__main__":
    main()
