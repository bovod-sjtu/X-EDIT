"""Check source and package contents against the documented release boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIRECTORIES = (
    "third_party/dots_tts_edit", "third_party/src", "third_party/service_assets",
)
DOCUMENTATION_ONLY_DIRECTORIES = (
    "x_editor/tools/catalog/sam_audio", "x_editor/tools/catalog/voice_sculptor",
)
EXTERNAL_BACKEND_FILES = {
    "x_editor/runtime/sam_audio_handler.py",
    "x_editor/capabilities/cards/scope_acoustic_timeline/sam_audio.yaml",
    "x_editor/capabilities/cards/content_voice/voice_sculptor.yaml",
    "x_editor/tools/catalog/sam_audio/service.py",
    "x_editor/tools/catalog/sam_audio/engine.py",
    "x_editor/tools/catalog/sam_audio/requirements.lock",
    "x_editor/tools/catalog/voice_sculptor/service.py",
    "x_editor/tools/catalog/voice_sculptor/setup.sh",
    "x_editor/tools/catalog/voice_sculptor/check_environment.sh",
    "x_editor/tools/catalog/voice_sculptor/pyproject.toml",
    "x_editor/tools/catalog/voice_sculptor/uv.lock",
    "deployment/requirements/sam_audio.lock",
    "deployment/requirements/voice_sculptor.lock",
}
FORBIDDEN_DIRECTORIES = {
    ".git", ".venv", "__pycache__", "models", "checkpoints", "runs", "datasets", "vendor",
}
FORBIDDEN_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".mp4",
    ".webm", ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".onnx",
    ".so", ".dll", ".dylib", ".exe", ".pyc", ".pyo", ".pem", ".key",
}
PATTERNS = {
    "cluster_path": re.compile(r"/(?:prodcpfs|cpfs|lustre|dssg)/|/mnt[/]petrelfs/|\.codex/(?:shared|worktrees)/"),
    "cluster_specific_reference": re.compile(r"\bPAI\b|\bDLC\b|\bpi\.(?:sjtu|hpc)|\bpilogin\b|\bdlc_entrypoint\b|\bscripts/dl[c]/", re.I),
    "private_network_endpoint": re.compile(r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?=[:/\s\"'])"),
    "credential": re.compile(r"(?:AKIA|LTAI)[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "cluster_job_identifier": re.compile(r"\bdlc[a-z0-9]{16,}\b", re.I),
}


def inspect_file(name, payload):
    path = PurePosixPath(name)
    reasons = []
    if name in EXTERNAL_BACKEND_FILES:
        reasons.append("operator_installed_backend")
    if any(name == prefix or name.startswith(prefix + "/") for prefix in DOCUMENTATION_ONLY_DIRECTORIES):
        reasons.append("documentation_only_extension_code")
    if path.is_absolute() or ".." in path.parts:
        reasons.append("unsafe_archive_path")
    if FORBIDDEN_DIRECTORIES.intersection(path.parts):
        reasons.append("excluded_directory")
    if any(name == prefix or name.startswith(prefix + "/") for prefix in EXTERNAL_DIRECTORIES):
        reasons.append("external_source_or_asset")
    if name.startswith("x_editor/accent_library/data/") and path.name != "README.md":
        reasons.append("reference_library_data")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES or re.search(r"\.so\.\d", path.name):
        reasons.append("model_media_or_binary")
    if path.name in {"ffmpeg", "ffprobe", ".env"} or (path.name.startswith(".env.") and path.name != ".env.example"):
        reasons.append("binary_or_environment_file")
    if payload.startswith((b"\x7fELF", b"MZ", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")):
        reasons.append("native_executable")
    try:
        contents = payload.decode("utf-8")
    except UnicodeDecodeError:
        reasons.append("nontext_release_content")
    else:
        reasons.extend(label for label, pattern in PATTERNS.items() if pattern.search(contents))
    return [{"file": name, "reason": reason} for reason in reasons]


def source_entries(root):
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and Path(result.stdout.strip()).resolve() == root.resolve():
        names = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        ).decode().split("\0")
        for name in sorted(set(names) - {""}):
            path = root / name
            if path.is_symlink():
                yield name, b"", "source_symlink"
            elif path.is_file():
                yield name, path.read_bytes(), None
        return
    for directory, directories, filenames in os.walk(root):
        directories.sort()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            name = path.relative_to(root).as_posix()
            if path.is_symlink():
                yield name, b"", "source_symlink"
            else:
                yield name, path.read_bytes(), None
        for dirname in list(directories):
            path = Path(directory) / dirname
            if path.is_symlink():
                directories.remove(dirname)
                yield path.relative_to(root).as_posix(), b"", "source_symlink"


def archive_entries(path):
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if stat.S_ISLNK(member.external_attr >> 16):
                    yield member.filename, b"", "archive_link"
                else:
                    yield member.filename, archive.read(member), None
        return
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        if len(roots) != 1:
            raise ValueError("A source archive must have one top-level directory")
        for member in members:
            if member.isdir():
                continue
            original = PurePosixPath(member.name)
            name = PurePosixPath(*original.parts[1:]).as_posix()
            if original.is_absolute() or ".." in original.parts:
                yield member.name, b"", "unsafe_archive_path"
            elif not member.isfile():
                yield name, b"", "archive_link_or_special_file"
            else:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Cannot read archive member: " + member.name)
                with stream:
                    yield name, stream.read(), None


def check_entries(entries):
    failures = []
    count = 0
    seen = set()
    for name, payload, error in entries:
        count += 1
        if name in seen:
            failures.append({"file": name, "reason": "duplicate_member"})
        seen.add(name)
        if error:
            failures.append({"file": name, "reason": error})
        failures.extend(inspect_file(name, payload))
    if not count:
        failures.append({"file": "", "reason": "empty_release"})
    return {"passed": not failures, "files_checked": count, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = check_entries(source_entries(args.source))
    artifacts = []
    for artifact in args.artifact:
        result = check_entries(archive_entries(artifact))
        result.update(name=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        artifacts.append(result)
    result = {
        "schema_version": "x-edit.release-content-audit/v1",
        "passed": source["passed"] and all(row["passed"] for row in artifacts),
        "source": source, "artifacts": artifacts,
        "scope": "Tracked and nonignored project files plus supplied archives; external gitlink checkouts and ignored runtime downloads are separate dependencies.",
        "legal_approval": False, "inference_verified": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
