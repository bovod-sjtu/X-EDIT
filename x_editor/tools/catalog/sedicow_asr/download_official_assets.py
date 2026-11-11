"""Download official SE-DiCoW / DiCoW assets for the local service wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPOS = (
    "BUT-FIT/SE-DiCoW",
    "BUT-FIT/DiCoW_v3_3",
    "BUT-FIT/DiCoW_v3_2",
    "BUT-FIT/diarizen-wavlm-large-s80-md",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="sedicow_asr catalog directory",
    )
    parser.add_argument(
        "--skip-repos",
        action="store_true",
        help="Do not clone/update official GitHub repositories.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Do not download Hugging Face model snapshots.",
    )
    parser.add_argument(
        "--hf-repo",
        action="append",
        dest="hf_repos",
        default=[],
        help="Additional or replacement HF repo. If provided, defaults are still included unless --only-hf-repos is set.",
    )
    parser.add_argument(
        "--only-hf-repos",
        action="store_true",
        help="Download only repos passed with --hf-repo.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    vendor = root / "vendor"
    checkpoints = root / "checkpoints" / "hf"
    vendor.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "root": str(root),
        "repos": [],
        "weights": [],
    }

    if not args.skip_repos:
        _ensure_git_repo(
            "https://github.com/BUTSpeechFIT/DiCoW",
            vendor / "DiCoW",
            summary["repos"],
        )
        _ensure_git_repo(
            "https://github.com/BUTSpeechFIT/TS-ASR-Whisper",
            vendor / "TS-ASR-Whisper",
            summary["repos"],
        )

    if not args.skip_weights:
        repos = tuple(args.hf_repos) if args.only_hf_repos else (*DEFAULT_REPOS, *tuple(args.hf_repos))
        seen: set[str] = set()
        for repo_id in repos:
            if repo_id in seen:
                continue
            seen.add(repo_id)
            started = time.time()
            target = checkpoints / repo_id.replace("/", "__")
            try:
                path = Path(snapshot_download(repo_id=repo_id, local_dir=target))
                files = [p for p in path.rglob("*") if p.is_file()]
                summary["weights"].append(
                    {
                        "repo_id": repo_id,
                        "status": "ok",
                        "path": str(path),
                        "file_count": len(files),
                        "size_bytes": sum(p.stat().st_size for p in files),
                        "elapsed_sec": round(time.time() - started, 2),
                    }
                )
            except Exception as exc:  # pragma: no cover - environment/network dependent
                summary["weights"].append(
                    {
                        "repo_id": repo_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_sec": round(time.time() - started, 2),
                    }
                )

    summary_path = checkpoints / "download_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path}")


def _ensure_git_repo(url: str, path: Path, summary: list[dict[str, object]]) -> None:
    started = time.time()
    try:
        if (path / ".git").exists():
            subprocess.run(["git", "-C", str(path), "fetch", "--all", "--prune"], check=True)
        else:
            subprocess.run(["git", "clone", url, str(path)], check=True)
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        summary.append(
            {
                "url": url,
                "status": "ok",
                "path": str(path),
                "commit": commit,
                "elapsed_sec": round(time.time() - started, 2),
            }
        )
    except Exception as exc:  # pragma: no cover - environment/network dependent
        summary.append(
            {
                "url": url,
                "status": "failed",
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.time() - started, 2),
            }
        )


if __name__ == "__main__":
    main()
