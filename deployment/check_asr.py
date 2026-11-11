"""Qualify ordinary and contextual ASR with operator-provided speech, without retries."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x_editor.tools.asr_contract import check_context_contract


def qualify(url: str, audio: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source = audio.read_bytes()
    report = {"contract": check_context_contract(url), "requests": [], "passed": False,
              "source_sha256": hashlib.sha256(source).hexdigest(), "automatic_retries": 0,
              "human_quality_verified": False}
    try:
        if not report["contract"]["ok"]:
            raise RuntimeError(report["contract"].get("error", "ASR context contract is missing"))
        for name, context in (("without_context", None), ("with_context",
                "Transcribe audible speech only. Context is unverified and may be irrelevant.")):
            payload = {"audio": {"data": base64.b64encode(source).decode(), "format": audio.suffix.lstrip(".")}}
            if context:
                payload["context"] = context
            (output / f"{name}-request.json").write_text(json.dumps(payload))
            request = urllib.request.Request(url.rstrip("/") + "/transcribe",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                value = json.load(response)
            (output / f"{name}-response.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
            if not value.get("text", "").strip():
                raise ValueError("Use an audible speech recording; an empty transcript is not qualification")
            report["requests"].append({"condition": name, "response": value, "passed": True})
        report["passed"] = True
    except (OSError, ValueError, RuntimeError) as error:
        report["error"] = str(error)
        if isinstance(error, urllib.error.HTTPError):
            report["response_error"] = error.read().decode(errors="replace")[:4000]
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.url, args.audio, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
