"""Fetch explicitly pinned public dependencies. Never discover private credentials."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reject_operator_assets(names, field):
    services = json.loads((ROOT / 'deployment/stack.json').read_text())['services']
    managed = {asset for service in services.values() if not service.get('external_only')
               for asset in service.get(field, [])}
    restricted = {asset for service in services.values() if service.get('external_only')
                  for asset in service.get(field, [])} - managed
    if set(names) & restricted:
        raise ValueError('Operator-installed external tools: download and install independently outside this checkout; '
                         'this command will not fetch ' + ', '.join(sorted(set(names) & restricted)))


def assets(names):
    """Fetch ancillary upstream files without redistributing them in the release."""
    manifest = json.loads((ROOT / 'third_party/manifest.json').read_text())
    rows = manifest.get('service_assets', [])
    known = {Path(row['file']).parts[1] for row in rows}
    if set(names) - known:
        raise ValueError('Unknown service asset groups: ' + str(sorted(set(names) - known)))
    for row in rows:
        if Path(row['file']).parts[1] not in names:
            continue
        path = ROOT / 'third_party' / row['file']
        if not path.resolve().is_relative_to((ROOT / 'third_party/service_assets').resolve()):
            raise ValueError('Service asset path escapes its declared directory')
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Refusing to replace changed service asset: ' + row['file'])
        else:
            with urllib.request.urlopen(row['url'], timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Downloaded service asset hash differs: ' + row['file'])
            path.parent.mkdir(parents=True, exist_ok=True)
            # Never silently replace an asset created by another installer.
            with path.open('xb') as stream:
                stream.write(content)
        print(row['file'] + ': ' + row['sha256'])


def repositories(names):
    reject_operator_assets(names, 'repositories')
    manifest = json.loads((ROOT / "third_party/manifest.json").read_text())
    by_name = {r["name"]: r for r in manifest["repositories"]}
    if set(names) - by_name.keys():
        raise ValueError('Unsupported source dependencies: ' + ', '.join(sorted(set(names) - by_name.keys())))
    for name in names:
        repo = by_name[name]
        dest = ROOT / "third_party/src" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        newly_cloned = not dest.exists()
        if newly_cloned:
            subprocess.run(["git", "clone", "--no-checkout", repo["url"], str(dest)], check=True)
        if not newly_cloned and subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"]):
            raise RuntimeError(f"Refusing to replace changes in {dest}")
        subprocess.run(["git", "-C", str(dest), "fetch", "origin", repo["commit"]], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", repo["commit"]], check=True)
        subprocess.run(["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"], check=True)
        print(f"{name}: {repo['commit']}")


def models(names, destination, output=None):
    reject_operator_assets(names, 'models')
    manifest = json.loads((ROOT / "third_party/manifest.json").read_text())
    rows = {r["repo_id"]: r for r in manifest["models"]}
    if set(names) - rows.keys():
        raise ValueError('Unsupported model dependencies: ' + ', '.join(sorted(set(names) - rows.keys())))
    from huggingface_hub import snapshot_download
    receipt = json.loads(output.read_text()) if output and output.exists() else {"schema_version": "x-edit.model-assets/v1", "models": {}}
    for name in names:
        revision = rows[name].get("revision")
        if not revision:
            raise ValueError(f"No pinned revision for {name}; resolve it before downloading")
        if name in receipt['models'] and receipt['models'][name]['revision'] != revision:
            raise RuntimeError(f'Refusing to change pinned model in an existing receipt: {name}')
        for attempt in range(3):
            try:
                path = snapshot_download(name, revision=revision, cache_dir=str(destination),
                                         allow_patterns=rows[name].get("allow_patterns"),
                                         max_workers=2)
                break
            except Exception as error:
                # snapshot_download can wrap a transient HTTP failure inside
                # LocalEntryNotFoundError. Never retry access denial or bad pins.
                cause=error; response=None
                while cause is not None:
                    candidate=getattr(cause,'response',None)
                    if candidate is not None:
                        response=candidate; break
                    cause=cause.__cause__
                if response is None or response.status_code not in (429,502,503,504) or attempt==2:
                    raise
                delay=30*(2**attempt)
                retry_after=response.headers.get('Retry-After','')
                if retry_after.isdigit():
                    delay=max(delay,int(retry_after))
                print(f'{name}: HTTP {response.status_code}; retry {attempt+1}/2 in {delay}s',flush=True)
                time.sleep(delay)
        # Some unchanged upstream loaders address the default ref. Pin that ref
        # in this deployment's isolated cache, never silently replace another one.
        ref = Path(path).parents[1] / "refs/main"
        if ref.exists() and ref.read_text().strip() != revision:
            raise RuntimeError(f"Cache default ref differs for {name}; use a new deployment cache")
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision)  # HF expects exact commit bytes, with no trailing newline.
        receipt["models"][name] = {"path": path, "revision": revision,
                                  "model_card": rows[name].get("model_card"),
                                  "allow_patterns": rows[name].get("allow_patterns")}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary=output.with_suffix('.tmp')
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(output)
        print(f"{name}@{revision}: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    repo = sub.add_parser("repos")
    repo.add_argument("names", nargs="+")
    asset = sub.add_parser("assets")
    asset.add_argument("names", nargs="+", help="Ancillary asset groups, e.g. roformer")
    model = sub.add_parser("models")
    model.add_argument("names", nargs="+")
    model.add_argument("--cache", type=Path, required=True)
    model.add_argument("--output", type=Path, help="Write an asset map for stack.py configure")
    args = p.parse_args()
    if args.action == "repos":
        repositories(args.names)
    elif args.action == "assets":
        assets(args.names)
    else:
        models(args.names, args.cache, args.output)


if __name__ == "__main__":
    main()
