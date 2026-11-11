"""Check the separately built FFmpeg distribution and its source notices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_binary(executable):
    executable = Path(executable).resolve(strict=True)
    configuration = subprocess.check_output(
        [str(executable), '-hide_banner', '-buildconf'], stderr=subprocess.STDOUT, text=True
    )
    license_text = subprocess.check_output(
        [str(executable), '-hide_banner', '-L'], stderr=subprocess.STDOUT, text=True
    )
    arguments = shlex.split(configuration)
    forbidden = sorted(set(arguments) & {'--enable-gpl', '--enable-nonfree', '--enable-version3'})
    normalized_license = ' '.join(license_text.split())
    permitted = ('GNU Lesser General Public License' in normalized_license
                 and 'version 2.1' in normalized_license)
    return {
        'executable': str(executable),
        'sha256': sha256(executable),
        'configuration': configuration,
        'license': license_text,
        'forbidden_flags': forbidden,
        'passed': not forbidden and permitted,
    }


def inspect_distribution(prefix):
    prefix = Path(prefix).resolve(strict=True)
    notices = prefix / 'share/xedit-ffmpeg'
    binaries = [inspect_binary(prefix / 'bin' / name) for name in ('ffmpeg', 'ffprobe')]
    expected = {'--disable-autodetect', '--disable-gpl', '--disable-nonfree',
                '--disable-version3', '--enable-shared', '--disable-static'}
    flags_valid = all(expected <= set(shlex.split(row['configuration'])) for row in binaries)
    source_valid = False
    manifest = notices / 'SHA256SUMS'
    if manifest.is_file():
        fields = manifest.read_text().split()
        if len(fields) == 2 and Path(fields[1]).name == fields[1]:
            archive = notices / fields[1]
            if archive.is_file():
                source_valid = sha256(archive) == fields[0]
    notice_files = ('COPYING.LGPLv2.1', 'LICENSE.md', 'SOURCE_URL', 'config.log',
                    'config.h', 'config_components.h', 'ffmpeg-buildconf.txt',
                    'ffprobe-buildconf.txt', 'ffmpeg-license.txt', 'ffprobe-license.txt')
    missing = [name for name in notice_files if not (notices / name).is_file()]
    return {
        'schema_version': 'x-edit.ffmpeg-distribution/v1',
        'passed': all(row['passed'] for row in binaries) and flags_valid and source_valid and not missing,
        'binaries': binaries,
        'required_flags_present': flags_valid,
        'corresponding_source_hash_valid': source_valid,
        'missing_notices': missing,
        'scope': 'Declared FFmpeg prefix only; does not certify unrelated system or Python-package libraries.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prefix', type=Path)
    group.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect_distribution(args.prefix) if args.prefix else inspect_binary(args.binary)
    serialized = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
