#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Ubuntu/Debian base-image preparation. Run as root in a dedicated container.
test "$(id -u)" = 0 || { echo 'Run as root in the target container.' >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
# GPU container runtimes can bind-mount driver GL libraries read-only. dpkg
# cannot rename/back up these files across mounts. Preserve only those exact
# mounted paths; do not unmount or overwrite the host-supplied driver libraries.
# These options apply to this invocation, not to the resulting Docker image.
apt_options=(-o Acquire::Retries=2 -o Acquire::http::Timeout=30)
while IFS= read -r library; do
  case "$library" in
    /usr/lib/*/libGL*.so*|/usr/lib/*/libEGL*.so*|/usr/lib/*/libGLES*.so*)
      test -f "$library" || continue
      echo "Preserving container-mounted GPU library: $library"
      apt_options+=(-o "Dpkg::Options::=--path-exclude=$library")
      ;;
  esac
done < <(awk '{print $5}' /proc/self/mountinfo)
apt-get "${apt_options[@]}" update
apt-get "${apt_options[@]}" install -y --no-install-recommends \
  ca-certificates curl xz-utils git git-lfs sox libsndfile1 libsndfile1-dev \
  libgomp1 libgl1 build-essential pkg-config python3 python3-pip python3-dev
bash "$script_dir/build_ffmpeg.sh"
python3 -m pip install --no-cache-dir uv==0.11.28
uv python install 3.11.15 3.10.12
