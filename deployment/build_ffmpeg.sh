#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

prefix=${XEDIT_FFMPEG_PREFIX:-/opt/xedit-ffmpeg}
jobs=${XEDIT_FFMPEG_JOBS:-4}
version=8.1.2
archive=ffmpeg-${version}.tar.xz
checksum=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
test "${prefix:0:1}" = / || { echo 'XEDIT_FFMPEG_PREFIX must be absolute' >&2; exit 1; }
test ! -e "$prefix" || { echo "Refusing to replace $prefix" >&2; exit 1; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl --fail --location --retry 3 --proto '=https' --tlsv1.2 \
  "https://ffmpeg.org/releases/$archive" --output "$work/$archive"
printf '%s  %s\n' "$checksum" "$work/$archive" | sha256sum --check --status
tar -xJf "$work/$archive" -C "$work"
cd "$work/ffmpeg-$version"
./configure --prefix="$prefix" --disable-autodetect --disable-gpl \
  --disable-nonfree --disable-version3 --disable-doc --disable-debug \
  --disable-ffplay --disable-x86asm --enable-shared --disable-static \
  --extra-ldflags="-Wl,-rpath,$prefix/lib"
make -j "$jobs"
make install
notices="$prefix/share/xedit-ffmpeg"
mkdir -p "$notices"
cp "$work/$archive" "$notices/"
cp COPYING.LGPLv2.1 LICENSE.md "$notices/"
printf '%s  %s\n' "$checksum" "$archive" > "$notices/SHA256SUMS"
printf '%s\n' "https://ffmpeg.org/releases/$archive" > "$notices/SOURCE_URL"
cp ffbuild/config.log config.h config_components.h "$notices/"
for executable in ffmpeg ffprobe; do
  "$prefix/bin/$executable" -hide_banner -buildconf > "$notices/$executable-buildconf.txt" 2>&1
  "$prefix/bin/$executable" -hide_banner -L > "$notices/$executable-license.txt" 2>&1
done
python3 "$script_dir/check_ffmpeg.py" --prefix "$prefix"
