#!/usr/bin/env bash
# Verify a Meson or CMake install of minimage: headers, cdylib, pkg-config,
# and find_package(minimage) / a tiny consumer configure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MESON_BUILD="${MINIMAGE_MESON_BUILD:-$ROOT/build-meson}"
CMAKE_BUILD="${MINIMAGE_CMAKE_BUILD:-$ROOT/build-cmake}"
MESON_DEST="${MINIMAGE_MESON_DEST:-$ROOT/destdir-meson}"
CMAKE_PREFIX="${MINIMAGE_CMAKE_PREFIX:-$ROOT/prefix-cmake}"

cargo_ver="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
if [[ -z "$cargo_ver" ]]; then
  echo "could not read version from Cargo.toml"
  exit 1
fi

have_lib() {
  local prefix="$1" stem="$2"
  local f
  for f in \
    "$prefix/lib/${stem}" \
    "$prefix/lib64/${stem}" \
    "$prefix/lib/x86_64-linux-gnu/${stem}" \
    "$prefix/bin/${stem}"; do
    if [[ -f "$f" || -L "$f" ]]; then
      return 0
    fi
  done
  return 1
}

find_pc_dir() {
  local prefix="$1" cand
  for cand in \
    "$prefix/lib/pkgconfig" \
    "$prefix/lib64/pkgconfig" \
    "$prefix/lib/x86_64-linux-gnu/pkgconfig" \
    "$prefix/share/pkgconfig"; do
    if [[ -f "$cand/minimage.pc" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

check_pkgconfig() {
  local pcdir="$1" label="$2"
  export PKG_CONFIG_PATH="$pcdir${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  pkg-config --exists --print-errors minimage
  local ver cflags libs
  ver="$(pkg-config --modversion minimage)"
  cflags="$(pkg-config --cflags minimage)"
  libs="$(pkg-config --libs minimage)"
  echo "$label pkg-config: $ver"
  echo "$label cflags: $cflags"
  echo "$label libs: $libs"
  if [[ "$ver" != "$cargo_ver" ]]; then
    echo "$label pkg-config version $ver != Cargo.toml $cargo_ver"
    exit 1
  fi
  case "$cflags" in
    *-I*) ;;
    *)
      echo "$label Cflags missing -I: $cflags"
      exit 1
      ;;
  esac
  case "$libs" in
    *-lminimage*) ;;
    *)
      echo "$label Libs must contain -lminimage: $libs"
      exit 1
      ;;
  esac
  unset PKG_CONFIG_PATH
}

check_headers() {
  local prefix="$1"
  if [[ ! -f "$prefix/include/minimage.h" ]]; then
    echo "missing $prefix/include/minimage.h"
    exit 1
  fi
  if [[ ! -f "$prefix/include/minimage.hpp" ]]; then
    echo "missing $prefix/include/minimage.hpp"
    exit 1
  fi
}

if [[ "${MINIMAGE_SKIP_BUILD:-0}" != "1" ]]; then
  echo "=== meson ==="
  meson setup "$MESON_BUILD" --wipe
  meson compile -C "$MESON_BUILD"
  meson test -C "$MESON_BUILD" --print-errorlogs
  rm -rf "$MESON_DEST"
  meson install -C "$MESON_BUILD" --destdir "$MESON_DEST"
fi

prefix_meson=""
for cand in \
  "$MESON_DEST/usr/local" \
  "$MESON_DEST/usr" \
  "$MESON_DEST"; do
  if [[ -f "$cand/include/minimage.h" ]]; then
    prefix_meson="$cand"
    break
  fi
done
if [[ -z "$prefix_meson" ]]; then
  echo "meson install did not write include/minimage.h under $MESON_DEST"
  exit 1
fi
check_headers "$prefix_meson"

pc_meson="$(find_pc_dir "$prefix_meson" || true)"
if [[ -z "$pc_meson" ]]; then
  echo "meson install did not write minimage.pc"
  exit 1
fi
check_pkgconfig "$pc_meson" "meson"

if [[ "${MINIMAGE_SKIP_BUILD:-0}" != "1" ]]; then
  echo "=== cmake ==="
  cmake -S "$ROOT" -B "$CMAKE_BUILD" -DCMAKE_INSTALL_PREFIX="$CMAKE_PREFIX"
  cmake --build "$CMAKE_BUILD"
  ctest --test-dir "$CMAKE_BUILD" --output-on-failure
  rm -rf "$CMAKE_PREFIX"
  cmake --install "$CMAKE_BUILD"
fi

check_headers "$CMAKE_PREFIX"
if [[ ! -f "$CMAKE_PREFIX/lib/cmake/minimage/minimageConfig.cmake" \
   && ! -f "$CMAKE_PREFIX/lib64/cmake/minimage/minimageConfig.cmake" ]]; then
  echo "cmake install did not write minimageConfig.cmake"
  exit 1
fi

echo "=== cmake consumer configure ==="
cmake -S "$ROOT/tests/cmake-consumer" -B "$CMAKE_BUILD/find-consumer" \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
  -DMINIMAGE_EXAMPLE="$ROOT/examples/wrap_pair.cpp"

echo "OK"
