#!/usr/bin/env bash
# Configure and build the C and C++ hosts against the minimage headers
# with no cargo and no rustc on PATH, then run both against a built
# shared library.
#
# The subproject is assembled the way packaging/wrapdb/minimage-headers.wrap
# does it: the minimage sources the headers subproject reads, with
# packaging/minimage-headers/meson.build laid over their meson.build.
#
# Usage: scripts/check-headers-only.sh <path to libminimage> [work dir]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBRARY="$(realpath "${1:?usage: check-headers-only.sh <libminimage> [work dir]}")"
WORK="${2:-$(mktemp -d)}"
mkdir -p "${WORK}"
WORK="$(realpath "${WORK}")"

HOST="${WORK}/host"
SUB="${HOST}/subprojects/minimage-headers"
mkdir -p "${SUB}/include" "${SUB}/scripts"
cp "${ROOT_DIR}/tests/headers-consumer/meson.build" "${HOST}/meson.build"
cp "${ROOT_DIR}/tests/c/abi_host.c" "${HOST}/abi_host.c"
cp "${ROOT_DIR}/tests/cpp/abi_host.cpp" "${HOST}/abi_host.cpp"
cp "${ROOT_DIR}/include/minimage.h" "${ROOT_DIR}/include/minimage_abi.hpp" "${SUB}/include/"
cp "${ROOT_DIR}/Cargo.toml" "${SUB}/Cargo.toml"
cp "${ROOT_DIR}/scripts/crate_version.py" "${SUB}/scripts/crate_version.py"
cp "${ROOT_DIR}/packaging/minimage-headers/meson.build" "${SUB}/meson.build"

# A PATH with meson, ninja, python3, and the compilers under their plain
# names and nothing else beside the system directories. The tools are
# reached through one-line wrappers, so a directory that also holds a
# Rust toolchain does not put cargo on PATH.
BARE="${WORK}/bin"
mkdir -p "${BARE}"
wrap() {
    local name="$1" real="$2"
    [[ -x "${real}" ]] || { echo "${name} not found" >&2; exit 1; }
    printf '#!/bin/sh\nexec "%s" "$@"\n' "${real}" >"${BARE}/${name}"
    chmod +x "${BARE}/${name}"
}
wrap meson "$(command -v meson)"
wrap ninja "$(command -v ninja)"
wrap python3 "$(command -v python3)"
wrap cc "$(command -v "${CC:-cc}")"
wrap c++ "$(command -v "${CXX:-c++}")"

env -i PATH="${BARE}:/usr/bin:/bin" HOME="${WORK}" CC=cc CXX=c++ \
    LDFLAGS="${LDFLAGS:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" bash -c '
    set -euo pipefail
    if command -v cargo >/dev/null 2>&1 || command -v rustc >/dev/null 2>&1; then
        echo "a Rust toolchain is still on PATH: $(command -v cargo rustc)" >&2
        exit 1
    fi
    echo "cargo on PATH: none; rustc on PATH: none"
    meson setup "$1/build" "$1" >"$1/setup.log" 2>&1 || { cat "$1/setup.log" >&2; exit 1; }
    meson compile -C "$1/build" >"$1/compile.log" 2>&1 || { cat "$1/compile.log" >&2; exit 1; }
' _ "${HOST}"
"${HOST}/build/abi_host_c" "${LIBRARY}"
"${HOST}/build/abi_host_cpp" "${LIBRARY}"
echo "headers-only hosts: ok"
