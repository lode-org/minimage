#!/usr/bin/env bash
# Regenerate or check the C API header (include/minimage.h) with cbindgen.
#
# The header is shipped pre-generated so downstream packagers never need
# cbindgen on the build host. Meson and CMake must not invoke this.
# Run this script when the FFI surface in src/capi.rs changes, then
# commit include/minimage.h.
#
# cbindgen parses src/ against cbindgen.toml. Do not add [parse.expand]
# crates (that runs cargo expand, a compile).
#
# Usage:
#   scripts/regen-capi-headers.sh           Regenerate the header in place.
#   scripts/regen-capi-headers.sh --check   Diff the shipped header against
#                                           fresh cbindgen output.
#
# Requires cbindgen on PATH. Install with: cargo install cbindgen
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEADER_DST="${ROOT_DIR}/include/minimage.h"
CHECK_ONLY=0

if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
fi

if ! command -v cbindgen >/dev/null 2>&1; then
    echo "cbindgen not found on PATH. Install with: cargo install cbindgen" >&2
    exit 1
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    tmp="$(mktemp)"
    trap 'rm -f "${tmp}"' EXIT
    cbindgen \
        --config "${ROOT_DIR}/cbindgen.toml" \
        --crate minimage \
        --output "${tmp}" >/dev/null
    if ! diff -u "${HEADER_DST}" "${tmp}"; then
        echo "drift detected: include/minimage.h is out of sync with cbindgen output" >&2
        echo "run: scripts/regen-capi-headers.sh" >&2
        exit 1
    fi
    echo "include/minimage.h is in sync with cbindgen output"
else
    cbindgen \
        --config "${ROOT_DIR}/cbindgen.toml" \
        --crate minimage \
        --output "${HEADER_DST}"
    echo "Regenerated ${HEADER_DST}"
fi
