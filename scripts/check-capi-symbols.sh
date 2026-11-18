#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${root}/target}"

cargo build --manifest-path "${root}/Cargo.toml" --lib \
  --no-default-features --features capi

case "$(uname -s)" in
  Darwin) library="${target_dir}/debug/libminimage.dylib" ;;
  Linux) library="${target_dir}/debug/libminimage.so" ;;
  *)
    echo "unsupported platform for C ABI symbol check: $(uname -s)" >&2
    exit 1
    ;;
esac

symbols="$(nm -g --defined-only "${library}" | awk '{print $NF}')"
printf '%s\n' "${symbols}" | awk '/^minimage_/ { print }'

if printf '%s\n' "${symbols}" | awk '/^mi_/ { found = 1 } END { exit !found }'; then
  echo "unexpected unprefixed C ABI symbol:" >&2
  printf '%s\n' "${symbols}" | awk '/^mi_/ { print }' >&2
  exit 1
fi
