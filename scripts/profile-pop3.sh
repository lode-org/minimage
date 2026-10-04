#!/usr/bin/env bash
# Time the hot path and, when the kernel allows it, read cycles and
# instructions. The hierarchy is POP3: a leaf below 0.8 is the one to
# open. This crate is one thread, so the parallel leaves are not a
# process count.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if cargo +stable --version >/dev/null 2>&1; then
  cargo +stable bench -p minimage --bench pop3
else
  cargo bench -p minimage --bench pop3
fi
