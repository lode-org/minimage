#!/usr/bin/env bash
# Run SymPy, Sollya, and Lean, then settle the three ballots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/validate/out"
mkdir -p "${OUT}"

export PATH="${HOME}/.elan/bin:${HOME}/.local/bin:${PATH}"
if [[ -d "${HOME}/.local/lib" ]]; then
  export LD_LIBRARY_PATH="${HOME}/.local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

python3 "${ROOT}/validate/sympy/check_mic.py" > "${OUT}/sympy.json"

# Sollya prints a banner before the script. Keep the JSON object.
sollya "${ROOT}/validate/sollya/check_mic.sollya" \
  | python3 -c 'import sys; rows=[ln for ln in sys.stdin if ln.lstrip().startswith("{")]; sys.stdout.write(rows[-1] if rows else "")' \
  > "${OUT}/sollya.json"

(
  cd "${ROOT}/validate/lean"
  if grep -n -E '\bsorry\b' MinimageValidate.lean; then
    echo "Lean file contains sorry" >&2
    exit 1
  fi
  for theorem in wrap_mem wrap_at_half selling_preserves_sum sellingBasis_det \
    selling_drops ortho_shell skew_shell dyadic_shell; do
    grep -q "^theorem ${theorem}" MinimageValidate.lean
  done
  lake exe cache get
  lake build
)
printf '%s\n' '{"agent":"lean","choice":"accept"}' > "${OUT}/lean.json"

# ljos-consensus 0.5 requires rustc 1.89. The library stays on its own MSRV.
if cargo +stable --version >/dev/null 2>&1; then
  cargo +stable run --locked --manifest-path "${ROOT}/Cargo.toml" -p minimage-consensus -- "${OUT}"
else
  cargo run --locked --manifest-path "${ROOT}/Cargo.toml" -p minimage-consensus -- "${OUT}"
fi
