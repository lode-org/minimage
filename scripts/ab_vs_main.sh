#!/usr/bin/env bash
# Time this tree against published 0.1.2 (origin/main) on the calls both export.
# Fails when a gated query region is not at least 5% faster here.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if ! git rev-parse --verify origin/main >/dev/null 2>&1; then
  git fetch --depth 1 origin main
fi

cargo_bin() {
  if command -v rustup >/dev/null 2>&1 && rustup toolchain list | grep -q '^stable'; then
    echo "cargo +stable"
  else
    echo "cargo"
  fi
}
CARGO="$(cargo_bin)"

wt="${AB_MAIN_WORKTREE:-/tmp/minimage-ab-main}"
if [[ ! -d "$wt/.git" && ! -f "$wt/.git" ]]; then
  rm -rf "$wt"
  git worktree add --detach "$wt" origin/main
fi

echo "building HEAD" >&2
# shellcheck disable=SC2086
$CARGO build -p minimage --release --locked >&2
echo "building origin/main" >&2
(
  cd "$wt"
  # shellcheck disable=SC2086
  $CARGO build -p minimage --release --locked >&2
)

stage() {
  local name="$1"
  local src="$2"
  local dir
  dir="$(mktemp -d "/tmp/ab-${name}.XXXXXX")"
  mkdir -p "$dir/src"
  cp "$root/scripts/ab_driver.rs" "$dir/src/main.rs"
  cat >"$dir/Cargo.toml" <<EOF
[package]
name = "ab-driver"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
minimage = { path = "$src" }

[profile.release]
lto = true
codegen-units = 1
EOF
  (
    cd "$dir"
    # shellcheck disable=SC2086
    $CARGO build --release --offline >&2 || $CARGO build --release >&2
  )
  echo "$dir/target/release/ab-driver"
}

head_bin="$(stage head "$root")"
main_bin="$(stage main "$wt")"

cc_one() {
  local label="$1"
  local tree="$2"
  local out="/tmp/ab-c-${label}"
  gcc -O2 -std=c11 -o "$out" "$root/scripts/ab_c.c" \
    -I "$tree/include" \
    -L "$tree/target/release" \
    -lminimage \
    -Wl,-rpath,"$tree/target/release" \
    -lm
  echo "$out"
}

head_c="$(cc_one head "$root")"
main_c="$(cc_one main "$wt")"

median_run() {
  local bin="$1"
  local dest="$2"
  local tmp
  tmp="$(mktemp)"
  for _ in 1 2 3; do
    "$bin" >>"$tmp"
  done
  python3 - "$tmp" "$dest" <<'PY'
import collections, statistics, sys
src, dest = sys.argv[1], sys.argv[2]
groups = collections.defaultdict(list)
for line in open(src):
    if "ns_per_pair=" not in line:
        continue
    name, rest = line.strip().split(" ns_per_pair=", 1)
    groups[name].append(float(rest.split()[0]))
with open(dest, "w", encoding="utf-8") as fh:
    for name, vals in groups.items():
        fh.write(f"{name} ns_per_pair={statistics.median(vals):.6f}\n")
PY
}

head_out="$(mktemp)"
main_out="$(mktemp)"
echo "timing HEAD" >&2
median_run "$head_bin" "$head_out"
median_run "$head_c" "$head_out.c"
cat "$head_out.c" >>"$head_out"
echo "timing origin/main" >&2
median_run "$main_bin" "$main_out"
median_run "$main_c" "$main_out.c"
cat "$main_out.c" >>"$main_out"

python3 - "$main_out" "$head_out" <<'PY'
import sys
main_path, head_path = sys.argv[1], sys.argv[2]

def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if "ns_per_pair=" not in line:
            continue
        name, rest = line.strip().split(" ns_per_pair=", 1)
        out[name] = float(rest.split()[0])
    return out

main = load(main_path)
head = load(head_path)
gated = {
    "ortho_many",
    "ortho_pairs",
    "eucl_far",
    "c_dist2",
    "c_dist2_many",
    "c_dist2_pairs",
}
print(f"{'region':<16} {'main_ns':>12} {'head_ns':>12} {'head/main':>10} {'gate':>8}")
failed = []
for name in sorted(set(main) | set(head)):
    m = main.get(name)
    h = head.get(name)
    if m is None or h is None or m <= 0.0:
        print(f"{name:<16} missing")
        failed.append(name)
        continue
    ratio = h / m
    gate = "report"
    if name in gated:
        gate = "pass" if ratio <= 0.95 else "FAIL"
        if gate == "FAIL":
            failed.append(name)
    print(f"{name:<16} {m:12.4f} {h:12.4f} {ratio:10.3f} {gate:>8}")
if failed:
    print("slower than 0.1.2 on: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
PY
