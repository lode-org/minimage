#!/usr/bin/env python3
"""Print the version from Cargo.toml, the one place it is written by hand.

`--set VERSION` first writes VERSION into [workspace.package]; cog bump runs
that before scripts/sync-version.py --release.

meson.build runs this inside project(), before find_program() exists, so it
stays on the Python that runs Meson (3.7 or newer) and has no imports past
the standard library. CMakeLists.txt runs it before project() too.
"""

import re
import sys
from pathlib import Path

CARGO_TOML = Path(__file__).resolve().parent.parent / "Cargo.toml"
VERSION_LINE = r'version\s*=\s*"([^"]+)"'


def crate_version(path=CARGO_TOML):
    section = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            section = line.strip()
        elif section in ("[workspace.package]", "[package]"):
            m = re.match(VERSION_LINE, line)
            if m:
                return m.group(1)
    sys.exit(f"no package version in {path}")


def set_crate_version(version, path=CARGO_TOML):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    section = None
    for i, line in enumerate(lines):
        if line.startswith("["):
            section = line.strip()
        elif section == "[workspace.package]":
            m = re.match(VERSION_LINE, line)
            if m:
                lines[i] = line[: m.start(1)] + version + line[m.end(1) :]
                path.write_text("".join(lines), encoding="utf-8")
                return
    sys.exit(f"no [workspace.package] version in {path}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--set":
        set_crate_version(sys.argv[2])
    elif len(sys.argv) != 1:
        sys.exit("usage: crate_version.py [--set VERSION]")
    print(crate_version())
