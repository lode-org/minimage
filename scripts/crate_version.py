#!/usr/bin/env python3
"""Print the version from Cargo.toml, the one place it is written by hand.

meson.build runs this inside project(), before find_program() exists, so it
stays on the Python that runs Meson (3.7 or newer) and has no imports past
the standard library.
"""

import re
import sys
from pathlib import Path

CARGO_TOML = Path(__file__).resolve().parent.parent / "Cargo.toml"


def crate_version(path=CARGO_TOML):
    section = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            section = line.strip()
        elif section in ("[workspace.package]", "[package]"):
            m = re.match(r'version\s*=\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    sys.exit(f"no package version in {path}")


if __name__ == "__main__":
    print(crate_version())
