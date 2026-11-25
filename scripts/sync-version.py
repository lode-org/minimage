#!/usr/bin/env python3
"""Rewrite every copy of the version from Cargo.toml [workspace.package].

CITATION.cff and burn/Cargo.toml, which sits outside the workspace, carry
cog blocks. Cargo.lock, codemeta.json and .zenodo.json keep no markers, so an
anchored pattern replaces any version there. The Meson wrap snippets name a
git tag, which has to exist before a wrap can fetch it, so they take
[workspace.metadata.release].tag instead of the version. Every
`find_package(minimage X.Y` line in the README and the getting started
snippets takes that tag's major.minor.
The CMake consumer test finds the package built from this tree, so its
`find_package(minimage X.Y` takes the major.minor of the version.
The lockstep check reads each of those lines. `--release` first sets that
tag to v<version>, and the CITATION.cff date-released and the codemeta.json
datePublished to today (UTC), for the release commit. CI runs this and then
`git diff --exit-code`.

Needs cogapp: python -m pip install cogapp
"""

import datetime
import re
import sys
from pathlib import Path

import tomllib
from check_version_lockstep import CONSUMER, SNIPPETS
from cogapp import Cog
from crate_version import crate_version

ROOT = Path(__file__).resolve().parent.parent
V = r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc|dev|pre)\.?\d*)?"

COG_FILES = ["CITATION.cff", "burn/Cargo.toml"]
LOCK = rf'^name = "minimage(?:-python)?"\nversion = "(?P<v>{V})"'
JSON_FILES = ["codemeta.json", ".zenodo.json"]
JSON_VERSION = rf'^  "version": "(?P<v>{V})",$'
WRAPS = [
    "README.md",
    "packaging/wrapdb/minimage.wrap",
    "docs/orgmode/getting-started.org",
    "website/getting-started.html",
]
REVISION = rf"^revision = (?P<v>v{V})$"
FIND_PACKAGE = r"find_package\(minimage (?P<v>\d+\.\d+) "
RELEASE_TAG = r'^(\[workspace\.metadata\.release\]\ntag = ")[^"]*(")'
RELEASED = r'^(date-released: ")[^"]*(")$'
PUBLISHED = r'^(  "datePublished": ")[^"]*(",)$'


def substitute(pattern, text, value):
    def put(m):
        start, end = m.span("v")
        return m.group(0)[: start - m.start()] + value + m.group(0)[end - m.start() :]

    return re.sub(pattern, put, text, flags=re.MULTILINE)


def rewrite(rel, change):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    new = change(text)
    if new != text:
        path.write_text(new, encoding="utf-8")


def set_release_tag(version):
    def change(text):
        new, count = re.subn(
            RELEASE_TAG, rf"\g<1>v{version}\g<2>", text, flags=re.MULTILINE
        )
        if count != 1:
            sys.exit("Cargo.toml: no tag line under [workspace.metadata.release]")
        return new

    rewrite("Cargo.toml", change)


def set_release_date():
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    def setter(rel, pattern, line):
        def change(text):
            new, count = re.subn(pattern, rf"\g<1>{today}\g<2>", text, flags=re.MULTILINE)
            if count != 1:
                sys.exit(f"{rel}: no {line} line")
            return new

        rewrite(rel, change)

    setter("CITATION.cff", RELEASED, 'date-released: "YYYY-MM-DD"')
    setter("codemeta.json", PUBLISHED, '"datePublished": "YYYY-MM-DD",')


def release_tag():
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return cargo["workspace"]["metadata"]["release"]["tag"]


def main(argv=()):
    version = crate_version(ROOT / "Cargo.toml")
    if "--release" in argv:
        set_release_tag(version)
        set_release_date()
    status = Cog().main(
        ["cog", "-r", "-D", f"VERSION={version}", *(str(ROOT / f) for f in COG_FILES)]
    )
    if status:
        return status
    rewrite("Cargo.lock", lambda text: substitute(LOCK, text, version))
    for rel in JSON_FILES:
        rewrite(rel, lambda text: substitute(JSON_VERSION, text, version))
    tag = release_tag()
    for rel in dict.fromkeys([*WRAPS, *SNIPPETS]):
        rewrite(rel, lambda text: substitute(REVISION, text, tag))
    minor = ".".join(tag.removeprefix("v").split(".")[:2])
    for rel in SNIPPETS:
        rewrite(rel, lambda text: substitute(FIND_PACKAGE, text, minor))
    built = ".".join(version.split(".")[:2])
    rewrite(CONSUMER, lambda text: substitute(FIND_PACKAGE, text, built))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
