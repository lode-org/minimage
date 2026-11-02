#!/usr/bin/env python3
"""Fail when a copy of the version or of the wrap tag disagrees with Cargo.toml.

Each copy is read with a parser for its format: tomllib for TOML,
configparser for the Meson wrap and Meson's own parser for meson.build. A
reformatted line, a second version key or a pre-release tag outside the sync
patterns therefore fails here even when scripts/sync-version.py leaves the
file alone. The wrap snippets in the docs sit inside code blocks, so every
`revision` key there is read with a free-form value.

`--tag NAME` also requires NAME to be v<version> and the wrap tag, so a
release tag pushed before the release commit moved the snippets fails.

Python 3.11 or newer, standard library only. meson.build is read when the
mesonbuild package is importable, and the CI version job installs it.
"""

import argparse
import configparser
import re
import subprocess
import sys
from pathlib import Path

import tomllib
from crate_version import crate_version

try:
    from mesonbuild import mparser
except ImportError:
    mparser = None

ROOT = Path(__file__).resolve().parent.parent
WRAP = "packaging/wrapdb/minimage.wrap"
SNIPPETS = [
    "README.md",
    "docs/orgmode/getting-started.org",
    "website/getting-started.html",
]
INHERIT = {"workspace": True}


def read(root, rel):
    return (root / rel).read_text(encoding="utf-8")


def load_toml(root, rel):
    return tomllib.loads(read(root, rel))


def load_cff(root, rel):
    """Top-level scalar keys of a CFF file, with a repeated key rejected."""
    keys = {}
    for number, line in enumerate(read(root, rel).splitlines(), 1):
        if not line.strip() or line[0] in " \t#-":
            continue
        key, colon, value = line.partition(":")
        key = key.strip().strip("'\"")
        if not colon:
            raise ValueError(f"line {number} is not a key")
        if key in keys:
            raise ValueError(f"line {number} repeats the {key!r} key")
        value = value.split(" #")[0].strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        keys[key] = value
    return keys


def sources_of_truth(root):
    """The hand-written version and wrap tag in Cargo.toml."""
    workspace = load_toml(root, "Cargo.toml")["workspace"]
    return workspace["package"]["version"], workspace["metadata"]["release"]["tag"]


def cargo_toml(root, rel, version, tag):
    data = load_toml(root, rel)
    yield "[package].version", data["package"]["version"], INHERIT
    if not re.fullmatch(r"v\d+\.\d+\.\d+\S*", tag):
        raise ValueError(f"wrap tag {tag!r} is not vX.Y.Z")


def helper(root, rel, version, tag):
    try:
        found = crate_version(root / "Cargo.toml")
    except SystemExit as err:
        raise ValueError(str(err)) from None
    yield "version it reads from Cargo.toml", found, version


def python_crate(root, rel, version, tag):
    yield "[package].version", load_toml(root, rel)["package"]["version"], INHERIT


def burn_crate(root, rel, version, tag):
    data = load_toml(root, rel)
    yield "[package].version", data["package"]["version"], version
    yield "minimage dependency", data["dependencies"]["minimage"]["version"], version


def cargo_lock(root, rel, version, tag):
    packages = load_toml(root, rel)["package"]
    for name in ("minimage", "minimage-python"):
        entries = [p for p in packages if p["name"] == name]
        if len(entries) != 1:
            raise ValueError(f"{len(entries)} {name} entries, expected 1")
        yield f"{name} version", entries[0]["version"], version


def pyproject(root, rel, version, tag):
    project = load_toml(root, rel)["project"]
    yield "project.version", project.get("version"), None
    yield "version in project.dynamic", "version" in project["dynamic"], True


def cff(root, rel, version, tag):
    yield "version", load_cff(root, rel)["version"], version


def wrap(root, rel, version, tag):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(read(root, rel), source=rel)
    yield "revision", parser["wrap-git"].get("revision"), tag


def snippet(root, rel, version, tag):
    pattern = r"^\s*revision\s*[=:]\s*(.*?)\s*$"
    found = re.findall(pattern, read(root, rel), re.MULTILINE)
    if not found:
        raise ValueError("no revision line")
    for value in found:
        yield "revision", value, tag


def meson_build(root, rel, version, tag):
    project = mparser.Parser(read(root, rel), rel).parse().lines[0]
    if project.func_name.value != "project":
        raise ValueError("the first statement is not project()")
    kwargs = {key.value: value for key, value in project.args.kwargs.items()}
    node = kwargs["version"]
    if isinstance(node, mparser.StringNode):
        yield "project() version", node.value, version
        return
    methods = []
    while isinstance(node, mparser.MethodNode):
        methods.append(node.name.value)
        node = node.source_object
    args = node.args.arguments
    call = (node.func_name.value, args[0].value if args else None, methods)
    wanted = ("run_command", "scripts/crate_version.py", ["strip", "stdout"])
    yield "project() version expression", call, wanted
    out = subprocess.run(
        [sys.executable, str(root / wanted[1])],
        capture_output=True,
        check=True,
        cwd=root,
        text=True,
    )
    yield "project() version", out.stdout.strip(), version


def sources(root):
    """(file, check) for every copy of the version or the wrap tag."""
    found = [
        ("Cargo.toml", cargo_toml),
        ("scripts/crate_version.py", helper),
        ("python/Cargo.toml", python_crate),
        ("burn/Cargo.toml", burn_crate),
        ("Cargo.lock", cargo_lock),
        ("pyproject.toml", pyproject),
        ("CITATION.cff", cff),
        (WRAP, wrap),
        *((rel, snippet) for rel in SNIPPETS),
    ]
    if mparser is not None:
        found.append(("meson.build", meson_build))
    return found


def check(root=ROOT, pushed_tag=None):
    """One message per copy under root that disagrees with Cargo.toml."""
    try:
        version, tag = sources_of_truth(root)
    except (OSError, LookupError, TypeError, ValueError) as err:
        return [f"Cargo.toml: cannot read the version and wrap tag: {err!r}"]
    errors = []
    if pushed_tag is not None:
        for what, wanted in (("v<version>", f"v{version}"), ("the wrap tag", tag)):
            if pushed_tag != wanted:
                errors.append(f"tag {pushed_tag}: expected {what}, {wanted!r}")
    for rel, read_copies in sources(root):
        try:
            for what, found, wanted in read_copies(root, rel, version, tag):
                if found != wanted:
                    errors.append(f"{rel}: {what} is {found!r}, expected {wanted!r}")
        except (OSError, LookupError, TypeError, ValueError, configparser.Error) as err:
            errors.append(f"{rel}: cannot read: {err!r}")
        except subprocess.CalledProcessError as err:
            errors.append(f"{rel}: {err.cmd} failed: {err.stderr.strip()}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="a release tag that is being pushed")
    args = parser.parse_args()
    if mparser is None:
        print("skip meson.build: mesonbuild is not importable", file=sys.stderr)
    errors = check(pushed_tag=args.tag)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print("check_version_lockstep: FAILED", file=sys.stderr)
        return 1
    version, tag = sources_of_truth(ROOT)
    print(f"check_version_lockstep: every copy matches {version}, wraps name {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
