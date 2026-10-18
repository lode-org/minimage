#!/usr/bin/env python3
"""Check the Meson cargo linker against the C compiler's argument syntax.

MSVC (cl, clang-cl) must not set CARGO_TARGET_<TRIPLE>_LINKER. That value
would be cl, and rustc would link build scripts with the compiler driver
(LNK2001 on memcpy, mainCRTStartup, __CxxFrameHandler3). GNU and Apple
drivers must still be passed through as the linker.

The self-test covers those three platform shapes. A build directory checks
the Meson configure that produced it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


MSVC_IDS = frozenset({"msvc", "clang-cl", "intel-cl", "intel-llvm-cl"})
LINKER_KEY = re.compile(r"^CARGO_TARGET_[A-Z0-9_]+_LINKER$")
ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


class LinkerCheckError(Exception):
    pass


def write_report(path: Path, syntax: str, host: str, key: str, cc: str) -> None:
    payload = {
        "argument_syntax": syntax,
        "host": host,
        "linker_key": key,
        "cc": cc,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def load_report(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LinkerCheckError(f"missing {path}") from exc
    except json.JSONDecodeError as exc:
        raise LinkerCheckError(f"invalid {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LinkerCheckError(f"{path} is not a JSON object")
    out: dict[str, str] = {}
    for field in ("argument_syntax", "host", "linker_key", "cc"):
        value = data.get(field)
        if not isinstance(value, str):
            raise LinkerCheckError(f"{path} missing string field {field}")
        out[field] = value
    return out


def rustc_host() -> str:
    try:
        out = subprocess.check_output(["rustc", "-vV"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LinkerCheckError(f"rustc -vV failed: {exc}") from exc
    for line in out.splitlines():
        if line.startswith("host: "):
            return line[len("host: ") :].strip()
    raise LinkerCheckError("rustc did not report its host target")


def linker_key_for(host: str) -> str:
    return "CARGO_TARGET_" + host.upper().replace("-", "_") + "_LINKER"


def ninja_split(text: str) -> list[str]:
    """Split a Ninja variable on unescaped spaces, honoring `$` escapes."""
    tokens: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "$" and i + 1 < len(text):
            cur.append(text[i + 1])
            i += 2
            continue
        if text[i] == " ":
            if cur:
                tokens.append("".join(cur))
                cur = []
            i += 1
            continue
        cur.append(text[i])
        i += 1
    if cur:
        tokens.append("".join(cur))
    return tokens


def strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("build "):
            if current:
                blocks.append(current)
            current = [line]
        elif current and (line.startswith(" ") or line.startswith("\t")):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _field(block: list[str], name: str) -> str:
    prefix = name.lower()
    for line in block:
        stripped = line.strip()
        head, sep, tail = stripped.partition("=")
        if sep and head.strip().lower() == prefix:
            return tail.lstrip()
    return ""


def linkers_from_pickle(path: Path) -> dict[str, str]:
    try:
        import mesonbuild.utils.core  # noqa: F401
        import pickle
    except ImportError as exc:
        raise LinkerCheckError(
            f"{path} holds a Meson exe pickle but mesonbuild is not importable"
        ) from exc
    try:
        obj = pickle.loads(path.read_bytes())
    except (OSError, pickle.UnpicklingError, AttributeError) as exc:
        raise LinkerCheckError(f"could not read {path}: {exc}") from exc
    env = getattr(obj, "env", None)
    if env is None:
        return {}
    got = env.get_env({})
    return {key: value for key, value in got.items() if LINKER_KEY.match(key)}


def linkers_from_command(build_dir: Path, command: str) -> dict[str, str]:
    tokens = [strip_quotes(token) for token in ninja_split(command)]
    found: dict[str, str] = {}
    if tokens and tokens[0] == "env":
        for token in tokens[1:]:
            if not ENV_ASSIGN.match(token):
                break
            key, value = token.split("=", 1)
            if LINKER_KEY.match(key):
                found[key] = value
    if "--unpickle" in tokens:
        idx = tokens.index("--unpickle")
        if idx + 1 >= len(tokens):
            raise LinkerCheckError("meson --unpickle is missing a path")
        raw = Path(tokens[idx + 1])
        path = raw if raw.is_absolute() else build_dir / raw
        found.update(linkers_from_pickle(path))
    return found


def _unescape(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "$" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def cargo_linkers(build_dir: Path) -> dict[str, str]:
    ninja_path = build_dir / "build.ninja"
    try:
        text = ninja_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise LinkerCheckError(f"missing {ninja_path}") from exc
    for block in split_blocks(text):
        if "minimage-cargo" not in _unescape(_field(block, "description")):
            continue
        return linkers_from_command(build_dir, _field(block, "COMMAND"))
    raise LinkerCheckError("minimage-cargo custom target not found in build.ninja")


def compiler_facts(build_dir: Path) -> tuple[str, list[str]]:
    path = build_dir / "meson-info" / "intro-compilers.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LinkerCheckError(f"missing {path}") from exc
    except json.JSONDecodeError as exc:
        raise LinkerCheckError(f"invalid {path}: {exc}") from exc
    try:
        compiler = data["host"]["c"]
        cid = compiler["id"]
        exelist = compiler["exelist"]
    except (KeyError, TypeError) as exc:
        raise LinkerCheckError(f"{path} has no host C compiler") from exc
    if not isinstance(cid, str) or not isinstance(exelist, list) or not exelist:
        raise LinkerCheckError(f"{path} has no host C compiler id and exelist")
    if not all(isinstance(item, str) for item in exelist):
        raise LinkerCheckError(f"{path} exelist is not a list of strings")
    return cid, exelist


def check_build(
    build_dir: Path,
    require_syntax: str | None = None,
    *,
    rustc: bool = True,
    quiet: bool = False,
) -> None:
    report = load_report(build_dir / "cargo-linker.env.json")
    syntax = report["argument_syntax"]
    host = report["host"]
    key = report["linker_key"]
    cc = report["cc"]
    if syntax not in {"gcc", "msvc"}:
        raise LinkerCheckError(f"unexpected argument syntax {syntax!r}")
    if require_syntax is not None and syntax != require_syntax:
        raise LinkerCheckError(
            f"required argument syntax {require_syntax}, Meson reported {syntax}"
        )
    expected_key = linker_key_for(host)
    if key != expected_key:
        raise LinkerCheckError(f"linker key {key} != {expected_key}")
    cid, exelist = compiler_facts(build_dir)
    inferred = "msvc" if cid in MSVC_IDS else "gcc"
    if inferred != syntax:
        raise LinkerCheckError(
            f"compiler id {cid} implies {inferred}, Meson reported {syntax}"
        )
    if exelist[-1] != cc:
        raise LinkerCheckError(f"cc command {cc!r} != exelist[-1] {exelist[-1]!r}")
    if rustc:
        live = rustc_host()
        if live != host:
            raise LinkerCheckError(f"rustc host {live} != Meson host {host}")
    actual = cargo_linkers(build_dir)
    if syntax == "msvc":
        if actual:
            detail = ", ".join(f"{name}={value}" for name, value in sorted(actual.items()))
            raise LinkerCheckError(f"MSVC set CARGO_TARGET_*_LINKER to {detail}")
    else:
        expected = {key: cc}
        if actual != expected:
            raise LinkerCheckError(f"linker env {actual} != {expected}")
    if not quiet:
        if syntax == "msvc":
            print(f"cargo linker: unset ({syntax}, compiler {cid})")
        else:
            print(f"cargo linker: {key}={cc} ({syntax}, compiler {cid})")


def _stage(
    root: Path,
    report: dict[str, str],
    compiler_id: str,
    exelist: list[str],
    command: str,
) -> None:
    (root / "meson-info").mkdir(parents=True)
    (root / "cargo-linker.env.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    (root / "meson-info" / "intro-compilers.json").write_text(
        json.dumps({"host": {"c": {"id": compiler_id, "exelist": exelist}}}) + "\n",
        encoding="utf-8",
    )
    decoy = "env CARGO_TARGET_WRONG_LINKER=nope /usr/bin/python3 -c 'skip'"
    ninja = (
        "rule CUSTOM_COMMAND\n"
        " command = $COMMAND\n"
        "\n"
        "build decoy: CUSTOM_COMMAND\n"
        f" COMMAND = {decoy}\n"
        " description = Generating$ something$ else\n"
        "\n"
        "build libminimage.so libminimage.a: CUSTOM_COMMAND\n"
        f" COMMAND = {command}\n"
        " description = Generating$ minimage-cargo$ with$ a$ custom$ command\n"
    )
    (root / "build.ninja").write_text(ninja, encoding="utf-8")


def self_test() -> int:
    linux_key = linker_key_for("x86_64-unknown-linux-gnu")
    macos_key = linker_key_for("aarch64-apple-darwin")
    win_key = linker_key_for("x86_64-pc-windows-msvc")
    arm_key = linker_key_for("aarch64-pc-windows-msvc")
    mingw_key = linker_key_for("x86_64-pc-windows-gnu")
    linux_cmd = (
        f"env {linux_key}=gcc /usr/bin/python3 -c "
        "'import$ shutil,$ subprocess'"
    )
    cases: list[dict[str, object]] = [
        {
            "name": "linux-gcc",
            "ok": True,
            "require": "gcc",
            "report": {
                "argument_syntax": "gcc",
                "host": "x86_64-unknown-linux-gnu",
                "linker_key": linux_key,
                "cc": "gcc",
            },
            "id": "gcc",
            "exelist": ["gcc"],
            "command": linux_cmd,
        },
        {
            "name": "linux-ccache",
            "ok": True,
            "require": "gcc",
            "report": {
                "argument_syntax": "gcc",
                "host": "x86_64-unknown-linux-gnu",
                "linker_key": linux_key,
                "cc": "gcc",
            },
            "id": "gcc",
            "exelist": ["ccache", "gcc"],
            "command": linux_cmd,
        },
        {
            "name": "macos",
            "ok": True,
            "require": "gcc",
            "report": {
                "argument_syntax": "gcc",
                "host": "aarch64-apple-darwin",
                "linker_key": macos_key,
                "cc": "cc",
            },
            "id": "clang",
            "exelist": ["cc"],
            "command": f"env {macos_key}=cc /usr/bin/python3 -c 'ok'",
        },
        {
            "name": "windows-gnu",
            "ok": True,
            "require": "gcc",
            "report": {
                "argument_syntax": "gcc",
                "host": "x86_64-pc-windows-gnu",
                "linker_key": mingw_key,
                "cc": "x86_64-w64-mingw32-gcc",
            },
            "id": "gcc",
            "exelist": ["x86_64-w64-mingw32-gcc"],
            "command": f"env {mingw_key}=x86_64-w64-mingw32-gcc python -c ok",
        },
        {
            "name": "windows-msvc",
            "ok": True,
            "require": "msvc",
            "report": {
                "argument_syntax": "msvc",
                "host": "x86_64-pc-windows-msvc",
                "linker_key": win_key,
                "cc": "cl",
            },
            "id": "msvc",
            "exelist": ["cl"],
            "command": '"C:/Python/python.exe" "-c" "print(1)"',
        },
        {
            "name": "windows-clang-cl",
            "ok": True,
            "require": "msvc",
            "report": {
                "argument_syntax": "msvc",
                "host": "aarch64-pc-windows-msvc",
                "linker_key": arm_key,
                "cc": "clang-cl",
            },
            "id": "clang-cl",
            "exelist": ["clang-cl"],
            "command": "C:/Python/python.exe -c print(1)",
        },
        {
            "name": "windows-msvc-rejects-cl",
            "ok": False,
            "error": "MSVC set CARGO_TARGET_*_LINKER",
            "report": {
                "argument_syntax": "msvc",
                "host": "x86_64-pc-windows-msvc",
                "linker_key": win_key,
                "cc": "cl",
            },
            "id": "msvc",
            "exelist": ["cl"],
            "command": f'"env" "{win_key}=cl" "C:/Python/python.exe"',
        },
        {
            "name": "linux-missing-linker",
            "ok": False,
            "error": "linker env",
            "report": {
                "argument_syntax": "gcc",
                "host": "x86_64-unknown-linux-gnu",
                "linker_key": linux_key,
                "cc": "gcc",
            },
            "id": "gcc",
            "exelist": ["gcc"],
            "command": "/usr/bin/python3 -c 'no-env'",
        },
        {
            "name": "linux-script-text-is-not-env",
            "ok": False,
            "error": "linker env",
            "report": {
                "argument_syntax": "gcc",
                "host": "x86_64-unknown-linux-gnu",
                "linker_key": linux_key,
                "cc": "gcc",
            },
            "id": "gcc",
            "exelist": ["gcc"],
            "command": f"/usr/bin/python3 -c '{linux_key}=gcc'",
        },
        {
            "name": "linux-wrong-driver",
            "ok": False,
            "error": "linker env",
            "report": {
                "argument_syntax": "gcc",
                "host": "x86_64-unknown-linux-gnu",
                "linker_key": linux_key,
                "cc": "gcc",
            },
            "id": "gcc",
            "exelist": ["gcc"],
            "command": f"env {linux_key}=cc /usr/bin/python3 -c ok",
        },
        {
            "name": "msvc-build-is-not-gcc",
            "ok": False,
            "error": "required argument syntax gcc",
            "require": "gcc",
            "report": {
                "argument_syntax": "msvc",
                "host": "x86_64-pc-windows-msvc",
                "linker_key": win_key,
                "cc": "cl",
            },
            "id": "msvc",
            "exelist": ["cl"],
            "command": "python -c ok",
        },
        {
            "name": "syntax-disagrees-with-compiler-id",
            "ok": False,
            "error": "compiler id gcc implies gcc",
            "report": {
                "argument_syntax": "msvc",
                "host": "x86_64-unknown-linux-gnu",
                "linker_key": linux_key,
                "cc": "gcc",
            },
            "id": "gcc",
            "exelist": ["gcc"],
            "command": "/usr/bin/python3 -c ok",
        },
    ]
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = case["report"]
            assert isinstance(report, dict)
            _stage(
                root,
                report,
                str(case["id"]),
                list(case["exelist"]),  # type: ignore[arg-type]
                str(case["command"]),
            )
            require = case.get("require")
            try:
                check_build(
                    root,
                    str(require) if require else None,
                    rustc=False,
                    quiet=True,
                )
            except LinkerCheckError as exc:
                if case["ok"]:
                    raise LinkerCheckError(f"{case['name']}: unexpected failure: {exc}") from exc
                expected = case.get("error")
                if expected and str(expected) not in str(exc):
                    raise LinkerCheckError(
                        f"{case['name']}: {exc} does not mention {expected}"
                    ) from exc
            else:
                if not case["ok"]:
                    raise LinkerCheckError(f"{case['name']}: expected failure")
    extra = _self_test_pickle()
    passed = len(cases) + extra
    print(f"cargo-linker self-test: {passed} passed")
    return passed


def _pickle_command(blob: bytes, root: Path) -> str:
    private = root / "meson-private"
    private.mkdir()
    (private / "meson_exe_python.dat").write_bytes(blob)
    return "/usr/bin/meson --internal exe --unpickle meson-private/meson_exe_python.dat"


def _self_test_pickle() -> int:
    """Meson pickles the env when `env` is not on PATH (typical on Windows)."""
    try:
        import pickle

        from mesonbuild.utils.core import EnvironmentVariables, ExecutableSerialisation
    except ImportError:
        return 0

    def dumped(host: str, value: str) -> bytes:
        key = linker_key_for(host)
        env = EnvironmentVariables({key: [value]})
        return pickle.dumps(ExecutableSerialisation(["python"], env=env))

    linux = "x86_64-unknown-linux-gnu"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        command = _pickle_command(dumped(linux, "gcc"), root)
        _stage(
            root,
            {
                "argument_syntax": "gcc",
                "host": linux,
                "linker_key": linker_key_for(linux),
                "cc": "gcc",
            },
            "gcc",
            ["gcc"],
            command,
        )
        check_build(root, "gcc", rustc=False, quiet=True)

    windows = "x86_64-pc-windows-msvc"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        command = _pickle_command(dumped(windows, "cl"), root)
        _stage(
            root,
            {
                "argument_syntax": "msvc",
                "host": windows,
                "linker_key": linker_key_for(windows),
                "cc": "cl",
            },
            "msvc",
            ["cl"],
            command,
        )
        try:
            check_build(root, "msvc", rustc=False, quiet=True)
        except LinkerCheckError as exc:
            if "MSVC set CARGO_TARGET_*_LINKER" not in str(exc):
                raise LinkerCheckError(f"windows-msvc-pickled-cl: {exc}") from exc
        else:
            raise LinkerCheckError("windows-msvc-pickled-cl: expected failure")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-report",
        nargs=5,
        metavar=("PATH", "SYNTAX", "HOST", "KEY", "CC"),
        help="write the configure-time linker report and exit",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-syntax", choices=("gcc", "msvc"))
    parser.add_argument("builddir", nargs="?")
    args = parser.parse_args(argv)
    try:
        if args.write_report is not None:
            path, syntax, host, key, cc = args.write_report
            write_report(Path(path), syntax, host, key, cc)
            return 0
        if args.require_syntax and not args.builddir:
            parser.error("--require-syntax needs a build directory")
        self_test()
        if args.builddir:
            check_build(Path(args.builddir), args.require_syntax)
        elif not args.self_test:
            parser.error("pass a Meson build directory or --self-test")
    except LinkerCheckError as exc:
        print(f"cargo linker check: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
