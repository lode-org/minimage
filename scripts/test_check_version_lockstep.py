#!/usr/bin/env python3
"""Drift that the sync patterns leave alone still fails the parser check.

Each case edits a copy of the tracked files the check reads. When cogapp is
importable, each case also runs scripts/sync-version.py on that copy and
asserts that the edit survives, so `git diff --exit-code` alone would pass it.
"""

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_version_lockstep as lockstep

try:
    import cogapp  # noqa: F401
except ImportError:
    sync = None
else:
    spec = importlib.util.spec_from_file_location(
        "sync_version", lockstep.ROOT / "scripts" / "sync-version.py"
    )
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)


class Drift(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for rel, _ in lockstep.sources(lockstep.ROOT):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lockstep.ROOT / rel, self.root / rel)
        self.v, self.tag = lockstep.sources_of_truth(self.root)

    def text(self, rel):
        return (self.root / rel).read_text(encoding="utf-8")

    def edit(self, rel, old, new):
        text = self.text(rel)
        self.assertIn(old, text)
        (self.root / rel).write_text(text.replace(old, new, 1), encoding="utf-8")

    def run_sync(self, *argv):
        sync.ROOT = self.root
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sync.main(argv), 0)

    def assertFails(self, rel):
        if sync is not None:
            before = self.text(rel)
            self.run_sync()
            self.assertEqual(self.text(rel), before, "the sync rewrote the edit")
        errors = lockstep.check(self.root)
        self.assertTrue(any(e.startswith(f"{rel}: ") for e in errors), errors)

    def test_clean_copy_passes(self):
        self.assertEqual(lockstep.check(self.root), [])

    def test_reformatted_readme_revision(self):
        line = f"revision = {self.tag}\n"
        self.edit("README.md", line, "  revision=v0.1.3\n")
        self.assertFails("README.md")

    def test_reformatted_cargo_lock(self):
        entry = f'name = "minimage"\nversion = "{self.v}"'
        self.edit("Cargo.lock", entry, "name = 'minimage'\nversion = '9.9.9'")
        self.assertFails("Cargo.lock")

    def test_second_version_key_in_burn_crate(self):
        rel = "burn/Cargo.toml"
        self.edit(rel, 'edition = "2021"', 'edition = "2021"\nversion = "8.8.8"')
        self.assertFails(rel)

    def test_second_version_key_in_citation(self):
        self.edit("CITATION.cff", "license: MIT\n", "license: MIT\nversion: 8.8.8\n")
        self.assertFails("CITATION.cff")

    def test_off_allowlist_tag_in_wrap(self):
        line = f"revision = {self.tag}\n"
        self.edit(lockstep.WRAP, line, f"revision = {self.tag}-vendor\n")
        self.assertFails(lockstep.WRAP)

    def test_off_allowlist_tag_in_cargo_lock(self):
        entry = f'name = "minimage-python"\nversion = "{self.v}"'
        self.edit("Cargo.lock", entry, f'{entry[:-1]}+local"')
        self.assertFails("Cargo.lock")

    def test_release_tag_needs_the_release_commit(self):
        errors = lockstep.check(self.root, pushed_tag=f"v{self.v}")
        self.assertIn(f"tag v{self.v}: expected the wrap tag, {self.tag!r}", errors)

    @unittest.skipIf(sync is None, "needs cogapp")
    def test_release_sync_moves_every_snippet(self):
        self.run_sync("--release")
        for rel in [lockstep.WRAP, *lockstep.SNIPPETS]:
            self.assertIn(f"revision = v{self.v}\n", self.text(rel))
        self.assertEqual(lockstep.check(self.root, pushed_tag=f"v{self.v}"), [])


if __name__ == "__main__":
    unittest.main()
