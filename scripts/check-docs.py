#!/usr/bin/env python3
"""Validate documentation links, HTML export freshness, and optional external URLs.

Python 3.10+, standard library only.
"""

from __future__ import annotations

import argparse
import html.parser
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    "index.html",
    "getting-started.html",
    "tutorial.html",
    "howto.html",
    "explanation.html",
    "simd.html",
    "reference.html",
]
ORG_ID_RE = re.compile(r"(outline-container-|text-|#)?org[0-9a-f]{7}")


class HtmlScanner(html.parser.HTMLParser):
    """Collect id attributes and href/src targets from one HTML file."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if attr_map.get("id"):
            self.ids.add(attr_map["id"])  # type: ignore[arg-type]
        if tag == "link" and attr_map.get("rel") in ("preconnect", "dns-prefetch"):
            return
        for key in ("href", "src"):
            val = attr_map.get(key)
            if val:
                self.links.append(val)


def normalize_org_html(text: str) -> str:
    return ORG_ID_RE.sub(lambda m: f"{m.group(1) or ''}orgHASH", text)


def check_internal_links(root: Path) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    external: set[str] = set()
    html_ids: dict[Path, set[str]] = {}
    html_links: dict[Path, list[str]] = {}

    for path in sorted((root / "website").glob("*.html")):
        scanner = HtmlScanner()
        scanner.feed(path.read_text(encoding="utf-8"))
        html_ids[path] = scanner.ids
        html_links[path] = scanner.links

    def resolve_web_target(base_dir: Path, raw: str, source: Path) -> None:
        if raw.startswith(("http://", "https://")):
            external.add(raw)
            return
        if raw.startswith(("mailto:", "javascript:", "data:")):
            return
        target_part, _, frag = raw.partition("#")
        if not target_part:
            if source in html_ids and frag and frag not in html_ids[source]:
                errors.append(f"{source.relative_to(root)}: missing fragment #{frag}")
            return
        cand = (base_dir / target_part).resolve()
        if not cand.exists() and base_dir == root / "website" and target_part.startswith("assets/"):
            # pages.yml copies branding assets into _site/assets/
            cand = (root / "assets" / "branding" / Path(target_part).name).resolve()
        if not cand.exists():
            errors.append(f"{source.relative_to(root)}: missing target {raw!r}")
            return
        if frag and cand in html_ids and frag not in html_ids[cand]:
            errors.append(f"{source.relative_to(root)}: missing fragment {raw!r}")

    for path, links in html_links.items():
        for raw in links:
            resolve_web_target(root / "website", raw, path)

    readme = root / "README.md"
    readme_text = readme.read_text(encoding="utf-8")
    for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme_text):
        resolve_web_target(root, raw.split()[0], readme)
    for raw in re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', readme_text):
        resolve_web_target(root, raw, readme)

    for org in sorted((root / "docs" / "orgmode").glob("*.org")):
        text = org.read_text(encoding="utf-8")
        for m in re.finditer(r"^\s*#\+setupfile:\s*(\S+)", text, re.MULTILINE):
            target = (org.parent / m.group(1)).resolve()
            if not target.exists():
                errors.append(f"{org.relative_to(root)}: missing setupfile {m.group(1)!r}")
        for m in re.finditer(r"\[\[([^\]]+?)\](?:\[[^\]]*\])?\]", text):
            raw = m.group(1)
            if raw.startswith("file:"):
                rel = raw.removeprefix("file:").split("::")[0]
                if not (org.parent / rel).resolve().exists():
                    errors.append(f"{org.relative_to(root)}: missing org file link {raw!r}")
            elif raw.startswith(("http://", "https://")):
                external.add(raw)
        for m in re.finditer(r'\b(?:href|src)="([^"]+)"', text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start : None if line_end == -1 else line_end]
            if 'rel="preconnect"' in line or 'rel="dns-prefetch"' in line:
                continue
            resolve_web_target(root / "website", m.group(1), org)

    return errors, external


def check_emacs_export(root: Path, require_emacs: bool) -> list[str]:
    emacs = shutil.which("emacs")
    if emacs is None:
        if require_emacs:
            return ["emacs is required for HTML export verification but was not found on PATH"]
        print("skip emacs export check: emacs not found on PATH", file=sys.stderr)
        return []

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        shutil.copytree(root / "docs", tmp_root / "docs")
        shutil.copytree(root / "website", tmp_root / "website")
        proc = subprocess.run(
            [emacs, "--batch", "-l", str(tmp_root / "docs" / "export.el")],
            cwd=tmp_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return [f"emacs export failed ({proc.returncode}): {proc.stderr.strip()}"]
        for page in PAGES:
            expected_path = root / "website" / page
            actual_path = tmp_root / "website" / page
            if not actual_path.exists():
                errors.append(f"emacs export did not generate website/{page}")
                continue
            if not expected_path.exists():
                errors.append(f"tracked website/{page} is missing")
                continue
            exp = normalize_org_html(expected_path.read_text(encoding="utf-8"))
            act = normalize_org_html(actual_path.read_text(encoding="utf-8"))
            if exp != act:
                errors.append(
                    f"website/{page} is out of sync with docs/orgmode (run: emacs --batch -l docs/export.el)"
                )
    return errors


def check_external_urls(urls: set[str]) -> list[str]:
    errors: list[str] = []
    headers = {"User-Agent": "minimage-docs-check/1.0"}
    for url in sorted(urls):
        if url.startswith("https://doi.org/"):
            doi = url.removeprefix("https://doi.org/")
            api_url = f"https://doi.org/api/handles/{doi}"
            req = urllib.request.Request(api_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    payload = json.load(resp)
                if resp.status != 200 or payload.get("responseCode") != 1:
                    errors.append(f"DOI handle lookup failed for {url}: {payload}")
            except Exception as err:
                errors.append(f"DOI lookup error for {url}: {err}")
            continue
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status >= 400:
                    errors.append(f"URL {url} returned HTTP {resp.status}")
        except urllib.error.HTTPError as err:
            if err.code == 405:
                req_get = urllib.request.Request(url, headers=headers, method="GET")
                try:
                    with urllib.request.urlopen(req_get, timeout=15) as resp:
                        if resp.status >= 400:
                            errors.append(f"URL {url} returned HTTP {resp.status}")
                except Exception as inner:
                    errors.append(f"URL {url} failed: {inner}")
            else:
                errors.append(f"URL {url} returned HTTP {err.code}")
        except Exception as err:
            errors.append(f"URL {url} failed: {err}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-emacs",
        action="store_true",
        help="fail if emacs is not installed instead of skipping HTML export check",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="verify external URLs and DOIs over HTTPS",
    )
    args = parser.parse_args()

    errors, external = check_internal_links(ROOT)
    errors.extend(check_emacs_export(ROOT, args.require_emacs))
    if args.external:
        errors.extend(check_external_urls(external))

    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    if errors:
        return 1
    print(
        f"check-docs: internal links OK ({len(external)} external URLs catalogued)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
