#!/usr/bin/env python3
"""Check consistency of QUARTIC2D release metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src" / "quartic2d" / "_version.py"
CITATION_FILE = ROOT / "CITATION.cff"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def _match(pattern: str, text: str, *, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise SystemExit(f"Could not read {label}.")
    return match.group(1).strip().strip('"').strip("'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        help="release tag to validate, for example v0.1.0; release-mode checks are enabled when supplied",
    )
    args = parser.parse_args()

    version_text = VERSION_FILE.read_text()
    citation_text = CITATION_FILE.read_text()
    changelog_text = CHANGELOG_FILE.read_text()

    version = _match(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        version_text,
        label="src/quartic2d/_version.py version",
    )
    citation_version = _match(
        r"^version:\s*(.+)$",
        citation_text,
        label="CITATION.cff version",
    )

    if citation_version != version:
        raise SystemExit(
            f"Version mismatch: package={version!r}, CITATION.cff={citation_version!r}."
        )

    changelog_match = re.search(
        rf"^## \[{re.escape(version)}\] - (.+)$",
        changelog_text,
        flags=re.MULTILINE,
    )
    if changelog_match is None:
        raise SystemExit(f"CHANGELOG.md has no heading for version {version}.")

    changelog_state = changelog_match.group(1).strip()

    if args.tag is None:
        print(f"package version : {version}")
        print(f"citation version: {citation_version}")
        print(f"changelog state : {changelog_state}")
        print("release metadata: consistent")
        return

    expected_tag = f"v{version}"
    if args.tag != expected_tag:
        raise SystemExit(
            f"Tag mismatch: supplied {args.tag!r}, expected {expected_tag!r}."
        )

    date_pattern = r"\d{4}-\d{2}-\d{2}"
    if not re.fullmatch(date_pattern, changelog_state):
        raise SystemExit(
            f"Release changelog heading must contain YYYY-MM-DD, found {changelog_state!r}."
        )

    citation_date = _match(
        r"^date-released:\s*(.+)$",
        citation_text,
        label="CITATION.cff date-released",
    )
    if citation_date != changelog_state:
        raise SystemExit(
            "Release-date mismatch: "
            f"CHANGELOG.md={changelog_state!r}, CITATION.cff={citation_date!r}."
        )

    print(f"release tag      : {args.tag}")
    print(f"package version  : {version}")
    print(f"release date     : {citation_date}")
    print("release metadata : consistent")


if __name__ == "__main__":
    main()
