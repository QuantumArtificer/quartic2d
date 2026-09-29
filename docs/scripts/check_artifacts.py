"""Check QUARTIC2D documentation generated-artifact consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "artifacts.json"
GENERATED_META = ROOT / "docs" / "source" / "_generated" / "validation" / "snapshot.json"

sys.path.insert(0, str((ROOT / "docs" / "scripts").resolve()))
from generate_evidence import publication_fingerprint  # noqa: E402

SVG_FINGERPRINT_RE = re.compile(r"publication-bundle-sha256:\s*([0-9a-f]{64})")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _expected_paths(manifest: dict) -> set[str]:
    return {
        path
        for group in manifest["groups"].values()
        for path in group["paths"]
    }


def _managed_actual(manifest: dict) -> set[str]:
    actual: set[str] = set()
    for directory in manifest["managed_directories"]:
        root = ROOT / directory
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in {".svg", ".md", ".json", ".txt"}:
                actual.add(path.relative_to(ROOT).as_posix())
    return actual


def _resolve_reference(source: Path, raw: str) -> str | None:
    raw = raw.rstrip("`'\"),.;")
    if raw.startswith("docs/source/"):
        candidate = ROOT / raw
    else:
        candidate = (source.parent / raw).resolve()
    try:
        rel = candidate.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None
    if rel.startswith("docs/source/_static/") or rel.startswith("docs/source/_generated/"):
        return rel
    return None


def _referenced_managed_paths() -> set[str]:
    refs: set[str] = set()
    sources = [ROOT / "README.md"]
    sources.extend(
        path
        for path in (ROOT / "docs" / "source").rglob("*")
        if path.suffix.lower() in {".md", ".rst"}
    )
    pattern = re.compile(
        r"(?P<path>(?:docs/source/|(?:\.\./)*)?(?:"
        r"_static/(?:examples|validation)/[^\s\)\}\]\"']+\.svg|"
        r"_generated/validation/[^\s\)\}\]\"']+\.(?:md|json)|"
        r"_generated/examples/[^\s\)\}\]\"']+\.txt"
        r"))"
    )
    for source in sources:
        text = source.read_text(errors="ignore")
        for match in pattern.finditer(text):
            resolved = _resolve_reference(source, match.group("path"))
            if resolved:
                refs.add(resolved)
    return refs


def check(results_dir: Path | None = None) -> list[str]:
    manifest = _load_manifest()
    expected = _expected_paths(manifest)
    actual = _managed_actual(manifest)
    referenced = _referenced_managed_paths()
    errors: list[str] = []

    missing = sorted(expected - actual)
    if missing:
        errors.append("Missing managed artifacts:\n  " + "\n  ".join(missing))

    orphan = sorted(actual - expected)
    if orphan:
        errors.append("Undeclared/orphan managed artifacts:\n  " + "\n  ".join(orphan))

    panel_label_re = re.compile(r"<!--\s*\([a-z]\)\s*-->")
    for path_text in sorted(expected):
        if not path_text.endswith(".svg"):
            continue
        path = ROOT / path_text
        if not path.is_file():
            continue
        if panel_label_re.search(path.read_text(errors="ignore")):
            errors.append(f"Generated SVG contains a forbidden subfigure label: {path_text}")

    undeclared_refs = sorted(referenced - expected)
    if undeclared_refs:
        errors.append("Referenced generated artifacts absent from docs/artifacts.json:\n  " + "\n  ".join(undeclared_refs))

    allowed_unreferenced = {
        "docs/source/_generated/validation/snapshot.json",
    }
    unreferenced = sorted(expected - referenced - allowed_unreferenced)
    if unreferenced:
        errors.append("Declared artifacts not referenced by documentation sources:\n  " + "\n  ".join(unreferenced))

    tracked_fingerprint = None
    if not GENERATED_META.is_file():
        errors.append(f"Missing generated evidence metadata: {GENERATED_META.relative_to(ROOT)}")
    else:
        meta = json.loads(GENERATED_META.read_text())
        tracked_fingerprint = meta.get("publication_bundle_sha256")
        if not isinstance(tracked_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", tracked_fingerprint):
            errors.append("Generated evidence metadata has no valid publication-bundle fingerprint.")

        output_hashes = meta.get("generated_outputs_sha256", {})
        for name, expected_hash in sorted(output_hashes.items()):
            path = GENERATED_META.parent / name
            if not path.is_file():
                errors.append(f"Generated evidence file recorded in snapshot is missing: {path.relative_to(ROOT)}")
                continue
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                errors.append(
                    f"Generated evidence file differs from tracked snapshot: {path.relative_to(ROOT)} "
                    f"tracked={expected_hash} current={actual_hash}"
                )

        if tracked_fingerprint:
            for path_text in manifest["groups"]["validation_figures"]["paths"]:
                path = ROOT / path_text
                if not path.is_file():
                    continue
                match = SVG_FINGERPRINT_RE.search(path.read_text(errors="ignore"))
                if match is None:
                    errors.append(f"Validation SVG lacks publication-bundle fingerprint: {path_text}")
                elif match.group(1) != tracked_fingerprint:
                    errors.append(
                        f"Validation SVG and generated evidence text come from different publication bundles: "
                        f"{path_text} svg={match.group(1)} text={tracked_fingerprint}"
                    )

    if results_dir is not None:
        fingerprint, input_hashes = publication_fingerprint(results_dir)
        if GENERATED_META.is_file():
            meta = json.loads(GENERATED_META.read_text())
            if meta.get("publication_bundle_sha256") != fingerprint:
                errors.append(
                    "Generated validation text belongs to a different publication bundle: "
                    f"tracked={meta.get('publication_bundle_sha256')} current={fingerprint}"
                )
            if meta.get("input_sha256") != input_hashes:
                errors.append("Generated validation metadata input hashes do not match the supplied publication bundle.")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="optional canonical publication result directory for bundle-freshness checks",
    )
    args = parser.parse_args()
    errors = check(args.results_dir)
    if errors:
        print("\n\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("documentation artifact manifest: PASS")


if __name__ == "__main__":
    main()
