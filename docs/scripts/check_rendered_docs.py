"""Check structural invariants of a rendered QUARTIC2D HTML documentation tree."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[str] = []
        self.images: list[tuple[str, str | None]] = []
        self.text_chunks: list[str] = []
        self.headings: list[str] = []
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_tag = tag
            self._heading_parts = []
        element_id = values.get("id")
        if element_id:
            self.ids.append(element_id)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"] or "")
        if tag == "img" and values.get("src"):
            self.images.append((values["src"] or "", values.get("alt")))

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text_chunks.append(data)
            if self._heading_tag is not None:
                self._heading_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._heading_tag == tag:
            heading = " ".join(part.strip() for part in self._heading_parts if part.strip())
            if heading:
                self.headings.append(heading)
            self._heading_tag = None
            self._heading_parts = []


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


def local_target(page: Path, root: Path, href: str) -> tuple[Path, str] | None:
    split = urlsplit(href)
    if split.scheme or split.netloc or href.startswith(("mailto:", "javascript:")):
        return None

    raw_path = unquote(split.path)
    if raw_path.startswith("/"):
        target = root / raw_path.lstrip("/")
    elif raw_path:
        target = page.parent / raw_path
    else:
        target = page

    if raw_path.endswith("/"):
        target = target / "index.html"
    return target.resolve(), unquote(split.fragment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "html_dir",
        nargs="?",
        default="docs/_build/html",
        type=Path,
        help="rendered Sphinx HTML directory",
    )
    args = parser.parse_args()
    root = args.html_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"rendered documentation directory not found: {root}")

    pages = sorted(root.rglob("*.html"))
    if not pages:
        raise SystemExit(f"no rendered HTML pages found under: {root}")

    parsed = {page.resolve(): parse_page(page) for page in pages}
    failures: list[str] = []
    forbidden_phrases = (
        "external reference for the pedagogical field",
        "direct-q reference",
        "observed refinement order",
        "observed asymptotic order",
    )

    for page, data in parsed.items():
        rel = page.relative_to(root)
        visible_text = " ".join(data.text_chunks).lower()
        for phrase in forbidden_phrases:
            if phrase in visible_text:
                failures.append(f"{rel}: forbidden documentation phrase: {phrase!r}")
        if any(heading.strip().lower() == "terminal output" for heading in data.headings):
            failures.append(f"{rel}: forbidden standalone 'Terminal output' heading")

        duplicates = sorted({value for value in data.ids if data.ids.count(value) > 1})
        for duplicate in duplicates:
            failures.append(f"{rel}: duplicate id #{duplicate}")

        for src, alt in data.images:
            target_info = local_target(page, root, src)
            if target_info is not None and not target_info[0].exists():
                failures.append(f"{rel}: missing image asset {src}")
            if alt is None or not alt.strip():
                failures.append(f"{rel}: image lacks non-empty alt text: {src}")

        for href in data.links:
            target_info = local_target(page, root, href)
            if target_info is None:
                continue
            target, fragment = target_info
            if not target.exists():
                failures.append(f"{rel}: broken internal link {href}")
                continue
            if fragment and target.suffix.lower() == ".html":
                target_data = parsed.get(target)
                if target_data is None:
                    target_data = parse_page(target)
                    parsed[target] = target_data
                if fragment not in target_data.ids:
                    failures.append(f"{rel}: missing anchor in internal link {href}")

    if failures:
        print("rendered documentation QA: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"rendered documentation QA: PASS ({len(pages)} HTML pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
