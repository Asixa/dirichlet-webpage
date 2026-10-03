"""Stamp local asset URLs in index.html with a content hash (?v=xxxxxxxx).

GitHub Pages serves every file with Cache-Control: max-age=600 and caches
HTML and JS independently, so after a deploy a browser can pair new HTML
with a stale script for up to 10 minutes. A per-file content hash makes
each changed file a new URL while unchanged files keep their cache.

Run from the repository root (the pre-commit hook does this):
    python tools/stamp_assets.py
"""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"
# Only our own code and styles; vendored fonts/KaTeX never change in place.
# data-worker: race2d.js creates its Web Worker from this attribute, since
# the worker has no tag of its own.
ASSET = re.compile(r'((?:src|href|data-worker)=")(static/(?:js|css)/[\w.-]+\.(?:js|css))(?:\?v=[0-9a-f]+)?(")')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def main():
    html = PAGE.read_text(encoding="utf-8")

    def stamp(m):
        f = ROOT / m.group(2)
        if not f.exists():
            sys.exit(f"stamp_assets: {m.group(2)} is referenced in index.html but missing")
        return f"{m.group(1)}{m.group(2)}?v={digest(f)}{m.group(3)}"

    new = ASSET.sub(stamp, html)
    if new != html:
        PAGE.write_text(new, encoding="utf-8", newline="")
        print("stamp_assets: index.html updated")


if __name__ == "__main__":
    main()
