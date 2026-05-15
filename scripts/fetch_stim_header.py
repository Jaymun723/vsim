#!/usr/bin/env python3
"""Download stim sources from PyPI into vendor/ for local/native builds."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_PYPI_SDIST_URL = "https://files.pythonhosted.org/packages/source/s/stim/stim-{version}.tar.gz"

# Known-good SHA-256 for the full stim-{version}.tar.gz sdist archive.
_KNOWN_SHA256: dict[str, str] = {
    "1.15.0": "cb0d01b76a596f97f2f46d6d8831274f95ed47f7688a14a3aafde25f5cf68f88",
}


def fetch_stim_sources(
    version: str,
    output_dir: Path,
    check: bool = True,
) -> None:
    url = _PYPI_SDIST_URL.format(version=version)
    print(f"Downloading {url}...", file=sys.stderr)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            tmp.write(resp.read())
        tmp_path = Path(tmp.name)

    archive_bytes = tmp_path.read_bytes()
    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    print(f"Archive SHA-256: {archive_digest}", file=sys.stderr)

    if check and version in _KNOWN_SHA256:
        expected = _KNOWN_SHA256[version]
        if archive_digest != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for stim sdist v{version}:\n"
                f"  expected: {expected}\n"
                f"  got:      {archive_digest}"
            )

    print("Extracting stim.h and src/stim/...", file=sys.stderr)
    extracted_root = Path(tempfile.mkdtemp(prefix="stim-src-"))
    try:
        with tarfile.open(tmp_path, "r:gz") as tf:
            tf.extractall(extracted_root)
        source_root = extracted_root / f"stim-{version}" / "src"
        if not source_root.exists():
            raise RuntimeError(f"Expected source root missing: {source_root}")

        stim_h_src = source_root / "stim.h"
        stim_dir_src = source_root / "stim"
        if not stim_h_src.exists() or not stim_dir_src.is_dir():
            raise RuntimeError("stim sdist does not contain expected src/stim.h and src/stim/")

        output_dir.mkdir(parents=True, exist_ok=True)

        stim_h_dst = output_dir / "stim.h"
        stim_dir_dst = output_dir / "stim"
        if stim_dir_dst.exists():
            shutil.rmtree(stim_dir_dst)

        shutil.copy2(stim_h_src, stim_h_dst)
        shutil.copytree(stim_dir_src, stim_dir_dst)
        print(f"Wrote {stim_h_dst}", file=sys.stderr)
        print(f"Wrote {stim_dir_dst}", file=sys.stderr)
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(extracted_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and vendor stim sources from the stim PyPI sdist."
    )
    parser.add_argument(
        "--version",
        default="1.15.0",
        help="stim version to fetch (default: 1.15.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="vendor",
        type=Path,
        help="destination directory for stim.h and stim/ (default: vendor)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip sdist SHA-256 integrity check",
    )
    args = parser.parse_args()

    fetch_stim_sources(
        version=args.version,
        output_dir=args.output_dir,
        check=not args.no_check,
    )


if __name__ == "__main__":
    main()
