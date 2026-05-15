#!/usr/bin/env python3
"""
Download and vendor the stim C++ amalgamation header (stim.h) and
the companion source tree used for building the _fast extension.

Usage:
    python scripts/fetch_stim_header.py [--version 1.15.0] [--output vendor/stim.h]

The script downloads the stim sdist from PyPI, extracts stim.h, and writes it
to vendor/stim.h.  It also prints a SHA-256 digest so you can verify integrity
against the known-good value.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_PYPI_SDIST_URL = (
    "https://files.pythonhosted.org/packages/source/s/stim/stim-{version}.tar.gz"
)

# Known-good SHA-256 for the stim.h file extracted from the 1.15.0 sdist.
_KNOWN_SHA256: dict[str, str] = {
    # Will be filled in after the first run; leave empty to skip check.
}


def fetch_stim_header(
    version: str,
    output: Path,
    check: bool = True,
) -> None:
    url = _PYPI_SDIST_URL.format(version=version)
    print(f"Downloading {url} …", file=sys.stderr)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            tmp.write(resp.read())
        tmp_path = Path(tmp.name)

    print("Extracting stim.h …", file=sys.stderr)
    member_name = f"stim-{version}/src/stim.h"
    with tarfile.open(tmp_path, "r:gz") as tf:
        member = tf.getmember(member_name)
        fobj = tf.extractfile(member)
        if fobj is None:
            raise RuntimeError(f"Could not extract {member_name}")
        content = fobj.read()

    tmp_path.unlink(missing_ok=True)

    digest = hashlib.sha256(content).hexdigest()
    print(f"SHA-256: {digest}", file=sys.stderr)

    if check and version in _KNOWN_SHA256:
        expected = _KNOWN_SHA256[version]
        if digest != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for stim.h v{version}:\n"
                f"  expected: {expected}\n"
                f"  got:      {digest}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    print(f"Written {len(content)} bytes → {output}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and vendor stim.h from the stim PyPI sdist."
    )
    parser.add_argument(
        "--version",
        default="1.15.0",
        help="stim version to fetch (default: 1.15.0)",
    )
    parser.add_argument(
        "--output",
        default="vendor/stim.h",
        type=Path,
        help="destination path (default: vendor/stim.h)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip SHA-256 integrity check",
    )
    args = parser.parse_args()

    fetch_stim_header(
        version=args.version,
        output=args.output,
        check=not args.no_check,
    )


if __name__ == "__main__":
    main()
