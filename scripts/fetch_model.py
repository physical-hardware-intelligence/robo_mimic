#!/usr/bin/env python
"""Fetch the SO-101 MuJoCo model from upstream. Run with `make model`.

NOT vendored: 16.4 MB of STL meshes across 28 files. Fetched instead, and
pinned by hash so a silent upstream change cannot alter the geometry every other
measurement in this repo is built on.

Verified 2026-09-17: phi's vendored copy of `so101_new_calib.xml` is
BYTE-IDENTICAL to upstream (sha256 d75253eb...), so fetching and copying from a
local phi checkout are equivalent. Fetching wins only because it keeps `mirror`
self-contained.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "model"

REPO = "TheRobotStudio/SO-ARM100"
BRANCH = "main"
UPSTREAM = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/Simulation/SO101"
API = f"https://api.github.com/repos/{REPO}/contents/Simulation/SO101"

#: The files `scene.xml` actually needs. The repo also ships old-calibration and
#: camera variants plus URDFs; none are used here, so none are fetched.
TOP_LEVEL = ("scene.xml", "so101_new_calib.xml", "joints_properties.xml")

#: Pinned so the geometry cannot drift under us. This is the file every
#: kinematic constant in the repo was measured from.
PINNED = {
    "so101_new_calib.xml": (
        "d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580"
    )
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return bytes(response.read())


def save(relative: str, data: bytes) -> None:
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    for name in TOP_LEVEL:
        target = OUT / name
        if target.exists():
            print(f"  have  {name}")
        else:
            try:
                data = fetch(f"{UPSTREAM}/{name}")
            except urllib.error.URLError as error:
                print(f"cannot fetch {name}: {error}", file=sys.stderr)
                return 1
            save(name, data)
            print(f"  got   {name}  ({len(data) / 1024:.1f} KB)")

        digest = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        expected = PINNED.get(name)
        if expected and digest != expected:
            print(
                f"\nREFUSING: {name} does not match its pinned hash.\n"
                f"  expected {expected}\n  got      {digest}\n"
                "Upstream changed the model. Every kinematic constant in this repo\n"
                "was measured from the pinned version, so re-verify before updating\n"
                "the pin -- do not just paste the new hash in.",
                file=sys.stderr,
            )
            return 1

    assets = OUT / "assets"
    have = {p.name for p in assets.glob("*")} if assets.exists() else set()
    try:
        import json

        listing = json.loads(fetch(f"{API}/assets").decode())
    except (urllib.error.URLError, ValueError) as error:
        print(f"cannot list assets: {error}", file=sys.stderr)
        return 1
    if isinstance(listing, dict):
        print(f"GitHub API said: {listing.get('message')}", file=sys.stderr)
        return 1

    wanted = [entry["name"] for entry in listing if entry["type"] == "file"]
    missing = [name for name in wanted if name not in have]
    if not missing:
        print(f"  have  assets/ ({len(wanted)} files)")
    else:
        total = 0
        for index, name in enumerate(missing, start=1):
            data = fetch(f"{UPSTREAM}/assets/{name}")
            save(f"assets/{name}", data)
            total += len(data)
            print(f"  got   assets/{name}  [{index}/{len(missing)}]")
        print(f"  {total / 1e6:.1f} MB of meshes")

    print(f"\nmodel ready at {OUT.relative_to(ROOT)}/scene.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
