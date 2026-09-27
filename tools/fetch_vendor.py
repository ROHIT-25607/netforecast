"""Download the dashboard's vendored front-end assets.

The dashboard deliberately loads no CDN: NetForecast is pitched for air-gapped
CII networks, so it has to render with zero outbound network access. Run this
once (with connectivity) to populate server/static/vendor/; after that the whole
application is offline.

    python tools/fetch_vendor.py
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

VENDOR = pathlib.Path(__file__).resolve().parent.parent / "server" / "static" / "vendor"

ASSETS = {
    "plotly.min.js": [
        "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js",
    ],
}
MIN_BYTES = 1_000_000


def main() -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, urls in ASSETS.items():
        dest = VENDOR / name
        if dest.exists() and dest.stat().st_size >= MIN_BYTES:
            print(f"  {name}: already present ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        for url in urls:
            try:
                print(f"  {name}: fetching {url}")
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = r.read()
                if len(data) < MIN_BYTES:
                    print(f"    unexpectedly small ({len(data)} bytes), trying next mirror")
                    continue
                dest.write_bytes(data)
                print(f"    saved {len(data)/1e6:.1f} MB  sha256={hashlib.sha256(data).hexdigest()[:16]}")
                break
            except Exception as exc:
                print(f"    failed: {exc}")
        else:
            failed.append(name)

    if failed:
        print(f"\nCould not fetch: {', '.join(failed)}. The dashboard will not render charts.")
        return 1
    print("\nVendor assets ready — the dashboard now runs fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
