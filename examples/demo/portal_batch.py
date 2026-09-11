#!/usr/bin/env python
"""Batch-submit images to the hosted wheat-count endpoint at a capped rate.

Mirrors the team's own rate-limit smoke test: the portal accepts bursts up
to ~25 requests/second before 429s kick in. PortalClient already retries a
429 with exponential backoff (see docs/portal.md); this script demonstrates
staying well under the burst limit in the first place, and reports each
image's outcome (including any rate-limit or marker-detection failure).

**Default run: the pacing and error reporting, not counts.** With no
--images-dir the script cycles the bundled `examples/data` samples, which are
pre-cropped quadrat interiors with no ArUco corner markers in frame. The
hosted endpoint locates the quadrat via those 4 markers, so *every* bundled
request comes back HTTP 422 / error_code 202 -> `MarkerDetectionError`, and
the run prints a column of expected failures. What it demonstrates is the
request pacing and per-image error reporting; no bundled sample can produce
a count from this endpoint.

**Real count path:** point it at a folder of your own *uncropped* quadrat
photos, each showing all 4 corner markers:

    python examples/portal_batch.py --images-dir /path/to/quadrat/photos

Needs a real API key — see examples/portal_quickstart.py / .env.example.

    python examples/portal_batch.py                       # error-path demo, 5 req/s
    python examples/portal_batch.py --rate 20 --images 40 # pacing demo, harder
    python examples/portal_batch.py --images-dir photos/  # real counts
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from cropcounter.portal import AuthError, MarkerDetectionError, PortalClient, RateLimitError

SAMPLE_DIR = Path(__file__).parent / "data" / "val" / "images"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

DEMO_BANNER = """\
No --images-dir given, so this is the ERROR-PATH DEMO.

  The bundled samples are pre-cropped quadrat interiors — the 4 printed ArUco
  markers at the corners of the physical quadrat frame are not in the picture,
  so the hosted endpoint rejects every one of them (HTTP 422 / error_code 202,
  surfaced as MarkerDetectionError). Expect a full column of FAILED lines:
  that is the point here — the pacing and the per-image error reporting are
  what this run demonstrates.

  For real counts, re-run with --images-dir pointing at a folder of your own
  uncropped quadrat photos showing all 4 corner markers.
"""


def find_images(directory: Path) -> List[Path]:
    """Every image file directly under ``directory``, sorted by name."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def call_one(client: PortalClient, image: Path, idx: int) -> Dict[str, Any]:
    try:
        result = client.count_wheat(image)
        return {"idx": idx, "image": image.name, "ok": True, "count": result.predicted_count}
    except MarkerDetectionError as exc:
        return {"idx": idx, "image": image.name, "ok": False, "error": f"marker detection: {exc}"}
    except RateLimitError as exc:
        return {"idx": idx, "image": image.name, "ok": False, "error": f"rate-limited: {exc}"}


def fire_requests(
    client: PortalClient, images: List[Path], rate_per_second: int
) -> List[Dict[str, Any]]:
    """Submit ``images`` in ticks of at most ``rate_per_second`` per second."""
    results = []
    with ThreadPoolExecutor(max_workers=max(len(images), 1)) as executor:
        futures = []
        submitted = 0
        while submitted < len(images):
            tick_start = time.time()
            tick_count = min(rate_per_second, len(images) - submitted)
            for i in range(tick_count):
                idx = submitted + i
                futures.append(executor.submit(call_one, client, images[idx], idx))
            submitted += tick_count
            elapsed = time.time() - tick_start
            if elapsed < 1.0 and submitted < len(images):
                time.sleep(1.0 - elapsed)
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rate", type=int, default=5, help="requests/second (portal burst cap ~25)")
    parser.add_argument(
        "--images-dir", type=Path, default=None,
        help="folder of your own uncropped quadrat photos (all 4 ArUco corner markers "
             "visible). Omit to run the marker-detection error-path demo on the bundled "
             "samples.",
    )
    parser.add_argument(
        "--images", type=int, default=None,
        help="how many requests to send (default: each image in the folder once; "
             "a larger count cycles through them)",
    )
    args = parser.parse_args()

    demo = args.images_dir is None
    source = SAMPLE_DIR if demo else args.images_dir
    if not source.is_dir():
        raise SystemExit(f"no such directory: {source}")
    pool = find_images(source)
    if not pool:
        raise SystemExit(f"no images found under {source}")
    if demo:
        print(DEMO_BANNER)

    n = args.images or len(pool)
    images = [pool[i % len(pool)] for i in range(n)]

    try:
        client = PortalClient()  # key from $INSIGHTML_API_KEY, env=prod
    except AuthError as exc:
        raise SystemExit(f"{exc}\ncopy .env.example to .env and add your key") from exc

    print(f"sending {len(images)} request(s) at {args.rate}/s to {client.base_url} ...")
    started = time.time()
    results = fire_requests(client, images, args.rate)
    elapsed = time.time() - started

    for r in sorted(results, key=lambda r: r["idx"]):
        if r["ok"]:
            print(f"{r['idx'] + 1:3d}  {r['image']:20s} count={r['count']}")
        else:
            print(f"{r['idx'] + 1:3d}  {r['image']:20s} FAILED: {r['error']}")

    ok = sum(1 for r in results if r["ok"])
    print(f"\n{elapsed:.2f}s | {ok} ok | {len(results) - ok} failed")
    if demo:
        print(
            "All failures are the expected marker-detection rejection of the pre-cropped "
            "bundled samples.\nRe-run with --images-dir <your uncropped quadrat photos> "
            "for real counts."
        )


if __name__ == "__main__":
    main()
