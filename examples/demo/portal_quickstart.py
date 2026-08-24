#!/usr/bin/env python
"""Call the hosted wheat-count endpoint — two paths, one script.

**Default run (no arguments): the error path, on purpose.** The bundled
`examples/data` images are pre-cropped quadrat *interiors* — the printed
ArUco markers at the corners of the physical quadrat frame were cropped away
long before the dataset was anonymised. The hosted endpoint locates the
quadrat via exactly those 4 markers, so posting a bundled sample fails marker
detection (HTTP 422, `error_code` 202 -> `MarkerDetectionError`). That is the
correct, expected outcome, and this script narrates it: no bundled sample can
produce a count from this endpoint, and none is claimed to.

**Real count path:** point the script at your own *uncropped* quadrat photo —
one that shows all 4 corner markers, unoccluded and in focus:

    python examples/portal_quickstart.py --image /path/to/quadrat.jpg

Needs a real API key either way: copy .env.example to .env and fill in
INSIGHTML_API_KEY (get one at https://portal.insightml.io -> sign up ->
API keys), or export it directly. See docs/portal.md for what this endpoint
currently returns (the previous-generation VGG count-only model).

    python examples/portal_quickstart.py                       # error-path demo
    python examples/portal_quickstart.py --image quadrat.jpg   # real count
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cropcounter.portal import AuthError, MarkerDetectionError, PortalClient, RateLimitError

SAMPLE_DIR = Path(__file__).parent / "data" / "val" / "images"

DEMO_BANNER = """\
No --image given, so this is the ERROR-PATH DEMO.

  The bundled sample below is a pre-cropped quadrat interior: the 4 printed
  ArUco markers at the corners of the physical quadrat frame are not in the
  picture. The hosted endpoint needs all 4 to locate the quadrat, so it will
  reject this image with HTTP 422 / error_code 202 and the client will raise
  MarkerDetectionError. Expected. Nothing in examples/data can be counted by
  the hosted endpoint.

  For a real count, re-run with --image pointing at your own uncropped
  quadrat photo showing all 4 corner markers.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--image", type=Path, default=None,
        help="your own uncropped quadrat photo (all 4 ArUco corner markers visible). "
             "Omit to run the marker-detection error-path demo on a bundled sample.",
    )
    args = parser.parse_args()

    demo = args.image is None
    if demo:
        images = sorted(SAMPLE_DIR.glob("*.jpg"))
        if not images:
            raise SystemExit(f"no sample images found under {SAMPLE_DIR}")
        sample = images[0]
        print(DEMO_BANNER)
    else:
        sample = args.image
        if not sample.is_file():
            raise SystemExit(f"no such image: {sample}")

    try:
        client = PortalClient()  # key from $INSIGHTML_API_KEY, env=prod
    except AuthError as exc:
        raise SystemExit(f"{exc}\ncopy .env.example to .env and add your key") from exc

    print(f"POST {sample.name} -> {client.base_url} (env={client.env}) ...")
    try:
        result = client.count_wheat(sample)
    except MarkerDetectionError as exc:
        if demo:
            print(f"\nMarkerDetectionError (as expected): {exc}")
            print(
                "\nThat is the demo working: the error path is exercised end to end.\n"
                "Re-run with --image <your uncropped quadrat photo> for a real count."
            )
            return
        raise SystemExit(
            f"marker detection failed: {exc}\n"
            "The photo must show all 4 ArUco markers at the corners of the quadrat "
            "frame, unoccluded and in focus — an already-cropped quadrat cannot work."
        ) from exc
    except RateLimitError as exc:
        raise SystemExit(f"rate-limited: {exc}") from exc

    if demo:
        # Only reachable if the endpoint stops requiring markers on this input.
        print("\nUnexpected: the bundled sample was counted rather than rejected.")
    print(f"predicted count: {result.predicted_count}")
    if result.inference_time_seconds is not None:
        print(f"inference time:  {result.inference_time_seconds:.2f}s")
    if result.metadata:
        print(f"metadata: {result.metadata}")


if __name__ == "__main__":
    main()
