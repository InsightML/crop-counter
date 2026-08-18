# Portal API client

`cropcounter.portal` is a thin client for the hosted InsightML portal API —
a managed service, not a wrapper around this package's local model. No GPU
or DINOv3 checkpoint needed to use it.

<!-- TODO: update when the DINOv3 build ships on the portal -->

> **The hosted wheat-count endpoint currently serves the previous-generation
> model** — a VGG density-map counter, not this repo's DINOv3 point model —
> and it returns **a count only, no per-plant coordinates**. This repo's
> DINOv3 pyramid-decoder point model (the one `notebooks/training.ipynb`
> trains and `notebooks/inference.ipynb` runs) is being rolled out to the
> portal; until then, `count_wheat()` gives you a count from a different,
> older model than the one you'd train locally with this package. See
> `notebooks/portal_api.ipynb` for the two side by side on the same image.

## Install

```bash
pip install "cropcounter[portal]"
```

Pulls in `requests` and `python-dotenv`; the core package has no HTTP
dependency, so installing it plain never drags in an HTTP client.

## Auth

Get a key at <https://portal.insightml.io> (sign up -> API keys). The
client resolves it, in order: the `api_key` argument, then
`$INSIGHTML_API_KEY` (loaded from a `.env` in the working directory or an
ancestor, if `python-dotenv` is installed and a file is present — see
`.env.example`).

```python
from cropcounter.portal import PortalClient

client = PortalClient()                          # env=prod, key from $INSIGHTML_API_KEY
client = PortalClient(api_key="...", env="staging")
```

`env` defaults to `"prod"`; if left at that default and `$INSIGHTML_API_ENV`
is set (see `.env.example`), the env var wins — set it once in `.env`
instead of passing `env=` everywhere. Every request sends
`Authorization: Bearer <key>`.

## Endpoints

| | Method | Path | `env="prod"` host |
|---|---|---|---|
| Wheat count | POST | `/v1/agriculture/wheat-count` | `https://api.insightml.io` |
| Virtual quadrat | POST | `/v1/tools/virtual-quadrat` | `https://api.insightml.io` |

`env="staging"` points at the pre-release execute-api host baked into the
client for InsightML-internal testing (`STAGING_BASE_URL` in
`cropcounter/portal.py`). It is unstable and subject to change without
notice, and the endpoint itself is auth-gated — production integrations
must use `env="prod"`.

Both endpoints locate the physical sample quadrat in the photo via **4
printed ArUco markers**, one at each corner of the quadrat frame. A photo
that doesn't show all 4 clearly fails marker detection (see Errors below).

### `count_wheat(image, id=None, json_data=None) -> WheatCountResult`

Request body: `{"image": "<base64>", "id": ..., "json_data": {...}}` (the
last two are optional pass-throughs, echoed back by the API). The response
is parsed into:

```python
@dataclass
class WheatCountResult:
    predicted_count: int
    cropped_image: bytes | None          # decoded quadrat crop, if the response included one
    quadrat_corners: list | None         # 4 [x, y] pixel pairs, if included
    inference_time_seconds: float | None # server-reported, else client-measured
    metadata: dict                       # whatever else the response carried
```

### `virtual_quadrat(image, quadrat_side="S") -> list`

Request body: `{"image": "<base64>", "parameters": {"quadrat_side": "S"}}`.
Returns `quadrat_corners` directly — 4 `[x, y]` pixel pairs.

### Image input

Both methods accept a file path (`str` / `Path`), raw `bytes`, a
`PIL.Image.Image`, or an already-base64-encoded string.

## Errors

| Exception | HTTP | Cause |
|---|---|---|
| `AuthError` | — / 401 | No key configured locally, or the portal rejected it (missing, stale, or revoked). Get a fresh one at the portal. |
| `RateLimitError` | 429 | Burst limit (~25 req/s) exceeded; the client retries with exponential backoff first (`max_retries`, default 3), then raises. |
| `MarkerDetectionError` | 422 (confirmed) or 400, `error_code` in 201-206 (202 = "Invalid number of markers detected") | The image doesn't show all 4 ArUco quadrat-corner markers. Live testing against prod (2026-08-18) confirmed **HTTP 422 with `error_code` 202**; HTTP 400 and the rest of the 201-206 family — the documented range for the sibling `/crop` endpoint — are kept as defensive tolerance, so the client treats both status codes and the whole range the same. |
| `PortalError` | any other non-2xx | Base class for all of the above; catch it alone to handle any portal error generically. |

## Example scripts

`examples/portal_quickstart.py` (one image) and `examples/portal_batch.py`
(a paced batch) each run two paths:

- **Default, no arguments — the marker-detection error path, on purpose.**
  The bundled `examples/data` images are pre-cropped quadrat *interiors*: the
  4 printed ArUco markers at the corners of the physical quadrat frame are
  not in the picture, so the hosted endpoint rejects them (HTTP 422,
  `error_code` 202 -> `MarkerDetectionError`) and the scripts narrate that as
  the expected outcome. No bundled sample can return a count from this
  endpoint, and neither script claims one; what the default run demonstrates
  is auth, encoding, request pacing and the error path end to end.
- **Your own imagery — the real count path.** `--image PATH` (quickstart) or
  `--images-dir PATH` (batch) points the same code at uncropped quadrat
  photos that do show all 4 corner markers.

## Rate limits

The portal accepts bursts up to **~25 requests/second** before throttling
kicks in. `examples/portal_batch.py` sends at a capped rate (default 5/s,
configurable) and reports each image's outcome (see
[Example scripts](#example-scripts) for what the bundled samples do and don't
return); `PortalClient`'s built-in retry means occasional 429s at the edge of
the burst window are absorbed automatically rather than surfacing as errors —
`RateLimitError` only comes back once `max_retries` is exhausted.

## Local model vs. hosted API — which to use

- **This package, local**: the current-generation DINOv3 point model —
  per-plant coordinates, no rate limit, no network call — but needs the
  (gated) DINOv3 backbone weights and a trained decoder checkpoint on disk.
  `cropcounter` is **AGPL-3.0-only**: if you build a networked service on
  top of the local model, that service's source must be made available to
  its users under AGPL too.
- **Hosted portal API**: no local weights, checkpoint, or GPU needed — but,
  as above, currently the older VGG count-only model, and it's InsightML's
  hosted service: the usual SaaS trade-off of convenience for a per-call
  cost, without control over the model version.

Prototyping, or don't want to deal with local GPU/MPS setup and the gated
DINOv3 download: use the hosted API. Need per-plant coordinates, the
current-generation model, offline operation, zero marginal cost, or don't
want to send imagery to a third party: run this package locally.

## Testing

`tests/test_portal.py` mocks every HTTP response — no live calls, no key
required to run it.

```bash
pytest tests/test_portal.py -q
```
