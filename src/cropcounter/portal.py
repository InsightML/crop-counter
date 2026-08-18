"""Client for the hosted InsightML portal API (https://api.insightml.io).

Thin HTTP wrapper: encode an image, POST it, get a typed result back. This
module never imports the rest of ``cropcounter`` (no torch, no DINOv3
backbone) — it works in an environment with only ``pip install
cropcounter[portal]``, no GPU or gated weights required::

    from cropcounter.portal import PortalClient

    client = PortalClient()  # key from $INSIGHTML_API_KEY, env="prod"
    result = client.count_wheat("field.jpg")
    print(result.predicted_count)

See ``docs/portal.md`` for endpoint shapes, rate limits, error semantics, and
the local-vs-hosted decision (the hosted wheat-count endpoint currently
serves the previous-generation VGG density model, not this repo's DINOv3
point model — see ``notebooks/portal_api.ipynb`` for the two side by side).
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is an optional extra
    load_dotenv = None

#: Accepted image inputs: a file path, raw bytes, a PIL image, or an
#: already-base64-encoded string.
ImageInput = Union[str, Path, bytes, Image.Image]

PROD_BASE_URL = "https://api.insightml.io"
#: Pre-release execute-api host for InsightML-internal testing. Unstable and
#: subject to change without notice; the endpoint itself is auth-gated, so
#: production integrations must use env="prod" (see docs/portal.md).
STAGING_BASE_URL = "https://mszbobqb2m.execute-api.eu-west-2.amazonaws.com"

WHEAT_COUNT_PATH = "/v1/agriculture/wheat-count"
VIRTUAL_QUADRAT_PATH = "/v1/tools/virtual-quadrat"

DEFAULT_TIMEOUT_SECONDS = 60.0
#: Automatic retries on HTTP 429 before RateLimitError is raised.
DEFAULT_MAX_RETRIES = 3
#: Base delay for exponential backoff on 429 (doubles each retry: 1s, 2s, 4s, ...).
DEFAULT_BACKOFF_SECONDS = 1.0

#: The ``error_code`` family a marker-detection-failure body carries. Live
#: testing against prod (2026-08-18) confirmed the wheat-count endpoint returns
#: HTTP 422 with ``error_code`` 202 ("fewer than 4 ArUco markers detected").
#: The wider 201-206 family is the sibling /tools/virtual-quadrat/crop
#: endpoint's documented marker-error range, kept here as defensive tolerance;
#: :meth:`PortalClient._post` likewise still accepts HTTP 400 as well as the
#: confirmed 422.
MARKER_ERROR_CODES = frozenset(range(201, 207))
#: Kept as an explicit name for the one code InsightML's docs call out.
MARKER_DETECTION_ERROR_CODE = 202

PORTAL_SIGNUP_URL = "https://portal.insightml.io"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class PortalError(Exception):
    """Base class for errors raised by :class:`PortalClient`."""


class AuthError(PortalError):
    """No usable API key, or the portal rejected it with HTTP 401.

    Covers both "never configured" (checked locally, before any request) and
    "rejected by the server" (missing, stale, or revoked key). Get a fresh
    key at https://portal.insightml.io (sign up -> API keys).
    """


class RateLimitError(PortalError):
    """HTTP 429 after :attr:`PortalClient.max_retries` automatic retries.

    The portal allows bursts up to ~25 requests/second; sustained traffic
    above that gets throttled. See ``examples/portal_batch.py`` for a
    rate-aware calling pattern.
    """


class MarkerDetectionError(PortalError):
    """HTTP 400 or 422 with an ``error_code`` in the marker-detection family
    (201-206; 202 = "Invalid number of markers detected"). Live testing against
    prod (2026-08-18) confirmed 422 + ``error_code`` 202; HTTP 400 and the rest
    of the 201-206 family are kept as defensive tolerance (that range is
    documented for the sibling /crop endpoint), so both status codes are
    treated the same here.

    Both endpoints locate the sample quadrat via 4 ArUco markers printed at
    the corners of the physical quadrat frame; the submitted image must show
    all 4, unoccluded and in focus.
    """


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass
class WheatCountResult:
    """Parsed response from ``POST /v1/agriculture/wheat-count``.

    Attributes:
        predicted_count: the headline count.
        cropped_image: decoded image bytes of the quadrat crop, if the
            response included one; ``None`` otherwise.
        quadrat_corners: 4 ``[x, y]`` pixel pairs locating the quadrat in the
            source image, if the response included them; ``None`` otherwise.
        inference_time_seconds: server-reported inference time if present,
            else the client-measured request round trip.
        metadata: whatever else the response carried (model version, request
            id, ...); ``{}`` if none.
    """

    predicted_count: int
    cropped_image: Optional[bytes]
    quadrat_corners: Optional[List[List[float]]]
    inference_time_seconds: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls, payload: Dict[str, Any], elapsed_seconds: Optional[float] = None
    ) -> "WheatCountResult":
        """Build a result from the parsed JSON body of a 200 response.

        Args:
            payload: the response JSON.
            elapsed_seconds: client-measured request time, used only if the
                response itself carries no timing.
        """
        cropped_b64 = payload.get("cropped_image")
        metadata = dict(payload.get("metadata") or {})
        inference_time = payload.get("inference_time_seconds")
        if inference_time is None:
            inference_time = metadata.get("inference_time_seconds", elapsed_seconds)
        return cls(
            predicted_count=payload["predicted_count"],
            cropped_image=base64.b64decode(cropped_b64) if cropped_b64 else None,
            quadrat_corners=payload.get("quadrat_corners"),
            inference_time_seconds=inference_time,
            metadata=metadata,
        )


# --------------------------------------------------------------------------- #
# Image encoding
# --------------------------------------------------------------------------- #


def _image_to_base64(image: ImageInput) -> str:
    """Coerce any :data:`ImageInput` to a base64-encoded string.

    A ``str``/``Path`` that names an existing file is read and encoded; a
    ``str`` that isn't a path on disk is assumed to be base64 content
    already and passed through unchanged.
    """
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format=image.format or "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode("ascii")
    if isinstance(image, Path):
        return base64.b64encode(image.read_bytes()).decode("ascii")
    if isinstance(image, str):
        try:
            is_file = Path(image).is_file()
        except (OSError, ValueError):
            # A real base64 string is typically far longer than the OS's max
            # path/filename length (e.g. ENAMETOOLONG), or may contain bytes
            # that aren't valid in a path — either way, it's not a file path.
            is_file = False
        if is_file:
            return base64.b64encode(Path(image).read_bytes()).decode("ascii")
        return image  # already base64
    raise TypeError(f"unsupported image input type: {type(image)!r}")


def _safe_json(response: "requests.Response") -> Dict[str, Any]:
    """``response.json()``, or ``{}`` for a non-JSON / empty body."""
    try:
        return response.json()
    except ValueError:
        return {}


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class PortalClient:
    """Thin HTTP client for the hosted InsightML portal API.

    Args:
        api_key: overrides ``$INSIGHTML_API_KEY``. Get one at
            https://portal.insightml.io (sign up -> API keys).
        env: ``"prod"`` (default, https://api.insightml.io) or ``"staging"``
            (the pre-release execute-api host). If left at the default and
            ``$INSIGHTML_API_ENV`` is set, that value wins — set it in
            ``.env`` (see ``.env.example``) rather than passing ``env=``
            everywhere.
        timeout: per-request timeout in seconds.
        max_retries: automatic retries on HTTP 429 before RateLimitError.
        backoff_seconds: base delay for exponential backoff on 429.
        session: an existing ``requests.Session`` to reuse (mainly for tests
            and connection pooling); a new one is created if omitted.

    Raises:
        AuthError: no API key was found (arg, env var, or ``.env``).
        ValueError: ``env`` isn't ``"prod"`` or ``"staging"``.
    """

    BASE_URLS = {"prod": PROD_BASE_URL, "staging": STAGING_BASE_URL}

    def __init__(
        self,
        api_key: Optional[str] = None,
        env: str = "prod",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()  # no-op if there's no .env; never overrides a set env var

        if env == "prod" and os.environ.get("INSIGHTML_API_ENV"):
            env = os.environ["INSIGHTML_API_ENV"]
        if env not in self.BASE_URLS:
            raise ValueError(f"env must be one of {sorted(self.BASE_URLS)}, got {env!r}")

        self.api_key = api_key or os.environ.get("INSIGHTML_API_KEY")
        if not self.api_key:
            raise AuthError(
                "no API key: pass api_key=, set $INSIGHTML_API_KEY, or add it to .env "
                f"(get one at {PORTAL_SIGNUP_URL} -> sign up -> API keys)"
            )

        self.env = env
        self.base_url = self.BASE_URLS[env]
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``json_body`` to ``path``, retrying 429s, raising named
        errors for 401 / marker-detection (400 or 422), and returning the
        parsed JSON body of any other 2xx response."""
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            response = self.session.post(
                url, headers=self._headers(), json=json_body, timeout=self.timeout
            )
            status = response.status_code

            if status == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise RateLimitError(
                        f"rate-limited after {self.max_retries} retries (POST {path}); "
                        "the portal allows bursts up to ~25 req/s — see examples/portal_batch.py"
                    )
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                continue

            if status == 401:
                raise AuthError(
                    "401 Unauthorized: API key is missing, stale, or revoked — "
                    f"get a fresh one at {PORTAL_SIGNUP_URL} (sign up -> API keys)"
                )

            if status in (400, 422):
                # 422 + error_code 202 is what prod returned in live testing
                # (2026-08-18); 400 and the rest of the 201-206 family are
                # defensive tolerance for the sibling /crop endpoint. Both
                # status codes are treated as the same marker-detection
                # failure, keyed off error_code rather than status alone.
                body = _safe_json(response)
                if body.get("error_code") in MARKER_ERROR_CODES:
                    raise MarkerDetectionError(
                        f"{status} (error_code {body.get('error_code')}): Invalid number of "
                        "markers detected — the image must show all 4 ArUco markers at the "
                        "corners of the physical quadrat frame, unoccluded and in focus"
                    )
                raise PortalError(f"{status} (POST {path}): {body}")

            if 200 <= status < 300:
                return response.json()

            raise PortalError(f"unexpected HTTP {status} (POST {path}): {_safe_json(response)}")

    def count_wheat(
        self,
        image: ImageInput,
        id: Optional[str] = None,  # matches the API's own field name
        json_data: Optional[Dict[str, Any]] = None,
    ) -> WheatCountResult:
        """``POST /v1/agriculture/wheat-count``.

        Args:
            image: path / bytes / ``PIL.Image`` / base64 string.
            id: optional caller-supplied request id, echoed back by the API.
            json_data: optional free-form metadata attached to the request.

        Returns:
            A :class:`WheatCountResult`.

        Raises:
            AuthError: HTTP 401.
            RateLimitError: HTTP 429, retries exhausted.
            MarkerDetectionError: HTTP 400 or 422, error_code in 201-206 (202 = marker count).
            PortalError: any other non-2xx response.
        """
        payload: Dict[str, Any] = {"image": _image_to_base64(image)}
        if id is not None:
            payload["id"] = id
        if json_data is not None:
            payload["json_data"] = json_data

        started = time.time()
        body = self._post(WHEAT_COUNT_PATH, payload)
        elapsed = time.time() - started
        return WheatCountResult.from_response(body, elapsed_seconds=elapsed)

    def virtual_quadrat(
        self, image: ImageInput, quadrat_side: str = "S"
    ) -> List[List[float]]:
        """``POST /v1/tools/virtual-quadrat``.

        Args:
            image: path / bytes / ``PIL.Image`` / base64 string.
            quadrat_side: physical quadrat size code (e.g. ``"S"``).

        Returns:
            ``quadrat_corners`` — 4 ``[x, y]`` pixel pairs.

        Raises:
            AuthError: HTTP 401.
            RateLimitError: HTTP 429, retries exhausted.
            MarkerDetectionError: HTTP 400 or 422, error_code in 201-206 (202 = marker count).
            PortalError: any other non-2xx response.
        """
        payload = {
            "image": _image_to_base64(image),
            "parameters": {"quadrat_side": quadrat_side},
        }
        body = self._post(VIRTUAL_QUADRAT_PATH, payload)
        return body["quadrat_corners"]
