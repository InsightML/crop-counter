"""Unit tests for cropcounter.portal — all HTTP mocked, no live calls, no key."""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from cropcounter.portal import (
    PROD_BASE_URL,
    STAGING_BASE_URL,
    AuthError,
    MarkerDetectionError,
    PortalClient,
    PortalError,
    RateLimitError,
    WheatCountResult,
    _image_to_base64,
)

SAMPLE_IMAGE = sorted(
    (Path(__file__).parent.parent / "examples" / "data" / "val" / "images").glob("*.jpg")
)[0]


# --------------------------------------------------------------------------- #
# Fakes — no real network, no requests-mock dependency
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body if json_body is not None else {}

    def json(self):
        return self._json_body


class FakeSession:
    """Returns queued responses in order; records every call it receives."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self._responses.pop(0)


def make_client(responses, **kwargs) -> PortalClient:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("session", FakeSession(responses))
    kwargs.setdefault("backoff_seconds", 0.0)  # keep retry tests fast
    return PortalClient(**kwargs)


# --------------------------------------------------------------------------- #
# Construction / auth-at-init
# --------------------------------------------------------------------------- #


def test_no_key_raises_autherror(monkeypatch, tmp_path):
    monkeypatch.delenv("INSIGHTML_API_KEY", raising=False)
    monkeypatch.delenv("INSIGHTML_API_ENV", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: no .env to accidentally pick up
    with pytest.raises(AuthError):
        PortalClient(session=FakeSession([]))


def test_key_from_env_var(monkeypatch):
    monkeypatch.setenv("INSIGHTML_API_KEY", "from-env")
    client = PortalClient(session=FakeSession([]))
    assert client.api_key == "from-env"


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("INSIGHTML_API_KEY", "from-env")
    client = PortalClient(api_key="explicit", session=FakeSession([]))
    assert client.api_key == "explicit"


def test_env_prod_and_staging_base_urls():
    prod = make_client([], env="prod")
    staging = make_client([], env="staging")
    assert prod.base_url == PROD_BASE_URL
    assert staging.base_url == STAGING_BASE_URL


def test_invalid_env_raises_valueerror():
    with pytest.raises(ValueError):
        make_client([], env="dev")


def test_insightml_api_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("INSIGHTML_API_ENV", "staging")
    client = make_client([])  # env left at default "prod"
    assert client.base_url == STAGING_BASE_URL
    monkeypatch.delenv("INSIGHTML_API_ENV", raising=False)


def test_explicit_env_wins_over_missing_env_var(monkeypatch):
    monkeypatch.delenv("INSIGHTML_API_ENV", raising=False)
    client = make_client([], env="staging")
    assert client.base_url == STAGING_BASE_URL


# --------------------------------------------------------------------------- #
# Image encoding
# --------------------------------------------------------------------------- #


def test_image_to_base64_from_path():
    encoded = _image_to_base64(SAMPLE_IMAGE)
    assert base64.b64decode(encoded) == SAMPLE_IMAGE.read_bytes()


def test_image_to_base64_from_str_path():
    encoded = _image_to_base64(str(SAMPLE_IMAGE))
    assert base64.b64decode(encoded) == SAMPLE_IMAGE.read_bytes()


def test_image_to_base64_from_bytes():
    raw = SAMPLE_IMAGE.read_bytes()
    assert _image_to_base64(raw) == base64.b64encode(raw).decode("ascii")


def test_image_to_base64_from_pil_image():
    with Image.open(SAMPLE_IMAGE) as im:
        im.load()
        encoded = _image_to_base64(im)
    decoded = base64.b64decode(encoded)
    assert len(decoded) > 0
    # Round-trips to a valid, same-sized image (re-encoding won't byte-match the JPEG).
    with Image.open(__import__("io").BytesIO(decoded)) as roundtrip:
        assert roundtrip.size == im.size


def test_image_to_base64_passthrough_for_non_path_string():
    already_b64 = "not-a-real-file-so-treated-as-base64=="
    assert _image_to_base64(already_b64) == already_b64


def test_image_to_base64_passthrough_for_realistic_length_base64():
    # Regression: a real base64-encoded image (hundreds of KB+) is far
    # longer than the OS's max path/filename length, so a naive
    # `Path(image).is_file()` raises OSError (ENAMETOOLONG) instead of
    # returning False. Must be treated as already-base64, not crash.
    already_b64 = base64.b64encode(b"\x00\x01\x02" * 500_000).decode("ascii")
    assert _image_to_base64(already_b64) == already_b64


def test_image_to_base64_unsupported_type_raises():
    with pytest.raises(TypeError):
        _image_to_base64(12345)


# --------------------------------------------------------------------------- #
# count_wheat — happy path
# --------------------------------------------------------------------------- #


def test_count_wheat_happy_path():
    cropped_bytes = b"fake-png-bytes"
    body = {
        "predicted_count": 42,
        "cropped_image": base64.b64encode(cropped_bytes).decode("ascii"),
        "quadrat_corners": [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
        "inference_time_seconds": 0.87,
        "metadata": {"model_version": "vgg-density-v1"},
    }
    client = make_client([FakeResponse(200, body)])
    result = client.count_wheat(SAMPLE_IMAGE)

    assert isinstance(result, WheatCountResult)
    assert result.predicted_count == 42
    assert result.cropped_image == cropped_bytes
    assert result.quadrat_corners == body["quadrat_corners"]
    assert result.inference_time_seconds == 0.87
    assert result.metadata == {"model_version": "vgg-density-v1"}

    # Request shape: bearer header, base64 image field, correct URL.
    call = client.session.calls[0]
    assert call["url"] == f"{PROD_BASE_URL}/v1/agriculture/wheat-count"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert base64.b64decode(call["json"]["image"]) == SAMPLE_IMAGE.read_bytes()
    assert "id" not in call["json"]
    assert "json_data" not in call["json"]


def test_count_wheat_passes_id_and_json_data():
    body = {"predicted_count": 1}
    client = make_client([FakeResponse(200, body)])
    client.count_wheat(SAMPLE_IMAGE, id="req-123", json_data={"key": "value"})
    call = client.session.calls[0]
    assert call["json"]["id"] == "req-123"
    assert call["json"]["json_data"] == {"key": "value"}


def test_count_wheat_falls_back_to_measured_inference_time():
    body = {"predicted_count": 7}  # no inference_time_seconds in the response
    client = make_client([FakeResponse(200, body)])
    result = client.count_wheat(SAMPLE_IMAGE)
    assert result.inference_time_seconds is not None
    assert result.inference_time_seconds >= 0.0


def test_count_wheat_without_cropped_image_or_corners():
    body = {"predicted_count": 3}
    client = make_client([FakeResponse(200, body)])
    result = client.count_wheat(SAMPLE_IMAGE)
    assert result.cropped_image is None
    assert result.quadrat_corners is None
    assert result.metadata == {}


# --------------------------------------------------------------------------- #
# virtual_quadrat — happy path
# --------------------------------------------------------------------------- #


def test_virtual_quadrat_happy_path():
    corners = [[10.0, 20.0], [110.0, 22.0], [108.0, 122.0], [8.0, 120.0]]
    client = make_client([FakeResponse(200, {"quadrat_corners": corners})])
    result = client.virtual_quadrat(SAMPLE_IMAGE, quadrat_side="S")

    assert result == corners
    call = client.session.calls[0]
    assert call["url"] == f"{PROD_BASE_URL}/v1/tools/virtual-quadrat"
    assert call["json"]["parameters"] == {"quadrat_side": "S"}


def test_virtual_quadrat_default_quadrat_side():
    client = make_client([FakeResponse(200, {"quadrat_corners": []})])
    client.virtual_quadrat(SAMPLE_IMAGE)
    call = client.session.calls[0]
    assert call["json"]["parameters"] == {"quadrat_side": "S"}


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #


def test_401_raises_autherror_with_helpful_message():
    client = make_client([FakeResponse(401, {})])
    with pytest.raises(AuthError, match="stale, or revoked"):
        client.count_wheat(SAMPLE_IMAGE)


@pytest.mark.parametrize("status", [400, 422])
def test_marker_error_code_raises_markerdetectionerror_on_400_or_422(status):
    # Real status code for these two endpoints is unconfirmed pending a live
    # smoke test (sibling /tools/virtual-quadrat/crop endpoint docs show 400;
    # our own demo scripts don't exercise the failure path) — the client
    # must treat both the same.
    body = {"error_code": 202, "message": "Invalid number of markers detected"}
    client = make_client([FakeResponse(status, body)])
    with pytest.raises(MarkerDetectionError, match="4 ArUco markers"):
        client.count_wheat(SAMPLE_IMAGE)


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize("error_code", [201, 203, 204, 205, 206])
def test_marker_error_code_family_201_to_206_all_raise_markerdetectionerror(status, error_code):
    body = {"error_code": error_code, "message": "marker trouble"}
    client = make_client([FakeResponse(status, body)])
    with pytest.raises(MarkerDetectionError):
        client.count_wheat(SAMPLE_IMAGE)


@pytest.mark.parametrize("status", [400, 422])
def test_other_error_code_raises_generic_portalerror_not_marker(status):
    body = {"error_code": 999, "message": "something else went wrong"}
    client = make_client([FakeResponse(status, body)])
    with pytest.raises(PortalError) as excinfo:
        client.virtual_quadrat(SAMPLE_IMAGE)
    assert not isinstance(excinfo.value, MarkerDetectionError)


def test_429_retries_then_succeeds():
    body = {"predicted_count": 5}
    client = make_client([FakeResponse(429), FakeResponse(429), FakeResponse(200, body)])
    result = client.count_wheat(SAMPLE_IMAGE)
    assert result.predicted_count == 5
    assert len(client.session.calls) == 3


def test_429_exhausts_retries_then_raises_ratelimiterror():
    responses = [FakeResponse(429) for _ in range(5)]
    client = make_client(responses, max_retries=2)
    with pytest.raises(RateLimitError, match="25 req/s"):
        client.count_wheat(SAMPLE_IMAGE)
    # 1 initial attempt + 2 retries = 3 calls total before giving up.
    assert len(client.session.calls) == 3


def test_unexpected_status_raises_portalerror():
    client = make_client([FakeResponse(500, {"message": "server error"})])
    with pytest.raises(PortalError, match="500"):
        client.count_wheat(SAMPLE_IMAGE)
