"""Unit tests for the per-device mixed-precision policy in cropcounter.

No GPU is touched: a ``cuda`` device object is cheap to construct, and the
compute-capability lookup that decides bf16 vs fp16 is monkeypatched, so both
GPU branches are exercised on a CPU-only machine.
"""
from __future__ import annotations

import contextlib

import pytest
import torch

from cropcounter.dinov3_pyramid import amp_dtype, autocast_context

# --------------------------------------------------------------------------- #
# amp_dtype
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", ["cpu", "mps"])
def test_amp_dtype_is_none_off_cuda(spec):
    """MPS and CPU keep fp32 — autocast disabled, exactly as before."""
    assert amp_dtype(torch.device(spec)) is None


def test_amp_dtype_accepts_a_device_string():
    assert amp_dtype("cpu") is None


def test_amp_dtype_turing_gets_float16(monkeypatch):
    """sm_75 (Colab's T4) has no native bf16: cuDNN can refuse it outright."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    assert amp_dtype(torch.device("cuda")) is torch.float16


def test_amp_dtype_ampere_gets_bfloat16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 0))
    assert amp_dtype(torch.device("cuda")) is torch.bfloat16


def test_amp_dtype_hopper_gets_bfloat16(monkeypatch):
    """Anything past Ampere stays on bf16 rather than falling off the check."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert amp_dtype(torch.device("cuda")) is torch.bfloat16


def test_amp_dtype_volta_gets_float16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 0))
    assert amp_dtype(torch.device("cuda")) is torch.float16


# --------------------------------------------------------------------------- #
# autocast_context
# --------------------------------------------------------------------------- #


def test_autocast_context_cpu_is_a_disabled_context_manager():
    ctx = autocast_context(torch.device("cpu"))
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")
    with ctx:
        assert torch.is_autocast_enabled("cpu") is False


def test_autocast_context_cpu_leaves_arithmetic_in_fp32():
    """The observable contract off CUDA: nothing is downcast."""
    x = torch.ones(2, 3)
    with autocast_context(torch.device("cpu")):
        out = torch.nn.Linear(3, 3)(x)
    assert out.dtype is torch.float32


def test_autocast_context_is_a_nullcontext_off_cuda():
    assert isinstance(autocast_context(torch.device("cpu")), contextlib.nullcontext)


@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_autocast_context_cuda_carries_the_chosen_dtype(monkeypatch):
    """Built, not entered — entering would initialise CUDA.

    torch warns when a cuda autocast is constructed on a CPU-only host; that is
    an artefact of testing the GPU branch without a GPU, not of the policy.
    """
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    ctx = autocast_context(torch.device("cuda"))
    assert isinstance(ctx, torch.autocast)
    assert ctx.fast_dtype is torch.float16
    assert ctx.device == "cuda"

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 0))
    assert autocast_context(torch.device("cuda")).fast_dtype is torch.bfloat16


# --------------------------------------------------------------------------- #
# package surface
# --------------------------------------------------------------------------- #


def test_helpers_are_exported_from_the_package():
    import cropcounter

    assert cropcounter.amp_dtype is amp_dtype
    assert cropcounter.autocast_context is autocast_context
    assert {"amp_dtype", "autocast_context"} <= set(cropcounter.__all__)
