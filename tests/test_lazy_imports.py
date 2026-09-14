"""Regression tests for the package's lazy (PEP 562) attribute resolution.

``cropcounter.train`` names both a submodule and the ``train`` function it
exports. Importing anything from the submodule (``TrainConfig``) binds the
*module* on the package, which used to shadow the function whenever it was
imported afterwards -- the exact ``from cropcounter import TrainConfig, train``
line the training notebook uses. Each case runs in a fresh interpreter so
the package's import state is genuinely cold.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

CASES = {
    "train_after_trainconfig": (
        "from cropcounter import TrainConfig, train; assert callable(train), type(train)"
    ),
    "train_alone": "from cropcounter import train; assert callable(train), type(train)",
    "attribute_after_lazy_access": (
        "import cropcounter; cropcounter.TrainConfig; "
        "assert callable(cropcounter.train), type(cropcounter.train)"
    ),
    "module_still_importable": (
        "from cropcounter.train import TrainConfig, train; assert callable(train)"
    ),
}


@pytest.mark.parametrize("snippet", CASES.values(), ids=list(CASES))
def test_train_function_is_not_shadowed_by_its_submodule(snippet):
    result = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
