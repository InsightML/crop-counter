"""The AWS launcher's placeholder contract.

``bootstrap.sh`` is the EC2 user-data. ``launch.sh`` fills its ``__NAME__``
placeholders with ``sed`` before handing it to ``run-instances``, and the
heredoc that runs everything as ``ubuntu`` is SINGLE-QUOTED — so a value
reaches the inner shell only if it is both substituted *and* listed in the
``env`` hand-off.

That is three separate places to remember, and forgetting any one of them fails
silently on a $2.50/h box: the RUNS list added for the trunk-size sweep was
read by ``run_all.sh`` but never forwarded, so a launch could only ever train
the default pair. These tests are text-level and cost nothing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

AWS = Path(__file__).resolve().parents[1] / "examples" / "FishDetection" / "scripts" / "aws"
BOOTSTRAP = (AWS / "bootstrap.sh").read_text()
LAUNCH = (AWS / "launch.sh").read_text()

#: Values bootstrap.sh must receive from launch.sh, and pass on to the ubuntu shell.
FORWARDED = ("RUNS", "RUN_BASELINES", "MEASURE_SIZES", "EPOCHS")


def placeholders_in_bootstrap() -> set:
    return set(re.findall(r"__([A-Z0-9_]+)__", BOOTSTRAP))


def test_bootstrap_still_uses_placeholders():
    found = placeholders_in_bootstrap()
    assert {"BRANCH", "S3_URI", "RESUME", "CODE_KEY", "REGION", "S3_REGION"} <= found


@pytest.mark.parametrize("name", sorted(placeholders_in_bootstrap()))
def test_every_placeholder_is_substituted_by_launch(name):
    """An unsubstituted placeholder reaches the box as the literal string."""
    assert f"s|__{name}__|" in LAUNCH, (
        f"bootstrap.sh uses __{name}__ but launch.sh never substitutes it"
    )


@pytest.mark.parametrize("name", FORWARDED)
def test_run_selection_values_cross_the_quoted_heredoc(name):
    """Defining a value above the heredoc is not enough — it needs the env list.

    The heredoc is ``<<'UBUNTU'``, so nothing expands host-side; ``sudo -u
    ubuntu -H env NAME="$NAME"`` is the only way in.
    """
    env_block = BOOTSTRAP.split("sudo -u ubuntu -H env", 1)[1].split("bash -s", 1)[0]
    assert f'{name}="${name}"' in env_block, (
        f"{name} is never handed to the ubuntu shell; it would be empty there"
    )


@pytest.mark.parametrize("name", FORWARDED)
def test_run_selection_values_are_exported_only_when_set(name):
    """An empty RUNS must not override run_all.sh's default with nothing."""
    assert f'if [ -n "${name}" ]; then export {name}; fi' in BOOTSTRAP


def test_launch_refuses_to_ship_an_unsubstituted_placeholder():
    assert "refusing to launch: unsubstituted placeholder" in LAUNCH
    # [A-Z_] alone would miss __S3_URI__ and __S3_REGION__ — the digit matters.
    assert "__[A-Z0-9_]" in LAUNCH


def test_launch_escapes_sed_replacements():
    """RUNS carries filesystem paths; & and | in a replacement corrupt the sed."""
    assert "sed_escape()" in LAUNCH
    for name in FORWARDED:
        assert f's|__{name}__|$(sed_escape "${name}")|g' in LAUNCH


def test_run_all_reads_everything_bootstrap_forwards():
    """The other end of the contract: run_all.sh must honour these names."""
    run_all = (AWS.parent / "run_all.sh").read_text()
    for name in FORWARDED:
        assert f'{name}="${{{name}:-' in run_all, f"run_all.sh ignores {name}"
