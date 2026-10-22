"""Shared test fixtures.

MPS tests use the ``mps_device`` fixture. Without a usable MPS device they are
skipped with an explicit reason (see tests/hardware.py). Set MINILM_REQUIRE_MPS=1
on Apple Silicon to turn such skips into failures, so the complete MPS suite must run:

    MINILM_REQUIRE_MPS=1 python -m pytest -q -rs
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hardware import MPS_REASON, MPS_USABLE

REQUIRE_MPS = os.environ.get("MINILM_REQUIRE_MPS") == "1"


@pytest.fixture
def mps_device() -> str:
    if not MPS_USABLE:
        if REQUIRE_MPS:
            pytest.fail(f"MINILM_REQUIRE_MPS=1 but {MPS_REASON}")
        pytest.skip(f"hardware: {MPS_REASON}")
    return "mps"
