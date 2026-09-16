"""Test bootstrap for the uninstalled ``src`` package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip Intel-ML tests when PyTorch is not installed."""
    if item.get_closest_marker("ml") is not None:
        pytest.importorskip("torch")
