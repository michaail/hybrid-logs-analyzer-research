"""Shared utilities for the hybrid log analyzer."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def is_ci() -> bool:
    """Return True when running inside a CI environment."""
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # pyright: ignore[reportMissingImports]  # noqa: F401

        return True
    except ImportError:
        return False


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
