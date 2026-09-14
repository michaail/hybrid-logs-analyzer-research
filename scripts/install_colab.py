#!/usr/bin/env python3
"""Install the project dependencies in a Google Colab runtime.

Colab supplies the CUDA-enabled PyTorch build.  Replacing it with a version
from ``requirements-colab.txt`` is a common source of PyG ABI
incompatibilities, so this installer instead obtains the matching extension
wheels from data.pyg.org.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _pip(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", *args]
    )


def _pyg_wheel_url() -> str:
    import torch

    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    cuda_version = torch.version.cuda
    platform = f"cu{cuda_version.replace('.', '')}" if cuda_version else "cpu"
    return f"https://data.pyg.org/whl/torch-{torch_version}+{platform}.html"


def install(project_root: Path, *, include_dev: bool = False) -> None:
    requirements = project_root / "requirements-colab.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements}")

    _pip("--upgrade", "pip")
    _pip("-r", str(requirements))

    wheel_url = _pyg_wheel_url()
    # These are the compiled dependencies used by GINEConv and sparse message
    # passing.  Requiring binary wheels avoids a long, error-prone Colab build.
    _pip("torch-geometric")
    # Optional acceleration wheels are unavailable for some new Python/Torch
    # combinations. GINEConv runs using torch-geometric's PyTorch fallbacks,
    # so do not fail a train-only Colab run when those optional wheels have not
    # been published yet.
    try:
        _pip(
            "--only-binary=:all:",
            "pyg_lib",
            "torch_scatter",
            "torch_sparse",
            "-f",
            wheel_url,
        )
    except subprocess.CalledProcessError:
        print("Optional PyG extension wheels are unavailable; using PyTorch fallbacks.")
    if include_dev:
        _pip("pytest==8.3.4")

    import torch
    import torch_geometric

    print(f"PyTorch {torch.__version__}")
    print(f"PyG {torch_geometric.__version__}")
    print(f"PyG wheel index: {wheel_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository checkout containing requirements-colab.txt.",
    )
    parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Install test-only dependencies as well.",
    )
    args = parser.parse_args()
    install(args.project_root.resolve(), include_dev=args.include_dev)


if __name__ == "__main__":
    main()
