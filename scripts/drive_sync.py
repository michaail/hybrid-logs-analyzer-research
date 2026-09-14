#!/usr/bin/env python3
"""Synchronize ignored experiment data between a workspace and Google Drive.

The script delegates authentication to an already configured rclone remote.
It copies only data/artifacts/results; repository source code and secrets are
never included.  Use ``--dry-run`` first to inspect the exact rclone command.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_FILTERS = {
    "bgl": [
        "data/raw/BGL_full.log",
        "data/processed/bgl/**",
        "artifacts/cache/bgl/**",
        "outputs/bgl/**",
        "artifacts/runs/**",
    ],
    "hdfs": [
        "data/raw/HDFS_full.log",
        "data/raw/anomaly_label.csv",
        "data/processed/hdfs/**",
        "artifacts/cache/hdfs/**",
        "outputs/hdfs/**",
        "artifacts/runs/**",
    ],
}
ALL_FILTERS = [
    "data/raw/**",
    "data/processed/**",
    "artifacts/**",
    "models/**",
    "outputs/**",
    "runs/**",
]


def _default_filters(dataset: str | None, run_id: str | None) -> list[str]:
    filters = list(DATASET_FILTERS[dataset] if dataset else ALL_FILTERS)
    if run_id:
        filters.extend(
            [
                f"artifacts/runs/{run_id}.json",
                f"outputs/**/{run_id}/**",
            ]
        )
    return filters


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the rclone command without executing it (useful for tests/dry runs)."""
    local_root = args.local_root.resolve()
    filters = args.include or _default_filters(args.dataset, args.run_id)
    source, destination = (
        (str(local_root), args.remote)
        if args.command == "push"
        else (args.remote, str(local_root))
    )
    command = ["rclone", "copy", source, destination, "--create-empty-src-dirs"]
    if args.checksum:
        command.append("--checksum")
    for pattern in filters:
        command.extend(["--include", pattern])
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("push", "pull"))
    parser.add_argument(
        "--remote",
        required=True,
        help="Configured rclone destination, e.g. gdrive:hybrid-log-analyzer-artifacts",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace containing data/, artifacts/, and outputs/.",
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_FILTERS))
    parser.add_argument("--run-id", help="Additionally include one run manifest/result folder.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Custom rclone include pattern; repeat to replace default filters.",
    )
    parser.add_argument(
        "--checksum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare checksums instead of timestamps and size (default: true).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned copy only.")
    args = parser.parse_args()

    if not args.local_root.exists() and args.command == "push":
        parser.error(f"Local workspace does not exist: {args.local_root}")
    command = build_command(args)
    print("Planned command:")
    print(" ".join(command))
    if args.dry_run:
        return 0
    if shutil.which("rclone") is None:
        print(
            "rclone is not installed or is not on PATH. Install it, run `rclone config`, "
            "then repeat this command.",
            file=sys.stderr,
        )
        return 2
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
