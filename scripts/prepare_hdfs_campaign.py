#!/usr/bin/env python3
"""Prepare a local HDFS Family A ablation campaign (parse → gzipped graphs).

Run from this repository root. Training stays on Colab via
``run_ablation.py --mode train-only``.

Example::

    python scripts/prepare_hdfs_campaign.py \\
        --campaign-id hdfs_ablation_20260916 \\
        --workspace-root /path/to/workspace \\
        --set experiment.dataset=hdfs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ablation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(os.environ.get("PIPELINE_WORKSPACE_ROOT", ".")),
    )
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "ablation_base.yaml",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "configs" / "ablation_representation.yaml",
    )
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.code_root / args.config
    config = run_ablation.load_config(config_path)
    config = run_ablation.apply_overrides(
        config, ["experiment.dataset=hdfs", *args.overrides]
    )
    config["__pipeline__"]["workspace_root"] = str(args.workspace_root.resolve())
    config["__pipeline__"]["code_root"] = str(args.code_root.resolve())
    prepared = run_ablation.prepare_representation_campaign(
        config,
        campaign_id=args.campaign_id,
        workspace_root=args.workspace_root,
        code_root=args.code_root,
        campaign_dir=args.campaign_dir,
        matrix_path=args.matrix,
        checkpoint_root=args.checkpoint_root,
    )
    print(json.dumps(prepared, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
