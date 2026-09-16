"""Atomic, resumable artifacts for the experiment pipeline.

The cache intentionally records file metadata rather than hashes large graph
bundles.  Hashing a 10 GB dataset at every Colab restart would defeat the
purpose of resumability; the manifest still records its size and mtime together
with the configuration and Git revision that created it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUCCESS_FILE = "_SUCCESS.json"


def git_revision(code_root: Path) -> str:
    """Return the checked-out commit, or ``unknown`` outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=code_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def input_metadata(paths: list[Path], *, workspace_root: Path) -> list[dict[str, Any]]:
    """Return cheap, serialisable provenance metadata for input files."""
    metadata: list[dict[str, Any]] = []
    for path in sorted({path.resolve() for path in paths}):
        if not path.exists():
            raise FileNotFoundError(f"Required pipeline input does not exist: {path}")
        stat = path.stat()
        try:
            label = str(path.relative_to(workspace_root))
        except ValueError:
            label = str(path)
        metadata.append(
            {
                "path": label,
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        )
    return metadata


def fingerprint(*, config: Mapping[str, Any], inputs: list[dict[str, Any]], revision: str) -> str:
    """Build a stable stage-cache key from config and inputs.

    Git revision is recorded on the success manifest for provenance but is
    intentionally excluded from the cache key so a Colab ``git pull`` does
    not bust a 10 GB graph cache mid-campaign.
    """
    del revision  # provenance-only; see ArtifactStore.stage manifest
    payload = {"config": config, "inputs": inputs}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


class ArtifactStore:
    """A stage cache with atomic publish semantics.

    Each successful stage lives under::

        <workspace>/artifacts/cache/<dataset>/<stage>/<fingerprint>/

    A stage is reusable only when its success manifest has the requested
    fingerprint and every declared output exists.  Builders write into a
    temporary sibling directory; no incomplete cache directory is ever exposed
    after an interrupted Colab runtime.
    """

    def __init__(self, workspace_root: str | Path, dataset: str, code_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.dataset = dataset.lower()
        self.code_root = Path(code_root).resolve()
        self.cache_root = self.workspace_root / "artifacts" / "cache" / self.dataset
        self.revision = git_revision(self.code_root)

    def cache_dir(self, stage: str, stage_fingerprint: str) -> Path:
        return self.cache_root / stage / stage_fingerprint

    def stage(
        self,
        *,
        stage: str,
        stage_config: Mapping[str, Any],
        inputs: list[Path],
        build: Callable[[Path], Mapping[str, str | Path]],
    ) -> tuple[dict[str, Path], dict[str, Any], bool]:
        """Reuse or build a cache entry.

        Returns ``(outputs, manifest, reused)``.  ``build`` must return a map
        from logical output names to paths relative to the supplied temporary
        directory.
        """
        metadata = input_metadata(inputs, workspace_root=self.workspace_root)
        stage_fingerprint = fingerprint(
            config=stage_config, inputs=metadata, revision=self.revision
        )
        final_dir = self.cache_dir(stage, stage_fingerprint)
        manifest_path = final_dir / SUCCESS_FILE
        manifest = self._read_valid_manifest(manifest_path, stage_fingerprint)
        if manifest is not None:
            return self._outputs_from_manifest(final_dir, manifest), manifest, True

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(prefix=f".{stage}-{stage_fingerprint}-", dir=final_dir.parent)
        )
        temp_dir_to_cleanup: Path | None = temp_dir
        try:
            raw_outputs = build(temp_dir)
            outputs = {name: Path(path) for name, path in raw_outputs.items()}
            resolved_outputs: dict[str, str] = {}
            for name, path in outputs.items():
                absolute = path if path.is_absolute() else temp_dir / path
                if not absolute.exists():
                    raise FileNotFoundError(
                        f"Stage {stage!r} declared missing output {name!r}: {absolute}"
                    )
                resolved_outputs[name] = str(absolute.relative_to(temp_dir))

            manifest = {
                "stage": stage,
                "dataset": self.dataset,
                "fingerprint": stage_fingerprint,
                "git_revision": self.revision,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "inputs": metadata,
                "config": stage_config,
                "outputs": resolved_outputs,
            }
            (temp_dir / SUCCESS_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True))

            # A concurrent process might have completed the same stage first.
            existing = self._read_valid_manifest(manifest_path, stage_fingerprint)
            if existing is None:
                os.replace(temp_dir, final_dir)
                temp_dir_to_cleanup = None  # ownership transferred to the final cache entry
                return self._outputs_from_manifest(final_dir, manifest), manifest, False
            return self._outputs_from_manifest(final_dir, existing), existing, True
        finally:
            if temp_dir_to_cleanup is not None:
                shutil.rmtree(temp_dir_to_cleanup, ignore_errors=True)

    @staticmethod
    def _read_valid_manifest(path: Path, stage_fingerprint: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if manifest.get("fingerprint") != stage_fingerprint:
            return None
        parent = path.parent
        outputs = manifest.get("outputs", {})
        if not outputs or not all((parent / relative_path).exists() for relative_path in outputs.values()):
            return None
        return manifest

    @staticmethod
    def _outputs_from_manifest(cache_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
        return {
            name: cache_dir / relative_path
            for name, relative_path in manifest["outputs"].items()
        }
