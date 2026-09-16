"""Immutable HDFS preprocessing-bundle contract. This module never imports PyTorch."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from src.modules.model_package import (
    MANIFEST_NAME,
    PACKAGE_FORMAT_V2,
    PackageArchitecture,
    PackageModel,
    PackageValidationIssue,
    PackageValidationResult,
    PreprocessingBundleRef,
    try_load_model_package_manifest,
    unpack_zip_bytes,
    validate_model_package,
)

BUNDLE_FORMAT = "hdfs-preprocessing-bundle-v1"
DRAIN_CONFIG_NAME = "drain.ini"
DRAIN_PARSER_NAME = "drain_parser.bin"
EMBEDDINGS_NAME = "embeddings.npz"
NODE_FEATURE_EXTRA_DIM = 9
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9_.-]+$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class BundleFiles(PackageModel):
    """Declared preprocessing-bundle members and their SHA-256 digests."""

    drain_config: str
    drain_parser: str
    embeddings: str
    checksums: dict[str, str]

    @field_validator("drain_config", "drain_parser", "embeddings")
    @classmethod
    def _relative_posix_path(cls, value: str) -> str:
        return _require_relative_posix(value)

    @field_validator("checksums")
    @classmethod
    def _lowercase_checksums(cls, value: dict[str, str]) -> dict[str, str]:
        for path, digest in value.items():
            _require_relative_posix(path)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"checksum for {path} must be a lowercase hex SHA-256")
        return value

    @model_validator(mode="after")
    def _checksums_match_declared_files(self) -> BundleFiles:
        expected = {self.drain_config, self.drain_parser, self.embeddings}
        if set(self.checksums) != expected:
            raise ValueError("checksums must contain exactly the declared bundle files")
        return self


class BundleManifest(PackageModel):
    """Closed inspectable contract for a frozen HDFS preprocessing bundle."""

    identifier: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    version: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN)
    source_compatibility: Literal["hdfs"]
    format: Literal["hdfs-preprocessing-bundle-v1"]
    digest: str = Field(pattern=_SHA256_PATTERN)
    files: BundleFiles


def bundle_digest(checksums: Mapping[str, str]) -> str:
    """Return the stable digest over declared member checksums."""

    material = "".join(f"{path}:{digest}\n" for path, digest in sorted(checksums.items()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def validate_preprocessing_bundle(
    root: Path,
    *,
    architecture: PackageArchitecture | None = None,
    expected: PreprocessingBundleRef | None = None,
) -> PackageValidationResult:
    """Collect every directory-bundle eligibility failure without loading a model."""

    issues: list[PackageValidationIssue] = []
    bundle_root = root.expanduser()
    if not bundle_root.is_dir():
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path=str(root),
                    reason="Bundle root must be an existing directory.",
                )
            ]
        )

    manifest, manifest_issues = _load_bundle_manifest(bundle_root)
    issues.extend(manifest_issues)
    if manifest is None:
        return PackageValidationResult.from_issues(issues)

    declared = (
        manifest.files.drain_config,
        manifest.files.drain_parser,
        manifest.files.embeddings,
    )
    resolved: dict[str, Path] = {}
    for relative in declared:
        located, path_issues = _resolve_declared_file(bundle_root, relative)
        issues.extend(path_issues)
        if located is not None:
            resolved[relative] = located

    for relative, digest in manifest.files.checksums.items():
        located = resolved.get(relative)
        if located is None:
            continue
        actual = hashlib.sha256(located.read_bytes()).hexdigest()
        if actual != digest:
            issues.append(
                PackageValidationIssue(
                    path=relative,
                    reason="SHA-256 does not match files.checksums.",
                )
            )

    computed_digest = bundle_digest(manifest.files.checksums)
    if computed_digest != manifest.digest:
        issues.append(
            PackageValidationIssue(
                path="digest",
                reason="Bundle digest does not match the declared member checksums.",
            )
        )

    issues.extend(_undeclared_file_issues(bundle_root, {MANIFEST_NAME, *declared}))

    embeddings_path = resolved.get(manifest.files.embeddings)
    if embeddings_path is not None:
        issues.extend(
            _embedding_issues(embeddings_path, manifest.files.embeddings, architecture)
        )

    if expected is not None:
        if expected.identifier != manifest.identifier:
            issues.append(
                PackageValidationIssue(
                    path="identifier",
                    reason="Bundle identifier does not match the model package descriptor.",
                )
            )
        if expected.version != manifest.version:
            issues.append(
                PackageValidationIssue(
                    path="version",
                    reason="Bundle version does not match the model package descriptor.",
                )
            )
        if expected.digest != manifest.digest:
            issues.append(
                PackageValidationIssue(
                    path="digest",
                    reason="Bundle digest does not match the model package descriptor.",
                )
            )

    return PackageValidationResult.from_issues(issues)


def validate_preprocessing_bundle_source(
    path: Path,
    *,
    architecture: PackageArchitecture | None = None,
    expected: PreprocessingBundleRef | None = None,
) -> PackageValidationResult:
    """Validate a directory bundle or a zip that unpacks to one."""

    source = path.expanduser()
    if source.is_dir():
        return validate_preprocessing_bundle(
            source, architecture=architecture, expected=expected
        )
    if not source.is_file():
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path=str(path),
                    reason="Bundle source must be an existing directory or zip file.",
                )
            ]
        )
    with tempfile.TemporaryDirectory(prefix="preprocessing-bundle-") as tmp:
        extract_root = Path(tmp)
        unpack_report = unpack_zip_bytes(source.read_bytes(), extract_root)
        if not unpack_report.valid:
            return unpack_report
        return validate_preprocessing_bundle(
            extract_root, architecture=architecture, expected=expected
        )


def validate_packaged_release(
    package_root: Path,
    bundle_root: Path | None = None,
    *,
    load_state_dict: Any = None,
) -> PackageValidationResult:
    """Validate a model package and its bound preprocessing bundle."""

    issues: list[PackageValidationIssue] = []
    package_result = validate_model_package(package_root, load_state_dict=load_state_dict)
    issues.extend(package_result.issues)
    manifest = try_load_model_package_manifest(package_root)
    if manifest is None:
        return PackageValidationResult.from_issues(issues)

    if manifest.format != PACKAGE_FORMAT_V2:
        return PackageValidationResult.from_issues(issues)

    if bundle_root is None:
        issues.append(
            PackageValidationIssue(
                path="preprocessing_bundle",
                reason="attribute-aware-gae-v2 packages require a preprocessing bundle.",
            )
        )
        return PackageValidationResult.from_issues(issues)

    bundle_result = validate_preprocessing_bundle(
        bundle_root,
        architecture=manifest.architecture,
        expected=manifest.preprocessing_bundle,
    )
    issues.extend(bundle_result.issues)
    return PackageValidationResult.from_issues(issues)


def materialize_declared_bundle_files(bundle_root: Path, destination: Path) -> list[str]:
    """Copy manifest.json and declared bundle members into destination."""

    source_root = bundle_root.expanduser().resolve()
    destination_root = destination.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest, manifest_issues = _load_bundle_manifest(source_root)
    if manifest is None:
        reason = manifest_issues[0].reason if manifest_issues else "manifest.json is missing."
        raise ValueError(reason)

    planned: list[tuple[str, Path]] = []
    for relative in (
        MANIFEST_NAME,
        manifest.files.drain_config,
        manifest.files.drain_parser,
        manifest.files.embeddings,
    ):
        located, path_issues = _resolve_declared_file(source_root, relative)
        if located is None:
            reason = path_issues[0].reason if path_issues else "Declared path is missing."
            raise ValueError(f"{relative}: {reason}")
        target = (destination_root / relative).resolve()
        try:
            target.relative_to(destination_root)
        except ValueError as error:
            raise ValueError(f"{relative}: destination path must stay inside the prefix.") from error
        planned.append((relative, located))

    copied: list[str] = []
    for relative, located in planned:
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(located.read_bytes())
        copied.append(relative)
    return copied


def _load_bundle_manifest(
    bundle_root: Path,
) -> tuple[BundleManifest | None, list[PackageValidationIssue]]:
    manifest_path = bundle_root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None, [
            PackageValidationIssue(path=MANIFEST_NAME, reason="manifest.json is missing.")
        ]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, [
            PackageValidationIssue(
                path=MANIFEST_NAME,
                reason="manifest.json must be readable UTF-8 JSON.",
            )
        ]
    if not isinstance(payload, dict):
        return None, [
            PackageValidationIssue(
                path=MANIFEST_NAME,
                reason="manifest.json must contain a JSON object.",
            )
        ]
    try:
        return BundleManifest.model_validate(payload), []
    except Exception as error:
        return None, _pydantic_issues(error)


def _pydantic_issues(error: Exception) -> list[PackageValidationIssue]:
    issues: list[PackageValidationIssue] = []
    errors = getattr(error, "errors", None)
    if callable(errors):
        for item in errors():
            location = ".".join(str(part) for part in item.get("loc", ()))
            path = f"{MANIFEST_NAME}:{location}" if location else MANIFEST_NAME
            issues.append(
                PackageValidationIssue(
                    path=path,
                    reason=str(item.get("msg", "Manifest field is invalid.")),
                )
            )
        if issues:
            return issues
    return [
        PackageValidationIssue(path=MANIFEST_NAME, reason="manifest.json does not match the contract.")
    ]


def _resolve_declared_file(
    bundle_root: Path,
    relative: str,
) -> tuple[Path | None, list[PackageValidationIssue]]:
    try:
        _require_relative_posix(relative)
    except ValueError as error:
        return None, [PackageValidationIssue(path=relative, reason=str(error))]
    located = bundle_root / relative
    if located.is_symlink() or not located.is_file():
        return None, [
            PackageValidationIssue(
                path=relative,
                reason="Declared path must be a regular file inside the bundle.",
            )
        ]
    candidate = located.resolve()
    try:
        candidate.relative_to(bundle_root.resolve())
    except ValueError:
        return None, [
            PackageValidationIssue(
                path=relative,
                reason="Declared path must stay inside the bundle root.",
            )
        ]
    if candidate.stat().st_size <= 0:
        return None, [
            PackageValidationIssue(
                path=relative,
                reason="Declared path must be a non-empty regular file.",
            )
        ]
    return candidate, []


def _undeclared_file_issues(bundle_root: Path, allowed: set[str]) -> list[PackageValidationIssue]:
    issues: list[PackageValidationIssue] = []
    for path in bundle_root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            issues.append(
                PackageValidationIssue(
                    path=relative,
                    reason="Bundle members must not be symbolic links.",
                )
            )
            continue
        if relative not in allowed:
            issues.append(
                PackageValidationIssue(
                    path=relative,
                    reason="Undeclared bundle files are not allowed.",
                )
            )
    return issues


def _embedding_issues(
    path: Path,
    relative: str,
    architecture: PackageArchitecture | None,
) -> list[PackageValidationIssue]:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as payload:
            names = list(payload.files)
            loaded = {name: np.array(payload[name], copy=True) for name in names}
    except (OSError, ValueError, KeyError):
        return [
            PackageValidationIssue(
                path=relative,
                reason="Embeddings archive must load as a safe NPZ without pickled objects.",
            )
        ]
    if set(loaded) != {"cluster_ids", "embeddings"}:
        return [
            PackageValidationIssue(
                path=relative,
                reason="Embeddings archive must declare cluster_ids and embeddings arrays.",
            )
        ]
    cluster_ids = loaded["cluster_ids"]
    embeddings = loaded["embeddings"]

    issues: list[PackageValidationIssue] = []
    if cluster_ids.ndim != 1 or cluster_ids.size == 0:
        issues.append(
            PackageValidationIssue(
                path=relative,
                reason="Embeddings cluster_ids must be a non-empty 1-D integer array.",
            )
        )
        return issues
    if not np.issubdtype(cluster_ids.dtype, np.integer):
        issues.append(
            PackageValidationIssue(
                path=relative,
                reason="Embeddings cluster_ids must be integers.",
            )
        )
        return issues
    ids = np.asarray(cluster_ids, dtype=np.int64)
    if ids.size != np.unique(ids).size:
        issues.append(
            PackageValidationIssue(
                path=relative,
                reason="Embeddings cluster_ids must be unique.",
            )
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != ids.size or embeddings.shape[1] <= 0:
        issues.append(
            PackageValidationIssue(
                path=relative,
                reason="Embeddings must be a non-empty 2-D array aligned with cluster_ids.",
            )
        )
        return issues
    vectors = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(vectors).all():
        issues.append(
            PackageValidationIssue(
                path=relative,
                reason="Embeddings values must be finite float32 values.",
            )
        )
    if architecture is not None:
        expected_node_dim = int(vectors.shape[1]) + NODE_FEATURE_EXTRA_DIM
        if expected_node_dim != architecture.node_dim:
            issues.append(
                PackageValidationIssue(
                    path=relative,
                    reason=(
                        "Embedding width plus node extras must equal architecture.node_dim."
                    ),
                )
            )
    return issues


def _require_relative_posix(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value or Path(value).is_absolute():
        raise ValueError("path must be a relative POSIX path")
    if ".." in Path(value).parts:
        raise ValueError("path must not contain '..' segments")
    return value
