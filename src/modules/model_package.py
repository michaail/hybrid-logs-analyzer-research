"""Trusted HDFS model-package contract. This module never imports PyTorch."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_NAME = "manifest.json"
DEFAULT_ARTIFACT_NAME = "model.pt"
DEFAULT_EVIDENCE_NAME = "evidence.json"
PACKAGE_FORMAT_V1 = "attribute-aware-gae-v1"
PACKAGE_FORMAT_V2 = "attribute-aware-gae-v2"
PACKAGE_FORMAT = PACKAGE_FORMAT_V2
MAX_ZIP_MEMBERS = 64
MAX_ZIP_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 96 * 1024 * 1024
_NESTED_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".gz")
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9_.-]+$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FLOAT_DTYPES = frozenset({"float32", "torch.float32"})
_INT64_DTYPES = frozenset({"int64", "torch.int64"})
_SHA256_PATTERN_COMPILED = re.compile(_SHA256_PATTERN)
_UNIX_SYMLINK_MASK = 0o170000
_UNIX_SYMLINK = 0o120000
StateDictLoader = Callable[[Path], Mapping[str, Any]]


class PackageModel(BaseModel):
    """Closed package-schema base that rejects unspecified fields."""

    model_config = ConfigDict(extra="forbid")


class PackageValidationIssue(PackageModel):
    """One completeness or compatibility failure inside a package."""

    path: str
    reason: str


class PackageValidationResult(PackageModel):
    """Collected eligibility report; valid only when issues is empty."""

    valid: bool
    issues: list[PackageValidationIssue]

    @classmethod
    def from_issues(cls, issues: Sequence[PackageValidationIssue]) -> PackageValidationResult:
        collected = list(issues)
        return cls(valid=len(collected) == 0, issues=collected)


class PackageMetrics(BaseModel):
    """Evaluation metrics; only best_threshold is required for eligibility."""

    model_config = ConfigDict(extra="allow")

    best_threshold: float

    @field_validator("best_threshold")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("best_threshold must be a finite number")
        return value


class PackageArchitecture(PackageModel):
    """Declared AttributeAwareGAE reconstruction fields."""

    node_dim: int = Field(gt=0)
    edge_dim: int = Field(ge=0)
    hidden_dim: int = Field(gt=0)
    latent_dim: int = Field(gt=0)
    gine_aggregation: Literal["sum", "mean", "max"]
    node_transformation: Literal["mlp", "linear"]
    structure_decoder: Literal["mlp", "inner_product"] = "inner_product"
    edge_mean: list[float] | None = None
    edge_std: list[float] | None = None
    feature_contract: Literal["notebook_raw_v1", "stabilized_v2"] = "stabilized_v2"

    @model_validator(mode="after")
    def _paired_normalization(self) -> PackageArchitecture:
        if (self.edge_mean is None) != (self.edge_std is None):
            raise ValueError("edge_mean and edge_std must both be null or both present")
        if self.edge_mean is None or self.edge_std is None:
            return self
        if len(self.edge_mean) != self.edge_dim or len(self.edge_std) != self.edge_dim:
            raise ValueError("edge_mean and edge_std must each have length edge_dim")
        if any(not math.isfinite(value) for value in (*self.edge_mean, *self.edge_std)):
            raise ValueError("edge_mean and edge_std values must be finite")
        if any(value < 0.1 for value in self.edge_std):
            raise ValueError("each edge_std value must be at least 0.1")
        return self


class PackageScoring(PackageModel):
    """Anomaly-score mix weights used with the declared threshold."""

    alpha: float
    beta: float
    gamma: float

    @model_validator(mode="after")
    def _non_negative_and_active(self) -> PackageScoring:
        values = (self.alpha, self.beta, self.gamma)
        if any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("alpha, beta, and gamma must be finite and non-negative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one of alpha, beta, or gamma must be positive")
        return self


class PreprocessingBundleRef(PackageModel):
    """Immutable pointer from a v2 model package to its preprocessing bundle."""

    identifier: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    version: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN)
    digest: str = Field(pattern=_SHA256_PATTERN)


class PackageFiles(PackageModel):
    """Declared package members and their SHA-256 digests."""

    artifact: str
    evidence: str
    checksums: dict[str, str]

    @field_validator("artifact", "evidence")
    @classmethod
    def _relative_posix_path(cls, value: str) -> str:
        return _require_relative_posix(value)

    @field_validator("checksums")
    @classmethod
    def _lowercase_checksums(cls, value: dict[str, str]) -> dict[str, str]:
        for path, digest in value.items():
            _require_relative_posix(path)
            if not _SHA256_PATTERN_COMPILED.match(digest):
                raise ValueError(f"checksum for {path} must be a lowercase hex SHA-256")
        return value

    @model_validator(mode="after")
    def _checksums_match_declared_files(self) -> PackageFiles:
        expected = {self.artifact, self.evidence}
        actual = set(self.checksums)
        if actual != expected:
            raise ValueError("checksums must contain exactly the artifact and evidence paths")
        return self


class ModelPackageManifest(PackageModel):
    """Closed inspectable contract for an HDFS AttributeAwareGAE package."""

    model_identifier: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    version: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN)
    source_compatibility: Literal["hdfs"]
    format: str
    pipeline_run_id: str | None = Field(default=None, min_length=1)
    metrics: PackageMetrics
    architecture: PackageArchitecture
    scoring: PackageScoring
    files: PackageFiles
    preprocessing_bundle: PreprocessingBundleRef

    @field_validator("format")
    @classmethod
    def _v2_format_only(cls, value: str) -> str:
        if value != PACKAGE_FORMAT_V2:
            raise ValueError(
                "format must be attribute-aware-gae-v2; "
                "attribute-aware-gae-v1 is not an accepted package format."
            )
        return value


class TensorSpec:
    """Expected tensor identity for one state-dict entry."""

    def __init__(self, shape: tuple[int, ...], dtypes: frozenset[str]) -> None:
        self.shape = shape
        self.dtypes = dtypes


def expected_state_dict_spec(architecture: PackageArchitecture) -> dict[str, TensorSpec]:
    """Return the AttributeAwareGAE tensor key set for an architecture."""

    node_dim = architecture.node_dim
    edge_dim = architecture.edge_dim
    hidden_dim = architecture.hidden_dim
    latent_dim = architecture.latent_dim
    float_spec = _FLOAT_DTYPES
    int_spec = _INT64_DTYPES
    spec: dict[str, TensorSpec] = {
        "raw_node_norm.running_mean": TensorSpec((node_dim,), float_spec),
        "raw_node_norm.running_var": TensorSpec((node_dim,), float_spec),
        "raw_node_norm.num_batches_tracked": TensorSpec((), int_spec),
        "node_proj.weight": TensorSpec((hidden_dim, node_dim), float_spec),
        "node_proj.bias": TensorSpec((hidden_dim,), float_spec),
        "edge_proj.weight": TensorSpec((hidden_dim, edge_dim), float_spec),
        "edge_proj.bias": TensorSpec((hidden_dim,), float_spec),
        "encoder_conv.eps": TensorSpec((1,), float_spec),
        "encoder_conv.lin.weight": TensorSpec((hidden_dim, hidden_dim), float_spec),
        "encoder_conv.lin.bias": TensorSpec((hidden_dim,), float_spec),
        "node_decoder.0.weight": TensorSpec((hidden_dim, latent_dim), float_spec),
        "node_decoder.0.bias": TensorSpec((hidden_dim,), float_spec),
        "node_decoder.2.weight": TensorSpec((node_dim, hidden_dim), float_spec),
        "node_decoder.2.bias": TensorSpec((node_dim,), float_spec),
        "edge_decoder.0.weight": TensorSpec((hidden_dim, latent_dim * 2), float_spec),
        "edge_decoder.0.bias": TensorSpec((hidden_dim,), float_spec),
        "edge_decoder.2.weight": TensorSpec((edge_dim, hidden_dim), float_spec),
        "edge_decoder.2.bias": TensorSpec((edge_dim,), float_spec),
    }
    if architecture.structure_decoder == "mlp":
        spec.update(
            {
                "structure_decoder.0.weight": TensorSpec((hidden_dim, latent_dim * 2), float_spec),
                "structure_decoder.0.bias": TensorSpec((hidden_dim,), float_spec),
                "structure_decoder.2.weight": TensorSpec((1, hidden_dim), float_spec),
                "structure_decoder.2.bias": TensorSpec((1,), float_spec),
            }
        )
    if architecture.node_transformation == "mlp":
        spec.update(
            {
                "encoder_conv.nn.0.weight": TensorSpec((hidden_dim, hidden_dim), float_spec),
                "encoder_conv.nn.0.bias": TensorSpec((hidden_dim,), float_spec),
                "encoder_conv.nn.1.weight": TensorSpec((hidden_dim,), float_spec),
                "encoder_conv.nn.1.bias": TensorSpec((hidden_dim,), float_spec),
                "encoder_conv.nn.1.running_mean": TensorSpec((hidden_dim,), float_spec),
                "encoder_conv.nn.1.running_var": TensorSpec((hidden_dim,), float_spec),
                "encoder_conv.nn.1.num_batches_tracked": TensorSpec((), int_spec),
                "encoder_conv.nn.3.weight": TensorSpec((latent_dim, hidden_dim), float_spec),
                "encoder_conv.nn.3.bias": TensorSpec((latent_dim,), float_spec),
            }
        )
    else:
        spec.update(
            {
                "encoder_conv.nn.weight": TensorSpec((latent_dim, hidden_dim), float_spec),
                "encoder_conv.nn.bias": TensorSpec((latent_dim,), float_spec),
            }
        )
    return spec


def validate_state_dict(
    payload: Mapping[str, Any],
    architecture: PackageArchitecture,
) -> list[PackageValidationIssue]:
    """Check a loaded mapping against the static AttributeAwareGAE tensor schema."""

    issues: list[PackageValidationIssue] = []
    if not payload:
        issues.append(
            PackageValidationIssue(
                path="files.artifact",
                reason="Artifact state dict must be a non-empty tensor mapping.",
            )
        )
        return issues
    expected = expected_state_dict_spec(architecture)
    actual_keys = set(payload)
    expected_keys = set(expected)
    for missing in sorted(expected_keys - actual_keys):
        issues.append(
            PackageValidationIssue(
                path=f"files.artifact:{missing}",
                reason="Required state-dict tensor is missing.",
            )
        )
    for unexpected in sorted(actual_keys - expected_keys):
        issues.append(
            PackageValidationIssue(
                path=f"files.artifact:{unexpected}",
                reason="Unexpected state-dict entry is not part of AttributeAwareGAE.",
            )
        )
    for key in sorted(actual_keys & expected_keys):
        issues.extend(_check_tensor_entry(key, payload[key], expected[key]))
    return issues


def validate_model_package(
    root: Path,
    *,
    load_state_dict: StateDictLoader | None = None,
) -> PackageValidationResult:
    """Collect every directory-package eligibility failure without running a model.

    The tensor probe runs only when ``load_state_dict`` is provided. The isolated
    package-validation process always supplies a ``weights_only=True`` loader.
    This module never imports PyTorch.
    """

    issues: list[PackageValidationIssue] = []
    package_root = root.expanduser()
    if not package_root.is_dir():
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path=str(root),
                    reason="Package root must be an existing directory.",
                )
            ]
        )

    manifest, manifest_issues = _load_manifest(package_root)
    issues.extend(manifest_issues)
    if manifest is None:
        return PackageValidationResult.from_issues(issues)

    declared_paths = (manifest.files.artifact, manifest.files.evidence)
    resolved: dict[str, Path] = {}
    for relative in declared_paths:
        located, path_issues = _resolve_declared_file(package_root, relative)
        issues.extend(path_issues)
        if located is not None:
            resolved[relative] = located

    for relative, digest in manifest.files.checksums.items():
        located = resolved.get(relative)
        if located is None:
            continue
        actual = _sha256_file(located)
        if actual != digest:
            issues.append(
                PackageValidationIssue(
                    path=relative,
                    reason="SHA-256 does not match files.checksums.",
                )
            )

    evidence_path = resolved.get(manifest.files.evidence)
    if evidence_path is not None:
        issues.extend(_validate_evidence_file(evidence_path, manifest.files.evidence))

    artifact_relative = manifest.files.artifact
    if not artifact_relative.endswith(".pt"):
        issues.append(
            PackageValidationIssue(
                path="files.artifact",
                reason="Artifact path must end with .pt.",
            )
        )
    artifact_path = resolved.get(artifact_relative)
    if artifact_path is not None and load_state_dict is not None:
        issues.extend(_probe_state_dict(artifact_path, artifact_relative, manifest, load_state_dict))

    return PackageValidationResult.from_issues(issues)


def validate_model_package_source(
    path: Path,
    *,
    load_state_dict: StateDictLoader | None = None,
) -> PackageValidationResult:
    """Validate a directory package or a zip that unpacks to one.

    Zip extraction is process-local and always deleted. This helper never
    returns a ZIP-member artifact reference and is not an admission-storage
    operation.
    """

    source = path.expanduser()
    if source.is_dir():
        return validate_model_package(source, load_state_dict=load_state_dict)
    if not source.is_file():
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path=str(path),
                    reason="Package source must be an existing directory or zip file.",
                )
            ]
        )
    if source.stat().st_size > MAX_ZIP_COMPRESSED_BYTES:
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path=source.name,
                    reason="Zip archive exceeds the 32 MiB compressed size limit.",
                )
            ]
        )
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile:
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path=source.name,
                    reason="Package source must be a directory or a readable zip archive.",
                )
            ]
        )
    with archive:
        zip_issues = _zip_transport_issues(archive)
        if zip_issues:
            return PackageValidationResult.from_issues(zip_issues)
        with tempfile.TemporaryDirectory(prefix="model-package-") as tmp:
            extract_root = Path(tmp)
            extract_issues = _extract_zip_archive(archive, extract_root)
            if extract_issues:
                return PackageValidationResult.from_issues(extract_issues)
            return validate_model_package(extract_root, load_state_dict=load_state_dict)


def package_zip_admission_preview_issues(archive_bytes: bytes) -> list[PackageValidationIssue]:
    """Read manifest.json from a ZIP member and reject v1 or a missing bundle.

    Transport failures are left to ``unpack_zip_bytes``. This preview never
    imports PyTorch and never extracts members.
    """

    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile:
        return []
    with archive:
        names = archive.namelist()
        if MANIFEST_NAME not in names:
            return []
        try:
            payload = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
    if not isinstance(payload, dict):
        return []
    if payload.get("format") == PACKAGE_FORMAT_V1:
        return [
            PackageValidationIssue(
                path="manifest.json:format",
                reason=(
                    "attribute-aware-gae-v1 is not an accepted package format."
                ),
            )
        ]
    if not payload.get("preprocessing_bundle"):
        return [
            PackageValidationIssue(
                path="preprocessing_bundle",
                reason="attribute-aware-gae-v2 packages require a preprocessing bundle.",
            )
        ]
    return []


def unpack_zip_bytes(archive_bytes: bytes, destination: Path) -> PackageValidationResult:
    """Unpack a zip into destination using the existing member and size caps.

    This is transport only: it does not run the directory contract or persist files.
    """

    if len(archive_bytes) > MAX_ZIP_COMPRESSED_BYTES:
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path="package",
                    reason="Zip archive exceeds the 32 MiB compressed size limit.",
                )
            ]
        )
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile:
        return PackageValidationResult.from_issues(
            [
                PackageValidationIssue(
                    path="package",
                    reason="Package source must be a readable zip archive.",
                )
            ]
        )
    with archive:
        zip_issues = _zip_transport_issues(archive)
        if zip_issues:
            return PackageValidationResult.from_issues(zip_issues)
        extract_issues = _extract_zip_archive(archive, destination)
        if extract_issues:
            return PackageValidationResult.from_issues(extract_issues)
    return PackageValidationResult.from_issues([])


def materialize_declared_package_files(package_root: Path, destination: Path) -> list[str]:
    """Copy manifest.json and declared artifact/evidence into destination.

    Extra undeclared files are not copied. Symlinks and path escape are refused.
    This helper does not unpack zips or import PyTorch.
    """

    source_root = package_root.expanduser().resolve()
    destination_root = destination.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    manifest, manifest_issues = _load_manifest(source_root)
    if manifest is None:
        reason = manifest_issues[0].reason if manifest_issues else "manifest.json is missing."
        raise ValueError(reason)

    planned: list[tuple[str, Path]] = []
    for relative in (MANIFEST_NAME, manifest.files.artifact, manifest.files.evidence):
        located, path_issues = _resolve_declared_file(source_root, relative)
        if located is None:
            reason = path_issues[0].reason if path_issues else "Declared path is missing."
            raise ValueError(f"{relative}: {reason}")
        target = (destination_root / relative).resolve()
        try:
            target.relative_to(destination_root)
        except ValueError as error:
            raise ValueError(
                f"{relative}: destination path must stay inside the prefix."
            ) from error
        planned.append((relative, located))

    copied: list[str] = []
    for relative, located in planned:
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(located.read_bytes())
        copied.append(relative)
    return copied


def try_load_model_package_manifest(package_root: Path) -> ModelPackageManifest | None:
    """Return a parsed manifest when the file is readable and contract-valid."""

    manifest, _issues = _load_manifest(package_root)
    return manifest


def _load_manifest(
    package_root: Path,
) -> tuple[ModelPackageManifest | None, list[PackageValidationIssue]]:
    manifest_path = package_root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None, [
            PackageValidationIssue(path=MANIFEST_NAME, reason="manifest.json is missing.")
        ]
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
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
        return ModelPackageManifest.model_validate(payload), []
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
    package_root: Path,
    relative: str,
) -> tuple[Path | None, list[PackageValidationIssue]]:
    try:
        _require_relative_posix(relative)
    except ValueError as error:
        return None, [PackageValidationIssue(path=relative, reason=str(error))]
    located = package_root / relative
    if located.is_symlink() or not located.is_file():
        return None, [
            PackageValidationIssue(
                path=relative,
                reason="Declared path must be a regular file inside the package.",
            )
        ]
    candidate = located.resolve()
    try:
        candidate.relative_to(package_root.resolve())
    except ValueError:
        return None, [
            PackageValidationIssue(
                path=relative,
                reason="Declared path must stay inside the package root.",
            )
        ]
    return candidate, []


def _validate_evidence_file(path: Path, relative: str) -> list[PackageValidationIssue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [
            PackageValidationIssue(
                path=relative,
                reason="Evidence file must be readable UTF-8 JSON.",
            )
        ]
    if not isinstance(payload, dict) or not payload:
        return [
            PackageValidationIssue(
                path=relative,
                reason="Evidence file must be a non-empty JSON object.",
            )
        ]
    return []


def _probe_state_dict(
    artifact_path: Path,
    relative: str,
    manifest: ModelPackageManifest,
    load_state_dict: StateDictLoader,
) -> list[PackageValidationIssue]:
    try:
        payload = load_state_dict(artifact_path)
    except Exception:
        return [
            PackageValidationIssue(
                path=relative,
                reason="Artifact must load as a weights_only tensor state dict.",
            )
        ]
    if not isinstance(payload, Mapping):
        return [
            PackageValidationIssue(
                path=relative,
                reason="Artifact state dict must be a mapping of tensor values.",
            )
        ]
    return validate_state_dict(payload, manifest.architecture)


def _check_tensor_entry(key: str, value: Any, spec: TensorSpec) -> list[PackageValidationIssue]:
    issues: list[PackageValidationIssue] = []
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return [
            PackageValidationIssue(
                path=f"files.artifact:{key}",
                reason="State-dict entry must be a tensor.",
            )
        ]
    actual_shape = tuple(int(dim) for dim in shape)
    if actual_shape != spec.shape:
        issues.append(
            PackageValidationIssue(
                path=f"files.artifact:{key}",
                reason=f"Tensor shape {actual_shape} does not match {spec.shape}.",
            )
        )
    dtype_name = _dtype_name(dtype)
    if dtype_name not in spec.dtypes:
        issues.append(
            PackageValidationIssue(
                path=f"files.artifact:{key}",
                reason=f"Tensor dtype {dtype_name} is not permitted for this key.",
            )
        )
    return issues


def _dtype_name(dtype: object) -> str:
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    text = str(dtype)
    return text.removeprefix("torch.") if text.startswith("torch.") else text


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_relative_posix(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value or Path(value).is_absolute():
        raise ValueError("path must be a relative POSIX path")
    if ".." in Path(value).parts:
        raise ValueError("path must not contain '..' segments")
    return value


def _zip_transport_issues(archive: zipfile.ZipFile) -> list[PackageValidationIssue]:
    issues: list[PackageValidationIssue] = []
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        issues.append(
            PackageValidationIssue(
                path=archive.filename or "package.zip",
                reason=f"Zip archive exceeds the {MAX_ZIP_MEMBERS}-member limit.",
            )
        )
    uncompressed = 0
    for info in members:
        uncompressed += max(info.file_size, 0)
        issues.extend(_zip_member_issues(info))
    if uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
        issues.append(
            PackageValidationIssue(
                path=archive.filename or "package.zip",
                reason="Zip archive exceeds the 96 MiB uncompressed size limit.",
            )
        )
    return issues


def _zip_member_issues(info: zipfile.ZipInfo) -> list[PackageValidationIssue]:
    name = info.filename.replace("\\", "/")
    issues: list[PackageValidationIssue] = []
    if not name or name.endswith("/") and name.strip("/") == "":
        issues.append(
            PackageValidationIssue(path=info.filename, reason="Zip member path is empty.")
        )
        return issues
    if name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
        issues.append(
            PackageValidationIssue(
                path=info.filename,
                reason="Zip member path must not be absolute.",
            )
        )
    if ".." in Path(name).parts:
        issues.append(
            PackageValidationIssue(
                path=info.filename,
                reason="Zip member path must not contain '..' segments.",
            )
        )
    lowered = name.rstrip("/").lower()
    if lowered.endswith(_NESTED_ARCHIVE_SUFFIXES):
        issues.append(
            PackageValidationIssue(
                path=info.filename,
                reason="Nested archive members are not allowed.",
            )
        )
    if _is_zip_symlink(info):
        issues.append(
            PackageValidationIssue(
                path=info.filename,
                reason="Zip member must not be a symbolic link.",
            )
        )
    return issues


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return (mode & _UNIX_SYMLINK_MASK) == _UNIX_SYMLINK


def _extract_zip_archive(
    archive: zipfile.ZipFile,
    destination: Path,
) -> list[PackageValidationIssue]:
    destination_root = destination.resolve()
    written_total = 0
    for info in archive.infolist():
        target, issue = _resolved_zip_member_path(destination_root, info.filename)
        if issue is not None:
            return [issue]
        if info.is_dir() or info.filename.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        declared = max(info.file_size, 0)
        member_written = 0
        try:
            with archive.open(info) as source, target.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    next_member = member_written + len(chunk)
                    next_total = written_total + len(chunk)
                    if next_member > declared:
                        return [
                            PackageValidationIssue(
                                path=info.filename,
                                reason="Zip member exceeds its declared uncompressed size.",
                            )
                        ]
                    if next_total > MAX_ZIP_UNCOMPRESSED_BYTES:
                        return [
                            PackageValidationIssue(
                                path=archive.filename or "package.zip",
                                reason="Zip archive exceeds the 96 MiB uncompressed size limit.",
                            )
                        ]
                    output.write(chunk)
                    member_written = next_member
                    written_total = next_total
        except zipfile.BadZipFile:
            return [
                PackageValidationIssue(
                    path=info.filename,
                    reason="Zip member is corrupt or exceeds its declared uncompressed size.",
                )
            ]
    return []


def _resolved_zip_member_path(
    destination_root: Path,
    member_name: str,
) -> tuple[Path, PackageValidationIssue | None]:
    relative = member_name.replace("\\", "/").lstrip("/")
    target = (destination_root / relative).resolve()
    try:
        target.relative_to(destination_root)
    except ValueError:
        return target, PackageValidationIssue(
            path=member_name,
            reason="Zip member path must stay inside the extract directory.",
        )
    return target, None
