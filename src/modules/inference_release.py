"""Export a trusted HDFS training result into a v2 model package and bundle.

This module is a training-side release tool. It never fits Drain, never loads
a public untrusted pickle, and never packages the training graph bundle.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.modules.artifacts import git_revision
from src.modules.inference_bundle import (
    BUNDLE_FORMAT,
    DRAIN_CONFIG_NAME,
    DRAIN_PARSER_NAME,
    EMBEDDINGS_NAME,
    BundleFiles,
    BundleManifest,
    bundle_digest as compute_bundle_digest,
)
from src.modules.model_package import (
    DEFAULT_ARTIFACT_NAME,
    DEFAULT_EVIDENCE_NAME,
    MANIFEST_NAME,
    PACKAGE_FORMAT_V2,
    ModelPackageManifest,
    PackageArchitecture,
    PackageFiles,
    PackageMetrics,
    PackageModel,
    PackageScoring,
    PreprocessingBundleRef,
)

MODEL_PACKAGE_FORMAT_V2 = PACKAGE_FORMAT_V2
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9_.-]+$"
_FIXED_ZIP_DATE = (2020, 1, 1, 0, 0, 0)


class InferenceReleaseError(ValueError):
    """Raised when a trusted HDFS training result cannot be exported."""


class HdfsReleaseSource(PackageModel):
    """Explicit completed-stage paths required to export an HDFS release."""

    checkpoint_path: Path
    parser_state_path: Path
    drain_config_path: Path
    embeddings_path: Path
    labels_path: Path
    metrics_path: Path


class HdfsReleaseIdentity(PackageModel):
    """Stable identifiers written into the model package and bundle manifests."""

    model_identifier: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    version: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN)
    bundle_identifier: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    bundle_version: str = Field(min_length=1, max_length=64, pattern=_IDENTIFIER_PATTERN)
    pipeline_run_id: str | None = Field(default=None, min_length=1)


class InspectedArchiveFile(PackageModel):
    """One declared archive member and its SHA-256 digest."""

    name: str
    sha256: str
    size_bytes: int = Field(ge=0)


class InspectedInferenceRelease(BaseModel):
    """Checksum-verified view of an exported model package and bundle."""

    model_config = ConfigDict(extra="forbid")

    model_manifest: ModelPackageManifest
    bundle_manifest: BundleManifest
    model_files: list[InspectedArchiveFile]
    bundle_files: list[InspectedArchiveFile]


class ExportedInferenceRelease(BaseModel):
    """Filesystem location of an atomically written HDFS inference release."""

    model_config = ConfigDict(extra="forbid")

    model_package_path: Path
    preprocessing_bundle_path: Path
    model_manifest: ModelPackageManifest
    bundle_manifest: BundleManifest


def export_hdfs_inference_release(
    *,
    config: Mapping[str, Any],
    source: HdfsReleaseSource,
    identity: HdfsReleaseIdentity,
    output_dir: Path,
    code_root: Path,
) -> ExportedInferenceRelease:
    """Write a v2 model ZIP and a separate preprocessing-bundle ZIP atomically."""

    _reject_non_hdfs_config(config)
    _reject_smoke_config(config)
    _require_source_files(source)

    drain_config_bytes = source.drain_config_path.read_bytes()
    parser_bytes = source.parser_state_path.read_bytes()
    embeddings_bytes = _canonical_embeddings_bytes(source.embeddings_path)
    metrics_payload = _load_metrics(source.metrics_path)
    checkpoint = _load_training_checkpoint(source.checkpoint_path)
    artifact_bytes = _tensor_only_state_dict_bytes(checkpoint["model_state_dict"])

    bundle_files = BundleFiles(
        drain_config=DRAIN_CONFIG_NAME,
        drain_parser=DRAIN_PARSER_NAME,
        embeddings=EMBEDDINGS_NAME,
        checksums={
            DRAIN_CONFIG_NAME: _sha256_bytes(drain_config_bytes),
            DRAIN_PARSER_NAME: _sha256_bytes(parser_bytes),
            EMBEDDINGS_NAME: _sha256_bytes(embeddings_bytes),
        },
    )
    declared_digest = compute_bundle_digest(bundle_files.checksums)
    bundle_manifest = BundleManifest(
        identifier=identity.bundle_identifier,
        version=identity.bundle_version,
        source_compatibility="hdfs",
        format=BUNDLE_FORMAT,
        digest=declared_digest,
        files=bundle_files,
    )
    bundle_manifest_bytes = _json_bytes(bundle_manifest.model_dump(mode="json"))

    architecture = _architecture_from_checkpoint(checkpoint, config)
    scoring = _scoring_from_checkpoint(checkpoint, config)
    metrics = PackageMetrics.model_validate(metrics_payload)
    evidence_bytes = _json_bytes(
        {
            "kind": "hdfs-inference-release",
            "code_revision": git_revision(code_root),
            "source": "run_ablation.py",
            "config": _public_config(config),
            "metrics": metrics_payload,
        }
    )
    model_files = PackageFiles(
        artifact=DEFAULT_ARTIFACT_NAME,
        evidence=DEFAULT_EVIDENCE_NAME,
        checksums={
            DEFAULT_ARTIFACT_NAME: _sha256_bytes(artifact_bytes),
            DEFAULT_EVIDENCE_NAME: _sha256_bytes(evidence_bytes),
        },
    )
    model_manifest = ModelPackageManifest(
        model_identifier=identity.model_identifier,
        version=identity.version,
        source_compatibility="hdfs",
        format=MODEL_PACKAGE_FORMAT_V2,
        pipeline_run_id=identity.pipeline_run_id,
        metrics=metrics,
        architecture=architecture,
        scoring=scoring,
        files=model_files,
        preprocessing_bundle=PreprocessingBundleRef(
            identifier=identity.bundle_identifier,
            version=identity.bundle_version,
            digest=declared_digest,
        ),
    )
    model_manifest_bytes = _json_bytes(model_manifest.model_dump(mode="json"))

    bundle_name = f"{identity.bundle_identifier}-{identity.bundle_version}.zip"
    model_name = f"{identity.model_identifier}-{identity.version}.zip"
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".inference-release-", dir=destination))
    bundle_tmp = staging / bundle_name
    model_tmp = staging / model_name
    final_bundle = destination / bundle_name
    final_model = destination / model_name
    try:
        _write_zip(
            bundle_tmp,
            {
                MANIFEST_NAME: bundle_manifest_bytes,
                DRAIN_CONFIG_NAME: drain_config_bytes,
                DRAIN_PARSER_NAME: parser_bytes,
                EMBEDDINGS_NAME: embeddings_bytes,
            },
        )
        _write_zip(
            model_tmp,
            {
                MANIFEST_NAME: model_manifest_bytes,
                DEFAULT_ARTIFACT_NAME: artifact_bytes,
                DEFAULT_EVIDENCE_NAME: evidence_bytes,
            },
        )
        os.replace(bundle_tmp, final_bundle)
        try:
            os.replace(model_tmp, final_model)
        except OSError:
            final_bundle.unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return ExportedInferenceRelease(
        model_package_path=final_model,
        preprocessing_bundle_path=final_bundle,
        model_manifest=model_manifest,
        bundle_manifest=bundle_manifest,
    )


def inspect_inference_release(
    model_package: Path,
    preprocessing_bundle: Path,
) -> InspectedInferenceRelease:
    """Validate declared members and checksums of an exported release pair."""

    model_members = _read_zip_members(model_package)
    bundle_members = _read_zip_members(preprocessing_bundle)
    model_manifest = ModelPackageManifest.model_validate(
        _load_manifest_payload(model_members, model_package.name)
    )
    bundle_manifest = BundleManifest.model_validate(
        _load_manifest_payload(bundle_members, preprocessing_bundle.name)
    )
    model_files = _declared_files(
        model_members,
        (model_manifest.files.artifact, model_manifest.files.evidence),
        model_manifest.files.checksums,
        archive_name=model_package.name,
    )
    bundle_files = _declared_files(
        bundle_members,
        (
            bundle_manifest.files.drain_config,
            bundle_manifest.files.drain_parser,
            bundle_manifest.files.embeddings,
        ),
        bundle_manifest.files.checksums,
        archive_name=preprocessing_bundle.name,
    )
    bundle_ref = model_manifest.preprocessing_bundle
    if bundle_ref is None:
        raise InferenceReleaseError("Model package is missing a preprocessing bundle descriptor.")
    if bundle_ref.digest != bundle_manifest.digest:
        raise InferenceReleaseError("Model package bundle digest does not match the bundle manifest.")
    if bundle_ref.identifier != bundle_manifest.identifier:
        raise InferenceReleaseError("Model package bundle identifier does not match the bundle manifest.")
    if bundle_ref.version != bundle_manifest.version:
        raise InferenceReleaseError("Model package bundle version does not match the bundle manifest.")
    _embeddings_from_bytes(bundle_members[bundle_manifest.files.embeddings])
    return InspectedInferenceRelease(
        model_manifest=model_manifest,
        bundle_manifest=bundle_manifest,
        model_files=model_files,
        bundle_files=bundle_files,
    )


def _reject_non_hdfs_config(config: Mapping[str, Any]) -> None:
    dataset = str(config.get("experiment", {}).get("dataset", "")).lower()
    if dataset != "hdfs":
        raise InferenceReleaseError("Inference releases can be exported only for HDFS, not BGL.")


def _reject_smoke_config(config: Mapping[str, Any]) -> None:
    training = config.get("training", {})
    if isinstance(training, Mapping) and bool(training.get("test_run")):
        raise InferenceReleaseError("Smoke/test-run outputs cannot be exported as an inference release.")


def _require_source_files(source: HdfsReleaseSource) -> None:
    required = (
        (source.labels_path, "HDFS labels file"),
        (source.parser_state_path, "Drain parser snapshot"),
        (source.embeddings_path, "Embeddings archive"),
        (source.drain_config_path, "Drain configuration"),
        (source.checkpoint_path, "Training checkpoint"),
        (source.metrics_path, "Training metrics"),
    )
    for path, label in required:
        _require_regular_file(path, label)


def _require_regular_file(path: Path, label: str) -> None:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size <= 0:
        raise InferenceReleaseError(f"{label} is missing.")


def _canonical_embeddings_bytes(path: Path) -> bytes:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as payload:
            files = set(payload.files)
            if "cluster_ids" not in files or "embeddings" not in files:
                raise InferenceReleaseError(
                    "Embeddings archive must declare cluster_ids and embeddings arrays."
                )
            cluster_ids = np.array(payload["cluster_ids"], copy=True)
            embeddings = np.array(payload["embeddings"], copy=True)
    except InferenceReleaseError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise InferenceReleaseError("Embeddings archive is missing or is not a safe NPZ.") from error
    return _serialize_embeddings(cluster_ids, embeddings)


def _embeddings_from_bytes(payload: bytes) -> tuple[Any, Any]:
    import numpy as np

    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
        with loaded:
            cluster_ids = np.array(loaded["cluster_ids"], copy=True)
            embeddings = np.array(loaded["embeddings"], copy=True)
    except (OSError, ValueError, KeyError) as error:
        raise InferenceReleaseError("Bundle embeddings.npz is missing or is not a safe NPZ.") from error
    _serialize_embeddings(cluster_ids, embeddings)
    return cluster_ids, embeddings


def _serialize_embeddings(cluster_ids: Any, embeddings: Any) -> bytes:
    import numpy as np

    if cluster_ids.ndim != 1 or cluster_ids.size == 0:
        raise InferenceReleaseError("Embeddings cluster_ids must be a non-empty 1-D integer array.")
    if not np.issubdtype(cluster_ids.dtype, np.integer):
        raise InferenceReleaseError("Embeddings cluster_ids must be integers.")
    ids = np.asarray(cluster_ids, dtype=np.int64)
    if ids.size != np.unique(ids).size:
        raise InferenceReleaseError("Embeddings cluster_ids must be unique.")
    if embeddings.ndim != 2 or embeddings.shape[0] != ids.size or embeddings.shape[1] <= 0:
        raise InferenceReleaseError("Embeddings must be a non-empty 2-D array aligned with cluster_ids.")
    vectors = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise InferenceReleaseError("Embeddings values must be finite float32 values.")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, cluster_ids=ids, embeddings=vectors)
    return buffer.getvalue()


def _load_metrics(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceReleaseError("Training metrics must be readable UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise InferenceReleaseError("Training metrics must include best_threshold.")
    raw_test = payload.get("test")
    raw_val = payload.get("val")
    test: Mapping[str, Any] = raw_test if isinstance(raw_test, Mapping) else {}
    val: Mapping[str, Any] = raw_val if isinstance(raw_val, Mapping) else {}
    flattened = {
        "best_threshold": payload.get("best_threshold"),
        "test_f1": payload.get("test_f1", test.get("f1")),
        "test_pr_auc": payload.get("test_pr_auc", test.get("pr_auc")),
        "test_roc_auc": payload.get("test_roc_auc", test.get("roc_auc")),
        "val_f1": payload.get("val_f1", val.get("f1")),
        "val_pr_auc": payload.get("val_pr_auc", val.get("pr_auc")),
        "val_roc_auc": payload.get("val_roc_auc", val.get("roc_auc")),
    }
    if flattened["best_threshold"] is None:
        raise InferenceReleaseError("Training metrics must include best_threshold.")
    return {key: value for key, value in flattened.items() if value is not None}


def _load_training_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise InferenceReleaseError(
            "Training checkpoint must load as a weights_only mapping with model_state_dict."
        ) from error
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise InferenceReleaseError("Training checkpoint must contain model_state_dict.")
    settings = _training_settings(payload)
    if bool(settings.get("test_run")):
        raise InferenceReleaseError("Smoke/test-run outputs cannot be exported as an inference release.")
    return payload


def _tensor_only_state_dict_bytes(state_dict: Any) -> bytes:
    import torch

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise InferenceReleaseError("model_state_dict must be a non-empty tensor mapping.")
    for key, value in state_dict.items():
        if getattr(value, "shape", None) is None or getattr(value, "dtype", None) is None:
            raise InferenceReleaseError(f"model_state_dict entry {key!r} must be a tensor.")
    buffer = io.BytesIO()
    torch.save(dict(state_dict), buffer)
    return buffer.getvalue()


def _training_settings(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    for key in ("training", "experiment"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    training = (config or {}).get("training")
    return training if isinstance(training, Mapping) else {}


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    raise TypeError("expected a sequence of floats")


def _architecture_from_checkpoint(
    checkpoint: Mapping[str, Any], config: Mapping[str, Any]
) -> PackageArchitecture:
    settings = _training_settings(checkpoint)
    release_config = config.get("inference_release", {})
    feature_contract = (
        release_config.get("feature_contract", "stabilized_v2")
        if isinstance(release_config, Mapping)
        else "stabilized_v2"
    )
    try:
        return PackageArchitecture.model_validate(
            {
                "node_dim": checkpoint["node_dim"],
                "edge_dim": checkpoint["edge_dim"],
                "hidden_dim": checkpoint.get("hidden_dim", settings.get("hidden_dim")),
                "latent_dim": checkpoint.get("latent_dim", settings.get("latent_dim")),
                "gine_aggregation": checkpoint.get(
                    "gine_aggregation", settings.get("gine_aggregation")
                ),
                "node_transformation": checkpoint.get(
                    "node_transformation", settings.get("node_transformation")
                ),
                "edge_mean": _as_float_list(checkpoint.get("edge_mean")),
                "edge_std": _as_float_list(checkpoint.get("edge_std")),
                "feature_contract": feature_contract,
            }
        )
    except Exception as error:
        raise InferenceReleaseError(
            "Training checkpoint does not contain a complete HDFS architecture."
        ) from error


def _scoring_from_checkpoint(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> PackageScoring:
    settings = _training_settings(checkpoint, config)
    try:
        return PackageScoring.model_validate(
            {
                "alpha": settings["alpha"],
                "beta": settings["beta"],
                "gamma": settings["gamma"],
            }
        )
    except Exception as error:
        raise InferenceReleaseError("Training checkpoint does not contain scoring weights.") from error


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "__pipeline__"}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_zip(destination: Path, members: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)


def _read_zip_members(path: Path) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            return {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
    except (OSError, zipfile.BadZipFile) as error:
        raise InferenceReleaseError("Release archive must be a readable zip file.") from error


def _load_manifest_payload(members: Mapping[str, bytes], archive_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(members[MANIFEST_NAME].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceReleaseError(f"{archive_name} is missing a readable manifest.json.") from error
    if not isinstance(payload, dict):
        raise InferenceReleaseError("manifest.json must contain a JSON object.")
    return payload


def _declared_files(
    members: Mapping[str, bytes],
    relative_names: tuple[str, ...],
    checksums: Mapping[str, str],
    *,
    archive_name: str,
) -> list[InspectedArchiveFile]:
    inspected: list[InspectedArchiveFile] = []
    for name in relative_names:
        if name not in members:
            raise InferenceReleaseError(f"{archive_name} is missing declared file {name}.")
        payload = members[name]
        digest = _sha256_bytes(payload)
        expected = checksums.get(name)
        if digest != expected:
            raise InferenceReleaseError(f"{archive_name} checksum mismatch for {name}.")
        inspected.append(InspectedArchiveFile(name=name, sha256=digest, size_bytes=len(payload)))
    return inspected
