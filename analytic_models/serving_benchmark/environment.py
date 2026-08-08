"""Frozen software, checkpoint, and quantization environment for a campaign."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path
from typing import Any

from .io import read_json, sha256_json, write_json_atomic
from .manifest import BenchmarkManifest
from .inventory import EPHEMERAL_MODEL_CACHE_STORAGE, PERSISTENT_WORKSPACE_STORAGE


LOCK_SCHEMA = "runpod-serving-environment-v1"


def storage_mode() -> str:
    return os.environ.get("PLENA_RUNPOD_STORAGE_MODE", PERSISTENT_WORKSPACE_STORAGE)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("vllm", "torch", "transformers", "nvidia-ml-py", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def software_identity(*, image_digest: str | None = None) -> dict[str, Any]:
    image = image_digest or os.environ.get("RUNPOD_IMAGE_DIGEST")
    return {
        "python": platform.python_version(),
        "packages": package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runpod_image_name": os.environ.get("RUNPOD_IMAGE_NAME"),
        "image_digest": image,
        "hf_home": os.environ.get("HF_HOME"),
        "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
        "storage_mode": storage_mode(),
    }


def resolve_model_revisions(manifest: BenchmarkManifest) -> dict[str, str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to resolve model revisions") from exc

    api = HfApi()
    revisions: dict[str, str] = {}
    for name, model in manifest.models.items():
        info = api.model_info(model.model_id, revision=model.revision)
        if not info.sha:
            raise RuntimeError(f"Hugging Face did not return a commit SHA for {model.model_id}")
        revisions[name] = str(info.sha)
    return revisions


def create_environment_lock(
    *,
    manifest: BenchmarkManifest,
    inventory: dict[str, Any],
    resolved_revisions: dict[str, str],
    quantization: str,
    image_digest: str,
    preflight_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    missing = set(manifest.models) - set(resolved_revisions)
    if missing:
        raise ValueError(f"missing resolved revisions for models: {sorted(missing)}")
    if not image_digest.startswith("sha256:") or len(image_digest) <= len("sha256:"):
        raise ValueError("formal preflight requires a container digest in sha256:<digest> form")
    lock = {
        "schema_version": LOCK_SCHEMA,
        "campaign": manifest.campaign,
        "manifest_hash": manifest.fingerprint,
        "inventory_hash": inventory.get("inventory_hash"),
        "storage_mode": inventory.get("storage_mode", PERSISTENT_WORKSPACE_STORAGE),
        "resolved_revisions": dict(sorted(resolved_revisions.items())),
        "quantization_backend": quantization,
        "software": software_identity(image_digest=image_digest),
        "preflight_artifacts": preflight_artifacts,
        "decode_semantics": "imported_kv_decode_proxy",
    }
    lock["environment_hash"] = sha256_json(lock)
    return lock


def write_environment_lock(path: Path, lock: dict[str, Any]) -> None:
    write_json_atomic(path, lock)


def load_environment_lock(path: Path) -> dict[str, Any]:
    lock = read_json(path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError(f"unsupported environment lock schema: {lock.get('schema_version')!r}")
    expected = lock.get("environment_hash")
    payload = dict(lock)
    payload.pop("environment_hash", None)
    if expected != sha256_json(payload):
        raise ValueError(f"environment lock hash mismatch: {path}")
    return lock


def validate_environment_lock(
    lock: dict[str, Any],
    *,
    manifest: BenchmarkManifest,
    image_digest: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if lock.get("manifest_hash") != manifest.fingerprint:
        errors.append("manifest hash differs from the frozen preflight manifest")
    current = software_identity(image_digest=image_digest)
    frozen = lock.get("software", {})
    if current.get("python") != frozen.get("python"):
        errors.append(f"Python changed: {frozen.get('python')} -> {current.get('python')}")
    for package, version in frozen.get("packages", {}).items():
        if current.get("packages", {}).get(package) != version:
            errors.append(
                f"package {package} changed: {version} -> {current.get('packages', {}).get(package)}"
            )
    if current.get("image_digest") != frozen.get("image_digest"):
        errors.append("container image digest differs from preflight")
    for path_name in ("hf_home", "vllm_cache_root"):
        if current.get(path_name) != frozen.get(path_name):
            errors.append(f"{path_name} differs from preflight")
    if set(lock.get("resolved_revisions", {})) != set(manifest.models):
        errors.append("environment lock does not cover every manifest model")
    return errors


def validate_runpod_persistent_paths(
    *paths: Path,
    required_storage_mode: str | None = None,
) -> None:
    selected_mode = required_storage_mode or storage_mode()
    if not os.environ.get("RUNPOD_POD_ID") and required_storage_mode is None:
        return
    workspace = Path("/workspace").resolve()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(workspace):
            raise ValueError(f"RunPod campaign artifacts must be under /workspace: {resolved}")
    for variable in ("HF_HOME", "VLLM_CACHE_ROOT"):
        value = os.environ.get(variable)
        if not value:
            raise ValueError(f"{variable} must be set before preflight")
        cache_path = Path(value).resolve()
        if selected_mode == PERSISTENT_WORKSPACE_STORAGE:
            if not cache_path.is_relative_to(workspace):
                raise ValueError(f"{variable} must point inside /workspace before preflight")
        elif selected_mode == EPHEMERAL_MODEL_CACHE_STORAGE:
            if cache_path.is_relative_to(workspace):
                raise ValueError(f"{variable} must point outside /workspace in ephemeral-cache mode")
        else:
            raise ValueError(f"unsupported storage mode: {selected_mode}")
