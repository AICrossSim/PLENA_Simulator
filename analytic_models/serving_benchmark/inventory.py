"""RunPod host inventory and A100 topology validation."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .io import sha256_json, write_json_atomic


CommandRunner = Callable[[Sequence[str]], str]
PERSISTENT_WORKSPACE_STORAGE = "persistent-workspace"
EPHEMERAL_MODEL_CACHE_STORAGE = "ephemeral-model-cache"
STORAGE_MODES = (PERSISTENT_WORKSPACE_STORAGE, EPHEMERAL_MODEL_CACHE_STORAGE)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _run(command: Sequence[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_gpu_query(raw: str) -> list[dict[str, Any]]:
    fields = (
        "index",
        "uuid",
        "name",
        "memory_total_mib",
        "power_limit_w",
        "max_sm_clock_mhz",
        "driver_version",
    )
    gpus: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise ValueError(f"unexpected nvidia-smi query row: {line!r}")
        item = dict(zip(fields, values, strict=True))
        item["index"] = int(item["index"])
        item["memory_total_mib"] = float(item["memory_total_mib"])
        item["power_limit_w"] = float(item["power_limit_w"])
        item["max_sm_clock_mhz"] = float(item["max_sm_clock_mhz"])
        gpus.append(item)
    return gpus


def _parse_topology(raw: str, gpu_count: int) -> dict[str, str]:
    links: dict[str, str] = {}
    # Recent nvidia-smi versions underline the header when stdout is attached to
    # a PTY. Strip presentation escapes before matching stable GPU labels.
    normalized = _ANSI_ESCAPE.sub("", raw)
    rows = [line.split() for line in normalized.splitlines() if line.strip()]
    header = next((row for row in rows if row and row[0] == "GPU0"), None)
    if header is None:
        return links
    gpu_columns = [index for index, value in enumerate(header) if re.fullmatch(r"GPU\d+", value)]
    for row in rows:
        if not row or not re.fullmatch(r"GPU\d+", row[0]):
            continue
        source = int(row[0][3:])
        for target, column in enumerate(gpu_columns[:gpu_count]):
            value_index = column + 1
            if value_index < len(row):
                links[f"{source}:{target}"] = row[value_index]
    return links


def collect_inventory(
    *,
    runner: CommandRunner = _run,
    storage_mode: str = PERSISTENT_WORKSPACE_STORAGE,
) -> dict[str, Any]:
    if storage_mode not in STORAGE_MODES:
        raise ValueError(f"unsupported storage mode: {storage_mode}")
    query = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,power.limit,clocks.max.sm,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    topology = runner(["nvidia-smi", "topo", "-m"])
    full_query = runner(["nvidia-smi", "-q"])
    gpus = _parse_gpu_query(query)
    software = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "vllm": _package_version("vllm"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "nvidia_ml_py": _package_version("nvidia-ml-py"),
    }
    runpod = {
        key: os.environ.get(key)
        for key in (
            "RUNPOD_POD_ID",
            "RUNPOD_DC_ID",
            "RUNPOD_GPU_COUNT",
            "RUNPOD_GPU_NAME",
            "RUNPOD_PUBLIC_IP",
            "RUNPOD_VOLUME_PATH",
            "RUNPOD_IMAGE_NAME",
        )
        if os.environ.get(key) is not None
    }
    workspace = Path("/workspace")
    workspace_usage = shutil.disk_usage(workspace) if workspace.exists() else None
    root_usage = shutil.disk_usage("/")
    memory: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, raw_value = line.split(":", maxsplit=1)
            value = raw_value.strip().split()[0]
            memory[f"{name.lower()}_bytes"] = int(value) * 1024
    inventory = {
        "schema_version": "runpod-inventory-v1",
        "storage_mode": storage_mode,
        "gpus": gpus,
        "topology_raw": topology,
        "topology_links": _parse_topology(topology, len(gpus)),
        "nvidia_smi_query_raw": full_query,
        "software": software,
        "runpod": runpod,
        "storage": {
            "workspace_exists": workspace.exists(),
            "workspace_total_bytes": workspace_usage.total if workspace_usage else None,
            "workspace_free_bytes": workspace_usage.free if workspace_usage else None,
            "root_total_bytes": root_usage.total,
            "root_free_bytes": root_usage.free,
        },
        "memory": memory,
    }
    inventory["inventory_hash"] = sha256_json(inventory)
    return inventory


def validate_a100_sxm_inventory(inventory: dict[str, Any], *, expected_gpus: int = 8) -> list[str]:
    errors: list[str] = []
    gpus = inventory.get("gpus", [])
    if len(gpus) != expected_gpus:
        errors.append(f"expected {expected_gpus} GPUs, found {len(gpus)}")
        return errors
    for gpu in gpus:
        name = str(gpu.get("name", ""))
        if "A100" not in name or "SXM" not in name or "80GB" not in name.replace(" ", ""):
            errors.append(f"GPU {gpu.get('index')} is not A100 SXM 80GB: {name!r}")
        if float(gpu.get("memory_total_mib", 0)) < 79_000:
            errors.append(f"GPU {gpu.get('index')} exposes less than 79,000 MiB")
    power_limits = {float(gpu.get("power_limit_w", 0)) for gpu in gpus}
    sm_clocks = {float(gpu.get("max_sm_clock_mhz", 0)) for gpu in gpus}
    if len(power_limits) != 1:
        errors.append(f"GPU power limits differ: {sorted(power_limits)}")
    if len(sm_clocks) != 1:
        errors.append(f"GPU maximum SM clocks differ: {sorted(sm_clocks)}")
    links = inventory.get("topology_links", {})
    for source in range(expected_gpus):
        for target in range(expected_gpus):
            if source == target:
                continue
            link = str(links.get(f"{source}:{target}", ""))
            if not link.startswith("NV"):
                errors.append(f"GPU{source}<->GPU{target} is not NVLink-connected: {link or 'missing'}")
    query = str(inventory.get("nvidia_smi_query_raw", ""))
    mig_matches = re.findall(r"MIG Mode\s*\n\s*Current\s*:\s*(\w+)", query)
    if len(mig_matches) < expected_gpus:
        errors.append(f"could not confirm MIG mode for every GPU (found {len(mig_matches)} records)")
    elif any(value.lower() != "disabled" for value in mig_matches):
        errors.append(f"MIG must be disabled, observed: {mig_matches}")
    storage = inventory.get("storage", {})
    storage_mode = inventory.get("storage_mode", PERSISTENT_WORKSPACE_STORAGE)
    if not storage.get("workspace_exists"):
        errors.append("/workspace artifact volume is not mounted")
    elif storage_mode == PERSISTENT_WORKSPACE_STORAGE:
        if int(storage.get("workspace_total_bytes") or 0) < 400_000_000_000:
            errors.append("/workspace is smaller than the expected 500 GB network volume")
    elif storage_mode == EPHEMERAL_MODEL_CACHE_STORAGE:
        if int(storage.get("workspace_free_bytes") or 0) < 10_000_000_000:
            errors.append("/workspace has less than 10 GB free for source and result artifacts")
        if int(storage.get("root_free_bytes") or 0) < 400_000_000_000:
            errors.append("root overlay has less than 400 GB free for ephemeral model caches")
    else:
        errors.append(f"unsupported storage mode: {storage_mode}")
    return errors


def validate_inventory_hash(inventory: dict[str, Any]) -> bool:
    expected = inventory.get("inventory_hash")
    payload = dict(inventory)
    payload.pop("inventory_hash", None)
    payload.pop("validation", None)
    return expected == sha256_json(payload)


def write_inventory(
    path: Path,
    *,
    validate: bool = True,
    storage_mode: str = PERSISTENT_WORKSPACE_STORAGE,
) -> dict[str, Any]:
    inventory = collect_inventory(storage_mode=storage_mode)
    errors = validate_a100_sxm_inventory(inventory) if validate else []
    inventory["validation"] = {"status": "pass" if not errors else "fail", "errors": errors}
    write_json_atomic(path, inventory)
    if errors:
        raise RuntimeError("RunPod inventory validation failed: " + "; ".join(errors))
    return inventory
