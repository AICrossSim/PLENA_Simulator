"""Auditable shared system profiles and fail-closed full-layer evidence gates.

Memory capacity follows the actual Ramulator organization. Increasing capacity
requires an explicit memory configuration, not a larger accounting constant.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class DmaResources:
    read_credits: int = 32
    write_credits: int = 32
    line_bytes: int = 64
    read_response_bytes: int = 2048
    write_staging_bytes: int = 2048
    tag_bits: int = 64
    policy: str = "bounded_batches"

    def __post_init__(self):
        if self.policy != "bounded_batches" or self.line_bytes != 64:
            raise ValueError("unsupported DMA service")
        if not 1 <= self.read_credits <= 64 or not 1 <= self.write_credits <= 64:
            raise ValueError("DMA credits must be bounded by 1..64")
        if self.read_response_bytes < self.read_credits * self.line_bytes:
            raise ValueError("read credits exceed response storage")
        if self.write_staging_bytes < self.write_credits * self.line_bytes:
            raise ValueError("write credits exceed staging storage")
        if self.tag_bits < 32:
            raise ValueError("insufficient destination tag width")

    @property
    def service(self):
        return f"bounded:{self.read_credits}:{self.write_credits}"

    @property
    def buffer_and_tag_bytes(self):
        return (
            self.read_response_bytes
            + self.write_staging_bytes
            + math.ceil((self.read_credits + self.write_credits) * self.tag_bits / 8)
        )


def memory_geometry(config):
    memory = config["memory_system"]
    if memory["channel_mapper"]["impl"] != "CacheLineInterleave":
        raise ValueError("unsupported channel mapping")
    capacity = bandwidth = 0
    for controller in memory["controllers"]:
        dram = controller["dram"]
        counts = dram["org"]["count"]
        if len(counts) != 6 or min(counts) < 1 or dram["org"]["dq"] != dram["channel_width"]:
            raise ValueError("unsupported DRAM organization")
        capacity += math.prod(counts) * dram["org"]["dq"] // 8
        bandwidth += dram["timing"][0] * 10**6 * dram["channel_width"] // 8
    return dict(
        controllers=len(memory["controllers"]),
        capacity_bytes=capacity,
        peak_bytes_per_second=bandwidth,
        physical_stack_count=None,
        topology="explicit independent interleaved controllers; stack packaging unspecified",
    )


def shared_profile(config, *, name, dma=DmaResources(), clock_hz=10**9):
    memory = memory_geometry(config)
    if clock_hz != 10**9:
        raise ValueError("captured memory timing has only been checked at 1 GHz")
    profile = dict(
        name=name,
        clock_hz=clock_hz,
        clock_status="simulation target, not closed RTL timing",
        memory=memory,
        memory_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        dma=asdict(dma),
        dma_buffer_and_tag_bytes=dma.buffer_and_tag_bytes,
        matrix_sram_bytes=1024**2,
        vector_sram_bytes=256 * 1024,
        matrix_banks=64,
        bank_bf16_words=32,
        bank_ports=dict(read=1, write=1),
        state="BF16",
        activation="BF16",
        kv="BF16",
        weight="per-model explicit NVFP4 policy, scales/padding included",
        update=dict(lanes=256, ii=2, latency=6, intermediate="FP32", store="BF16 RN"),
        dot="BF16 tree",
        dma_integration_status="finite-credit execution model; integrated RTL pending",
    )
    profile["sha256"] = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    return profile


REQUIRED = ("producer", "projection", "attention_mla", "moe", "norm_conv_gate", "output_head", "connected_layer")


def formal_blockers(evidence, *, profile_sha256, required_bytes, capacity_bytes):
    blockers = []
    if required_bytes > capacity_bytes:
        blockers.append("full model allocations exceed configured HBM capacity")
    for stage in REQUIRED:
        record = evidence.get(stage, {})
        if not record.get("passed"):
            blockers.append(f"{stage}: validation missing or failed")
            continue
        if record.get("profile_sha256") != profile_sha256:
            blockers.append(f"{stage}: validation belongs to another profile")
        if not record.get("held_out_cases", 0):
            blockers.append(f"{stage}: no held-out validation")
        if not record.get("component_comparison_passed"):
            blockers.append(f"{stage}: component errors unchecked")
        if not record.get("numerical_passed"):
            blockers.append(f"{stage}: numerical path unchecked")
        files = record.get("sources", [])
        if not files:
            blockers.append(f"{stage}: no pinned source evidence")
        for item in files:
            path = Path(item["path"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
                blockers.append(f"{stage}: missing or changed source {path}")
    return blockers
