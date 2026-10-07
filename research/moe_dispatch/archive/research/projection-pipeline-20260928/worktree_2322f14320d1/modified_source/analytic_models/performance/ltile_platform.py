"""Auditable shared system profiles and fail-closed full-layer evidence gates.

Memory capacity follows the actual Ramulator organization. Increasing capacity
requires an explicit memory configuration, not a larger accounting constant.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

from .ltile_cost import Machine
from .matrix_service import MatrixService
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


@dataclass(frozen=True)
class CodecResources:
    """Finite NVFP4 dequantization candidate, separate from weight storage.

    Scale multiplication precedes value multiplication/conversion. No overlap
    with Matrix execution is assumed. These are budgeted services, not PPA.
    """

    lanes: int = 256
    ii: int = 1
    latency: int = 6
    scale_lanes: int = 16
    scale_latency: int = 2
    input_bytes: int = 8192
    output_bytes: int = 16384

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("positive finite codec resources required")


@dataclass(frozen=True)
class ExecutionProfile:
    """One execution contract shared by the layer runner and decode model."""

    machine: Machine = field(
        default_factory=lambda: Machine(
            sfu_lanes=32,
            vector_exp_cycles=8,
            vector_softplus_cycles=16,
            vector_reciprocal_cycles=8,
            reduction_tree_bf16=True,
        )
    )
    matrix: MatrixService = field(default_factory=MatrixService)
    dma: DmaResources = field(default_factory=DmaResources)
    codec: CodecResources = field(default_factory=CodecResources)
    state_rounding: str = "rn"
    delta: str = "bf16_rational_from_log"
    scheduling: str = "serial_retirement"
    projection_schedule: str = "resident"
    projection_n_panel_tile: int = 1
    # Named projection stages may use different software loop nests. These
    # choices affect the program/cache identity, never physical resources.
    projection_panel_overrides: tuple = ()
    projection_template_overrides: tuple = ()
    projection_vector_rows: int = 58
    projection_request_tile: int = 16
    projection_k_tile: int = 256
    projection_codec_rows: int = 6
    gather_vector_rows: int = 64
    hbm_controllers: int = 8

    def __post_init__(self):
        if self.hbm_controllers not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("power-of-two HBM controller count up to 64 required")
        if self.state_rounding not in ("rn", "sr"):
            raise ValueError("unknown state rounding")
        if self.delta != "bf16_rational_from_log" or self.scheduling != "serial_retirement":
            raise ValueError("unvalidated arithmetic or scheduling contract")
        if self.projection_schedule not in ("resident", "compact", "batch", "transposed", "matrix"):
            raise ValueError("unknown projection schedule")
        if type(self.projection_n_panel_tile) is not int or self.projection_n_panel_tile not in (1, 2, 4, 8):
            raise ValueError("projection N panel tile must be 1, 2, 4 or 8")
        if self.projection_schedule == "resident" and self.projection_n_panel_tile != 1:
            raise ValueError("N panel tiling requires the compact Matrix interface")
        templates = tuple(sorted(tuple(x) for x in self.projection_template_overrides))
        if any(len(x) != 3 or not isinstance(x[0], str) or x[1] not in ("resident", "transposed", "matrix") or x[2] not in (256,512,1024) or (x[1] == "resident" and x[2] != 256) for x in templates):
            raise ValueError("invalid per-projection original-instruction template")
        if len({x[0] for x in templates}) != len(templates):
            raise ValueError("duplicate projection template")
        if templates and self.projection_schedule not in ("resident", "transposed", "matrix"):
            raise ValueError("original-instruction templates cannot mix with projection extensions")
        if (templates or self.projection_schedule == "matrix") and (self.matrix.weight_replay or self.matrix.projection_segments != 1 or self.matrix.edge != 4):
            raise ValueError("bounded original matrix templates require the common four-row array")
        object.__setattr__(self, "projection_template_overrides", templates)
        overrides = self.projection_panel_overrides
        if not isinstance(overrides, (tuple, list)) or any(
            not isinstance(item, (tuple, list)) or len(item) != 2
            or not isinstance(item[0], str) or not item[0]
            or type(item[1]) is not int or item[1] not in (1, 2, 4, 8)
            for item in overrides
        ):
            raise ValueError("projection overrides require named stages and legal panel tiles")
        if len({item[0] for item in overrides}) != len(overrides):
            raise ValueError("duplicate projection stage override")
        if overrides and self.projection_schedule == "resident":
            raise ValueError("projection overrides require the compact Matrix interface")
        object.__setattr__(self, "projection_panel_overrides", tuple(sorted(tuple(item) for item in overrides)))
        if type(self.projection_codec_rows) is not int or self.projection_codec_rows not in (0, 6):
            raise ValueError("codec reservation must be zero or six Vector rows")
        if type(self.projection_request_tile) is not int or self.projection_request_tile not in (1, 2, 4, 8, 16):
            raise ValueError("projection request tile must be 1/2/4/8/16")
        if self.projection_schedule != "resident" and self.projection_request_tile != 16:
            raise ValueError("request grouping is only exposed for the resident software path")
        if type(self.projection_k_tile) is not int or self.projection_k_tile not in (256, 512, 1024) or (
            self.projection_schedule not in ("transposed", "matrix") and self.projection_k_tile != 256
        ):
            raise ValueError("larger K grouping requires the transposed software path")
        if self.projection_schedule in ("transposed", "matrix") and (
            self.matrix.weight_replay or self.matrix.projection_segments != 1
            or self.projection_n_panel_tile != 1 or overrides
        ):
            raise ValueError("transposed software path uses no projection extensions")
        if not 1 <= self.projection_vector_rows <= 64 - self.projection_codec_rows or not 1 <= self.gather_vector_rows <= 64:
            raise ValueError("invalid projection/gather Vector workspace")
        if self.projection_schedule != "resident" and self.matrix.accumulator != "BF16":
            raise ValueError("this projection schedule requires BF16 partial sums")
        if self.matrix.matrix_capacity_bytes != 1024**2 or self.matrix.vector_capacity_bytes != 256 * 1024:
            raise ValueError("connected ISA geometry requires 1 MiB Matrix / 256 KiB Vector")

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def runtime_environment(self):
        m = self.machine
        return dict(
            PLENA_VECTOR_REC_ALU=m.vector_rec_alu,
            PLENA_VECTOR_REC_ALU_LATENCY=str(m.vector_rec_alu_latency),
            PLENA_HBM_CONTROLLERS=str(self.hbm_controllers),
            PLENA_UNIFIED_SERIAL_TIMING="1",
            PLENA_EXACT_VIEW_DMA="1",
            PLENA_DMA_READ_WINDOW=str(self.dma.read_credits),
            PLENA_DMA_WRITE_WINDOW=str(self.dma.write_credits),
            PLENA_VECTOR_SFU_LANES=str(m.sfu_lanes or m.vector_width),
            PLENA_VECTOR_SFU_II=str(m.sfu_ii),
            PLENA_VECTOR_SFU_EXP_LATENCY=str(m.vector_exp_cycles),
            PLENA_VECTOR_SFU_SOFTPLUS_LATENCY=str(m.vector_softplus_cycles),
            PLENA_VECTOR_SFU_RECI_LATENCY=str(m.vector_reciprocal_cycles),
            PLENA_VECTOR_REDUCE_BF16_TREE="1" if m.reduction_tree_bf16 else "0",
            PLENA_V2_STREAM_ENGINE="1",
            PLENA_V2_UPDATE_LANES=str(m.lanes),
            PLENA_V2_UPDATE_INTERVAL=str(m.update_ii),
            PLENA_V2_UPDATE_LATENCY=str(m.update_latency),
            PLENA_V2_DOT=m.dot,
            PLENA_V2_DOT_LATENCY=str(m.dot_latency),
            PLENA_V2_DOT_II=str(m.dot_ii),
            PLENA_V2_SRAM_PORT=str(m.sram_cycles),
            PLENA_V2_CONTEXT_PORT=str(m.context_cycles),
            PLENA_V2_STATE_ROUNDING=self.state_rounding,
            PLENA_NATIVE_READ_WIDTH=str(m.native_read_width),
            PLENA_NATIVE_RESULT_SLOTS=str(m.native_result_slots),
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
