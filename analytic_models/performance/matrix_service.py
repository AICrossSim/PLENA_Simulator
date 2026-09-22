"""Bounded split-K mini-array reference contract for Matrix calibration.

Topology follows PLENA's parallel square mini-arrays plus a reduction tree.
This is NOT the old MatrixCoreProfile rows-by-cols output-stationary shortcut.
The reference deliberately serializes K tiles and drains final writeback.
It does not price HBM, MX/NVFP codecs, or claim integrated RTL acceptance.
"""

from dataclasses import asdict, dataclass
from math import ceil, log2


@dataclass(frozen=True)
class MatrixService:
    edge: int = 4
    reduction_lanes: int = 1024
    mac_latency: int = 2
    mac_ii: int = 1
    tree_add_latency: int = 2
    matrix_read_elements: int = 2048
    vector_read_elements: int = 2048
    vector_write_elements: int = 2048
    matrix_capacity_bytes: int = 1024**2
    vector_capacity_bytes: int = 256 * 1024
    accumulator: str = "BF16"

    def __post_init__(self):
        integers = {k: v for k, v in asdict(self).items() if k != "accumulator"}
        if any(type(v) is not int or v < 1 for v in integers.values()):
            raise ValueError("Matrix resource counts must be positive integers")
        if self.edge > 16 or self.reduction_lanes % self.edge:
            raise ValueError("reduction lanes must comprise whole square mini-arrays")
        if self.groups & (self.groups - 1):
            raise ValueError("cross-array reduction must be a power-of-two tree")
        if self.accumulator not in ("BF16", "FP32"):
            raise ValueError("explicit BF16 or FP32 accumulator required")
        if self.operand_bytes > self.matrix_capacity_bytes:
            raise ValueError("weight tile exceeds Matrix SRAM")
        if self.operand_bytes + self.edge**2 * 2 > self.vector_capacity_bytes:
            raise ValueError("input tile and output tile exceed Vector SRAM")

    @property
    def groups(self):
        return self.reduction_lanes // self.edge

    @property
    def operand_bytes(self):
        return self.edge * self.reduction_lanes * 2

    def resources(self):
        scalar_bytes = 2 if self.accumulator == "BF16" else 4
        partials = self.groups * self.edge**2
        return dict(
            multipliers=self.edge * self.reduction_lanes,
            pe_accumulator_bytes=partials * scalar_bytes,
            cross_array_adders=(self.groups - 1) * self.edge**2,
            # All tree stages registered; no free reduction network.
            tree_register_bytes=(self.groups - 1) * self.edge**2 * scalar_bytes,
            tile_accumulator_bytes=self.edge**2 * scalar_bytes,
            matrix_operand_bytes=self.operand_bytes,
            vector_operand_bytes=self.operand_bytes,
            input_latch_bytes=2 * self.operand_bytes,
            output_hold_bytes=self.edge**2 * 2,
        )


def matrix_cost(m, n, k, hardware=MatrixService(), *, write_stall=0):
    """Tiled [m,k] x [k,n], padding every edge/K tail with explicit zeros.

    Independent Matrix/Vector ports read concurrently, then compute, then
    writeback. One live output tile; no double buffering or inter-tile overlap.
    A PE has one dependent accumulator, hence launch spacing >= feedback delay.
    SRAM counters count occupied cycles, not an additive sum of concurrent ports.
    """
    if any(type(v) is not int or v < 1 for v in (m, n, k)) or write_stall < 0:
        raise ValueError("positive M/N/K and nonnegative write stall required")
    h = hardware
    tiles = ceil(m / h.edge) * ceil(n / h.edge)
    chunks = ceil(k / h.reduction_lanes)
    values = h.edge * h.reduction_lanes
    mr = ceil(values / h.matrix_read_elements)
    vr = ceil(values / h.vector_read_elements)
    # Last padded PE, last local-K operation, then registered tree and final
    # cross-K accumulator. MAC II alone cannot hide a dependent feedback delay.
    array = 2 * (h.edge - 1) + (h.edge - 1) * max(h.mac_ii, h.mac_latency) + h.mac_latency
    tree = int(log2(h.groups)) * h.tree_add_latency
    write = ceil(h.edge**2 / h.vector_write_elements)
    issue = tiles * (chunks + 1)  # compute chunks plus explicit writeout
    sram = tiles * (chunks * max(mr, vr) + write)
    arithmetic = tiles * chunks * (array + tree + h.tree_add_latency)
    dependency = tiles * write_stall
    return dict(
        issue=issue,
        sram=sram,
        arithmetic=arithmetic,
        dependency=dependency,
        total=issue + sram + arithmetic + dependency,
        matrix_read_elements=tiles * chunks * values,
        vector_read_elements=tiles * chunks * values,
        vector_write_elements=tiles * h.edge**2,
        logical_macs=m * n * k,
        issued_macs=tiles * chunks * h.edge**2 * h.reduction_lanes,
        tiles=tiles,
        reduction_chunks=chunks,
        resources=h.resources(),
        status="bounded_matrix_reference_candidate",
        excluded=[
            "HBM/DMA",
            "weight unpack/scale conversion",
            "physical bank placement",
            "Compiler/ISA integration",
            "integrated RTL timing",
        ],
    )
