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
    # Opt-in local payload retention; the historical M_MV reread schedule is
    # unchanged by default. This is a candidate finite buffer, not free reuse.
    weight_replay: bool = False

    def __post_init__(self):
        integers = {k: v for k, v in asdict(self).items() if k not in ("accumulator", "weight_replay")}
        if any(type(v) is not int or v < 1 for v in integers.values()):
            raise ValueError("Matrix resource counts must be positive integers")
        if self.edge > 16 or self.reduction_lanes % self.edge:
            raise ValueError("reduction lanes must comprise whole square mini-arrays")
        if self.groups & (self.groups - 1):
            raise ValueError("cross-array reduction must be a power-of-two tree")
        if self.accumulator not in ("BF16", "FP32"):
            raise ValueError("explicit BF16 or FP32 accumulator required")
        if type(self.weight_replay) is not bool:
            raise ValueError("weight_replay must be an explicit boolean")
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
            projection_replay=self.projection_resources() if self.weight_replay else None,
        )

    def arithmetic_cycles(self):
        return (
            2 * (self.edge - 1)
            + (self.edge - 1) * max(self.mac_ii, self.mac_latency)
            + self.mac_latency
            + int(log2(self.groups)) * self.tree_add_latency
            + self.tree_add_latency
        )

    @staticmethod
    def projection_resources():
        # M_MM.P requires these finite resources even when legacy M_MV replay
        # is disabled. SRAM capacity is unchanged; these are local latches.
        return dict(
            weight_replay_bytes=256 * 32 * 2,
            vector_row_transfer_bytes=2048 * 2,
            compact_input_bytes=4 * 256 * 2,
            output_hold_bytes=4 * 32 * 2,
            payload_bytes=256 * 32 * 2 + 2048 * 2 + 4 * 256 * 2 + 4 * 32 * 2,
            metadata_and_selection="additional; not included in payload bytes",
        )

    def replay_cost(self, k, batch, matrix_cycles, bank_words, *, partial_writeback):
        """One bounded K-by-32 panel, including finite latch feed bandwidth.

        Matrix preload, input-row reads, subgroup feed/compute and output
        read-modify-write all serialize. It does not assume implicit overlap,
        masked SRAM ports, or multiply a B1 time by a fitted factor.
        """
        if not (1 <= k <= 256 and 1 <= batch <= 4 and batch <= self.edge):
            raise ValueError("projection replay needs K<=256 and 1..4 active rows")
        if 32 % self.edge or k > self.reduction_lanes:
            raise ValueError("projection replay exceeds the Matrix geometry")
        if self.accumulator != "BF16":
            raise ValueError("projection replay requires the BF16 accumulation contract")
        groups = 32 // self.edge
        preload = max(matrix_cycles, ceil(bank_words * 32 / self.matrix_read_elements))
        inputs = batch * ceil(2048 / self.vector_read_elements)
        feed = groups * (
            ceil(k * self.edge / self.matrix_read_elements)
            + ceil(batch * k / self.vector_read_elements)
        )
        write = batch * (
            ceil(2048 / self.vector_read_elements) + ceil(2048 / self.vector_write_elements)
        ) if partial_writeback else 0
        return dict(
            sram=preload + inputs + feed + write,
            # The one full Vector-row latch serializes each request's final
            # BF16 partial merge. The array computes all active rows together.
            arithmetic=groups * (
                self.arithmetic_cycles() - self.tree_add_latency
                + batch * self.tree_add_latency
            ) if partial_writeback else groups * self.arithmetic_cycles(),
            matrix_bank_words=bank_words,
            vector_read_rows=batch * (2 if partial_writeback else 1),
            vector_write_rows=batch if partial_writeback else 0,
            projection_latch_service_cycles=feed,
            projection_weight_latch_bytes_loaded=k * 32 * 2,
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
    if h.weight_replay:
        raise ValueError("generic Matrix reference does not model projection replay; use compiled M_MV/M_MM.P")
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
