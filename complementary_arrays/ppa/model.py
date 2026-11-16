"""Area and energy constants of the chip blocks and the array area model, at 1 GHz.

Logic areas are cell areas after synthesis in ASAP7 (RVT, TT, 0.7 V), not after routing. Cell energies come from
gate-level simulation of the synthesized netlists with real weights and activations; SRAM and HBM energies per byte
are taken from the literature (low / mid / high bounds). The clock tree is not included (see CLOCK_TREE_FACTOR).
"""

DFF = 0.2916  # um^2 per flip-flop

# Dot-product cell of Q multipliers (BF16 x BF16 products into an FP32 sum), um^2 at 1 GHz.
DPU_AREA = {
    1: 226.748159,
    16: 1367.92474,
    24: 1959.90189,
    32: 2545.2743,
    48: 4166.628587,
    64: 5511.356543,
    128: 11405.029847,
    256: 22785.754835,
}

# Busy cell power in mW at 1 GHz (= pJ per cell-cycle), every cycle valid. 'bf16': BF16 weights; 'w4g32p': decoded
# NVFP4 weights (at most 6 significant bits). FP8-decoded weights were not simulated and use the BF16 numbers.
DPU_BUSY_MW = {
    "bf16": {1: 0.7883, 16: 6.4320, 32: 12.44, 48: 20.48, 64: 27.42, 128: 51.9115},
    "w4g32p": {32: 11.30, 48: 18.70, 64: 25.43},
}
FMT_POWER = {"w4g32p": "w4g32p", "fp8b128": "bf16"}
# A row without a token computes on zeros (one valid bit per array): mW with NVFP4 weights.
DPU_ZEROACT_MW = {32: 5.68, 48: 10.52, 64: 13.94}
# Clock-gated idle cell (no valid input), uW, leakage included.
DPU_IDLE_UW = {32: 19.6 + 2.1, 48: 30.3 + 3.4, 64: 36.0 + 4.5}


def _near(table, q):
    return min(table, key=lambda k: abs(k - q))


def dpu_busy_pj(q, fmt="w4g32p"):
    """pJ per cell-cycle at full activity and the source ('meas' or how it was estimated). Unmeasured widths take the
    per-multiplier energy of the nearest measured width; NVFP4 at an unmeasured width scales BF16 by the NVFP4 / BF16
    ratio of the nearest measured width."""
    pf = FMT_POWER.get(fmt, fmt)
    b = DPU_BUSY_MW["bf16"]
    if pf == "w4g32p":
        m = DPU_BUSY_MW["w4g32p"]
        if q in m:
            return m[q], "meas"
        k = _near(m, q)
        return dpu_busy_pj(q, "bf16")[0] * m[k] / b[k], f"est (BF16 x NVFP4/BF16 ratio of q{k})"
    if q in b:
        return b[q], "meas"
    if q == 24:
        return 24 * (b[16] / 16 + b[32] / 32) / 2, "est (mean pJ/MAC of q16, q32)"
    k = _near(b, q)
    return q * b[k] / k, f"est (pJ/MAC of q{k})"


def zeroact_factor(q):
    k = _near(DPU_ZEROACT_MW, q)
    return DPU_ZEROACT_MW[k] / DPU_BUSY_MW["w4g32p"][k], "meas" if q == k else f"est (q{k})"


def dpu_idle_pj(q):
    k = _near(DPU_IDLE_UW, q)
    return q * DPU_IDLE_UW[k] / k * 1e-3, "meas" if q == k else f"est (per-multiplier idle of q{k})"


# Operand-movement area per cell of the segmented systolic array (16-row segments), from synthesized 2 x 32 x Q slices
# (Q = 32, 48, 64; the four tree nodes of a slice are removed here because seg_tree_flops adds the tree for every n) and
# an unsegmented 16-wide slice.
E_MEAS = {
    16: 227.0,
    32: 375.0 - 2 * 2 * 16 * 32 / 64 * DFF,
    48: 564.0 - 2 * 2 * 16 * 48 / 64 * DFF,
    64: 745.0 - 2 * 2 * 16 * 64 / 64 * DFF,
}
NORM3 = 111.2  # three-stage FP32 normalizer per column


def e_cell(q):
    if q in E_MEAS:
        return E_MEAS[q], "meas"
    c = sum(E_MEAS[k] - 32 * k * DFF for k in E_MEAS) / len(E_MEAS)  # two 16q-bit operand registers + mean constant
    return 32 * q * DFF + c, "est (32q flops + mean constant of q16/32/64)"


def seg_tree_flops(m, n, q, s=16):
    """Flops of the activation distribution tree of segmented delivery: per row, one 16q-bit register per node of a
    tree over the ceil(n / s) segments; a node with one child keeps its register so that latencies stay equal."""
    if n <= s:
        return 0
    k = -(-n // s)
    d = (k - 1).bit_length()
    nodes = sum(-(-k // (1 << level)) for level in range(d))
    return m * nodes * 16 * q


def array_area(m, n, k):
    """um^2 of one m x n array of dot-product cells of k / m multipliers: cells, operand movement, normalizers,
    activation skew and drain registers, distribution tree."""
    q = k // m
    edge = DFF * (m * (m - 1) / 2 * q * 16 + n * (59 + 32))
    tree = DFF * seg_tree_flops(m, n, q)
    return m * n * (DPU_AREA[q] + e_cell(q)[0]) + n * NORM3 + edge + tree


# ---- other blocks (um^2) ----
VEC_LANE = 690.042234  # FP32 multiply-add lane of the vector unit
VEC_LANES = 2048
DEC = {  # dequantizer per 64-lane instance
    "w4g32p": (825.4, "NVFP4 (E2M1 x E4M3 -> BF16, exact)"),
    "fp8b128": (4681.0, "FP8 E4M3 with FP32 block scale, correctly rounded"),
}
DEC_PJ_PER_WEIGHT = {"w4g32p": 0.061, "fp8b128": 0.246}  # gate-level, real weight streams
# Weight data plane: arrival buffers, write queues, reorder buffers per stream, release crossbars (256 or 128 beats per
# cycle), control.
DP = dict(
    arrival_per_core=140845 / 2,
    write_queues=475061,
    rob_per_stream=661697 / 4,
    rel256=207232,
    rel128=205860 / 2 * 207232 / (381620 / 2),
    control=33330,
    wrel=1146,
)
RE_ROB_2048_EXTRA = 165424 + 27898  # 2,048-slot re-delivery reorder buffer and its 16:1 crossbar, per array


def dataplane_area(arrays):
    """Weight data plane for a list of (m, n, k) arrays: two streams (first pass, re-delivery) per array; release width
    256 beats when the array takes more than 8 KB of BF16 weights per cycle, else 128."""
    a = DP["write_queues"] + DP["control"] + DP["wrel"]
    for m, n, k in arrays:
        q = k // m
        a += (
            DP["arrival_per_core"] + 2 * DP["rob_per_stream"] + 2 * (DP["rel256"] if n * q * 2 > 8192 else DP["rel128"])
        )
        a += RE_ROB_2048_EXTRA
    return a


WPORT = 66826.4  # weight-port scheduler, 512 groups
CTRL = 24369.7  # engine control plane
DISP = {"pincer": 14407.6 + 4788.2, "winscan": 18752.5 + 15381.3}  # dispatcher front end + core
SRAM = {"low": 4.38e6, "mid": 5.42e6, "high": 5.93e6}  # 16 MiB; high = CACTI including global wires
PORT_REP = 0.08e6  # weight-port repeaters along the array edge
SRAM_MM2_PER_MIB = {k: v / 16 for k, v in {"low": 4.38, "mid": 5.42, "high": 5.93}.items()}


def stream_buffer_bytes(m, n, k, slots):
    """Per-array streaming buffer: max(2, ceil(slots / m) + 1) K-blocks of n x k BF16 weights."""
    return max(2, -(-slots // m) + 1) * n * k * 2


def dec_area(fmt):
    return DEC[fmt][0]


def policy_dispatcher(policy):
    return "winscan" if policy == 2 else "pincer"


# ---- energy constants ----
E_SRAM = {"low": 0.20, "mid": 0.42, "high": 1.06}  # pJ/B, bank access
E_WIRE = {"low": 0.38, "mid": 0.48, "high": 0.58}  # pJ/B, bank to array, on the bytes the weight port delivers
E_BUF_WIRE = {"low": 0.06, "mid": 0.08, "high": 0.10}  # pJ/B, streaming buffer to array
SRAM_LEAK_W = 8.7e-3  # 16 MiB, CACTI 22 nm low-standby-power cells
E_HBM_BIT = {"low": 1.95, "mid": 3.97, "high": 7.0}  # pJ/bit
# Upper bound of the clock tree: the only clock-tree-synthesized cell spends 42.5% of its power in the clock network;
# applied as 0.8 x the dynamic energy of every logic block and reported separately.
CLOCK_TREE_FACTOR = 0.8
DPU_DENSITY = DPU_BUSY_MW["bf16"][32] / DPU_AREA[32]  # mW per um^2 of a busy q32 cell, for unsimulated logic

# Transfer and distribution power of the segmented array periphery, measured on 2 x 32 x Q slices with all rows busy
# (NVFP4 weights), mW per slice.
PERIPH_MW = {32: 124.4, 48: 2.78 * 64, 64: 3.72 * 64}


def periph_density(fmt, q=32):
    """mW per um^2 of array periphery during matrix phases: the measured slice nearest in Q, scaled from NVFP4 to the
    weight format by the cell energy ratio."""
    k = min(PERIPH_MW, key=lambda x: (abs(x - q), x))
    base = PERIPH_MW[k] / (array_area(2, 32, 2 * k) - 64 * DPU_AREA[k])
    return base * dpu_busy_pj(32, fmt)[0] / dpu_busy_pj(32, "w4g32p")[0]
