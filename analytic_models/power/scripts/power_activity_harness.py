#!/usr/bin/env python3
"""Reusable cocotb stimulus for RTL-activity power calibration.

The file is copied into a disposable PLENA_RTL worker.  Its Verilator binary
is built once per mapped configuration and then reused for all scenarios.
Only the measurement interval described by the action sidecar is consumed by
Design Compiler; reset and register initialization remain in the VCD for
debugging but are outside that interval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.utils import get_sim_time
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH


COMPONENT = os.environ["PLENA_POWER_COMPONENT"]
PATTERN = os.environ["PLENA_POWER_PATTERN"]
REPEATS = int(os.environ["PLENA_POWER_REPEATS"])
MICROKERNEL = os.environ.get("PLENA_POWER_MICROKERNEL", "mixed")
SIDECAR = Path(os.environ["PLENA_POWER_SIDECAR"])
PARAMS = json.loads(os.environ.get("PLENA_POWER_PARAMS_JSON", "{}"))
SEED = int(os.environ.get("PLENA_POWER_SEED", "20260722"))
ACTIVITY_FINGERPRINT = os.environ.get("PLENA_POWER_FINGERPRINT", "")
MIX_SEQUENCE = tuple(json.loads(os.environ.get("PLENA_POWER_MIX_SEQUENCE_JSON", "[]")))
MIX_HASH = os.environ.get("PLENA_POWER_MIX_HASH", "")
CLOCK_NS = 1.0


def _features() -> dict[str, float]:
    return {
        "matrix.active_mac_bit_product": 0.0,
        "matrix.matrix_vector_bit_product": 0.0,
        "matrix.output_bit": 0.0,
        "vector.lane_add_sub_bit": 0.0,
        "vector.lane_movement_bit": 0.0,
        "vector.lane_multiply_bit2": 0.0,
        "vector.lane_sfu_bit2": 0.0,
        "vector.reduction_node_bit": 0.0,
        "vector.compact_stats_mul": 0.0,
        "vector.compact_stats_add": 0.0,
        "vector.compact_stats_rsqrt": 0.0,
        "scalar.fp_add_sub_move_bit": 0.0,
        "scalar.fp_multiply_bit2": 0.0,
        "scalar.fp_sfu_bit2": 0.0,
        "scalar.integer_alu_bit": 0.0,
        "scalar.register_access_bit": 0.0,
        "scalar.vector_lane_access_bit": 0.0,
        "control.frontend_issue": 0.0,
        "agu.config": 0.0,
        "agu.loop_setup": 0.0,
        "agu.loop_boundary": 0.0,
        "agu.stream_step": 0.0,
        "agu.offset_read": 0.0,
        "hbm.dma_issue": 0.0,
        "hbm.line": 0.0,
        "hbm.byte": 0.0,
    }


def _clock_features(cycles: int) -> dict[str, float]:
    result = {
        "matrix_pe": 0.0,
        "vector_lane": 0.0,
        "scalar_machine": 0.0,
        "control_frontend": 0.0,
        "hbm_controller": 0.0,
    }
    if COMPONENT == "matrix":
        block = int(PARAMS.get("BLOCK_DIM", 1))
        result["matrix_pe"] = float(cycles * block * block)
    elif COMPONENT == "vector":
        result["vector_lane"] = float(cycles * int(PARAMS["VLEN"]))
    elif COMPONENT == "scalar":
        result["scalar_machine"] = float(cycles)
    elif COMPONENT in {"control", "agu"}:
        result["control_frontend"] = float(cycles)
    else:
        result["hbm_controller"] = float(cycles)
    return result


async def _cycles(dut, count: int) -> None:
    for _ in range(count):
        await RisingEdge(dut.clk)


def _one(exp: int, mant: int) -> int:
    return ((1 << (exp - 1)) - 1) << mant


def _mix64(value: int) -> int:
    """Fold high action bits into the low payload bits deterministically."""

    mask = (1 << 64) - 1
    mixed = (value + 0x9E3779B97F4A7C15) & mask
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
    return mixed ^ (mixed >> 31)


def _finite_fp_value(
    exp: int,
    mant: int,
    index: int,
    pattern: str,
    rng: random.Random,
) -> int:
    """Return a finite nonzero value with pattern-specific switching.

    Random packed bits frequently encode NaN/Inf, while repeating one constant
    makes a mapped FP datapath quiescent after startup.  Constructing the sign,
    exponent, and fraction separately gives sustained but bounded activity.
    """

    bias = (1 << (exp - 1)) - 1
    max_finite_exp = (1 << exp) - 2
    mixed = _mix64(index)
    if pattern == "low-toggle":
        # Keep the Hamming distance small while ensuring that action bits above
        # log2(VLEN) affect every supported FP format.
        folded = index ^ (index >> max(1, mant)) ^ (index >> max(1, exp))
        sign = (folded >> 1) & 1
        exponent = max(1, min(max_finite_exp, bias + (folded & 1)))
        fraction = 1 << (folded % max(mant, 1)) if mant else 0
    elif pattern == "representative-qwen":
        sign = mixed & 1
        exponent = max(1, min(max_finite_exp, bias - 3 + ((mixed >> 1) & 3)))
        fraction = (mixed >> 3) & ((1 << mant) - 1)
    elif pattern == "mixed-kernel-holdout":
        sign = (mixed >> 7) & 1
        exponent = max(1, min(max_finite_exp, bias - 1 + ((mixed >> 8) % 3)))
        fraction = (mixed >> 11) & ((1 << mant) - 1)
    else:
        sign = rng.getrandbits(1)
        # Keep random training operands in the transformer-like finite range.
        # Sampling above the bias made EXP-heavy vector scenarios saturate to
        # a constant output, so their switching could fall below the all-zero
        # idle baseline.  Sign and mantissa remain random, while exponent
        # values stay in [bias-4, bias-1].
        exponent = max(1, min(max_finite_exp, bias - 4 + rng.randrange(4)))
        fraction = rng.getrandbits(mant) if mant else 0
    return (sign << (exp + mant)) | (exponent << mant) | fraction


def _pattern_bits(width: int, index: int, pattern: str, rng: random.Random) -> int:
    """Generate non-FP payloads with distinct train and holdout statistics."""

    mask = (1 << width) - 1
    if pattern == "low-toggle":
        return 1 << (index % max(width, 1))
    if pattern == "representative-qwen":
        # Low-magnitude signed payloads approximate quantized transformer
        # tensors without making the holdout identical to the random train set.
        payload_width = max(2, min(width, 6))
        value = ((index * 5 + 3) & ((1 << payload_width) - 1))
        return value & mask
    if pattern == "mixed-kernel-holdout":
        alternating = int("10" * ((width + 1) // 2), 2) & mask
        return (alternating ^ (index * 0x9E37)) & mask
    return rng.getrandbits(width)


def _packed_lanes(values: list[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (value & mask) << (lane * width)
    return packed


def _packed_pattern_lanes(
    *,
    lane_width: int,
    lane_count: int,
    action: int,
    pattern: str,
    rng: random.Random,
    fp_format: tuple[int, int] | None = None,
    salt: int = 0,
) -> int:
    """Generate one independently varying value per physical input lane."""

    values: list[int] = []
    for lane in range(lane_count):
        index = ((action + 1) << 20) ^ (lane * 0x1F123BB5) ^ salt
        if fp_format is None:
            values.append(_pattern_bits(lane_width, _mix64(index), pattern, rng))
        else:
            exp, mant = fp_format
            values.append(_finite_fp_value(exp, mant, index, pattern, rng))
    return _packed_lanes(values, lane_width)


def _precision_width(value: str) -> int:
    if value.startswith("MXINT"):
        return int(value.removeprefix("MXINT"))
    match = re.fullmatch(r"MXFP_E(\d+)M(\d+)", value)
    if not match:
        raise ValueError(f"unsupported calibration precision: {value}")
    return 1 + int(match.group(1)) + int(match.group(2))


def _next_hbm_element_width(raw_width: int) -> int:
    # Matches configuration.svh: 1 << clog2(HBM_ELE_WIDTH_RAW * 2).
    return 1 << max(0, (raw_width * 2 - 1).bit_length())


async def _reset_common(dut) -> None:
    dut.rst.value = 1
    await _cycles(dut, 5)
    dut.rst.value = 0
    await _cycles(dut, 4)


async def _matrix_activity(dut, active: bool, rng: random.Random, features: dict[str, float]) -> int:
    mode = str(PARAMS.get("mode", "leaf"))
    block = int(PARAMS.get("BLOCK_DIM", 2))
    slot_cycles = 4 * block + 16
    if mode == "mxint":
        for name in ("load_a_row", "load_b_col", "load_a_scale", "load_b_scale", "load_valid"):
            getattr(dut, name).value = 0
    elif mode == "mxfp":
        for name in ("clear_accumulator", "in_top_element", "in_top_scale", "system_top_valid",
                     "in_left_element", "in_left_scale", "system_left_valid"):
            getattr(dut, name).value = 0
    else:
        for name in ("in_valid", "int_data", "fp_data", "acc_in", "scale_in"):
            getattr(dut, name).value = 0
    await _reset_common(dut)

    for action in range(REPEATS):
        if mode == "mxint":
            if active:
                width_a = int(PARAMS["L_BITS"])
                width_b = int(PARAMS["T_BITS"])
                for feed in range(block):
                    stimulus_action = action * block + feed
                    dut.load_a_row.value = _packed_pattern_lanes(
                        lane_width=width_a, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        salt=0xA1,
                    )
                    dut.load_b_col.value = _packed_pattern_lanes(
                        lane_width=width_b, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        salt=0xB2,
                    )
                    scale_a_width = len(dut.load_a_scale) // block
                    scale_b_width = len(dut.load_b_scale) // block
                    dut.load_a_scale.value = _packed_pattern_lanes(
                        lane_width=scale_a_width, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        salt=0xC3,
                    )
                    dut.load_b_scale.value = _packed_pattern_lanes(
                        lane_width=scale_b_width, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        salt=0xD4,
                    )
                    dut.load_valid.value = 1
                    await RisingEdge(dut.clk)
                dut.load_valid.value = 0
                features["matrix.active_mac_bit_product"] += block**3 * width_a * width_b
                features["matrix.output_bit"] += block * block * (width_a + width_b + 4)
            await _cycles(dut, slot_cycles - (block if active else 0))
        elif mode == "mxfp":
            width_a = 1 + int(PARAMS["L_EXP"]) + int(PARAMS["L_MANT"])
            width_b = 1 + int(PARAMS["T_EXP"]) + int(PARAMS["T_MANT"])
            if active:
                dut.clear_accumulator.value = int(action == 0)
                for feed in range(block):
                    stimulus_action = action * block + feed
                    dut.in_top_element.value = _packed_pattern_lanes(
                        lane_width=width_b, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        fp_format=(int(PARAMS["T_EXP"]), int(PARAMS["T_MANT"])),
                        salt=0xE5,
                    )
                    dut.in_left_element.value = _packed_pattern_lanes(
                        lane_width=width_a, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        fp_format=(int(PARAMS["L_EXP"]), int(PARAMS["L_MANT"])),
                        salt=0xF6,
                    )
                    top_scale_width = len(dut.in_top_scale) // block
                    left_scale_width = len(dut.in_left_scale) // block
                    dut.in_top_scale.value = _packed_pattern_lanes(
                        lane_width=top_scale_width, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        salt=0x17,
                    )
                    dut.in_left_scale.value = _packed_pattern_lanes(
                        lane_width=left_scale_width, lane_count=block,
                        action=stimulus_action, pattern=PATTERN, rng=rng,
                        salt=0x28,
                    )
                    dut.system_top_valid.value = 1
                    dut.system_left_valid.value = 1
                    await RisingEdge(dut.clk)
                    dut.clear_accumulator.value = 0
                dut.system_top_valid.value = 0
                dut.system_left_valid.value = 0
                features["matrix.active_mac_bit_product"] += block**3 * width_a * width_b
                features["matrix.output_bit"] += block * block * 16
            await _cycles(dut, slot_cycles - (block if active else 0))
        else:
            if active:
                dut.int_data.value = _packed_pattern_lanes(
                    lane_width=16, lane_count=len(dut.int_data) // 16,
                    action=action, pattern=PATTERN, rng=rng, salt=0x5B,
                )
                dut.fp_data.value = _packed_pattern_lanes(
                    lane_width=16, lane_count=len(dut.fp_data) // 16,
                    action=action, pattern=PATTERN, rng=rng,
                    fp_format=(8, 7), salt=0x6C,
                )
                dut.acc_in.value = _pattern_bits(len(dut.acc_in), action + 23, PATTERN, rng)
                dut.scale_in.value = _pattern_bits(
                    len(dut.scale_in), action + 31, PATTERN, rng
                )
                dut.in_valid.value = 1
                await RisingEdge(dut.clk)
                dut.in_valid.value = 0
                features["matrix.output_bit"] += 16 * 16
            await _cycles(dut, slot_cycles - (1 if active else 0))
    return REPEATS * slot_cycles


async def _vector_activity(dut, active: bool, rng: random.Random, features: dict[str, float]) -> int:
    exp = int(PARAMS["V_FP_EXP_WIDTH"])
    mant = int(PARAMS["V_FP_MANT_WIDTH"])
    width = 1 + exp + mant
    vlen = int(PARAMS["VLEN"])
    one = _one(exp, mant)
    low = _packed_lanes([one] * vlen, width)
    for name in ("element_op", "reduction_op", "reduction_segment_log2", "reduction_segment_index",
                 "compact_active_lanes", "compact_stats_en", "reduction_overwrite_en",
                 "broadcast_fp2", "segment_broadcast_en", "lane_store_en", "vector_mask",
                 "element_mask_enable", "v_a", "v_a_valid", "v_b", "v_b_valid", "scalar_in",
                 "scalar_in_valid", "scalar_target", "result_waddr", "result_waddr_update"):
        getattr(dut, name).value = 0
    await _reset_common(dut)
    slot_cycles = 32
    # Different train/validation mixes make operation-family slopes
    # identifiable. Keeping one fixed mix made ADD and movement exactly
    # collinear even though their datapaths have different switching costs.
    if PATTERN == "low-toggle":
        operations = ("move", "add", "move", "reduce")
    elif PATTERN == "representative-qwen":
        operations = ("add", "mul", "reduce", "add", "move")
    elif PATTERN == "mixed-kernel-holdout":
        operations = ("add", "mul", "exp", "reduce", "move", "mul", "add")
    else:
        operations = ("add", "mul", "exp", "reduce", "move")
    selected_operations = {
        "add_vv": "add_vv", "add_vf": "add_vf", "add_vseg": "add_vseg",
        "mul_vv": "mul_vv", "mul_vf": "mul_vf", "mul_vseg": "mul_vseg",
        "exp": "exp", "reciprocal": "reciprocal", "reduce_sum": "reduce_sum",
        "reduce_max": "reduce_max", "reduce_sum_seg": "reduce_sum_seg",
        "reduce_max_seg": "reduce_max_seg", "reduce_sum_segs": "reduce_sum_segs",
        "reduce_max_segs": "reduce_max_segs", "shift": "move",
        "lane_load": "lane_load", "lane_store": "lane_store",
        "compact_stats_mul": "compact_stats_mul",
        "compact_stats_add": "compact_stats_add",
        "compact_stats_rsqrt": "compact_stats_rsqrt",
        "reduce_sum_ovr": "reduce_sum_ovr",
        "reduce_max_ovr": "reduce_max_ovr",
        "reduce_sum_seg_ovr": "reduce_sum_seg_ovr",
        "reduce_max_seg_ovr": "reduce_max_seg_ovr",
    }
    for action in range(REPEATS):
        op = selected_operations.get(
            MICROKERNEL,
            MIX_SEQUENCE[action % len(MIX_SEQUENCE)]
            if MIX_SEQUENCE
            else operations[action % len(operations)],
        )
        if active:
            if PATTERN == "low-toggle":
                vector = low
            else:
                vector = _packed_lanes(
                    [
                        _finite_fp_value(exp, mant, action * vlen + lane, PATTERN, rng)
                        for lane in range(vlen)
                    ],
                    width,
                )
            dut.v_a.value = vector
            dut.v_b.value = ((vector << width) | (vector >> max(width, vlen * width - width))) & ((1 << (vlen * width)) - 1)
            dut.scalar_in.value = one
            dut.broadcast_fp2.value = 0
            dut.reduction_segment_log2.value = min(4, max(0, (vlen.bit_length() - 1) // 2))
            compact_lanes = int(PARAMS.get("COMPACT_STATS_LANES", min(64, vlen)))
            dut.compact_active_lanes.value = compact_lanes
            dut.result_waddr.value = action
            dut.result_waddr_update.value = 1
            dut.v_a_valid.value = 1
            if op in {"add", "add_vv", "add_vf", "add_vseg"}:
                dut.element_op.value = 1
                dut.v_b_valid.value = int(op in {"add", "add_vv", "add_vseg"})
                dut.scalar_in_valid.value = int(op == "add_vf")
                dut.broadcast_fp2.value = int(op == "add_vf")
                dut.segment_broadcast_en.value = int(op == "add_vseg")
                features["vector.lane_add_sub_bit"] += vlen * width
            elif op in {"mul", "mul_vv", "mul_vf", "mul_vseg"}:
                dut.element_op.value = 3
                dut.v_b_valid.value = int(op in {"mul", "mul_vv", "mul_vseg"})
                dut.scalar_in_valid.value = int(op == "mul_vf")
                dut.broadcast_fp2.value = int(op == "mul_vf")
                dut.segment_broadcast_en.value = int(op == "mul_vseg")
                features["vector.lane_multiply_bit2"] += vlen * width * width
            elif op == "exp":
                dut.element_op.value = 4
                features["vector.lane_sfu_bit2"] += vlen * width * width
            elif op == "reciprocal":
                dut.element_op.value = 5
                features["vector.lane_sfu_bit2"] += vlen * width * width
            elif op in {
                "compact_stats_mul",
                "compact_stats_add",
                "compact_stats_rsqrt",
            }:
                dut.element_op.value = {
                    "compact_stats_mul": 10,
                    "compact_stats_add": 11,
                    "compact_stats_rsqrt": 12,
                }[op]
                dut.compact_stats_en.value = 1
                dut.scalar_in_valid.value = int(op != "compact_stats_rsqrt")
                features[f"vector.{op}"] += compact_lanes
            elif op in {
                "reduce", "reduce_sum", "reduce_max", "reduce_sum_seg", "reduce_max_seg",
                "reduce_sum_segs", "reduce_max_segs", "reduce_sum_ovr",
                "reduce_max_ovr", "reduce_sum_seg_ovr",
                "reduce_max_seg_ovr", "lane_load",
            }:
                dut.reduction_op.value = {
                    "reduce": 1, "reduce_sum": 1, "reduce_max": 2,
                    "reduce_sum_seg": 3, "reduce_max_seg": 4,
                    "reduce_sum_segs": 5, "reduce_max_segs": 6,
                    "reduce_sum_ovr": 1, "reduce_max_ovr": 2,
                    "reduce_sum_seg_ovr": 3, "reduce_max_seg_ovr": 4,
                    "lane_load": 7,
                }[op]
                dut.reduction_overwrite_en.value = int(op.endswith("_ovr"))
                if op == "lane_load":
                    dut.reduction_segment_index.value = action % vlen
                    features["scalar.vector_lane_access_bit"] += width
                else:
                    segment_log2 = int(dut.reduction_segment_log2.value)
                    if op in {
                        "reduce_sum_seg",
                        "reduce_max_seg",
                        "reduce_sum_seg_ovr",
                        "reduce_max_seg_ovr",
                    }:
                        reduction_nodes = max((1 << segment_log2) - 1, 0)
                    elif op in {"reduce_sum_segs", "reduce_max_segs"}:
                        reduction_nodes = max(vlen - ((vlen + (1 << segment_log2) - 1) >> segment_log2), 0)
                    else:
                        reduction_nodes = max(vlen - 1, 0)
                    features["vector.reduction_node_bit"] += reduction_nodes * width
            elif op == "lane_store":
                dut.element_op.value = 9
                dut.broadcast_fp2.value = 1
                dut.lane_store_en.value = 1
                dut.reduction_segment_index.value = action % vlen
                dut.scalar_in_valid.value = 1
                features["scalar.vector_lane_access_bit"] += width
            else:
                dut.element_op.value = 8
                dut.reduction_segment_index.value = 1
                features["vector.lane_movement_bit"] += vlen * width
            await RisingEdge(dut.clk)
            dut.element_op.value = 0
            dut.reduction_op.value = 0
            dut.v_a_valid.value = 0
            dut.v_b_valid.value = 0
            dut.scalar_in_valid.value = 0
            dut.broadcast_fp2.value = 0
            dut.segment_broadcast_en.value = 0
            dut.compact_stats_en.value = 0
            dut.reduction_overwrite_en.value = 0
            dut.lane_store_en.value = 0
            dut.result_waddr_update.value = 0
        await _cycles(dut, slot_cycles - (1 if active else 0))
    return REPEATS * slot_cycles


async def _scalar_activity(dut, active: bool, rng: random.Random, features: dict[str, float]) -> int:
    exp = int(PARAMS["S_FP_EXP_WIDTH"])
    mant = int(PARAMS["S_FP_MANT_WIDTH"])
    width = 1 + exp + mant
    int_width = int(PARAMS["INT_DATA_WIDTH"])
    for name in ("scalar_fp_op", "scalar_int_op", "fps1", "fps2", "fpd", "map_addr",
                 "external_fp_in", "external_fp_in_valid", "external_fp_wtarget",
                 "int_rs1", "int_rs2", "int_rd", "int_imm"):
        getattr(dut, name).value = 0
    await _reset_common(dut)

    # Keep f1-f8 read-only during measurement.  Rotating among distinct finite
    # operands avoids RAW stalls and keeps the mapped datapath switching.
    for register in range(1, 9):
        dut.external_fp_in.value = _finite_fp_value(exp, mant, register, PATTERN, rng)
        dut.external_fp_wtarget.value = register
        dut.external_fp_in_valid.value = 1
        await RisingEdge(dut.clk)
        dut.external_fp_in_valid.value = 0
        await RisingEdge(dut.clk)
    for _ in range(64):
        if not int(dut.backend_busy.value):
            break
        await RisingEdge(dut.clk)
    assert not int(dut.backend_busy.value), "scalar register preload did not retire"

    # Keep the action density high enough that active-minus-idle measures the
    # datapath rather than tiny state-dependent changes in clock-pin internal
    # power.  Eight cycles remains conservative for the 8-entry scalar ROB and
    # the mixed ADD/MUL/SFU stream used here.
    slot_cycles = 8
    if PATTERN == "representative-qwen":
        fp_ops = (4, 1, 14, 4, 12, 6)
    elif PATTERN == "mixed-kernel-holdout":
        fp_ops = (1, 2, 3, 4, 5, 6, 7, 12)
    elif PATTERN == "low-toggle":
        fp_ops = (12, 1, 4, 12, 2)
    else:
        fp_ops = (1, 4, 5, 12, 6, 7)
    scalar_microkernels = {
        "fp_alu": ("fp", 1), "fp_mul": ("fp", 4), "fp_exp": ("fp", 5),
        "fp_reciprocal": ("fp", 6), "fp_sqrt": ("fp", 7),
        "fp_rsqrt": ("fp", 14), "int_alu": ("int", 2),
        "int_mul": ("int", 4), "register_access": ("fp", 12),
    }
    for action in range(REPEATS):
        if active:
            selected = scalar_microkernels.get(MICROKERNEL)
            if selected is None and MIX_SEQUENCE:
                selected = scalar_microkernels[MIX_SEQUENCE[action % len(MIX_SEQUENCE)]]
            use_integer = selected is not None and selected[0] == "int"
            if use_integer or (selected is None and action % 6 == 5):
                # ADDI creates changing GP values.  Reading two reset-zero
                # registers previously exercised mostly control logic.
                int_reg = 1 + (action // 6) % 7
                int_op = int(selected[1]) if selected is not None else 2
                dut.scalar_int_op.value = int_op
                dut.int_rs1.value = int_reg
                dut.int_rs2.value = 0
                dut.int_rd.value = int_reg
                dut.int_imm.value = 1 + ((action * 13) & ((1 << len(dut.int_imm)) - 1))
                features["scalar.integer_alu_bit"] += int_width
                features["scalar.register_access_bit"] += 2 * int_width
            else:
                op = int(selected[1]) if selected is not None else fp_ops[action % len(fp_ops)]
                dut.scalar_fp_op.value = op
                dut.fps1.value = 1 + action % 8
                dut.fps2.value = 1 + (action * 3 + 1) % 8
                dut.fpd.value = 9 + action % 7
                if op in (1, 2, 3, 12):
                    features["scalar.fp_add_sub_move_bit"] += width
                elif op == 4:
                    features["scalar.fp_multiply_bit2"] += width * width
                else:
                    features["scalar.fp_sfu_bit2"] += width * width
                features["scalar.register_access_bit"] += 3 * width
                await Timer(1, units="ps")
                assert not int(dut.frontend_stall.value), (
                    f"scalar action {action} was not accepted by the frontend"
                )
            await RisingEdge(dut.clk)
            dut.scalar_fp_op.value = 0
            dut.scalar_int_op.value = 0
        await _cycles(dut, slot_cycles - (1 if active else 0))

    # Include the same fixed drain in active and matched-idle windows.  This
    # makes completed_actions meaningful and absorbs the final SFU tail.
    drain_cycles = 32
    await _cycles(dut, drain_cycles)
    if active:
        assert not int(dut.backend_busy.value), "scalar actions did not drain from the ROB"
    return REPEATS * slot_cycles + drain_cycles


async def _control_activity(dut, active: bool, _rng: random.Random, features: dict[str, float]) -> int:
    inputs = (
        "decode_m_op", "decode_v_element_op", "decode_v_reduction_op", "decode_s_fp_op",
        "decode_c_op", "decode_h_op", "decode_v_broadcast_en", "hbm_m_prefetch_in_progress",
        "hbm_v_prefetch_in_progress", "continuous_write_to_v_sram", "fp_stall_req",
        "fp_sram_stall_req", "m_load_in_process", "m_mcu_active", "s_received_v_reduct_result",
        "mem_wreq_m_sram", "mem_wreq_s_sram_port_a", "mem_wreq_s_sram_port_b",
        "mem_wreq_from_m",
    )
    for name in inputs:
        getattr(dut, name).value = 0
    await _reset_common(dut)
    slot_cycles = 8
    for action in range(REPEATS):
        if active:
            selector = action % 5
            if selector == 0:
                dut.decode_v_element_op.value = 1
            elif selector == 1:
                dut.decode_v_reduction_op.value = 1
            elif selector == 2:
                dut.decode_s_fp_op.value = 4
            elif selector == 3:
                dut.decode_m_op.value = 5
            else:
                dut.decode_h_op.value = 3
            features["control.frontend_issue"] += 1
            await RisingEdge(dut.clk)
            for name in ("decode_v_element_op", "decode_v_reduction_op", "decode_s_fp_op", "decode_m_op", "decode_h_op"):
                getattr(dut, name).value = 0
        await _cycles(dut, slot_cycles - (1 if active else 0))
    return REPEATS * slot_cycles


async def _agu_activity(dut, active: bool, rng: random.Random, features: dict[str, float]) -> int:
    """Exercise setup, affine stepping, and resolved-read paths independently."""

    inputs = (
        "config_valid", "config_reg", "config_stride", "frame_start",
        "frame_counter_reg", "boundary_step", "boundary_exit",
        "gp_write_valid", "gp_write_addr", "gp_read_addr_1",
        "gp_read_addr_2", "gp_read_valid_1", "gp_read_valid_2",
        "gp_base_1", "gp_base_2",
    )
    for name in inputs:
        getattr(dut, name).value = 0
    await _reset_common(dut)
    slot_cycles = 12
    selected = MICROKERNEL

    if active and selected.startswith("boundary_"):
        stream_count = int(selected.rsplit("_", 1)[1])
        for stream in range(stream_count):
            dut.config_valid.value = 1
            dut.config_reg.value = stream + 1
            dut.config_stride.value = 16 + stream
            await RisingEdge(dut.clk)
        dut.config_valid.value = 0
        dut.frame_start.value = 1
        dut.frame_counter_reg.value = 15
        await RisingEdge(dut.clk)
        dut.frame_start.value = 0

    for action in range(REPEATS):
        consumed_cycles = 0
        if active:
            if selected.startswith("boundary_"):
                stream_count = int(selected.rsplit("_", 1)[1])
                dut.gp_read_addr_1.value = 1 + action % stream_count
                dut.gp_read_addr_2.value = 1 + (action + 1) % stream_count
                dut.gp_read_valid_1.value = 1
                dut.gp_read_valid_2.value = 1
                dut.gp_base_1.value = _pattern_bits(32, action, PATTERN, rng)
                dut.gp_base_2.value = _pattern_bits(32, action + 17, PATTERN, rng)
                dut.boundary_step.value = 1
                features["agu.loop_boundary"] += 1
                features["agu.stream_step"] += stream_count
                features["agu.offset_read"] += 2
                await RisingEdge(dut.clk)
                consumed_cycles = 1
                dut.boundary_step.value = 0
            elif selected == "offset_read":
                dut.gp_read_addr_1.value = 1 + action % 6
                dut.gp_read_addr_2.value = 1 + (action + 3) % 6
                dut.gp_read_valid_1.value = 1
                dut.gp_read_valid_2.value = 1
                dut.gp_base_1.value = _pattern_bits(32, action, PATTERN, rng)
                dut.gp_base_2.value = _pattern_bits(32, action + 29, PATTERN, rng)
                features["agu.offset_read"] += 2
                await RisingEdge(dut.clk)
                consumed_cycles = 1
            elif selected.startswith("setup_"):
                stream_count = int(selected.rsplit("_", 1)[1])
                for stream in range(stream_count):
                    dut.config_valid.value = 1
                    dut.config_reg.value = stream + 1
                    dut.config_stride.value = 16 + stream
                    await RisingEdge(dut.clk)
                    consumed_cycles += 1
                dut.config_valid.value = 0
                dut.frame_start.value = 1
                dut.frame_counter_reg.value = 15
                await RisingEdge(dut.clk)
                consumed_cycles += 1
                dut.frame_start.value = 0
                dut.boundary_step.value = 1
                dut.boundary_exit.value = 1
                await RisingEdge(dut.clk)
                consumed_cycles += 1
                dut.boundary_step.value = 0
                dut.boundary_exit.value = 0
                features["agu.config"] += stream_count
                features["agu.loop_setup"] += 1
                features["agu.loop_boundary"] += 1
                features["agu.stream_step"] += stream_count
            else:
                raise AssertionError(f"unsupported AGU microkernel {selected}")
        await _cycles(dut, slot_cycles - consumed_cycles)
    return REPEATS * slot_cycles


async def _hbm_activity(dut, active: bool, rng: random.Random, features: dict[str, float]) -> int:
    for name in ("h_op", "addr_1", "addr_2", "prefetch_v_ready", "write_high_valid",
                 "write_low_valid", "write_high_element", "write_low_element", "write_scale"):
        getattr(dut, name).value = 0
    dut.prefetch_v_ready.value = 1
    await _reset_common(dut)
    lane_width = max(
        _precision_width(str(PARAMS["ACT_WIDTH"])),
        _precision_width(str(PARAMS["KV_WIDTH"])),
        _precision_width(str(PARAMS["WEIGHT_WIDTH"])),
    )
    element_width = _next_hbm_element_width(int(PARAMS["MLEN"]) * lane_width)
    scale_width = (
        int(PARAMS["MX_SCALE_WIDTH"])
        * (int(PARAMS["MLEN"]) // int(PARAMS["BLOCK_DIM"]))
    )
    element_bytes = (element_width + 7) // 8
    scale_bytes = (scale_width + 7) // 8
    slot_cycles = 96
    for action in range(REPEATS):
        before_lines = int(dut.accepted_lines.value)
        before_element_lines = int(dut.accepted_element_lines.value)
        before_scale_lines = int(dut.accepted_scale_lines.value)
        if active:
            selected_microkernel = (
                MIX_SEQUENCE[action % len(MIX_SEQUENCE)] if MICROKERNEL == "mixed" and MIX_SEQUENCE
                else MICROKERNEL
            )
            op = {
                "matrix_prefetch": 1,
                "vector_prefetch": 3,
                "vector_writeback": 5,
            }.get(selected_microkernel, (1, 3, 5)[action % 3])
            dut.h_op.value = op
            dut.addr_1.value = action * 64
            dut.addr_2.value = action * 64 + 32
            if op == 5:
                dut.write_high_valid.value = 1
                high_lane_count = int(PARAMS["VLEN"])
                high_lane_width = len(dut.write_high_element) // high_lane_count
                dut.write_high_element.value = _packed_pattern_lanes(
                    lane_width=high_lane_width, lane_count=high_lane_count,
                    action=action, pattern=PATTERN, rng=rng,
                    salt=0x39,
                )
                scale_lane_count = max(
                    1, int(PARAMS["VLEN"]) // int(PARAMS["BLOCK_DIM"])
                )
                scale_lane_width = len(dut.write_scale) // scale_lane_count
                dut.write_scale.value = _packed_pattern_lanes(
                    lane_width=scale_lane_width, lane_count=scale_lane_count,
                    action=action, pattern=PATTERN, rng=rng,
                    salt=0x4A,
                )
            await RisingEdge(dut.clk)
            dut.h_op.value = 0
            dut.write_high_valid.value = 0
            features["hbm.dma_issue"] += 1
        await _cycles(dut, slot_cycles - (1 if active else 0))
        lines = max(0, int(dut.accepted_lines.value) - before_lines)
        element_lines = max(
            0, int(dut.accepted_element_lines.value) - before_element_lines
        )
        scale_lines = max(0, int(dut.accepted_scale_lines.value) - before_scale_lines)
        features["hbm.line"] += lines
        features["hbm.byte"] += (
            element_lines * element_bytes + scale_lines * scale_bytes
        )
    return REPEATS * slot_cycles


@cocotb.test()
async def generate_power_activity(dut):
    cocotb.start_soon(Clock(dut.clk, CLOCK_NS, units="ns").start())
    rng = random.Random(SEED)
    features = _features()
    active = PATTERN != "idle"
    if COMPONENT == "matrix":
        cycles = await _matrix_activity(dut, active, rng, features)
    elif COMPONENT == "vector":
        cycles = await _vector_activity(dut, active, rng, features)
    elif COMPONENT == "scalar":
        cycles = await _scalar_activity(dut, active, rng, features)
    elif COMPONENT == "control":
        cycles = await _control_activity(dut, active, rng, features)
    elif COMPONENT == "agu":
        cycles = await _agu_activity(dut, active, rng, features)
    elif COMPONENT == "hbm":
        cycles = await _hbm_activity(dut, active, rng, features)
    else:
        raise AssertionError(f"unsupported component {COMPONENT}")
    end = float(get_sim_time(units="ns"))
    # The component routines include reset before their action loop. Derive the
    # exact action window from its fixed cycle count so reset is excluded.
    measurement_start = end - cycles * CLOCK_NS
    payload = {
        "schema_version": 1,
        "activity_fingerprint": ACTIVITY_FINGERPRINT,
        "component": COMPONENT,
        "pattern": PATTERN,
        "microkernel": MICROKERNEL,
        "qwen_mix_semantic_hash": MIX_HASH or None,
        "mix_action_counts": dict(Counter(MIX_SEQUENCE)) if MIX_SEQUENCE else {},
        "requested_actions": REPEATS,
        "accepted_actions": REPEATS if active else 0,
        "completed_actions": REPEATS if active else 0,
        "measurement_start_ns": measurement_start,
        "measurement_end_ns": end,
        "measurement_cycles": cycles,
        "clock_period_ns": CLOCK_NS,
        "seed": SEED,
        "dynamic_features": features,
        "clock_features": _clock_features(cycles),
        "params": PARAMS,
    }
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    component_paths = {
        "matrix": [
            "basic_components/systolic_gemm_mx", "basic_components/systolic_gemm_mxint",
            "basic_components/mx_fp_operation", "basic_components/mx_int_operation",
            "basic_components/gemv", "basic_components/conversion",
        ],
        "vector": ["vector_machine", "basic_components/hadamard_transform", "basic_components/synopsis_ip_inst"],
        "scalar": ["scalar_machine", "memory/scalar_sram", "memory/vector_sram"],
        "control": ["control"],
        "agu": ["scalar_machine"],
        "hbm": ["memory/HBM", "memory/HBM/TileLink_Lib"],
    }
    shared = [
        "basic_components/buffer", "basic_components/common", "basic_components/fp_operation",
        "basic_components/int_operation", "basic_components/cast", "basic_components/fixed_operation",
        "basic_components/conversion", "basic_components/synopsis", "basic_components/synopsis_ip_inst",
    ]
    include_paths = [str(SRC_PATH / path) for path in dict.fromkeys(shared + component_paths[COMPONENT])]
    veri_runner(
        group="power_activity",
        module="power_activity_tb",
        additional_include_paths=include_paths,
        definitions_path=[str(SRC_PATH / "definitions"), str(SRC_PATH / "memory/HBM/TileLink_Lib")],
        trace=True,
        skip_build=os.environ.get("PLENA_POWER_SKIP_BUILD", "0") == "1",
        test_module=Path(__file__).stem,
        test_dir=Path(os.environ["PLENA_POWER_TEST_DIR"]),
        sim_build_dir=Path(os.environ["PLENA_POWER_VCD_DIR"]),
        # Keep the repository-supported struct tracing mode. Replay translates
        # Verilator's nested packed-member scopes through the exact DC SAIF
        # name map instead of relying on simulator-specific flat names.
        extra_build_args=["-DSIMULATION"],
    )
