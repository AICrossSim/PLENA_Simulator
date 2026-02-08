#!/usr/bin/env python3
"""
Correct & readable test for `M_BTMM` + `M_BMM_WO`.

What this test guarantees:
- Writes/compares **all 4 heads** (BROADCAST_AMOUNT) instead of only head0.
- Uses the simulator's actual tiling semantics:
  - `HLEN=16`, `BROADCAST_AMOUNT=4`, `MLEN=64`
  - `M_BTMM` accumulates partial sums into `h_accum`, slicing K in 16-wide chunks by `head_offset`
  - `M_BMM_WO` writes back all heads: head0..head3, each 64x64
- Matches Q VRAM layout to mm_load_stride=4 by writing Q with stride-mult=4.
"""

from __future__ import annotations

import os
import sys
import json
import tomlkit
from pathlib import Path

# Make imports work no matter where this script is run from.
ROOT = Path(__file__).resolve().parents[2]  # /.../Coprocessor_for_Llama
TESTBENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTBENCH_DIR))

try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "This testbench requires `torch` because the existing sim env builder loads "
        "inputs from `.pt` and writes `golden_result.txt` using torch. "
        "Install it (CPU is fine) or run in the repo's expected Python environment.\n"
        "Example: `pip install torch --index-url https://download.pytorch.org/whl/cpu`"
    ) from e

from compiler.asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm
from config_utils import update_plena_config
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from check_mem import compare_with_golden, print_comparison_results, parse_golden_output, read_bin_file_as_array


PLENA_SETTINGS_PATH = ROOT / "src" / "definitions" / "plena_settings.toml"


def _read_plena_bmm_scale() -> float:
    # NOTE: bmm_scale is currently not in `src/definitions/plena_settings.toml`.
    # The behavioral simulator initializes it to 0.25 by default in `behavioral_simulator/src/main.rs`.
    # If you later move it into TOML, update this function.
    return float(os.environ.get("PLENA_BMM_SCALE", "0.25"))


def _set_hbm_v_prefetch_amount(value: int) -> int:
    """Set HBM_V_Prefetch_Amount in plena_settings.toml and return the previous value."""
    with open(PLENA_SETTINGS_PATH, "r") as f:
        config = tomlkit.load(f)
    old_value = config["CONFIG"]["HBM_V_Prefetch_Amount"]["value"]
    config["CONFIG"]["HBM_V_Prefetch_Amount"]["value"] = value
    with open(PLENA_SETTINGS_PATH, "w") as f:
        tomlkit.dump(config, f)
    return int(old_value)


def main() -> None:
    # Ensure we are using the expected tiling config for this test.
    update_plena_config(vlen=64, mlen=64, blen=4, verbose=False)
    
    # Temporarily set HBM_V_Prefetch_Amount=1 so each H_PREFETCH_V writes one row.
    # This allows us to write Q with stride-mult=4 to match mm_load_stride=4.
    old_hbm_v_prefetch = _set_hbm_v_prefetch_amount(1)

    mlen = 64
    d = 64
    hlen = 16
    broadcast_amount = 4
    mm_load_stride = 4  # This is hardcoded in main.rs
    assert broadcast_amount * hlen == mlen, "Expected BROADCAST_AMOUNT*HLEN == MLEN"

    batch = 1
    real_data_ratio = 1.125
    bmm_scale = _read_plena_bmm_scale()  # default is 0.25

    # Random Q/K like linear_test.py for a more realistic distribution.
    torch.manual_seed(42)
    q = torch.randn(mlen, d, dtype=torch.bfloat16)
    k = torch.randn(mlen, d, dtype=torch.bfloat16)

    # Golden must match BTMM semantics:
    # Q is interpreted as [mlen, broadcast_amount, hlen] (4 heads), K is sliced by head_offset.
    # Each BTMM uses the SAME q_head and accumulates over 4 K slices, so:
    # result_head = q_head @ (sum over K_slices).T
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    q_heads = q_f32.reshape(mlen, broadcast_amount, hlen)          # [64,4,16]
    k_slices = k_f32.reshape(mlen, broadcast_amount, hlen)         # [64,4,16]
    k_sum = k_slices.sum(dim=1)                                    # [64,16]
    golden_heads = []
    for h in range(broadcast_amount):
        q_head = q_heads[:, h, :]                                  # [64,16]
        golden_heads.append(q_head.matmul(k_sum.T) * float(bmm_scale))  # [64,64]
    golden_all_heads = torch.stack(golden_heads, dim=0)            # [4,64,64]

    input_tensor = {
        "q": q.reshape(batch, -1),  # [1,4096]
        "k": k.reshape(batch, -1),  # [1,4096]
    }
    golden_result_dict = {
        "input_tensor": input_tensor,
        # Write golden in the same flattened order we will read from VRAM:
        # head0(64x64) then head1 then head2 then head3
        "original_output": golden_all_heads.reshape(1, -1),
    }

    # -----------------------
    # Assembly generation
    # -----------------------
    asm = "; BTMM + BMM_WO correctness test (all heads)\n"

    # HBM layout: [q | padding/alignment | k]
    q_hbm_size = int(mlen * d * real_data_ratio)
    k_hbm_addr = ((q_hbm_size + 63) // 64) * 64

    asm += preload_addr_reg_asm(
        addr_reg_to_set=[0, 1],  # a0=q, a1=k
        available_registers=[1, 2],
        addr_reg_val=[0, k_hbm_addr],
    )

    # Load Q -> VRAM with stride-mult to match mm_load_stride (4).
    # With HBM_V_Prefetch_Amount=1 and preload_len=1, each H_PREFETCH_V writes one row.
    # vram_stride_mult=4 means rows are written at offsets 0, 4*64, 8*64, ... in VRAM.
    # This matches how BTMM reads with mm_load_stride=4.
    asm += reset_reg_asm(alive_registers=[3, 4, 5, 6, 7])
    asm += preload_act_asm(
        vlen=mlen,
        preload_len=1,  # One row per H_PREFETCH_V
        batch=batch,
        hidden_size=mlen * d,
        alive_registers=[3, 4, 5, 6, 7],
        act_vram_offset=0,
        activation_offset_reg=0,  # a0
        stride_size=mlen * d,
        vram_stride_mult=mm_load_stride,  # Write Q with stride=4 layout
    )

    # Prefetch K -> Matrix SRAM base 0
    asm += reset_reg_asm(alive_registers=[1, 2, 3])
    asm += f"S_ADDI_INT gp3, gp0, {mlen * d} \n"
    asm += "C_SET_SCALE_REG gp3 \n"
    asm += f"S_ADDI_INT gp3, gp0, {d} \n"
    asm += "C_SET_STRIDE_REG gp3 \n"
    asm += "S_ADDI_INT gp1, gp0, 0 \n"  # matrix sram addr
    asm += "S_ADDI_INT gp2, gp0, 0 \n"  # hbm offset (relative to a1)
    asm += "H_PREFETCH_M gp1, gp2, a1, 1, 1 \n"

    # BTMM: inner-dim tiling (head_offset 0/16/32/48).
    asm += reset_reg_asm(alive_registers=[1, 2])
    asm += "; BTMM inner-tiling: head_offset=0/16/32/48, accumulate to h_accum\n"
    # NOTE: The assembler parses 3-operand instructions as: `OPCODE rd, rs1, rs2`
    # (see `tools/assembler/parser.py`), and the simulator executes `M_BTMM { rs1, rs2, rd }` as:
    #   btmm(m_addr = gp[rs1] + gp[rd], v_addr = gp[rs2], stride_len = mm_load_stride, ...)
    # Therefore we must pass:
    #   - rs2 = v_addr (VRAM base for Q)  -> must be VLEN-aligned
    #   - rd  = head_offset              -> added into m_addr to select K inner tile
    asm += "S_ADDI_INT gp2, gp0, 0 \n"  # v_addr (VRAM base for Q)
    for head_offset in (0, 16, 32, 48):
        asm += f"; head_offset={head_offset}\n"
        asm += f"S_ADDI_INT gp1, gp0, {head_offset} \n"
        # Correct operand order (string): rd=gp1 (head_offset), rs1=gp0 (base), rs2=gp2 (v_addr)
        # IMPORTANT: use `gp0` (not bare `0`) so the assembler parses it as a register, not an immediate.
        asm += "M_BTMM gp1, gp0, gp2 \n"

    # Write output
    # Q occupies VRAM rows 0..63*4 (stride-mult layout), so result starts after that.
    result_vram_addr = mlen * mm_load_stride * mlen  # 64 * 4 * 64 = 16384
    asm += f"; BMM_WO writeback: VRAM[{result_vram_addr}] (writes all 4 heads, each 64x64)\n"
    asm += f"S_ADDI_INT gp1, gp0, {result_vram_addr} \n"
    asm += "M_BMM_WO gp1, 0 \n"

    # -----------------------
    # Emit build artifacts
    # -----------------------
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)

    # Comparison: read ALL heads (256 rows) starting from result_vram_addr/64
    result_start_row = result_vram_addr // mlen  # 256
    num_rows = mlen * broadcast_amount  # 256 rows = 4 heads
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_rows,
        "row_dim": mlen,
        "num_batches": broadcast_amount,
        "elements_per_batch": mlen * mlen,  # 4096 per head
        "use_stride_mode": False,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    fp_preload = [0.0, 1e-6, 1.0]
    create_sim_env(input_tensor, asm, golden_result_dict, fp_preload)
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="btmm_bmmwo_test",
        data=None,
        specified_data_order=["q", "k"],
    )

    # -----------------------
    # If simulator already ran, compare now
    # -----------------------
    vram_file = Path(__file__).parent.parent.parent / "behavioral_simulator" / "vram_dump.bin"
    golden_file = build_dir / "golden_result.txt"
    report_file = build_dir / "golden_vs_simulated_btmm_bmmwo-2.txt"

    if not (vram_file.exists() and golden_file.exists()):
        print("Build artifacts generated. Now run the behavioral simulator, then re-run this script to compare.")
        print(f"- asm: {build_dir / 'generated_asm_code.asm'}")
        print(f"- golden: {golden_file}")
        print(f"- vram expected at: {vram_file}")
        _set_hbm_v_prefetch_amount(old_hbm_v_prefetch)
        return

    results = compare_with_golden(
        str(vram_file),
        str(golden_file),
        exp_width=8,
        man_width=7,
        num_bytes_per_val=2,
        row_dim=mlen,
        start_row_idx=result_start_row,
        num_rows=num_rows,
        num_batches=broadcast_amount,
        elements_per_batch=mlen * mlen,
        tolerance=0.2,
        use_stride_mode=False,
        atol=0.1,
        rtol=0.1,
    )

    golden_np = parse_golden_output(str(golden_file))
    simulated_np = read_bin_file_as_array(str(vram_file), 8, 7, mlen, 2, result_start_row, num_rows)
    expected_vals = num_rows * mlen
    if len(simulated_np) < expected_vals:
        _set_hbm_v_prefetch_amount(old_hbm_v_prefetch)
        raise RuntimeError(
            f"VRAM dump too short: got {len(simulated_np)} vals, expected {expected_vals} "
            f"(start_row={result_start_row}, num_rows={num_rows}, row_dim={mlen})."
        )

    golden = torch.from_numpy(golden_np).to(torch.float32).reshape(broadcast_amount, mlen, mlen)
    simulated = torch.from_numpy(simulated_np[:expected_vals]).to(torch.float32).reshape(broadcast_amount, mlen, mlen)

    with open(report_file, "w") as f:
        f.write("BTMM + BMM_WO comparison (ALL HEADS)\n")
        f.write(f"MLEN={mlen}, HLEN={hlen}, BROADCAST_AMOUNT={broadcast_amount}, bmm_scale={bmm_scale}\n")
        f.write(f"VRAM result base addr={result_vram_addr} (start_row={result_start_row})\n\n")
        f.write("Summary:\n")
        f.write(json.dumps({k: results[k] for k in ("mse", "mae", "max_error", "match_rate", "allclose_pass")}, indent=2))
        f.write("\n\n")
        for h in range(broadcast_amount):
            f.write("=" * 80 + "\n")
            f.write(f"Head {h}\n")
            f.write("=" * 80 + "\n")
            f.write("Golden (first 8 rows, first 16 cols):\n")
            for i in range(8):
                row = " ".join(f"{golden[h, i, j].item():7.3f}" for j in range(16))
                f.write(f"Row {i:02d}: {row}\n")
            f.write("\nSimulated (first 8 rows, first 16 cols):\n")
            for i in range(8):
                row = " ".join(f"{simulated[h, i, j].item():7.3f}" for j in range(16))
                f.write(f"Row {i:02d}: {row}\n")
            f.write("\nSimulated nonzero row count (threshold=1e-6): ")
            f.write(str(int((simulated[h].abs().sum(dim=1) > 1e-6).sum().item())))
            f.write("\n\n")

    print(f"✅ Report written: {report_file}")
    print_comparison_results(results, verbose=True, comparison_params=comparison_params)
    
    # Restore original HBM_V_Prefetch_Amount
    _set_hbm_v_prefetch_amount(old_hbm_v_prefetch)


if __name__ == "__main__":
    main()
