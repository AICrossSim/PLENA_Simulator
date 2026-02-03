"""Output computation test for flashattn.output module.

Tests computing_o_code and computing_row_wise_scaling_code functions:
- computing_o_code: O = diag(m_res) * O_old + PV
- computing_row_wise_scaling_code: O = O / l (final normalization)

Memory layout:
- HBM: O_old, PV
- VSRAM: O_old (preloaded), PV (preloaded), O (output)
- FPSRAM: m_res, l values preloaded
"""

import sys
import json
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compiler.asm_templates import preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from compiler.asm_templates.flashattn import (
    computing_o_code, computing_row_wise_scaling_code
)
from compiler.asm_templates.reset_reg_asm import reset_fpreg_asm, reset_vmask_asm
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    # Test configuration
    batch_size = 1
    num_q_heads = 4
    num_kv_heads = 1
    h_qkv = 16
    mlen = 64
    vlen = 64
    blen = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)

    q_index_2_kv_index_ratio = num_q_heads // num_kv_heads
    hidden_size = h_qkv * num_q_heads

    device = torch.device("cpu")
    print(f"flashattn.output Test Config:")
    print(f"  batch_size={batch_size}")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, h_qkv={h_qkv}")
    print(f"  q_index_2_kv_index_ratio={q_index_2_kv_index_ratio}")
    print(f"  blen={blen}, mlen={mlen}, hidden_size={hidden_size}")

    torch.manual_seed(42)

    # Test with single head for simplicity
    test_q_head = 0

    # Generate test data
    # O_old: (batch, mlen, hidden_size) - previous output (packed format)
    o_old = torch.randn(batch_size, mlen, mlen, dtype=torch.bfloat16, device=device)

    # PV: (mlen, mlen) - new PV result for a single head (will be placed in packed format)
    pv_single_head = torch.randn(mlen, h_qkv, dtype=torch.bfloat16, device=device)

    # m_res: (mlen,) - scaling factors (exp(m_new - m_old)), use values between 0 and 1
    m_res = torch.rand(mlen, dtype=torch.float32, device=device) * 0.5 + 0.5  # Between 0.5 and 1.0

    # l: (mlen,) - row-wise sums for normalization, positive values
    l_values = torch.rand(mlen, dtype=torch.float32, device=device) * 10 + 1.0  # Between 1.0 and 11.0

    print(f"\nTensor shapes:")
    print(f"  O_old: {o_old.shape}")
    print(f"  PV (single head): {pv_single_head.shape}")
    print(f"  m_res: {m_res.shape}")
    print(f"  l: {l_values.shape}")

    # Compute golden output
    # Step 1: O = diag(m_res) * O_old + PV (for the test head only)
    o_golden = o_old.clone()
    for i in range(mlen):
        # Scale the test head's portion of O_old by m_res
        head_start = test_q_head * h_qkv
        head_end = (test_q_head + 1) * h_qkv
        o_golden[0, i, head_start:head_end] = (
            m_res[i].to(torch.bfloat16) * o_old[0, i, head_start:head_end] +
            pv_single_head[i, :]
        )

    # Step 2: O = O / l (for the test head only)
    for i in range(mlen):
        head_start = test_q_head * h_qkv
        head_end = (test_q_head + 1) * h_qkv
        o_golden[0, i, head_start:head_end] = (
            o_golden[0, i, head_start:head_end] / l_values[i].to(torch.bfloat16)
        )

    print(f"\nGolden output shape: {o_golden.shape}")

    # Create PV in packed format (zeros for other heads)
    pv_packed = torch.zeros(mlen, hidden_size, dtype=torch.bfloat16, device=device)
    pv_packed[:, test_q_head * h_qkv : (test_q_head + 1) * h_qkv] = pv_single_head

    # Memory layout in VSRAM
    o_old_base_address = 0
    o_old_total_size = mlen * hidden_size
    pv_base_address = o_old_base_address + o_old_total_size

    print(f"\nVSRAM Layout:")
    print(f"  O_old Base: {o_old_base_address}")
    print(f"  O_old Total Size: {o_old_total_size}")
    print(f"  PV Base: {pv_base_address}")

    # HBM layout
    o_old_hbm_size = int(mlen * hidden_size * batch_size * real_data_ratio)
    pv_hbm_offset = o_old_hbm_size

    print(f"\nHBM Layout:")
    print(f"  O_old: 0 - {o_old_hbm_size}")
    print(f"  PV offset: {pv_hbm_offset}")

    # FPSRAM layout
    # Address 0: zero (0.0)
    # Address 1-mlen: m_res values
    # Address mlen+1 to 2*mlen: l values
    fp_sram_m_res_start = 1
    fp_sram_l_start = fp_sram_m_res_start + mlen

    print(f"\nFPSRAM Layout:")
    print(f"  m_res start: {fp_sram_m_res_start}")
    print(f"  l start: {fp_sram_l_start}")

    # Prepare fp_preload: [0.0, m_res..., l...]
    fp_preload = [0.0]  # Address 0: zero
    fp_preload.extend(m_res.tolist())  # Address 1 to mlen: m_res
    fp_preload.extend(l_values.tolist())  # Address mlen+1 to 2*mlen: l

    print(f"  fp_preload length: {len(fp_preload)}")

    # Input tensors for HBM
    # pv_packed: shape (mlen, hidden_size), but we want to create (mlen, mlen) where each row contains h_qkv real values padded with zeros to mlen width
    # Step 1: For each row, put pv_single_head[i, :] in the first h_qkv positions, pad with zeros to length mlen
    pv_square = torch.zeros(mlen, mlen, dtype=torch.bfloat16, device=device)
    for i in range(mlen):
        pv_square[i, :h_qkv] = pv_single_head[i, :]  # real values in first h_qkv positions, rest zero

    input_tensor = {
        "o_old": o_old.reshape(batch_size, -1),
        "pv": pv_square.reshape(1, -1),  # (1, mlen * mlen)
    }

    # Generate assembly
    gen_assembly_code = "; flashattn.output Test \n"

    # Set PV addr offset register
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1, 2],
        addr_reg_val=[pv_hbm_offset]
    )

    # Preload O_old to VSRAM
    gen_assembly_code += "; Preload O_old to VSRAM\n"
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=batch_size * mlen,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=o_old_base_address,
        activation_offset_reg=0
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    # Preload PV to VSRAM
    gen_assembly_code += "; Preload PV to VSRAM\n"
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=mlen,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=pv_base_address,
        activation_offset_reg=1
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    # Set V_MASK for the test head
    gen_assembly_code += reset_vmask_asm(1, 1 << test_q_head)

    # Computing O: O = diag(m_res) * O_old + PV
    gen_assembly_code += computing_o_code(
        mlen=mlen,
        alive_registers_int=[1, 2, 3, 4],
        alive_registers_fp=[1],
        m_res_base_address=fp_sram_m_res_start,
        pv_base_address=pv_base_address,
        o_old_base_address=o_old_base_address,
        head_dim=h_qkv,
        q_head_num=num_q_heads,
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4])
    gen_assembly_code += reset_fpreg_asm(alive_registers=[1])

    # Row-wise scaling: O = O / l
    gen_assembly_code += computing_row_wise_scaling_code(
        mlen=mlen,
        alive_registers_int=[1, 2, 3],
        alive_registers_fp=[1],
        o_old_base_address=o_old_base_address,
        l_old_base_address=fp_sram_l_start,
        o_row_stride=hidden_size,
        use_mask=True,
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])
    gen_assembly_code += reset_fpreg_asm(alive_registers=[1])

    # Golden result - output in packed format
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": o_golden.reshape(-1).unsqueeze(0)
    }

    print(f"\nGolden output packed shape: {o_golden.shape}")

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm=None, data=None, specified_data_order=["o_old", "pv"], build_path=build_path)

    result_start_row = o_old_base_address // vlen
    num_result_rows = (mlen * hidden_size) // vlen

    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": mlen,
        "elements_per_batch": hidden_size,
        "row_dim": vlen,
        "use_stride_mode": False
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print(f"\nResult location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("=" * 60)
