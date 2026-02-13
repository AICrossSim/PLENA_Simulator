"""
Full Transformer Block Test (128x128)

Pipeline:
X -> QKV -> FlashAttention -> Wo -> residual + RMSNorm -> W1 -> W2 -> residual + RMSNorm
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import math
import torch

from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from plena_program import PLENAProgram


def rms_norm_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


if __name__ == "__main__":
    print("=" * 80)
    print("Full Transformer Block Test (128x128)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Config (target: 128 x 128)
    # -------------------------------------------------------------------------
    seq_len = 128
    hidden_size = 128
    mlen = 64
    inter_size = hidden_size
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    row_blocks = seq_len // mlen
    hidden_col_blocks = hidden_size // mlen
    num_q_blocks = seq_len // mlen
    num_k_blocks = seq_len // mlen

    qk_scale = 1.0 / math.sqrt(hidden_size)
    norm_eps = 1e-6
    norm_reci_hid = 1.0 / hidden_size

    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # Inputs + Golden
    # -------------------------------------------------------------------------
    X = torch.randn(seq_len, hidden_size) * 0.1
    Wq = torch.randn(hidden_size, hidden_size) * 0.1
    Wk = torch.randn(hidden_size, hidden_size) * 0.1
    Wv = torch.randn(hidden_size, hidden_size) * 0.1
    Wo = torch.randn(hidden_size, hidden_size) * 0.1
    W1 = torch.randn(hidden_size, inter_size) * 0.1
    W2 = torch.randn(inter_size, hidden_size) * 0.1

    Q = X.float() @ Wq.float()
    K = X.float() @ Wk.float()
    V = X.float() @ Wv.float()
    S = Q @ K.T
    P = torch.softmax(S * qk_scale, dim=-1)
    O_attn = P @ V
    O_proj = O_attn @ Wo.float()
    X_attn = rms_norm_ref(O_proj + X.float(), norm_eps)
    F1 = X_attn @ W1.float()
    F2 = F1 @ W2.float()
    golden_out = rms_norm_ref(F2 + X_attn, norm_eps)

    print(f"Input shape: {X.shape}")
    print(f"Golden output shape: {golden_out.shape}")

    # -------------------------------------------------------------------------
    # Program
    # -------------------------------------------------------------------------
    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    x_input = prog.input("X", shape=(seq_len, hidden_size))
    wq_input = prog.input("Wq", shape=(hidden_size, hidden_size))
    wk_input = prog.input("Wk", shape=(hidden_size, hidden_size))
    wv_input = prog.input("Wv", shape=(hidden_size, hidden_size))
    wo_input = prog.input("Wo", shape=(hidden_size, hidden_size))
    w1_input = prog.input("W1", shape=(hidden_size, inter_size))
    w2_input = prog.input("W2", shape=(inter_size, hidden_size))

    wq_sub = prog.register_sub_matrix(wq_input, name="Wq_sub")
    wk_sub = prog.register_sub_matrix(wk_input, name="Wk_sub")
    wv_sub = prog.register_sub_matrix(wv_input, name="Wv_sub")
    wo_sub = prog.register_sub_matrix(wo_input, name="Wo_sub")
    w1_sub = prog.register_sub_matrix(w1_input, name="W1_sub")
    w2_sub = prog.register_sub_matrix(w2_input, name="W2_sub")

    @prog.function
    def linear_fn(act_in, weight_sub, out_features: int):
        act = prog.load_batch(act_in, name="act")
        act_sub = prog.register_vram_sub_matrix(act, name="act_sub")
        out = prog.alloc("linear_out", seq_len, out_features)
        out_col_blocks = out_features // mlen

        for r in range(row_blocks):
            for c in range(out_col_blocks):
                prog.reset_mram()
                weight_sub.load_col(c)
                prog.vram_sub_projection_to(
                    vram_row=act_sub.row(r),
                    mram_col=weight_sub.col(c),
                    target=out,
                    target_row_idx=r,
                    target_col_idx=c,
                )
        return out

    @prog.function
    def linear_store_fn(act_in, weight_sub, out_features: int, store_name: str):
        y = linear_fn(act_in, weight_sub, out_features)
        y_store = prog.store(y, name=store_name)
        return y_store

    @prog.function
    def qkv_projection_fn(x_in):
        q_store = linear_store_fn(x_in, wq_sub, hidden_size, "Q_proj")
        k_store = linear_store_fn(x_in, wk_sub, hidden_size, "K_proj")
        v_store = linear_store_fn(x_in, wv_sub, hidden_size, "V_proj")
        return q_store, k_store, v_store

    @prog.function
    def flash_attention_core_store_fn(q_in, k_in, v_in, store_name: str):
        q_batch = prog.load_batch(q_in, name="Q")
        q_sub = prog.register_vram_sub_matrix(q_batch, name="Q_sub")
        k_sub = prog.register_sub_matrix(k_in, name="K_sub")
        v_sub = prog.register_sub_matrix(v_in, name="V_sub")

        s_block = prog.alloc("S_block", mlen, mlen)
        pv = prog.alloc("PV", mlen, hidden_size)
        o = prog.alloc("O", seq_len, hidden_size)

        for q_idx in range(num_q_blocks):
            prog.init_online_softmax(q_idx, o)
            for k_idx in range(num_k_blocks):
                prog.reset_mram()
                k_sub.load_row(k_idx)
                prog.vram_sub_projection_T_to(
                    q_sub.row(q_idx),
                    k_sub.row(k_idx),
                    s_block,
                    target_row_idx=0,
                    target_col_idx=0,
                )
                prog.online_softmax_block(s_block, qk_scale)
                prog.compute_pv(s_block, v_sub, k_idx, pv)
                prog.scale_o_row(o, q_idx)
                prog.vram_add(o, pv, dst_row_offset=q_idx * mlen)
            prog.final_scale_o(q_idx, o)

        o_store = prog.store(o, name=store_name)
        return o_store

    # Keep full-block helpers in this file for later use.
    @prog.function
    def add_inplace_fn(dst, src, col_blocks: int):
        for r in range(row_blocks):
            for c in range(col_blocks):
                prog.vram_block_add_to(
                    src1=dst,
                    src1_row_idx=r,
                    src1_col_idx=c,
                    src2=src,
                    src2_row_idx=r,
                    src2_col_idx=c,
                    target=dst,
                    target_row_idx=r,
                    target_col_idx=c,
                )
        return dst

    @prog.function
    def attention_block_fn(x_in):
        q_store, k_store, v_store = qkv_projection_fn(x_in)
        o_store = flash_attention_core_store_fn(q_store, k_store, v_store, "O_attn_full")
        o_proj_store = linear_store_fn(o_store, wo_sub, hidden_size, "O_proj_full")
        o_proj = prog.load_batch(o_proj_store, name="O_proj_load_full")
        x_res = prog.load_batch(x_in, name="X_attn_res_full")
        x_attn = add_inplace_fn(o_proj, x_res, hidden_size // mlen)
        x_attn = prog.norm(x_attn, mode="rms", eps_offset=3, reci_hid_offset=4)
        return prog.store(x_attn, name="X_attn_full")

    @prog.function
    def forward_block_fn(x_in):
        f1_store = linear_store_fn(x_in, w1_sub, inter_size, "FFN1_full")
        f2_store = linear_store_fn(f1_store, w2_sub, hidden_size, "FFN2_full")
        f2 = prog.load_batch(f2_store, name="FFN2_load_full")
        x_res = prog.load_batch(x_in, name="X_ffn_res_full")
        x_out = add_inplace_fn(f2, x_res, hidden_size // mlen)
        x_out = prog.norm(x_out, mode="rms", eps_offset=3, reci_hid_offset=4)
        return prog.store(x_out, name="X_out_full")

    @prog.function
    def merged_attention_fn(x_in):
        q_store, k_store, v_store = qkv_projection_fn(x_in)
        o_store = flash_attention_core_store_fn(q_store, k_store, v_store, "O_attn")
        o_proj_store = linear_store_fn(o_store, wo_sub, hidden_size, "O_proj")
        o_out = prog.load_batch(o_proj_store, name="O_out")
        x_res = prog.load_batch(x_in, name="X_res_add")
        o_out = add_inplace_fn(o_out, x_res, hidden_size // mlen)
        o_out = prog.norm(
            o_out,
            mode="rms",
            eps_offset=3,
            reci_hid_offset=4,
        )
        return prog.store(o_out, name="O_final")

    @prog.function
    def transformer_block_fn(x_in):
        x_attn_store = merged_attention_fn(x_in)
        x_out_store = forward_block_fn(x_attn_store)
        return x_out_store

    x_out_store = transformer_block_fn(x_input)
    x_out = prog.load_batch(x_out_store, name="X_out_load")
    prog.result(x_out)

    gen_code = prog.compile()
    print(f"Generated ISA lines: {len(gen_code.splitlines())}")

    # -------------------------------------------------------------------------
    # Sim env
    # -------------------------------------------------------------------------
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "X": X.reshape(1, -1),
        "Wq": Wq.reshape(1, -1),
        "Wk": Wk.reshape(1, -1),
        "Wv": Wv.reshape(1, -1),
        "Wo": Wo.reshape(1, -1),
        "W1": W1.reshape(1, -1),
        "W2": W2.reshape(1, -1),
    }
    golden_result = {"original_output": golden_out}

    fp_preload = [0.0, qk_scale, float("-inf"), norm_eps, norm_reci_hid] + [0.0] * 5
    create_sim_env(input_tensor, gen_code, golden_result, fp_preload=fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="transformer_block_function_test",
        data=None,
        specified_data_order=["X", "Wq", "Wk", "Wv", "Wo", "W1", "W2"],
        build_path=build_dir,
    )

    symbol_table = prog._compiler.symbol_table.table
    out_info = symbol_table[x_out.name]
    comparison_params = {
        "start_row_idx": out_info.vram_addr // mlen,
        "num_rows": (seq_len * hidden_size) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": True,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"Build dir: {build_dir}")
    print(f"Result location: row {out_info.vram_addr // mlen}, {(seq_len * hidden_size) // mlen} rows")
    print(f"Comparison params: {comparison_params}")
