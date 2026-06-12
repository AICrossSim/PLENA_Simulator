"""COMPILE-ONLY single_stream_block: 只编译出每个 kernel 的 .isa, 不算 golden,
不写大 HBM bin, 不跑 emulator, 不比对。产物在 <build_root>/ir/<kernel>/<kernel>.isa,
可直接喂 analytic model (tools/power/plena_isa_energy.py)。

复用 _validate_block.py 的 graph (tensors + nodes) 定义, 但:
  - 不构建 golden chain (torch 全精度 + 每 hop MX roundtrip — 那段最慢且这里无用)
  - data=None / compare=None (compile_kernel 只需 spec+kwargs+地址, 不需数值)
  - Manager(compile_only=True): run_pipeline 每步 compile+assemble 即停

Run: tools/manager/run.sh _compile_block
     tools/manager/run.sh _compile_block <build_root>            # 自定义输出目录
     tools/manager/run.sh _compile_block <build_root> <toml>     # 指定 settings toml
       e.g. tools/manager/run.sh _compile_block managerbuild_SSB_fresh plena_settings.SSB_build_1.toml

注意: load_behavior_settings 永远读 toml 的 [BEHAVIOR] 段 (与 emulator 一致)。
SSB_build_1.toml 的 BEHAVIOR 段是大几何 (MLEN=1024); 默认 plena_settings.toml
的 BEHAVIOR 段是小几何 (MLEN=64)。要大几何就传 SSB_build_1.toml。
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager


# ============================================================================
# 真实 Open-Sora v2 维度 (padding 到 MLEN=1024 的倍数, mlp_ratio=4)
#   joint seq  8828 → 9216 (NSB=9)
#   hidden     3072       (HD/MLEN=3, HEAD=HD/HLEN=24)
#   mlp_ratio  4          (mlp hidden = 4*HD = 12288)
# 这些是模型维度, 不再从 toml 的 lane_count 派生 (toml 只给 MLEN/HLEN 等 HW)。
# ============================================================================
NSB_SINGLE = 9          # joint seq blocks: S = 9*1024 = 9216 (真实 8828 pad)
HD_REAL    = 3072       # hidden
MLP_RATIO  = 4


def build_graph(s):
    """真实 single_stream_block 维度 (padding + mlp_ratio=4) → graph dict。"""
    MLEN, HLEN = s.mlen, s.hlen
    NSB = NSB_SINGLE
    S = NSB * MLEN          # 9216
    HD = HD_REAL           # 3072
    HEAD = HD // HLEN      # 24
    MHD = MLP_RATIO * HD   # 12288  (mlp hidden)
    MHEAD = MHD // HLEN    # 96     (mlp head_count)
    CATD = HD + MHD        # 15360  (concat attn[HD] + mlp[4HD] = 5HD)
    EPS = 1e-6
    HALF = HLEN // 2
    assert HEAD % s.hardware_lane_count == 0, f"HEAD {HEAD} % lane {s.hardware_lane_count}"

    K = "tilelang_tvm_compiler.kernels."
    LN, MOD = K+"layernorm_min:make_layernorm_min", K+"modulate_min:make_modulate_min"
    LIN, GELU = K+"linear_min:make_linear_min", K+"gelu_min:make_gelu_min"
    RMS, ROPE = K+"rmsnorm_min:make_rmsnorm_min", K+"rope_min:make_rope_min"
    FA, CC = K+"flash_attention_min:make_flash_attention_min", K+"concat_min:make_concat_min"
    RG = K+"residual_gate_min:make_residual_gate_min"

    hd1 = [1, S, 1, HD]
    bshd = [1, S, HEAD, HLEN]
    mlp_hd1 = [1, S, 1, MHD]          # mlp hidden, flat
    mlp_bshd = [1, S, MHEAD, HLEN]    # mlp hidden, head-packed
    wKN = [1, HD, 1, HD]
    wMLP = [1, MHD, 1, HD]            # mlp_in weight: (4HD, HD)
    biasMN = [1, S, 1, HD]
    biasMLP = [1, S, 1, MHD]

    lin_kw  = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": HD//MLEN, "with_bias": True}
    mlp_kw  = {"m_blocks": S//MLEN, "n_blocks": MHD//MLEN, "k_blocks": HD//MLEN, "with_bias": True}  # H -> 4H
    lin2_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": CATD//MLEN, "with_bias": True} # 5H -> H
    norm_geo = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "num_s_blocks": NSB, "batch": 1}
    gelu_kw = {"rows": MLEN, "hlen": HLEN, "head_count": MHEAD, "num_s_blocks": NSB, "batch": 1}  # on 4HD
    rope_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "half_dim": HALF,
               "num_s_blocks": NSB, "batch": 1}
    fa_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "active_lane": 2,
             "num_kv_blocks": NSB, "num_q_blocks": NSB}
    cc_kw = {"rows": MLEN, "a_dim": HD, "b_dim": MHD, "num_s_blocks": NSB, "batch": 1}  # attn[HD] + gelu[4HD]

    def t(shape, role): return {"shape": shape, "role": role}
    tensors = {
        "X": t(hd1, "io"),
        "LN_W_S": t(hd1, "weight"), "LN_W_B": t(hd1, "weight"),
        "LN_Y": t(bshd, "activation"),
        "MOD_S": t(bshd, "weight"), "MOD_H": t(bshd, "weight"),
        "MOD_Y": t(hd1, "activation"),
        "WQ": t(wKN,"weight"),"BQ": t(biasMN,"weight"),
        "WK": t(wKN,"weight"),"BK": t(biasMN,"weight"),
        "WV": t(wKN,"weight"),"BV": t(biasMN,"weight"),
        "WM": t(wMLP,"weight"),"BM": t(biasMLP,"weight"),   # mlp_in: H->4H
        "Q": t(bshd,"activation"),"Kk": t(bshd,"activation"),
        "V": t(bshd,"activation"),"MLP": t(mlp_bshd,"activation"),   # 4HD
        "QKN_W": t(bshd,"weight"),
        "QKN_Q": t(hd1,"activation"),"QKN_K": t(hd1,"activation"),
        "COS": t(hd1,"weight"),"SGN": t(hd1,"weight"),"P": t([1,MLEN,1,MLEN],"weight"),
        "ROPE_Q": t(bshd,"activation"),"ROPE_K": t(bshd,"activation"),
        "ATTN": t(hd1,"activation"),
        "GELU": t(mlp_hd1,"activation"),                    # 4HD
        "CONCAT": t([1,S,1,CATD],"activation"),             # 5HD
        "W2": t([1,HD,1,CATD],"weight"),"B2": t(biasMN,"weight"),   # 5H -> H
        "LIN2": t(bshd,"activation"),
        "GATE": t(bshd,"weight"),
        "BLOCK_OUT": t(bshd,"io"),
    }
    nodes = [
        {"name":"layernorm","kernel":LN,"kwargs":{"rows":MLEN,"hidden_size":HD,"num_s_blocks":NSB,"batch":1,"eps":EPS},
         "in":{"X_hbm":"X","SCALE_hbm":"LN_W_S","BIAS_hbm":"LN_W_B"},"out":{"Y_hbm":"LN_Y"}},
        {"name":"modulate","kernel":MOD,"kwargs":norm_geo,
         "in":{"X_hbm":"LN_Y","SCALE1P_hbm":"MOD_S","SHIFT_hbm":"MOD_H"},"out":{"Y_hbm":"MOD_Y"}},
        {"name":"linear_q","kernel":LIN,"kwargs":lin_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WQ","BIAS_hbm":"BQ"},"out":{"C_hbm":"Q"}},
        {"name":"linear_k","kernel":LIN,"kwargs":lin_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WK","BIAS_hbm":"BK"},"out":{"C_hbm":"Kk"}},
        {"name":"linear_v","kernel":LIN,"kwargs":lin_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WV","BIAS_hbm":"BV"},"out":{"C_hbm":"V"}},
        {"name":"linear_mlp","kernel":LIN,"kwargs":mlp_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WM","BIAS_hbm":"BM"},"out":{"C_hbm":"MLP"}},
        {"name":"qknorm_q","kernel":RMS,"kwargs":norm_geo,"in":{"X_hbm":"Q","SCALE_hbm":"QKN_W"},"out":{"Y_hbm":"QKN_Q"}},
        {"name":"qknorm_k","kernel":RMS,"kwargs":norm_geo,"in":{"X_hbm":"Kk","SCALE_hbm":"QKN_W"},"out":{"Y_hbm":"QKN_K"}},
        {"name":"rope_q","kernel":ROPE,"kwargs":rope_kw,"in":{"XQ_hbm":"QKN_Q","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"},"out":{"Q_OUT_hbm":"ROPE_Q"}},
        {"name":"rope_k","kernel":ROPE,"kwargs":rope_kw,"in":{"XQ_hbm":"QKN_K","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"},"out":{"Q_OUT_hbm":"ROPE_K"}},
        {"name":"gelu","kernel":GELU,"kwargs":gelu_kw,"in":{"X_hbm":"MLP"},"out":{"Y_hbm":"GELU"}},
        {"name":"flash_attention","kernel":FA,"kwargs":fa_kw,"in":{"Q_hbm":"ROPE_Q","K_hbm":"ROPE_K","V_hbm":"V"},"out":{"O_hbm":"ATTN"}},
        {"name":"concat","kernel":CC,"kwargs":cc_kw,"in":{"A_hbm":"ATTN","B_hbm":"GELU"},"out":{"Y_hbm":"CONCAT"}},
        {"name":"linear2","kernel":LIN,"kwargs":lin2_kw,"in":{"A_hbm":"CONCAT","B_hbm":"W2","BIAS_hbm":"B2"},"out":{"C_hbm":"LIN2"}},
        {"name":"residual_gate","kernel":RG,"kwargs":norm_geo,"in":{"X_hbm":"X","GATE_hbm":"GATE","Y_hbm":"LIN2"},"out":{"OUT_hbm":"BLOCK_OUT"}},
    ]
    return {"tensors": tensors, "nodes": nodes}


def main() -> int:
    import time as _time
    build_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    toml_path = sys.argv[2] if len(sys.argv) > 2 else None
    s = load_behavior_settings(toml_path)
    print(f"[compile-only] settings toml: {toml_path or 'plena_settings.toml (default)'} "
          f"| MLEN={s.mlen} HLEN={s.hlen} BLEN={s.blen} VLEN={s.vlen}")
    graph = build_graph(s)

    mgr = Manager(settings=s, build_root=build_root, compile_only=True)
    print(f"[compile-only] single_stream_block -> {mgr.ir_dir}")
    _t0 = _time.time()
    out = mgr.run_graph(graph, data=None, compare=None)
    _wall = _time.time() - _t0

    print(f"\n[compile-only] done: {len(out['results'])} kernels compiled in {_wall:.1f}s")
    print(f"  ISA at: {mgr.ir_dir}/<kernel>/<kernel>.isa")
    for name, r in out["results"].items():
        isa = mgr.ir_dir / name / f"{name}.isa"
        tag = "ok" if isa.exists() else "MISSING"
        print(f"  {tag:>8}  {name}: {isa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
