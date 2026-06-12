"""COMPILE-ONLY full SingleStreamBlock BACKWARD chain.

Lowers the backward of every SSB kernel through the manager pipeline so
each .isa lands in <build_root>/ir/<kernel>/. Feeds the analytic model
(tools/power/plena_isa_energy.py); not the simulator. Correctness aside.

Backward kernel coverage (one per forward operator class):
  linear_q/k/v/mlp/2  -> linear_bwd_dx + linear_bwd_dw   (2 GEMMs each)
  flash_attention     -> flash_attention_bwd             (5 GEMMs)
  rope_q/k            -> rope_bwd                        (1 shuffle-GEMM)
  qknorm_q/k         -> rmsnorm_bwd                     (vector)
  layernorm          -> layernorm_bwd                   (vector)
  gelu               -> gelu_bwd                        (vector)
  modulate/res_gate  -> affine_bwd                      (vector)
  concat             -> (split, ~free; omitted)

Run:
  PLENA_ALLOC_MODE=stable tools/manager/run.sh _compile_bwd_chain managerbuild_bwd_chain
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager

NSB = 9
HD_REAL = 3072
MLP_RATIO = 4


def build_graph(s):
    MLEN, HLEN = s.mlen, s.hlen
    S = NSB * MLEN          # 9216
    HD = HD_REAL           # 3072
    HEAD = HD // HLEN      # 24
    MHD = MLP_RATIO * HD   # 12288
    MHEAD = MHD // HLEN    # 96
    CATD = HD + MHD        # 15360 (concat width, 5H)

    K = "tilelang_tvm_compiler.kernels."
    LDX = K + "linear_bwd_min:make_linear_bwd_dx_min"
    LDW = K + "linear_bwd_min:make_linear_bwd_dw_min"
    FA  = K + "flash_attention_bwd_min:make_flash_attention_bwd_min"
    ROPE = K + "rope_bwd_min:make_rope_bwd_min"
    RMS = K + "rmsnorm_bwd_min:make_rmsnorm_bwd_min"
    LN  = K + "layernorm_bwd_min:make_layernorm_bwd_min"
    GELU = K + "gelu_bwd_min:make_gelu_bwd_min"
    AFF = K + "affine_bwd_min:make_affine_bwd_min"

    bshd  = [1, S, HEAD, HLEN]
    hd1   = [1, S, 1, HD]
    mlp_bshd = [1, S, MHEAD, HLEN]
    wKN   = [1, HD, 1, HD]
    wMLP  = [1, MHD, 1, HD]
    wMLPT = [1, HD, 1, MHD]
    w2    = [1, HD, 1, CATD]

    def t(shape):
        return {"shape": shape, "role": "io"}

    # linear blocks: each = (m_blocks, n_blocks, k_blocks) of the FORWARD Y=X@W^T
    #   q/k/v : X[S,HD] @ W[HD,HD]^T -> [S,HD]    (M=S,N=HD,K=HD)
    #   mlp   : X[S,HD] @ W[4HD,HD]^T -> [S,4HD]  (M=S,N=4HD,K=HD)
    #   proj  : X[S,5HD]@ W[HD,5HD]^T -> [S,HD]   (M=S,N=HD,K=5HD)
    lin_kw  = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": HD//MLEN}
    mlp_kw  = {"m_blocks": S//MLEN, "n_blocks": MHD//MLEN, "k_blocks": HD//MLEN}
    proj_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": CATD//MLEN}
    norm_geo = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "num_s_blocks": NSB}
    gelu_kw  = {"rows": MLEN, "hlen": HLEN, "head_count": MHEAD, "num_s_blocks": NSB}
    rope_kw  = {"hlen": HLEN, "head_count": HEAD, "half_dim": HLEN//2, "num_s_blocks": NSB}
    ln_kw    = {"hidden_size": HD, "num_s_blocks": NSB}
    fa_kw    = {"hlen": HLEN, "head_count": HEAD, "num_kv_blocks": NSB, "num_q_blocks": NSB}

    tensors = {
        # generic grad/activation/weight pools (io role = bump-allocated HBM)
        "dY_S_HD": t(hd1), "dY_S_MHD": t([1,S,1,MHD]), "dY_S_CAT": t([1,S,1,CATD]),
        "X_S_HD": t(hd1), "X_S_CAT": t([1,S,1,CATD]),
        "Wq": t(wKN), "Wk": t(wKN), "Wv": t(wKN), "Wm": t(wMLP), "W2": t(w2),
        "dX_S_CAT": t([1,S,1,CATD]), "dX_mlpin": t(hd1),
        "dX_q": t(hd1), "dX_k": t(hd1), "dX_v": t(hd1),
        "dWq": t(wKN), "dWk": t(wKN), "dWv": t(wKN), "dWm": t(wMLP), "dW2": t(w2),
        # bshd grads for attention / rope / norms / affine
        "Q": t(bshd), "Kk": t(bshd), "V": t(bshd), "dO": t(bshd),
        "dQ": t(bshd), "dK": t(bshd), "dV": t(bshd),
        "COS": t(hd1), "SGN": t(hd1), "P": t([1,MLEN,1,MLEN]),
        "dROPEQ": t(hd1), "dROPEK": t(hd1),
        "SC": t(bshd), "FAC": t(bshd),
        "dX_bshd_a": t(bshd), "dX_bshd_b": t(bshd), "dX_bshd_c": t(bshd),
        "dX_bshd_d": t(bshd), "dX_bshd_e": t(bshd), "dX_bshd_f": t(bshd),
        "dX_mlp": t(mlp_bshd), "GELU_X": t(mlp_bshd), "GELU_dY": t(mlp_bshd),
        "dX_ln": t(hd1),
    }

    nodes = [
        # --- linear backward: each forward linear -> dX + dW ---
        {"name":"linear2_dx","kernel":LDX,"kwargs":proj_kw,"in":{"dY_hbm":"dY_S_HD","W_hbm":"W2"},"out":{"dX_hbm":"dX_S_CAT"}},
        {"name":"linear2_dw","kernel":LDW,"kwargs":proj_kw,"in":{"dY_hbm":"dY_S_HD","X_hbm":"X_S_CAT"},"out":{"dW_hbm":"dW2"}},
        {"name":"linear_mlp_dx","kernel":LDX,"kwargs":mlp_kw,"in":{"dY_hbm":"dY_S_MHD","W_hbm":"Wm"},"out":{"dX_hbm":"dX_mlpin"}},
        {"name":"linear_mlp_dw","kernel":LDW,"kwargs":mlp_kw,"in":{"dY_hbm":"dY_S_MHD","X_hbm":"X_S_HD"},"out":{"dW_hbm":"dWm"}},
        {"name":"linear_q_dx","kernel":LDX,"kwargs":lin_kw,"in":{"dY_hbm":"dY_S_HD","W_hbm":"Wq"},"out":{"dX_hbm":"dX_q"}},
        {"name":"linear_q_dw","kernel":LDW,"kwargs":lin_kw,"in":{"dY_hbm":"dY_S_HD","X_hbm":"X_S_HD"},"out":{"dW_hbm":"dWq"}},
        {"name":"linear_k_dx","kernel":LDX,"kwargs":lin_kw,"in":{"dY_hbm":"dY_S_HD","W_hbm":"Wk"},"out":{"dX_hbm":"dX_k"}},
        {"name":"linear_k_dw","kernel":LDW,"kwargs":lin_kw,"in":{"dY_hbm":"dY_S_HD","X_hbm":"X_S_HD"},"out":{"dW_hbm":"dWk"}},
        {"name":"linear_v_dx","kernel":LDX,"kwargs":lin_kw,"in":{"dY_hbm":"dY_S_HD","W_hbm":"Wv"},"out":{"dX_hbm":"dX_v"}},
        {"name":"linear_v_dw","kernel":LDW,"kwargs":lin_kw,"in":{"dY_hbm":"dY_S_HD","X_hbm":"X_S_HD"},"out":{"dW_hbm":"dWv"}},
        # --- attention backward ---
        {"name":"flash_attention_bwd","kernel":FA,"kwargs":fa_kw,
         "in":{"Q_hbm":"Q","K_hbm":"Kk","V_hbm":"V","dO_hbm":"dO"},
         "out":{"dQ_hbm":"dQ","dK_hbm":"dK","dV_hbm":"dV"}},
        # --- rope backward (q, k) ---
        {"name":"rope_q_bwd","kernel":ROPE,"kwargs":rope_kw,
         "in":{"dOUT_hbm":"dROPEQ","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"},"out":{"dX_hbm":"dX_bshd_a"}},
        {"name":"rope_k_bwd","kernel":ROPE,"kwargs":rope_kw,
         "in":{"dOUT_hbm":"dROPEK","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"},"out":{"dX_hbm":"dX_bshd_b"}},
        # --- norm backward ---
        {"name":"qknorm_q_bwd","kernel":RMS,"kwargs":norm_geo,
         "in":{"X_hbm":"Q","SCALE_hbm":"SC","dY_hbm":"dO"},"out":{"dX_hbm":"dX_bshd_c"}},
        {"name":"qknorm_k_bwd","kernel":RMS,"kwargs":norm_geo,
         "in":{"X_hbm":"Kk","SCALE_hbm":"SC","dY_hbm":"dO"},"out":{"dX_hbm":"dX_bshd_d"}},
        {"name":"layernorm_bwd","kernel":LN,"kwargs":ln_kw,
         "in":{"X_hbm":"X_S_HD","SCALE_hbm":"X_S_HD","dY_hbm":"dY_S_HD"},"out":{"dX_hbm":"dX_ln"}},
        # --- gelu backward (on 4HD) ---
        {"name":"gelu_bwd","kernel":GELU,"kwargs":gelu_kw,
         "in":{"X_hbm":"GELU_X","dY_hbm":"GELU_dY"},"out":{"dX_hbm":"dX_mlp"}},
        # --- affine backward (modulate, residual_gate) ---
        {"name":"modulate_bwd","kernel":AFF,"kwargs":norm_geo,
         "in":{"dY_hbm":"dO","FAC_hbm":"FAC"},"out":{"dX_hbm":"dX_bshd_e"}},
        {"name":"residual_gate_bwd","kernel":AFF,"kwargs":norm_geo,
         "in":{"dY_hbm":"dO","FAC_hbm":"FAC"},"out":{"dX_hbm":"dX_bshd_f"}},
    ]
    return {"tensors": tensors, "nodes": nodes}


def main() -> int:
    import time as _time
    build_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    toml_path = sys.argv[2] if len(sys.argv) > 2 else None
    s = load_behavior_settings(toml_path)
    print(f"[bwd-chain] toml: {toml_path or 'default'} | MLEN={s.mlen} HLEN={s.hlen} BLEN={s.blen}")
    graph = build_graph(s)
    mgr = Manager(settings=s, build_root=build_root, compile_only=True)
    print(f"[bwd-chain] -> {mgr.ir_dir}")
    _t0 = _time.time()
    out = mgr.run_graph(graph, data=None, compare=None)
    print(f"\n[bwd-chain] done: {len(out['results'])} kernels in {_time.time()-_t0:.1f}s")
    print(f"  ISA at: {mgr.ir_dir}/<kernel>/<kernel>.isa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
