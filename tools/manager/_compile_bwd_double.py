"""COMPILE-ONLY full DoubleStreamBlock (DSB) BACKWARD chain.

DSB forward = two parallel streams (image S=9216, text S=1024), each with
its own norm/modulate/linear-qkv/qknorm/rope/mlp/proj/res, whose q/k/v are
concatenated into one joint sequence (10240), run through a SINGLE flash
attention, then split back per stream.

Reuses the SSB backward kernels (linear dx/dw, flash bwd,
rope/gelu/rmsnorm/layernorm/affine); only the graph differs — each
operator's backward appears once per stream, plus one joint flash-attention
backward. concat/split backward (= split/concat) is ~free and omitted.

Address note: this is compile-only (every kernel is compiled independently;
no real dataflow). Kernel INPUTS are therefore drawn from a small set of
SHARED, read-only buffer pools (one per (stream, shape)); only OUTPUTS get
unique tensors. This keeps the total HBM tensor count — and the bump
allocator's address range — well under the int32 (2^31 byte) limit.

Run:
  PLENA_ALLOC_MODE=stable tools/manager/run.sh _compile_bwd_double managerbuild_bwd_double
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager

NSB_IMG = 9
NSB_TXT = 1
HD_REAL = 3072
MLP_RATIO = 4


def build_graph(s):
    MLEN, HLEN = s.mlen, s.hlen
    HD = HD_REAL
    HEAD = HD // HLEN            # 24
    MHD = MLP_RATIO * HD         # 12288
    MHEAD = MHD // HLEN          # 96
    CATD = HD + MHD              # 15360
    NSBJ = NSB_IMG + NSB_TXT     # 10

    K = "tilelang_tvm_compiler.kernels."
    LDX = K + "linear_bwd_min:make_linear_bwd_dx_min"
    LDW = K + "linear_bwd_min:make_linear_bwd_dw_min"
    FA  = K + "flash_attention_bwd_min:make_flash_attention_bwd_min"
    ROPE = K + "rope_bwd_min:make_rope_bwd_min"
    RMS = K + "rmsnorm_bwd_min:make_rmsnorm_bwd_min"
    LN  = K + "layernorm_bwd_min:make_layernorm_bwd_min"
    GELU = K + "gelu_bwd_min:make_gelu_bwd_min"
    AFF = K + "affine_bwd_min:make_affine_bwd_min"

    tensors = {}
    nodes = []

    def pool(name, shape):
        """A shared read-only input buffer (declared once)."""
        if name not in tensors:
            tensors[name] = {"shape": shape, "role": "io"}
        return name

    def out(name, shape):
        """Shared output scratch. compile-only does no dataflow, so a kernel's
        output can REUSE any existing same-shape io pool (input or output) —
        the address may overlap an input, which is irrelevant since nothing
        runs. This caps the HBM tensor set at one buffer per distinct shape
        per stream, keeping the bump allocator under the int32 limit."""
        prefix = name.split("_", 1)[0]            # stream tag (I / T / J)
        sig = 'x'.join(str(d) for d in shape)
        # reuse any existing same-stream, same-shape pool
        for nm, sp in tensors.items():
            if nm.startswith(prefix) and sp["shape"] == list(shape):
                return nm
        key = f"{prefix}_o_{sig}"
        tensors[key] = {"shape": shape, "role": "io"}
        return key

    def node(name, kernel, kwargs, ins, outs):
        nodes.append({"name": name, "kernel": kernel, "kwargs": kwargs,
                      "in": ins, "out": outs})

    def stream(p, NSB):
        S = NSB * MLEN
        hd1  = [1, S, 1, HD]
        bshd = [1, S, HEAD, HLEN]
        mlp_hd1  = [1, S, 1, MHD]
        mlp_bshd = [1, S, MHEAD, HLEN]
        wKN  = [1, HD, 1, HD]
        wMLP = [1, MHD, 1, HD]
        w2   = [1, HD, 1, CATD]

        # --- shared read-only input pools (one each, reused by every node) ---
        dY_HD  = pool(f"{p}_dY_HD",  hd1)        # HD-wide upstream grad
        dY_MHD = pool(f"{p}_dY_MHD", mlp_hd1)    # 4HD-wide upstream grad
        X_HD   = pool(f"{p}_X_HD",   hd1)        # saved HD activation
        X_CAT  = pool(f"{p}_X_CAT",  [1,S,1,CATD])
        W_KN   = pool(f"{p}_W_KN",   wKN)        # any HD×HD weight (q/k/v)
        W_MLP  = pool(f"{p}_W_MLP",  wMLP)       # mlp weight
        W_2    = pool(f"{p}_W_2",    w2)         # proj weight
        bshd_X = pool(f"{p}_bshd",   bshd)       # any bshd activation (Q/K/dO/SC/FAC)
        COS    = pool(f"{p}_COS",    hd1)
        SGN    = pool(f"{p}_SGN",    hd1)
        Pm     = pool(f"{p}_P",      [1,MLEN,1,MLEN])
        gX     = pool(f"{p}_gX",     mlp_bshd)   # gelu fwd input / dY (4HD)

        lin_kw  = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": HD//MLEN}
        mlp_kw  = {"m_blocks": S//MLEN, "n_blocks": MHD//MLEN, "k_blocks": HD//MLEN}
        mlpout_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN, "k_blocks": MHD//MLEN}
        proj_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": CATD//MLEN}
        norm_geo = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "num_s_blocks": NSB}
        gelu_kw  = {"rows": MLEN, "hlen": HLEN, "head_count": MHEAD, "num_s_blocks": NSB}
        rope_kw  = {"hlen": HLEN, "head_count": HEAD, "half_dim": HLEN//2, "num_s_blocks": NSB}
        ln_kw    = {"hidden_size": HD, "num_s_blocks": NSB}

        P = p
        node(f"{P}_proj_dx",  LDX, proj_kw,  {"dY_hbm":dY_HD,"W_hbm":W_2},   {"dX_hbm":out(f"{P}_o_projdx",[1,S,1,CATD])})
        node(f"{P}_proj_dw",  LDW, proj_kw,  {"dY_hbm":dY_HD,"X_hbm":X_CAT}, {"dW_hbm":out(f"{P}_o_projdw",w2)})
        node(f"{P}_mlpout_dx",LDX, mlpout_kw,{"dY_hbm":dY_HD,"W_hbm":W_MLP}, {"dX_hbm":out(f"{P}_o_mlpodx",mlp_hd1)})
        node(f"{P}_mlpout_dw",LDW, mlpout_kw,{"dY_hbm":dY_HD,"X_hbm":X_HD},  {"dW_hbm":out(f"{P}_o_mlpodw",[1,HD,1,MHD])})
        node(f"{P}_mlpin_dx", LDX, mlp_kw,   {"dY_hbm":dY_MHD,"W_hbm":W_MLP},{"dX_hbm":out(f"{P}_o_mlpidx",hd1)})
        node(f"{P}_mlpin_dw", LDW, mlp_kw,   {"dY_hbm":dY_MHD,"X_hbm":X_HD}, {"dW_hbm":out(f"{P}_o_mlpidw",wMLP)})
        node(f"{P}_gelu",     GELU,gelu_kw,  {"X_hbm":gX,"dY_hbm":gX},       {"dX_hbm":out(f"{P}_o_gelu",mlp_bshd)})
        for q in ("q","k","v"):
            node(f"{P}_lin{q}_dx", LDX, lin_kw, {"dY_hbm":dY_HD,"W_hbm":W_KN}, {"dX_hbm":out(f"{P}_o_lin{q}dx",hd1)})
            node(f"{P}_lin{q}_dw", LDW, lin_kw, {"dY_hbm":dY_HD,"X_hbm":X_HD}, {"dW_hbm":out(f"{P}_o_lin{q}dw",wKN)})
        node(f"{P}_rope_q", ROPE, rope_kw, {"dOUT_hbm":dY_HD,"COS_hbm":COS,"SGN_SIN_hbm":SGN,"P_hbm":Pm}, {"dX_hbm":out(f"{P}_o_ropeq",hd1)})
        node(f"{P}_rope_k", ROPE, rope_kw, {"dOUT_hbm":dY_HD,"COS_hbm":COS,"SGN_SIN_hbm":SGN,"P_hbm":Pm}, {"dX_hbm":out(f"{P}_o_ropek",hd1)})
        node(f"{P}_qknorm_q", RMS, norm_geo, {"X_hbm":bshd_X,"SCALE_hbm":bshd_X,"dY_hbm":bshd_X}, {"dX_hbm":out(f"{P}_o_qknq",bshd)})
        node(f"{P}_qknorm_k", RMS, norm_geo, {"X_hbm":bshd_X,"SCALE_hbm":bshd_X,"dY_hbm":bshd_X}, {"dX_hbm":out(f"{P}_o_qknk",bshd)})
        node(f"{P}_norm1", LN, ln_kw, {"X_hbm":X_HD,"SCALE_hbm":X_HD,"dY_hbm":dY_HD}, {"dX_hbm":out(f"{P}_o_norm1",hd1)})
        node(f"{P}_norm2", LN, ln_kw, {"X_hbm":X_HD,"SCALE_hbm":X_HD,"dY_hbm":dY_HD}, {"dX_hbm":out(f"{P}_o_norm2",hd1)})
        node(f"{P}_mod1", AFF, norm_geo, {"dY_hbm":bshd_X,"FAC_hbm":bshd_X}, {"dX_hbm":out(f"{P}_o_mod1",bshd)})
        node(f"{P}_mod2", AFF, norm_geo, {"dY_hbm":bshd_X,"FAC_hbm":bshd_X}, {"dX_hbm":out(f"{P}_o_mod2",bshd)})
        node(f"{P}_res1", AFF, norm_geo, {"dY_hbm":bshd_X,"FAC_hbm":bshd_X}, {"dX_hbm":out(f"{P}_o_res1",bshd)})
        node(f"{P}_res2", AFF, norm_geo, {"dY_hbm":bshd_X,"FAC_hbm":bshd_X}, {"dX_hbm":out(f"{P}_o_res2",bshd)})

    which = build_graph._which     # img | txt | joint | all
    if which in ("img", "all"):
        stream("I", NSB_IMG)
    if which in ("txt", "all"):
        stream("T", NSB_TXT)
    if which in ("joint", "all"):
        bshdJ = [1, NSBJ*MLEN, HEAD, HLEN]
        jin = pool("J_in", bshdJ)
        fa_kw = {"hlen": HLEN, "head_count": HEAD, "num_kv_blocks": NSBJ, "num_q_blocks": NSBJ}
        node("flash_attention_bwd", FA, fa_kw,
             {"Q_hbm":jin,"K_hbm":jin,"V_hbm":jin,"dO_hbm":jin},
             {"dQ_hbm":out("J_dQ",bshdJ),"dK_hbm":out("J_dK",bshdJ),"dV_hbm":out("J_dV",bshdJ)})

    return {"tensors": tensors, "nodes": nodes}


build_graph._which = "img"


def main() -> int:
    import time as _time
    build_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    which = sys.argv[2] if len(sys.argv) > 2 else "img"
    toml_path = sys.argv[3] if len(sys.argv) > 3 else None
    build_graph._which = which
    s = load_behavior_settings(toml_path)
    print(f"[bwd-double:{which}] MLEN={s.mlen} HLEN={s.hlen} BLEN={s.blen}")
    graph = build_graph(s)
    mgr = Manager(settings=s, build_root=build_root, compile_only=True)
    print(f"[bwd-double] -> {mgr.ir_dir}  ({len(graph['nodes'])} kernels, {len(graph['tensors'])} tensors)")
    _t0 = _time.time()
    out_ = mgr.run_graph(graph, data=None, compare=None)
    print(f"\n[bwd-double] done: {len(out_['results'])} kernels in {_time.time()-_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
