"""COMPILE-ONLY double_stream_block (real Open-Sora v2 dims, ASYMMETRIC streams).

Only emits each kernel's .isa (no golden, no emulator, no compare). Two streams
(image + text) with INDEPENDENT weights; pre-attn per stream -> s_concat to one
joint sequence -> single flash_attention -> s_split -> post-attn per stream.

Real dims (padded to MLEN=1024 multiples, mlp_ratio=4):
  image stream  : 8316 -> 9216  (NSB_img = 9)
  text  stream  :  512 -> 1024  (NSB_txt = 1)
  joint (concat): 10240          (NSB_joint = 10)
  hidden        : 3072           (HEAD = 24)
  mlp_ratio     : 4              (mlp hidden = 4*HD = 12288)

Run: tools/manager/run.sh _compile_double_block [build_root] [toml]
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager

# real dims
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
    CATD = HD + MHD              # 15360 (concat attn[HD] + mlp[4HD])
    EPS = 1e-6
    HALF = HLEN // 2
    NSBJ = NSB_IMG + NSB_TXT     # joint = 10
    assert HEAD % s.hardware_lane_count == 0

    K = "tilelang_tvm_compiler.kernels."
    LN, MOD = K+"layernorm_min:make_layernorm_min", K+"modulate_min:make_modulate_min"
    LIN, GELU = K+"linear_min:make_linear_min", K+"gelu_min:make_gelu_min"
    RMS, ROPE = K+"rmsnorm_min:make_rmsnorm_min", K+"rope_min:make_rope_min"
    FA = K+"flash_attention_min:make_flash_attention_min"
    SC, SS = K+"s_concat_min:make_s_concat_min", K+"s_split_min:make_s_split_min"
    RG = K+"residual_gate_min:make_residual_gate_min"

    tensors, nodes = {}, []
    def add_t(name, shape, role): tensors[name] = {"shape": shape, "role": role}
    def add_n(name, kernel, kwargs, ins, outs):
        nodes.append({"name": name, "kernel": kernel, "kwargs": kwargs, "in": ins, "out": outs})

    # shared rope tables
    add_t("COS", [1, MLEN, 1, HD], "weight")   # per-stream rope uses its own S; tables sized per stream below
    add_t("SGN", [1, MLEN, 1, HD], "weight")
    add_t("P", [1, MLEN, 1, MLEN], "weight")

    def stream(p, NSB):
        """Build one stream (prefix p, its own NSB). Returns rope_q/k, v names."""
        S = NSB * MLEN
        hd1 = [1, S, 1, HD]; bshd = [1, S, HEAD, HLEN]
        mlp_hd1 = [1, S, 1, MHD]; mlp_bshd = [1, S, MHEAD, HLEN]
        wKN = [1, HD, 1, HD]; wMLP = [1, MHD, 1, HD]
        biasMN = [1, S, 1, HD]; biasMLP = [1, S, 1, MHD]
        lin_kw  = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": HD//MLEN, "with_bias": True}
        mlp_kw  = {"m_blocks": S//MLEN, "n_blocks": MHD//MLEN, "k_blocks": HD//MLEN, "with_bias": True}
        lin2_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN,  "k_blocks": CATD//MLEN, "with_bias": True}
        norm_geo = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "num_s_blocks": NSB, "batch": 1}
        gelu_kw = {"rows": MLEN, "hlen": HLEN, "head_count": MHEAD, "num_s_blocks": NSB, "batch": 1}
        rope_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "half_dim": HALF,
                   "num_s_blocks": NSB, "batch": 1}

        # --- IO + pre-attn weights/acts ---
        add_t(f"{p}_X", hd1, "io")
        add_t(f"{p}_LNS", hd1, "weight"); add_t(f"{p}_LNB", hd1, "weight")
        add_t(f"{p}_MOD1S", bshd, "weight"); add_t(f"{p}_MOD1H", bshd, "weight")
        add_t(f"{p}_WQ", wKN, "weight"); add_t(f"{p}_BQ", biasMN, "weight")
        add_t(f"{p}_WK", wKN, "weight"); add_t(f"{p}_BK", biasMN, "weight")
        add_t(f"{p}_WV", wKN, "weight"); add_t(f"{p}_BV", biasMN, "weight")
        add_t(f"{p}_QKN", bshd, "weight")
        add_t(f"{p}_LN1", bshd, "activation"); add_t(f"{p}_MOD1", hd1, "activation")
        add_t(f"{p}_Q", bshd, "activation"); add_t(f"{p}_K", bshd, "activation"); add_t(f"{p}_V", bshd, "activation")
        add_t(f"{p}_QKNQ", hd1, "activation"); add_t(f"{p}_QKNK", hd1, "activation")
        add_t(f"{p}_RQ", bshd, "activation"); add_t(f"{p}_RK", bshd, "activation")
        add_n(f"{p}_norm1", LN, {"rows":MLEN,"hidden_size":HD,"num_s_blocks":NSB,"batch":1,"eps":EPS},
              {"X_hbm":f"{p}_X","SCALE_hbm":f"{p}_LNS","BIAS_hbm":f"{p}_LNB"}, {"Y_hbm":f"{p}_LN1"})
        add_n(f"{p}_mod1", MOD, norm_geo,
              {"X_hbm":f"{p}_LN1","SCALE1P_hbm":f"{p}_MOD1S","SHIFT_hbm":f"{p}_MOD1H"}, {"Y_hbm":f"{p}_MOD1"})
        add_n(f"{p}_linq", LIN, lin_kw, {"A_hbm":f"{p}_MOD1","B_hbm":f"{p}_WQ","BIAS_hbm":f"{p}_BQ"}, {"C_hbm":f"{p}_Q"})
        add_n(f"{p}_link", LIN, lin_kw, {"A_hbm":f"{p}_MOD1","B_hbm":f"{p}_WK","BIAS_hbm":f"{p}_BK"}, {"C_hbm":f"{p}_K"})
        add_n(f"{p}_linv", LIN, lin_kw, {"A_hbm":f"{p}_MOD1","B_hbm":f"{p}_WV","BIAS_hbm":f"{p}_BV"}, {"C_hbm":f"{p}_V"})
        add_n(f"{p}_qknq", RMS, norm_geo, {"X_hbm":f"{p}_Q","SCALE_hbm":f"{p}_QKN"}, {"Y_hbm":f"{p}_QKNQ"})
        add_n(f"{p}_qknk", RMS, norm_geo, {"X_hbm":f"{p}_K","SCALE_hbm":f"{p}_QKN"}, {"Y_hbm":f"{p}_QKNK"})
        add_n(f"{p}_ropeq", ROPE, rope_kw, {"XQ_hbm":f"{p}_QKNQ","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"}, {"Q_OUT_hbm":f"{p}_RQ"})
        add_n(f"{p}_ropek", ROPE, rope_kw, {"XQ_hbm":f"{p}_QKNK","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"}, {"Q_OUT_hbm":f"{p}_RK"})
        return f"{p}_RQ", f"{p}_RK", f"{p}_V", (lin_kw, mlp_kw, lin2_kw, norm_geo, gelu_kw, S, hd1, bshd, mlp_hd1, mlp_bshd, wKN, wMLP, biasMN, biasMLP)

    def stream_post(p, attn_tensor, kws):
        lin_kw, mlp_kw, lin2_kw, norm_geo, gelu_kw, S, hd1, bshd, mlp_hd1, mlp_bshd, wKN, wMLP, biasMN, biasMLP = kws
        CAT = [1, S, 1, CATD]
        add_t(f"{p}_WPROJ", wKN, "weight"); add_t(f"{p}_BPROJ", biasMN, "weight")
        add_t(f"{p}_GATE1", bshd, "weight"); add_t(f"{p}_GATE2", bshd, "weight")
        add_t(f"{p}_LNS2", hd1, "weight"); add_t(f"{p}_LNB2", hd1, "weight")
        add_t(f"{p}_MOD2S", bshd, "weight"); add_t(f"{p}_MOD2H", bshd, "weight")
        # double 不融合 proj+mlp_out (那是 single 的 linear2 才做的); double 是
        # 两条独立 residual: res1 用 proj(HD->H), res2 用 mlp_out(4H->H)。
        mlpout_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN, "k_blocks": MHD//MLEN, "with_bias": True}  # 4H->H
        add_t(f"{p}_WMI", wMLP, "weight"); add_t(f"{p}_BMI", biasMLP, "weight")       # H->4H
        add_t(f"{p}_WMO", [1,HD,1,MHD], "weight"); add_t(f"{p}_BMO", biasMN, "weight") # 4H->H
        add_t(f"{p}_PROJ", bshd, "activation"); add_t(f"{p}_X1", bshd, "activation")
        add_t(f"{p}_LN2", bshd, "activation"); add_t(f"{p}_MOD2", hd1, "activation")
        add_t(f"{p}_MH", mlp_bshd, "activation"); add_t(f"{p}_GELU", mlp_hd1, "activation")
        add_t(f"{p}_MLP", bshd, "activation"); add_t(f"{p}_OUT", bshd, "io")
        # proj + residual1
        add_n(f"{p}_proj", LIN, lin_kw, {"A_hbm":attn_tensor,"B_hbm":f"{p}_WPROJ","BIAS_hbm":f"{p}_BPROJ"}, {"C_hbm":f"{p}_PROJ"})
        add_n(f"{p}_res1", RG, norm_geo, {"X_hbm":f"{p}_X","GATE_hbm":f"{p}_GATE1","Y_hbm":f"{p}_PROJ"}, {"OUT_hbm":f"{p}_X1"})
        # norm2 -> mod2 -> mlp_in(H->4H) -> gelu -> mlp_out(4H->H) -> residual2
        add_n(f"{p}_norm2", LN, {"rows":MLEN,"hidden_size":HD,"num_s_blocks":norm_geo["num_s_blocks"],"batch":1,"eps":EPS},
              {"X_hbm":f"{p}_X1","SCALE_hbm":f"{p}_LNS2","BIAS_hbm":f"{p}_LNB2"}, {"Y_hbm":f"{p}_LN2"})
        add_n(f"{p}_mod2", MOD, norm_geo, {"X_hbm":f"{p}_LN2","SCALE1P_hbm":f"{p}_MOD2S","SHIFT_hbm":f"{p}_MOD2H"}, {"Y_hbm":f"{p}_MOD2"})
        add_n(f"{p}_mlpin", LIN, mlp_kw, {"A_hbm":f"{p}_MOD2","B_hbm":f"{p}_WMI","BIAS_hbm":f"{p}_BMI"}, {"C_hbm":f"{p}_MH"})
        add_n(f"{p}_gelu", GELU, gelu_kw, {"X_hbm":f"{p}_MH"}, {"Y_hbm":f"{p}_GELU"})
        add_n(f"{p}_mlpout", LIN, mlpout_kw, {"A_hbm":f"{p}_GELU","B_hbm":f"{p}_WMO","BIAS_hbm":f"{p}_BMO"}, {"C_hbm":f"{p}_MLP"})
        add_n(f"{p}_res2", RG, norm_geo, {"X_hbm":f"{p}_X1","GATE_hbm":f"{p}_GATE2","Y_hbm":f"{p}_MLP"}, {"OUT_hbm":f"{p}_OUT"})

    iq, ik, iv, ikws = stream("I", NSB_IMG)
    tq, tk, tv, tkws = stream("T", NSB_TXT)

    # joint: s_concat (txt first, img second) -> flash -> s_split
    SJ = NSBJ * MLEN
    hd1j = [1, SJ, 1, HD]; bshdj = [1, SJ, HEAD, HLEN]
    sc_kw = {"hd": HD, "num_s_blocks_a": NSB_TXT, "num_s_blocks_b": NSB_IMG, "batch": 1}
    fa_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "active_lane": 2,
             "num_kv_blocks": NSBJ, "num_q_blocks": NSBJ}
    add_t("QJ", hd1j, "activation"); add_t("KJ", hd1j, "activation"); add_t("VJ", hd1j, "activation")
    add_t("ATTNJ", bshdj, "activation")
    add_t("T_ATTN", [1, NSB_TXT*MLEN, 1, HD], "activation")
    add_t("I_ATTN", [1, NSB_IMG*MLEN, 1, HD], "activation")
    add_n("concat_q", SC, sc_kw, {"A_hbm":tq, "B_hbm":iq}, {"Y_hbm":"QJ"})
    add_n("concat_k", SC, sc_kw, {"A_hbm":tk, "B_hbm":ik}, {"Y_hbm":"KJ"})
    add_n("concat_v", SC, sc_kw, {"A_hbm":tv, "B_hbm":iv}, {"Y_hbm":"VJ"})
    add_n("flash", FA, fa_kw, {"Q_hbm":"QJ","K_hbm":"KJ","V_hbm":"VJ"}, {"O_hbm":"ATTNJ"})
    add_n("split_attn", SS, sc_kw, {"X_hbm":"ATTNJ"}, {"A_hbm":"T_ATTN","B_hbm":"I_ATTN"})

    stream_post("I", "I_ATTN", ikws)
    stream_post("T", "T_ATTN", tkws)

    return {"tensors": tensors, "nodes": nodes}


def main() -> int:
    import time as _time
    build_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    toml_path = sys.argv[2] if len(sys.argv) > 2 else None
    s = load_behavior_settings(toml_path)
    print(f"[compile-only] double_stream_block | MLEN={s.mlen} HLEN={s.hlen} BLEN={s.blen} VLEN={s.vlen}")
    print(f"  img NSB={NSB_IMG} txt NSB={NSB_TXT} joint NSB={NSB_IMG+NSB_TXT} HD={HD_REAL} mlp_ratio={MLP_RATIO}")
    graph = build_graph(s)
    mgr = Manager(settings=s, build_root=build_root, compile_only=True)
    _t0 = _time.time()
    out = mgr.run_graph(graph, data=None, compare=None)
    _wall = _time.time() - _t0
    print(f"\n[compile-only] done: {len(out['results'])} kernels in {_wall:.1f}s")
    for name, r in out["results"].items():
        isa = mgr.ir_dir / name / f"{name}.isa"
        print(f"  {'ok' if isa.exists() else 'MISSING':>8}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
