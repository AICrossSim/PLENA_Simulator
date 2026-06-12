"""COMPILE-ONLY final_layer (MMDiT LastLayer): 只编译每个 kernel 的 .isa。

Open-Sora MMDiT final_layer = LastLayer(hidden_size=3072, patch_size=1, out_channels=64):
    shift, scale = adaLN_modulation(vec).chunk(2)      # SiLU + Linear(HD -> 2*HD), on vec only
    x = (1 + scale) * norm_final(x) + shift            # LayerNorm(no affine) + modulate
    x = linear(x)                                       # Linear(HD -> patch^2 * out_ch = 64)

PLENA kernels:
  1. modulation_gen_min ×2  — silu(vec) @ W -> shift / scale  (vec is one (1,HD) row)
  2. layernorm_min          — norm_final (no affine -> scale=1, bias=0 constants)
  3. modulate_min           — (1+scale)*x + shift
  4. linear_min             — x[S,HD] @ W[Npad,HD], out 64 padded to MLEN (n_blocks=1)

Run: tools/manager/run.sh _compile_final_layer <build_root> <toml>
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager

NSB_SINGLE = 9          # joint seq blocks: S = 9*1024 = 9216 (真实 8828 pad)
HD_REAL    = 3072       # hidden
OUT_DIM    = 64         # patch^2 * out_channels = 1*64; padded to MLEN for linear


def build_graph(s):
    MLEN, HLEN = s.mlen, s.hlen
    NSB = NSB_SINGLE
    S = NSB * MLEN          # 9216
    HD = HD_REAL           # 3072
    HEAD = HD // HLEN      # 24
    NOUT = MLEN            # out 64 padded to one MLEN block (n_blocks=1)
    EPS = 1e-6
    assert HEAD % s.hardware_lane_count == 0

    K = "tilelang_tvm_compiler.kernels."
    MODGEN = K+"modulation_gen_min:make_modulation_gen_min"
    LN  = K+"layernorm_min:make_layernorm_min"
    MOD = K+"modulate_min:make_modulate_min"
    LIN = K+"linear_min:make_linear_min"

    hd1  = [1, S, 1, HD]
    bshd = [1, S, HEAD, HLEN]
    vec1 = [1, MLEN, 1, HD]          # vec staged as one MLEN tile (1 real row)
    wMod = [1, HD, 1, HD]            # adaLN linear weight (HD -> HD per chunk)
    modbc = [1, S, 1, HD]            # shift/scale broadcast over SEQ
    wLIN = [1, NOUT, 1, HD]          # final linear weight (NOUT, HD)
    biasLIN = [1, S, 1, NOUT]
    outT = [1, S, 1, NOUT]

    modgen_kw = {"hd": HD, "seq": S}
    norm_geo = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "num_s_blocks": NSB, "batch": 1}
    lin_kw = {"m_blocks": S//MLEN, "n_blocks": NOUT//MLEN, "k_blocks": HD//MLEN, "with_bias": True}

    def t(shape, role): return {"shape": shape, "role": role}
    tensors = {
        "VEC": t(vec1, "io"),
        "W_SHIFT": t(wMod, "weight"), "W_SCALE": t(wMod, "weight"),
        "MOD_H": t(modbc, "activation"), "MOD_S": t(modbc, "activation"),
        "X": t(hd1, "io"),
        "LN_ONE": t(hd1, "weight"), "LN_ZERO": t(hd1, "weight"),  # no-affine: scale=1, bias=0
        "LN_Y": t(hd1, "activation"),
        "MOD_Y": t(bshd, "activation"),
        "WL": t(wLIN, "weight"), "BL": t(biasLIN, "weight"),
        "OUT": t(outT, "io"),
    }
    nodes = [
        {"name":"adaln_shift","kernel":MODGEN,"kwargs":modgen_kw,
         "in":{"VEC_hbm":"VEC","W_SHIFT":"W_SHIFT"},"out":{"SHIFT_hbm":"MOD_H"}},
        {"name":"adaln_scale","kernel":MODGEN,"kwargs":modgen_kw,
         "in":{"VEC_hbm":"VEC","W_SHIFT":"W_SCALE"},"out":{"SHIFT_hbm":"MOD_S"}},
        {"name":"norm_final","kernel":LN,"kwargs":{"rows":MLEN,"hidden_size":HD,"num_s_blocks":NSB,"batch":1,"eps":EPS},
         "in":{"X_hbm":"X","SCALE_hbm":"LN_ONE","BIAS_hbm":"LN_ZERO"},"out":{"Y_hbm":"LN_Y"}},
        {"name":"modulate","kernel":MOD,"kwargs":norm_geo,
         "in":{"X_hbm":"LN_Y","SCALE1P_hbm":"MOD_S","SHIFT_hbm":"MOD_H"},"out":{"Y_hbm":"MOD_Y"}},
        {"name":"linear_out","kernel":LIN,"kwargs":lin_kw,
         "in":{"A_hbm":"MOD_Y","B_hbm":"WL","BIAS_hbm":"BL"},"out":{"C_hbm":"OUT"}},
    ]
    return {"tensors": tensors, "nodes": nodes}


def main() -> int:
    import time as _time
    build_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    toml_path = sys.argv[2] if len(sys.argv) > 2 else None
    s = load_behavior_settings(toml_path)
    print(f"[compile-only] settings: {toml_path or 'default'} | MLEN={s.mlen} HLEN={s.hlen} BLEN={s.blen} VLEN={s.vlen}")
    graph = build_graph(s)
    mgr = Manager(settings=s, build_root=build_root, compile_only=True)
    print(f"[compile-only] final_layer -> {mgr.ir_dir}")
    _t0 = _time.time()
    out = mgr.run_graph(graph, data=None, compare=None)
    print(f"\n[compile-only] done: {len(out['results'])} kernels in {_time.time()-_t0:.1f}s")
    for name, r in out["results"].items():
        isa = mgr.ir_dir / name / f"{name}.isa"
        print(f"  {'ok' if isa.exists() else 'MISSING':>8}  {name}: {isa}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
