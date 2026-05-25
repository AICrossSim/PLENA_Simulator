"""First REAL heterogeneous chain via the graph driver: layernorm -> modulate
(the head of single_stream_block_graph.svg).

  X --layernorm(scale,bias)--> LN_Y --modulate(scale1p,shift)--> MOD_Y

layernorm sees BSHD (1,S,1,H); modulate sees (1,S,head,hlen). H == head*hlen
== 128, so LN_Y is the same flat bytes both ways -> shared address -> bin relay.

Run:
  cd PLENA_Simulator
  LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib \
    PYTHONPATH=tools:compiler:transactional_emulator/testbench \
    ./.venv/bin/python3 tools/manager/_validate_ln_mod.py
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager
from _tb_cache import mx_roundtrip


def main() -> int:
    s = load_behavior_settings()
    BATCH = 1
    NUM_S_BLOCKS = 2
    ROWS = s.mlen
    HLEN = s.hlen
    HEAD_COUNT = 8
    H = HEAD_COUNT * HLEN              # 128 == layernorm hidden_size
    SEQ_LEN = NUM_S_BLOCKS * ROWS
    EPS = 1e-6
    assert H % s.mlen == 0, f"hidden_size {H} must be a multiple of mlen {s.mlen}"

    ln_shape  = [BATCH, SEQ_LEN, 1, H]                  # layernorm view
    mod_shape = [BATCH, SEQ_LEN, HEAD_COUNT, HLEN]      # modulate view (same bytes)

    LN  = "tilelang_tvm_compiler.kernels.layernorm_min:make_layernorm_min"
    MOD = "tilelang_tvm_compiler.kernels.modulate_min:make_modulate_min"
    ln_kw  = {"rows": ROWS, "hidden_size": H, "num_s_blocks": NUM_S_BLOCKS,
              "batch": BATCH, "eps": EPS}
    mod_kw = {"rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
              "num_s_blocks": NUM_S_BLOCKS, "batch": BATCH}

    torch.manual_seed(0)
    # --- layernorm inputs (weights expanded to (B,S,1,H), as the test does) ---
    x       = torch.randn(BATCH, SEQ_LEN, 1, H, dtype=torch.float32) * 0.5
    ln_sc   = torch.randn(H, dtype=torch.float32) * 0.3 + 1.0
    ln_bi   = torch.randn(H, dtype=torch.float32) * 0.1
    ln_sc_f = ln_sc.view(1, 1, 1, H).expand(BATCH, SEQ_LEN, 1, H).contiguous()
    ln_bi_f = ln_bi.view(1, 1, 1, H).expand(BATCH, SEQ_LEN, 1, H).contiguous()

    x_eff   = mx_roundtrip(x)
    sc_eff  = mx_roundtrip(ln_sc_f)
    bi_eff  = mx_roundtrip(ln_bi_f)
    mu      = x_eff.mean(dim=-1, keepdim=True)
    xc      = x_eff - mu
    var     = (xc * xc).mean(dim=-1, keepdim=True)
    ln_y    = xc * torch.rsqrt(var + EPS) * sc_eff + bi_eff
    ln_y    = mx_roundtrip(ln_y)                       # LN_Y golden (1,S,1,H)

    # --- modulate inputs: x = LN_Y (reshaped), plus its own scale/shift ---
    mod_sc   = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3
    mod_shft = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3
    s1p_eff  = mx_roundtrip((1.0 + mod_sc).to(torch.float32))
    shft_eff = mx_roundtrip(mod_shft)
    ln_y_mod = ln_y.reshape(BATCH, SEQ_LEN, HEAD_COUNT, HLEN)
    mod_y    = mx_roundtrip(s1p_eff * ln_y_mod + shft_eff)   # MOD_Y golden

    graph = {
        "tensors": {
            "X":        {"shape": ln_shape,  "role": "io"},
            "LN_SCALE": {"shape": ln_shape,  "role": "weight"},
            "LN_BIAS":  {"shape": ln_shape,  "role": "weight"},
            "LN_Y":     {"shape": mod_shape, "role": "activation"},  # shared
            "MOD_S1P":  {"shape": mod_shape, "role": "weight"},
            "MOD_SHIFT":{"shape": mod_shape, "role": "weight"},
            "MOD_Y":    {"shape": mod_shape, "role": "io"},
        },
        "nodes": [
            {"name": "layernorm", "kernel": LN, "kwargs": ln_kw,
             "in":  {"X_hbm": "X", "SCALE_hbm": "LN_SCALE", "BIAS_hbm": "LN_BIAS"},
             "out": {"Y_hbm": "LN_Y"}},
            {"name": "modulate", "kernel": MOD, "kwargs": mod_kw,
             "in":  {"X_hbm": "LN_Y", "SCALE1P_hbm": "MOD_S1P", "SHIFT_hbm": "MOD_SHIFT"},
             "out": {"Y_hbm": "MOD_Y"}},
        ],
    }

    mgr = Manager(settings=s)
    out = mgr.run_graph(
        graph,
        data={"X": x, "LN_SCALE": ln_sc_f, "LN_BIAS": ln_bi_f,
              "MOD_S1P": (1.0 + mod_sc).to(torch.float32), "MOD_SHIFT": mod_shft},
        compare={"LN_Y": ln_y.reshape(-1).numpy(),
                 "MOD_Y": mod_y.reshape(-1).numpy()},
    )

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.85)
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: cosine={cmp.cosine:.6f} "
              f"nrmse={cmp.nrmse:.6f}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
