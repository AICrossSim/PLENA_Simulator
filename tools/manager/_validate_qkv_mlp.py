"""MLP/projection segment of single_stream_block: 4 linears + gelu.

    x_mod --linear_q(Wq)--> Q
          --linear_k(Wk)--> K
          --linear_v(Wv)--> V
          --linear_mlp(Wm)--> MLP --gelu--> GELU_OUT

All four linears read the SAME input (x_mod) with their own weight; gelu
follows linear_mlp. x_mod is fed as io input (so this segment is validated on
its own, not requiring the upstream layernorm->modulate each run).

Geometry: x_mod is (1,S,H,D)=(1,128,8,16); flat S x 128. Each linear treats it
as A=(1,M,1,K) with M=S=128, K=H*D=128 -> m_blocks=k_blocks=2 (MLEN=64). N=128
-> n_blocks=2. gelu sees MLP (1,128,1,128) as (1,128,8,16) — same flat bytes.

Run:
  cd PLENA_Simulator
  LD_LIBRARY_PATH="<gcc>:<zlib>:<torch>" \
    PYTHONPATH=tools:compiler:transactional_emulator/testbench \
    ./.venv/bin/python3 tools/manager/_validate_qkv_mlp.py
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
    MLEN = s.mlen
    S = 2 * MLEN              # 128
    HD = 128                  # H*D = head_count*hlen = 8*16
    M, K, N = S, HD, HD
    mb, kb, nb = M // MLEN, K // MLEN, N // MLEN
    HEAD_COUNT, HLEN = 8, s.hlen
    lin_shape = [1, M, 1, K]            # linear A view
    w_shape   = [1, N, 1, K]            # linear B (weight) view (N,K)
    bias_shape = [1, M, 1, N]
    out_shape = [1, M, 1, N]            # linear C view
    gelu_shape = [1, S, HEAD_COUNT, HLEN]   # gelu view of MLP (same bytes)

    LINEAR = "tilelang_tvm_compiler.kernels.linear_min:make_linear_min"
    GELU   = "tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min"
    lin_kw = {"m_blocks": mb, "n_blocks": nb, "k_blocks": kb, "with_bias": True}
    gelu_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD_COUNT,
               "num_s_blocks": S // MLEN, "batch": 1}

    torch.manual_seed(0)
    x_mod = torch.randn(1, M, 1, K, dtype=torch.float32) * 0.25
    a_eff = mx_roundtrip(x_mod.reshape(M, K))

    def lin_golden(W2d, bias1d):
        b_eff = mx_roundtrip(W2d)
        bias_eff = mx_roundtrip(bias1d.view(1, N).expand(M, N).contiguous())
        return mx_roundtrip(a_eff @ b_eff.T + bias_eff)   # (M,N)

    # four independent weights
    Wq = torch.randn(N, K) * 0.25; bq = torch.randn(N) * 0.1
    Wk = torch.randn(N, K) * 0.25; bk = torch.randn(N) * 0.1
    Wv = torch.randn(N, K) * 0.25; bv = torch.randn(N) * 0.1
    Wm = torch.randn(N, K) * 0.25; bm = torch.randn(N) * 0.1
    Q_g = lin_golden(Wq, bq); K_g = lin_golden(Wk, bk)
    V_g = lin_golden(Wv, bv); MLP_g = lin_golden(Wm, bm)
    # gelu on MLP (MX-roundtripped both sides)
    GELU_g = mx_roundtrip(torch.nn.functional.gelu(MLP_g, approximate="tanh"))

    def w4(W):  # (N,K) -> (1,N,1,K)
        return W.view(1, N, 1, K).contiguous()
    def b4(b):  # (N,) -> (1,M,1,N) bias tile
        return b.view(1, 1, 1, N).expand(1, M, 1, N).contiguous()

    graph = {
        "tensors": {
            "XMOD": {"shape": lin_shape, "role": "io"},
            "WQ": {"shape": w_shape, "role": "weight"}, "BQ": {"shape": bias_shape, "role": "weight"},
            "WK": {"shape": w_shape, "role": "weight"}, "BK": {"shape": bias_shape, "role": "weight"},
            "WV": {"shape": w_shape, "role": "weight"}, "BV": {"shape": bias_shape, "role": "weight"},
            "WM": {"shape": w_shape, "role": "weight"}, "BM": {"shape": bias_shape, "role": "weight"},
            "Q":   {"shape": out_shape, "role": "io"},
            "K":   {"shape": out_shape, "role": "io"},
            "V":   {"shape": out_shape, "role": "io"},
            "MLP": {"shape": gelu_shape, "role": "activation"},   # linear_mlp out == gelu in
            "GELU_OUT": {"shape": gelu_shape, "role": "io"},
        },
        "nodes": [
            {"name": "linear_q", "kernel": LINEAR, "kwargs": lin_kw,
             "in": {"A_hbm": "XMOD", "B_hbm": "WQ", "BIAS_hbm": "BQ"}, "out": {"C_hbm": "Q"}},
            {"name": "linear_k", "kernel": LINEAR, "kwargs": lin_kw,
             "in": {"A_hbm": "XMOD", "B_hbm": "WK", "BIAS_hbm": "BK"}, "out": {"C_hbm": "K"}},
            {"name": "linear_v", "kernel": LINEAR, "kwargs": lin_kw,
             "in": {"A_hbm": "XMOD", "B_hbm": "WV", "BIAS_hbm": "BV"}, "out": {"C_hbm": "V"}},
            {"name": "linear_mlp", "kernel": LINEAR, "kwargs": lin_kw,
             "in": {"A_hbm": "XMOD", "B_hbm": "WM", "BIAS_hbm": "BM"}, "out": {"C_hbm": "MLP"}},
            {"name": "gelu", "kernel": GELU, "kwargs": gelu_kw,
             "in": {"X_hbm": "MLP"}, "out": {"Y_hbm": "GELU_OUT"}},
        ],
    }

    mgr = Manager(settings=s)
    out = mgr.run_graph(
        graph,
        data={"XMOD": x_mod,
              "WQ": w4(Wq), "BQ": b4(bq), "WK": w4(Wk), "BK": b4(bk),
              "WV": w4(Wv), "BV": b4(bv), "WM": w4(Wm), "BM": b4(bm)},
        compare={"Q": Q_g.reshape(-1).numpy(), "K": K_g.reshape(-1).numpy(),
                 "V": V_g.reshape(-1).numpy(), "MLP": MLP_g.reshape(-1).numpy(),
                 "GELU_OUT": GELU_g.reshape(-1).numpy()},
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
