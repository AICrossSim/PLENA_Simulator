"""Single linear (GEMM) end-to-end through the manager — fast smoke test for a
new geometry (compiles ONE kernel). Geometry all from plena_settings.toml.

  C = A @ B^T + bias,  A=(M,K) input, B=(N,K) weight, bias (M,N)
  M = N = K = HD = (mlen//hlen*2) * hlen  (= head_count * hlen)

Run: tools/manager/run.sh _validate_linear
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.tensor import Role
from manager.pipeline import Manager
from _tb_cache import mx_roundtrip as mxr


def main() -> int:
    s = load_behavior_settings()
    MLEN, HLEN = s.mlen, s.hlen
    HEAD = s.hardware_lane_count * 2
    HD = HEAD * HLEN
    M = N = K = HD
    mb, nb, kb = M // MLEN, N // MLEN, K // MLEN
    print(f"geometry: mlen={MLEN} hlen={HLEN} lane={s.hardware_lane_count} "
          f"HEAD={HEAD} HD={HD}  (M=N=K={HD}, blocks m={mb} n={nb} k={kb})")

    A_shape = [1, M, 1, K]
    B_shape = [1, N, 1, K]
    bias_shape = [1, M, 1, N]
    C_shape = [1, M, 1, N]

    torch.manual_seed(0)
    a2d = torch.randn(M, K) * 0.25
    b2d = torch.randn(N, K) * 0.25
    bias1d = torch.randn(N) * 0.1
    a = a2d.view(1, M, 1, K).contiguous()
    b = b2d.view(1, N, 1, K).contiguous()
    bias_tile = bias1d.view(1, 1, 1, N).expand(1, M, 1, N).contiguous()

    a_eff = mxr(a2d); b_eff = mxr(b2d)
    bias_eff = mxr(bias1d.view(1, N).expand(M, N).contiguous())
    C_golden = mxr(a_eff @ b_eff.T + bias_eff)

    mgr = Manager(settings=s)
    mgr.place("A_hbm", A_shape, Role.IO, data=a)
    mgr.place("B_hbm", B_shape, Role.WEIGHT, data=b)
    mgr.place("BIAS_hbm", bias_shape, Role.WEIGHT, data=bias_tile)
    mgr.place("C_hbm", C_shape, Role.IO, data=torch.zeros(1, M, 1, N))

    out = mgr.run_kernel(
        "tilelang_tvm_compiler.kernels.linear_min:make_linear_min",
        asm_name="linear_solo",
        kernel_kwargs={"m_blocks": mb, "n_blocks": nb, "k_blocks": kb, "with_bias": True},
        compare={"C_hbm": C_golden.reshape(-1).numpy()},
    )
    cmp = out["compares"][0]
    print(f"\nC_hbm: cosine={cmp.cosine:.6f} nrmse={cmp.nrmse:.6f}")
    ok = cmp.ok(cos_thresh=0.8)
    print(f"\n{'PASS' if ok else 'FAIL'} (cosine >= 0.8)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
