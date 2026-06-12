"""Adam optimizer step end-to-end through the manager.

One fused weight update (m/v moments + bias correction + Newton rsqrt + W
update), vector-elementwise only. PLENA has no element-wise sqrt, so
``1/(sqrt(vhat)+eps)`` is one Newton-Raphson rsqrt step (seed y0=1/(vhat+1)).

The golden **mirrors the kernel's Newton approximation** (NOT torch's exact
rsqrt) and runs in f16, so the comparison checks the kernel computes the same
approximate update — this is a cost/lowering study, the approximation is the
intended behaviour.

Run: tools/manager/run.sh _validate_adam_step
"""

import os
import sys
from pathlib import Path

os.environ["PLENA_ALLOC_MODE"] = "stable"

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.tensor import Role
from manager.pipeline import Manager


def _h(x):
    """f16 round-trip (match the hardware's f16 elementwise path)."""
    return x.half().float()


def main() -> int:
    s = load_behavior_settings()
    MLEN, HLEN = s.mlen, s.hlen
    HEAD = s.hardware_lane_count          # one lane-group of heads
    head_count = HEAD
    NSB = 2
    ROWS = MLEN
    SEQ = NSB * ROWS

    lr, beta1, beta2 = 1e-3, 0.9, 0.999
    weight_decay = 1e-2          # AdamW decoupled weight decay (matches kernel default)
    # step t=1 bias-correction reciprocals
    bc1 = 1.0 / (1.0 - beta1)             # 10.0
    bc2 = 1.0 / (1.0 - beta2)             # 1000.0
    one_m_b1 = 1.0 - beta1
    one_m_b2 = 1.0 - beta2

    print(f"geometry: mlen={MLEN} hlen={HLEN} head_count={head_count} "
          f"NSB={NSB} SEQ={SEQ}  lr={lr} b1={beta1} b2={beta2} bc1={bc1} bc2={bc2}")

    torch.manual_seed(0)
    shp = (1, SEQ, head_count, HLEN)
    W = torch.randn(shp) * 0.1
    G = torch.randn(shp) * 0.05          # grad dW
    M = torch.randn(shp) * 0.01          # 1st moment
    V = (torch.randn(shp) ** 2) * 0.001  # 2nd moment (>= 0)

    # ---- golden: replicate the kernel's f16 Newton-rsqrt update exactly ----
    Wf, Gf, Mf, Vf = _h(W), _h(G), _h(M), _h(V)
    m = _h(_h(beta1 * Mf) + _h(one_m_b1 * Gf))
    v = _h(_h(beta2 * Vf) + _h(one_m_b2 * _h(Gf * Gf)))
    mhat = _h(m * bc1)
    vhat = _h(v * bc2)
    # Newton rsqrt, one step, seed y0 = 1/(vhat+1)
    y0 = _h(1.0 / _h(vhat + 1.0))
    r = _h(y0 * _h(1.5 - _h(_h(0.5 * vhat) * _h(y0 * y0))))
    # AdamW: w = W - lr*mhat*r - lr*wd*W  (decoupled weight decay)
    Wo_g = _h(_h(Wf - _h(_h(lr * mhat) * r)) - _h((lr * weight_decay) * Wf))
    Mo_g = m
    Vo_g = v

    mgr = Manager(settings=s)
    mgr.place("W_hbm",  list(shp), Role.IO, data=W)
    mgr.place("G_hbm",  list(shp), Role.IO, data=G)
    mgr.place("M_hbm",  list(shp), Role.IO, data=M)
    mgr.place("V_hbm",  list(shp), Role.IO, data=V)
    mgr.place("Wo_hbm", list(shp), Role.IO, data=torch.zeros(shp))
    mgr.place("Mo_hbm", list(shp), Role.IO, data=torch.zeros(shp))
    mgr.place("Vo_hbm", list(shp), Role.IO, data=torch.zeros(shp))

    out = mgr.run_kernel(
        "tilelang_tvm_compiler.kernels.adam_step_min:make_adam_step_min",
        asm_name="adam_step_solo",
        kernel_kwargs={
            "head_count": head_count, "num_s_blocks": NSB,
            "lr": lr, "beta1": beta1, "beta2": beta2, "bc1": bc1, "bc2": bc2,
            "weight_decay": weight_decay,
        },
        compare={
            "Wo_hbm": Wo_g.reshape(-1).numpy(),
            "Mo_hbm": Mo_g.reshape(-1).numpy(),
            "Vo_hbm": Vo_g.reshape(-1).numpy(),
        },
    )

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.8)
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: "
              f"cosine={cmp.cosine:.6f} nrmse={cmp.nrmse:.6f}")
        if not ok and cmp.got is not None:
            import numpy as _np
            g, gold = cmp.got, cmp.golden
            print(f"      got[:8]   = {_np.round(g[:8],5)}")
            print(f"      golden[:8]= {_np.round(gold[:8],5)}")
            ratio = g[:8] / (gold[:8] + 1e-12)
            print(f"      got/golden= {_np.round(ratio,4)}")
            # is got == some other quantity? check vs raw inputs
            print(f"      got nonzero frac = {float((_np.abs(g)>1e-6).mean()):.3f}, "
                  f"golden nonzero frac = {float((_np.abs(gold)>1e-6).mean()):.3f}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'} (cosine >= 0.8)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
