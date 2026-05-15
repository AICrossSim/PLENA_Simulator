"""Run a CLM-60M sliced decoder test using the local PLENA_RTL config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "tools", REPO_ROOT / "PLENA_Compiler"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def main() -> int:
    from transactional_emulator.testbench.rtl_config import default_rtl_root, rtl_plena_settings
    from transactional_emulator.testbench.sliced_layer_test_builder import build_and_run_sliced_decoder_layer_test

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", type=Path, default=None, help="Path to local PLENA_RTL checkout")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length; defaults to RTL MLEN")
    parser.add_argument("--hidden-size", type=int, default=None, help="Sliced hidden size; defaults to RTL MLEN")
    parser.add_argument("--inter-dim", type=int, default=None, help="Sliced FFN intermediate; defaults to 2*MLEN")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/clm60m_rtl_config"))
    args = parser.parse_args()

    rtl_root = (args.rtl_root or default_rtl_root(REPO_ROOT)).resolve()
    plena_toml = REPO_ROOT / "plena_settings.toml"

    with rtl_plena_settings(plena_toml, rtl_root) as rtl:
        mlen = rtl["MLEN"]
        blen = rtl["BLEN"]
        seq_len = args.seq_len or mlen
        hidden_size = args.hidden_size or mlen
        inter_dim = args.inter_dim or (2 * mlen)

        print("Using RTL configuration:")
        print(f"  rtl_root={rtl_root}")
        print(
            f"  MLEN={rtl['MLEN']} VLEN={rtl['VLEN']} BLEN={rtl['BLEN']} "
            f"HLEN={rtl['HLEN']} BROADCAST_AMOUNT={rtl['BROADCAST_AMOUNT']}"
        )
        print(
            f"  M_FP=e{rtl['M_FP_EXP_WIDTH']}m{rtl['M_FP_MANT_WIDTH']} "
            f"V_FP=e{rtl['V_FP_EXP_WIDTH']}m{rtl['V_FP_MANT_WIDTH']} "
            f"S_FP=e{rtl['S_FP_EXP_WIDTH']}m{rtl['S_FP_MANT_WIDTH']}"
        )
        print(
            f"  CLM-60M sliced decoder: seq_len={seq_len}, "
            f"hidden_size={hidden_size}, inter_dim={inter_dim}"
        )

        build_and_run_sliced_decoder_layer_test(
            model_id="AICrossSim/clm-60m",
            asm_name=f"clm60m_rtl_m{mlen}_b{blen}",
            build_dir=args.build_dir,
            layer_idx=args.layer_idx,
            seq_len=seq_len,
            hidden_size=hidden_size,
            inter_dim=inter_dim,
            mlen=mlen,
            blen=blen,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
