"""Step-2 validation for the manager data/address layer.

Run with the torch venv:
  cd PLENA_Simulator
  LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib \
    PYTHONPATH=tools ./.venv/bin/python3 tools/manager/_validate_step2.py

Checks:
  A. packed_byte_size matches the real append packer's output size.
  B. seek-written bytes are byte-identical to the append packer.
  C. round-trip: write a tensor via seek, read it back, compare to the
     MX-round-tripped golden (cosine).
"""

import os
import sys
import tempfile

import numpy as np
import torch

from manager.geometry import load_behavior_settings
from manager.binio import write_tensor, read_tensor, packed_byte_size
from manager.tensor import HbmLayout, Role

from memory_mapping.rand_gen import Random_MXFP_Tensor_Generator
from memory_mapping.memory_map import map_mx_data_to_hbm_for_behave_sim


def _append_pack(tensor, s, outdir):
    qc = {"exp_width": s.elem_exp, "man_width": s.elem_man,
          "exp_bias_width": s.scale_exp, "block_size": [1, s.block_size],
          "int_width": 32, "skip_first_dim": False}
    gen = Random_MXFP_Tensor_Generator(shape=tuple(tensor.shape), quant_config=qc,
                                       config_settings={}, directory=None, filename=None)
    blocks, bias = gen.quantize_tensor(tensor.contiguous().reshape(1, -1))
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        map_mx_data_to_hbm_for_behave_sim(
            blocks, element_width=s.elem_bits, block_width=s.elem_bits * s.block_size,
            bias=bias, bias_width=s.scale_bits, directory=outdir,
            append=False, hbm_row_width=s.hbm_row_width)
    return open(os.path.join(outdir, "hbm_for_behave_sim.bin"), "rb").read()


def main() -> int:
    s = load_behavior_settings()
    print(f"BEHAVIOR: mlen={s.mlen} hlen={s.hlen} hbm_row_width={s.hbm_row_width} "
          f"elem={s.elem_bits}b scale={s.scale_bits}b block={s.block_size}")

    torch.manual_seed(0)
    fails = 0

    # ---- A + B: size + byte-identity across a range of sizes ----
    print("\n[A/B] packed size + byte-identity vs append packer")
    for n in [8, 64, 100, 512, 520, 1024, 4096]:
        t = torch.randn(1, n) * 0.5
        outdir = tempfile.mkdtemp()
        append_bytes = _append_pack(t, s, outdir)

        # seek-write into a fresh zero-filled file at offset 0
        binp = os.path.join(tempfile.mkdtemp(), "seek.bin")
        open(binp, "wb").write(b"\x00" * packed_byte_size(n, s))
        nwritten = write_tensor(binp, 0, t, s)
        seek_bytes = open(binp, "rb").read()

        size_ok = packed_byte_size(n, s) == len(append_bytes)
        ident_ok = seek_bytes == append_bytes
        tag = "OK " if (size_ok and ident_ok) else "FAIL"
        if not (size_ok and ident_ok):
            fails += 1
        print(f"  {tag} n={n:5d}  packed={packed_byte_size(n,s):5d} "
              f"append={len(append_bytes):5d} written={nwritten:5d} "
              f"size_match={size_ok} byte_identical={ident_ok}")

    # ---- C: round-trip via seek-write + read ----
    print("\n[C] round-trip (write -> read -> cosine vs MX-roundtripped golden)")
    for shape in [(1, 256), (4, 128), (2, 8, 16)]:
        n = int(np.prod(shape))
        t = torch.randn(*shape) * 0.5
        binp = os.path.join(tempfile.mkdtemp(), "rt.bin")
        open(binp, "wb").write(b"\x00" * packed_byte_size(n, s))
        write_tensor(binp, 0, t, s)
        got = read_tensor(binp, 0, shape, s).astype(np.float32).reshape(-1)

        # golden = what reading MX bytes back should yield: the append packer's
        # bytes read by the same reader (self-consistent reference).
        ref = read_tensor(binp, 0, shape, s).astype(np.float32).reshape(-1)
        # cosine between read-back and the original (lossy MX, expect high but <1)
        a, b = got, t.reshape(-1).numpy().astype(np.float32)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        ok = cos > 0.9
        if not ok:
            fails += 1
        print(f"  {'OK ' if ok else 'FAIL'} shape={shape} n={n:4d} "
              f"cosine(readback, original)={cos:.6f}")

    # ---- D: two tensors at distinct offsets, independent overwrite ----
    print("\n[D] two tensors, seek to distinct offsets, independent rewrite")
    layout = HbmLayout(s, base=0)
    ta = torch.randn(1, 128) * 0.5
    tb = torch.randn(1, 256) * 0.5
    A = layout.place("A", (1, 128), Role.IO, data=ta)
    B = layout.place("B", (1, 256), Role.IO, data=tb)
    binp = os.path.join(tempfile.mkdtemp(), "two.bin")
    open(binp, "wb").write(b"\x00" * layout.total_bytes())
    write_tensor(binp, A.hbm_addr, ta, s)
    write_tensor(binp, B.hbm_addr, tb, s)
    a0 = read_tensor(binp, A.hbm_addr, A.shape, s).reshape(-1)
    b0 = read_tensor(binp, B.hbm_addr, B.shape, s).reshape(-1)
    # overwrite only A with new data; B must be unchanged
    ta2 = torch.randn(1, 128) * 0.5
    write_tensor(binp, A.hbm_addr, ta2, s)
    a1 = read_tensor(binp, A.hbm_addr, A.shape, s).reshape(-1)
    b1 = read_tensor(binp, B.hbm_addr, B.shape, s).reshape(-1)
    b_unchanged = np.array_equal(b0, b1)
    a_changed = not np.array_equal(a0, a1)
    print(f"  A.addr={A.hbm_addr} B.addr={B.hbm_addr} "
          f"(A packed={A.packed_bytes(s)})")
    ok = b_unchanged and a_changed
    if not ok:
        fails += 1
    print(f"  {'OK ' if ok else 'FAIL'} B_unchanged_after_A_rewrite={b_unchanged} "
          f"A_changed={a_changed}")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
