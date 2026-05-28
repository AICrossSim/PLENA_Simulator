"""Configurable ATen-style normalization test (RMSNorm or LayerNorm).

python norm_test.py [--norm-type rms|layer] [--mlen 128] [--blen 16] [--batch 8] [--hidden 256]
"""

from pathlib import Path
import argparse
import json
import torch

from compiler.aten.ops.registry import OpRegistry, Backend
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.golden import golden_rms_norm, golden_layer_norm
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--norm-type", choices=["rms", "layer"], default="rms")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: blen)")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden size (default: 2*mlen)")
    parser.add_argument("--build-dir", type=Path, default=None)
    args = parser.parse_args()

    norm_type = args.norm_type
    mlen = args.mlen
    blen = args.blen
    batch = args.batch or blen
    hidden = args.hidden or 2 * mlen

    if batch % blen != 0:
        raise ValueError(f"batch ({batch}) must be divisible by BLEN ({blen})")
    if hidden % mlen != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by MLEN ({mlen})")

    build_dir = args.build_dir or (Path(__file__).parent / "build" / f"{norm_type}_norm")
    hw = setup_hw(args, build_dir)

    label = "RMSNorm" if norm_type == "rms" else "LayerNorm"
    print(
        f"{'=' * 80}\nATen-style {label} Test  (mlen={mlen}, blen={blen}, batch={batch}, hidden={hidden})\n{'=' * 80}"
    )

    torch.manual_seed(args.seed)
    X = torch.randn(batch, hidden)
    eps = 1e-5

    if norm_type == "rms":
        golden = golden_rms_norm(X.clone(), eps)
    else:
        golden = golden_layer_norm(X.clone(), eps)

    print(f"  golden: {golden.shape}  golden[0,:4]: {golden[0, :4].tolist()}")

    import compiler.aten.ops as ops

    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)
    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_input = prog.input("X", shape=(batch, hidden))
    x_batch = prog.load_batch(x_input, name="X")

    if norm_type == "rms":
        ops.rms_norm(prog, x_batch, eps_offset=0, reci_hid_offset=1)
    else:
        ops.layer_norm(prog, x_batch)

    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA code")

    y_vram_addr = prog._compiler.get_vram_addr(x_batch.name)
    physical_rows = ((batch + blen - 1) // blen) * blen
    physical_hidden = ((hidden + mlen - 1) // mlen) * mlen

    comparison_params = {
        "start_row_idx": y_vram_addr // mlen,
        "num_rows": (physical_rows * physical_hidden) // mlen,
        "num_batches": physical_rows,
        "elements_per_batch": physical_hidden,
        "row_dim": mlen,
        "physical_rows": physical_rows,
        "use_slice_mode": True,
        "slice_per_row": hidden,
    }

    input_tensors = {"X": X}
    golden_result = {"original_output": golden}

    create_sim_env(input_tensors, gen_code, golden_result, fp_preload=None, build_dir=str(build_dir))
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear_aten",
        data=None,
        specified_data_order=["X"],
        build_path=build_dir,
        input_tensors=input_tensors,
    )

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    run_and_assert(build_dir, f"{norm_type}_norm", mlen=mlen, blen=blen)
