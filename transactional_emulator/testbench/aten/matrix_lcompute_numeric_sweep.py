"""Seeded BF16 recurrence regression, with every intermediate state read from Rust HBM.

Snapshot DMA is diagnostic overhead and is not a performance measurement.
"""

import argparse
import json
from pathlib import Path

import torch

from transactional_emulator.testbench.aten.matrix_lcompute_recurrence_test import (
    SEED,
    NEMOTRON_MAMBA,
    KIMI_KDA,
    RecurrenceKind,
    RecurrenceLayout,
    _mamba_inputs,
    _kda_inputs,
    _state_seed,
    run_prepared_case,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED, 17, 65537])
    parser.add_argument("--layouts", nargs="+", choices=["affine", "fixed"], default=["affine", "fixed"])
    parser.add_argument("--model", choices=["mamba", "kda", "both"], default="both")
    args = parser.parse_args()
    if args.tokens < 4:
        parser.error("use at least four tokens")
    torch.set_num_threads(1)
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = {"mamba": NEMOTRON_MAMBA, "kda": KIMI_KDA}
    for spec in specs.values() if args.model == "both" else (specs[args.model],):
        inputs = _mamba_inputs if spec.kind is RecurrenceKind.MAMBA else _kda_inputs
        for seed in args.seeds:
            operands = tuple(inputs(token, seed) for token in range(args.tokens))
            for layout in map(RecurrenceLayout, args.layouts):
                result = run_prepared_case(
                    spec,
                    layout,
                    args.output_dir,
                    initial_state=_state_seed(spec, seed),
                    operands_by_token=operands,
                    check_intermediate=True,
                    exact=layout is RecurrenceLayout.AFFINE,
                    case_name=f"{spec.name}_seed{seed}_tokens{args.tokens}",
                ).report
                result["seed"] = seed
                case = f"{spec.name}_seed{seed}_tokens{args.tokens}_{layout.value}.json"
                (args.output_dir / case).write_text(json.dumps(result, indent=2) + "\n")
                results.append(result)
                (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{spec.name} {layout} seed={seed}: passed {args.tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
