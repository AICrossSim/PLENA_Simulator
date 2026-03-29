# TileTensor Kernel Programs

This note points to the cleaned-up location for the new `TileTensorProgram`
kernel rewrites.

## Files

All new kernel rewrites live under:

- `transactional_emulator/testbench/tile_tensor_kernel_programs/`

Main files:

- `linear.py`
- `layernorm.py`
- `rmsnorm.py`
- `attention.py`
- `elementwise.py`
- `rope.py`
- `activations.py`
- `testbench_runner.py`

## Why The New Layout Matters

These rewrites are not only a file move. They are the first group of kernels
written against the newer `TileTensorProgram` runtime model, where symbolic
parallel regions are a core feature rather than an implementation detail.

The important shift is:

- kernels describe logical tensor work
- the runtime records symbolic parallel assignments
- the thread manager derives cache and execution plans
- the emulator lowering emits the final ISA-oriented sequence

That `parallel` capability is what makes this directory strategically important:
the kernels here are the place where the stronger execution model becomes
visible and reusable.

## Parallel Programming Model

The main public entrypoint is:

- `program.parallel_region3d((S, H, D), name="...")`

Inside the region, the runtime returns symbolic axes, typically `(s, h, d)`,
and indexing expressions like `x[s, h, d]` become symbolic loads. Assignments
like `y[s, h, d] = ...` register a parallel task instead of executing
immediately.

Supported helpers include:

- `program.where(predicate, on_true, on_false)`
- `program.if_then_else(...)`
- `program.pair(d)` for RoPE-style even/odd partner lanes
- `program.half_index(d)` for coefficient-group addressing

When the region closes, the runtime finalizes:

- the parallel assignment graph
- cache-slot planning
- cycle-by-cycle load / compute / writeback steps
- execution-plan lowering into the transactional emulator flow

## Current Constraints

The current lowering is intentionally focused and does not claim to support
arbitrary tensor syntax yet.

- Parallel writes must target tensor-backed destinations.
- Destination indexing must use the active 3D axes of the region.
- Arithmetic expressions currently lower `add`, `sub`, and `mul`.
- Predicate expressions currently lower comparison operators such as `lt`,
  `le`, `gt`, `ge`, and `eq`.
- Cycle lowering currently assumes one full-width contiguous row per cycle.

## How To Run

From `/home/a13247568123124/project/PLENA_Simulator`:

- `just build-emulator-debug linear`
- `just build-emulator-debug layernorm`
- `just build-emulator-debug rmsnorm`
- `just build-emulator-debug attention`
- `just build-emulator-debug elementwise`

For `elementwise`, the default mode is `modulate`.

To run `residual_gate` instead:

- `TILE_TENSOR_ELEMENTWISE_KIND=residual_gate just build-emulator-debug elementwise`

## Output Artifacts

Generated files still land in:

- `transactional_emulator/testbench/build/`

Important outputs:

- `generated_machine_code.mem`
- `hbm_for_behave_sim.bin`
- `fp_sram.bin`
- `int_sram.bin`
- `comparison_params.json`
- `*_generated_asm_code.asm`
- `*_operation_report.txt`
