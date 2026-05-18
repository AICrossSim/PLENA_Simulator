# Attention Looping Step 2 Report

## Goal

Reduce ATen MHA attention instruction memory without changing the amount of
matrix/vector attention work. This implements the second step of the large-tile
codegen plan: roll the obvious row-wise attention helper code with hardware
loops, while accepting small dynamic loop overhead as the tradeoff.

## Branches Checked

- Simulator repo: `asm-count-verification`
- Simulator base hash before these edits: `90c88701ba46a1ae86eeb201e2fe8358ee6972bd`
- Nested compiler repo: `feat/codegen-addr-reg-init`
- Nested compiler base hash before these edits: `68615ff3d1894c7c7720f79b62269f1c09c6dab0`

## Implementation

Changed nested compiler files:

- `compiler/aten/plena/isa_attention.py`
- `compiler/aten/plena/isa_compiler.py`

The old Python-unrolled attention code is still available through
`prog.unroll_attention = True`. Default ATen MHA attention emission now uses
hardware loops for:

- `_online_softmax_asm`
- `_scale_o_asm`
- `_final_scaling_asm`
- `_pv_multiply_asm`
- `_reset_fpsram_asm`
- `_reset_vram_asm`

The new looped path keeps dynamic matrix/vector work equivalent to the old
unrolled path. It does add scalar pointer increments and `C_LOOP_*` control
instructions, which is the expected instruction-memory tradeoff.

Added simulator-repo harness:

- `transactional_emulator/testbench/aten/attention_looping_compare.py`
- `transactional_emulator/testbench/aten/flash_attention_mha_test.py`

The harness forces generic ATen GEMM lowering to looped mode for both rows
(`ATEN_UNROLL=0`) and toggles only `prog.unroll_attention`.

## Commands Run

```bash
.venv/bin/python -m py_compile \
  compiler/aten/plena/isa_attention.py \
  compiler/aten/plena/isa_compiler.py \
  transactional_emulator/testbench/isa_analysis.py \
  transactional_emulator/testbench/aten/attention_looping_compare.py

.venv/bin/python transactional_emulator/testbench/aten/attention_looping_compare.py \
  --out-dir transactional_emulator/testbench/aten/build/attention_looping_compare

.venv/bin/python transactional_emulator/testbench/aten/attention_codegen_compare.py \
  --out-dir transactional_emulator/testbench/aten/build/attention_codegen_compare_after_attention_looping

.venv/bin/python transactional_emulator/testbench/aten/flash_attention_mha_test.py \
  --build-dir transactional_emulator/testbench/aten/build/flash_attention_mha_looped
```

## Focused ATen MHA Result

Shape:

```text
seq_len = 64
head_dim = 64
causal = true
MLEN = 64
BLEN = 4
DC_EN = 1
```

| Mode | Source lines | Static instr | Dynamic instr | Semantic dynamic instr | Loop/pointer overhead instr | Est cycles | Est ms @1GHz | C_LOOP_START |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `looped` | 176 | 128 | 7,664 | 2,761 | 4,903 | 40,558 | 0.040558 | 13 |
| `unrolled` | 3,087 | 2,960 | 6,485 | 2,761 | 3,724 | 39,379 | 0.039379 | 5 |

Ratios:

- Source reduction, unrolled / looped: `17.540x`
- Static instruction reduction, unrolled / looped: `23.125x`
- Dynamic instruction ratio, looped / unrolled: `1.182x`
- Semantic dynamic instruction ratio, looped / unrolled: `1.000x`
- Estimated cycle ratio, looped / unrolled: `1.030x`

Interpretation: static attention code drops substantially, while all
non-overhead dynamic opcode counts are unchanged. Estimated cycles rise by
about 3% from loop control and pointer increments. The harness compares every
dynamic opcode and only allows `C_LOOP_START`, `C_LOOP_END`, and `S_ADDI_INT`
to differ between looped and unrolled emission.

## Broader Attention Compare After Change

| Case | Source lines | Static instr | Dynamic instr | Est cycles | Est ms @1GHz | C_LOOP_START |
|---|---:|---:|---:|---:|---:|---:|
| `direct_emit_mha_1h_d64_teststyle` | 232 | 208 | 18,442 | 293,240 | 0.293240 | 12 |
| `aten_mha_1h_d64` | 176 | 128 | 7,664 | 40,558 | 0.040558 | 13 |
| `direct_emit_gqa_4h_d16_teststyle` | 525 | 469 | 12,118 | 30,848 | 0.030848 | 26 |
| `aten_gqa_fused_4h_d16` | 467 | 412 | 12,061 | 30,807 | 0.030807 | 26 |

The GQA rows are unchanged in spirit: ATen fused GQA still dispatches into the
George-style `flash_attn_asm` path. The large static bloat was in the ATen MHA
primitive helper path.

## Multi-Column-Block Sanity Check

The harness was also run with `head_dim=128` to cover two MLEN-wide output
column blocks. This checks that row-looped scaling does not repeat scalar
`m_res` or `l` work per column block.

| Mode | Source lines | Static instr | Dynamic instr | Semantic dynamic instr | Loop/pointer overhead instr | Est cycles | Est ms @1GHz | C_LOOP_START |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `looped` | 221 | 169 | 11,486 | 3,787 | 7,699 | 76,634 | 0.076634 | 19 |
| `unrolled` | 4,545 | 4,399 | 9,263 | 3,787 | 5,476 | 74,411 | 0.074411 | 6 |

The harness assertions passed for both `head_dim=64` and `head_dim=128`.

## Golden Transactional Emulator Check

The default looped ATen MHA primitive path was also run through the Rust
transactional emulator and checked against PyTorch
`scaled_dot_product_attention`.

Shape:

```text
seq_len = 64
head_dim = 64
causal = false
MLEN = 64
BLEN = 4
```

Result:

```text
Generated 163 lines of ISA
Simulation completed. Latency 40662.000ns
Allclose Check: PASS
Match Rate: 100.00%
Max Absolute Error: 0.072266
MAE: 0.01232910
```

The golden checker uses the existing `run_and_assert` path, so this verifies
that the looped helper emission assembles, runs in the transactional emulator,
and matches the PyTorch SDPA golden within the repository's standard
`atol=0.2, rtol=0.2` threshold.

## Caveats

- This is static codegen analysis plus loop-expanded dynamic/cycle estimation.
  It is not a full transactional simulator numerical run.
- The cycle model is loaded from `plena_settings.toml` with `DC_EN=1`.
- The static model does not charge a fixed latency for `H_PREFETCH_*` or
  `H_STORE_*`; those are async memory operations in the simulator.
- Dynamic instruction count is expected to increase under the looped path. The
  point of this change is instruction memory pressure, not a cycle reduction.
