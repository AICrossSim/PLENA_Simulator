# Decode reference

How a decode step is lowered onto PLENA, what the hardware contract is, and what
has been measured against it. Results are labelled by evidence source: RTL
observations, transactional-emulator measurements, compiler-counted fixtures,
or analytic sensitivities. RTL claims cite the source file or probe that supports
them.

Contents:

1. [Matrix operand addressing](#1-matrix-operand-addressing)
2. [Broadcast matmul granularity](#2-broadcast-matmul-granularity)
3. [Validating the attention stage](#3-validating-the-attention-stage)
4. [Scaling across array geometries](#4-scaling-across-array-geometries)
5. [KV cache traffic](#5-kv-cache-traffic)

---

## 1. Matrix operand addressing

The matrix opcodes do not share one address convention. Each decomposes the
operand address differently, and the differences are deliberate: broadcast ops
carry a head offset in the bits below MLEN, plain ops do not. An emitter that
assumes a single convention produces assembly that runs on one backend and not
the other.

`multiple_and_offset(x, m)` returns `(floor(x / m) * m, x mod m)`.

| Opcode | Tile modulus | Address decomposition | Selects |
| --- | --- | --- | --- |
| `M_MM` | `MLEN²` | `column = (addr mod MLEN²) / MLEN` | columns `[column, column+BLEN)`; requires `column + BLEN <= MLEN` |
| `M_TMM` | `MLEN²` | `row = (addr mod MLEN²) / MLEN` | rows `[row, row+BLEN)` of `W`, i.e. columns of `Wᵀ` |
| `M_MV` | `MLEN²` | `column = (addr mod MLEN²) / MLEN` | one column group |
| `M_TMV` | `MLEN²` | `column = addr mod MLEN²` | transposed single-vector variant |
| `M_BMM` | `MLEN²` | split at `MLEN`: aligned part / MLEN = BLEN column group, remainder = head offset | head rows `[head, head+HLEN)`, columns `[column, column+BLEN)` |
| `M_BTMM` | `MLEN²` | split at `MLEN`: aligned part / MLEN = BLEN row block, remainder = head offset, plus `head_selector · HLEN` | rows `[row, row+BLEN)`, columns `[head, head+HLEN)` |
| `M_BMV` | `MLEN·BLEN` | split at `MLEN` | as `M_BMM`, single vector |
| `M_BTMV` | `MLEN²` | split at `MLEN` | as `M_BTMM`, single vector |

### Why the column index is MLEN-scaled

The matrix SRAM is addressed in element units but always returns a whole
MLEN-wide vector, so the address bits below MLEN never reach the storage array.
Three pieces of RTL fix the convention:

- `memory/matrix_sram/rtl/matrix_sram_without_rounding.sv` computes
  `raddr_for_sub_sram = sram_raddr >> $clog2(MLEN * PARALLEL_DIM)`, with
  `PARALLEL_DIM = 1` (`core/rtl/plena.sv`). The low `log2(MLEN)` bits of the
  instruction's address are discarded.
- `memory/matrix_sram/rtl/subsram.sv` skews the per-bank address by the low bits
  of the shifted address: with `transposed_read == 0` sub-bank `i` reads
  `sram_index - raddr`, which gathers a **column**; with `transposed_read == 1`
  every bank reads the same address, which returns a **row**. So the low bits of
  the *shifted* address are the column index for `M_MM`/`M_MV` and the row index
  for `M_TMM`.
- `control/rtl/data_flow_control.sv` advances the operand by `MLEN` per load step
  (`m_sram_raddr_offset = m_sram_load_counter * MLEN`) and loads
  `MATRIX_LOAD_ITERATION_GEMM = BLEN` steps per matmul.

One BLEN-wide column group is therefore `BLEN * MLEN` in the operand address,
exactly as one BLEN-wide row group is for `M_TMM`.

### Broadcast head offsets are not an addressing error

The broadcast opcodes select an HLEN-wide window using the bits below MLEN, and
the machine adds `head_selector · HLEN` to the operand address itself. The RTL
does the equivalent: a non-transposed per-head load adds
`head_selector * HLEN * MLEN` to the SRAM address (the same window in the
MLEN-scaled encoding); a transposed per-head load selects the HLEN lanes in the
datapath instead. The sub-MLEN bits here are a deliberate head offset and must
not be normalised away.

### Emitter contract

Every emitter advances the matrix operand by `BLEN * MLEN` per column or row
group. The same loop usually also carries a result cursor in VRAM, which is
element-addressed and advances by `BLEN`. Where one register served both roles
the two cursors are separate registers:

- `asm_templates/ffn_asm.py` — all four variants carry `result_column_register`
  alongside `w_actual_register`
- `asm_templates/flashattn/pv.py` — `v_base_register` scaled, `out_col_register` not
- `asm_templates/projection_asm.py` — both the K-split path and the transposed
  path split `column_group` from `column_group * mlen`
- `aten/plena/isa_attention.py` — `gp_v` scaled, `gp_pv_col_base` not
- `asm_templates/vram_sub_projection_asm.py`, `aten/plena/isa_matrix.py` —
  already separated `mat_col_stride` from the result column

`src/matrix_machine.rs` pins the decomposition of `M_MM`, `M_MV`, `M_TMM`,
`M_BMM`, `M_BMV` and `M_BTMM` against an MRAM seeded with
`W[r][c] = r · MLEN + c` in bf16, so a changed convention fails on addressing
rather than silently altering results.

---

## 2. Broadcast matmul granularity

`M_BTMM` retires a **BLEN × BLEN block for each of the `MLEN / HLEN` head
lanes**, not an `MLEN × MLEN` tile per head. The compiler and the emulator both
assumed the wide shape, which is why they agreed with each other and the
end-to-end check passed while neither agreed with the RTL.

### The measurement

Run on SimTop at the RTL's own parameterisation (MLEN=16, BLEN=4, HLEN=8, so
HEAD_COUNT=2) with `PLENA_RTL/tools/testworkloads/utils/btmm_granularity_probe.py`,
which installs one `M_BTMM` plus one `M_BMM_WO` over the `linear` workload's
prefetch prologue and counts the rows the writeout commits:

    one M_BTMM + M_BMM_WO wrote 8 rows: [0, 1, 2, 3, 16, 17, 18, 19]

Eight rows is `HEAD_COUNT * BLEN`. The pattern makes it unambiguous: two groups
of exactly BLEN rows, separated by exactly MLEN. Rows 4–15 stayed zero while rows
16–19 were written, so the gap is not missing input — the same staged activations
feed both groups. Each written row carries BLEN non-zero elements, so one issue
covers a BLEN × BLEN block per head.

Three independent sources agree:

- `mxint_systolic_mcu.sv`: `localparam int PH_DRAIN_ROWS = HEAD_COUNT * BLEN;`
  with `drain_last = per_head_exe ? (PH_DRAIN_ROWS - 1) : (BLEN - 1)`.
- `customISA_lib.json`: `M_BMM_WO.alone = 6 + (MLEN // HLEN) * BLEN`, which is
  `PH_DRAIN_ROWS + 6`.
- The architecture: the array is partitioned into `MLEN / HLEN` cores, each
  running a `(BLEN, HLEN) × (HLEN, BLEN)` GEMM in parallel.

The reduction therefore runs over HLEN — the head dimension — inside one issue,
so a score block needs no accumulation loop: every issue completes its block and
is drained by its own `M_BMM_WO`.

### The contract

`M_BTMM`'s operand address decomposes exactly as `M_TMM`'s, with a head lane
added:

| field | source | selects |
| --- | --- | --- |
| tile | `addr / MLEN²` | the matrix SRAM tile |
| row block | `(addr mod MLEN²) / MLEN`, BLEN-aligned | BLEN rows of the tile |
| head window | `addr mod MLEN`, HLEN-aligned, plus `head_selector · HLEN` | the HLEN columns of one KV head |

The activation operand supplies BLEN VRAM rows, each read as `broadcast_amount`
HLEN-wide head lanes. `M_BMM_WO` decomposes like `M_MM_WO`: the MLEN-aligned part
of its address is the destination row, the remainder is the BLEN-wide column
group, and head lane `j` sits one MLEN-row block further on, so head `j` row `i`
lands at `base + (j * MLEN + i) * MLEN`.

Covering one `MLEN × MLEN` score tile therefore takes `(MLEN / BLEN)²` issue and
drain pairs. `asm_templates/flashattn/qkt.py` emits them as a two-deep nest:

    outer : query row blocks, Q advancing by BLEN * MLEN
    inner : key row blocks,   the matrix operand advancing by BLEN * MLEN
            and the score column advancing by BLEN

`aten/plena/program_attention.py` already used this addressing, so the two
compiler paths agree.

### What changed

| component | before | now |
| --- | --- | --- |
| `matrix_machine.rs` `btmm`/`bmm` | read MLEN activation rows, produced `[broadcast, MLEN, MLEN]` | read BLEN rows, produce `[broadcast, BLEN, BLEN]` |
| `matrix_machine.rs` `bmm_wo` | wrote MLEN full rows per head | writes BLEN rows × BLEN columns per head |
| broadcast accumulate cost | `(MLEN/BLEN)² · (3·BLEN + 11)` | `3·BLEN + 11`, one issue |
| `flashattn/qkt.py` batched path | one `M_BTMM` + one `M_BMM_WO` per tile | `ceil(q_rows/BLEN) · (MLEN/BLEN)` pairs in a nest |

Total matmul cycles are unchanged by construction — `(MLEN/BLEN)²` issues at
`3·BLEN + 11` each is what the single wide issue was already charged — so the
correction shows up as instruction count and as drain cycles, which the wide
model never charged.

### Cost of the drain

Each issue carries its own `M_BMM_WO` at `(MLEN/HLEN) · BLEN + 6` cycles against
an issue of `3 · BLEN + 11`. At MLEN=64, BLEN=4, HLEN=16 that is 22 against 23;
at the reported MLEN=1024, BLEN=4, HLEN=128 point it is 38 against 23. The
serialized drain is therefore a material matrix-side cost.

### Overlapping the drain

`mxint_systolic_mcu.sv` accepts a new instruction only when it is not draining:

```systemverilog
end else if (!draining && control != OP_STALL_M) begin
    control_exe  <= control;
```

so every accumulate waits for the previous writeout to stream out. Serialized,
one issue and drain pair costs `(3·BLEN + 11) + ((MLEN/HLEN)·BLEN + 6)`.

Overlapping them is one concrete change: the drain reads the live accumulator,
which the next accumulate immediately overwrites, so an implementation needs a
second bank. At MLEN=1024, HLEN=128 and BLEN=4, duplicating the packed and plain
32-bit accumulator state is **576 B of raw registers per chip**, or 4,032 B over
the seven-chip headline configuration, before muxing, control, clocking, routing,
or implementation overhead. This is not a silicon-area result; no synthesis
evidence exists.

The bound is `(drain + issue) / max(drain, issue)`, so it peaks where the two are
balanced and falls off as either dominates.

`DRAIN_OVERLAPPED = 1` in the emulator's `TRANSACTIONAL.CONFIG` prices a
writeout at one issue slot, and the analytic model carries the matching
`drain_overlapped` timing contract. It is a co-design sensitivity: the current
publication table generates its throughput effect from the analytic model and
leaves power and energy efficiency blank pending synthesis and power
characterisation. The canonical emulator calibrations in
`analytic_models/performance/calibration/` use the implemented serialized
contract. No overlap result is presented here as an RTL measurement.

The contract is off by default because it does not describe the current RTL.
`ideal_matrix_pipeline` is a different and much stronger claim: it also shortens
the accumulate itself from `3·BLEN + 11` to `BLEN`, which no implementation in
this tree has demonstrated.

### Which opcodes the RTL decodes

Read from `PLENA_RTL/src/frontend/rtl/decoder.sv` and
`src/definitions/operation.svh`, pinned by
`compiler/asm_templates/tests/test_rtl_decoder_opcode_coverage.py`.
`operation.svh` assigns an encoding to the whole matrix family, but the decoder's
matrix case arm matches only:

    M_MM, M_TMM, M_BMM, M_BTMM, M_MM_WO, M_BMM_WO, M_MV, M_TMV, M_MV_WO

`M_BMV` (6'h09), `M_BTMV` (6'h0A) and `M_BMV_WO` (6'h0C) have encodings but no
decoder arm, so they fall through to `STALL_M` and never issue. This splits the
two attention lowerings:

| lowering | QKᵀ | PV | decoded |
| --- | --- | --- | --- |
| batch-packed (`q_len > 1`) | `M_BTMM` + `M_BMM_WO` | `M_MM` + `M_MM_WO` | yes |
| single-token (`q_len == 1`) | `M_BTMV` + `M_BMV_WO` | `M_MV` + `M_MV_WO` | no |

The batch-packed formulation is therefore the only decode attention path the
current RTL can execute; a single-token program stalls whatever the granularity
is. The decode program takes it, because `stage` is `"prefill"` whenever
`q_len > 1`.

### Why the broadcast form and not per-head `M_TMM`

`_qkt_per_head_prefill` is also BLEN-granular and `M_TMM` is decoded, so it was
the other way to make decode attention hardware-correct. It was not taken:

- `M_TMM` reduces over BLEN per issue while `M_BTMM` reduces over HLEN, and it
  covers one head where `M_BTMM` covers `MLEN / HLEN`. Per score tile the
  per-head route costs `broadcast_amount · (HLEN / BLEN)` times as many issues —
  16× at MLEN=64/BLEN=4/HLEN=16 — which is exactly the head parallelism the
  flattened array exists to exploit.
- `M_TMM` takes an MLEN-scaled operand index with no head lane, so per-head QKᵀ
  needs every head's Q MLEN-aligned: the packed group layout, not the row-packed
  layout decode uses. That is a re-layout of Q, the O accumulator, and the
  softmax and PV addressing that read them.

The per-head path remains for geometries where the broadcast group does not match
the GQA ratio; its operand addressing is the open item in
[section 4](#the-remaining-geometry-restriction).

---

## 3. Validating the attention stage

### The fixed-tolerance check could not see attention

`decoder_decode_test.py` gates on `|err| <= atol + rtol·|golden|` with
`atol = 0.2`, chosen against the residual stream, whose elements are order 1.
Attention output elements are far smaller and shrink as the cache grows, because
softmax averages over more keys, so the constant swamps the signal. Measured at
MLEN=64/BLEN=4/HLEN=16 while attention was reading only the first key tile:

| kv | key tiles | rms(O) | mean&#124;err&#124; / rms | correlation | fixed-tolerance verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 2 | 0.0592 | **87%** | +0.52 | PASS 99.80% |
| 256 | 4 | 0.0284 | 163% | +0.22 | PASS 99.80% |
| 512 | 8 | 0.0201 | 249% | +0.43 | PASS 99.88% |
| 1024 | 16 | 0.0166 | **344%** | +0.26 | PASS 99.51% |

At kv=1024 the mean absolute error was more than three times the rms of the
correct answer and the correlation was +0.26. The end-to-end check still passed
because the layer output is dominated by the residual stream:
`O_proj = x + W_O(O) + FFN(...)` with `rms(x) ≈ 0.5` against `rms(O) ≈ 0.017`.
A cross-correlation of every emulator head against every reference head was
diagonal-dominant, so this was not a permutation artefact in the checker.

### The signal-relative rule

`misc/decode_signal_relative_check.py` applies

    tolerance(stage) = 1e-3 + 0.25 · rms(golden_stage)

The bound follows the stage's own rms because block-scaled MXFP8 accumulates a
*relative* error set by the format and the reduction length, not an absolute one.
One rule then holds at every geometry and cache length, which a constant atol
cannot. The floor covers exactly-zero regions such as KV zero padding. The rule
discriminates rather than merely tightening: the stages that were correct passed
it comfortably and the broken one failed.

The bound is per stage. A stage at the end of a long chain also carries every
upstream stage's error, so its end-to-end residue is not a statement about its own
arithmetic. The tool therefore also reports *local* agreement for stages whose
input survives in the dump — the same reference recomputed from the emulator's own
input.

### The defect the rule found: every key tile re-read the first

The K/V cache is `(kv_len, MLEN)` row-major, so key tile `t` begins at
`t · MLEN · MLEN` elements. Every prefetch address was built from the KV-head
index alone:

- `flashattn/overall.py` — `_emit_k_prefetch` / `_emit_v_prefetch` took only a
  head index, and the pipelined schedule passed `kv_iters[idx][0]`, discarding the
  tile index in `kv_iters[idx][1]`.
- `flashattn/qkt.py` — the non-pipelined K prefetch used `k_head_index * d`.
- `flashattn/pv.py` — the V prefetch used `v_head_index * head_dim`.

Each now adds the key-tile offset, threaded from a bound `k_tile_index`. Effect on
the signal-relative check:

| kv | O before (corr / mean&#124;err&#124;/rms) | O after |
| ---: | --- | --- |
| 128 | 0.516 / 87% | **0.999 / 3.7%** |
| 256 | 0.217 / 163% | **0.999 / 3.7%** |
| 512 | 0.432 / 249% | **0.999 / 3.3%** |
| 1024 | 0.263 / 344% | **0.999 / 3.5%** |

`compiler/asm_templates/tests/test_flashattn_kv_tile_addressing.py` asserts that
consecutive key tiles are fetched from distinct addresses, that the stride between
them is exactly one `MLEN · MLEN` cache tile, and that a single-tile cache still
uses only the base address. Reverting the offset fails two of the three.

### How it was found, and one hypothesis that was wrong

Reasoning from the layer output would not have found it, because the layer output
is dominated by the residual stream. The kernel was bisected instead: the `_gqa_S`
and `_gqa_PV` scratch buffers were dumped from VRAM and matched against every
reference orientation, head and key tile rather than an assumed layout. `_gqa_PV`
turned out to be packed by head into column lanes, and `O` to be a single packed
tile — either assumption would have produced a false negative. Every buffer's best
match was key tile 0, at correlation 0.66–0.96.

An earlier hypothesis was refuted: the flash-attention template reset the running
max, the running sum and the output accumulator inside the key-tile loop and
applied the `1/l` row scaling per tile, which is not how flash attention folds
tiles into one state. Hoisting all four out is the structurally correct shape and
removed 84 duplicated instructions per layer at kv=128 (3202 → 3118), but left the
numerical result **bit-identical**. The hoist is kept because it is correct and is
a measured instruction reduction, not because it fixed anything.

### Current retained agreement

The source-current MLEN=64/BLEN=4/HLEN=16 runs at cache lengths 128, 256, 512,
and 1024 all pass the unchanged end-to-end allclose gate at 100.00%. Their
content-addressed per-stage timing agreement is reported in
`analytic_models/performance/calibration.md`; each artifact binds the assembly,
op-stats, settings, ISA library, run manifest, run receipt, emulator binary, and
analytic sources. The earlier signal-relative sweep remains a diagnostic that
found the repeated-key-tile defect, not a retained publication table.

---

## 4. Scaling across array geometries

The decode testbench derives its shape from `TRANSACTIONAL.CONFIG` rather than
carrying its own constants (`decoder_decode_asm_gen.decode_geometry`):

- HLEN is the head lane width, so **head_dim = HLEN**.
- The array holds MLEN/HLEN head lanes, which is what one broadcast matmul
  covers, so **GQA ratio = query heads = MLEN/HLEN** with one KV head.
- **hidden = heads · head_dim = MLEN**.
- BLEN is independent: it is the systolic block length. Packed GQA uses the
  batched attention path when the group ratio fits `BROADCAST_AMOUNT`; the
  decoder geometry uses the full `MLEN / HLEN` broadcast group.
- VRAM matrices register in whole MLEN-row tiles, so the packed decode batch is
  **one full query tile, batch = MLEN**.

### Geometry sensitivity

The larger-array sweep is a synthetic compiler/emulator sensitivity. Matching
HLEN=128 to a model head dimension does not reproduce Qwen3-32B: hidden width,
layer count, vocabulary, batching, cache placement, chip partitioning, precision
support, and RTL elaboration remain separate contracts. Consequently the sweep
is useful for exposing allocation, immediate-width, and watchdog failures, but
is not retained as Qwen correctness, RTL validation, or model-throughput
evidence. Only the MLEN=64 calibration points above are currently receipt-bound
to source-current emulator runs.

The committed RTL configuration is MLEN=16, BLEN=4, HLEN=8, VLEN=16. Any other
geometry requires its own emitted configuration and RTL run before it can support
an RTL claim. The emulator records its geometry in `decode_run_manifest.json`,
and stage validation rejects a requested shape that contradicts that manifest.

### The MLEN=256 abort: HBM allocation

Below MLEN=256 the program ran; at MLEN=256 the emulator aborted serialising a
non-finite quantized tensor. Four candidates were tested and refuted: softmax
overflow, VRAM capacity, 18-bit immediate overflow, and non-finite input data.

The cause was found by making the emulator name the address. A magnitude trap on
SRAM traffic (`PLENA_TRAP_VRAM_MAGNITUDE`, in `lib/sram`) reported the first
out-of-range value as **MRAM tile 0, element 0 = 4.3e17** — a weight tile, before
any arithmetic had touched it. Comparing the compiler's HBM addresses against the
stager's layout showed W_Q at 655,360 where the stager had written it to 368,640.

`PlenaCompiler._allocate_hbm` rounded every allocation up to an `MLEN · MLEN`
boundary once `MLEN >= 256`, while `create_mem_for_sim` writes tensors back to
back as `[elements][scales]` pairs padded only to an HBM row. Every tensor after
the first sat at an address the stager never wrote, and the weights decoded from a
neighbour's bytes. The justification recorded for the padding — that
`continous_write_delayed` requires tile alignment — applies to that function's
**MRAM destination**, not to the HBM source address it was applied to.

The padding is removed; the allocator advances by exactly the footprint
`hbm_tensor_size` reports, which is the stager's own rule.
`compiler/asm_templates/tests/test_hbm_allocation_matches_stager.py` pins the two
together across the ladder. At MLEN=64 and 128 the alignment was already inactive,
so those rungs are unchanged.

### Immediate-field overflow above MLEN=256

`S_ADDI_INT` carries an 18-bit immediate, and `MLEN · MLEN` reaches exactly `2¹⁸`
at MLEN=512. The emitters that build strides from raw text — `ffn_asm`,
`projection_asm`, `gemv_asm`, `batched_matmul_asm` — overflowed it, and the
assembler correctly refused to truncate.

Rather than edit each of the thirty-odd sites, `PlenaCompiler.compile()` runs
`asm_templates._imm.legalize_immediates` over the emitted text: a `gp0` source
becomes a wide load, any other source becomes the chunked relative add, which
needs no scratch register and so is safe after register allocation. Every template
is covered, including ones that do not route their arithmetic through the helpers.

### Scalar FP SRAM bounds live GQA groups

The online softmax keeps three running scalars per query row and query head on top
of six shared constants. Query rows tile freely, so an untiled full-array count is
not a hardware cost. The binding allocation is one full BLEN-row query block for
each simultaneously live KV group:

    6 + 3 · BLEN · (MLEN / HLEN) · groups

At MLEN=1024, HLEN=128 and BLEN=4, four groups require 390 of the RTL's 512
slots. Keeping all eight groups live requires 774 slots, or **393 B more scalar
SRAM**. This is the cost attached to the one-read traffic bound.

`softmax_row_tile` snaps the row tile to a multiple of BLEN, which makes the
emitted query-block count exactly `ceil(rows / BLEN)`. The emulator's FP SRAM is
driven by `TRANSACTIONAL.CONFIG.FP_SRAM_DEPTH`, so an infeasible configuration is
rejected explicitly.

The tiling costs no KV traffic: packed query rows are separate sequences holding
their own caches, so a row tile reads its own rows' caches and nothing else. The
QKᵀ nest scores only the tile's rows; the compiler tests pin that instruction
structure and the final emulator measurements above cover the resulting program.

### The remaining geometry restriction

`ratio != BLEN` selects the per-head path (`use_batched` in
`flashattn/overall.py`), which addresses Q at `head_dim` granularity while `M_TMM`
takes an MLEN-scaled index with no head lane. Per
[section 1](#1-matrix-operand-addressing) only the `M_B*` forms carry a head
remainder, so per-head QKᵀ needs every head's Q MLEN-aligned — the packed group
layout, not the row-packed layout decode uses. The synthetic sweep keeps
`BLEN == MLEN / HLEN` so every point takes the batched path, which is also the
only one the RTL decoder can execute.

### Runaway-loop watchdog

`MAX_LOOP_INSTRUCTIONS` defaults to 100,000. At MLEN=1024 the QKᵀ nest retires
more than that inside one outer iteration, so the guard fires before the program
finishes. It is a configured safety limit rather than a hardware bound; the
MLEN=1024 rung raises it in its settings copy.

---

## 5. KV cache traffic

The default lowering prefetches the packed KV row once for each KV head, so its
attention traffic amplification is `kv_heads`. The Qwen3-32B hkv=8 case is an
analytic 8× traffic bound, conditional on a selector-capable implementation;
it is not one of the emulator geometries below. `M_BTMM` carries a head-selector
field that picks a head window out of a resident tile, so the reuse schedule moves
the KV-head loop inside the key-tile loop and selects each head from one resident
row.

The cost is scalar FP SRAM. Keeping `g` KV groups live means their online-softmax
state is live together, so a query block of BLEN rows holds

    6 + 3 · BLEN · (MLEN / HLEN) · g

slots. Only a full M_BTMM query block has to fit, not the whole MLEN row: packed
query rows are separate sequences, so tiling the batch splits it rather than
re-reading any cache, and `query_blocks = ceil(q_rows / BLEN)` means a tile below
BLEN wastes array fill while a tile above it buys nothing.

For the analytic Qwen3-32B co-design point at MLEN=1024, HLEN=128, BLEN=4,
against `FP_SRAM_DEPTH = 512` in
`PLENA_RTL/src/definitions/configuration.svh`:

| KV groups live | slots needed | vs 512 | reads per token |
| ---: | ---: | ---: | ---: |
| 1 (today) | 102 | 0.20× | 8× |
| 2 | 198 | 0.39× | 4× |
| 4 | 390 | 0.76× | 2× |
| 8 (one read per token) | 774 | 1.51× | 1× |

The configured depth affords four live groups in this analytic mapping. One read
per token needs 774 slots, **393 B more scalar SRAM per chip**, or 2,751 B over
seven chips. These are raw storage deltas, not synthesized area. The slot formula
is pinned by the multi-head compiler tests and reported by
`analytic_models/performance/fp_sram_sweep.py`; the source-current emulator
evidence retained below isolates the KV-head traffic axis at MLEN=64.

Two earlier readings of this are superseded. The head selector was first claimed to
be worth 1.56× throughput and 4.8× tokens per joule, which double-counted; it was
then argued to cancel identically against the query-row tile, on the assumption
that each row tile re-sweeps a shared cache. Row tiles cover disjoint sequences, so
nothing cancels.

### The schedule, and what it emits

`flash_attn_asm(..., kv_head_reuse=True)` hoists the KV-head loop inside the
key-tile loop. One `H_PREFETCH_M` brings the packed row into the matrix SRAM and
each head is read out of it with `M_BTMM <selector>`; `M_MM` walks the same
resident tile for PV, offset by `selector * head_dim * MLEN` because the matrix
operand scales its column index by MLEN. Both schedules given the same
query-row tile, at MLEN=64, HLEN=16, BLEN=4, 4 key tiles:

| KV heads | schedule | slots | prefetches | per key tile | saving |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | per-head | 54 | 128 | 32 | — |
| 1 | reuse | 54 | 128 | 32 | 1.00× |
| 2 | per-head | 54 | 256 | 64 | — |
| 2 | reuse | 102 | 128 | 32 | 2.00× |
| 4 | per-head | 54 | 512 | 128 | — |
| 4 | reuse | 198 | 128 | 32 | 4.00× |

Prefetches per key tile stop depending on the KV head count, so the saving is
exactly `hkv`, and the slot counts are the `6 + 3 · BLEN · lanes · g` above. The
row-tile axis does not cancel it: the two schedules are compared at one tile, and
the saving is the same whether that tile is one `M_BTMM` query block or the whole
MLEN row. `compiler/asm_templates/tests/test_kv_head_reuse.py` pins both halves.

The complete decoder was emulator-run at kv=128 with the row tile fixed at four
for both schedules. K, V, and non-attention bytes are issue-origin physical
traffic from source-tagged dynamic instructions; their sum reconciles exactly
to the emulator global counter in every receipt:

| hkv | cycles per-head | cycles reuse | K=V per-head | K=V reuse | other bytes | global per-head | global reuse | allclose |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 223,899 | 223,903 | 262,144 B | 262,144 B | 139,264 B | 663,552 B | 663,552 B | 100.00% |
| 2 | 319,124 | 317,184 | 524,288 B | 262,144 B | 155,648 B | 1,204,224 B | 679,936 B | 100.00% |
| 4 | 506,991 | 501,179 | 1,048,576 B | 262,144 B | 188,416 B | 2,285,568 B | 712,704 B | 100.00% |

The complete VRAM dump is bit-identical between schedules for each hkv, so the
traffic change does not change numerical output. K and V each fall by exactly
2× at hkv=2 and 4× at hkv=4. The reuse nest pipelines K before QK, V behind the
first QK, and the next K behind row finalization; measured latency is neutral at
hkv=1 and improves 0.61% at hkv=2 and 1.15% at hkv=4.

This verifies the mechanism and traffic at the recorded emulator geometry, not
the Qwen3 headline. That row remains an analytic co-design traffic bound: it
uses MLEN=1024, HLEN=128 and hkv=8, its compute model differs from the per-head
row, and the active MXFP RTL profile marks the selector unsupported (the RTL
selector implementation is MXINT-only).

### Retained traffic evidence

The manifest below records the six emulator runs behind the table above. It is
provenance, not prose: **a reader following the argument can skip it.** It is
embedded here rather than kept as a separate file so that the evidence cannot
drift away from the claim it supports, and
`analytic_models/performance/test_decode_stage_validation.py` parses it straight
out of this document and re-verifies it on every test run.

Reading it, if you need to:

- `behavior_contract` is the hardware configuration the runs used — array
  geometry, SRAM sizes, HBM generation and channel count, and the precision of
  each memory region.
- `cells` holds one entry per run, each with its command, byte counters and
  result checksums.
- The `*_sha256` fields are SHA-256 digests of the exact bytes of each input and
  output: the assembly, machine code, HBM preload image, golden result, run
  manifest and receipt. They exist so a later run can be proved identical to
  this one, and are not meant to be read by eye.
- `aggregate_sha256` is the digest of the whole manifest — SHA-256 over the
  canonical compact JSON with sorted keys, computed after removing that field
  itself. The test recomputes it, so editing any value inside this block will
  fail the suite until the digest is recomputed too.
- `command_prefix`, `environment` and `working_directory` record the machine
  where the runs happened. They are historical facts about that execution, not
  paths anything reads today; the evidence is bound by content hash, so it stays
  valid on any checkout.

Each receipt was revalidated against its settings, executable, input inventory,
op-statistics totals, issue-origin ledger, and numerical result.

<!-- decode-kv-traffic-evidence:start -->
```json
{
  "aggregate_sha256": "b3b0fb568240b010eaf55b936ce9d742867753772eb03b2a2828e2b8a9c382e6",
  "behavior_contract": {
    "BLEN": 4,
    "BROADCAST_AMOUNT": 4,
    "DRAIN_OVERLAPPED": false,
    "FP_SRAM_DEPTH": 512,
    "HBM_CHANNELS": 8,
    "HBM_GEN": "HBM2",
    "HBM_M_Prefetch_Amount": 64,
    "HBM_SIZE": 17179869184,
    "HBM_V_Prefetch_Amount": 4,
    "HBM_V_Writeback_Amount": 4,
    "HLEN": 16,
    "MATRIX_SRAM_SIZE": 4096,
    "MLEN": 64,
    "PRECISION": {
      "HBM_M_KV_TYPE": {
        "ELEM": {"exponent": 4, "mantissa": 3, "sign": true, "type": "Fp"},
        "SCALE": {"exponent": 8, "mantissa": 0, "sign": false, "type": "Fp"},
        "block": 8,
        "format": "Mx"
      },
      "HBM_M_WEIGHT_TYPE": {
        "ELEM": {"exponent": 4, "mantissa": 3, "sign": true, "type": "Fp"},
        "SCALE": {"exponent": 8, "mantissa": 0, "sign": false, "type": "Fp"},
        "block": 8,
        "format": "Mx"
      },
      "HBM_V_ACT_TYPE": {
        "ELEM": {"exponent": 4, "mantissa": 3, "sign": true, "type": "Fp"},
        "SCALE": {"exponent": 8, "mantissa": 0, "sign": false, "type": "Fp"},
        "block": 8,
        "format": "Mx"
      },
      "HBM_V_KV_TYPE": {
        "ELEM": {"exponent": 4, "mantissa": 3, "sign": true, "type": "Fp"},
        "SCALE": {"exponent": 8, "mantissa": 0, "sign": false, "type": "Fp"},
        "block": 8,
        "format": "Mx"
      },
      "MATRIX_SRAM_TYPE": {
        "DATA_TYPE": {"exponent": 8, "mantissa": 7, "sign": true, "type": "Fp"},
        "format": "Plain"
      },
      "VECTOR_SRAM_TYPE": {
        "DATA_TYPE": {"exponent": 8, "mantissa": 7, "sign": true, "type": "Fp"},
        "format": "Plain"
      }
    },
    "VECTOR_SRAM_SIZE": 65536,
    "VLEN": 64
  },
  "command_prefix": [
    "/home/sr1325/PLENA_Software/.venv/bin/python",
    "transactional_emulator/testbench/misc/decoder_decode_test.py",
    "--kv-size",
    "128"
  ],
  "emulator_binary_sha256": "88bc0e748168eaa6d3575e1da138f715dbdb79191d399b2e93c8480d28065c72",
  "environment": {
    "PLENA_SETTINGS_TOML": "/home/sr1325/PLENA_Simulator/plena_settings.toml",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/home/sr1325/PLENA_Simulator:/home/sr1325/PLENA_Simulator/PLENA_Tools:/home/sr1325/PLENA_Simulator/compiler:/home/sr1325/PLENA_Simulator/transactional_emulator/testbench"
  },
  "schema": "plena-decode-kv-traffic-evidence-v1",
  "settings_sha256": "16e57a9a81018a558120b54d034f021f79ba029aa21a4d6b1e7f7e6025629cb7",
  "working_directory": "/home/sr1325/PLENA_Simulator",
  "cells": [
    {
      "allclose_pass": true,
      "arguments": ["--kv-heads", "1", "--softmax-row-tile", "4", "--build-dir", "/tmp/plena_decode_final_matrix.8ityPN/hkv1_default"],
      "asm_sha256": "896b2df7f6a277db67b055ad3738b6811457904f43ed22e7fb547fcf15ab3f25",
      "behavior_contract_sha256": "309c4ee98b3e05a0f262842150e7eed2e9823cc455ca4087c7af1eadd6201ad0",
      "cell": "hkv1_default",
      "comparison_params_sha256": "160516ce79feebf2c5a7eeead05828cf43653c0acd1217010fd286c16e6e27c6",
      "cycles": 223899,
      "global_bytes": 663552,
      "golden_result_sha256": "a52981e2f12a189d8622609048c54aeac29e1cb728bfa6457f5c8d12cf6980a9",
      "hbm_preload_sha256": "92a00a11063bba73754f74721a6ede641fd5d53d4edef27ea3257b4d9b4ebbbc",
      "key_bytes": 262144,
      "kv_heads": 1,
      "machine_code_sha256": "ec4488ccec468f7226623c69133d6efd5843d8244296a276c64d4921fc91454e",
      "non_attention_bytes": 139264,
      "op_stats_sha256": "5568f04ebfe8f62a7f5cf63504c158e8f2c9a2e400f044c8784550707f480f8e",
      "receipt_sha256": "44b6512e0bef3ee44e8a9aa2c3b329e484dfb866a3e55300f74fe0b8f49afcb6",
      "reuse": false,
      "run_manifest_sha256": "d9e5a274888f093f3396bbfceeeb6a555b061010684f37a24ba631872d10a06e",
      "value_bytes": 262144,
      "vram_sha256": "cfb04c3b9f222e564a3b519fafcff655357d826b5b9731e560c6beaac4181f14"
    },
    {
      "allclose_pass": true,
      "arguments": ["--kv-heads", "1", "--softmax-row-tile", "4", "--kv-head-reuse", "--build-dir", "/tmp/plena_decode_final_matrix.8ityPN/hkv1_reuse"],
      "asm_sha256": "1d05cf34c64ec81de0a9134ec225e1a0240ce9896498c7989916dd9f3c1cb829",
      "behavior_contract_sha256": "309c4ee98b3e05a0f262842150e7eed2e9823cc455ca4087c7af1eadd6201ad0",
      "cell": "hkv1_reuse",
      "comparison_params_sha256": "160516ce79feebf2c5a7eeead05828cf43653c0acd1217010fd286c16e6e27c6",
      "cycles": 223903,
      "global_bytes": 663552,
      "golden_result_sha256": "a52981e2f12a189d8622609048c54aeac29e1cb728bfa6457f5c8d12cf6980a9",
      "hbm_preload_sha256": "92a00a11063bba73754f74721a6ede641fd5d53d4edef27ea3257b4d9b4ebbbc",
      "key_bytes": 262144,
      "kv_heads": 1,
      "machine_code_sha256": "f2248ba5d49face838f2050ccc0eebd2379ce8da91b3b46f736ca05a7446ce3f",
      "non_attention_bytes": 139264,
      "op_stats_sha256": "17383d73341492b08eaa3c9a1eb16479a4ba1ae6ab481420dc1aa6bc7d5d00d4",
      "receipt_sha256": "7608b241addfc75a47102f1c7e321f54d3cc483e3a196b7caf0799b59b9055d5",
      "reuse": true,
      "run_manifest_sha256": "7c03cc68ddfa4a7d13497f0ea6341532a2cbad79b4b7c2afdd58567def51b4dd",
      "value_bytes": 262144,
      "vram_sha256": "cfb04c3b9f222e564a3b519fafcff655357d826b5b9731e560c6beaac4181f14"
    },
    {
      "allclose_pass": true,
      "arguments": ["--kv-heads", "2", "--softmax-row-tile", "4", "--build-dir", "/tmp/plena_decode_final_matrix.8ityPN/hkv2_default"],
      "asm_sha256": "d2eb279c1a6a6736d8f3df0183f305ad2372427f723ee0e97153dd60fecb1ca1",
      "behavior_contract_sha256": "309c4ee98b3e05a0f262842150e7eed2e9823cc455ca4087c7af1eadd6201ad0",
      "cell": "hkv2_default",
      "comparison_params_sha256": "11ad2765822d9ff3d35c8dac11a4672164046d231bfe1369cfe446d437e39a54",
      "cycles": 319124,
      "global_bytes": 1204224,
      "golden_result_sha256": "c30e203bace8cab57ae3de0488aadd7484ab57c52812a9d39193471b78a2ef96",
      "hbm_preload_sha256": "04fb0889ae0e5a4916ae0701164166231acaefb8f85facf158c1e80b9165d772",
      "key_bytes": 524288,
      "kv_heads": 2,
      "machine_code_sha256": "b2d82ec30069dfdb9ff7c2a715ae3cbb45ef1424bd7ff22cca40654d16aaaeca",
      "non_attention_bytes": 155648,
      "op_stats_sha256": "da308e3ad8ff0ae02af5c72b25694b664e5b5a102650288302dc86fd4f21cfc0",
      "receipt_sha256": "64d4ea62fa83a5d1048f8cb3e1ec44924b4e143ba08eca72bb6c98d57ed0685e",
      "reuse": false,
      "run_manifest_sha256": "7e1a1f7b0ec420987d0c896f6f03a89fc099ef1a4117b570e4b0106c7526d186",
      "value_bytes": 524288,
      "vram_sha256": "8794e56e64f9e716910bdd12fc93970e92ff43f0c98478cdd3ae2dd770437c07"
    },
    {
      "allclose_pass": true,
      "arguments": ["--kv-heads", "2", "--softmax-row-tile", "4", "--kv-head-reuse", "--build-dir", "/tmp/plena_decode_final_matrix.8ityPN/hkv2_reuse"],
      "asm_sha256": "2fbbaecd0bad1bcdd6923e3df82e49907613f439958ec342c11e9cf8e3347a95",
      "behavior_contract_sha256": "309c4ee98b3e05a0f262842150e7eed2e9823cc455ca4087c7af1eadd6201ad0",
      "cell": "hkv2_reuse",
      "comparison_params_sha256": "11ad2765822d9ff3d35c8dac11a4672164046d231bfe1369cfe446d437e39a54",
      "cycles": 317184,
      "global_bytes": 679936,
      "golden_result_sha256": "c30e203bace8cab57ae3de0488aadd7484ab57c52812a9d39193471b78a2ef96",
      "hbm_preload_sha256": "04fb0889ae0e5a4916ae0701164166231acaefb8f85facf158c1e80b9165d772",
      "key_bytes": 262144,
      "kv_heads": 2,
      "machine_code_sha256": "8cd56c2bad6756bd38cc4fc20fe648da14ed4798f5f57f6a641ebb0acdf7ca82",
      "non_attention_bytes": 155648,
      "op_stats_sha256": "2ce319894c571b296407954d52cfc1cb48e02723d0248729f05ab6541d378121",
      "receipt_sha256": "01d27d2dad63169590a19f92ba019d043b626721d5f5e797de6c7fdb21c589bf",
      "reuse": true,
      "run_manifest_sha256": "8d5b34bb307c3d4b666c391c3ed5b66e3df325cf5a8080535ec59f3a00f5f7d2",
      "value_bytes": 262144,
      "vram_sha256": "8794e56e64f9e716910bdd12fc93970e92ff43f0c98478cdd3ae2dd770437c07"
    },
    {
      "allclose_pass": true,
      "arguments": ["--kv-heads", "4", "--softmax-row-tile", "4", "--build-dir", "/tmp/plena_decode_final_matrix.8ityPN/hkv4_default"],
      "asm_sha256": "ba2475bbf4ddb4da721616b6626f70db6a86bbd6cc3009a50a548935c932aa11",
      "behavior_contract_sha256": "309c4ee98b3e05a0f262842150e7eed2e9823cc455ca4087c7af1eadd6201ad0",
      "cell": "hkv4_default",
      "comparison_params_sha256": "5dfb7c43d0d16aac25c7d59e11c207abc032d857d39a5f9cda6f35965ffcb5ce",
      "cycles": 506991,
      "global_bytes": 2285568,
      "golden_result_sha256": "87fa25cbbae535e3313581c02a35a021622210d7cd20aa957201b5f6fc8a2caa",
      "hbm_preload_sha256": "debe4f8a38712f99702141bc3116fecbb952d339947a92632eadf7fa3502df61",
      "key_bytes": 1048576,
      "kv_heads": 4,
      "machine_code_sha256": "d43631ac970ae4f90d36cbfb863564425ce980c0c97c5bbb18151da3f372736b",
      "non_attention_bytes": 188416,
      "op_stats_sha256": "d45116bb1a455c2efc631dd21a3e6fa6fa969fb53a65ca8fdd569fe2ac1b60ee",
      "receipt_sha256": "f2094c47ced32baa29400bea54e4f8ad472aab179c6acd46aa8bcbfe025a4c07",
      "reuse": false,
      "run_manifest_sha256": "0be34e9994ab88dee5de413c08338c5a7b58d3a6f455ea4c603c97968b350aa7",
      "value_bytes": 1048576,
      "vram_sha256": "0023aa7f20694bdca30117f99cc9f51722fcb3d5c6e5121aa02dced8cdd3cb23"
    },
    {
      "allclose_pass": true,
      "arguments": ["--kv-heads", "4", "--softmax-row-tile", "4", "--kv-head-reuse", "--build-dir", "/tmp/plena_decode_final_matrix.8ityPN/hkv4_reuse"],
      "asm_sha256": "e1c72c60342012d28c867fbe0bceac2390ca20eb8addc12cee3b29b126cb010a",
      "behavior_contract_sha256": "309c4ee98b3e05a0f262842150e7eed2e9823cc455ca4087c7af1eadd6201ad0",
      "cell": "hkv4_reuse",
      "comparison_params_sha256": "5dfb7c43d0d16aac25c7d59e11c207abc032d857d39a5f9cda6f35965ffcb5ce",
      "cycles": 501179,
      "global_bytes": 712704,
      "golden_result_sha256": "87fa25cbbae535e3313581c02a35a021622210d7cd20aa957201b5f6fc8a2caa",
      "hbm_preload_sha256": "debe4f8a38712f99702141bc3116fecbb952d339947a92632eadf7fa3502df61",
      "key_bytes": 262144,
      "kv_heads": 4,
      "machine_code_sha256": "9fe7ed3de1fbfa914e501ad6ef67a2e59800e671c9b32cd6255bf0905f369ba4",
      "non_attention_bytes": 188416,
      "op_stats_sha256": "a5d3b65d6e6ed510c4958fd42d04d92ccb98040b3961c3003325dd8d8952e61b",
      "receipt_sha256": "6e6844a45bd2ede50858fd993740e043f9b4ffae44b21100041b8a66814ffd5d",
      "reuse": true,
      "run_manifest_sha256": "4562ad59caf788ce0b49b0dff1ca35dbb4e30401a8a2e4fa64d3ce25b9edb1e9",
      "value_bytes": 262144,
      "vram_sha256": "0023aa7f20694bdca30117f99cc9f51722fcb3d5c6e5121aa02dced8cdd3cb23"
    }
  ]
}
```
<!-- decode-kv-traffic-evidence:end -->
