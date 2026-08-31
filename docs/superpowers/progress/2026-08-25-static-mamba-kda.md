# Static Mamba-2 + KDA — execution log

> **Superseded ISA note (2026-08-31).** This file is a historical execution
> log. The final conflict-free ABI reserves `0x39`-`0x3C` for routed MoE,
> encodes `V_FMA_VF` as the `V_MUL_VF` accumulate variant, and assigns
> `V_SOFTPLUS_V=0x3D`, `S_MAP_FP_V=0x3E`, and `L_CFG=0x3F`. Old opcode values
> below record the implementation history and are not the current contract.

Plan: [`../plans/2026-08-25-static-mamba-kda.md`](../plans/2026-08-25-static-mamba-kda.md)
Branches: `feat/static-kda` in `PLENA_Simulator` (from `65cf6f09`) and in the
`PLENA_Compiler` submodule (from `f5eb36a`), both branched off `feat/mamba2-support`.

## Environment

Not provisioned on this machine at the start. Recorded because every number below
depends on it.

| | |
|---|---|
| Python | `uv` 0.12.5, `.venv` at `PLENA_Simulator/.venv`, Python 3.12 |
| torch | 2.7.1+cu126 (via `uv sync`; CPU execution) |
| Test command | `uv run pytest`, with `PYTHONPATH=$SIM:$SIM/PLENA_Compiler:$SIM/PLENA_Tools:$SIM/transactional_emulator/testbench` |
| Rust / emulator | **`nix develop`** — the flake supplies the toolchain, Ramulator2 and libtorch |

**Correction.** I first ran `cargo` outside the Nix dev shell, hit
`unable to find library -lramulator`, and recorded it as "a C++ dependency that
is not built" blocking whole-model execution. That was wrong. `flake.nix:78`
pulls `customPkgs.ramulator2` into `buildInputs` and pre-fetches libtorch
(`:39`), `transactional_emulator/pkgs/` carries both derivations, and Nix was
already installed. Inside the shell:

```
Ramulator2:   library at /nix/store/...-ramulator-2.1-unstable-2026-07-30/lib
Libtorch:     /nix/store/...-libtorch-2.7.0/lib
cargo test -> 106 passed; 0 failed
```

The rustup toolchain I installed is redundant; the flake provides its own.
**Emulator work is not blocked.**

## Baseline — `feat/mamba2-support`, before any change

`pytest PLENA_Compiler/aten/tests` at `f5eb36a`, verified in a clean worktree:

**3 failed, pre-existing, not caused by this work:**

- `test_moe_stage_attribution.py::test_stage_parameters_have_no_default` — three
  existing emitters give `stage` a default (`program_mamba_common.py:362`,
  `program_ssd.py:321`, `program_ssm_recurrent.py:213`)
- `test_plena_compiler.py::test_qwen_packed_skinny_router_rowpacked_compiles_for_128_experts`
- `test_quantization_ablation.py::test_mxfp8_is_sole_gap_source`

Any run below reporting exactly these three is at baseline.

---

## Task 1 — KDA golden reference ✅

**Result: 3 failed (all baseline), 112 passed.** New: `test_kda_reference.py`, 9 passed.

### What landed

| File | |
|---|---|
| `PLENA_Compiler/aten/models/kda/reference.py` | 355 lines, the torch KDA forward |
| `PLENA_Compiler/aten/models/kda/state_precision.py` | 95 lines, a dependency of the above |
| `PLENA_Compiler/aten/models/kda/__init__.py` | new |
| `PLENA_Compiler/aten/tests/test_kda_reference.py` | 253 lines + 1 contract test |
| `PLENA_Compiler/.github/workflows/ci.yml` | wired the new test in |

### Two corrections to the plan, made while executing

**1. The reference belongs in the Compiler, not the Simulator.** The plan put it at
`PLENA_Simulator/analytic_models/reference/` and had the Compiler import it. That
direction does not exist: `transactional_emulator/testbench/` imports
`aten.models.gpt_oss.*` from the Compiler, and the Compiler imports nothing from
the Simulator. `aten/models/mamba2/reference.py` is the precedent. Moved to
`aten/models/kda/`; the Simulator can reach it because `PLENA_Compiler` is on its
`PYTHONPATH` (`justfile:135`).

**2. The reference API is not what the plan assumed.** Real signature:

```python
kda_step(q, k, v, gate, beta_logit, state: KdaState, a_log, dt_bias, shape: KdaShape,
         *, scale=None, state_storage=StateStorage.FP32) -> tuple[Tensor, KdaState]
```

Batched `[batch, heads, dim]`; `gate_lower_bound` lives on `KdaShape` (default
`-5.0`); `scale` defaults to `1/sqrt(key_dim)`.

Most important: normalisation is **`x * rsqrt(sum(x²) + 1e-6)`**, matching FlashKDA's
kernel, *not* `torch.nn.functional.normalize`, which clamps instead. The plan's draft
test used `F.normalize` and would have disagreed. Downstream lowering must use the
rsqrt form.

### The contract test

`test_decode_step_matches_the_transposed_formulation` pins `T[k, v] == S[v, k]` —
the load-bearing design decision of the whole plan. It replays the reference in the
transposed orientation the lowering will use and asserts both the output and the
final state agree to `rtol=1e-5, atol=1e-6`. Without it a lowering bug could hide
behind a plausible reimplementation of the maths.

### Deviation from the plan worth noting

The reference package's `__init__.py` also pulled in
`nemotron3_mamba.py`. Not carried across: a second Mamba reference alongside the
Compiler's existing `aten/models/mamba2/reference.py` is exactly the two-drifting-
copies problem the plan warns about. The KDA-only `__init__` records why.

### Guard that did its job

`test_every_test_file_here_is_wired_into_ci` failed the moment the new test file
appeared and was not in `ci.yml` — caught before commit. This is the fail-closed
guard style the review recommended, working as intended.

### Review round — `a5753d2`

A subagent ran mutation testing against the commit. **Six functional defects, all
false-greens from degenerate inputs.** The transposition contract itself was sound
(nine mutations to `reference.py` correctly turned it red), but the inputs it ran
on made four classes of bug invisible.

| # | Defect | Mutation that stayed green | Fix |
|---|---|---|---|
| D1 | `batch=1`; no test in the suite compared numerics at batch>1 | `beta`, `k_normalized`, `log_decay` each recomputed from batch 0 | `batch=2` |
| D2 | `key_dim == value_dim == 8` | output scale `1/sqrt(key_dim)` → `sqrt(value_dim)` | `key_dim=8, value_dim=5` |
| D3 | `q`/`k` at unit variance: `‖x‖²≈O(1)`, so the `1e-6` inside `rsqrt` perturbs by ~2e-7, **below the test's own `atol=1e-6`** | dropping the epsilon; swapping in `F.normalize` | scale `q`/`k` by `0.05` |
| D4 | `activate_log_decay` tested with `a_log=zeros`, `dt_bias=zeros`, and used as its own oracle everywhere else | dropping `dt_bias`; dropping `exp(a_log)` | new `test_log_decay_matches_its_closed_form` |
| D5 | `test_sequence_matches_repeated_steps` only asserted shape + `isfinite` | dropping the carried state; reversing token order | compare against repeated `kda_step` |
| D6 | `state_precision.py` present, its tests not — MX8_B128 path uncovered | — | ported both tests |

Every fix was verified by re-running the mutation it targets. All six mutants now die.

**D3 is the one to remember.** The comment in the test asserted that the FlashKDA
`rsqrt(sum + eps)` convention differs from `F.normalize` — and it does, but not at
an amplitude the test could see. A correct claim, an input that could not test it.

Also switched the package from bare `aten.` to `compiler.aten.`, matching
`aten/models/mamba2/__init__.py` and 37 other files. Both resolve to the same
directory via `compiler/__init__.py`'s `__path__` hack but produce *distinct module
objects*, so `KdaShape` imported both ways would be two classes — breaking
`isinstance` and dataclass equality as soon as the lowering imports it.

**After fixes: 3 failed (baseline), 115 passed.** `test_kda_reference.py`: 12 passed,
also verified under the CI invocation (`cd PLENA_Compiler && PYTHONPATH=. pytest`).

---

## Task 2 — KDA state layout and data movement ✅

**Result: 3 failed (baseline), 135 passed.** `test_kda_lowering.py` 14, `test_kda_stage_contract.py` 6.

### What landed

`aten/plena/program_kda_common.py` — `ProgramKdaCommonMixin`, wired into `PlenaCompiler`'s
MRO after `ProgramSSMRecurrentMixin`. State residency (`kda_pin_state_v0`,
`kda_load_state_v0`, `kda_store_state_v0`), conv history roll, L2 normalisation,
FPRAM constants. Plus `aten/models/kda/shape.py` and two new test files.

State is `[num_heads * key_dim, value_dim]` — head `h` owns rows
`[h*key_dim, (h+1)*key_dim)`. Data movement is the existing H-type path; no
descriptor, no queue, no residency table.

### Review round — `055223c` / `e4110d47`

A subagent ran mutation testing. **13 mutations survived the first version**, plus
three defects no test could have caught. All fixed; all 13 now fail the suite.

**1. I broke a CI job.** `compiler.py` imports the KDA lowering at module scope →
`reference.py` → `torch`. So every `compiler.aten.plena.*` import began requiring
torch, breaking `moe-stage-guard`, which installs only pytest and pyyaml by design.
Verified with a `meta_path` hook blocking torch: clean at parent `a5753d2`, broken at
`34692ce`, clean now. Fix: `KdaShape` moved to a torch-free `shape.py`, and
`aten/models/kda/__init__.py` re-exports lazily via PEP 562.

*(My first reproduction wrongly showed "OK" — I used the pre-3.12 `find_module`
protocol, which Python 3.12 ignores. The reviewer's finding was right and mine was
the broken experiment.)*

**2. The FP32 precision claim was false, and `4` was wrong under every config.**
`precision=1` selects the keyvalue *class*; the width comes from the active
PRECISION table, and the shipped `HBM_V_KV_TYPE` is `format = "Mx"` with e4m3
elements — 1 byte plus a scale stream. `storage_precision` only feeds compiler-side
address arithmetic. There is no FP32 path through that class at all.

Corrected to **BF16, 2 bytes** on load, store and the pinned reservation, and the
module now *requires* Plain BF16 via `kda_require_state_precision_v0` rather than
asserting it — reusing the guard `f5eb36a` added for this exact bug on the Mamba
path. This also fixes the under-reserved pinned region: under Plain BF16 no scale
stream is written, so the reservation is correct *because* of the guard.

**3. The stage markers were not markers.** Four emitters wrote `stage=kda_...`
with no `@`. `extract_stage_tag` matches `"@stage="`, so those comments were
invisible and every KDA instruction billed to the **preceding** stage — silently,
producing a profile that looks correct. Added `KDA_STAGES` + `kda_stage_marker`,
ten matching `StageKind` variants in the emulator (`cargo check` clean), and a
cross-repo contract test that also greps for hand-formatted markers.

**4. Four tests asserted nothing.**

| Test | Why it could not fail |
|---|---|
| `test_state_store_mirrors_the_load_precision` | `stored.hbm_addr == addr` is a tautology — `store()` builds the InputVar from the address it was given |
| `test_pinned_state_region_does_not_collide...` | compared an **element count** against a **byte address** |
| `test_conv_roll_shifts_history_and_appends` | counted `V_ADD_VV`; a reversed shift and a wrong append source both emit 3 |
| `test_l2_normalize_uses_the_scalar_sqrt_path` | `any(opcode in ...)` — `V_MUL_VF` comes from the zeroing idiom, `V_RED_SUM` from any surviving reduction |

Rewritten around a small `_trace()` that resolves `S_ADDI_INT gpN, gp0, imm` so
assertions name **resolved operand addresses**. One mutation (block-copy src/dst
swap, which zeroes the vector) survived even the repair — every operand I checked
was unchanged — and needed an explicit assertion on the copy's direction.

**5. Smaller functional fixes.** `mamba_fp_constants` now takes no shape (it never
read one), removing a `Mamba2Shape.__new__` uninitialised-frozen-dataclass hack.
`kda_l2_normalize_v0` rejects a scratch aliased to its input (would silently return
zeros), an undersized accumulator, and a vector wider than one column tile (the
`tile_col_idx=0` default would normalise by a partial norm — and `KdaShape.kimi_k3()`
has `key_dim=128` against a default `mlen=64`).

### Lesson carried forward

Both review rounds found the same shape of bug: **the assertion was right, the
setup could not exercise it**. Task 1 was degenerate inputs; Task 2 was
opcode-presence instead of operand identity. Tests for later tasks assert resolved
addresses and counts from the start.

---

## Task 3 — The KDA decode step (existing instructions only) ✅

**Result: 3 failed (baseline), 160 passed.** `test_kda_decode_step.py` 12,
`test_isa_interpreter.py` 12.

### The claim Phase 0 exists to make, now demonstrated

`kda_decode_step_v0` reproduces `kda_step`'s output **and** updated state across
four geometries, **using no instruction that did not already exist**. Seven
opcodes: `S_ADDI_INT`, `S_LD_FP`, `V_MUL_VF`, `V_ADD_VV`, `V_SUB_VV`,
`C_LOOP_START`, `C_LOOP_END`.

**Phase 0 baseline for Task 8: `static=505, dynamic=540`** (1 head, key_dim=8).

### How it is validated without the emulator

`aten/tests/isa_interpreter.py` — a ~200-line interpreter for exactly those seven
opcodes — executes the emitted assembly, so the tests compare numbers against the
CPU reference rather than asserting which opcodes appear. It raises on anything
outside its subset, so a lowering that starts emitting something new fails loudly.

This is a second implementation of ISA semantics and could drift; the whole-model
emulator run (Task 12) is what finally cross-checks it. It is scoped to opcodes
whose meaning is one line each.

### Review round — `8c23ffc`

18 of 20 lowering mutations were caught. The two that were not, plus three oracle
defects:

| Defect | Why it mattered |
|---|---|
| **Both accumulator zeroings deletable** | Every test ran *one* step on a freshly-zeroed `Machine`. A real decode loop reuses `pred`/`o` every token. Reviewer measured token-1 error at 1.9e-2 / 7.6e-2 |
| **`doc/plena_isa_spec.md` had `V_SUB_VV` backwards** | Spec said `rd = rs2 - rs1`; hardware does `rs1 - rs2`. Subtraction does not commute — **"fixing" the interpreter to match the spec would have flipped KDA's error sign** |
| **`gp0`/`f0` are not hardwired zero** | They are ordinary writable registers; `RegisterAllocator` reserving them is the entire guarantee. The oracle asserted hardware behaviour that does not exist |
| **No VLEN-alignment check in the oracle** | The emulator's vector SRAM asserts it; unaligned addresses panic there, passed silently here |
| **The oracle had no tests of its own** | Mutating `V_MUL_VF` to read its destination left every decode test green — the lowering only emits the in-place form |

Fixes: a two-token test that carries state on chip and rewrites only FPRAM between
tokens; the spec corrected with a note; the register model made faithful with the
invariant guarded where it lives (a test that the lowering writes neither `gp0`
nor `f0`); an alignment guard; and `test_isa_interpreter.py`, which exercises every
opcode in a form the lowering does not produce. All four oracle mutations now fail.

### Known restriction, carried forward

**`value_dim` must not exceed `mlen`.** The `tile_row_*` family defaults to
`tile_col_idx=0` while `vram_fill_zero` walks every column block and
`vram_matrix_add`'s wide path needs mlen-aligned row offsets, which cross-row
single-row copies do not have. Asserted, not implied.

**Kimi K3 is `value_dim=128` against a default `mlen=64`, so this must be lifted
before any whole-model run.** It is a change to the shared `vram_*` helpers, not
to KDA.

### Recurring lesson, third round running

Every review has found the same shape: **the assertion was right, the setup could
not exercise it.** Degenerate inputs (Task 1), opcode presence instead of operand
identity (Task 2), a single step instead of two (Task 3). The generalisation is
that a test must be run against a deliberately broken implementation before it is
trusted — which is now the working habit, not an afterthought.

---

## Interlude — unblocking the emulator found a bug in Task 2 ✅

Running `cargo test` for the first time (see the environment correction above)
immediately failed one test:

```
stage_marker_names_match_the_compiler_vocabulary
  StageKind has stages no compiler marker produces:
  ["kda_qkv_proj", "kda_conv1d", ... "kda_state_store"]
```

`e4110d47` added ten KDA variants to `StageKind` but not to the guard that reads
the vocabulary *back* out of the compiler's Python. The guard is bidirectional and
unions several compiler modules — MoE from `program_routed_moe.py`, Mamba from
`program_mamba_common.py` — because one `StageKind` is fed by all of them. KDA was
the third and was missing.

`cargo check` could not catch this: it is a test, not a compile error. Fixed in
`6271e982`; verified bidirectional by dropping `kda_readout` from the Python
vocabulary and confirming the Rust test names exactly that stage.

**Emulator suite: 106 passed, 0 failed.**

The lesson is narrower than it looks: I recorded an environment failure as a
project blocker without checking how the project builds. The `flake.nix` and
`docker/Dockerfile` both said so in the first ten lines.

---

## Interlude 2 — column-block folding ✅ `993d863` / `595e812`

Lifts the `value_dim <= mlen` restriction that Task 3 asserted against, **without
touching any shared helper**.

### The choice

Three options were on the table:

| | verdict |
|---|---|
| Add `tile_col_idx` to the six helpers that lack it | Rejected. Touches emitters Mamba and attention are using, for a bug **no current caller triggers** — `Mamba2Shape.validate` already constrains Mamba to `== mlen`. And its error text names this exact fix as *"deliberately not done yet"*, so it would reverse a decision the codebase made on purpose. |
| Loop `tile_col_idx` inside KDA only | Rejected. Measured: only `tile_row_mul_fp` accepts it. `tile_row_sum`, `tile_row_mul`, `tile_row_sub`, `vram_fill_zero`, `mamba_row_copy`, `mamba_row_add` do not — so the copy/multiply/add triple could not be told which block to work on. |
| **Fold the block into the row** | **Chosen.** `row = (head * blocks + block) * key_dim + key`. Every tile is exactly one block wide, no helper needs a column argument, and fixing `(head, block)` leaves keys at unit stride so each sweep is still one hardware loop. |

At `value_dim == mlen` the folded layout is byte-identical to the previous one —
verified by diffing the emitted assembly against `8c23ffc`: **776 lines each, every
instruction line identical**, differing only in stage-marker comments that gained
` block=0`.

### Why it is sound

Nothing in the KDA recurrence contracts over `value`. Decay, rank-1 update and
read-out are elementwise along it; prediction and read-out contract over `key` at a
fixed value lane. So block `c` of `pred`/`out` reads only block `c` of the state,
and blocks are independent. The reviewer confirmed this against the emitted code by
resolving every VRAM operand back to `(tile, row)` and grouping by stage marker:
**zero cross-block or cross-head accesses**, and the state rows touched are exactly
the full tile.

### Performance data

MLEN = 8, one decode token:

| heads | key_dim | blocks | static | dynamic | static / (head·block) |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1 | 505 | 540 | 505 |
| 1 | 8 | 2 | 1010 | 1080 | 505 |
| 1 | 8 | 4 | 2020 | 2160 | 505 |
| 2 | 4 | 1 | 554 | 584 | 277 |
| 2 | 4 | 2 | 1108 | 1168 | 277 |
| 3 | 5 | 3 | 3006 | 3186 | 334 |

**Exactly linear in blocks, zero per-block overhead.** Per-`(head, block)` cost
depends only on `key_dim`. Dynamic exceeds static by ~7%, from the `C_LOOP` bodies
in the decay and fill-zero sweeps.

### Review round — `595e812`

No correctness bug; 17 of 20 mutations killed. The three survivors shared one blind
spot: **nothing pinned row *ordering* or residency *sizing* at `blocks > 1`.**

| Survivor | Why it mattered |
|---|---|
| `kda_state_row` head/block swapped | Still a bijection, still unit-stride, so every layout test accepted it. But it *is* the HBM byte layout of the pinned state — a cross-component contract as soon as prefill or a checkpoint loader writes an initial state |
| `kda_pin_state_v0` dropping the blocks factor | Under-reserves, letting a later `input()`/`store()` land on the carried state. All four residency tests ran at `value_dim == mlen`, where old and new formulas coincide |
| `kda_head_base` ignoring its `block` argument | Same root cause |

Parametrising the residency tests over 1/2/3 blocks was **not sufficient on its
own**: the collision test compares against the size `pin` recorded, so an
under-reservation is self-consistent. What kills it is the round-trip test, which
computes the required size independently.

**Seeding/readback symmetry held up.** Five harness mutations; only a *globally
consistent* relabelling of the block axis survived, and that is an equivalent
convention, not a bug — the gather side is anchored to the reference in true
`[heads, key, value]` order.

### The trap this uncovered for Task 4

`kda_l2_normalize_v0`'s guard told callers to "block it with `kda_vector_row`
first". **That is actively wrong.** KDA normalises `q` and `k`, which are
*key*-width, and Kimi K3 is `key_dim = 128` against `mlen = 64` — but the L2 norm
**contracts over the key axis**, so the two halves of one `q` share one norm.
Following that hint normalises each half by its own partial norm: finite,
plausible, wrong.

Row folding solves `value_dim` precisely *because* nothing contracts over value. It
cannot solve `key_dim`.

And the obvious workaround does not work either: `tile_row_sum` emits
`S_ADD_FP f1, f0, f0` before each `V_RED_SUM`, so a second call **overwrites** the
FPRAM slot rather than adding to it — even though the `V_RED_SUM` instruction
itself accumulates into its FP register. Verified by reading the emitted assembly.

**Task 4 must therefore reduce each block into its own FPRAM slot and combine them
with scalar `S_ADD_FP` before the rsqrt** — `blocks - 1` extra scalar adds per
vector.

---

## Task 4a — The causal conv step ✅ `de3eaae` / `08a7a2c`

**Result: 3 failed (baseline), 194 passed.** `test_kda_conv.py` 9,
`test_isa_interpreter.py` 19.

KDA runs three short depthwise convolutions — q, k, v — before the recurrence.
Validated numerically against `_causal_conv_step` across four geometries including
multi-block, plus direct assertions that the history *shifts* rather than being
overwritten and that perturbing one channel block moves only that block.

### Reuse, and where it stopped

`mamba_conv1d_v0` is the same kernel and makes the same argument about `V_MUL_VF`
being wrong for a per-lane tap weight. Two differences made it its own emitter:
Mamba convolves a sequence held in `x` (so the shift is a free row offset), while
KDA decode has a **carried** state (a real row copy every step); and KDA's channel
counts are ~12,288, so the block index has to reach the inner loop and
`mamba_conv1d_v0` has nowhere to put it.

Adding a base-row argument there would have avoided the duplication — **rejected
for the same reason as the `tile_row_*` question**: that emitter is on Mamba's
decode and prefill paths.

### The oracle refused, as designed

`V_MUL_VV`, `V_EXP_V`, `V_RECI_V`, `V_ADD_VF` were outside its modelled subset, so
it raised instead of guessing. Extended deliberately with those plus `V_SUB_VF`,
**semantics taken from the emulator, not the spec** — `V_EXP_V` clamps to
`[-88, 88]` *before* exp, so an unclamped model could not return `inf` and have a
comparison pass for the wrong reason.

### Review round — `08a7a2c`

| Defect | Why it mattered |
|---|---|
| **`out` zero-fill unanchored** | Same class as `8c23ffc`. The harness built a fresh zero-initialised `Machine` and never wrote `out`, so "the program clears it" and "it happened to start at zero" were indistinguishable. `out` is persistent and reused by q/k/v |
| **SiLU billed to a Mamba stage** | `mamba_silu_v0` emitted `@stage=mamba_gated_norm` *inside* the `kda_conv1d` region — and markers are sticky, so everything after it billed there too. ~8% of conv instructions, ×3 convs ×61 layers |
| **Stage guard scanned one file** | `test_kda_stage_contract.py` hardcoded `program_kda_common.py`, so the grep guards passed vacuously on the new module |
| **`V_RECI_V` diverged on zero** | Oracle raised where `tensor.reciprocal()` returns `inf` |
| **Alias guard omitted `weight`/`bias`** | `mamba_row_copy` is zero-then-add; it would wipe them mid-loop |
| **`kda_conv_state_row` formula unanchored** | Emitter and both harness sides funnel through it, so any bijection cancels |

`mamba_silu_v0` now takes an optional `marker` overriding the emitted string
verbatim — additive, default-preserving, so Mamba is untouched.

### Performance

mlen=8, `apply_silu=True`: **`static = 39 + 104·blocks`** at kernel 4, and `+22`
per unit of kernel per block. Exactly linear; SiLU amortises to `~31 + 8·blocks`.

**The conv body is fully unrolled.** Every history shift and tap multiply is a
separate row op with `num_rows=1`, which `_row_progression` cannot fold — only the
SiLU tail becomes a hardware loop. At Kimi K3 scale (12,288 channels ÷ 64 = 192
blocks, kernel 4) that is **~20,000 static instructions per conv, ×3 per layer**.

The layout was chosen so the history shift is a run of adjacent row copies, but
nothing folds them yet — **the adjacency buys nothing today.** And `V_FMA_VF` will
not help: the tap multiply is by a per-channel *vector* (`V_MUL_VV`), not a
broadcast scalar. Folding the conv would need either a block-copy-based shift or a
`V_FMA_VV`. Recorded for the Phase 1 measurement rather than acted on.

---

## Task 4b(i) — Cross-block L2 normalisation ✅ `f57a45f` / `9671ed1`

**Result: 3 failed (baseline), 211 passed.**

The trap Task 4a's review identified, now solved. Everywhere else in KDA a wide
vector splits across rows because nothing contracts over `value`. **`q` and `k`
are key-width and the norm contracts over `key`**, so the halves of one `q` share
one norm — the column-block folding does not apply.

The obvious workaround does not work either: `tile_row_sum` emits
`S_ADD_FP f1, f0, f0` before each `V_RED_SUM`, so a second call into the same slot
overwrites it. So each block reduces into its own slot and the slots fold with
scalar `S_ADD_FP`. Partial sums are laid out **block-major**, `part[c*vectors + v]`,
so the fold is `blocks-1` contiguous FPRAM adds of length `vectors` rather than
`vectors` separate reductions.

### The `_fpram_row_map` trap, second occurrence

With >1 row it walks `base_offset + i`; with exactly one it **ignores
`base_offset`** and uses `single_offset`. At `vectors == 1` every block's sum
landed in slot 0 and only the last survived — invisible at `vectors > 1`, which is
why the parametrisation runs 1, 2, 3 and 4 vectors. Both call sites now go through
`_kda_row_sum` / `_kda_scale_rows`, which own the branch. `_kda_scale_rows` moved
into `program_kda_common` so the common module no longer depends on a method
defined in a specialisation.

### Review round — `9671ed1`

**The lowering was clean — all 13 mutations caught.** The failures were all in the
**oracle**, and they were exactly the semantics this task had gone to the emulator
to establish:

| Survivor | Why the lowering cannot reach it |
|---|---|
| `V_RED_SUM` overwrite vs accumulate | Every emitted `V_RED_SUM` is preceded by `S_ADD_FP f1, f0, f0`, so f1 always enters at zero |
| `f0` write guard (×3) | `RegisterAllocator` hands out f1..f7, so f0 is never a destination |
| `S_SUB_FP` direction | Emitted **zero** times by any KDA path |
| `S_ST_FP` immediate | Non-zero only under `ATEN_OPS_UNROLL=1` |
| `S_RECI_FP(0)`, `S_SQRT_FP(neg)` | Inputs are always positive here |
| `V_RED_SUM` alignment check | Addresses are always aligned |

All eight now pinned by direct unit tests. Two genuine oracle corrections fell out:
`S_RECI_FP(-0.0)` returned `+inf` where bf16 division gives `-inf` (Python's
`a == 0.0` is true for `-0.0`), and `V_RED_SUM` accepted a mask it does not model.

### Performance

One-line fix worth recording: `mamba_block_copy` was copying the **padded** tile
height, not the live rows. VRAM row counts must be a multiple of `mlen`, so at
`mlen=64` with two live rows it copied 64.

| Kimi shape (mlen=64, 1 vector, 2 blocks) | dynamic |
|---|---:|
| before | 598 |
| after | **94** |

**6.4×**, and the padding never contributed to the result. `kda_l2_normalize_v0`
has the same pattern — an existing convention worth revisiting.

Scaling: static `+29` per block, `+11–13` per vector (`mamba_rsqrt_fpram` is
Python-unrolled at 9 instructions per vector).

### Flagged, not fixed — shared code

`tile_row_max_asm` calls `_emit_tile_row_reduce` **without**
`clear_accumulator=True`, while `tile_row_sum_asm` passes it. `V_RED_MAX`
accumulates the same way `V_RED_SUM` does, so a multi-row `tile_row_max` returns a
**running maximum** seeded with whatever the previous user left in `f1`. That is
Mamba and attention code — spawned as its own task rather than fixed inline.

---

## Task 4b(ii) — Decay and beta scalars ✅ `6329a62` / `ff7fa6b`

**Result: 3 failed (baseline), 225 passed.**

`decay[h,k] = exp(lower_bound · sigmoid(rate[h] · (gate + dt_bias)))` and
`beta[h] = sigmoid(beta_logit)`, both landing in FPRAM via `S_MAP_FP_V`.
`rate = exp(a_log)` depends only on a weight, so the host precomputes it rather
than paying a vector `exp` per token.

Sigmoid is four vector ops in place with no scratch. `mamba_silu_v0` runs the same
sequence but needs scratch because it keeps `x` alive for a final multiply.

**The FPRAM addresses fall out of the row order rather than being computed.**
`tile_row_to_fpram` puts row `i` at `base + i·mlen`, so iterating `(head, key
block)` lands head `h`'s block `kb` at `h·key_dim + kb·mlen` — exactly what the
recurrence indexes. A test asserts that slot by slot rather than trusting it.

### Review round — `ff7fa6b`

**No surviving lowering mutation** (17 of 18 red; the one green was a verified
semantic no-op). But one real out-of-bounds write:

`kda_beta_scalars_v0` checked `beta_fp.size >= num_heads`. **`S_MAP_FP_V` moves a
whole `mlen` row**, so the write is `rows · mlen`. At Kimi's 96 heads against
`mlen` 64 that is **32 slots written outside the allocation**. Nothing downstream
catches it: `_resolve_fpram_addr` bounds-checks only the base offset, and the
emulator's scalar SRAM knows the file but not allocation boundaries.

My own commit message contained the tell: *"harmless because they are never
indexed"* — that argument is about **reads**. The lanes are never read. They are
written. And the test masked it by sizing `beta_fp` with `_rows_up(num_heads)` —
the padded size the emitter should have required but didn't.

The oracle would not have caught it either: **Python slice assignment past the end
extends the list** rather than failing, so an OOB `S_MAP_FP_V` silently landed at
the wrong address and stayed green. Now refuses, matching the emulator's assert.

---

## ⚠ Design constraint found here — this decides the mixer's shape

**FPRAM is 512 slots in RTL** (`FP_SRAM_DEPTH`, `configuration.svh:30`); 1024 in
the compiler's allocator.

The current scalar layout wants `decay_fp`, `q_hat` and `k_hat` at
`num_heads · key_dim` each:

| | Kimi K3 |
|---|---:|
| `decay_fp` | 12,288 |
| `q_hat` | 12,288 |
| `k_hat` | 12,288 |
| `beta_fp` | 96 |
| **total** | **36,960** |
| **available** | **512** |

**72× over.** The reviewer reproduced the overflow at (96 heads, 2 key blocks)
even at `mlen=8`.

So the mixer **cannot** compute every head's scalars and then run the recurrence.
It has to **stream per head**: produce one head's decay / q̂ / k̂, run that head's
recurrence, reuse the window. That needs `3·key_dim + 1 = 385` slots, which fits.

`kda_decode_step_v0` already takes `head_rows` for exactly this — the per-head
path was built in Task 3 and is about to earn its keep.

### Performance

Static counts scale **linearly in `heads × key_blocks`, ~12 per row** — not
constant. `_arith_progression` returns `None` for a constant sequence (step 0 with
count > 1 would spin a hardware loop forever), and the three broadcast-constant
steps in the sigmoid have identical FPRAM addresses per row, so they unroll.
Kimi's gate path is ~2.4k static instructions per token, mostly that unrolling.
**Preloading the constants into registers once is a clear optimisation**, recorded
for the Phase 1 measurement.

---

## Task 4c — The assembled per-head mixer step — next

Committed as `85be73f`. 246 → passed, 3 pre-existing failures. Reviewed below.

### Review — four surviving mutations, and the FPRAM budget was wrong

The review found the number this whole design rests on was miscounted.

**1. The mixer did not fit FP_SRAM_DEPTH.** `kda_mixer_fpram_slots` returned
`3·key_dim + 1` and the test asserted `385 ≤ 512`. Two omissions:

* `beta_fp` is not one slot. `S_MAP_FP_V` moves a whole `mlen` row, so it costs
  `ceil(num_heads/mlen)·mlen` — **128** at Kimi K3, not 1. (Task 4b(ii) fixed the
  out-of-bounds *write* this causes but did not carry the size into the budget.)
* `rate_fp` is indexed by head number, so it costs `num_heads` — **96**.

Real high-water was **620 against 512**. And `3·key_dim + beta = 512` on its own,
so no rearrangement of the rest could have saved it.

The fix is a liveness observation: **`decay` and `q_hat` are never live at the
same time.** Decay is dead once the prediction has run; `q_hat` is not read until
the read-out. So `kda_decode_step_v0` split into `kda_decode_predict_v0` and
`kda_decode_update_v0`, the mixer calls the two halves, and normalises `q`
*between* them — into the window decay just vacated. The wrapper still exists and
`test_the_two_halves_emit_the_same_program_as_the_whole_step` pins it.

| | slots |
|---|---|
| `k_hat` | 128 |
| `decay_or_q_hat` (shared) | 128 |
| `beta` | 128 |
| `rate` | 96 |
| `part`/`acc`/`scale`/`lower` | 5 |
| `consts` | 7 |
| **total** | **492** |
| **FP_SRAM_DEPTH** | **512** |

20 slots of headroom. `kda_mixer_fpram_slots` now returns the itemised dict, and
`test_fpram_accounting_matches_what_is_actually_allocated` checks it against the
allocator's real high-water mark — because `FPRAMAllocator` defaults to **1024**
and will not catch an overflow itself. That 1024-vs-512 disagreement between
`memory.py` and `configuration.svh` is still open; the SystemVerilog is
authoritative and both copies of it (compiler `doc/`, `PLENA_RTL/src/`) say 512.

**2. `head_rows` gave wrong answers, and nothing caught it.** Mutating the mixer
to ignore `head_rows` entirely left the suite green: the test named
`test_heads_can_be_lowered_individually` never passed the argument. Actually using
it was worse than untested — lowering three heads in three calls gave head 0 the
right answer and heads 1–2 an error of 1.2e-2, because `kda_beta_scalars_v0` ran
at the top of *every* call and consumes `beta_logit` in place. Sigmoid is not
idempotent.

`kda_beta_scalars_v0` moved out of the mixer; the caller runs it once. The test
now lowers per head and asserts bit-identical output against one call for all
heads, and a second test checks that lowering head 0 leaves the other heads'
state untouched.

**3. The idempotency docstring was backwards.** It warned about the L2
normalisation, which is the one in-place step that *is* idempotent (a unit vector
normalises to itself, 2.4e-7), and omitted the two that are not: `beta_logit`
(0.43) and `gate` (0.90). Rewritten with the measured numbers.

**4. A head subset bounded the tile by row *count*, not row *number*.** With
`heads=[12]` the row list has `key_blocks` entries but indexes row 12, so a
12-row check passed a tile of 8 rows and the emitter then read and wrote four
rows past the allocation — silently, into the next VRAM object. Bound is now
`max(rows) + 1`.

**5. The squares copy was quadratic in head count.** `kda_l2_normalize_blocked_v0`
copied rows `0 .. first_row + n` instead of the `n` live ones, and the mixer walks
`first_row` across every head. At Kimi K3 that is **18,624 rows moved for 384 rows
of live data, 48×**, at ~80 instructions per row. `mamba_block_copy` already took
`src_row_offset`/`dst_row_offset`; using them makes the cost flat in `first_row`
(2,374 → 2,401 instructions across a 256-row offset — the residue is the wider
address immediate). Pinned by `test_the_squares_copy_is_only_as_long_as_the_slice`.

Also fixed: both `fp_head_stride` guards were untested (no test passed the
parameter at all), and the `sq_scratch` `first_row` bound was untested because
`vec`'s bound always fired first.

### Mutation results after the fixes

All ten go red, including the four that previously survived:

| | mutation | |
|---|---|---|
| M1 | mixer ignores `head_rows` | 2 failed |
| M2 | drop the `stride==0` multi-head guard | 1 failed |
| M3 | drop the negative-stride guard | 1 failed |
| M4 | `sq_scratch` bound drops `first_row` | 1 failed |
| M5 | gates bound back to `len(rows)` | 1 failed |
| M6 | block copy back to row 0 | 1 failed |
| M7 | beta counted as 1 slot | 2 failed |
| M8 | rate left out of the budget | 2 failed |
| M9 | `q` normalised before the predict half | 9 failed |
| M10 | update half reads decay as `q` | 8 failed |

### Performance

The 723 recorded above for Task 4c is right. I briefly "corrected" it to 27,243
and that correction was wrong: `get_code()` returns assembly **text**, so
`len(get_code())` counts characters. The codebase counts instructions as
non-blank non-comment lines -- see `test_records_the_phase0_instruction_count`.

One head, `key = value = mlen = 8`:

| stage | static |
|---|---|
| beta (once, all heads) | 33 |
| normalize x2 | 110 |
| `tile_row_to_fpram` x2 | 14 |
| decay | 61 |
| **predict half** | **188** |
| **update half** | **317** |
| **mixer total** | **723** |

| shape | static | per head |
|---|---|---|
| 1 head, key=value=8 | 723 | 723 |
| 3 heads, key=value=8 | 2,103 | 701 |
| 2 heads, key=16 value=8 | 2,469 | 1,234 |
| 3 heads, key=value=16 | 6,570 | 2,190 |
| 6 heads, key=value=16 | 13,107 | 2,184 |

**The recurrence is 70% of the mixer** (505 of 723), and it is all the
`copy + multiply + add` triple per key row -- exactly what `V_FMA_VF` collapses.
That is the number Task 8 is measured against.

Per-head cost is flat in head count (701 at 3 heads against 723 at 1), which is
what the copy fix bought; before it, it grew.

---

## Phase 1, Task 5 — `V_FMA_VF` in the emulator — next

## Task 5 — `V_FMA_VF` in the emulator — done (`df2ae9fa`)

`Vector[rd] += Vector[rs1] * fp_reg<rs2>` at 0x3B. Two places the plan
under-specified, both found by writing the test first:

* **The masked path starts from the destination, not the source.** `mul_scalar`
  starts from the source because there the source *is* what the result is built
  from. For an accumulate, a masked-off head must keep what the destination had.
  Pinned by `fma_scalar_leaves_masked_off_heads_holding_the_destination`.
* **Timing gets its own arm.** FMA touches two vector rows, like the VV ops, not
  the one row the rest of the VF family does. The plan put it in the single-read
  group, which would have under-reported exactly the traffic Task 8 is judged on.

`test_mamba_opcodes_do_not_collide_with_existing_encodings` failed, correctly --
it pinned 0x3B as free. Boundary moved to 0x3C. 109 passed.

`cargo clippy` and `cargo fmt` are **not installed** in the Nix dev shell, so the
plan's lint steps could not run.

## Task 6 — the ISA definition, assembler and spec — done (`93bc504`, `84ef4848`)

Mnemonic `V_FMA_VF rd, rs1, fp2, rmask`, encoding identical to `V_MUL_VF` except
the opcode field.

**PLENA_RTL's `operation.svh` stops at 0x34.** 0x35 through 0x3A were already
ahead of the RTL before this work, so `V_FMA_VF` joins six existing opcodes the
compiler and emulator implement and the hardware does not. Not a regression, but
it is the honest status: this instruction needs RTL work that does not exist.

### A guard, and two pre-existing gaps it found

`decode` carried three comments saying encodings "must stay in sync with
PLENA_Compiler's `doc/operation.svh`" -- enforced by author diligence alone,
which is the arrangement that let FPRAM 1024-vs-512 drift unnoticed. Now a test
parses the header (the submodule is checked out recursively in every CI job) and
asserts nothing in it decodes to `Invalid`.

It found `V_PS_V` (0x31) and `C_HADAMARD_TRANSFORM` (0x33): declared by PLENA,
implemented nowhere. Both known -- `program_ssd.py` already documents that
`V_PS_V` "assembles silently and then decodes to Invalid, so emitting it is worse
than not emitting it" -- but that lived in one docstring. They are now an
explicit exemption list carrying the reason. Verified by deleting the 0x3B arm:
the guard names the opcode and fails.

## Task 7 — the FMA row-sweep emitter — done

`_emit_tile_row_fma` with four GP pointers, plus `tile_row_fma_fp_asm`,
`tile_row_fma_fp_sweep_asm` and `tile_row_fma_fp_broadcast_asm`.

**Named `_sweep`, not `_broadcast` as the plan said.** In `isa_tile_rows.py`
`_broadcast` means one FPRAM slot applied to *every* row; the plan's function
walks `fpram_base + i`, which is the opposite. Using the plan's name would have
made every call site read backwards. Both now exist, with the file's meanings.

### `_arith_progression` refused a step of 0, and its stated reason was wrong

The comment read "Constant sequence (step=0, count>1) would cause infinite HW
loop". That is not what happens: `C_LOOP_START` takes its trip count as an
immediate and `C_LOOP_END` decrements a dedicated loop register
(`loop_state.rs`), neither of which the address step touches. The single-element
case has always returned step 0 and looped fine.

It matters because **a pinned side is exactly a step of 0**, and every recurrent
contraction has one -- prediction and read-out walk the state rows against one
accumulator row. Removing the restriction breaks nothing (265 passed, same 3
pre-existing failures) and changes no existing output.

**It also buys nothing yet, and it is worth being clear about that.** The mixer
is 723 instructions before and after. The current sweeps loop in *Python* --
`mamba_row_copy` is called once per row with a single row argument, so there is
no row list for `_row_progression` to see. Step-0 support is a **precondition**
for Task 8, which passes whole row lists to the FMA emitter; it is not a win on
its own.

### The oracle had to learn the opcode

`isa_interpreter.py` correctly refused `V_FMA_VF`, so it is now modelled -- as
one expression, `dst + src * f`, matching `fma_scalar`'s single quantisation
rather than a multiply followed by an add. `test_unmodelled_opcode_raises` used
`V_FMA_VF` as its example of something unmodelled and now uses `V_MAX_VF`. Two
new oracle tests pin the accumulate and the `dst == src` case.

265 passed, 3 pre-existing failures.

---

### Review of Task 5 — no bug in the instruction, three gaps around it

The arithmetic, the masked-path base, the timing classification and the
cross-repo encoding all checked out. What did not:

* **The profiler charged `V_FMA_VF` one cycle instead of five.**
  `isa_analysis.py`'s `instruction_cycles` is a prefix cascade ending in a
  catch-all `return 1`, and `V_FMA_VF` does not start with `"V_MUL"`. The
  emulator charges `VECTOR_MUL_CYCLES`. A **5x under-count per FMA, in the one
  tool Phase 4 measures with**, on the one opcode being measured. It was also
  missing from `selected_opcodes`, so the recurrence would not have appeared in
  the per-opcode breakdown at all. Both fixed.
* **`analytic_models/performance/customISA_lib.json` had no entry**, and
  `perf_model.py:147` raises `KeyError` on a miss rather than defaulting. Added,
  matching `V_MUL_VF`'s cycle expressions.
* **The dispatch arm and the timing arm were pinned by nothing.** Swapping
  `rd`/`rs1` in dispatch -- which makes the instruction compute
  `V[rs1] += V[rd] * f` -- left 110/110 green, because *nothing in the emulator
  executes a decoded instruction*; every test calls the machine methods
  directly. Same for the timing arm, the commit's own headline justification.

The timing arm now has a real test. The dispatch arm has a source-parsing guard,
the technique `stage_profile.rs` already uses, because closing it properly needs
an `Accelerator` built in a test with Ramulator and a `MatrixMachine`. **Every
opcode has this gap**, not just the new one -- spawned as separate work.

Two numbers worth keeping from the review:

* **FMA is 1.20x more accurate than the triple in RMS** (bf16, VLEN 64, 64
  trials): max_abs 3.89e-3 against 5.81e-3. Direct difference 7.81e-3, about 2
  bf16 ulp, growing to 6.25e-2 over a 64-step sweep.
* **No test observes that difference.** `isa_interpreter.py` is float64, where
  the FMA and the triple are algebraically identical, and nothing runs KDA
  numerics through the transactional emulator. The difference is real, it
  favours the FMA, and it is currently unmeasured.

Also noted: `QuantTensor::quantize` is a **no-op** (`tensor.rs:128`, "TODO: add
actual quantization") -- it only attaches a type tag, and all rounding happens at
`VectorSram::write`. So "one rounding versus two" is true, but it is the SRAM
store that rounds.

---

## Task 8 — both recurrences on `V_FMA_VF` — done

### `ssm_decode_step_v0` had no test at all

The conversion landed and the whole suite stayed green. `grep -rl
ssm_decode_step_v0` returns the emitter, its caller in `aten/ops/plena/mamba_ops.py`,
and a docstring mention. **No test.** A rewrite of Mamba's state update and
output contraction was therefore unverified.

`aten/tests/test_ssm_decode_step.py` now runs it against
`mamba2_recurrent_reference` through the ISA interpreter -- five geometries
including multi-group, plus a seeded-garbage check that the output accumulator is
actually cleared. Eight tests.

### The measurement

Static instructions, one head:

| | before | after |
|---|---|---|
| KDA decode, key_dim 8 | 505 | **76** |
| KDA decode, key_dim 16 | 961 | **76** |
| KDA decode, key_dim 64 | 3,697 | **76** |
| KDA decode, key_dim 128 | 7,345 | **76** |
| Mamba decode, state 4 | 208 | **54** |
| Mamba decode, state 8 | 376 | **54** |
| Mamba decode, state 16 | 712 | **54** |
| Mamba decode, state 32 | 1,384 | **54** |

**The Mamba "before" column was wrong when first written** (200 / 352 / 656). I
measured it by reverting only the conversion, at a HEAD that already carried the
step-0 relaxation -- which had itself shrunk `ssm_decode_step_v0` from 712 to
656. The true pre-Phase-1 numbers, at `ede7454`, are above. The KDA figures were
measured the same way and are unaffected, because the KDA recurrence had no
step-0 progression to gain from.

**The cost went from linear in the contracted dimension to constant.** At Kimi
K3's key_dim of 128 that is 7,345 -> 76, a factor of **97**. That, and not the
saved multiply, is the case for spending opcode 0x3B: the triple needed a scratch
row per step, and because it was the *same* row every iteration the destination
never formed a progression, so the sweep could not become a hardware loop.

Whole mixer, per token:

`kda_mixer_step_v0` alone, **not** counting the one-off `kda_beta_scalars_v0`
(33 instructions) that the caller now runs separately:

| shape | after |
|---|---|
| 1 head, key=8 value=8 | 241 |
| 1 head, key=16 value=8 | 306 |
| 1 head, key=16 value=16 | 378 |
| 2 heads, key=8 value=8 | 482 |
| 3 heads, key=16 value=16 | 1,134 |
| 6 heads, key=16 value=16 | 2,268 |

Exactly linear in head count at a fixed shape, which is what the per-head
streaming is for. The earlier version of this table mixed `kda_beta_scalars_v0`
into some rows and compared across differing `value_dim`, so its "before" column
was not comparable row to row; the decode-step figures above are the ones to
quote.

`test_instruction_budget.py` gates both kernels, and
`test_the_cost_is_flat_in_the_contracted_dimension` asserts the flatness
directly. A budget alone would still pass if the cost grew and someone raised
the number.

### The scratch tile is gone

`kda_decode_step_v0` no longer takes one: the FMA accumulates in place, which is
most of what it bought. `KdaMixerBuffers` loses the field too. The aliasing check
that still matters is `pred` and `err` against `state`.

Seven mutations of the conversion, all red: dst/src swapped in both kernels'
rank-1 updates, the contraction reading the wrong FPRAM array, the D skip
losing its accumulate, the read-out losing `fp_base`, and the emitter emitting
`V_MUL_VF` instead (32 failures).

284 passed, 3 pre-existing failures. Emulator 112 passed.

---

## Task 9 — KDA chunk primitives — done

### The plan's formula was wrong, and the reference has no chunked path to check against

`kda_state_engine_prefill` is a loop of `kda_state_engine_step`. So the only
oracle for a chunked form is "it must equal the sequential one", and the algebra
had to be derived and checked before any of it was worth lowering. That was done
in plain torch first.

**KDA's decay is channel-wise on the key axis** -- `reference.py` applies
`exp(log_decay)[:, :, None, :]`, one factor per key channel, not one scalar per
timestep. The textbook chunking assumes a scalar decay, and the plan's
`_dense_ut` follows it: `tril(diag(beta) @ (K @ K.T), -1)`, with no decay
weighting. Against the sequential reference that form is **wrong by 1.8e-1**.
The correct gram is `M[t,s] = sum_key k_t k_s A_t/A_s`, which reproduces the
sequential output to **4.5e-8** and the state to 1.6e-7.

### Three numerical findings, all measured

**1. `chunk * |gate_lower_bound|` must stay under 88.7.** `M` and `N` are formed
as a matmul of `k*A` against `k/A`, and `1/A` reaches `exp(chunk*|lower_bound|)`.
bf16 tops out at 3.39e38, `ln` 88.72. At Kimi K3's -5 the **last chunk that works
is 17**; chunk 18 is `inf`, and then `nan` through the whole solve. The plan said
"do not raise the chunk size without redoing this analysis" -- this is the
number, and it is now an assertion rather than a comment.

**2. The cumulative decay must be a running product, not `exp` of a running
sum.** Following `program_ssd.py`'s cumsum was the obvious move, and it is wrong
here: `c` reaches -80, where bf16's ulp is 0.31, and `exp(0.31)` is a 36%
relative error on `A`. Multiplying per-step decays keeps every intermediate in
`[e^-5, 1]`, where bf16's error is relative rather than magnified.

| route | output error | state error |
|---|---|---|
| cumsum in f32 (the algebra's own floor) | 8.1e-05 | 9.5e-04 |
| cumsum in bf16 | 3.7e-04 | **1.1e-02** |
| **cumprod in bf16** | **5.9e-05** | **2.4e-03** |

**3. The matmul form's cancellation does not reach the output.** Forming `M` from
bf16 `k*A` and `k/A` gives up to **315% relative error on M's smallest entries**
-- which looked disqualifying. It is not: those entries contribute nothing, and
end-to-end the matmul form and an exactly-computed one agree to 6e-5. So the
matrix engine can do this contraction after all, and the vector-unit fallback
(C^2 contractions, no cheaper than just running decode 16 times) is not needed.

### What was built

`aten/plena/program_kda_chunk.py`:

* `kda_chunk_decay_cumprod_v0` -- running product per key channel, in place.
  `mamba_block_copy`'s independent row offsets stage one whole timestep at a
  time, so the serial scan needs no new shared emitter.
* `kda_ut_transform_v0` -- `T = (I + tril(diag(beta) M, -1))^-1 diag(beta)` by
  forward substitution. `T[i] = beta_i (e_i - sum_{j<i} M[i,j] T[j])` is **one
  FMA sweep per row**: destination pinned to row `i`, source walking `0..i-1`,
  FPRAM walking `M[i, 0..i-1]`. Both progressions, so one hardware loop per row
  -- 15 of them at chunk 16, whatever `key_dim` is. `M` is negated once up front
  because the sweep accumulates and the substitution subtracts.
* `kda_chunk_check_range` / `kda_max_chunk_for` -- the bf16 bound.

The identity is host-staged like `ssd_lower_triangular_ones`, for the reason that
helper gives: materialising a constant tile on chip costs one `S_ST_FP` per
element.

### Measured

| | max abs error | static | dynamic |
|---|---|---|---|
| cumulative decay, chunk 16, key 8 | **0** (exact) | 225 | |
| cumulative decay, chunk 16, key 16 | **0** | 270 | |
| UT transform, chunk 8 | 1.5e-08 | 246 | |
| UT transform, chunk 16 | 3.0e-08 | 502 | 1,102 |

The chunk-16 UT transform is 502 static instructions with 63 hardware loops, per
head. Sixteen tokens of decode is 1,216 at one key block -- so the transform
alone already costs less than the sequence it replaces, before the four matmuls
carry the rest.

Ten mutations, all red after one fix. **`C6` survived at first**: extending the
substitution sweep from `range(i)` to `range(i+1)` left everything green,
because every test fed a strictly lower-triangular `M` and `M[i,i]` was
therefore zero. In the real pipeline `M` is a gram matrix whose diagonal is
`|k|^2`, near 1, so an off-by-one mask would be a real wrong answer.
`test_ut_transform_ignores_m_on_and_above_the_diagonal` now feeds a polluted
diagonal and demands a bit-identical result.

**Deferred:** these emitters bill to the existing `kda_decay` and
`kda_state_update` stages. Prefill deserves its own stage names so Phase 4 can
separate its cost, but `KDA_STAGES` is guarded bidirectionally against the
emulator's `stage_profile.rs`, so adding one means editing both repos together.
Done with the prefill layer.

306 passed, 3 pre-existing failures.

---

## Task 10 — the KDA prefill layer — next

### KDA on the real emulator — the gap the Phase 1 review named

Every KDA test so far ran on `aten/tests/isa_interpreter.py`, which is float64
and says so. The Phase 1 review's most useful non-defect observation was that
**nothing runs KDA numerics through the transactional emulator**, so the bf16
behaviour the FMA was chosen partly for was unmeasured, and the dispatch arm that
translates a decoded `V_FMA_VF` into a `fma_scalar` call was executed by nothing
at all.

`transactional_emulator/testbench/kda/kda_stage_test.py` closes it, on the
pattern `mamba2/mamba2_stage_test.py` already established: compile one stage,
assemble it, run the Rust emulator, compare against a float32 torch golden.

| case | result | max error |
|---|---|---|
| `cumprod` — the cumulative decay | **PASS** | 1.95e-3 (one bf16 ulp) |
| `ut` — the UT transform | **PASS** | 1.95e-3 (one bf16 ulp) |

Both errors are exactly one bf16 ulp, which is the floor rather than a defect.
The Mamba cases can assert bit-exactness by drawing inputs that e4m3 represents
exactly; the cumulative decay cannot, because a product of sixteen
3-mantissa-bit values needs more than bf16's 8 — the inputs are exact and the
fifteen multiplies are not.

**This closes M12 for real.** The Phase 1 review found that swapping `rd`/`rs1`
in the dispatch arm — which makes the instruction compute `V[rs1] += V[rd]*f` —
left all 110 emulator tests green, because nothing executed a decoded
instruction. The `ut` case emits `V_FMA_VF`, and re-running it against that
mutation gives max error **0.5 against 0.002**. The source-parsing guard added
earlier stays as a fast local check, but this is the real one.

Run: `nix develop . --command bash -c 'cd transactional_emulator/testbench/kda &&
uv run python kda_stage_test.py --case ut'` (needs the Nix shell for Ramulator
and libtorch, and `uv` for torch).

### Review of Phase 1 — one surviving mutation, one false claim of mine

The conversion itself is **bit-identical** to what it replaced. The reviewer
built `ede7454` and `c833465` as separate worktrees and ran both recurrences on
byte-identical seeded VRAM and FPRAM, comparing the full `state`, `o`/`y` and
`err` tiles across twelve geometries — multi-block, non-power-of-two `key_dim`,
multi-group Mamba. All identical.

**1. A branch no test reached.** `_emit_tile_row_fma`'s unrolled fallback: both
swapping its two pointers and replacing the whole branch with `raise` left the
suite green. It is reachable in production two ways — a scattered `row_map`
through `tile_row_fma_fp_asm`, and `ATEN_OPS_UNROLL=1`, which routes *every*
emitter through it. Two tests now cover it, including the repeated-destination
case, which is the read-modify-write ordering the FMA makes possible: two
entries on the same row must compose, not the later win.

**2. "Removing it changes no existing output" was false.** The step-0 relaxation
changed 18 of the 24 `tile_row_*` shapes, at live call sites in
`program_ssm_recurrent`, `program_ssd`, `program_kda_gates` and
`program_mamba_common`. All numerically equivalent — checked word for word on
identical inputs — but the claim as written was wrong, and it is what made the
Mamba baseline above wrong too.

**3. It also cost dynamic instructions, and that is now fixed.** The looped form
dispatches `3 + 5N` where the unrolled dispatched `3N`, so the relaxation traded
static for dynamic: `_kda_sigmoid_inplace` over 32 rows went 266 → 26 static but
452 → 522 dynamic, **+15%**. The cause was that `_emit_tile_row_reduce` and
`_emit_tile_row_fp_scalar` still emitted `S_ADDI_INT gp, gp, 0` unconditionally
— the `if step:` guard existed only in the new FMA emitter. Adding it to both:

| | static | dynamic |
|---|---|---|
| sigmoid, 32 rows — unrolled (pre-step-0) | 266 | 452 |
| sigmoid, 32 rows — looped, no guard | 26 | 522 (+15%) |
| **sigmoid, 32 rows — looped, guarded** | **24** | **458 (+1.3%)** |
| `ssm_decode_step_v0` state 16 — no guard | 54 | 279 |
| **`ssm_decode_step_v0` state 16 — guarded** | **51** | **261** |

So the 10x static win keeps essentially all of it and the dynamic cost is gone.

**4. A range check the conversion dropped.** `_kda_scale_rows(..., fp_base + i)`
resolved and bounds-checked one FPRAM slot per iteration;
`tile_row_fma_fp_sweep` resolves the base once and lets the hardware walk, so an
over-long sweep read into whatever `FPVar` came next with nothing to say so.
Restored at the var layer, where the `FPVar`'s size is known.

**5. The accumulator clear was a single-point defence.** Deleting either
`vram_fill_zero` in the KDA recurrence killed exactly *one* test, because
`_Harness.run` never seeded `pred`/`o` and a fresh `Machine` is all zeros — the
same trap that was fixed in the new Mamba harness but not retrofitted. Seeded
with 7.5; the same deletion now fails 11.

Also: `program_kda_recurrent`'s docstring still opened with a "Phase 0 form …
**no new opcode**" section the conversion had made false, and `test_kda_mixer`
still allocated a `scratch` tile nothing consumed. Both gone.

Dynamic instruction counts for the converted recurrences fell 50–65%, so the
hardware loop is not trading dynamic for static there — it wins both:

| | static before → after | dynamic before → after |
|---|---|---|
| KDA decode, key_dim 128 | 7,345 → 76 | 7,980 → 2,616 |
| Mamba decode, state 32 | 1,384 → 54 | 1,384 → 519 |

## Task 10 — the KDA prefill layer — working end to end

`aten/plena/program_kda_prefill.py` composes the seven chunk-level products
against the incoming state, and `kda/kda_stage_test.py` runs the whole thing on
the transactional emulator against the sequential recurrence. **PASS** for both
the token outputs and the carried state.

### Two layout decisions that fall out of the hardware

**The error is carried transposed.** The final state needs `E^T @ k_end`, a
contraction over time on *both* operands, and the systolic array contracts a VRAM
operand's lanes against an MRAM operand's rows. Holding `v` and `E` as
`[value_dim, chunk]` — time on lanes — makes all seven products land on
`vram_sub_projection_to` or its `_T_to` sibling with **no explicit transpose
anywhere**. Same move `ssd_state_update_v0` makes when it demands `b_t_chunk`
with time on lanes.

**Every product's second operand is spilled to HBM.** There is no weight in this
kernel — only activations and state — and MRAM is writable only by
`H_PREFETCH_M`. So `k_tilde`, `k_hat`, `T`, `state`, `E` and `k_end` each go out
through `store()` and come back as MRAM. That is six spills per chunk per head,
and it is the cost of using the matrix engine at all here.

### Three things that had to be found by running it

**1. The spill path NaNs under the shipped precision settings.** Every output was
`nan`. `SPILLED_ACTIVATION` selects the `keyvalue` precision class with
`set_scale=False`, and `[TRANSACTIONAL.PRECISION.HBM_*_KV_TYPE]` makes that
Mx/e4m3 with a separate scale stream — so the read walks into the scale stream,
whose `0x7f` bytes decode to e4m3 NaN. `ProgramSSDMixin.require_bf16_kv_precision`
documents exactly this and **is called by nothing**; Mamba's prefill path has the
same exposure and no test that would show it. The KDA stage test now rewrites its
per-build TOML to Plain BF16 and says why.

**2. The emulator's HBM was sized from the staged inputs alone.** The spills sit
past all of them, and the emulator indexed off the end of its HBM vector and
panicked rather than reporting a size problem. The harness now writes
`hbm_size.txt` from the compiler's own allocator high-water mark.

**3. `[*, chunk]` tiles must be zero-padded to `mlen`.** The projections contract
whole blocks, so a tile narrower than `mlen` contracts against the wrong lanes.

### The chunk-size claim in this entry was wrong — see the review below

I recommended chunk 8 over 16 on the strength of a conditioning table. The table
was measured with **unnormalised** `q`/`k`, which is not what reaches the
recurrence. Corrected numbers are in the review section.

### Known limitation

`key_dim <= mlen`. The decode path folds the key block into the row index;
prefill cannot yet, because the state's rows here are values and the projections
index it by value block alone. Kimi K3's `key_dim` of 128 against `mlen` 64 needs
this. It is **refused with a clear error** rather than silently mis-tiled.

## Review of Tasks 9–10 — a defect pair that masked itself, and a conclusion of mine that was wrong

The algebra was independently re-derived from `kda_step` and confirmed to
2.7e-16 in float64 across chunk 4/8/16 and three seeds. Everything below is
about what surrounded it.

### 1. Two defences for one mistake, each hiding the other

`M` was masked strictly **and** the substitution reads only `j < i`. Three
mutations — swapping `M`'s mask for the inclusive one, dropping the `-1` from
it, extending the sweep to `j <= i` — each left the emulator's answer
**bit-identical to baseline**. Only two applied together broke it, by seven
orders of magnitude.

The mask is gone. The substitution's bound is the one that stays, and it is
pinned by `test_ut_transform_ignores_m_on_and_above_the_diagonal`, which feeds a
polluted diagonal and demands an identical result. `M`'s upper triangle is now
left holding large-but-finite values that nothing reads — which is what
`kda_chunk_check_range` actually guarantees. Its docstring claimed the bound
stops `M` becoming `nan`; the real thing it protects is `k_tilde` being storable
in bf16 at all.

### 2. My conditioning conclusion was an artefact of the test's inputs

The stage test fed raw projections (`|k| ≈ 5`) and unnaturally mild decays.
`reference.py::kda_step` normalises `q` and `k` with `rsqrt(sum + 1e-6)` first,
and so does the lowering. With the distribution that actually reaches the
recurrence:

| | cond(L) | f32 | bf16 out | bf16 state |
|---|---|---|---|---|
| normalised, decay `e^-5..1` | **1.0 – 1.1** | 2e-08 | 4e-04 | 4e-03 |
| raw projections, `\|k\| ≈ 5` | 1e6 – 6e10 | 3e+21 | 2e+25 | — |

`cond(L)` is **1.0 at every chunk size**, the error is **flat in chunk**, and on
the emulator chunk 16 now passes at every seed — including the one that failed
before. Max absolute error dropped about 100× (0.875 → 0.008). **Chunk 16 stands;
the chunk-8 recommendation is withdrawn.**

Nor does it compound. Eight chunks chained against an equal-length sequential
reference stay flat at 2–5e-3, and on the emulator three chunks give a *smaller*
error than one (0.0083 against 0.0132) — the decay makes the recurrence strongly
contracting, so old error is damped rather than accumulated.

### 3. The spill path: two defects that had to be fixed together

**Over-read.** `store()` sizes an HBM region from the tile's real height;
`load_sub_matrix_*` prefetch a whole `mlen × mlen` block regardless
(`k_block_count` selects whole blocks and cannot trim a partial one). At chunk 16
that is 2 KB written and 8 KB read. The six spills sat **1 KB apart**, so
`k_tilde`'s prefetch ran through the next seven regions. Only
`kda_prefill_state_tail_v0` contracts over MRAM *rows*, and it reads `k_end` —
which happened to be allocated last, with nothing after it. Adding a spill or
reordering them would have broken it silently.

**Never reclaimed.** Each call allocated six fresh regions, ~20 KB per chunk —
about 10 MB per head over a 4k-token prefill.

The interlock: reusing addresses to fix the second makes the first read the
*previous chunk's* real data instead of unallocated zeros. Fixed together —
one region per name, tiles `mlen` rows tall with the tail zeroed before each
store. **HBM growth after the first chunk is now zero**, and the regions sit
8 KB apart so each prefetch lands inside its own.

### 4. `require_bf16_kv_precision` rejected the configuration it demands

It read `kind` where the TOML spells `format`, so it returned `None` for every
configuration — rejecting the Plain BF16 build it exists to require as well as
the MX one it exists to refuse. A check that could not pass is why **nothing
called it**: not the four SSD emitters, not the KDA one, not a test.

Fixed, pinned in both directions, and the emitters now check the active build
themselves — `SPILLED_ACTIVATION` is chosen by the emitter, so putting the
burden on a caller was the wrong division. Under the shipped settings KDA prefill
produced every output `nan`; it now refuses at compile time and names the
setting. **Mamba's four SSD emitters had the identical exposure** and are covered
by the same change.

### 5. The prefill layer had no pytest coverage, and every tiling loop ran once

Its only caller was the hand-run emulator script. And `chunk > mlen` is refused,
`key_blocks != 1` is refused, and the test pinned `value_dim == mlen` — so
`t_blocks`, `key_blocks` and `val_blocks` were all 1 and *any* block-index error
was undetectable.

`test_kda_prefill_structure.py` parses the emitted assembly, where each
projection announces its operands and indices, and pins the seven products in
order. Seven mutations, all red — including dropping the spill tail zero-fill,
which is **invisible on the emulator**: every spilled tile is freshly allocated
so its tail is already zero, and the state projection's other operand happens to
be zero there too. One accident masking a whole class of over-read, now pinned as
an invariant.

### 6. Prefill and decode hold the state transposed relative to each other

decode is `[key, value]` (row progressions become hardware loops; shared with
Mamba). prefill is `[value, key]` (all seven products land on the projection
primitives without a transpose). Both deliberate — and at Kimi K3
`key_dim == value_dim == 128`, so the shapes match and passing one to the other
is a finite plausible wrong answer.

`kda_prefill_state_to_decode_layout_v0` converts at the boundary, one projection
against a staged identity. Verified on the emulator: **exact, zero error**.

### Performance

Broadcasting one row across a tile was a copy per row — 64 at `mlen` 64, 128 at
Kimi's `value_dim`. Doubling the filled span makes it `ceil(log2(n)) + 1`:
prefill at chunk 16 goes **1,803 → 1,476** static instructions.

Against decode, at `mlen` 64, one head, `key_dim = value_dim = 64`:

| | static | per token |
|---|---|---|
| prefill, chunk 16 | 1,476 | 92.2 |
| prefill, chunk 8 | 1,039 | 129.9 |
| decode × 16 | 1,152 | 72 |

**Prefill is still 1.28× more static instructions than running decode 16 times.**
That comparison is not the case for prefill, though — instruction count is not
cycles. Prefill's seven `M_MM_WO` each drive 64³ MACs through the systolic array
while decode's are cheap `V_FMA_VF`. One chunk of 16 tokens measures
`sim_latency_cycles = 173,939`; the decode comparison at the same `mlen` is
Task 14's job and is the number that decides it.

328 passed, 3 pre-existing failures. Emulator: 112 passed, and the KDA stage
tests `cumprod`, `ut`, `prefill_out`, `prefill_state`, `prefill_chain_out`,
`prefill_chain_state`, `state_transpose` all pass.

---

## Task 11 — the whole-model KDA lowering — next

## Task 11 — whole-model lowering — done

### I claimed a blocker. It was not one.

My first pass said the layer could not be assembled without giving
`kda_conv_step_v0` and `kda_l2_normalize_blocked_v0` a column-block index, and
recommended that as the fix. An adversarial pass took the claim apart and built
a working layer instead. Both halves of my premise were wrong:

* **`kda_conv_step_v0` does not walk feature blocks as rows.** Of its six tile
  operands only `x_new` comes from the projection, and it is read at exactly one
  place — a single row index. Everything else is a dense tile the compiler
  allocates.
* **`kda_l2_normalize_blocked_v0` is not on the projection path at all.** The
  mixer's `q` and `k` are the *convolutions'* output, which is dense — my own
  `test_conv_channel_blocks_and_mixer_key_blocks_are_the_same_rows` pins that
  seam. The normalisation never sees a projection tile.

And "`tile_col_idx` already threads through the `tile_row_*` family" was true of
the ISA layer and false of the Var layer the KDA emitters actually call, where
every wrapper drops it. So my recommended option was not "add a parameter" but
"build a cross-column-block copy primitive". Measured, it cost **+25%** on the
normalisation and lost a cross-block hardware loop — to solve a problem that is
not on the path.

My cost estimate for the alternative was wrong by two orders of magnitude too: I
said a scatter would be ~700 block operations per layer; done naively it is
7,674 static instructions (6.7% of the layer), and done properly it is **70**.

### What the layer actually needed

Column block `c` sits at `base + c * physical_rows * mlen` and rows within a
block are linear, so a projection tile's bytes already **are** a dense
`[blocks * physical_rows, mlen]` tile — feature block `c`'s token `t` at row
`c * stride + t`. `vram_tall_view` names it; `kda_gather_projection_v0` moves a
section into place with **one `V_FMA_VF` sweep**, because both the source rows
(step `stride`) and the destination rows (step 1) are progressions.

**14 static instructions per section, independent of block count.** A 96-head
gather costs exactly what a 2-head one costs — pinned by a test. All five
sections of a Kimi K3 layer: **70**.

`test_the_assembled_layer_matches_kda_state_engine_step` now runs gather → three
convolutions → gates → recurrence against the reference's own boundary. **Zero
emitter changes were needed.**

### The address bug both reviewers found

`vram_column_block_view` used `base + c * mlen * mlen`. The real stride is
`c * physical_rows * mlen` (`memory.py`'s `col_block_base`); they coincide only
when a tile is exactly `mlen` rows tall, which was the one shape I tested. A
projection allocated `strict=False` is padded to `blen` rows, so a real decode
projection was off by `mlen / blen` and the view pointed into a different tensor.

**Both the wrong formula and the right one passed the whole suite.** The test
seeded and read at the view's *own* address, so it was self-consistent with any
address the view invented. It now anchors against `_tile_addr` on the parent, at
four shapes including two row blocks and a `strict=False` padded one, and checks
a neighbouring block is untouched. The old formula now fails four tests.

Also fixed: the duplicate-name hazard (`register_vram_matrix` overwrites silently
and views resolve by name at emit time, so with a default prefix the first two
splits in a program collide — and the layer runs 93 times); reading the var's
`physical_shape` where the layout is authoritative; and validating `key_width`
where the invariant is per *head* — 4 heads of `key_dim` 6 gives a `key_width`
divisible by `mlen` with three heads starting mid-block. `kda_key_blocks` already
raised exactly that error and was imported but never called.

`alloc_at` already existed and did what I had reimplemented; the view now goes
through it.

### Measured, Kimi K3 at mlen 64

| | static | share |
|---|---|---|
| three convolutions | **53,757** | 58% |
| mixer | **39,526** | 42% |
| gather | **70** | 0.07% |
| **layer** | **93,353** | |
| **× 93 layers** | **8,681,829** | |

**The convolutions are the larger half and were ungated** — the budget covered
42% of what it was meant to protect. Now gated at 59,000, with a whole-layer gate
at 103,000 and a test that the gather stays a rounding error.

A whole-model figure that includes projections, MoE and embeddings is not comparable;
8.68M here is the state-engine path only, without projections, MoE or
embeddings, so the two are still not comparable.

### Prefill is closer than I said

`kda_l2_normalize_v0`, `mamba_conv1d_v0` and `kda_chunk_decay_cumprod_v0` all run
directly on a column-block view today. What prefill still lacks is a `v_t`
transpose, a beta stride, a per-token decay activation, and an assembler — and
above all **`key_dim <= mlen`**, which Kimi K3 (128 against 64) fails. That is
the real prefill gap, and it is larger than the seam I had been calling one.

*(Task 16 lifts the `key_dim <= mlen` limit, and `value_dim <= mlen` with it.)*

351 passed, 3 pre-existing failures. Emulator 112 passed.


## Task 12 — the layer executed in the emulator, first layer to last

This path had never been run as machine code. It now is.

### On the transactional emulator, against `kda_state_engine_step`

| case | result | max abs error |
|---|---|---|
| `layer` — one assembled layer | **PASS** | 2.14e-04 |
| `layer_chain` — four, back to back | **PASS** | 2.75e-04 |

Gather, three convolutions, the gates and the recurrence, compiled, assembled,
and executed by the Rust emulator — not the float64 ISA interpreter. Three heads,
`key_dim` 128, so `q` and `k` each span two blocks.

Four layers give 2.75e-04 against one layer's 2.14e-04: composing layers costs
essentially nothing, because each carries its own state and the error does not
chain.

### Two things only depth could find

**1. The second layer collided on view names.** `kda_layer_from_projected_v0`
named its tall view with a fixed prefix, so layer 1 and layer 2 asked for the
same VRAM object. The duplicate-name guard added in the Task 11 review turned
that into an error at compile time; without it,
`register_vram_matrix` overwrites silently and views resolve by name at emit
time, so **layer 1's views would have been repointed at layer 2's projection**.
Fixed by naming views after the projection tile, which is unique per layer — so
stacking needs nothing from the caller.

**2. FPRAM is scratch, not storage.** Allocating the mixer's window per layer
overflowed the compiler's optimistic 1024 slots **at the fourth layer** — and the
hardware has 512. One window serves the whole model, the same reuse the mixer
already does per head. Pinned: FPRAM use must not grow with depth.

### The HBM address space — the constraint that decides executability

The emulator preloads HBM from a flat file starting at offset 0, so the
allocation and the file both have to span every address the program touches. A
layout whose regions sit far apart needs a span-sized allocation whatever the
live data comes to, and stops being executable long before it stops being
describable. Addressing for the state-engine path, excluding the projection
weights any implementation needs:

| | |
|---|---|
| per layer | **2,336,832 bytes** (2,282 KiB) |
| 93 layers | **217,325,376 bytes** (0.202 GiB) |
| emulator `HBM_SIZE` | 16 GiB |

**Two orders of magnitude below the wall.** (The first version of this said 0.33
MB per layer — seven times too low, because the measuring helper put the state
and the projection in VRAM while the emulator test staged them from HBM. See the
review below.)

### What Task 12 does not claim

This is the KDA state-engine path first layer to last, not a whole model: the
input and output projections, the MoE blocks, the norms and the embeddings are
shared code with no Nemotron-3 or Kimi-K3 config in this repository to
instantiate them from. Task 13's real-checkpoint binding needs that config
first, and building it is a larger piece of work than anything in Phases 0–2.

354 passed, 3 pre-existing failures.

## Task 14 — the banking study and the measurement report — done

### The banking question has a decisive answer: stalls ≈ 0

`analytic_models/performance/vector_sram_banking.py` extracts every hardware
loop's pointer stride **from the emitted assembly** — not from a description of
the algorithm — and models row-interleaved banking.

| kernel | loops | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| KDA recurrence (Kimi K3) | 20 | 0.6% | 0.9% | 1.0% | 1.1% | 1.1% |
| KDA mixer, one head | 48 | 1.2% | 1.8% | 2.1% | 2.3% | 2.3% |
| Mamba-2 decode step | 7 | 1.0% | 1.5% | 1.8% | 1.9% | 2.0% |

The dominant loops are **stride 1 over 128 trips**, which visits every bank in
turn and never conflicts. The 1–2% residue is entirely single-trip loops with a
pinned pointer. **No evidence for a Vector SRAM bank-mapping mode.**

Worth noting the emulator does not model banking at all — `VectorSram` is a flat
row vector — so this is analytic, and a word-interleaved mapping would moot the
question regardless.

### The follow-up the plan predicted, now quantified

| kernel | dynamic | `S_ADDI_INT` | `S_LD_FP` | `V_FMA_VF` |
|---|---|---|---|---|
| KDA recurrence (Kimi K3) | 4,179 | **50.0%** | 24.6% | 18.4% |
| KDA mixer, one head | 4,398 | **50.2%** | 23.9% | 17.5% |
| Mamba-2 decode step | 730 | **46.4%** | 26.6% | 17.7% |

**Half the dynamic stream is pointer arithmetic and under a fifth is
arithmetic.** Post-increment addressing on the vector pointers, plus an
auto-advancing FP pointer, would remove up to three-quarters of the dynamic
instructions on the recurrent path — a far larger effect than banking, and the
next instruction-set question worth asking.

`doc/static_path_measurements.md` collects everything, with hard facts and
uncalibrated model output kept apart and **no comparison drawn against any other
uncalibrated latency figure**, because two of those decide nothing between them.

## Task 15 — nothing to retire, and a guard so it stays that way

There is no descriptor machinery, no command queue, no residency cache,
no `spec/`, no descriptor, no queue, no residency cache. This branch was cut from
`a4b3e7de` on main, which never had any of them — the plan's deletion steps have
nothing to delete.

That is the answer to the question this whole project started from. The static
path needs **three** new instructions -- `V_SOFTPLUS_V` (`0x39`), `S_MAP_FP_V`
(`0x3A`) and `V_FMA_VF` (`0x3B`) -- and every one is an ordinary fixed-function
op with its operands named in the instruction word. Everything else — the transposed state layout, the per-head
streaming, the shared FPRAM window, the chunked prefill, the projection gather —
is compiler work.

`aten/tests/test_no_state_engine.py` makes that a property rather than an
observation: no run-time state-engine vocabulary in any tracked source file,
those directories absent, `0x3D` and `0x3F` still free, and the three opcodes
this work adds each pinned to its slot. A grep is a check that ran once; a test
is one that keeps running.

Two docstrings in the CPU reference still named the boundary after a
descriptor-driven instruction. Reworded — the boundary is the right place to
compare against, and the name was for something this design does not have. The
state dataclass moved with them, from `KdaXState` to `KdaRecurrentState`.

358 passed, 3 pre-existing failures.

## Task 13 — real checkpoint weights — done for Mamba-2, blocked for KDA

Whole-model
artifacts were *"never bound to a real checkpoint"*. One is bound now.

### `AntonV/mamba2-130m-hf`, layer 0

A published Mamba-2: 24 layers, `hidden_size` 768, `state_size` 128, 24 heads of
`head_dim` 64. `ssm_decode_step_v0` runs on its real `A_log`, `D` and `dt_bias`
and matches `mamba2_recurrent_reference` to **rtol 1e-4 / atol 1e-5**, on both
the output and the carried state.

What real weights test that `randn` cannot:

* **Structure.** Layer 0's `A` is not a spread — 23 of 24 heads sit between
  −0.73 and −0.27, and one sits at **−5.06**, about ten times the rest. A tight
  cluster plus a single outlier is what training produces and a symmetric draw
  does not, and that outlier head is the one most likely to expose a decay that
  underflows. Pinned as a test.
* **Layout.** `in_proj` packs `z | x | B | C | dt` into one 3352-wide tensor, and
  `conv1d.weight` is `[channels, 1, kernel]` rather than `[channels, kernel]` —
  the reference accepts both, so a lowering that assumed the wrong one would
  silently transpose a 4-tap filter. Pinned.

I first asserted `A.min() < -10` on the strength of nothing; the real value is
−5.06. Corrected to assert the structure that is actually there.

The test **skips when the checkpoint is not cached**, so CI never needs network.

### KDA's half is blocked, and on what

No published KDA checkpoint was found whose parameter semantics match
`aten/models/kda/reference.py` closely enough to bind without guessing. The
gated-delta-net models on the Hub (`m-a-p/*-GatedDeltaNet-*`,
`linear-moe-hub/Gated-Deltanet-*`, `Idiap/gated-deltanet-swa-1.4B-30B`) are the
right family, but matching a checkpoint's exact gate parameterisation to this
reference is a verification task of its own — binding the wrong semantics would
produce a green test that proves nothing, which is worse than no test.

That is the remaining gap, and it is bounded: one checkpoint, read carefully
against `activate_log_decay` and `kda_step`.

361 passed, 3 pre-existing failures.

---

## Where the plan stands

| Phase | | |
|---|---|---|
| 0 — KDA decode, no new opcode | Tasks 1–4 | **done** |
| 1 — `V_FMA_VF`, measured | Tasks 5–8 | **done** |
| 2 — chunked prefill | Tasks 9–10 | **done** |
| 3 — whole model, executed | Tasks 11–12 | **done** |
| 3 — real checkpoint | Task 13 | **Mamba done, KDA blocked on a checkpoint** |
| 4 — measure, retire | Tasks 14–15 | **done** |

The two things the project set out to establish both hold, and both are now
demonstrated rather than argued:

**KDA and Mamba-2 need no descriptor machinery.** Three new instructions --
`V_SOFTPLUS_V`, `S_MAP_FP_V` and `V_FMA_VF`, at `0x39`-`0x3B` -- each an
ordinary fixed-function op. No descriptors read at run time, no queue, no
residency cache — and a test that keeps it that way.

*(An earlier version of this record said "one new instruction". That counted
only `V_FMA_VF`, the one the performance argument turns on, and omitted the two
the Mamba path needs. Relative to `main` this work adds three.)*

**The static form is smaller, not merely equivalent.** The recurrences went from
linear in the contracted dimension to constant — 7,345 static instructions to 76
at Kimi K3's `key_dim` — because removing the scratch row let the sweeps become
hardware loops. The whole path's HBM footprint is 0.029 GiB, three orders of
magnitude inside the emulator's flat allocation.

What is still open, in order of size: a Nemotron-3 / Kimi-K3 model config (which
Tasks 11–13 all touch), prefill for `key_dim > mlen`, and post-increment
addressing — which the measurements say is worth more than anything else on the
list, since half the dynamic instruction stream is pointer arithmetic.

*(`key_dim > mlen` is closed by Task 16, below. The model config and
post-increment addressing are still open.)*


## Review of Task 12 — the emulator case could not fail

The most serious finding in this project so far, and it was mine.

### `case_layer`'s tolerance was larger than its signal

`atol=2e-2` against a golden whose largest value is **0.01215**. Every one of
the 192 compared values sat inside the absolute bound, so **writing zeros into
the output tile scored 100%**. Verified, along with three more that all passed:

| mutation | before | after |
|---|---|---|
| zero the output tile | PASS | **FAIL** |
| skip the `v` convolution | PASS | **FAIL** |
| swap the `q` and `k` gather sections | PASS | **FAIL** (6.25% match) |
| zero every layer but the last (`layer_chain`) | PASS | **FAIL** |

It is `atol=0` with a relative bound now. The honest numbers, which the
absolute figure was hiding: **mean relative error 4.3%**, and 96.35% of lanes
inside 12% (the harness passes at ≥90% of lanes, not all).

4.3% is what bf16 gives — the read-out contracts 128 terms, so error grows like
`sqrt(128) * 2^-9 = 2.2%` before the rest of the chain. **The lowering is right
to 5e-4 in float64**, which `test_the_assembled_layer_matches_kda_state_engine_step`
checks on the ISA interpreter. The emulator case adds bf16 realism, not
verification power — and quoting only its absolute error implied otherwise.

### `layer_chain` tested one layer

`target = outs[-1]` with `golden = golden_rows[-1]`, so layers 0–2 were never
compared. Zeroing all their staged inputs left the verdict **bit-identical**.

So **"the error does not chain" was not a measurement** — nothing was chained
(the four layers are independent by construction), and the earlier layers were
outside the comparison anyway. Worse, the code to pack every layer existed but
was gated on `if heads == 1 else None`, and the only invocation the commit
recorded was `--num-heads 3`. Every layer is compared now, at every head count.

Still not compared: the updated conv history and recurrent state. For a decode
token the state is the layer's only persistent effect, so a mixer that produces
the right output and corrupts the state would pass.

### The FPRAM test proved something else

No KDA emitter allocates FPRAM at all — every slot comes from the caller, and
the test's own helper shares the window. So it could not catch the failure its
commit described, which is a *caller* allocating per layer.

It now asserts the number that actually binds: **492 slots at 93 layers against
`FP_SRAM_DEPTH` 512**. The compiler's allocator defaults to 1024 and would not
catch an overflow, which is why the hardware figure is asserted directly.

### The HBM figure counted one seventh of the traffic

The helper put the recurrent state, `dt_bias` and the projection tile in **VRAM**
while `case_layer` staged them from **HBM** — two tests disagreeing about where a
layer's state lives. Counted consistently, a layer is **2,336,832 bytes**, not
331,776, and **4.5× over the gate I had set**. 93 layers is 0.202 GiB, which
still sits three orders of magnitude inside the emulator's flat allocation.

### Cycles, newly measured

| | cycles |
|---|---|
| `layer` (1 token, 3 heads, `key_dim` 128) | **23,715** |
| `layer_chain` (4 layers) | **96,114** (24,029/layer, linear) |

By stage marker, inside the real program: `kda_state_update` + `kda_readout` =
**24.9%**, convolutions 13.8%, normalisation 9.2%, gather **0.9%**. The 38%
attributed to "other" is HBM prefetch of this harness's per-layer staged inputs,
which is a property of the harness rather than of a decode loop.

361 passed, 3 pre-existing failures.

## Task 16 — prefill past one block: `key_dim` and `value_dim` above `mlen`

The open item Task 12's summary put second is closed. Kimi K3's 128-wide head
prefills at `mlen` 64, and so does its 128-wide value axis, which nothing had
ever run.

### The fix is a layout choice, not new machinery

The key axis moved from **rows** onto **lanes**: a `[chunk, key_dim]` tile is now
`chunk` rows across `key_dim / mlen` column blocks rather than
`chunk * key_blocks` rows one block wide.

That is what the systolic array already wanted. Five of the seven products
contract over key, and `vram_sub_projection_asm_impl` walks a VRAM operand's
column blocks at stride `physical_rows * mlen`, accumulating in the array — so a
key axis on lanes contracts across as many blocks as it needs with no extra
instruction and no explicit sum. **Not one projection call changed.** What
changed is the tile widths and the row ops, which now run once per column block.

The alternative — keeping the folded rows and computing the gram of the folded
tile — computes the `kb != kb'` cross terms and throws half of them away, then
needs a strided gather to recover the diagonal. It was never worth writing down.

### Verified as machine code, every case at 100% of values in tolerance

| case | `key_dim` × `value_dim` | max abs error |
|---|---|---|
| state layout transpose | 128 × 128 | **0.00** |
| prefill, out | 128 × 128 | 6.71e-04 |
| prefill, state | 128 × 128 | 9.28e-03 |
| three chunks chained, out | 128 × 128 | 6.10e-04 |
| three chunks chained, state | 128 × 128 | 7.32e-03 |

Static instructions at chunk 16, one head: 1,476 at 64 × 64 — **unchanged** by
the layout move, so the lift cost nothing at the shape that already worked — and
2,426 at Kimi K3's 128 × 128, which is 1.6x the instructions for 4x the state.

### The defect this turned up, and the tolerance that reported it as a pass

`out = scale * (...)` ended in a single `tile_row_mul_fp_broadcast`, and a row op
reaches one column block per call. At `value_dim` 128 the upper half of every
token went unscaled — `sqrt(key_dim)` = 11.3x too large.

The case printed **PASSED**. `check_mem`'s `allclose_pass` is
`match_rate >= 90.0`, and 94.68% of values sat inside an `atol` of 5e-2 on data
of order 1e-3 — a tolerance fifty times the signal. Comparing the value blocks
separately is what showed it: block 0 at 6.4e-04, block 1 at 1.4e-01, mean
relative error 11.28 against a predicted 11.31.

So the stage test now restates the bar at 100% on the same numbers.
`check_mem` is shared and its rule is left alone. Under the new bar four
mutations of the column-block loops die, and **only one of them needed it**:

| mutation | match rate | caught by the 90% rule? |
|---|---|---|
| output scale on block 0 only *(the real defect)* | 94.68% | **no** |
| `k_hat`/`q_hat` on block 0 only | 0.00% | yes |
| `A_C * state` on block 0 only | 55.79% | yes |
| `k / A` on block 0 only | 85.80% | yes |

This is the second time in this project a tolerance larger than the signal has
hidden a real error, after `case_layer`'s. The layer cases moved to `atol`
2.5e-04 alongside their `rtol` 0.12 for the mirror-image reason: three of their
sixty-four outputs are near 1e-03 and bf16 delivers ~1.8e-04 absolute whatever
the value, so a *pure relative* bound is unmeetable there. They had been passing
at 95.31% on the 90% rule, not on the tolerance. The floor keeps the check real —
an all-zero output scores 1.6% against it.

### Two smaller things fixed on the way

* `kda_prefill_spill_v0` zero-filled `range(live_rows, mlen)`. For a tile taller
  than one block — the state at `value_dim` 128 — that range is empty, so the
  dead rows past `mlen` were never zeroed. It is `range(live_rows, tile.shape[0])`
  now.
* `contrib` was one scratch tile serving three products of three different
  shapes. They are `contrib`, `readout_contrib` and `state_contrib` now. Sizing
  one tile to the maximum would have passed `vram_add`'s width check by
  coincidence whenever `key_dim >= chunk`, and added the wrong lanes when not.

`kda_prefill_tile_shapes` is now the single source of the tile shapes, consulted
by the emitter's validation *and* by every caller. They used to be written
separately — the emitter kept a hand-written list of row counts and the callers
allocated `mlen x mlen` for everything, which agreed only at the one shape anyone
ran. That divergence is how the spill over-read got in.

377 passed, 2 pre-existing failures. All 17 emulator stage cases at 100%.

### Still refused

`key_dim` and `value_dim` must each be a whole multiple of `mlen`, and `chunk`
must fit one block. The state transpose additionally requires its output to be
exactly `key_dim` rows when there is more than one value block, since that is
what makes its column-block layout coincide with decode's row indexing.


## Correction — one of the "3 pre-existing failures" was mine

Every task record above closes with "N passed, **3** pre-existing failures". Two
of those three are genuinely pre-existing. The third was introduced by this
branch, and calling it pre-existing was wrong. CI is what caught it: the
`moe-stage-guard` job failed on the PR, and it passes on `origin/main`.

`test_moe_stage_attribution.py::test_stage_parameters_have_no_default` forbids an
emitter from giving its `stage` parameter a default, because a caller that
forgets the argument is then billed to the wrong stage silently rather than
failing. Three emitters **this branch adds** gave it one:

    program_mamba_common.py   mamba_silu_v0(stage='mamba_gated_norm')
    program_ssd.py            ssd_transposed_projection_v0(stage='mamba_in_proj')
    program_ssm_recurrent.py  ssm_decode_scalars_to_fpram_v0(stage='mamba_dt')

Checked against `origin/main` in a separate worktree: the guard passes there (9
passed) and all three files are additions on this branch. The other two failures
(`test_qwen_packed_skinny_router_rowpacked_compiles_for_128_experts`,
`test_mxfp8_is_sole_gap_source`) do fail on `origin/main`, so those labels stand.

The fix. `mamba_silu_v0` takes a **required `marker`** instead of a defaulted
`stage`: it serves both the Mamba and the KDA vocabulary, so no default is right
for both, and its own KDA caller was already working around the default by
passing `marker=`. Markers are sticky, so accepting `mamba_gated_norm` there
billed KDA's SiLU *and everything emitted after it* to a Mamba stage — which is
precisely the failure the guard exists to prevent, sitting in the tree
unreported. The other two emitters have no callers at all; `stage` is now a
required keyword there, so whoever wires them has to name one.

**Why the local runs never showed this.** The local suite reported the same three
failures every time and I read the count rather than the names, then carried
"3 pre-existing" forward from record to record without re-deriving it. The check
that settles it — run the failing test against `origin/main` — takes one command
and I did not run it until CI forced the question. A failure inherited from the
base branch and a failure you introduced look identical in a summary line.

## Correction — the second CI failure was also mine

`unit-tests` ("Generator unit tests") installs `torch` and `transformers`. Its
original steps run the generator tests as plain scripts, so it never needed
pytest. This branch adds sixteen steps to that job that invoke
`python3 -m pytest`, and every one of them failed with `No module named pytest`.

The job now installs pytest. Nothing was wrong with the tests themselves — they
pass locally and in the `moe-stage-guard` job, which installs pytest explicitly.
Adding a step to a job without checking what that job has installed is the whole
of the mistake.

## Correction — the parent repo's CI never ran at all, for two more reasons

The compiler PR's checks were the visible failure. The parent's were worse: all
three jobs failed, twice at `actions/checkout` and once at formatting, so the
emulator was never built or tested in CI on this branch even once.

### `actions/checkout` died on the `PLENA_Tools` submodule

`08db1bb9` is on no branch of `AICrossSim/PLENA_Tools`, so
`git submodule update --init --recursive` cannot fetch it and the job ends
before any step runs. This branch has been carrying that gitlink since
`6ec1e259`, and it has been noted in the PR description as "the submodule will
show as broken here" — which undersold it. It does not show as broken. **It
makes the branch impossible to check out**, for CI and for any reviewer.

The bump is now dropped: the gitlink is back to `6b31fe00`, which is what
`origin/main` points at. That is safe because nothing here needs the newer
commit, and it was checked rather than assumed — all six KDA stage cases
(`cumprod`, `ut`, `state_transpose`, `prefill_out`, `prefill_state`, `layer`)
pass against `6b31fe00`. The only `check_mem.py` change between the two is a
NaN/Inf hard-failure rule, and this branch's stage tests carry a stricter bar of
their own. The `physical_rows` / `col_block_stride` support the multi-block
comparison depends on is already present at `6b31fe00`.

If that Tools commit is wanted, the fix is to push it to
`AICrossSim/PLENA_Tools` and bump again — not to carry an unreachable pin.

### `cargo fmt --all -- --check` failed on three hunks of this branch's Rust

Two in `op.rs`, one in `vector_machine.rs`'s `fma_scalar`. The Nix dev shell
ships `rustfmt` but not the `cargo fmt` subcommand, so `cargo fmt` reports
"not installed for the toolchain" and I took that to mean formatting could not
be checked locally. It could: `rustfmt --edition 2024 --check <files>` runs fine
in the same shell and reproduces exactly what CI reports. The files are
formatted now.

`cargo clippy` is genuinely unavailable in that shell, and CI's clippy step never
ran because the formatting step failed first. So clippy is still unverified
here.

## Correction — the simulator CI was not "stuck", the PR had conflicts

I reported that the parent repo's Actions was backed up, on the evidence that no
workflow runs were being created for the branch while two `main` runs sat queued
for hours. That was the wrong conclusion from a true observation.

`gh api .../pulls/115 --jq .mergeable_state` says **`dirty`**. A `pull_request`
workflow runs against `refs/pull/<n>/merge`, and GitHub cannot compute that ref
when the merge conflicts — so no run is created at all, which looks exactly like
a stalled queue from the outside. One field settles it and I did not check it.

`main` had moved one commit ahead: #114, "pipelined `--timing-model` scoreboard
with real DMA overlap". It conflicts with this branch in `dispatch.rs` and
`stage_profile.rs`.

### The merge was not mechanical

#114 **deleted** `classify_timing_access` from `dispatch.rs` and replaced it with
an `OpAccess` model in a new `accelerator/access.rs`. Of this branch's seven
hunks in `dispatch.rs` — 139 lines, all insertions — four still applied (two
execution arms, two `resource_kind_for_opcode` arms) and three had to be
re-derived against the new model.

The three opcodes are classified in `access.rs` now:

* `V_FMA_VF` gets its own arm rather than joining the `V_*_VF` family, because it
  reads its **destination** as well as its source. It lists
  `vector(gp(rd), vector_tile)` in `reads`; grouping it with the others
  under-reports its vector-SRAM traffic by a row, which is the number the FMA
  conversion is judged on.
* `V_SOFTPLUS_V` joins `V_EXP_V` / `V_RECI_V` unchanged.
* `S_MAP_FP_V` is `Unit::Vector`, not `Scalar`: it holds the vector SRAM read
  port for a whole row even though its destination is FP_MEM. Its `S_MAP_V_FP`
  mirror is the inverse and stays `Scalar`.

The exhaustive match is what forced this: adding an opcode to `op.rs` fails the
build until `access.rs` classifies it. That guard is the reason a re-derivation
happened at all rather than a silent omission.

`main`'s dispatch tests moved to `pipeline_tests.rs`, so this branch's
`mod tests` in `dispatch.rs` was restored with the two guards that still mean
something after the merge.

### And clippy finally ran

`flake.nix` never listed `clippy` or `rustfmt` in the toolchain extensions, so
the dev shell had neither, while CI gates on both. `cargo fmt`'s error —
"not installed for the toolchain, run `rustup component add rustfmt`" — is
cargo's generic advice; there is no rustup here and the components were simply
absent from the override. I read that message as "cannot be checked locally"
rather than as a one-line flake fix, which is how a formatting failure reached
CI.

Both are in the extensions now. On the merged tree: `cargo fmt --all -- --check`
clean, `cargo clippy --workspace --all-targets -- -D warnings` clean — **the
first time clippy has run on this work at all** — and 129 Rust tests pass, up
from 112, the difference being #114's pipeline tests plus this branch's.

## Official KDA decode layer connected end to end

The previous `layer` case began at an already packed projection and stopped at
the recurrent output. The `official_layer` case now executes the complete
official Kimi K3 decode order on the Rust transactional emulator:

```text
hidden
  -> q / k / v / decay_a / decay_b / beta / output_gate projections
  -> q / k / v short convolutions
  -> recurrent KDA update and readout
  -> per-head RMSNorm * learned weight * sigmoid(output_gate)
  -> output projection
```

The connected regression uses production tensor layouts and formulas at a
small shape (`MLEN=8`, two heads). It emits 2,829 ISA lines, runs in 7,249
simulator cycles, and compares all 16 output values against the FP32 CPU
reference. Every value passes (`max_abs_error=0.011719`). This is synthetic
weight validation, not a Kimi checkpoint run.

The connected machine uses its configured BF16 Vector SRAM, including for the
recurrent tensor. The official GPU implementation keeps that tensor in FP32.
Consequently this test proves the static dataflow and BF16 numerical path; it
does not claim bit-equivalence to the official FP32-state kernel.

Building this test exposed two generic Plain-BF16 HBM bugs in the compiler:
input regions were allocated at the configured MX size instead of two bytes per
element, and `H_PREFETCH_V` advanced row/chunk offsets in elements although the
DMA address is byte-based. Both now have direct regression tests.

The analytic model also owns a separate `HBM_STATE_TYPE` precision parameter,
defaulting to FP32 as measured on the official GPU path. Attention KV precision
can now change without silently changing recurrent-state traffic.
