# Static Mamba-2 + KDA on PLENA Implementation Plan

> **Superseded ISA note (2026-08-31).** This is the original implementation
> plan. The final conflict-free ABI reserves `0x39`-`0x3C` for routed MoE,
> encodes `V_FMA_VF` as the `V_MUL_VF` accumulate variant, and assigns
> `V_SOFTPLUS_V=0x3D`, `S_MAP_FP_V=0x3E`, and `L_CFG=0x3F`. Old opcode values
> below are retained only as historical task records.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Nemotron-3 Mamba-2 and Kimi-K3 KDA on PLENA — **prefill and decode, whole model, on real checkpoint weights** — using only statically-scheduled instructions: no memory-resident descriptors, no command queues, no residency cache.

**Architecture:** Recurrent state is an ordinary tensor: it lives in Vector SRAM at compiler-chosen addresses and moves with `H_PREFETCH_V` / `H_STORE_V`, exactly like a KV cache. The recurrence is a loop of row-wise vector ops. Both Mamba-2 and KDA store per-head state with the **contracted axis as the row axis** (Mamba: `state_size`; KDA: `key_dim`), which turns every step of both recurrences — decay, rank-1 update, and state contraction — into one broadcast-scalar operation per row.

**Phasing.** Functionality first, optimisation second, each measured before the next is designed:

| Phase | What lands | New opcodes | Why here |
|---|---|---:|---|
| **0** (Tasks 1–4) | KDA **decode** runs end to end, validated against the golden | **0** | Proving KDA needs no new instruction is itself the result worth being able to state. Uses the same `copy + multiply + add` triple the Mamba kernel uses today. |
| **1** (Tasks 5–8) | `V_FMA_VF` (`rd += rs1 * fp2`), both decode kernels converted onto it | 1 (`0x3B`) | Removing the scratch row makes each sweep a single arithmetic-progression row walk, which the existing emitter turns into a **hardware loop** instead of an unrolled block. Phase 0 supplies the measured baseline. |
| **2** (Tasks 9–10) | KDA **chunked prefill** on the Matrix Engine | 0 | Decode alone is half the function. Prefill is the half that had never been mapped onto the Matrix Engine. |
| **3** (Tasks 11–13) | Whole model lowered, **executed end to end in Rust**, bound to **real checkpoint weights** | 0 | The largest credibility gap in the project: no whole model has ever been run through the emulator here, and none has ever used real weights. |
| **4** (Tasks 14–15) | Banking study, uncalibrated latency report, no-descriptor invariant confirmed | 0 | Measurement and confirmation, last. |

KDA needs **no new instruction** to work: `V_SOFTPLUS_V` is Mamba's (`dt`), and `S_MAP_FP_V` — the one primitive the recurrence genuinely requires, to get per-key scalars from a VRAM row into FP registers — already ships on `feat/mamba2-support`. `V_FMA_VF` is therefore a measured optimisation, never a prerequisite.

**Tech Stack:** Python 3 (compiler / `aten` lowering, pytest), Rust (transactional emulator, `cargo test`), SystemVerilog headers (`doc/operation.svh`, shared ISA definition), `just` recipes for the test matrix.

## Global Constraints

- **Base branches.** Compiler work branches from `feat/mamba2-support` (Compiler `f5eb36a`). Emulator work branches from `feat/mamba2-support` (Simulator `65cf6f09`). Do **not** branch from any descriptor-driven line of work.
- **New branch names.** Compiler: `feat/static-kda`. Simulator: `feat/static-kda`.
- **Opcode allocation.** `V_FMA_VF = 6'h3B`. `0x39` (`V_SOFTPLUS_V`) and `0x3A` (`S_MAP_FP_V`) are already taken by `feat/mamba2-support`. Do not use `0x3D` or `0x3F` — leaving them unallocated keeps the two histories distinguishable.
- **No new memory-resident control structures.** No instruction may read a descriptor, command block, or any control word from HBM. Every instruction is fully defined by its 32-bit word plus sticky control registers.
- **No new runtime traps.** Shape, precision, and address violations are compile-time errors in Python or assembler errors — never a status code returned by hardware.
- **State precision is BF16 for Mamba-2, FP32 for KDA**, matching the profiled runtime configurations. Never store recurrent state as MX8: the docstring at `aten/plena/program_ssm_recurrent.py:104-115` gives the error-amplification argument; it applies unchanged to KDA and must be preserved in any new docstring.
- **Every state load and its matching store must agree on `storage_precision` and `hbm_element_bytes`.** Mismatch produces a wrong answer rather than an error — see `aten/plena/program_ssm_recurrent.py:120-126`.
- **Every new row-walking emitter must go through `self._row_progression(rows)` and emit a hardware loop when it returns non-`None`.** Falling back to the unrolled path for a long walk is the single biggest static-footprint regression available; it is the single biggest contributor to a bloated whole-model static footprint.
- **`MAX_LOOP_DEPTH = 4`** (`PLENA_RTL/src/definitions/configuration.svh:35`). No emitted loop nest may exceed 4 levels.
- **All emitters allocate GP registers via `self._reg.allocate_gp(n)` inside a `try` / `finally: self._reg.free_gp(gp_regs)`.** Never hardcode a GP index — `mamba_rsqrt_fpram` (`program_mamba_common.py:568-596`) documents why: the allocator hands registers out descending, so a hardcoded index silently clobbers a live value.
- **Test commands.** Compiler: `pytest` from the Compiler root. Emulator: `cargo test` from `transactional_emulator/`, plus `cargo clippy --all-targets -- -D warnings`.
- **Commit style.** Conventional commits (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`), matching existing history.

## Out of scope

Named here so a reviewer does not read them as gaps. Each is a separate subsystem needing its own plan.

- **Post-increment vector addressing.** See the Reference section: emitted instruction count is dominated by address arithmetic, not by arithmetic. A sticky "advance by STRIDE after each V-type op" mode would cut the inner loop from 6 instructions to 2. That is the highest-value follow-up, and it is a general ISA improvement rather than a Mamba/KDA one.
- **Vector SRAM bank mapping.** Task 14 measures whether row-walking a head tile conflicts. Only if it does should a bank-mapping mode be designed, in its own plan.

---

## Reference: the two recurrences

Stated in the exact form the lowering implements. Task authors should not re-derive them.

**Mamba-2 decode step**, per head `h`, group `g = h // heads_per_group`, state `S[n, :]` for `n` in `0..state_size`, each row `head_dim` wide:

```
xs[:]    = dt[h] * x[h, :]
S[n, :]  = dA[h] * S[n, :] + B[g, n] * xs[:]        for each n
y[h, :]  = sum_n C[g, n] * S[n, :]
y[h, :] += D[h] * x[h, :]
```

**KDA decode step**, per head `h`, state stored **transposed** as `T[k, :]` for `k` in `0..key_dim`, each row `value_dim` wide (`T[k, v] == S[v, k]` in the row-major reference):

```
q̂ = normalize(q[h]);  k̂ = normalize(k[h])            # L2 over key_dim
beta      = sigmoid(beta_proj[h])
decay[k]  = exp(gate_lower_bound * sigmoid(exp(a_log[h]) * (gate[h, k] + dt_bias[h, k])))

# sweep 1: decay, then predict
T[k, :]   = decay[k] * T[k, :]                        for each k
pred[:]   = sum_k k̂[k] * T[k, :]

err[:]    = beta * (v[h, :] - pred[:])

# sweep 2: update, then read out (state must be updated before it is read)
T[k, :]  += err[:] * k̂[k]                             for each k
out[:]    = sum_k q̂[k] * T[k, :]
o[h, :]   = output_scale * out[:]
```

The transposition is the load-bearing design decision. In the reference orientation (`state[v][k]`, with `key` contiguous) the per-key decay is strided and both contractions run within a row, needing `V_MUL_VV` + `V_RED_SUM` per row. Transposed, `key` becomes the row axis and all four steps reduce to one broadcast-scalar op per row.

### Instruction accounting — read this before setting any budget

The emitters do **not** emit one instruction per vector operation. `_emit_tile_row_fp_scalar` (`isa_tile_rows.py:371-409`) shows the real hardware-loop body:

```asm
C_LOOP_START gp_loop, row_count
S_LD_FP      f1, gp_fp, 0
<V-op>       gp_src, gp_src, f1, 0
S_ADDI_INT   gp_src, gp_src, row_step * mlen
S_ADDI_INT   gp_fp,  gp_fp,  fp_step
C_LOOP_END   gp_loop
```

So a per-row-scalar sweep costs **5 dynamic instructions per row**, of which 1 is arithmetic. A `V_FMA_VF` sweep needs a third pointer and costs **6 per row**. Two consequences:

**1. The win from `V_FMA_VF` is mostly static, not dynamic.** The `copy + multiply + add` per state row — what Mamba does today and what Phase 0's KDA will do — is three separate emitter calls with `rows=[1]`, so each falls to the *unrolled* path: roughly 12 fully-expanded instructions per state row, every one a distinct word in the program image. With FMA each sweep becomes one arithmetic progression and therefore one hardware loop:

| | static instr / layer | dynamic instr / state row |
|---|---:|---:|
| Mamba-2 today (unrolled triples) | ~98,000 | ~12 |
| Mamba-2 with `V_FMA_VF` (3 hardware loops per head) | **~1,900** | 17 |

A ~50× static reduction. That is what attacks the 23.663 MiB / 44.490 MiB program images directly, and it is the reason `V_FMA_VF` earns an opcode.

**2. Dynamic cost is dominated by address arithmetic, and no latency number in this project is calibrated.** Each row costs 5–6 emitted instructions of which 1 is arithmetic. The lever is post-increment addressing (out of scope above). A state coprocessor would *not* fix this — it has the same 64 lanes (`2 head_lanes × 32 fma_lanes` = 64, identical to `VLEN=64`); it only hides the addressing inside a private sequencer.

**Do not convert instruction counts into microseconds and then into a decision.** Every PLENA latency available anywhere in this project is an uncalibrated analytic model output. A µs figure of the form `bytes ÷ bytes-per-cycle` is a roofline the model produces by construction, not a measurement, and two such figures decide nothing between them.

**Two different kinds of number, and only one is admissible as a gate:**

| Quantity | Status | Use |
|---|---|---|
| **Static instruction count** | Hard fact — a property of the compiled artifact | **Gate on it.** Tasks 8 and 11. |
| Dynamic instruction count | Hard fact, given a fixed program and token count | Report it. |
| Cycles / µs / TPOT | Uncalibrated model output on both branches | **Report only, labelled uncalibrated. Never a gate, never a comparison against another uncalibrated model.** |

Task 3 records the Phase 0 static count, Task 8 Step 9 records the Phase 1 count, and **their ratio is the justification for spending opcode `0x3B`** — not the estimates in this section.

---

## File Structure

**Simulator (`PLENA_Simulator`, branch `feat/static-kda`)**

| Path | Responsibility |
|---|---|
| `transactional_emulator/src/op.rs` | Modify: decode `0x3B` into `Opcode::V_FMA_VF` |
| `transactional_emulator/src/vector_machine.rs` | Modify: add `fma_scalar` next to `mul_scalar` |
| `transactional_emulator/src/accelerator/dispatch.rs` | Modify: execute `V_FMA_VF`; add to the VRAM-read operand list |
| `analytic_models/reference/kimi_k3_kda.py` | Create: KDA golden, single oracle for both repos |
| `analytic_models/reference/test_kimi_k3_kda.py` | Create: reference self-test |
| `analytic_models/performance/vector_sram_banking.py` | Create (Task 14): bank model retargeted at Vector SRAM |
| `transactional_emulator/testbench/models/kda_static_test.py` | Create: end-to-end KDA layer, emulator vs golden |
| `transactional_emulator/testbench/models/whole_model_test.py` | Create (Task 12): first layer to last, both models |
| `transactional_emulator/testbench/models/checkpoint_test.py` | Create (Task 13): real published weights vs HuggingFace |
| `doc/checkpoint_validation.md` | Create (Task 13): exactly what was validated, and what was not |
| `doc/static_path_measurements.md` | Create (Task 14): hard facts and uncalibrated model output, kept apart |

**Compiler (`PLENA_Compiler`, branch `feat/static-kda`)**

| Path | Responsibility |
|---|---|
| `doc/operation.svh` | Modify: `V_FMA_VF = 6'h3B` |
| `doc/plena_isa_spec.md` | Modify: document `V_FMA_VF` |
| `assembler/assembly_to_binary.py` | Modify: encode `V_FMA_VF` |
| `aten/plena/isa_tile_rows.py` | Modify: `_emit_tile_row_fma` + `tile_row_fma_fp_asm` + `tile_row_fma_fp_broadcast` |
| `aten/plena/program_ssm_recurrent.py` | Modify: rewrite `ssm_decode_step_v0` onto FMA sweeps |
| `aten/plena/program_kda_common.py` | Create: `ProgramKdaCommonMixin` — state load/store, conv roll, L2 normalize |
| `aten/plena/program_kda_recurrent.py` | Create: the two-sweep KDA decode step |
| `aten/plena/program_kda_layer.py` | Create: full KDA decode layer |
| `aten/plena/program_kda_chunk.py` | Create (Task 9): chunk decay cumsum and the UT transform |
| `aten/plena/program_kda_prefill.py` | Create (Task 10): chunked prefill on the Matrix Engine |
| `aten/models/kda/reference.py` | Create: re-export of the Simulator's golden, incl. the single `KdaShape` |
| `aten/native_ops.yaml` | Modify: register `kda_decode` for CPU and PLENA backends |
| `aten/ops/cpu/kda_ops.py`, `aten/ops/plena/kda_ops.py` | Create: backend entry points |
| `aten/tests/test_v_fma_vf.py` | Create: encoding + emitter tests |
| `aten/tests/test_kda_recurrent.py` | Create: per-head and per-layer numeric tests |
| `aten/tests/test_instruction_budget.py` | Create: measured **static** instruction-count gates |
| `aten/tests/test_kda_prefill.py` | Create (Tasks 9–10): UT transform, prefill/decode equivalence |
| `aten/tests/test_whole_model_lowering.py` | Create (Task 11): 52- and 93-layer lowering |

**`KdaShape` is defined exactly once**, in `analytic_models/reference/kimi_k3_kda.py`, and re-exported by `aten/models/kda/reference.py`. `program_kda_common.py` imports it; it must not define its own.

---

## Phase 0 — KDA decode runs at all

**Zero new instructions.** Everything here is lowering work on top of what `feat/mamba2-support` already ships. Nothing in this phase touches `doc/operation.svh`, the decoder, or the assembler, so it carries no RTL implication and no ISA risk. At the end of Phase 0, KDA decode works; prefill is Phase 2.

---

### Task 1: KDA golden reference

Establishes the reference maths so the static lowering has an oracle independent of any descriptor machinery.

**Files:**
- Create: `analytic_models/reference/kimi_k3_kda.py`, `analytic_models/reference/test_kimi_k3_kda.py` (Simulator)
- Create: `aten/models/kda/__init__.py`, `aten/models/kda/reference.py` (Compiler)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `KdaShape(num_heads: int, key_dim: int, value_dim: int, conv_kernel: int, heads_per_group: int = 1)` with `state_rows -> num_heads * key_dim` and `head_base(head) -> head * key_dim`. **This is the only definition of `KdaShape` in either repo.**
  - `kda_decode_step(state, q, k, v, gate, beta_logit, a_log, dt_bias, gate_lower_bound, output_scale) -> tuple[Tensor, Tensor]` returning `(new_state, output)`, with `state` in the **reference** orientation `[heads, value_dim, key_dim]`.
  - `kda_layer_reference(weights, tokens, shape) -> Tensor`.

- [ ] **Step 1: Establish the reference**

```bash
# Write analytic_models/reference/kimi_k3_kda.py and its self-test: the KDA
# forward in torch, transposed-state formulation, no PLENA concepts.
```

- [ ] **Step 2: Run the reference self-test**

Run: `pytest analytic_models/reference/test_kimi_k3_kda.py -v`
Expected: PASS. If an import fails, bring that module across too — but nothing that imports from a runtime state engine.

- [ ] **Step 3: Add `KdaShape` if the reference module lacks it**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class KdaShape:
    """KDA head geometry.

    `state_rows` is expressed in the TRANSPOSED layout the PLENA lowering
    uses -- one row per (head, key) -- because that is the orientation in
    which every step of the recurrence is a single broadcast-scalar op. The
    reference functions in this module keep the [heads, value_dim, key_dim]
    orientation; the lowering transposes on load.
    """

    num_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    heads_per_group: int = 1

    @property
    def state_rows(self) -> int:
        return self.num_heads * self.key_dim

    def head_base(self, head: int) -> int:
        return head * self.key_dim
```

- [ ] **Step 4: Write the failing transposition-contract test**

Append to `analytic_models/reference/test_kimi_k3_kda.py`:

```python
def test_decode_step_matches_the_transposed_formulation():
    """T[k, v] == S[v, k]. This is the contract between the reference and
    aten/plena/program_kda_recurrent.py."""
    import torch

    torch.manual_seed(0)
    heads, key_dim, value_dim = 2, 8, 8
    state = torch.randn(heads, value_dim, key_dim, dtype=torch.float32)
    q = torch.randn(heads, key_dim); k = torch.randn(heads, key_dim)
    v = torch.randn(heads, value_dim)
    gate = torch.randn(heads, key_dim)
    beta_logit = torch.randn(heads)
    a_log = torch.randn(heads); dt_bias = torch.randn(heads, key_dim)

    new_state, out = kda_decode_step(
        state, q, k, v, gate, beta_logit, a_log, dt_bias,
        gate_lower_bound=-5.0, output_scale=1.0,
    )

    t = state.transpose(1, 2).clone()                       # [heads, key, value]
    qn = torch.nn.functional.normalize(q, dim=-1)
    kn = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.sigmoid(beta_logit)
    decay = torch.exp(-5.0 * torch.sigmoid(a_log[:, None].exp() * (gate + dt_bias)))
    t = t * decay[:, :, None]
    pred = (kn[:, :, None] * t).sum(dim=1)
    err = beta[:, None] * (v - pred)
    t = t + err[:, None, :] * kn[:, :, None]
    out_t = (qn[:, :, None] * t).sum(dim=1)

    torch.testing.assert_close(t.transpose(1, 2), new_state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_t, out, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 5: Run it**

Run: `pytest analytic_models/reference/test_kimi_k3_kda.py::test_decode_step_matches_the_transposed_formulation -v`
Expected: PASS.

If it fails, the reference orders decay, prediction, or normalisation differently than the replay above. **Fix the replay to match the reference**, then update this plan's Reference section to the corrected ordering — the lowering must follow the reference, not the other way round.

- [ ] **Step 6: Expose it to the Compiler**

Create `aten/models/kda/__init__.py` (empty) and `aten/models/kda/reference.py`:

```python
"""KDA golden reference.

The maths lives in the Simulator repo (analytic_models/reference/kimi_k3_kda.py)
so the compiler tests and the emulator tests share one oracle instead of two
drifting copies. This module only re-exports it.
"""

from analytic_models.reference.kimi_k3_kda import (  # noqa: F401
    KdaShape,
    kda_decode_step,
    kda_layer_reference,
)

__all__ = ["KdaShape", "kda_decode_step", "kda_layer_reference"]
```

If `kda_layer_reference` does not exist in the reference module, write it there now — a whole-layer forward taking the weight dict from Task 4 — and add a self-test for it before moving on.

- [ ] **Step 7: Commit (both repos)**

```bash
# Simulator
git add analytic_models/reference/
git commit -m "test(kda): add the KDA golden reference and pin the transposed formulation"
# Compiler
git add aten/models/kda/
git commit -m "test(kda): re-export the KDA golden reference for compiler tests"
```

---

---

### Task 2: KDA state layout and data movement

**Files:**
- Create: `aten/plena/program_kda_common.py`
- Create: `aten/tests/test_kda_recurrent.py`

**Interfaces:**
- Consumes: `KdaShape` from `aten.models.kda.reference` (Task 1); `pin_hbm_region`, `pinned_hbm_region`, `load_batch`, `store` from `ProgramSSMRecurrentMixin` (`program_ssm_recurrent.py:63-136`); `mamba_row_copy`, `mamba_block_copy` (`program_mamba_common.py:325-342`); `tile_row_mul`, `tile_row_sum` (`isa_tile_rows.py:188, 52`); `mamba_rsqrt_fpram` (`program_mamba_common.py:568`).
- Produces `ProgramKdaCommonMixin` with:
  - `kda_load_state_v0(name: str, shape: KdaShape, hbm_addr: int) -> VRAMMatrixVar` — `[state_rows, value_dim]`, FP32
  - `kda_store_state_v0(state, name: str, hbm_addr: int) -> InputVar`
  - `kda_conv_state_roll_v0(conv_state, new_row_src, new_row_idx, shape)`
  - `kda_l2_normalize_v0(vec, row: int, sq_scratch, sq_row: int, acc_fp, consts)`

- [ ] **Step 1: Write the failing state tests**

Create `aten/tests/test_kda_recurrent.py`:

```python
"""KDA static lowering: state layout, decode step, and full layer."""

import torch

from aten.models.kda.reference import KdaShape


def test_state_is_loaded_in_the_transposed_layout(kda_program):
    shape = KdaShape(num_heads=2, key_dim=4, value_dim=kda_program.mlen, conv_kernel=4)
    addr = kda_program.pin_hbm_region(
        "kda_state", shape.state_rows * shape.value_dim, hbm_element_bytes=4
    )
    state = kda_program.kda_load_state_v0("kda_state", shape, addr)
    assert state.shape == (shape.num_heads * shape.key_dim, shape.value_dim)


def test_state_store_mirrors_the_load_precision(kda_program):
    """A load/store precision mismatch is a wrong answer, not an error."""
    shape = KdaShape(num_heads=1, key_dim=4, value_dim=kda_program.mlen, conv_kernel=4)
    addr = kda_program.pin_hbm_region(
        "s", shape.state_rows * shape.value_dim, hbm_element_bytes=4
    )
    state = kda_program.kda_load_state_v0("s", shape, addr)
    stored = kda_program.kda_store_state_v0(state, "s", addr)
    assert stored.hbm_element_bytes == 4
    assert stored.precision == 1


def test_conv_roll_shifts_history_and_appends(kda_program):
    shape = KdaShape(num_heads=1, key_dim=4, value_dim=kda_program.mlen, conv_kernel=4)
    conv = kda_program.alloc_vram("conv", (shape.conv_kernel - 1, shape.key_dim))
    new = kda_program.alloc_vram("new", (1, shape.key_dim))
    asm = kda_program.capture_asm(
        lambda: kda_program.kda_conv_state_roll_v0(conv, new, 0, shape)
    )
    body = [l for l in asm.splitlines() if l.strip() and not l.strip().startswith(";")]
    assert body, "roll emitted nothing"
```

Build the `kda_program` fixture, and `alloc_vram` / `capture_asm`, from whatever `test_mamba2_reference.py` already uses — reuse its fixture wholesale and mix in `ProgramKdaCommonMixin`. Do not invent a new program builder.

- [ ] **Step 2: Run and confirm it fails**

Run: `pytest aten/tests/test_kda_recurrent.py -v`
Expected: FAIL — no module `aten.plena.program_kda_common`.

- [ ] **Step 3: Implement the module**

Create `aten/plena/program_kda_common.py`:

```python
"""KDA (Kimi Delta Attention) state layout and data movement.

State layout
------------
Per head the recurrent state is stored ``[key_dim, value_dim]`` -- the
*transpose* of the reference's ``[value_dim, key_dim]``. This is deliberate.
In the reference orientation the per-key decay is strided and both
contractions run within a row, so each needs ``V_MUL_VV`` + ``V_RED_SUM``.
Transposed, ``key`` becomes the row axis and all four steps of the recurrence
reduce to one broadcast-scalar operation per row:

    decay     T[k, :] *= decay[k]
    predict   pred[:] += k_hat[k] * T[k, :]
    update    T[k, :] += err[:] * k_hat[k]
    read out  out[:]  += q_hat[k] * T[k, :]

Each is then a single arithmetic row progression over the state tile, which is
what lets Phase 1 turn every one of them into a hardware loop once V_FMA_VF
removes the scratch row. Phase 0 still pays copy + multiply + add.

Precision
---------
FP32, not MX8. The argument in ``ssm_load_state_v0``'s docstring applies
unchanged: the state is a multiplicative accumulator carried across the whole
sequence, so quantisation error is amplified by ``1 / sqrt(1 - lambda^2)``.
KDA's decay is per-key and driven toward 1 by ``gate_lower_bound``, which is
exactly the long-memory regime where the amplification is worst. The profiled
Kimi runtime also uses FP32 recurrent state with BF16 convolution state, and
this lowering matches that split.
"""

from __future__ import annotations

from aten.models.kda.reference import KdaShape

__all__ = ["KdaShape", "ProgramKdaCommonMixin"]


class ProgramKdaCommonMixin:
    def kda_load_state_v0(self, name: str, shape: KdaShape, hbm_addr: int):
        """Prefetch the pinned FP32 state tile into VRAM, transposed."""
        self.emit_comment(
            f"stage=kda_state_load {name} [{shape.state_rows},{shape.value_dim}]"
        )
        var = self.input(
            name,
            (shape.state_rows, shape.value_dim),
            hbm_addr=hbm_addr,
            real_data_ratio=1.0,
        )
        return self.load_batch(
            var, name=f"{name}_vram", storage_precision=3, precision=1
        )

    def kda_store_state_v0(self, state, name: str, hbm_addr: int):
        """Write the updated state back over the same pinned HBM range.

        Must mirror kda_load_state_v0 exactly -- same precision class, same
        bytes per element -- or the state read back is not the state written,
        and the failure mode is a wrong answer rather than an error.
        """
        self.emit_comment(f"stage=kda_state_store {name}")
        return self.store(
            state,
            name=name,
            hbm_addr=hbm_addr,
            precision=1,
            hbm_element_bytes=4,
            real_data_ratio=1.0,
        )

    def kda_conv_state_roll_v0(self, conv_state, new_row_src, new_row_idx, shape: KdaShape):
        """Shift the q/k/v conv1d history by one timestep and append the new one.

        Structurally identical to ssm_conv_state_roll_v0: a physical copy,
        because address immediates are baked at ASM-gen time and a runtime
        ring pointer is not expressible.
        """
        history = shape.conv_kernel - 1
        if history <= 0:
            return conv_state
        if conv_state.shape[0] < history:
            raise ValueError(f"conv_state needs {history} rows, has {conv_state.shape[0]}")
        self.emit_comment(f"stage=kda_conv1d roll history={history}")
        for i in range(history - 1):
            self.mamba_row_copy(conv_state, i, conv_state, i + 1)
        self.mamba_row_copy(conv_state, history - 1, new_row_src, new_row_idx)
        return conv_state

    def kda_l2_normalize_v0(self, vec, row: int, sq_scratch, sq_row: int, acc_fp, consts):
        """In-place L2 normalisation of one VRAM row.

        Same sequence mamba_gated_rmsnorm_v0 uses for RMSNorm, with the mean
        scale set to 1: square, V_RED_SUM into FPRAM, then mamba_rsqrt_fpram,
        which computes 1/sqrt(acc * reci_group + eps). `consts.reci_group`
        must be 1.0 here -- L2 is sqrt(sum), not sqrt(mean).
        """
        self.emit_comment(f"stage=kda_normalize row={row}")
        self.mamba_row_copy(sq_scratch, sq_row, vec, row)
        self.tile_row_mul(sq_scratch, vec, rows=[sq_row])
        self.tile_row_sum(acc_fp, sq_scratch, rows=[sq_row], target_base_offset=0)
        self.mamba_rsqrt_fpram(acc_fp, consts, count=1)
        self.tile_row_mul_fp_broadcast(vec, acc_fp, rows=[row], fpram_offset=0)
        return vec
```

Check `tile_row_mul`'s and `tile_row_sum`'s real signatures at `isa_tile_rows.py:188` and `:52` and match them; the calls above follow `mamba_gated_rmsnorm_v0` (`program_mamba_common.py:543-556`).

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest aten/tests/test_kda_recurrent.py -v`
Expected: PASS, three tests.

- [ ] **Step 5: Commit**

```bash
git add aten/plena/program_kda_common.py aten/tests/test_kda_recurrent.py
git commit -m "feat(kda): transposed FP32 state layout, conv roll, and L2 normalize"
```

---

---

### Task 3: The KDA decode step (existing instructions only)

**Files:**
- Create: `aten/plena/program_kda_recurrent.py`
- Modify: `aten/tests/test_kda_recurrent.py`

**Interfaces:**
- Consumes: `KdaShape`, `ProgramKdaCommonMixin` (Task 2); `tile_row_mul_fp_broadcast` (`isa_tile_rows.py:258`); `tile_row_sub` (`isa_tile_rows.py:167`); `vram_fill_zero` (`isa_tile_rows.py:272`); `mamba_row_copy`, `mamba_row_add` (`program_mamba_common.py:325-337`).
- Produces: `ProgramKdaRecurrentMixin.kda_decode_step_v0(*, state, scratch, q_fp, k_fp, decay_fp, beta_fp, v, o, pred, err, shape, output_scale_fp, head_rows=None) -> VRAMMatrixVar` returning `o`.

**Existing instructions only.** This task uses the same `copy + multiply + add` triple that `ssm_decode_step_v0` uses today. It is deliberately the un-optimised form: Phase 0 exists to prove that the static ISA expresses KDA at all, and to produce the baseline that Phase 1's `V_FMA_VF` is measured against. Task 8 converts both kernels together.

  `q_fp` / `k_fp` / `decay_fp` are FPRAM arrays holding `q̂[h, k]`, `k̂[h, k]`, `decay[h, k]`, `k`-major within each head. `beta_fp` holds `beta[h]`. `v`, `o`, `pred`, `err` are `[num_heads, value_dim]` VRAM tiles.

**Column blocking.** `value_dim` may exceed `VLEN`. One emitter row is `self.mlen` wide, so a state row spans `blocks = value_dim // mlen` emitter rows. Sweeping all of them in one walk would need the FPRAM cursor to advance every `blocks` rows, which is not an arithmetic progression. Instead emit **one hardware loop per column block**, with the row walk stepping by `blocks` and the FPRAM cursor stepping by 1. `value_dim % mlen == 0` is required and asserted.

- [ ] **Step 1: Write the failing tests**

Append to `aten/tests/test_kda_recurrent.py`:

```python
def test_single_head_decode_matches_the_golden(kda_program, run_on_emulator):
    from aten.models.kda.reference import kda_decode_step

    torch.manual_seed(7)
    # value_dim MUST be a multiple of the vector width -- the lowering asserts
    # it, because a partial row would leave the lanes past value_dim polluting
    # the accumulation. Derive it rather than hardcoding.
    shape = KdaShape(
        num_heads=1, key_dim=8, value_dim=kda_program.mlen, conv_kernel=4
    )

    state = torch.randn(1, shape.value_dim, shape.key_dim, dtype=torch.float32)
    q = torch.randn(1, shape.key_dim); k = torch.randn(1, shape.key_dim)
    v = torch.randn(1, shape.value_dim)
    gate = torch.randn(1, shape.key_dim); beta_logit = torch.randn(1)
    a_log = torch.randn(1); dt_bias = torch.randn(1, shape.key_dim)

    expected_state, expected_out = kda_decode_step(
        state, q, k, v, gate, beta_logit, a_log, dt_bias,
        gate_lower_bound=-5.0, output_scale=1.0,
    )

    actual_state, actual_out = run_on_emulator(
        kda_program, shape,
        state=state.transpose(1, 2).reshape(shape.state_rows, shape.value_dim),
        q=q, k=k, v=v, gate=gate, beta_logit=beta_logit,
        a_log=a_log, dt_bias=dt_bias,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        actual_state.reshape(1, shape.key_dim, shape.value_dim).transpose(1, 2),
        expected_state, rtol=2e-3, atol=2e-3,
    )


def test_decode_uses_only_instructions_that_already_exist(kda_program):
    """Phase 0 introduces no new opcode. This test is the gate on that."""
    shape = KdaShape(
        num_heads=1, key_dim=8, value_dim=2 * kda_program.mlen, conv_kernel=4
    )
    asm = kda_program.capture_asm(lambda: _emit_decode(kda_program, shape))
    body = [l for l in asm.splitlines() if l.strip() and not l.strip().startswith(";")]
    assert body, "decode emitted nothing"
    assert not any("V_FMA_VF" in l for l in body), (
        "Phase 0 must not depend on V_FMA_VF; that arrives in Task 7"
    )
    mnemonics = {l.split()[0] for l in body}
    allowed = {
        "S_ADDI_INT", "S_LD_FP", "V_MUL_VF", "V_MUL_VV", "V_ADD_VV", "V_SUB_VV",
        "C_LOOP_START", "C_LOOP_END", "S_MAP_FP_V", "S_MAP_V_FP",
    }
    assert mnemonics <= allowed, f"unexpected mnemonics: {mnemonics - allowed}"


def test_decode_records_its_phase0_instruction_count(kda_program):
    """Not a budget -- a baseline. Task 8 must beat this number, and the
    amount by which it does is the measured case for spending an opcode."""
    shape = KdaShape(
        num_heads=1, key_dim=8, value_dim=kda_program.mlen, conv_kernel=4
    )
    asm = kda_program.capture_asm(lambda: _emit_decode(kda_program, shape))
    body = [l for l in asm.splitlines() if l.strip() and not l.strip().startswith(";")]
    print(f"PHASE0_KDA_DECODE_STATIC_INSTR_PER_HEAD={len(body)}")
    assert len(body) > 0


def test_multi_head_decode_matches_the_golden(kda_program, run_on_emulator):
    """Four heads: catches a per-head FPRAM offset bug that one head hides."""
    from aten.models.kda.reference import kda_decode_step

    torch.manual_seed(9)
    shape = KdaShape(
        num_heads=4, key_dim=8, value_dim=kda_program.mlen, conv_kernel=4
    )
    state = torch.randn(4, shape.value_dim, shape.key_dim, dtype=torch.float32)
    q = torch.randn(4, shape.key_dim); k = torch.randn(4, shape.key_dim)
    v = torch.randn(4, shape.value_dim); gate = torch.randn(4, shape.key_dim)
    beta_logit = torch.randn(4); a_log = torch.randn(4)
    dt_bias = torch.randn(4, shape.key_dim)

    _, expected_out = kda_decode_step(
        state, q, k, v, gate, beta_logit, a_log, dt_bias,
        gate_lower_bound=-5.0, output_scale=1.0,
    )
    _, actual_out = run_on_emulator(
        kda_program, shape,
        state=state.transpose(1, 2).reshape(shape.state_rows, shape.value_dim),
        q=q, k=k, v=v, gate=gate, beta_logit=beta_logit,
        a_log=a_log, dt_bias=dt_bias,
    )
    torch.testing.assert_close(actual_out, expected_out, rtol=2e-3, atol=2e-3)
```

Add `_emit_decode(program, shape)` as a small local helper in the test file that allocates the VRAM and FPRAM operands and calls `kda_decode_step_v0`. Build `run_on_emulator` from the harness `test_mamba2_reference.py` already uses; do not write a new emulator driver.

- [ ] **Step 2: Run and confirm it fails**

Run: `pytest aten/tests/test_kda_recurrent.py -k decode -v`
Expected: FAIL — no module `aten.plena.program_kda_recurrent`.

- [ ] **Step 3: Implement the decode step**

Create `aten/plena/program_kda_recurrent.py`:

```python
"""The KDA recurrent decode step: two sweeps over the transposed state tile.

Sweep 1 applies the per-key decay and accumulates the prediction. The error is
then a value-length vector. Sweep 2 applies the rank-1 update and reads the
output out of the *updated* state -- the reference reads the updated value
inside the same loop (``reduced += updated * q``), so update and read-out
cannot be reordered into separate passes over the state.

State is read twice and written once. That is inherent to the delta rule, not
an artefact of this lowering.

Phase 0 form: this module uses only instructions that already exist, with the
same copy + multiply + add triple ssm_decode_step_v0 uses today. Task 8
converts it -- and the Mamba kernel -- onto V_FMA_VF together, so the win is a
measured before/after on two real programs rather than an estimate.
"""

from __future__ import annotations

from typing import Sequence

from aten.models.kda.reference import KdaShape

__all__ = ["ProgramKdaRecurrentMixin"]


class ProgramKdaRecurrentMixin:
    def kda_decode_step_v0(
        self,
        *,
        state,
        scratch,
        q_fp,
        k_fp,
        decay_fp,
        beta_fp,
        v,
        o,
        pred,
        err,
        shape: KdaShape,
        output_scale_fp,
        head_rows: Sequence[int] | None = None,
    ):
        if shape.value_dim % self.mlen:
            raise ValueError(
                f"value_dim {shape.value_dim} must be a multiple of "
                f"the vector width {self.mlen}"
            )
        blocks = shape.value_dim // self.mlen
        heads = list(range(shape.num_heads)) if head_rows is None else list(head_rows)
        k_dim = shape.key_dim
        self.emit_comment(
            f"stage=kda_state_update decode heads={len(heads)} key_dim={k_dim}"
        )

        for h in heads:
            base = shape.head_base(h)
            fp_base = h * k_dim
            # One emitter row per (key, column block). Column block `c` of key
            # `j` lives at row (base + j) * blocks + c, so a walk over keys at
            # fixed `c` steps by `blocks` -- an arithmetic progression, which
            # is what keeps each sweep a single hardware loop.
            def key_rows(c: int) -> list[int]:
                return [(base + j) * blocks + c for j in range(k_dim)]

            for c in range(blocks):
                rows = key_rows(c)
                acc = h * blocks + c

                # --- sweep 1: decay, then predict ------------------------
                # decay is already a single broadcast over an arithmetic row
                # progression, so this one is a hardware loop even in Phase 0.
                self.tile_row_mul_fp_broadcast(
                    state, decay_fp, rows=rows, fpram_offset=fp_base
                )
                # predict: copy + multiply + add per key, via a scratch row.
                # The scratch breaks the row progression, so this unrolls --
                # that is exactly the cost Task 7 removes.
                self.vram_fill_zero(pred, rows=[acc])
                for i, r in enumerate(rows):
                    self.mamba_row_copy(scratch, 0, state, r)
                    self.tile_row_mul_fp_broadcast(
                        scratch, k_fp, rows=[0], fpram_offset=fp_base + i
                    )
                    self.mamba_row_add(pred, acc, scratch, 0)

            # --- error: err[h, :] = beta_h * (v[h, :] - pred[h, :]) -------
            self.emit_comment(f"stage=kda_error head={h}")
            err_rows = [h * blocks + c for c in range(blocks)]
            for r in err_rows:
                self.mamba_row_copy(err, r, v, r)
            self.tile_row_sub(err, pred, rows=err_rows)
            self.tile_row_mul_fp_broadcast(err, beta_fp, rows=err_rows, fpram_offset=h)

            for c in range(blocks):
                rows = key_rows(c)
                acc = h * blocks + c

                # --- sweep 2: update, then read out -----------------------
                # Every key's update touches only that key's row, so updating
                # all rows and then reducing over all rows is identical to the
                # reference's interleaved `updated = ...; reduced += updated*q`.
                for i, r in enumerate(rows):
                    self.mamba_row_copy(scratch, 0, err, acc)
                    self.tile_row_mul_fp_broadcast(
                        scratch, k_fp, rows=[0], fpram_offset=fp_base + i
                    )
                    self.mamba_row_add(state, r, scratch, 0)

                self.vram_fill_zero(o, rows=[acc])
                for i, r in enumerate(rows):
                    self.mamba_row_copy(scratch, 0, state, r)
                    self.tile_row_mul_fp_broadcast(
                        scratch, q_fp, rows=[0], fpram_offset=fp_base + i
                    )
                    self.mamba_row_add(o, acc, scratch, 0)

                self.tile_row_mul_fp_broadcast(
                    o, output_scale_fp, rows=[acc], fpram_offset=0
                )

        return o
```

Confirm `tile_row_sub`'s signature at `isa_tile_rows.py:167` — if it takes `(dst, src, rows=...)` the call above is correct; if it takes row offsets like `vram_add`, adapt.

- [ ] **Step 4: Run the single-head numeric test**

Run: `pytest aten/tests/test_kda_recurrent.py -k single_head -v`
Expected: PASS.

Diagnosis if it fails: output right but state wrong → the update sweep ran after read-out. Both off by a constant → `output_scale` applied twice, or normalisation skipped. Output right for key 0 only → the FPRAM cursor is not advancing (check `fpram_offset` and Task 3 Step 4).

- [ ] **Step 5: Run the loop-structure and multi-head tests**

Run: `pytest aten/tests/test_kda_recurrent.py -k "hardware_loops or multi_head" -v`
Expected: PASS, both.

- [ ] **Step 6: Commit**

```bash
git add aten/plena/program_kda_recurrent.py aten/tests/test_kda_recurrent.py
git commit -m "feat(kda): two-sweep recurrent decode step on FMA hardware loops"
```

---

---

### Task 4: The full KDA layer

**Files:**
- Create: `aten/plena/program_kda_layer.py`
- Create: `aten/ops/cpu/kda_ops.py`, `aten/ops/plena/kda_ops.py`
- Modify: `aten/native_ops.yaml`
- Modify: `aten/tests/test_kda_recurrent.py`

**Interfaces:**
- Consumes: everything from Tasks 2 and 3; the projection and RMSNorm helpers in `program_mamba_common.py` and `program_tensors.py`; `kda_layer_reference` (Task 1).
- Produces: `ProgramKdaLayerMixin.kda_layer_v0(*, hidden, weights, shape, hbm) -> VRAMMatrixVar`, emitting the stage sequence `kda_qkv_projection` → `kda_decay_beta_projection` → `kda_conv1d` → `kda_normalize` → `kda_state_update` → `kda_output_gate` → `kda_output_gate_rmsnorm` → `kda_out_projection`.

- [ ] **Step 1: Write the failing full-layer tests**

Append to `aten/tests/test_kda_recurrent.py`:

```python
def _random_kda_weights(shape):
    """Non-degenerate weights.

    Every tensor is randn-derived on purpose. A connected
    tests set W_kda_decay_a, W_kda_decay_b and W_kda_beta to zeros, which made
    decay and the delta-rule subtraction constants -- so the two paths that
    distinguish KDA from a plain linear attention were never exercised.
    """
    d = shape.num_heads * shape.value_dim
    return {
        "W_kda_q": torch.randn(d, shape.num_heads * shape.key_dim) * 0.05,
        "W_kda_k": torch.randn(d, shape.num_heads * shape.key_dim) * 0.05,
        "W_kda_v": torch.randn(d, d) * 0.05,
        "W_kda_gate": torch.randn(d, d) * 0.05,
        "W_kda_out": torch.randn(d, d) * 0.05,
        "W_kda_decay_a": torch.randn(d, shape.num_heads * shape.key_dim) * 0.05,
        "W_kda_decay_b": torch.randn(
            shape.num_heads * shape.key_dim, shape.num_heads * shape.key_dim
        ) * 0.05,
        "W_kda_beta": torch.randn(d, shape.num_heads) * 0.05,
        "W_kda_norm": torch.ones(1, shape.value_dim),
        "Q_CONV_WEIGHT": torch.randn(shape.num_heads * shape.key_dim, shape.conv_kernel) * 0.3,
        "K_CONV_WEIGHT": torch.randn(shape.num_heads * shape.key_dim, shape.conv_kernel) * 0.3,
        "V_CONV_WEIGHT": torch.randn(d, shape.conv_kernel) * 0.3,
        "A_LOG": torch.randn(shape.num_heads) * 0.5,
        "DT_BIAS": torch.randn(shape.num_heads, shape.key_dim) * 0.5,
    }


def test_weights_used_by_the_layer_test_are_not_degenerate():
    """Guard the failure mode a descriptor-driven state engine invites."""
    w = _random_kda_weights(KdaShape(4, 16, 64, 4))
    for key in ("W_kda_decay_a", "W_kda_decay_b", "W_kda_beta", "A_LOG", "DT_BIAS"):
        assert w[key].abs().max() > 1e-3, f"{key} is degenerate"


def test_full_layer_matches_the_golden_over_four_tokens(kda_program, run_layer_on_emulator):
    """Four sequential tokens: catches state-carry bugs one token hides."""
    from aten.models.kda.reference import kda_layer_reference

    torch.manual_seed(11)
    shape = KdaShape(num_heads=4, key_dim=16, value_dim=kda_program.mlen, conv_kernel=4)
    weights = _random_kda_weights(shape)
    tokens = torch.randn(4, shape.num_heads * shape.value_dim)

    expected = kda_layer_reference(weights, tokens, shape)
    actual = run_layer_on_emulator(kda_program, weights, tokens, shape)

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `pytest aten/tests/test_kda_recurrent.py -k full_layer -v`
Expected: FAIL — no module `aten.plena.program_kda_layer`.

- [ ] **Step 3: Implement the layer**

Create `aten/plena/program_kda_layer.py`, following the Mamba layer in `program_mamba_common.py` step for step: project, scatter into per-head rows, convolve with `kda_conv_state_roll_v0`, L2-normalise q and k with `kda_l2_normalize_v0`, move the per-head scalars into FPRAM with `ssm_decode_scalars_to_fpram_v0`, run `kda_decode_step_v0`, apply the sigmoid output gate, RMSNorm with `W_kda_norm`, project out.

Emit a `stage=` marker comment at each boundary using the same convention as `mamba_stage_marker`, so the emulator's stage profiler attributes the work.

- [ ] **Step 4: Register the op**

Add a `kda_decode` entry to `aten/native_ops.yaml` following the `mamba2` entries `feat/mamba2-support` added, and create `aten/ops/cpu/kda_ops.py` and `aten/ops/plena/kda_ops.py` mirroring `aten/ops/cpu/mamba_ops.py` and `aten/ops/plena/mamba_ops.py`.

- [ ] **Step 5: Run the full-layer test and confirm it passes**

Run: `pytest aten/tests/test_kda_recurrent.py -k "full_layer or degenerate" -v`
Expected: PASS, both.

- [ ] **Step 6: Run the whole compiler suite**

Run: `pytest`
Expected: PASS, no regressions in the Mamba tests.

- [ ] **Step 7: Commit**

```bash
git add aten/plena/program_kda_layer.py aten/ops/cpu/kda_ops.py aten/ops/plena/kda_ops.py aten/native_ops.yaml aten/tests/test_kda_recurrent.py
git commit -m "feat(kda): full KDA layer lowering, validated against the golden over four tokens"
```

---

---

## Phase 1 — One instruction, measured against a real baseline

Phase 0 gave us two working programs. Now add `V_FMA_VF` and convert both kernels onto it, so the instruction's value is a measured before/after rather than an estimate.

---

### Task 5: `V_FMA_VF` in the emulator

Adds the instruction to the reference implementation first, so the compiler has something to test against.

**Files:**
- Modify: `transactional_emulator/src/op.rs` (enum near line 105, decode near line 448)
- Modify: `transactional_emulator/src/vector_machine.rs` (after `mul_scalar`, which ends at line 137)
- Modify: `transactional_emulator/src/accelerator/dispatch.rs` (dispatch near line 263, operand list near line 870)
- Test: inline `#[cfg(test)]` in `vector_machine.rs` and `op.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `Opcode::V_FMA_VF { rd: u8, rs1: u8, rs2: u8, rmask: u8 }` decoded from opcode `0x3B`; `VectorMachine::fma_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32)` with semantics `vram[vd] = vram[vd] + vram[vs1] * f`.

- [ ] **Step 1: Write the failing decode test**

Add to the `#[cfg(test)] mod tests` block in `transactional_emulator/src/op.rs`:

```rust
#[test]
fn v_fma_vf_decodes_like_v_mul_vf() {
    // R-type layout: opcode[5:0], rd[9:6], rs1[13:10], rs2[17:14], rs3[21:18]
    let word: u32 = 0x3B | (3 << 6) | (4 << 10) | (2 << 14);
    match Opcode::decode(word) {
        Opcode::V_FMA_VF { rd, rs1, rs2, rmask } => {
            assert_eq!((rd, rs1, rs2, rmask), (3, 4, 2, 0));
        }
        other => panic!("expected V_FMA_VF, got {other:?}"),
    }
}
```

If the surrounding tests build words with a helper rather than by hand, use that helper instead and keep the same field values.

- [ ] **Step 2: Run it and confirm it fails**

Run: `cargo test --manifest-path transactional_emulator/Cargo.toml v_fma_vf_decodes_like_v_mul_vf`
Expected: FAIL — `V_FMA_VF` is not a variant of `Opcode`.

- [ ] **Step 3: Add the enum variant and the decode arm**

In `op.rs`, immediately after the `V_MUL_VF` variant:

```rust
    /// `Vector[rd] += Vector[rs1] * fp_reg<rs2>` — fused multiply-add, one
    /// rounding. Unlike every other V-type op, `rd` is read as well as written.
    V_FMA_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
    },
```

In the decode `match`, after the `0x3A` arm:

```rust
            0x3B => Self::V_FMA_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
```

- [ ] **Step 4: Run the decode test and confirm it passes**

Run: `cargo test --manifest-path transactional_emulator/Cargo.toml v_fma_vf_decodes_like_v_mul_vf`
Expected: PASS.

- [ ] **Step 5: Write the failing execution test**

Add to the `#[cfg(test)] mod tests` block in `vector_machine.rs`. Construct the machine and read/write rows using whatever the nearest existing `#[tokio::test]` in that file already does — do not invent helpers:

```rust
#[tokio::test]
async fn fma_scalar_accumulates_into_the_destination() {
    let vm = /* same construction as the neighbouring vector_machine test */;
    /* write [1.0, 2.0, 3.0, 4.0] at row 0 (destination) */
    /* write [10.0, 20.0, 30.0, 40.0] at row 1 (source)  */
    vm.fma_scalar(0, 1, 0.5, 0, u32::MAX).await;
    /* assert row 0 == [6.0, 12.0, 18.0, 24.0] */
    /* assert row 1 is unchanged == [10.0, 20.0, 30.0, 40.0] */
}
```

- [ ] **Step 6: Run it and confirm it fails**

Run: `cargo test --manifest-path transactional_emulator/Cargo.toml fma_scalar_accumulates`
Expected: FAIL — no method `fma_scalar`.

- [ ] **Step 7: Implement `fma_scalar`**

In `vector_machine.rs`, directly after `mul_scalar`. It mirrors `mul_scalar` exactly, except that it also reads `vd`:

```rust
    /// `Vector[vd] += Vector[vs1] * f`.
    ///
    /// Mirrors `mul_scalar` but reads the destination too. The accumulate is
    /// what lets one instruction carry a rank-1 state update or a state
    /// contraction that otherwise costs copy + multiply + add, and — because
    /// it removes the scratch row — lets a whole sweep become one arithmetic
    /// row progression, which the compiler turns into a hardware loop.
    pub(crate) async fn fma_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        let d = self.vram.read(vd).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(
                d.as_tensor() + a.as_tensor() * (f as f64),
                d.data_type(),
            );
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = d.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let addend = a.as_tensor().narrow(0, start, end - start) * (f as f64);
                    let updated = &sliced + &addend;
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, d.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        }
    }
```

- [ ] **Step 8: Run the execution test and confirm it passes**

Run: `cargo test --manifest-path transactional_emulator/Cargo.toml fma_scalar_accumulates`
Expected: PASS.

- [ ] **Step 9: Wire it into dispatch**

In `dispatch.rs`, after the `V_MUL_VF` arm:

```rust
                op::Opcode::V_FMA_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .fma_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
```

In the VRAM-read operand list near line 870, add:

```rust
        | op::Opcode::V_FMA_VF { rs1, .. }
```

- [ ] **Step 10: Confirm the whole suite and lints are clean**

Run: `cargo test --manifest-path transactional_emulator/Cargo.toml`
Expected: PASS, no new failures.

Run: `cargo clippy --manifest-path transactional_emulator/Cargo.toml --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 11: Commit**

```bash
git add transactional_emulator/src/op.rs transactional_emulator/src/vector_machine.rs transactional_emulator/src/accelerator/dispatch.rs
git commit -m "feat(isa): V_FMA_VF, a fused broadcast multiply-add on the vector machine"
```

---

---

### Task 6: `V_FMA_VF` in the ISA definition and assembler

**Files:**
- Modify: `doc/operation.svh` (`CUSTOM_ISA_OPCODE`, after `S_MAP_FP_V = 6'h3A`)
- Modify: `doc/plena_isa_spec.md` (after the `### V_MUL_VF` section, near line 290)
- Modify: `assembler/assembly_to_binary.py`
- Test: `aten/tests/test_v_fma_vf.py`

**Interfaces:**
- Consumes: `Opcode::V_FMA_VF` decoding `0x3B` (Task 5).
- Produces: assembly mnemonic `V_FMA_VF rd, rs1, fp2, rmask` encoding to a word whose low 6 bits are `0x3B`.

- [ ] **Step 1: Write the failing encoding test**

Create `aten/tests/test_v_fma_vf.py`:

```python
"""V_FMA_VF encodes into the same R-type slots as V_MUL_VF."""

import tempfile
from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assemble(text: str) -> list[int]:
    with tempfile.NamedTemporaryFile("w", suffix=".asm", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        converter = AssemblyToBinary(
            str(REPO_ROOT / "doc" / "operation.svh"),
            str(REPO_ROOT / "doc" / "configuration.svh"),
        )
        return [converter._convert_to_binary(i) for i in parse_asm_file(str(path))]
    finally:
        path.unlink(missing_ok=True)


def test_v_fma_vf_opcode_and_operand_slots():
    (word,) = _assemble("V_FMA_VF gp3, gp4, f2, 0\n")
    assert word & 0x3F == 0x3B
    assert (word >> 6) & 0xF == 3    # rd
    assert (word >> 10) & 0xF == 4   # rs1
    assert (word >> 14) & 0xF == 2   # fp2
    assert (word >> 18) & 0xF == 0   # rmask


def test_v_fma_vf_matches_v_mul_vf_layout():
    (fma,) = _assemble("V_FMA_VF gp5, gp6, f1, 0\n")
    (mul,) = _assemble("V_MUL_VF gp5, gp6, f1, 0\n")
    assert fma >> 6 == mul >> 6, "only the opcode field may differ"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest aten/tests/test_v_fma_vf.py -v`
Expected: FAIL — the assembler does not know `V_FMA_VF`.

- [ ] **Step 3: Add the opcode to `doc/operation.svh`**

In `CUSTOM_ISA_OPCODE`, put a comma after `S_MAP_FP_V` and append:

```systemverilog
    V_SOFTPLUS_V           = 6'h39,
    S_MAP_FP_V             = 6'h3A,
    V_FMA_VF               = 6'h3B   // Vector[rd] += Vector[rs1] * fp_reg<rs2>
```

- [ ] **Step 4: Teach the assembler the mnemonic**

In `assembler/assembly_to_binary.py`, find where `V_MUL_VF` is handled (the 4-operand `rd, rs1, fp2, rmask` form) and add `V_FMA_VF` to exactly that handling. Do not add a new code path — if `V_MUL_VF` appears in a name list, append `"V_FMA_VF"` to that list.

- [ ] **Step 5: Run the encoding tests and confirm they pass**

Run: `pytest aten/tests/test_v_fma_vf.py -v`
Expected: PASS, both tests.

- [ ] **Step 6: Document it in the ISA spec**

In `doc/plena_isa_spec.md`, immediately after `### V_MUL_VF`:

````markdown
### V_FMA_VF

**Format:** `V_FMA_VF rd, rs1, fp2, rmask`

**Operation:** `Vector[gp_reg<rd>] += Vector[gp_reg<rs1>] * fp_reg<fp2>`

**Description:**

Fused broadcast multiply-add. Multiplies a (VLEN, 1) vector by a scalar from
`fp_reg<fp2>` and accumulates into the destination vector, with a single
rounding.

Unlike every other V-type instruction, `rd` is a **source as well as a
destination**. The recurrent-state kernels are the motivating case: a rank-1
state update (`S[n,:] += B[n] * xs[:]`) and a state contraction
(`y[:] += C[n] * S[n,:]`) are otherwise three instructions each — copy,
multiply, add — plus a scratch row. Removing the scratch is what lets a whole
sweep be expressed as one arithmetic row progression, and therefore as one
hardware loop rather than an unrolled block. The same shape appears in
attention and FFN accumulation.

**Operands:**
- `rd`: Register containing the destination/accumulator Vector SRAM address
- `rs1`: Register containing the source Vector SRAM address
- `fp2`: FP register index holding the broadcast scalar
- `rmask`: `0` = all lanes; otherwise use `V_MASK_REG`

**Example:**
```asm
S_ADDI_INT gp3, gp0, 512           ; accumulator row
S_ADDI_INT gp4, gp0, 1024          ; source row
V_FMA_VF gp3, gp4, f2, 0           ; Vector[512..] += Vector[1024..] * f2
```
````

- [ ] **Step 7: Confirm the cross-repo opcode tables agree**

Run `grep -n "6'h3B" doc/operation.svh` in the Compiler and `grep -n "0x3B" transactional_emulator/src/op.rs` in the Simulator.
Expected: both name `V_FMA_VF`.

- [ ] **Step 8: Commit**

```bash
git add doc/operation.svh doc/plena_isa_spec.md assembler/assembly_to_binary.py aten/tests/test_v_fma_vf.py
git commit -m "feat(isa): declare and assemble V_FMA_VF at 0x3B"
```

---

---

### Task 7: The FMA row-sweep emitter

The whole plan's static-footprint claim rests on this task emitting a hardware loop.

**Files:**
- Modify: `aten/plena/isa_tile_rows.py` — add `_emit_tile_row_fma` next to `_emit_tile_row_fp_scalar` (line 371), and the two public wrappers next to `tile_row_mul_fp_asm` (line 513) and `tile_row_mul_fp_broadcast_asm` (line 534)
- Test: `aten/tests/test_v_fma_vf.py` (extend)

**Interfaces:**
- Consumes: the `V_FMA_VF` mnemonic (Task 6); the existing `self._reg`, `self._row_progression`, `self._emit`, `self.mlen`, `IsaBuilder`, `gp`, `fp`.
- Produces:
  - `tile_row_fma_fp_asm(self, dst_addr: int, src_addr: int, row_map: list[tuple[int, int, int]]) -> str` where each entry is `(dst_row, src_row, fpram_addr)`.
  - `tile_row_fma_fp_broadcast_asm(self, dst_addr: int, src_addr: int, fpram_base: int, dst_rows: list[int], src_rows: list[int]) -> str` — walks `fpram_base + i` for entry `i`.

  Semantics: `dst[dst_row] += src[src_row] * FPRAM[fpram_addr]`, entries applied in order.

- [ ] **Step 1: Write the failing emitter tests**

Append to `aten/tests/test_v_fma_vf.py`. `tile_program` should be built the same way the nearest existing test that calls `tile_row_mul_fp_broadcast_asm` builds its program object:

```python
def _body(asm: str) -> list[str]:
    return [l.strip() for l in asm.splitlines()
            if l.strip() and not l.strip().startswith(";")]


def test_fma_sweep_over_a_progression_emits_a_hardware_loop(tile_program):
    """A 128-row sweep must be a loop, not 128 unrolled blocks.

    This is the whole static-footprint argument: the copy/multiply/add
    predecessor needed a scratch row per step, which broke the progression and
    forced the unrolled path.
    """
    asm = tile_program.tile_row_fma_fp_broadcast_asm(
        dst_addr=0,
        src_addr=8192,
        fpram_base=16,
        dst_rows=list(range(128)),
        src_rows=[0] * 128,
    )
    body = _body(asm)
    assert sum("C_LOOP_START" in l for l in body) == 1
    assert sum("C_LOOP_END" in l for l in body) == 1
    assert sum("V_FMA_VF" in l for l in body) == 1, "loop body holds one FMA"
    assert len(body) < 12, f"expected a compact loop, got {len(body)} instructions"


def test_fma_sweep_holds_a_constant_destination(tile_program):
    """The contraction walks the source and pins the destination."""
    asm = tile_program.tile_row_fma_fp_broadcast_asm(
        dst_addr=0,
        src_addr=8192,
        fpram_base=16,
        dst_rows=[3] * 64,
        src_rows=list(range(64)),
    )
    body = _body(asm)
    assert sum("C_LOOP_START" in l for l in body) == 1
    assert sum("V_FMA_VF" in l for l in body) == 1


def test_fma_uses_no_scratch_and_no_separate_multiply(tile_program):
    asm = tile_program.tile_row_fma_fp_broadcast_asm(
        dst_addr=0, src_addr=8192, fpram_base=16,
        dst_rows=[0, 1, 2, 3], src_rows=[0, 1, 2, 3],
    )
    body = _body(asm)
    assert not any("V_ADD_VV" in l for l in body)
    assert not any("V_MUL_VF" in l for l in body)


def test_fma_rejects_mismatched_row_counts(tile_program):
    import pytest
    with pytest.raises(ValueError, match="row counts"):
        tile_program.tile_row_fma_fp_broadcast_asm(
            dst_addr=0, src_addr=8192, fpram_base=16,
            dst_rows=[0, 1], src_rows=[0],
        )
```

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest aten/tests/test_v_fma_vf.py -k fma -v`
Expected: FAIL — no attribute `tile_row_fma_fp_broadcast_asm`.

- [ ] **Step 3: Implement the emitter**

In `aten/plena/isa_tile_rows.py`, immediately after `_emit_tile_row_fp_scalar`. This mirrors that method exactly, with a third pointer and two independent row progressions:

```python
    def _emit_tile_row_fma(
        self,
        dst_addr: int,
        src_addr: int,
        row_map: list[tuple[int, int, int]],
    ) -> str:
        """`dst[d] += src[s] * FPRAM[f]` for each `(d, s, f)` in `row_map`.

        Four pointers, so four GP registers. When both row walks and the FPRAM
        walk are arithmetic progressions this collapses to one hardware loop --
        the case that matters, since every recurrent sweep either steps a state
        tile with a pinned accumulator or the reverse. A step of 0 is a valid
        progression, which is how the pinned side is expressed.
        """
        gp_regs = self._reg.allocate_gp(4)
        gp_dst, gp_src, gp_fp, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(
                f"Tile Row FMA: VRAM[{dst_addr}] += VRAM[{src_addr}] * FPRAM"
            )
            dst_rows = [d for d, _, _ in row_map]
            src_rows = [s for _, s, _ in row_map]
            fp_addrs = [f for _, _, f in row_map]

            dst_prog = self._row_progression(dst_rows)
            src_prog = self._row_progression(src_rows)
            fp_prog = self._row_progression(fp_addrs)

            if dst_prog is not None and src_prog is not None and fp_prog is not None:
                dst_start, count, dst_step = dst_prog
                src_start, _, src_step = src_prog
                fp_start, _, fp_step = fp_prog
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr + dst_start * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr + src_start * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_start)
                asm.instr("C_LOOP_START", gp(gp_loop), count)
                asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                asm.instr("V_FMA_VF", gp(gp_dst), gp(gp_src), fp(1), 0)
                if dst_step:
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), dst_step * self.mlen)
                if src_step:
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), src_step * self.mlen)
                if fp_step:
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(gp_fp), fp_step)
                asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for dst_row, src_row, fpram_addr in row_map:
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr + dst_row * self.mlen)
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr + src_row * self.mlen)
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fpram_addr)
                    asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                    asm.instr("V_FMA_VF", gp(gp_dst), gp(gp_src), fp(1), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def tile_row_fma_fp_asm(
        self,
        dst_addr: int,
        src_addr: int,
        row_map: list[tuple[int, int, int]],
    ) -> str:
        return self._emit_tile_row_fma(dst_addr, src_addr, row_map)

    def tile_row_fma_fp_broadcast_asm(
        self,
        dst_addr: int,
        src_addr: int,
        fpram_base: int,
        dst_rows: list[int],
        src_rows: list[int],
    ) -> str:
        """Walk one FPRAM slot per row pair, starting at `fpram_base`."""
        if len(dst_rows) != len(src_rows):
            raise ValueError(
                f"tile_row_fma row counts differ: "
                f"{len(dst_rows)} destinations, {len(src_rows)} sources"
            )
        row_map = [
            (d, s, fpram_base + i)
            for i, (d, s) in enumerate(zip(dst_rows, src_rows))
        ]
        return self._emit_tile_row_fma(dst_addr, src_addr, row_map)
```

Confirm `fp(1)` is the right scratch FP register by checking that `_emit_tile_row_fp_scalar` uses it; if it uses a different one, match that.

- [ ] **Step 4: Confirm `_row_progression` accepts a constant list**

Run:

```bash
python -c "
from aten.plena.isa_tile_rows import IsaTileRowsMixin as M
print(M._row_progression(None, [3]*8))
"
```
Expected: a `(start, count, step)` tuple with `step == 0`. If it returns `None` for a constant list, fix `_row_progression` to treat a constant run as a step-0 progression, and add a test for that — the pinned-accumulator sweeps depend on it. Adjust the import path / call convention to whatever the class actually is.

- [ ] **Step 5: Add the `VRAMMatrixVar`-level wrapper**

Next to `tile_row_mul_fp_broadcast` (line 258), following that method's signature style:

```python
    def tile_row_fma_fp_broadcast(
        self,
        dst,
        src,
        fp,
        *,
        dst_rows: list[int],
        src_rows: list[int],
        fpram_offset: int,
    ):
        """`dst[dst_rows[i]] += src[src_rows[i]] * FPRAM[fp + fpram_offset + i]`."""
        return self.tile_row_fma_fp_broadcast_asm(
            dst_addr=dst.address,
            src_addr=src.address,
            fpram_base=fp.address + fpram_offset,
            dst_rows=dst_rows,
            src_rows=src_rows,
        )
```

Match how `tile_row_mul_fp_broadcast` resolves `.address` on its arguments and follow it exactly.

- [ ] **Step 6: Run and confirm all four tests pass**

Run: `pytest aten/tests/test_v_fma_vf.py -v`
Expected: PASS, six tests total.

- [ ] **Step 7: Commit**

```bash
git add aten/plena/isa_tile_rows.py aten/tests/test_v_fma_vf.py
git commit -m "feat(lowering): FMA row sweeps that collapse to a hardware loop"
```

---

---

### Task 8: Collapse both decode steps onto FMA sweeps

Proves the emitter on a kernel that already has a passing numeric test.

**Files:**
- Modify: `aten/plena/program_ssm_recurrent.py:143-211` (`ssm_decode_step_v0`)
- Modify: `aten/plena/program_kda_recurrent.py` (`kda_decode_step_v0`, from Task 3)
- Create: `aten/tests/test_instruction_budget.py`
- Test: `aten/tests/test_mamba2_reference.py` and `aten/tests/test_kda_recurrent.py` (both existing, both must stay green)

**Interfaces:**
- Consumes: `tile_row_fma_fp_broadcast` (Task 7).
- Produces: `ssm_decode_step_v0` and `kda_decode_step_v0`, both with unchanged numerical output. `kda_decode_step_v0` keeps its signature; its `scratch` parameter becomes unused and is dropped in Step 8. Both emit hardware loops instead of unrolled triples.

Doing both kernels in one task is deliberate: they share the emitter, so a bug in it shows up twice, and the measured before/after covers both programs at once.

- [ ] **Step 1: Record the current static count as a baseline**

Create `aten/tests/test_instruction_budget.py`:

```python
"""Static instruction-count gates.

The only guard against a lowering change that is numerically correct but
quietly unrolls a sweep that used to be a hardware loop. Unrolling is what
makes a whole-model program image tens of megabytes.

Budgets are MEASURED, not derived: run the test, read the reported count,
set the constant ~10% above it, and note the date. Raising a budget requires
a line in docs/superpowers/plans/2026-08-25-static-mamba-kda.md saying why.
"""

# Measured 2026-08-25 after Task 8. Three hardware loops per head plus
# per-head scalar setup.
MAMBA_DECODE_STATIC_INSTR_PER_HEAD_MAX = 40


def test_mamba_decode_step_static_instruction_count(mamba_decode_program):
    per_head = (
        mamba_decode_program.instruction_count // mamba_decode_program.num_heads
    )
    assert per_head <= MAMBA_DECODE_STATIC_INSTR_PER_HEAD_MAX, (
        f"{per_head} static instructions per head exceeds "
        f"{MAMBA_DECODE_STATIC_INSTR_PER_HEAD_MAX}; a sweep probably fell off "
        f"the hardware-loop path"
    )
```

Build the `mamba_decode_program` fixture from whatever `aten/tests/test_mamba2_reference.py` already uses to construct a decode program.

- [ ] **Step 2: Run it and confirm it fails, and record the number**

Run: `pytest aten/tests/test_instruction_budget.py -v`
Expected: FAIL, reporting roughly 1,100–1,200 static instructions per head (the unrolled `9 * state_size` pattern). Write the observed number into a comment in the test file.

- [ ] **Step 3: Rewrite the update and contraction loops**

In `ssm_decode_step_v0`, replace the whole per-head body:

```python
        for h in heads:
            group = h // shape.heads_per_group
            base = h * n_state
            state_rows = list(range(base, base + n_state))

            # xs = dt_h * x[h] -- fold dt into x once per head so the per-row
            # scalars below are only B[g, n].
            self.mamba_row_copy(scratch, 0, x, h)
            self.tile_row_mul_fp_broadcast(scratch, dt_fp, rows=[0], fpram_offset=h)

            # decay: h[h, n, :] *= dA_h   (one hardware loop, fp step 0)
            self.tile_row_mul_fp_broadcast(state, da_fp, rows=state_rows, fpram_offset=h)

            # rank-1 update: h[h, n, :] += B[g, n] * xs
            # dst walks the state tile, src is pinned to scratch row 0.
            self.tile_row_fma_fp_broadcast(
                state, scratch, b_fp,
                dst_rows=state_rows,
                src_rows=[0] * n_state,
                fpram_offset=group * n_state,
            )

            # contraction: y[h] = sum_n C[g, n] * h[h, n, :]
            # dst is pinned to y row h, src walks the state tile.
            self.vram_fill_zero(y, rows=[h])
            self.tile_row_fma_fp_broadcast(
                y, state, c_fp,
                dst_rows=[h] * n_state,
                src_rows=state_rows,
                fpram_offset=group * n_state,
            )

            # skip: y[h] += D_h * x[h]
            self.emit_comment(mamba_stage_marker("mamba_skip", f"head={h}"))
            self.tile_row_fma_fp_broadcast(
                y, x, d_fp, dst_rows=[h], src_rows=[h], fpram_offset=h
            )

        return y
```

- [ ] **Step 4: Run the numeric test and confirm the answer is unchanged**

Run: `pytest aten/tests/test_mamba2_reference.py -v`
Expected: PASS, unchanged.

If it fails: the maths is identical, so the bug is ordering. Check that `b_fp` / `c_fp` are laid out `n`-major within each group (the docstring at `program_ssm_recurrent.py:157-165` states this), and that `vram_fill_zero(y, rows=[h])` still runs *before* the contraction.

- [ ] **Step 5: Run the budget test and confirm it passes**

Run: `pytest aten/tests/test_instruction_budget.py -v`
Expected: PASS. If the measured count is above 40 but the emitted asm does contain 3 `C_LOOP_START`s per head, raise the constant to measured + 10% and note it.

- [ ] **Step 6: Confirm the sweeps really are loops**

Run: `pytest aten/tests/test_mamba2_reference.py -v -k asm` or dump one head's assembly and check by eye:

```bash
grep -c "C_LOOP_START" <dumped-head.asm>
```
Expected: 3 per head. If it is 0, `_row_progression` rejected one of the row lists — most likely the pinned `[0] * n_state` (see Task 7 Step 4).

- [ ] **Step 7: Convert the KDA decode step**

In `aten/plena/program_kda_recurrent.py`, replace each of the three unrolled
`for i, r in enumerate(rows)` blocks with one FMA sweep. `dst` below is the
pinned accumulator row list, `rows` the state-tile walk:

```python
                # sweep 1: predict
                self.vram_fill_zero(pred, rows=[acc])
                self.tile_row_fma_fp_broadcast(
                    pred, state, k_fp,
                    dst_rows=[acc] * k_dim, src_rows=rows, fpram_offset=fp_base,
                )

                # sweep 2: update, then read out
                self.tile_row_fma_fp_broadcast(
                    state, err, k_fp,
                    dst_rows=rows, src_rows=[acc] * k_dim, fpram_offset=fp_base,
                )
                self.vram_fill_zero(o, rows=[acc])
                self.tile_row_fma_fp_broadcast(
                    o, state, q_fp,
                    dst_rows=[acc] * k_dim, src_rows=rows, fpram_offset=fp_base,
                )
```

The decay sweep is already a broadcast over a progression and does not change.

- [ ] **Step 8: Run the KDA tests and confirm the answer is unchanged**

Run: `pytest aten/tests/test_kda_recurrent.py -v`
Expected: PASS, unchanged numerically.

`test_decode_uses_only_instructions_that_already_exist` from Task 3 will now
fail, correctly — Phase 1 is where `V_FMA_VF` enters. Delete that test and add
`V_FMA_VF` to the allowed set in nothing else; the count baseline test from
Task 3 stays and should now report a much smaller number.

Also drop the now-unused `scratch` parameter from `kda_decode_step_v0` and from
its call sites in `program_kda_layer.py`.

- [ ] **Step 9: Record the measured Phase 0 → Phase 1 improvement**

Run: `pytest aten/tests/test_kda_recurrent.py -k records_its_phase0 -s`
Expected: prints the new per-head static count.

Write both numbers — the Task 3 baseline and this one — into a comment in
`aten/tests/test_instruction_budget.py`. **This ratio is the measured
justification for spending opcode `0x3B`.** If it is below about 3×, say so
plainly rather than proceeding on the assumption that the instruction earned
its place.

- [ ] **Step 10: Commit**

```bash
git add aten/plena/program_ssm_recurrent.py aten/plena/program_kda_recurrent.py aten/plena/program_kda_layer.py aten/tests/test_instruction_budget.py aten/tests/test_kda_recurrent.py
git commit -m "perf: fold both recurrent decode steps into FMA hardware loops"
```

---

---

## Phase 2 — KDA chunked prefill

Decode alone is half the function. Prefill for KDA is the half that had never been mapped onto the PLENA Matrix Engine at all.

This phase mirrors what `feat/mamba2-support` already did for Mamba-2 in `aten/plena/program_ssd.py`, whose module docstring (lines 1-45) is the model to follow: identify which chunk-level products drop onto the existing flash-attention Matrix templates unchanged, and isolate the one that does not.

**The structural difference from Mamba.** Mamba-2's chunked form is four matmuls, three of which map directly onto `M_MM` / `M_TMM`. The gated delta rule does not chunk that simply: within a chunk the update is a product of rank-1 projectors `∏(I − β_t k_t k_tᵀ)`, and collapsing it requires the WY / UT transform, whose core is the inverse of a `[chunk, chunk]` lower-triangular matrix. That is why `chunk = 16` matters — a 16×16 triangular solve by forward substitution is 16 rank-1 updates, which the vector unit does directly. Do not raise the chunk size without redoing this analysis.

---

### Task 9: KDA chunk primitives — decay cumsum and the UT transform

**Files:**
- Create: `aten/plena/program_kda_chunk.py`
- Test: `aten/tests/test_kda_prefill.py`

**Interfaces:**
- Consumes: `KdaShape` (Task 1); `ssd_lower_triangular_ones` and `ssd_chunk_cumsum_v0` patterns from `aten/plena/program_ssd.py:177-269`; `tile_row_fma_fp_broadcast` (Task 7).
- Produces `ProgramKdaChunkMixin` with:
  - `kda_chunk_decay_cumsum_v0(decay_log, cs_fp, *, chunk, shape) -> FPVar` — per-key cumulative log-decay within the chunk, into FPRAM.
  - `kda_ut_transform_v0(k, beta_fp, t_out, scratch, *, chunk, shape) -> VRAMMatrixVar` — builds `T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)`, `[chunk, chunk]`, by forward substitution.

- [ ] **Step 1: Write the failing UT-transform test against a dense reference**

Create `aten/tests/test_kda_prefill.py`:

```python
"""KDA chunked prefill: chunk primitives, then the full prefill layer."""

import torch

from aten.models.kda.reference import KdaShape


def _dense_ut(k: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """T = (I + tril(diag(beta) K K^T, -1))^-1 diag(beta), computed densely."""
    chunk = k.shape[0]
    a = torch.tril(torch.diag(beta) @ (k @ k.T), diagonal=-1)
    return torch.linalg.inv(torch.eye(chunk) + a) @ torch.diag(beta)


def test_ut_transform_matches_a_dense_inverse(kda_program, run_on_emulator):
    torch.manual_seed(3)
    chunk = 16
    shape = KdaShape(num_heads=1, key_dim=8, value_dim=kda_program.mlen, conv_kernel=4)
    k = torch.nn.functional.normalize(torch.randn(chunk, shape.key_dim), dim=-1)
    beta = torch.sigmoid(torch.randn(chunk))

    expected = _dense_ut(k, beta)
    actual = run_on_emulator(
        kda_program, "kda_ut_transform_v0", chunk=chunk, shape=shape, k=k, beta=beta
    )
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


def test_ut_transform_is_lower_triangular(kda_program, run_on_emulator):
    """Forward substitution must never write above the diagonal."""
    torch.manual_seed(4)
    chunk = 16
    shape = KdaShape(num_heads=1, key_dim=8, value_dim=kda_program.mlen, conv_kernel=4)
    k = torch.nn.functional.normalize(torch.randn(chunk, shape.key_dim), dim=-1)
    beta = torch.sigmoid(torch.randn(chunk))
    actual = run_on_emulator(
        kda_program, "kda_ut_transform_v0", chunk=chunk, shape=shape, k=k, beta=beta
    )
    assert torch.triu(actual, diagonal=1).abs().max() < 1e-6
```

- [ ] **Step 2: Run and confirm it fails**

Run: `pytest aten/tests/test_kda_prefill.py -v`
Expected: FAIL — no module `aten.plena.program_kda_chunk`.

- [ ] **Step 3: Implement the UT transform by forward substitution**

Create `aten/plena/program_kda_chunk.py`. The solve is row-by-row: row `i` of `T` is
`β_i·e_i − Σ_{j<i} a_ij · T[j, :]`, where `a_ij = β_i · ⟨k_i, k_j⟩`. Each subtraction
is one `V_FMA_VF` with a negated scalar, so row `i` costs `i` FMAs — 120 for
`chunk = 16`, one hardware loop per row.

```python
"""KDA chunk-level primitives for the prefill path.

The gated delta rule does not chunk the way Mamba-2's SSD does. Within a chunk
the state update is a product of rank-1 projectors,

    S_chunk = prod_t (I - beta_t k_t k_t^T)

which cannot be reassociated into a matmul directly. The WY / UT transform
collapses it into two dense factors,

    T = (I + tril(diag(beta) K K^T, -1))^-1 diag(beta)      [chunk, chunk]
    W = T K,  U = T V

after which the rest of the chunk is ordinary matmuls that land on the same
flash-attention Matrix templates program_ssd.py already reuses.

The inverse is the only hard part, and it is small on purpose: at chunk = 16 it
is a 16x16 unit-lower-triangular solve, done by forward substitution as 16 rank-1
row updates. Row i costs i fused multiply-adds -- 120 in total -- with one
hardware loop per row. Raising the chunk size makes this quadratic term grow and
requires redoing the analysis; chunk = 16 is an ABI constant, not a tuning knob.
"""
```

Implement `kda_chunk_decay_cumsum_v0` by following `ssd_chunk_cumsum_v0`
(`program_ssd.py:213-269`) — the structure is identical, with the per-head scalar
`a_t = A_h · dt_t` replaced by the per-key `log_decay[t, k]`.

- [ ] **Step 4: Run and confirm both tests pass**

Run: `pytest aten/tests/test_kda_prefill.py -v`
Expected: PASS.

If the triangularity test fails, forward substitution is writing `T[i, j]` for `j > i` — the inner loop bound is `range(i)`, not `range(chunk)`.

- [ ] **Step 5: Commit**

```bash
git add aten/plena/program_kda_chunk.py aten/tests/test_kda_prefill.py
git commit -m "feat(kda): chunk decay cumsum and the UT transform by forward substitution"
```

---

### Task 10: The KDA chunked prefill layer

**Files:**
- Create: `aten/plena/program_kda_prefill.py`
- Modify: `aten/tests/test_kda_prefill.py`, `analytic_models/reference/kimi_k3_kda.py` (add `kda_prefill_reference`)

**Interfaces:**
- Consumes: `ProgramKdaChunkMixin` (Task 9); `kda_decode_step_v0` (Tasks 3, 8) for the equivalence test; the Matrix helpers `ssd_transposed_projection_v0` and `ssd_chunk_head_v0` patterns (`program_ssd.py:321-436`).
- Produces: `ProgramKdaPrefillMixin.kda_prefill_v0(*, hidden, weights, shape, chunk, hbm) -> tuple[VRAMMatrixVar, VRAMMatrixVar]` returning `(output, final_state)`.

**Chunk size is an ABI constant.** `chunk = 16` for prefill; decode's physical row capacity is `BLEN = 4`. The two must not be conflated: giving decode prefill's row pitch creates 4–16× zero-padded compute and address holes.

- [ ] **Step 1: Write the failing equivalence test**

The strongest available correctness statement, and the one `program_ssd.py` uses for Mamba: **a chunked prefill over S tokens must produce the same final state and the same outputs as S sequential decode steps.**

```python
def test_prefill_equals_sequential_decode(kda_program, run_layer_on_emulator):
    """The chunked path and the recurrent path must agree exactly.

    This is the only test that can catch a UT-transform sign error, a decay
    cumsum off-by-one, or an intra/inter-chunk mixup -- each of which produces
    plausible-looking output on its own.
    """
    from aten.models.kda.reference import kda_layer_reference

    torch.manual_seed(13)
    shape = KdaShape(num_heads=2, key_dim=16, value_dim=kda_program.mlen, conv_kernel=4)
    weights = _random_kda_weights(shape)
    tokens = torch.randn(32, shape.num_heads * shape.value_dim)   # 2 chunks of 16

    seq_out, seq_state = kda_layer_reference(
        weights, tokens, shape, return_state=True
    )
    pre_out, pre_state = run_layer_on_emulator(
        kda_program, weights, tokens, shape, mode="prefill", chunk=16
    )

    torch.testing.assert_close(pre_out, seq_out, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(pre_state, seq_state, rtol=5e-3, atol=5e-3)


def test_prefill_then_decode_carries_state(kda_program, run_layer_on_emulator):
    """Prefill 32 tokens, then decode 4 more, against 36 sequential steps."""
    from aten.models.kda.reference import kda_layer_reference

    torch.manual_seed(17)
    shape = KdaShape(num_heads=2, key_dim=16, value_dim=kda_program.mlen, conv_kernel=4)
    weights = _random_kda_weights(shape)
    tokens = torch.randn(36, shape.num_heads * shape.value_dim)

    expected, _ = kda_layer_reference(weights, tokens, shape, return_state=True)
    actual = run_layer_on_emulator(
        kda_program, weights, tokens, shape,
        mode="prefill_then_decode", chunk=16, prefill_len=32,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
```

Import `_random_kda_weights` from `aten/tests/test_kda_recurrent.py` rather than
duplicating it; extend `kda_layer_reference` with `return_state` and a
`kda_prefill_reference` in the Simulator's reference module first, with its own
self-test, exactly as Task 1 Step 6 required.

- [ ] **Step 2: Run and confirm it fails**

Run: `pytest aten/tests/test_kda_prefill.py -k prefill_equals -v`
Expected: FAIL — no module `aten.plena.program_kda_prefill`.

- [ ] **Step 3: Implement the prefill layer**

Create `aten/plena/program_kda_prefill.py`. Per chunk, per head:

1. `A = tril(diag(β) K Kᵀ, −1)` — `M_TMM`, the `qkt.py` template with `key_dim` substituted for `head_dim`.
2. `T` — `kda_ut_transform_v0` (Task 9).
3. `W = T K`, `U = T V` — two `M_MM`.
4. Intra-chunk output: `Y_intra = tril(Q Kᵀ ⊙ decay_mask) U` — `M_TMM` then `M_MM`, mirroring `ssd_chunk_head_v0`.
5. Inter-chunk output: `Y_inter = Q S_prev`, rows scaled by the decay cumsum — mirrors `ssd_inter_chunk_output_v0`.
6. State update: `S_new = decay_chunk ⊙ S_prev + Wᵀ (U − W S_prev)`.

Step 6 contracts over the **row** axis of both operands, which neither `M_MM` nor
`M_TMM` expresses — the identical obstacle `program_ssd.py:38-45` documents for
Mamba's `h[n,p] = Σ_t B[t,n] X[t,p]`. Use the same fix: stage the transposed
operand at its source via the `ssd_transposed_projection_v0` pattern
(`program_ssd.py:321-366`). Do not invent a transpose instruction.

- [ ] **Step 4: Run the equivalence test**

Run: `pytest aten/tests/test_kda_prefill.py -k prefill_equals -v`
Expected: PASS.

Diagnosis: outputs match but final state does not → step 6's transposed staging. First chunk matches, second does not → the inter-chunk decay scaling in step 5. Everything off by a per-row factor → the decay cumsum is exclusive where it should be inclusive.

- [ ] **Step 5: Run the prefill-then-decode test**

Run: `pytest aten/tests/test_kda_prefill.py -k then_decode -v`
Expected: PASS. This is the gate on prefill and decode agreeing on the state layout.

- [ ] **Step 6: Confirm the full suite**

Run: `pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aten/plena/program_kda_prefill.py aten/tests/test_kda_prefill.py
git commit -m "feat(kda): chunked prefill on the Matrix Engine, equivalent to sequential decode"
```

---

## Phase 3 — Whole model, executed, on real weights

This is the phase that closes the project's two largest credibility gaps: whole-model artifacts that are never bound to a real checkpoint, and never executed first layer to last.

Generating machine code is not the same as running it. Until Task 12 passes, "PLENA runs Mamba-2 and KDA" is not a statement anyone should make.

---

### Task 11: Whole-model lowering and static instruction gates

**Files:**
- Create: `aten/tests/test_instruction_budget.py` additions
- Create: `aten/tests/test_whole_model_lowering.py`
- Modify: `.github/workflows/` (both repos)

**Interfaces:**
- Consumes: `kda_layer_v0` (Task 4), `kda_prefill_v0` (Task 10), `ssm_decode_step_v0` (Task 8).
- Produces: full Nemotron-3 (52 layers) and Kimi-K3 (93 layers) lowered programs, plus CI gates on their static instruction counts.

- [ ] **Step 1: Write the failing whole-model lowering test**

```python
"""Whole-model lowering. Generating the program is the precondition for
Task 12 executing it; passing this test alone proves nothing about numerics."""

def test_nemotron3_52_layers_lower_and_assemble(nemotron_model_program):
    words = nemotron_model_program.assemble()
    assert len(words) > 0
    assert all(0 <= w <= 0xFFFF_FFFF for w in words)
    print(f"NEMOTRON_STATIC_INSTR={len(words)} BYTES={len(words) * 4}")


def test_kimi_k3_93_layers_lower_and_assemble(kimi_model_program):
    words = kimi_model_program.assemble()
    assert len(words) > 0
    assert all(0 <= w <= 0xFFFF_FFFF for w in words)
    print(f"KIMI_STATIC_INSTR={len(words)} BYTES={len(words) * 4}")
```

- [ ] **Step 2: Run, and record both counts**

Run: `pytest aten/tests/test_whole_model_lowering.py -v -s`
Expected: PASS, printing two counts.

The recurrent layers should be hardware loops rather than unrolled triples. **If the count is larger than the loop structure predicts, a sweep is unrolled** — find it with `grep -c C_LOOP_START` on the dumped layer before proceeding.

- [ ] **Step 3: Add the static gates**

Append to `aten/tests/test_instruction_budget.py`, with the ceilings set to the Step 2 measurements plus 10% and the date recorded:

```python
# MEASURED 2026-__-__. Static instruction count is a property of the compiled
# artifact, not a model output -- it is the only quantity in this project
# admissible as a gate. See the plan's "Instruction accounting" section.
NEMOTRON_STATIC_INSTR_MAX = None   # fill in from Task 11 Step 2
KIMI_STATIC_INSTR_MAX = None       # fill in from Task 11 Step 2
```

- [ ] **Step 4: Re-run and confirm the gates pass**

Run: `pytest aten/tests/test_instruction_budget.py -v`
Expected: PASS.

- [ ] **Step 5: Wire into CI and prove it runs**

Run: `grep -rn "test_whole_model_lowering\|test_instruction_budget" .github/workflows/`
Expected: at least one hit each. A guard that is defined but never invoked is the failure mode this step exists to prevent — a guard that is defined but never invoked stays green forever without executing once.

- [ ] **Step 6: Commit**

```bash
git add aten/tests/test_whole_model_lowering.py aten/tests/test_instruction_budget.py .github/workflows/
git commit -m "test: lower both whole models and gate their static instruction counts"
```

---

### Task 12: Execute a whole model end to end in the emulator

This has never been done here. It is the difference between "we generated machine code" and "PLENA runs the model".

**Files:**
- Create: `transactional_emulator/testbench/models/whole_model_test.py`
- Modify: `justfile`, `.github/workflows/`

**Interfaces:**
- Consumes: the lowered programs from Task 11.
- Produces: a first-layer-to-last-layer emulator run of both models with synthetic weights, numerically compared against a full PyTorch reference forward.

- [ ] **Step 1: Write the failing end-to-end test**

```python
"""First layer to last, in the emulator, against a full reference forward.

Start with a reduced layer count so the test is runnable in CI, then run the
full depth as a nightly. A shape-reduced model exercises every code path a
full one does; what it cannot catch is HBM address-space exhaustion, which
Step 4 checks separately.
"""

def test_nemotron3_reduced_depth_matches_reference(whole_model_harness):
    out = whole_model_harness.run("nemotron3", layers=4, tokens=4)
    torch.testing.assert_close(out.actual, out.expected, rtol=1e-2, atol=1e-2)


def test_kimi_k3_reduced_depth_matches_reference(whole_model_harness):
    out = whole_model_harness.run("kimi_k3", layers=4, tokens=4)
    torch.testing.assert_close(out.actual, out.expected, rtol=1e-2, atol=1e-2)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `python transactional_emulator/testbench/models/whole_model_test.py`
Expected: FAIL — the harness does not exist.

- [ ] **Step 3: Build the harness and get 4 layers passing**

Reuse the connected-test harness on `feat/mamba2-support` for program loading, HBM image construction, and result extraction. The new part is chaining layer outputs into layer inputs rather than testing one layer in isolation.

- [ ] **Step 4: Check the HBM address space**

Run the harness with the full layer count and assert the required HBM allocation is within the emulator's configured `HBM_SIZE` (`plena_settings.toml`, currently 16 GiB).

This is a real wall: the emulator preloads HBM from a flat file starting at offset 0, so the allocation and the file both have to span every address the program touches. A layout whose regions sit far apart needs a span-sized allocation whatever the live data comes to, and stops being executable long before it stops being describable. If ours does that, pack the regions into one bounded arena.

- [ ] **Step 5: Run full depth as a nightly**

Add a `just test-whole-model-full` recipe running 52 and 93 layers, wired to a scheduled workflow rather than to every push.

- [ ] **Step 6: Commit**

```bash
git add transactional_emulator/testbench/models/whole_model_test.py justfile .github/workflows/
git commit -m "test: execute both whole models first layer to last in the emulator"
```

---

### Task 13: Bind to real checkpoint weights

**Files:**
- Create: `transactional_emulator/testbench/models/checkpoint_test.py`
- Modify: `justfile`, `doc/` (record the result)

**Interfaces:**
- Consumes: the harness from Task 12.
- Produces: both models running on real published weights, compared against the HuggingFace reference forward.

- [ ] **Step 1: Write the failing checkpoint test**

```python
"""Real weights, not randn.

Synthetic weights validate the datapath; they cannot validate the layout
contract -- a transposed projection, a wrong head-group mapping, or a swapped
q/k ordering all produce plausible numbers on symmetric random inputs and
wrong numbers on trained ones.
"""

import pytest


@pytest.mark.checkpoint
def test_nemotron3_layer0_matches_huggingface(checkpoint_harness):
    out = checkpoint_harness.run_layer("nemotron3", layer=0, tokens=8)
    torch.testing.assert_close(out.actual, out.expected, rtol=2e-2, atol=2e-2)


@pytest.mark.checkpoint
def test_kimi_k3_layer0_matches_huggingface(checkpoint_harness):
    out = checkpoint_harness.run_layer("kimi_k3", layer=0, tokens=8)
    torch.testing.assert_close(out.actual, out.expected, rtol=2e-2, atol=2e-2)
```

Mark them `checkpoint` and deselect by default, so contributors without the
weights are not blocked; run them in a scheduled workflow that has them.

- [ ] **Step 2: Run and confirm it fails**

Run: `pytest -m checkpoint -v`
Expected: FAIL — no harness.

- [ ] **Step 3: Load one real layer and get it passing**

Load the layer's weights from the published checkpoint, run the HF reference
forward for that layer, lower and run the same layer on the emulator, compare.
Start with layer 0 of each model.

Expect this step to surface layout bugs rather than arithmetic bugs. When one
appears, fix the **lowering**, and add a synthetic test that would have caught
it — a synthetic case with a deliberately asymmetric weight, so the regression
is guarded without needing the checkpoint.

- [ ] **Step 4: Extend to the whole model at reduced depth**

Run layers 0–3 chained, on real weights, against the HF forward.

- [ ] **Step 5: Record the result**

Write a short `doc/checkpoint_validation.md` stating exactly what was validated:
which checkpoint revision, which layers, which tolerance, prefill or decode.
State plainly what was **not** validated. This document is what licenses the
claim "PLENA runs Nemotron-3 and Kimi-K3"; keep it narrower than the claim.

- [ ] **Step 6: Commit**

```bash
git add transactional_emulator/testbench/models/checkpoint_test.py justfile doc/checkpoint_validation.md
git commit -m "test: validate both models against real checkpoint weights"
```

---

## Phase 4 — Measure, then retire

Measurement and cleanup, last. No new mechanism is designed here.

---

### Task 14: Banking study and the uncalibrated latency report

**Files:**
- Create: `analytic_models/performance/vector_sram_banking.py`
- Create: `doc/static_path_measurements.md`

**Interfaces:**
- Consumes: green Tasks 11–13.
- Produces: a recorded answer to the Vector SRAM banking question, and a latency report that is honest about its own status.

- [ ] **Step 1: Retarget the bank-conflict model**

A bank model that consumes only `(rows, columns)` is algorithm-independent. Port it into `analytic_models/performance/vector_sram_banking.py`, driven by the Vector SRAM row/column geometry of the lowered Mamba and KDA tiles.

- [ ] **Step 2: Run it and record the answer**

Run: `python -m analytic_models.performance.vector_sram_banking`
Expected: stall-cycle counts for the Mamba and KDA state sweeps.

Record in `doc/static_path_measurements.md`:
- Stalls ≈ 0 → the vector path is limited by instruction issue, and the follow-up is post-increment addressing, not SRAM work.
- Stalls material → that is the evidence for a Vector SRAM bank-mapping mode. Separate plan.

- [ ] **Step 3: Write the measurement report**

`doc/static_path_measurements.md` must separate the two kinds of number explicitly, with the table from this plan's "Instruction accounting" section:

- **Hard facts:** static instruction counts (Task 11), dynamic instruction counts, HBM bytes moved.
- **Uncalibrated model output:** cycles, µs, TPOT — labelled as such on every line, and not compared against any other uncalibrated model output; two such numbers decide nothing between them.

- [ ] **Step 4: Commit**

```bash
git add analytic_models/performance/vector_sram_banking.py doc/static_path_measurements.md
git commit -m "docs: banking study and an explicitly uncalibrated latency report"
```

---

### Task 15: Confirm the tree carries no runtime state machinery

The static path is defined by what it does *not* have: no descriptors read at
run time, no command queue, no residency cache. This task is the check that it
stayed that way, not a removal -- this line of work was cut from `main`, which
has none of it, so there has never been anything to delete.

**Interfaces:**
- Consumes: green Tasks 11-14.
- Produces: a tree with no descriptor fetch, no queue, no residency cache.

- [ ] **Step 1: Confirm nothing introduces runtime state machinery**

```bash
grep -rniE "descriptor|residency|state_queue|StateDescriptor" \
  --include="*.rs" --include="*.py" --include="*.svh" --include="*.json" . \
  | grep -v "^./docs/superpowers/"
```
Expected: no hit that describes a run-time fetch. `aten/tests/test_no_state_engine.py`
is the CI form of this and must be green.

- [ ] **Step 2: Confirm the emulator builds and passes**

Run: `cargo test --manifest-path transactional_emulator/Cargo.toml`
Expected: PASS.

Run: `cargo clippy --manifest-path transactional_emulator/Cargo.toml --all-targets -- -D warnings`
Expected: no warnings.
## What is deliberately not carried over

Recorded so a reviewer does not read these as oversights.

| Dropped | Why |
|---|---|
| 256-byte HBM descriptor | Every field was a compile-time constant; the indirection carried no runtime information and cost a serially-dependent HBM read per command. |
| 16 async queues, events, `FENCE`, completion records | Never exercised by any emitted program; the event map had no eviction path, so it was not implementable as an RTL spec. |
| `StateCache` residency table | Residency is a compile-time allocation. The compiler already emitted explicit spill code; the runtime table only re-checked what the compiler guaranteed. |
| `cache_hit_rate` metric | It was `!descriptor.is_streaming()` replayed — a pure function of the program image, not a measured effect. |
| 7 runtime status codes | All collapsed to a `panic!` at the dispatch site. Shape and precision violations are now Python or assembler errors. |
| Named layout modes carried in a descriptor | The projection co-layout finding is real and worth publishing, but it is a Matrix-writeback question, not a state-engine one. It needs its own plan, gated on Task 14's banking result. |

## What is carried over

| Kept | Where it lands |
|---|---|
| KDA golden reference and its self-test | Task 1 |
| Mixed-precision error study (Mamba MX8 2.07% vs KDA 0.35% at 32K) | Justifies the FP32/BF16 split in Task 2's docstring |
| Bank-conflict model | Task 14, retargeted at Vector SRAM |
| Full-system HBM-bandwidth DSE | Untouched; out of scope for this plan |
| Multi-token state-carry testing with a real convolution kernel and non-zero decay | Task 4's four-token test, plus the explicit degeneracy guard |
