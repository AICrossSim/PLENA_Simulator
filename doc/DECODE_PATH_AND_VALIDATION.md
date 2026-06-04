# Decode Path and Numerical Validation Design

This document describes (1) the **latency-only** decode (TPOT) path that is already
wired through the compiler and `run_model`, and (2) the **full numerical-validation**
design to add later (design only — not yet implemented). It also records the known
single-token limitation and the repro commands.

File references use `<file>:<line>` against the current tree. The decode path lives in
two branches: `feat/decode-path` (compiler) and `feat/decode-ttft-tpot` (simulator /
`run_model`).

---

## 1. Latency-only decode path (implemented) and why TPOT is value-independent

### Goal

Measure decode **TPOT** (time-per-output-token) latency for a single new token that
attends to a KV cache of length `past_len`. TPOT latency is an **op/cycle count**: the
emulator's `sim_latency_ns` is a function of *which* ops run and *how many* tiles they
touch, not of the numeric *values* in those tiles. A GEMM over a `(1, head_dim) @
(kv_seq_len, head_dim)^T` score, a softmax over `kv_seq_len` columns, and a `(1,
kv_seq_len) @ (kv_seq_len, head_dim)` PV all cost the same regardless of whether the
cached K/V rows hold real data or zeros. Therefore **the KV cache values do not need to
be correct** for a latency measurement — the only requirement is that
`flash_attention` actually **reads `kv_seq_len` rows** (i.e. iterates
`ceil(kv_seq_len / mlen)` key/value tiles) so the op/cycle count is right.

### Param contract (already in place)

`compile_native_hf_decoder(...)` gained `past_len: int = 0` —
`PLENA_Compiler/aten/plena_frontend.py:2935`, docstring `:2937-2944`:

- `past_len == 0` → **prefill**, byte-identical to the pre-decode compiler.
- `past_len > 0` → **decode**: a single new token (`seq_len`, usually `1`) attends to
  `past_len` cached keys plus itself. `kv_seq_len = past_len + seq_len` is computed
  **internally** (`plena_frontend.py:1112`), so callers never pass `kv_seq_len`.
- `past_len < 0` → `ValueError` (`plena_frontend.py:3049-3050`).
- Decode requires **head packing OFF**: `past_len > 0` with a non-`None`
  `head_packing` raises `NotImplementedError`
  (`plena_frontend.py:3051-3058`) because the packed-attention kernels take
  `valid_cols` but not `kv_seq_len`, so they cannot grow the key range.
- Because `run_model` auto-enables packing when `head_dim < mlen` **iff** `hlen` and
  `broadcast_amount` are passed, decode callers must pass
  `attention_head_packing=False` (or omit `hlen`/`broadcast_amount`). See the auto-pack
  rule at `plena_frontend.py:3023-3024` and the decode call site
  `transactional_emulator/testbench/run_model.py:201-207`.

### RoPE at absolute position N

The new token sits at absolute position `past_len`, so its RoPE angles use positions
`[past_len, past_len + seq_len)`:

- `make_rope_inputs(seq_len, config, position_offset=0)` —
  `PLENA_Compiler/aten/reference.py:155-167`. Decode passes
  `position_offset=past_len`. Default `0` keeps prefill tables identical.
- `_make_rope_tables(seq_len, head_dim, theta=10000.0, position_offset=0)` —
  `aten/reference.py:127-142`; `positions = torch.arange(position_offset,
  position_offset + seq_len)` at `:136`.
- Call site that threads `past_len` into the angles:
  `plena_frontend.py:3216` — `make_rope_inputs(seq_len, model_cfg,
  position_offset=past_len)`.

### KV region sizing — how `K_stored`/`V_stored` are sized to `kv_seq_len`

This is the crux of "reads `kv_seq_len` rows". In `_emit_attention_block`
(`plena_frontend.py:1077`, `past_len: int = 0` at `:1098`):

- `is_decode = past_len > 0` (`:1111`), `kv_seq_len = past_len +
  active_seq_len_per_batch` (`:1112`).
- `kv_physical_rows = _ceil_to_multiple(max(prog.mlen, kv_seq_len), prog.mlen) if
  is_decode else None` (`:1115-1116`) — round `kv_seq_len` **up to whole MLEN tiles**,
  because the kernel reads `ceil(kv_seq_len / mlen)` blocks.
- `attn_causal_mask = None if is_decode else causal_mask` (`:1118`) — the single
  last-position query attends to **all** keys, so the triangular mask is dropped. This
  also avoids the kernel's `q_offset != 0` `NotImplementedError` (see §3).
- `kv_physical_rows` is forwarded to `_emit_kv_stores(...)` (`:1175-1187`).

In `_emit_kv_stores` (`plena_frontend.py:729`, new arg `kv_physical_rows: int | None =
None` at `:740`):

- Projections K/V are emitted at the natural (new-token) row count, then for decode
  the helper widens them: `if kv_physical_rows is not None and kv_physical_rows >
  K_h.physical_shape[0]:` (`:809`) → `K_full = _grow_kv_region(prog, K_h,
  kv_physical_rows, ...)` / `V_full = _grow_kv_region(...)` (`:812-813`) → `prog.store`
  as `K_stored_{layer}_h{kv_h}` / `V_stored_{layer}_h{kv_h}` (`:814-815`).
- Prefill path (`else` at `:819-822`) stores `K_h`/`V_h` unchanged — byte-identical.

New helper `_grow_kv_region(prog, source, target_rows, name)` —
`plena_frontend.py:830-845`: allocates a `target_rows`-tall region with
`physical_shape=(target_rows, source.physical_shape[1])`, `vram_fill_zero`s it, then
`vram_add`s `source` into the top `source.shape[0]` rows. **The cached rows below the
new token stay zero** — latency-only: their values do not matter.

The stored `K_stored`/`V_stored` therefore present `kv_physical_rows` (≥ `kv_seq_len`,
MLEN-rounded) physical rows to `flash_attention`, which is called with
`kv_seq_len=kv_seq_len` (`plena_frontend.py:1214-1224`). The kernel then iterates all
`num_k_blocks = ceil(kv_seq_len / mlen)` tiles (`program_attention.py:238-239`), giving
the correct op/cycle count.

### `run_model` wiring (simulator side)

- `--past-len` flag — `run_model.py:301-309`; `--seq-len` defaults to `1` for decode
  (`:174`). Decode gets its own build dir suffix `_decode_p{past_len}` so prefill
  artifacts are not clobbered (`_build_dir` at `:62-65`, `phase` at `:387`).
- Decode compile call omits `hlen`/`broadcast_amount` and passes
  `attention_head_packing=False`, `past_len=past_len` (`:201-207`).
- Decode is latency-only, so verification is auto-skipped: `if args.past_len > 0:
  args.no_verify = True` (`:334-335`). `emulate_from_result(..., verify=not
  args.no_verify)` (`:389-398`) → `run_and_assert(..., verify=False)` runs the emulator
  and captures `sim_latency_ns` but **skips** the golden comparison
  (`emulator_runner.py:375-380`).

### Latency interpretation caveats

The reported decode **TPOT here is a compute / op-count figure, not a
weight/KV-bandwidth-bound TPOT.** Two things to keep in mind when reading these numbers:

- **HBM KV prefetch is async / overlapped**, not the bandwidth-bound number. The
  emulator's `sim_latency_ns` reflects the compute-side op/cycle count of the attention
  GEMMs and softmax; it does **not** model the weight/KV-cache HBM read as a serial
  bottleneck. Real decode is typically memory-bound (it must stream the full KV cache +
  weights per token), so this figure is an optimistic compute-side lower bound, not the
  bandwidth-bound TPOT.
- **The latency-only construction adds small on-chip artifacts** that slightly inflate
  the compute count relative to a hand-tuned decode kernel:
  - `_grow_kv_region`'s per-head `vram_fill_zero` over the padded `kv_physical_rows`
    tile plus a `vram_add` of the new-token rows (`plena_frontend.py:830-845`) — pure
    on-chip zeroing/copy that a value-correct preload path would not need.
  - an **unused `causal_mask` `Load_Batch`** still emitted in decode even though the
    single-token query drops the triangular mask (`attn_causal_mask = None`,
    `plena_frontend.py:1118`).

  These are minor and value-independent, but they mean the count is a slightly padded
  compute figure rather than the minimal decode op count.

---

## 2. Full numerical-validation design (NOT yet implemented)

To make a decode step **numerically correct** (so the seq_len=1 output matches HF with
a real KV cache), three pieces must be added. The KV *values* must be real, the
reference must model attend-all decode with RoPE-at-N, and the cache must be preloaded
into the emulator's HBM. None of these are needed for latency; they are purely for
correctness validation.

### 2a. Reference: a `decode_mode` in `aten/reference.py`

The CPU/scheduled reference currently always uses `causal=True`
(`reference.py:288`, `:406`, `:472`) and does **not** accept a KV cache. Add a decode
mode threaded through the reference attention blocks. The natural insertion region is
`reference.py:157-241` (right after `make_rope_inputs` and into the
`run_*_reference`/attention-block signatures). Design:

- **Accept `past_key_values`**: extend the reference entry points
  (`run_decoder_reference` `:170`, `run_native_decoder_scheduled_reference` `:209`,
  and the per-block `_attention_block_ref` `:257` / `_scheduled_attention_block_ref`
  `:295`) with an optional `past_key_values` (list of `(K_cached, V_cached)` per layer
  per KV head, each `(past_len, head_dim)`).
- **RoPE-at-N for the new token only**: the reference already receives
  `cos_table`/`sin_table` built with `position_offset=past_len` (caller passes the same
  tables it gives the compiler). The new K row is RoPE'd at position `past_len`
  (`_rope_ref` at the K-projection, currently `reference.py:278`). **Do NOT re-RoPE the
  cached K** — the cache stores keys that were already rotated at their own absolute
  positions when they were first produced; rotating them again would double-apply RoPE.
  So: RoPE the new K/Q only, then `K_eff = cat([K_cached, K_new_rotated], dim=0)`,
  `V_eff = cat([V_cached, V_new], dim=0)`.
- **`causal=False` / attend-all**: the single new query at the last position attends to
  *every* key in `K_eff` (all past + itself), so call `_flash_attn_ref(q_h, K_eff,
  V_eff, scale, causal=False)` (`reference.py:651-659`) and
  `_flash_attn_scheduled_ref(..., causal=False)` (`:662-684`) in decode mode. This
  mirrors the compiler dropping the triangular mask (`plena_frontend.py:1118`). For
  `seq_len == 1` a full causal mask over a `(1, kv_seq_len)` score is a no-op anyway,
  but `causal=False` is the honest expression of "attend to all cached keys".
- Default `past_key_values=None` keeps prefill paths byte-identical.

### 2b. KV-cache PRELOAD via `sim_env_utils`

For numerical validation, the cached rows in `K_stored`/`V_stored` must hold the real
(MXFP8-quantized) cached keys/values instead of the `_grow_kv_region` zeros. The
preload mechanism already exists in `transactional_emulator/testbench/sim_env_utils.py`:

- `create_mem_for_sim(..., hbm_addrs=...)` — `sim_env_utils.py:315-325`. It writes each
  tensor's MXFP-quantized blocks to HBM at its compiler-assigned offset
  `hbm_addrs.get(name)` via `add_mx_file(..., hbm_addr=...)` (`:396-406`).
- `add_mx_file(..., hbm_addr=...)` — `sim_env_utils.py:262-285`; the padding/seek to
  `hbm_addr` is in the file-writer at `:143-156`.

Design for numerical validation:

1. In the reference, after producing the real cached K/V for each layer/KV head,
   **quantize them to MXFP8** with the same `HBM_V_ACT_TYPE` config used for activations
   (`sim_env_utils.py:347-354`, `_mx_quant_config` at `:300+`) — this is the precision
   the `K_stored`/`V_stored` regions hold.
2. Add an `input_tensors` entry (and `data_order` / `tensor_layouts` entry) for each
   `K_stored_{layer}_h{kv_h}` / `V_stored_{layer}_h{kv_h}` region, sized
   `(kv_physical_rows, head_dim)` with the cached rows in the **top** `past_len` rows
   and the new token's row at index `past_len` (matching `_grow_kv_region`'s top-load
   layout, `plena_frontend.py:836-844`).
3. `create_mem_for_sim` then writes that MXFP8 data to the `K_stored`/`V_stored`
   region's `hbm_addrs` offset (`sim_env_utils.py:405`), so the emulator reads real
   cached keys/values during `flash_attention`.

This requires the compiler to **expose** the `K_stored`/`V_stored` regions as
preloadable inputs in the result dict (`input_tensors` + `hbm_addrs` already carry
their offsets — see `hbm_addrs.json` dumped at `run_model.py:371-379`). For latency
that exposure is unnecessary (zeros are fine), so it is gated behind the
numerical-validation mode.

### 2c. `run_and_assert` numerical check for `seq_len=1`

With (2a) producing a real golden output and (2b) preloading real K/V, drop the
auto-`--no-verify` for decode and let the existing comparison run:

- Remove / gate the `if args.past_len > 0: args.no_verify = True` short-circuit
  (`run_model.py:334-335`) behind a `--validate-decode` opt-in.
- `emulate_from_result(..., verify=True)` (`emulator_runner.py:395-403`) →
  `run_and_assert(..., verify=True)` (`:332-392`) then runs
  `compare_emulator_output` and asserts (`:382-390`). The compared output is the
  decoder's final `(seq_len=1, hidden)` row; the golden side is the decode reference
  from (2a). For `seq_len=1` only that single output row is asserted.

---

## 3. Known limitation: single-token (seq_len=1) only

The decode path is correct **only for a single new token** (`seq_len == 1`, which is
the `run_model` default for decode, `run_model.py:174`). Multi-token / chunked decode
(a block of new queries at offset `past_len` attending to a longer cache) is **not**
supported because of the static causal mask in the kernel:

- `program_attention.py:259-265`: when a key block straddles the causal diagonal
  (`needs_triangular_mask`) and `query_first != key_first` (i.e. `q_offset != 0`), the
  kernel raises `NotImplementedError`. The static `(mlen, mlen)` mask only encodes the
  zero diagonal, which is exact only when the straddle sits on the `q_idx == k_idx`
  diagonal (prefill, `q_offset == 0`).
- Single-token decode sidesteps this entirely: it sets `attn_causal_mask = None`
  (`plena_frontend.py:1118`), so the new query attends to all keys with no triangular
  mask, and the `q_offset != 0` branch is never reached.

To generalize to multi-token / chunked decode, the kernel needs a **`q_offset`-aware
triangular mask**: the `(mlen, mlen)` mask at `program_attention.py:259-265` must be
shifted by `query_first - key_first` so a straddling block off the zero diagonal masks
the correct entries. Until that lands, keep `seq_len == 1`.

---

## 4. Repro commands

Always activate conda first:

```bash
source /home/khl22/miniconda3/etc/profile.d/conda.sh && conda activate plena
```

### Prefill regression (must stay byte-identical: past_len == 0)

```bash
bash /home/khl22/new_plena/PLENA_Simulator/run.sh \
  python3 transactional_emulator/testbench/run_model.py smollm2 --case decoder --layers 1
```

### Decode latency-only (TPOT) — single token, KV length = past_len

```bash
# past_len cached keys, one new token (seq_len defaults to 1); verify auto-skipped.
bash /home/khl22/new_plena/PLENA_Simulator/run.sh \
  python3 transactional_emulator/testbench/run_model.py smollm2 --case decoder --layers 1 \
  --past-len 64
```

- Compile-only (ISA + hbm_addrs.json, no emulator): add `--compile-only`.
- `sim_latency_ns` is captured in
  `<build_dir>/rust_emulator_run_stats.json`; the decode build dir is suffixed
  `_decode_p<past_len>` (`run_model.py:62-65`).

### Compiler unit reference

```bash
bash /home/khl22/new_plena/PLENA_Simulator/run.sh \
  python3 -m pytest PLENA_Compiler/aten/tests/test_plena_compiler.py -k decoder
```

### Numerical validation (after §2 is implemented)

```bash
# (planned flag) real KV preload + decode reference + golden assert at seq_len=1
bash /home/khl22/new_plena/PLENA_Simulator/run.sh \
  python3 transactional_emulator/testbench/run_model.py smollm2 --case decoder --layers 1 \
  --past-len 64 --validate-decode
```
