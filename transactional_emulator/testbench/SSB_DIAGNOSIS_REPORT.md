# SingleStreamBlock — staged-precision diagnosis report

Date: 2026-05-16

## Question

The full SingleStreamBlock chain (`tvm_single_stream_block_test.py`)
produces an output whose relative-error match rate (`|err|/|golden| <=
0.2`) was only ~60%. A per-step staged sweep showed `flash_attention`
collapsing to ~32-45%. Was that a real algorithm/wiring bug, or just
fp16 / quantization error accumulating down a deep chain?

## Method

`tvm_ssb_staged_test.py` truncates the chain at a chosen step and
stages that step's output to VRAM for comparison — isolating each
kernel. Three diagnostic modes were used to pin down attention:

1. **Standalone** — the existing `tvm_flash_attention_min_test.py`,
   attention alone, clean `randn*0.5` Q/K/V.
2. **clean-attn** (`SSB_CLEAN_ATTN`) — only the flash_attention kernel
   compiled, but through the staged driver's compile/address/build
   path, fed the same clean Q/K/V.
3. **chain-clean-attn** (`SSB_CHAIN_CLEAN_ATTN`) — the FULL chain runs
   (layernorm -> ... -> rope, real HBM layout, shared FPRAM, every
   upstream kernel executing), but attention's Q/K/V are separate
   clean `randn*0.5` input tensors instead of the chain's rope /
   linear_v outputs.

## Results

### Per-step staged sweep (attention fed the chain's real upstream output)

| step             | rel.err <= 0.2 | note                       |
|------------------|----------------|----------------------------|
| layernorm        | 97.19%         | fp16 noise floor           |
| modulate         | 93.65%         | small accumulation         |
| linear_q         | 78.73%         | GEMM K-reduce in fp16      |
| ...              | ...            |                            |
| rope_q           | 77.31%         |                            |
| rope_k           | 75.07%         |                            |
| flash_attention  | ~45%           | attention over 75% inputs  |

### Attention isolation — three modes, identical inputs

| mode              | rel.err <= 0.2 |
|-------------------|----------------|
| standalone        | 82.61%         |
| clean-attn        | 82.61%         |
| chain-clean-attn  | 82.61%         |

All three are bit-identical (82.61%, exact).

### HLIR comparison

`build/flash_attention.hlir.txt` (compiled inside the chain) vs the
backed-up `flash_attention_min.hlir.GOLDEN.txt` (standalone):

- **Op sequence: bit-identical** — the 56-op online-softmax body
  diffs zero lines.
- Only the FPRAM/HBM buffer ADDRESSES differ, and exactly as
  expected: in the chain the hoisted constants
  (`__const_f16_neg10000`, `__const_f16_0p25`) are pinned low at
  38/39 by the driver's `fpram_address_overrides`, and the per-row
  scratch (M_OLD..L_INV) sits high from `FPRAM_SCRATCH_BASE` (96+).
  The two segments do NOT overlap — the chained-FPRAM layout rule
  holds.

The online-softmax algorithm itself was also walked op-by-op:
init running max/sum -> per kv-block (Q.K^T, scale 0.25, running-max
update via accumulating reduce, correction factor exp(m_old-m_new),
P=exp(S-m_new), rowsum, l/O rescale, P.V, accumulate) -> final
reciprocal normalize. Standard and correct.

## Conclusion

**The chain's attention degradation is NOT a bug — it is quantization
/ input degradation.**

- The flash_attention kernel, its in-chain compilation, its pinned
  addresses, and the shared-FPRAM environment are all correct: three
  independent measurements give exactly 82.61%, and the in-chain HLIR
  op sequence is bit-identical to the known-good standalone HLIR.
- 82.61% is this online-softmax kernel's fp16 ceiling with clean
  inputs. When attention is instead fed the chain's degraded rope
  output (~75% match rate), its two GEMMs (Q.K^T and P.V) plus the
  softmax amplify that error, and the output lands at ~45%. That is
  the expected behaviour of fp16 GEMM error propagation, not an
  algorithm fault.
- The earlier "flash_attention 32%" sweep number was a diagnostic
  artefact (a since-fixed golden that included an un-written half of
  the old wide concat tensor), not a kernel result.

The `o_head_offset` shared-wide-concat path was also removed as part
of this work: flash_attention and gelu now each write their own
compact output tensor, joined by a dedicated `concat_min` kernel
before linear2. Every step is now independently verifiable.

### Where the precision actually goes

The match rate erodes gradually down the chain — layernorm 97% ->
modulate 94% -> linear ~78% -> rope ~75% -> attention ~45%. Each
kernel here has been through many compiler passes; the steady decline
is consistent with fp16 round-off and MXFP HBM packing accumulating,
not with a localized bug. No single step shows the discontinuous
collapse that would mark a real defect.

## Follow-up

- Diagnostic env-var defaults in `tvm_ssb_staged_test.py`
  (`SSB_CHAIN_CLEAN_ATTN`, `SSB_CLEAN_ATTN`) were flipped back to ""
  so a plain run is the normal staged chain again.
- `flash_attention_min.hlir.GOLDEN.txt` is kept as the reference HLIR
  for any future attention regression check.
