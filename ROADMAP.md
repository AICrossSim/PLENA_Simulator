# PLENA Simulator — Development Roadmap

Professor-approved. Based on SmolVLM2-256M profiling (see `smolvlm2_profile.md`).

---

## Step 1: Conv2d Hardware Hookup to PLENA RTL

Connect the software conv2d/im2col implementation to the RTL-level hardware.

**Context:**
- Current impl: software im2col → systolic matmul (`compiler/asm_templates/im2col_asm_no_shift.py`)
- Conv2d = 83% of vision encoder instructions, 86% of vision encoder cycles
- At VQA scale (20 tokens), conv2d ≈ 0.9% of end-to-end cost — but needed for correctness

**Tasks:**
- [ ] Map im2col ISA instructions to RTL hardware units
- [ ] Verify timing/cycle counts match behavioral sim
- [ ] Integration test: smolvlm2_vision_encoder_test passes against RTL sim

---

## Step 2: Behavioral Simulator — Host CPU memcpy Support

SmolVLM2-256M memory footprint exceeds on-chip SRAM. Need to model host CPU ↔ chip DMA transfers.

**Context:**
- SmolVLM2-256M weights: ~500MB (MXFP8) — exceeds chip HBM budget
- Current sim assumes all weights pre-loaded in HBM (infinite memory assumption)
- Real chip: host CPU must DMA weight tiles from DRAM into HBM between layers

**Tasks:**
- [ ] Model HBM capacity limit (configurable in `plena_settings.toml`)
- [ ] Add memcpy/DMA instruction to PLENA ISA (host → HBM, HBM → host)
- [ ] Scheduler: auto-insert memcpy ops when weight tile doesn't fit in HBM
- [ ] Cycle model: add memcpy latency to ASM profiler (`analytic_models/roofline/asm_profiler.py`)
- [ ] Test: multi-layer decoder where weights exceed HBM, verify DMA scheduling

---

## Step 3: Code Cleanup

Clean up implementations before paper submission and FPGA mapping.

**Tasks:**
- [ ] Consistent test output formatting across all `*_aten_test.py` files
- [ ] Remove debug prints / dead code in `compiler/asm_templates/`
- [ ] Document VRAM layout conventions (currently scattered across CLAUDE.md + test files)
- [ ] Unify `model_layer_test_builder.py` API (currently mixes `prog.rms_norm` and `ops.rms_norm`)
- [ ] Add missing justfile/run.sh entries for all tests (bmm, aten-compiler-rms-norm, etc.)
- [ ] Clean up compiler submodule (`compiler/.omc/` untracked junk)
- [ ] Accuracy table: collect allclose % for all ops at multiple configs (paper table)

---

## Step 4: Map to Real FPGA Implementation

Synthesize and validate PLENA on FPGA hardware.

**Tasks:**
- [ ] RTL synthesis: map PLENA ISA decoder + datapath to FPGA (Xilinx/Intel target TBD)
- [ ] Timing closure: meet frequency target for systolic array
- [ ] Memory interface: connect HBM controller to on-chip SRAM
- [ ] Run behavioral sim test suite against FPGA bitstream
- [ ] Measure real cycle counts vs behavioral sim estimates
- [ ] Update cycle cost model in `asm_profiler.py` with measured FPGA timings

---

## Step 5: Differential Precision for Encode / Decode

Professor note: use different numerical precision for vision encoder vs LM decoder.

**Context:**
- Current: MXFP8 (E4M3) for all HBM weights
- Potential: higher precision (FP16/BF16) for vision encoder (quality-sensitive patch features)
- Potential: lower precision (INT4/FP4) for LM decoder weights (larger, less sensitive)
- Matches industry trend: Google Gemma, Apple MLX use mixed precision by layer type

**Tasks:**
- [ ] Extend `quantize_to_mxfp()` in `model_layer_test_builder.py` to support multiple formats
- [ ] Add precision config per-module to `plena_settings.toml`
- [ ] ISA: extend HBM load instructions to specify precision format
- [ ] Accuracy study: allclose % vs precision for vision encoder vs decoder
- [ ] Profile: cycle cost change with wider precision (FP16 = 2× memory bandwidth)

---

## Current Status

| Step | Status | Notes |
|---|---|---|
| 1. Conv2d RTL hookup | 🔲 Not started | Software impl complete, passes tests |
| 2. Memcpy / DMA sim | 🔲 Not started | All tests assume infinite HBM |
| 3. Code cleanup | 🔄 In progress | Test suite passing, formatting TBD |
| 4. FPGA mapping | 🔲 Not started | Needs RTL synthesis |
| 5. Diff precision | 🔲 Not started | Professor future note |

## Immediate Next: Step 3 (Cleanup)

Start here before RTL work — clean code is easier to port to hardware.
Priority cleanup items:
1. Consistent test output format
2. Accuracy table for paper
3. Missing justfile entries
4. Dead code / debug prints in asm_templates
