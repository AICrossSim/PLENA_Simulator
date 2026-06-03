# SmolVLM2-256M ISA Cycle Profile

**Config:** 12L vision encoder + connector + 1L text decoder, mlen=256, blen=64, seq=256  
**Board:** Nexys Video — Artix-7 XC7A200T @ 100MHz, DDR3-1600 x16 (1.6 GB/s peak, ~1.0 GB/s effective)  
**Note:** mlen=256 exceeds Artix-7 synthesis resources. This profile is a what-if analysis. Realisable configs use mlen=64 (proven) or mlen=128 (untested).  
**ASM source:** `transactional_emulator/testbench/conv/build/smolvlm2_native_module/`  
**Profiler:** `transactional_emulator/testbench/aten/compare/asm_profiler.py --board nexys_a7 --mlen 256`  
**Date:** 2026-05-28 (corrected from 05-27: x16 bus width, validated against published references)

## Summary

| Stage | ISA lines | Dynamic instrs | Est cycles | Est time @100MHz |
|-------|-----------|----------------|-----------|------------------|
| Vision + Connector (12L) | 1,444,035 | 5,580,752 | 28.0M | **280 ms** |
| Text Decoder (1L) | 1,206,857 | 2,638,488 | 17.5M | **175 ms** |
| **Total VLM e2e** | 2,650,892 | 8,219,240 | 45.5M | **455 ms** |

## Vision + Connector (12L vision + connector)

**28.0M cycles @ Nexys A7 (dc_lib_dis, DDR3)**

### Per-opcode breakdown (top 15)

| Op | Dynamic count | Cyc/op | Total cycles | % |
|----|--------------|--------|-------------|---|
| M_MM | 44,496 | 256 | 11,390,976 | 40.7% |
| V_RED_SUM | 271,872 | 20 | 5,437,440 | 19.5% |
| S_ADDI_INT | 2,054,228 | 1 | 2,054,228 | 7.3% |
| V_MUL_VF | 297,216 | 5 | 1,486,080 | 5.3% |
| V_MUL_VV | 271,872 | 5 | 1,359,360 | 4.9% |
| H_PREFETCH_M | 3,033 | 400 | 1,213,200 | 4.3% |
| C_LOOP_END | 766,161 | 1 | 766,161 | 2.7% |
| V_ADD_VV | 343,296 | 2 | 686,592 | 2.5% |
| M_TMM | 2,304 | 256 | 589,824 | 2.1% |
| V_EXP_V | 73,728 | 6 | 442,368 | 1.6% |
| S_ST_FP | 381,125 | 1 | 381,125 | 1.4% |
| S_ADD_FP | 326,450 | 1 | 326,450 | 1.2% |
| H_PREFETCH_V | 15,492 | 20 | 309,840 | 1.1% |
| V_RECI_V | 36,864 | 7 | 258,048 | 0.9% |
| S_MAP_V_FP | 832 | 256 | 212,992 | 0.8% |

### Category summary

| Category | Cycles | % |
|----------|--------|---|
| **Matrix compute** | 11,998,272 | **42.9%** |
| **Vector compute** | 10,076,928 | **36.0%** |
| Scalar overhead | 3,537,049 | 12.7% |
| DDR3 memory | 1,546,080 | **5.5%** |
| Control | 796,938 | 2.9% |

## Text Decoder (1L)

**17.5M cycles @ Nexys A7 (dc_lib_dis, DDR3)**

### Per-opcode breakdown (top 15)

| Op | Dynamic count | Cyc/op | Total cycles | % |
|----|--------------|--------|-------------|---|
| M_MM | 32,040 | 256 | 8,202,240 | 46.9% |
| S_MAP_V_FP | 7,680 | 256 | 1,966,080 | 11.2% |
| H_PREFETCH_M | 3,240 | 400 | 1,296,000 | 7.4% |
| V_RED_SUM | 63,552 | 20 | 1,271,040 | 7.3% |
| S_ADDI_INT | 875,152 | 1 | 875,152 | 5.0% |
| V_MUL_VV | 161,472 | 5 | 807,360 | 4.6% |
| V_MUL_VF | 138,432 | 5 | 692,160 | 4.0% |
| V_EXP_V | 63,360 | 6 | 380,160 | 2.2% |
| V_RECI_V | 46,080 | 7 | 322,560 | 1.8% |
| C_LOOP_END | 267,883 | 1 | 267,883 | 1.5% |
| V_ADD_VV | 103,872 | 2 | 207,744 | 1.2% |
| S_LUI_INT | 201,149 | 1 | 201,149 | 1.1% |
| S_ST_FP | 197,760 | 1 | 197,760 | 1.1% |
| V_SUB_VF | 63,360 | 2 | 126,720 | 0.7% |
| V_ADD_VF | 46,080 | 2 | 92,160 | 0.5% |

### Category summary

| Category | Cycles | % |
|----------|--------|---|
| **Matrix compute** | 8,235,810 | **47.1%** |
| **Vector compute** | 4,055,424 | **23.2%** |
| Scalar overhead | 3,615,382 | 20.7% |
| DDR3 memory | 1,310,640 | **7.5%** |
| Control | 279,514 | 1.6% |

## DDR3 Memory Impact

The Nexys Video uses a single **MT41K256M16HA** DDR3-1600 chip with a **x16 bus** (2 bytes/transfer).

| Parameter | Value | Notes |
|-----------|-------|-------|
| Chip | MT41K256M16HA | Single x16, 512 MiB |
| Peak bandwidth | **1.6 GB/s** | 2B × 800 MT/s (x16 bus) |
| Effective bandwidth | **~1.0 GB/s** | ~60-65% MIG efficiency for sequential streaming |
| MIG read latency | ~27.7 cycles | ZipCPU measured on Kintex-7 MIG |
| Burst latency | ~5 PLENA cycles (50ns) | CAS + transfer for 64B |
| Row miss penalty | +3 cycles (30ns) | tRP+tRCD = 27.5ns |
| H_PREFETCH_V cost | 25 cycles | 256B / 1.0 GB/s = 256ns |
| H_PREFETCH_M cost | 400 cycles | 4KB / 1.0 GB/s = 4000ns |

| Stage | DDR3 cycles | % of total | Static prefetch count |
|-------|------------|-----------|---------------------|
| Vision+Connector | 1,629,300 | **5.8%** | 15,492 V + 3,033 M + 288 store |
| Text Decoder | 1,314,300 | **7.5%** | 5 V + 3,240 M + 180 store |

The text decoder is **more DDR3-bound** (7.5%) despite fewer total prefetches because M_PREFETCH dominates (3,240 × 400 = 1.3M cycles) — each matrix prefetch loads 4KB of weights from DDR3.

## Per-Kernel Breakdown

### Vision + Connector (12L) — 280 ms

| Kernel | Cycles | % | ms | Description |
|--------|--------|---|----|-------------|
| **Attention** | 13.5M | **48.4%** | 135 | Q/K/V projections + flash attention × 12 layers |
| **Layer Norm** | 8.4M | **30.0%** | 84 | RMS norm × 25 (2 per layer + final). V_RED_SUM (20 cyc) + V_MUL_VV (5 cyc) per element |
| **Conv2d im2col** | 6.0M | **21.4%** | 60 | SigLIP K=16 patch embedding via vector reductions (no matrix unit) |
| Connector | 0.07M | 0.2% | 0.7 | Linear projection 12288→576 |

### Text Decoder (1L) — 182 ms

| Kernel | Cycles | % | ms | Description |
|--------|--------|---|----|-------------|
| **FFN** | 8.7M | **48.0%** | 88 | Gate+Up linear (M_MM) + SiLU + Down linear (M_MM) |
| **Attention** | 5.3M | **29.1%** | 53 | Packed GQA (3 heads × online softmax + P@V via M_MM) |
| **Q/K/V Projection** | 2.5M | **13.8%** | 25 | Weight load from DDR3 + M_MM for Q/K/V/O |
| **RMS Norm** | 1.6M | **8.6%** | 16 | 2× per layer + final norm |
| Layer setup | 0.07M | 0.4% | 0.7 | VRAM zero-fill, residual copy |

### Observations

1. **Vision is attention-dominated (48%)** — 12 layers of self-attention with head projections dominate
2. **Vision layer norm is 30%** — 25 norm operations at 3.4ms each, driven by expensive V_RED_SUM (20 cycles on Artix-7)
3. **Conv2d im2col is 21% of vision** — patch embedding done entirely with vector ops, matrix unit idle
4. **Text decoder is FFN-dominated (48%)** — 3 large M_MM operations (gate, up, down projections: 576→1536→576)
5. **QKV projection is 14% of text** — weight loading from DDR3 is a significant fraction
6. **Connector is trivial (0.2%)** — single linear projection, not a bottleneck

## Optimization Opportunities

Ranked by estimated cycle savings on Nexys A7 (with DDR3):

| # | Target | Cycles saved | % of total | Approach |
|---|--------|-------------|------------|----------|
| 1 | **V_RED_SUM in vision im2col** | ~5.4M | 11.9% | Conv2d patch embedding uses V_RED_SUM (271K × 20 cyc) for dot products. Restructure im2col to gather patches into matrix SRAM and dispatch M_MM. |
| 2 | **DDR3 prefetch stalls** | ~2.9M | 6.3% | 6,273 M_PREFETCH × 400 cyc. Prefetch scheduling, double-buffering, or weight caching in BRAM could hide latency. |
| 3 | **S_ADDI_INT loop overhead** | ~2.9M | 6.4% | Loop counter increments (2.9M calls). Hardware auto-increment or compiler loop fusion. |
| 4 | **S_MAP_V_FP in text decoder** | ~2.0M | 4.4% | 7,680 scalar→vector broadcasts × 256 cyc. Vector fill instruction or register caching. |
| 5 | **V_MUL_VV + V_MUL_VF** | ~3.5M | 7.7% | 509K calls combined. Fused multiply-accumulate (V_MAC) would halve cycles. |
| 6 | **Softmax chain** | ~1.5M | 3.3% | V_EXP + V_RECI + V_RED_MAX + V_SUB form softmax. Fused instruction reduces pipeline stalls. |

### Conv2d analysis

The vision encoder's conv2d (SigLIP patch embedding, K=16, C=3) compiles via im2col → vector reductions. The matrix unit handles linear projections inside transformer layers (M_MM at 40.7%) but **not the initial patch convolution**.

Restructuring im2col to fill matrix SRAM and dispatch M_MM would reduce patch embedding from ~5.4M cycles to ~100K cycles — a **17% improvement** in vision encoder time.

### DDR3 vs HBM analysis

On V80 with HBM2e (819 GB/s), memory prefetch is effectively free (pipelined). On Nexys A7 with DDR3, memory accounts for 5-7% of cycles. For memory-bound workloads (e.g. weight loading for large models), DDR3 becomes the primary bottleneck — this profile is compute-bound because SmolVLM2-256M is a small model.

## Memory Footprint vs Nexys Video DDR3

### Actual HBM binary sizes (MXFP8 quantised weights + activations)

| Stage | HBM binary | Notes |
|-------|-----------|-------|
| Vision + Connector (12L) | 295 MB | 12 vision layers + connector weights |
| Text Decoder (1L) | 205 MB | 1/30 decoder layers |
| **Combined (1v+1t)** | **500 MB** | Fits in 512 MB DDR3 (barely) |
| Text Decoder (30L, estimated) | ~6 GB | Full 30-layer decoder |

### The 512 MB DDR3 problem

The Nexys Video board has **512 MB DDR3**. SmolVLM2-256M at MXFP8 quantisation:
- 12L vision + connector: 295 MB
- 1L text decoder: 205 MB → 30L would be ~6 GB

A single VLM e2e pass (12L vision + 1L text) barely fits at 500 MB. **Full 30-layer
text decoding is impossible** without weight streaming — loading decoder layers one
at a time from host, running each layer, then replacing it with the next.

### Implications for the emulator

The Rust transactional emulator currently models HBM as a flat address space
(configurable size, default 16 GB). It does not model:

1. **DDR3 capacity limits** — no OOM when allocations exceed 512 MB
2. **Weight streaming** — no mechanism to swap layer weights in/out during execution
3. **DDR3 bank conflicts** — prefetch latency is modeled as constant, not address-dependent
4. **Refresh cycles** — DDR3 refresh steals bandwidth periodically

To model the Nexys Video memory constraint accurately, the emulator would need:
- A configurable HBM capacity cap (512 MB) that raises an error on overflow
- A weight-streaming ISA extension or runtime mechanism to load weights per-layer
- Bank-conflict-aware prefetch latency (address → bank → row hit/miss → cycles)

For now, the cycle profile assumes all weights fit in DDR3. The 1L text decoder
config is realistic for the Nexys Video; the 30L config is V80-only.

## Validation

### Static profiler vs Rust emulator

The ASM profiler was validated against the Rust transactional emulator (Ramulator HBM2 timing, dc_en=1) on the linear test (64×128 × 128×64 matmul, mlen=64):

| Source | Cycles |
|--------|--------|
| Rust emulator (Ramulator HBM2) | 146,935 |
| Static profiler (dc_lib_en) | 144,938 |
| **Difference** | **1.4%** |

### DDR3 bandwidth cross-references

| Source | Bandwidth | Notes |
|--------|-----------|-------|
| **Our model** | 1.0 GB/s effective | x16 DDR3-1600, ~60-65% efficiency |
| Vitis AI DPU B1152 (PG338) | 684-1017 MB/s avg | Production DDR3/DDR4 accelerator |
| Flare on XC7A100T (PMC 2024) | 3.125 GB/s | x32 bus (2× our x16) |
| ultraembedded DDR3 on Arty A7 | 400 MB/s | Open-source controller @100MHz |
| ZipCPU MIG memtest (2025) | 27.7 cyc avg read | MIG controller latency measured |
| Gemmini (DAC 2021) | 8 bytes/cycle | LPDDR4 model for ASIC |

Our 1.0 GB/s effective matches the Vitis AI DPU B1152 average range (684-1017 MB/s). The 400 cycles per 4KB M_PREFETCH derives from 4KB / 1.0 GB/s = 4μs = 400 cycles @100MHz.

### Weight streaming architecture validation

Published FPGA accelerators use the same DDR layer-by-layer streaming pattern as our LayerSwapping model:

| System | Architecture | Reference |
|--------|-------------|-----------|
| Apache TVM VTA | Tiled DRAM-to-SRAM DMA, access-execute decoupled | arXiv 1807.04188 |
| Vitis AI DPU | Layer-sequential DDR streaming via micro-coded dispatch | AMD PG338 |
| FINN | On-chip default; `load_external_weights()` for DDR streaming | finn.readthedocs.io |
| DNNWeaver | DDR3 bandwidth as tiling constraint | MICRO 2016 |
| AutoWS | Automated on-chip/DDR weight partitioning for layer-pipelined | arXiv 2311.04764 |
| FlightLLM | HBM weight streaming at 65.9% utilization | FPGA 2024 |

Host-to-SRAM direct streaming (our HostStream model) is **not standard practice** for real-time inference — USB 2.0 at 38 MB/s is too slow. It is valid for modeling what-if scenarios (e.g., PCIe at 2 GB/s) and batch/offline inference.

### Weight streaming emulator validation

All 3 memory models produce **identical VRAM output** (verified via binary diff). Only simulated latency differs. Tested on the ATen linear test (64×128 × 128×64 matmul, mlen=64):

| Model | CLI flag | Latency (ns) | Slowdown | Notes |
|-------|----------|-------------|----------|-------|
| HBM (Ramulator) | `--memory-model hbm` | 146,935 | 1.0x | Pipelined HBM2, sub-cycle prefetch |
| LayerSwap (DDR3) | `--memory-model layer-swap` | 1,120,727 | **7.6x** | DDR3 SimpleTiming replaces HBM2 — no swap penalty (46KB fits 512MB), slowdown is pure DDR3 vs HBM timing |
| HostStream (USB) | `--memory-model host-stream` | 1,869,354 | **12.7x** | 38 MB/s serialized — each 64B weight chunk costs 1,684ns |

The 7.6x LayerSwap slowdown represents the **DDR3 vs HBM access latency gap** — this is the real cost of running on Nexys Video DDR3 instead of V80 HBM. The 12.7x HostStream shows the additional penalty of streaming weights from host over USB.

## Reproducing

```bash
# Profile with Nexys A7 DDR3 timing
python3 transactional_emulator/testbench/aten/compare/asm_profiler.py \
  transactional_emulator/testbench/conv/build/smolvlm2_native_module/vision_connector/generated_asm_code.asm \
  --board nexys_a7 --mlen 256

python3 transactional_emulator/testbench/aten/compare/asm_profiler.py \
  transactional_emulator/testbench/conv/build/smolvlm2_native_module/text_decoder/generated_asm_code.asm \
  --board nexys_a7 --mlen 256

# Compare with V80 (HBM, 400MHz)
python3 transactional_emulator/testbench/aten/compare/asm_profiler.py \
  transactional_emulator/testbench/conv/build/smolvlm2_native_module/text_decoder/generated_asm_code.asm \
  --board v80 --mlen 256
```
