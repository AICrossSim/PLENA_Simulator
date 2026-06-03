# Memory Footprint & Host-Link Streaming Analysis

Reasoning notes on whether SmolVLM2-256M needs weight streaming on the PLENA
target boards, and how local memory (DDR3 / HBM) compares to the host link
(USB2 / PCIe). Summary up front: **for SmolVLM2-256M at 8-bit, everything fits
in 512 MB of onboard DDR — keep weights resident and skip weight streaming.**

## 1. Does SmolVLM2-256M fit in 512 MB?

Yes, comfortably.

| Component | Size (MXFP8, 8-bit) | Notes |
|-----------|---------------------|-------|
| Weights | **~288 MB** | 256M params × (8 bits/elem + 8-bit scale per block-of-8 ≈ +12.5%) = ~1.125 B/elem |
| KV cache | tens of MB | ~27 MB @ 1K tokens, ~106 MB @ 4K; sub-MB for the sub-64 bring-up |
| Activations | single-MB / layer | small |
| **Total** | **~300–400 MB** | fits in 512 MB with ~120–200 MB headroom |

**The "~1 GB" you might observe on a GPU is not the model — it's CUDA
overhead:** the CUDA context (~300–500 MB), cuBLAS/cuDNN workspaces, allocator
fragmentation, and framework tensors. PLENA has none of that. (For reference,
fp32 weights *would* be ~1 GB — 256M × 4 B — so int8 is a 4× cut.)

At long context the **KV cache** — not the weights — is the only thing that
eventually pressures the 512 MB ceiling.

## 2. Implication: the streaming models are not for this model

`transactional_emulator/lib/memory/src/streaming.rs` provides two models, both
designed for the case where **weights do not fit in DDR**:

- **`HostStream`** — weight reads bypass DDR and pay a fixed host-transfer cost
  (stream weights host→SRAM).
- **`LayerSwapping`** — capacity-limited DDR; a read to a non-resident layer
  pays a swap-from-host penalty (read-only weights, so eviction is free).

Since SmolVLM2-256M fits in 512 MB, you use **neither** — you use the plain
**resident-DDR** model: load all weights into DDR once, no streaming, no
swapping. The streaming models remain valuable for models that genuinely exceed
DDR (larger models, or future targets), but they are unnecessary here.

## 3. Instructions: load once, don't stream

The ISA program is small (~5–20 MB for sub-64 SmolVLM2) and **fixed/reused
across the decode loop**. With ~200 MB of DDR free after weights, load the
program **once** into DDR and fetch it locally. The host link's only job is the
one-time load (program + weights + input) and the output readback; during
compute it is idle. Continuously fetching instructions from the host adds stall
risk for zero benefit.

## 4. DDR3 vs the host link (PCIe / USB)

First, a framing point: **DDR3 is attached memory; PCIe is a transport** to host
RAM. A "PCIe fetch" pays the PCIe link latency **plus** the host's own DRAM
latency on the far end — they are not the same kind of thing.

### Raw bandwidth (config-dependent)

| Link | Bandwidth |
|------|-----------|
| DDR3 single ×16 chip (Nexys-class) | ~3.2 GB/s peak (DDR3-1600, 16-bit); ~1 GB/s effective |
| DDR3 desktop DIMM (64-bit channel) | ~12.8 GB/s/channel (25.6 dual-channel) |
| PCIe Gen3 ×16 | ~16 GB/s |
| PCIe Gen4 ×16 | ~32 GB/s |
| PCIe Gen5 ×16 | ~64 GB/s |

On raw bandwidth a *single low-end DDR3 chip* is slower than a modern PCIe ×16
link; a *full DDR3 DIMM* is comparable; and wide **HBM (V80: 819 GB/s) dwarfs**
any PCIe.

### Latency (local memory always wins)

- DDR3 access: **~15–50 ns** (directly attached).
- PCIe round-trip: **~0.5–1 µs+** (serialization + protocol + host memory
  controller) — roughly **10–50× higher latency**.

For random / latency-sensitive access (weight & activation fetch during
compute), local DDR3 wins regardless of the bandwidth headline. PCIe is good for
**bulk one-time loads**, not for being in the per-cycle fetch path.

## 5. For the PLENA target boards specifically

| Board | Local memory | Host link | Verdict |
|-------|--------------|-----------|---------|
| **Nexys Video** (Artix-7) | DDR3 ~1–3.2 GB/s | **USB2 ~38 MB/s** (FT2232H) — *no PCIe* | local DDR3 **~30–80× faster** + far lower latency |
| **V80** (Versal) | HBM **819 GB/s** | PCIe Gen5 ×16 ~64 GB/s | HBM **~13× faster** |
| **Custom Artix-7 + PCIe 2.0 ×4** | DDR3 ~1–3.2 GB/s | **PCIe 2.0 ×4 ~1.6–1.8 GB/s** | *comparable* — streaming becomes viable (§5.1) |

For **Nexys Video** and **V80** the local memory beats the host link on bandwidth
*and* latency — which is exactly why "keep weights resident, don't stream from
host" is the right call there. (Note: the host link on Nexys Video is **USB2, not
PCIe**; PCIe is the V80 path, and the V80's 32 GB HBM makes the whole model a
rounding error.) The custom Artix-7 board below is the interesting exception.

### 5.1 Custom Artix-7 with PCIe 2.0 ×4 — streaming flips to viable

Put the *same* Artix-7 on a custom board with a **PCIe 2.0 ×4** host link and the
calculus changes:

- **The link:** PCIe 2.0 = 5 GT/s/lane; ×4 = **20 GT/s raw** ("20 Gbps"). After
  8b/10b encoding (×0.8) → **16 Gbps = 2.0 GB/s** usable; after TLP/protocol
  overhead (~10–20%) → **~1.6–1.8 GB/s effective payload.**
- **vs local DDR3** (single ×16 DDR3-1600: ~3.2 GB/s peak, ~1–2 GB/s effective):
  PCIe 2.0 ×4 is now **at parity with — or faster than — the local DDR3.** That is
  ~40× the USB2 figure; the host link is no longer the weak point.

What that unlocks:

1. **Weight streaming over PCIe becomes viable — even preferable.** The
   `HostStream` model (stream weights host→SRAM, bypassing DDR3) runs at ~1.6 GB/s,
   matching or beating the modeled DDR3 effective (~1.0 GB/s). On this board the
   streaming models (`HostStream` / `LayerSwapping`) are the *right* tool — the
   opposite of the USB2 verdict.
2. **Capacity is the real win:** streaming from host RAM (GBs) removes the 512 MB
   ceiling → run models **larger than 512 MB** on this Artix-7. (SmolVLM2-256M
   doesn't need it; a larger LM would.)
3. **Best-of-both split:** weights (read-only, sequential, prefetchable) streamed
   over PCIe; activations/KV (read-write, random) in local DDR3.

The one caveat is **latency, not bandwidth:** PCIe round-trip ~µs vs DDR3 ~tens of
ns. Hide it with **weight-prefetch FIFOs** — weight access is sequential, so this
works well (the board config already exposes `hbm_m_prefetch_amount` / prefetch
knobs). Latency would only bite random access, which weights are not.

**Perf sketch:** streaming 288 MB at ~1.6 GB/s ≈ 180 ms/token → ~5–6 tok/s, vs
~3–4 tok/s for DDR3-resident (at the config's 1.0 GB/s) — a modest speedup *plus*
the bigger-model capability. Stream the **weights** (large, sequential,
bottlenecking), not the instructions (still tiny → load once into DDR).

## 6. The real bottleneck is DDR bandwidth

On Nexys, effective DDR ≈ **1.0 GB/s**, so reading ~288 MB of weights per token
≈ **~290 ms → ~3–4 tok/s**. That is the performance wall, *not* the host link.
For contrast, streaming those weights from the host over USB2 (38 MB/s) would be
**~7.6 s/token — ~25× worse** — which is the quantitative reason resident DDR is
the only sane choice. (On V80, HBM at 819 GB/s removes the wall entirely.)

## 7. Caveat + how to validate in the emulator

The emulator currently **models DDR as 128 GiB** (`modeled_size_bytes` in
`transactional_emulator/testbench/board_configs/nexys_a7.yaml`), not the physical
512 MB — so it does **not** enforce the real capacity today.

To turn the "fits in 512 MB" argument into a measured result, run SmolVLM2-256M
through **`LayerSwapping` with `capacity = 512 MB`** and confirm **zero swaps**
(every layer loads once and never evicts). If it ever *does* swap at some
context length, that is the KV cache hitting the ceiling — the genuinely
interesting limit.

## Bottom line

For SmolVLM2-256M at 8-bit: keep everything resident in 512 MB DDR and skip
weight streaming. Caveats: (a) the Nexys host link is **USB2, not PCIe**;
(b) load the instruction program into DDR once rather than streaming it; and
(c) the performance wall is **DDR bandwidth (~3–4 tok/s on Nexys)**, not the
host link.

The exception is a **custom Artix-7 board with PCIe 2.0 ×4** (§5.1): there the
host link (~1.6 GB/s) reaches parity with local DDR3, so weight streaming
becomes viable — and worthwhile mainly because it lifts the 512 MB capacity
ceiling (run from host RAM), with weights over PCIe and activations/KV in DDR3.
