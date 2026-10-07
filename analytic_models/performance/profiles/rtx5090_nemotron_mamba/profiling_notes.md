# Nemotron 3 Nano Mamba layer profiling notes

## Conclusion first

The official Nemotron 3 Nano Mamba-2 mixer fast path runs correctly on the RTX 5090 with true model shapes and random BF16 weights. Standalone median latency is 0.498/0.752/1.213/4.046 ms for B1 prefill S=128/512/2048/8192, and 0.234/0.249/0.245/0.253 ms for one-token decode B=1/4/8/16. All outputs had the expected shape, all values were finite, and preallocated state storage kept the same CUDA pointers.

This is a **BF16 single-mixer microbenchmark**, not a BF16 30B-model result. The optional full-model NVFP4 baseline was not run: neither vLLM nor SGLang is installed, the official repository is about 19.4 GB, and the user quota had only about 9 GB remaining. Downloading it into the 31 GB tmpfs would leave too little safe room for the engine, staging, and runtime. No empty or invented full-model CSV was created.

## What was measured

- Official class: `NemotronHMamba2Mixer` from pinned model revision `ce1b118...`.
- Scope: one mixer only. The outer block residual/RMSNorm, MoE, attention, embedding, LM head, tokenizer, and checkpoint weights are excluded.
- Shape: hidden 2688; projection 10304; 64 Mamba heads; head dim 64; state dim 128; 8 groups; convolution width 4; chunk 128.
- Parameters: 38,742,208 parameters, about 73.9 MiB in BF16.
- Prefill state: conv `[B,6144,4]`; SSM `[B,64,64,128]`.
- Decode: one 128-token prefill initializes the state, then 20 one-token warmups occur before 100 measured updates.
- Timing: one CUDA event pair per iteration, explicit synchronization before measurement and after all event records. Peak VRAM is PyTorch allocated memory after warmup, including resident layer, input, state, output, and measured temporaries.

## Latency and memory

| Phase | B | S | Mean ms | Median ms | P95 ms | Peak MiB | Incremental peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 128 | 0.500 | 0.498 | 0.512 | 94.5 | 9.1 |
| prefill | 1 | 512 | 0.768 | 0.752 | 0.828 | 120.9 | 31.6 |
| prefill | 1 | 2048 | 1.214 | 1.213 | 1.234 | 220.9 | 115.8 |
| prefill | 1 | 8192 | 4.085 | 4.046 | 4.173 | 624.7 | 456.6 |
| decode | 1 | 1 | 0.236 | 0.234 | 0.243 | 84.1 | 0.05 |
| decode | 4 | 1 | 0.258 | 0.249 | 0.284 | 87.4 | 0.19 |
| decode | 8 | 1 | 0.247 | 0.245 | 0.261 | 91.9 | 0.38 |
| decode | 16 | 1 | 0.266 | 0.253 | 0.316 | 100.7 | 0.75 |

Batching is effective for decode: B=8 and B=16 take only about 1.04x and 1.08x the B=1 median while updating 8x and 16x as many streams. Prefill scales close to linearly once the fixed launch cost is amortized: S=8192 is 3.34x the S=2048 median for 4x as many tokens.

## Real fusion boundaries and kernels

The requested NVTX names mark existing official calls; no official kernel was split. In inference prefill, `mamba_state_update_output_fused` is the official `mamba_chunk_scan_combined` call and launches five Triton kernels. In decode it is the official `selective_state_update` call and launches one `_selective_scan_update_kernel`. The final output projection is a separate official linear operation and remains under `mamba_out_projection`.

Measured Systems kernel time inside the state/scan range:

| Case | Kernel(s) | Total kernel time |
|---|---|---:|
| prefill B1/S2048 | chunk scan, chunk state, state passing, BMM chunk, chunk cumsum | 89.3 us |
| decode B1 | `_selective_scan_update_kernel` | 1.8 us |
| decode B8 | `_selective_scan_update_kernel` | 6.3 us |

The state/scan is therefore only about 7.4% of standalone B1/S2048 mixer latency, 0.8% of B1 decode, and 2.5% of B8 decode. The projections and remaining operations dominate this single-layer latency.

## Nsight Compute metrics

Nsight Compute 2026.2.1 recognizes the GB202. The installed system NCU 2022.4.1 did not and was not used. Blackwell does not expose the requested `dram__bytes_read.sum` and `dram__bytes_write.sum` base names; after `--query-metrics --devices 0`, the direct replacements `dram__bytes_op_read.sum` and `dram__bytes_op_write.sum` were collected. This substitution is explicit in every normalized NCU CSV.

| Case / kernel | DRAM read | DRAM write | SM cycles avg | SM throughput | DRAM throughput |
|---|---:|---:|---:|---:|---:|
| decode B1 / selective update | 1,067,008 B | 0 B | 9,634 | 13.67% | 15.96% |
| decode B8 / selective update | 8,493,824 B | 0 B | 22,346 | 46.69% | 53.12% |
| prefill / chunk scan | 44,069,120 B | 3,072 B | 141,941 | 28.59% | 34.62% |
| prefill / state passing | 33,591,552 B | 1,792 B | 56,730 | 4.76% | 85.41% |

The zero DRAM write count is a measured physical-DRAM counter value, not a missing value: state stores can remain in cache during the profiled kernel. NCU replay timings are intentionally not used as latency measurements. NCU ran with `--clock-control none` to avoid changing global GPU clocks.

## Reproduction and caveats

- Main scripts: `scripts/setup_nemotron3_mamba_5090.sh`, `scripts/setup_nsight_compute_2026_2_1.sh`, `scripts/profile_nemotron3_mamba_5090.py`, and `scripts/run_nemotron3_mamba_5090.sh`.
- Dependencies and exact CUDA-extension wheel URLs are pinned in the setup script and `requirements-profile.txt`; no container was used.
- Systems reports trace warmups too, but the supplied measured kernel summaries select scan range index 20 for prefill and state-update range index 21 for decode (zero-based). This avoids reporting Triton compilation/autotuning as steady state.
- The standalone latency matrix and Systems reports were collected with no compute process other than desktop graphics. A separate `LevistoneIndoorSwarm` process started later, before NCU collection, and was idle at the environment snapshot. The NCU target process was correctly filtered, but those counter files should be considered concurrency-qualified unless re-collected after that process exits.
