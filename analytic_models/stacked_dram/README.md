# Stacked 3D DRAM model

This package models a 3D-stacked DRAM attached to a PLENA chip and adds a
memory term to PLENA's analytic latency estimate. It answers questions such as
"how do TTFT and TPS change if the HBM is replaced by a stack of *N* DRAM
layers with *M* of them vertically connected?".

`PerfModel` (`analytic_models/performance/perf_model.py`) counts pipelined
instruction cycles and charges HBM prefetch/store instructions one cycle, so
`llama_model.py` assumes that memory always keeps up. `estimate_decoder_latency`
keeps that compute term unchanged and adds the time to move each stage's DRAM
traffic through a chosen memory system.

## Provenance

The memory mechanisms are ported from DeepStack
([arXiv:2604.04750](https://arxiv.org/abs/2604.04750), MICRO 2026;
[tile-ai/DeepStack](https://github.com/tile-ai/DeepStack) at `8509061`). PLENA
does not depend on DeepStack; the equations were re-implemented here, and each
module names its source:

| Here | DeepStack source |
|---|---|
| `StackedDramConfig`, `DramTimingConfig` | `DramInterfaceConfig`, `DramTimingConfig` in `src/deepstack/mosaic/arch/custom_profile.py` |
| `dram_connectivity_efficiency`, peak/effective bandwidth, capacity | `dram_connectivity_efficiency`, `ConfigurableStackedGpu.update_ddr` in the same file |
| `BufferingPolicy` (Little's law), `ThermalPolicy` | `DramDsePolicy`, `apply_littles_law`, `compute_thermal_freq_scale` in `src/deepstack/mosaic/dse_space/case_study_dram_layer/dram_layer_config.py` |
| wave quantisation (`apply_wave_quantization`) | DRAM wave rounding in `src/tilesight/tilesight/fused_op_dtype_wave/matmul_fused_op_new_api_wave.py` |
| `DramEnergyConfig` | DRAM terms of `ChipEnergyConfig` in `src/deepstack/mosaic/cost/energy.py` |
| `"stage-roofline"` overlap | per-operator resource maximum in `src/tilesight/tilesight/fusion_support/hete_post_process_single_op.py` |

GPU-specific inputs were dropped (L2 bandwidth multiplier, uncached
utilisation, per-SM shared memory). DeepStack's per-SM buffering becomes a
per-*requester* buffer: an independent DMA stream and the on-chip SRAM reserved
for it.

**No hardware values ship with this package.** Every field is required and
must describe the caller's own design. The two profiles in `examples/` contain
conspicuously fictional numbers that only illustrate the format; most of them
are reused from DeepStack's public interface tests.

## Memory systems

`MemorySystem` is the small interface the estimator uses: usable bandwidth,
capacity, a compute-clock scale, per-transfer quantisation and optional energy.

* `StackedDramModel`: a `StackedDramConfig` plus optional policies.
  * Peak bandwidth is `connected_layers * channels_per_connected_layer *
    bytes_per_channel_transfer * transfers_per_memory_clock *
    memory_frequency_hz`, or `connected_layers *
    direct_peak_bandwidth_per_connected_layer_bytes_per_s`.
  * Connectivity efficiency is 1 while at most half of the layers are
    connected, then falls linearly to the fully connected efficiency, which is
    either given or derived from `bank_timing` as `row_read_cycles /
    (row_read_cycles + recharge_cycles)`.
  * Capacity is `total_layers * capacity_per_layer_bytes`.
  * With `apply_wave_quantization`, each transfer is rounded up to whole waves
    of `transaction_bytes * channels_per_connected_layer`.
  * `buffering` caps the bandwidth by Little's law: each of `requesters` streams
    needs `(bandwidth / requesters) * latency * buffering_factor` bytes in
    flight.
  * `thermal` scales the compute clock with the stack height.
  * `energy` gives read and write pJ/bit.
* `FixedBandwidthMemory`: a sustained bandwidth and capacity, for example an
  HBM baseline taken from a datasheet.

`with_layers(total, connected)` re-evaluates a stack at another height, which
makes layer sweeps one line.

## Profiles

Profiles are JSON in SI units. Unknown keys are rejected, and loading records
the file path and SHA-256 in `provenance`. See `profile.py` for the schema and
`examples/` for complete files.

```json
{
  "schema_version": 1,
  "kind": "stacked_dram",
  "name": "my-design",
  "provenance": {"source": "where these numbers come from"},
  "dram": {
    "total_layers": 0, "connected_layers": 0, "channels_per_connected_layer": 0,
    "bytes_per_channel_transfer": 0, "transfers_per_memory_clock": 0, "memory_frequency_hz": 0,
    "capacity_per_layer_bytes": 0, "transaction_bytes": 0,
    "bank_timing": {"row_bytes": 0, "sector_bytes": 0, "sector_cycles": 0, "recharge_cycles": 0,
                    "round_trip_latency_cycles": 0, "latency_clock_hz": 0},
    "apply_wave_quantization": true
  },
  "buffering": {"requesters": 0, "buffer_bytes_per_requester": 0, "buffering_factor": 0},
  "thermal": {"resistance_base_c_per_w": 0, "resistance_per_layer_c_per_w": 0, "baseline_layers": 0,
              "design_power_w": 0, "static_power_w": 0, "dynamic_power_exponent": 0},
  "energy": {"read_pj_per_bit": 0, "write_pj_per_bit": 0}
}
```

The zeros are placeholders, not defaults; validation rejects them.
`buffering`, `thermal` and `energy` are optional.

## Latency estimate

`estimate_decoder_latency(shape, perf, precision, memory, ...)` returns TTFT,
TPS and per-phase compute time, memory time, memory-bound time, DRAM bytes and
optional DRAM energy, plus weight and KV-cache footprints checked against the
memory capacity.

* The compute term calls `PerfModel` exactly as `llama_model.py` does,
  including its decode factor of two. With unbounded bandwidth it reproduces
  `llama_model.py`, and a test checks this. `head_dim` is read from the model
  config when present.
* The memory term is first-order:
  * a block's weights are read once per forward pass;
  * decode reads the layer's KV cache once per token and writes the new K/V;
  * prefill attention re-reads K/V once per MLEN-row query tile;
  * activations reach DRAM only where `PerfModel` spills them past Vector SRAM.
* Storage precisions come from `[ANALYTIC.PRECISION]` in `plena_settings.toml`.
* `overlap_policy="stage-roofline"` takes the slower of compute and memory per
  stage; `"serial"` adds them.
* Single chip, dense decoders only. Mixture-of-experts configs are rejected for
  now, and the LM head is left out unless `include_lm_head=True`, as in
  `llama_model.py`.

## Command line

From the repository root (or through the `just` recipes):

```bash
# Bandwidth, efficiency, capacity and clock scale over stack heights
python -m analytic_models.stacked_dram describe \
    --profile analytic_models/stacked_dram/examples/fictional_stacked_dram.json \
    --total-layers 4,8,12 --connected-layers 2,4,8,12
just stacked-dram-describe analytic_models/stacked_dram/examples/fictional_stacked_dram.json --total-layers 4,8,12

# Memory-aware TTFT/TPS with a stacked-DRAM profile, a fixed-bandwidth profile,
# or an ad-hoc fixed bandwidth
just stacked-dram-estimate llama-3.1-8b analytic_models/stacked_dram/examples/fictional_stacked_dram.json
just stacked-dram-estimate llama-3.1-8b analytic_models/stacked_dram/examples/fictional_stacked_dram.json 4 2048 1024 --total-layers 8 --json
python -m analytic_models.stacked_dram estimate --model llama-3.1-8b \
    --model-lib PLENA_Compiler/doc/Model_Lib --config plena_settings.toml \
    --isa-lib analytic_models/performance/customISA_lib.json --fixed-bandwidth-gbs 1000

# Tests
just test-stacked-dram
```

## Citation

```bibtex
@inproceedings{mo2026deepstack,
  title     = {DeepStack: Facilitating Co-Design Exploration of 3D DRAM-Stacked Accelerators for Distributed LLM Inference},
  author    = {Zhiwen Mo and Guoyu Li and Hao (Mark) Chen and Yu Cheng and Zhengju Tang and Qianzhou Wang and Lei Wang and Shuang Liang and Lingxiao Ma and Yuxiao Guo and Wayne Luk and Jilong Xue and Hongxiang Fan},
  booktitle = {59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year      = {2026},
  note      = {arXiv:2604.04750}
}
```
