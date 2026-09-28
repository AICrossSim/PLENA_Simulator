# Projection schedule and fixed reduction study, 2026-09-28

The 120-case sweep compares four bounded Compiler panel schedules and three static Matrix implementations on Mamba/KDA B1/2/4/8/16. All points include input norm, input projections, coefficient production, convolution, recurrence and output projection. Outer residual and FFN/MoE are excluded. These are compiled analytical predictions, with representative machine-code validation; not complete-model, GPU or silicon measurements.

Baseline: the previous optimized M_MM.P projection and native recurrence, one N32 panel and one reduction segment. It already includes replay, compact input slices and up-to-four-request packets. It is not an untouched or best-proven original PLENA mapping.

## Complete recurrent sublayer

Time is milliseconds for the entire batch advancing one token, at 1 GHz. The selected mapping is the minimum of this finite offline search; this is not an implemented autonomous Compiler cost-policy or a global-optimality claim.

| Model | B | Previous ms | Compiler only ms | Selected ms | Speedup | N32 panels / segments |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mamba | 1 | 2.909800 | 2.909800 | 2.253843 | 1.2910 | 1 / 4 |
| mamba | 2 | 3.047989 | 3.047989 | 2.434093 | 1.2522 | 8 / 4 |
| mamba | 4 | 3.378375 | 3.378375 | 2.750301 | 1.2284 | 1 / 4 |
| mamba | 8 | 5.297095 | 5.297095 | 4.017895 | 1.3184 | 1 / 4 |
| mamba | 16 | 9.155280 | 9.155280 | 6.648544 | 1.3770 | 1 / 4 |
| kda | 1 | 32.034199 | 32.034199 | 25.017326 | 1.2805 | 1 / 4 |
| kda | 2 | 33.324016 | 33.324016 | 26.338800 | 1.2652 | 1 / 4 |
| kda | 4 | 35.976288 | 35.976288 | 29.049808 | 1.2384 | 1 / 4 |
| kda | 8 | 55.509874 | 55.509874 | 41.660692 | 1.3324 | 1 / 4 |
| kda | 16 | 101.932709 | 96.770110 | 68.859100 | 1.4803 | 8 / 4 |

Compiler-only panel expansion helps KDA B16; the existing one-panel schedule wins the other nine Compiler-only comparisons. Four reduction segments win every complete search point. Mamba B2 selects eight panels because of memory-service ordering, with a very small advantage; this is not evidence that a larger panel is generally better.

## Stage breakdown

Entries are previous → selected milliseconds. DMA is already included in each stage. Other includes norm, convolution, gates and coefficient placement.

| Model | B | Input projections | Output projection | Recurrence | Other |
| --- | ---: | --- | --- | --- | --- |
| mamba | 1 | 2.060814 → 1.577641 | 0.786439 → 0.614198 | 0.039873 → 0.039744 | 0.022674 → 0.022260 |
| mamba | 2 | 2.101923 → 1.658862 | 0.818950 → 0.648523 | 0.081248 → 0.080517 | 0.045868 → 0.046191 |
| mamba | 4 | 2.260939 → 1.804113 | 0.863163 → 0.691389 | 0.163886 → 0.163886 | 0.090387 → 0.090913 |
| mamba | 8 | 3.474628 → 2.531566 | 1.319640 → 0.984240 | 0.324724 → 0.323507 | 0.178103 → 0.178582 |
| mamba | 16 | 5.898626 → 4.079853 | 2.248553 → 1.559217 | 0.647827 → 0.650936 | 0.360274 → 0.358538 |
| kda | 1 | 25.528919 → 19.883063 | 6.262069 → 4.891303 | 0.147879 → 0.147831 | 0.095332 → 0.095129 |
| kda | 2 | 26.349385 → 20.736985 | 6.483036 → 5.110187 | 0.295686 → 0.296460 | 0.195909 → 0.195168 |
| kda | 4 | 28.043634 → 22.557504 | 6.949745 → 5.510565 | 0.593763 → 0.592296 | 0.389146 → 0.389443 |
| kda | 8 | 42.897306 → 31.805506 | 10.639535 → 7.881953 | 1.187171 → 1.188498 | 0.785862 → 0.784735 |
| kda | 16 | 78.544555 → 52.019312 | 19.450768 → 12.901059 | 2.371378 → 2.371378 | 1.566008 → 1.567351 |

Recurrence arithmetic and traffic do not change. Small differences in recurrence/other totals come from the address-based memory timeline, including different refresh/row timing after projection. They are not additional recurrence speedups.

## Implementation and resource contract

- Compiler: reuse each K2048 input chunk across 1/2/4/8 N32 panels; retain the original schedule as a candidate. At most 64 K256×N32 weight views occupy the existing 1 MiB Matrix SRAM. No concurrent recurrent state residency is assumed while the full capacity is used.
- Matrix candidate: split the existing 256 four-by-four mini-arrays into four fixed groups. K256 operands otherwise use only one quarter of the K1024 reduction geometry. Different groups compute different N4 outputs; N32 now requires two waves instead of eight. Total multipliers remain 4096.
- Preserve the BF16 reduction order, including the serial upper-tree additions with zero and the K256 output merge. Rust keeps the original numerical column operation. Do not interpret segmentation as changing projection or recurrent precision.
- Charge two input-distribution cycles per wave, two root-selection and two collection cycles per root, finite operand-latch throughput, serial upper-tree additions, bank reads and Vector output read-modify-write. No free overlap between waves or between projection and recurrence.
- Four root payloads cost 128 B. This is additional to the existing candidate's 16 KiB weight replay, 4 KiB row transfer, 2 KiB compact inputs and 256 B output hold. The common Matrix operand latches have 8 KiB on each side. Metadata, masks, selectors, broadcast, control, area and integrated timing are not estimated by those capacity numbers.
- Common profile: 1 MiB Matrix SRAM/64 banks; 256 KiB Vector SRAM; 16 HBM2 controllers, 32 GiB modeled capacity, 256 GB/s peak; 32 read and 32 write DMA credits. BF16 weights, activations and state; recurrent FP32 update intermediates, BF16 RN commit and BF16 pairwise reduction. The same versioned BF16 rational delta producer is used on both sides. No NVFP4 runtime-decoder claim.
- WS/IS/OS reuse occurs at different levels: weights across requests, inputs across output panels, and local partial sums. No general-purpose dataflow switching fabric is introduced.

## Validation and provenance

- 24 representative machine-code cases, 53,637 output-value comparisons exact against an independent implementation of the declared arithmetic. Includes distinct requests, K/N tails, output-row crossings and B16 input-cache pressure. Numerical equality does not certify real-model long-chain quality.
- All seven cycle components agree with Rust exactly on those cases. Python and Rust share contracts and Ramulator, so agreement is not independent evidence for physical timing.
- 329 Rust workspace tests; format and clippy pass. Analytical Python: 276 passed, 9 skipped. Affected Compiler interfaces: 111 passed. Known unrelated full-Compiler failures remain documented in its existing projection note.
- Compiler implementation: `0b2f549ac9cebeb3f328b65fc8ea6f36e1c1be49`. Simulator panel implementation: `5939f07fefa14261ac2312e51de21779bdb1faf0`; segmented candidate: `c6fda247839b0b26bb44467ce5a3ff3015b84d50`.
- `sweep.csv`: all 120 total/component/traffic results. `stages.csv`: per-stage components. `selected.csv`: minima and before/after stage totals. `comparison.csv`: selected stage components and read/write traffic. `machine_validation.json`: machine evidence. `manifest.json`: common profile, source/result hashes and provenance. `frontend` is issue+scalar, not an extra exclusive cost.

## Reproduce

Follow the [checkout and environment instructions](../../../doc/l_tile_projection.md), initialize the pinned Compiler, and prepare the 16-controller memory backend. Use /tmp or another filesystem with space for traces. The results archive omits weights, raw HBM images, caches and builds.

```sh
python -m analytic_models.performance.ltile_dma --prepare "$RUN_ROOT/memory16" --controllers 16
python -m analytic_models.performance.projection_campaign \
  --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/projection-search" \
  --models mamba kda --batches 1 2 4 8 16 --stages batch \
  --panel-tiles 1 2 4 8 --segments 1 2 4 --workers 4
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" \
  python -m transactional_emulator.testbench.models.unified_service_test \
    --only panel --segments 1 2 4 \
    --runtime "$CARGO_TARGET_DIR/release/transactional_emulator" \
    --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/projection-validation"
```

For each model/batch, compare `(panel=1, segment=1)` with the minimum total over panels at S1 (Compiler only), then with the minimum over all 12 candidates. Break ties by fewer segments and smaller panels. Stage sums and profiles are checked before selection.

## What remains before freezing a paper architecture

Check the fixed mux/broadcast paths at the target clock, including their effect on original full-width GEMM. Establish the best legal original long-K M_MM and accumulation format; the current BF16 K256 reference is not proven optimal. Evaluate joint projection/recurrent residency and a bounded handoff separately. Only then add complete-model, physical area/power and task-quality claims. Segmented reduction and tiling already have prior art; the present result alone is not evidence of a novel general-purpose Matrix array.
