# Step2: bounded packed-weight prefetch with separate operand lifetime

Step1's cohort sequencer is retained. Step2 separates the packed-weight landing slot from the decoded BF16 operand stage. A slot can accept another tile after decoding, while the earlier tile's M cohort still consumes its operand stage. The native memory controller, weight representation, matrix shapes, arithmetic order and shared compute/vector throughput are unchanged.

The split path is opt-in through `refinement.stream_ctrl.split_slot_lifetime`; `split_window` supplies W=3/6, the matched Step1 load-latency sum/count and aging multiplier 2/4/8/null. Cohort control remains the accepted default in experiment manifests; all new mechanism switches default off in the schema. `dma.reserved_byte_credits` independently selects proportional core grants, with FIFO request order within each core. Native `per_channel` remains required. `demand_aware` is excluded by the existing 44 KiB frontend budget.

## Ownership and execution

1. Admit a complete M cohort into the existing Q output records. Each N band keeps its increasing K cursor and lifetime ID. There is only one active expert per core, and projection barriers remain.
2. Populate a finite window of at most W candidate headers. Each installation consumes one cycle on the existing shared descriptor port. The 16-byte header holds an 8-byte window-entry timestamp, 4-byte N index, 2-byte band ID and 2-byte validity/dependency flags. An N band has at most one queued tile, so the band ID identifies its current K cursor. Headers are registers for the bounded comparator, not free parallel SRAM reads.
3. Select an eligible header in descriptor N/K order; an eligible header older than the calibrated threshold gets oldest-first priority. Selection and packed-slot reservation retain the existing two-cycle descriptor service. No request leaves before the whole cohort and a packed landing slot are reserved.
4. Read the same element/scale spans and native 32-byte sectors as before. Both streams must arrive before the slot is eligible for decoding. Credits remain bounded by the existing 128 entries and 8 KiB response staging.
5. Reserve one of the existing two BF16 operand stages. Read the packed tile through the existing weight SRAM port, decode through the existing shared vector unit, then write BF16 directly to that stage through the same SRAM port. Release the packed slot only after the write finishes.
6. The accepted cohort actor issues M blocks from the operand stage. `prev_k_done` still means ordered accumulator writeback completed. Prefetch may overlap pending K feedback; arithmetic may not bypass it. Release the operand stage after its final M consumer issues. Burst completion updates remain unchanged.

For the proportional arbiter, outstanding reserved bytes count native sectors committed to tile storage but not yet copied into it. Among waiting FIFO heads, grant to the minimum `used_credits / reserved_bytes`, using integer cross-products and rotating exact ties. Idle cores do not strand credits. Per-core FIFO heads prevent coroutine notification order from becoming an unmodeled memory scheduling policy. Comparator and queue links use paid frontend control state and existing fragment descriptors; no extra data port is added.

## Storage and control service

Let `P=blen`, `R=mlen`, `C=ceil(Me/Mt)`, `T=number of native weight tiles`, `I=number of M-block issues`, `B=number of cohort bursts`, and `H=sum_bursts ceil(C/8)`.

- A packed tile costs `P*R*9/8` bytes (local block8 E4M3 elements and E8M0 scales).
- An operand stage costs `P*R*2` bytes; exactly two are allocated.
- Physical weight reservation is `W*P*R*9/8 + 2*P*R*2`.
- Implicit nonoverlapping addresses are packed slot `i: [i*packed,(i+1)*packed)` and operand stage `j: [W*packed+j*stage,W*packed+(j+1)*stage)`.
- Decode-in-flight is the reserved operand destination, not a third storage allocation. Lifetime reports partition bytes into packed, ready operand, and decode destination. The sum at each transition must fit the weight budget. Per-cycle reports retain the largest **simultaneous** tuple inside each cycle, then run-length encode equal tuples.
- Accumulator control reservation is `128Q + 64(W+2) + 24ceil(Q/64) + 16W + 8 + 64 + 64 + 16W`. The last term is candidate headers; the earlier `16W` is the bounded event FIFO. For Q32: Step1 W3=4,624 B, Step2 W3=4,672 B, Step2 W6=4,960 B.
- Proportional credits add `32 B/core + 64 B shared` inside the existing DMA frontend budget. Doubling W also reserves the additional fragment descriptors; all are checked before execution.

The existing 64-byte N-band metadata can hold: N/K indices (8 B), output-record base/count and slot/stage/flags (8 B), remaining operations (8 B), lifetime ID (8 B), pending/unissued masks or counts (8 B), queue links (8 B), and spare/pointer fields (16 B). A queued timestamp used only for reporting is simulation telemetry; the scheduling age is in the separately paid candidate header. Context records and current expert/projection metadata remain the existing allocations.

One tile costs ten descriptor cycles: candidate-header installation 1, load/reservation 2, packed-arrival event 1, operand binding 2, packed retirement 2, and operand retirement 2. Therefore:

`control_cycles = sum_bands(C+1) + 10*T + I + 4*B + H`.

`sequencer_cycles = I + not_ready_checks`, on the separate issue actor. The report reconstructs every term independently from completed projections. This formula replaces Step1's `8*T` term only on the split path.

## Measurement and limits

Use `scripts/moe_stream_ctrl/step2.py`; the immutable Step1 common table freezes cohort/charged-N3 comparisons and per-case/core L. Aging is `ceil(multiplier * Step1_load_sum_ps / Step1_tile_count)`, never recalibrated after enabling the mechanism. W3/W6 run all four aging settings; shared-credit/aging-off controls separate depth from credit/aging policy effects. Each configuration executes real arithmetic twice with native Ramulator, and checks BF16, FP32/pre-round bits, unchanged HBM bytes, finite resources, drained requests and deterministic native counters.

`regress_step2.py` verifies the complete old result objects and native counters when the mechanism is disabled. The measurement envelope, executable, native library, inputs and source snapshot are hash-linked. `cycle_peaks_*.json.gz` preserves cycle-level storage evidence; raw lifetime transitions remain in each compressed full result.

Wait counters describe concurrent queues/actors and cannot be added into an exclusive wall-time decomposition. Report dispatch ownership alongside aggregate performance: faster supply may change which expert the work-conserving dispatcher assigns to a slow small core. Step2 does not add ECT dispatch, cross-expert overlap, local partial sums, short down-K issue segments or tail handoff. Step1's Me32 accumulator dependency wait of 0.519 to 1.587 us remains a Step4 target.
