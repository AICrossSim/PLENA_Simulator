# Bounded band assignment and independent weight prefetch

This opt-in study extends the spatial-M **private-core** model. Dimensions are
M_tile x N_tile x K_tile. It does not change BF16, the native HBM preset, weight
layout, MAC latency/II, SRAM payload budgets, or per-core data-port rates.

## Four comparable configurations

All use the same finite control SRAM reservation, including the matched baseline:

|Mode|Output assignment|Weight requests|
|---|---|---|
|baseline|Existing per-output-row affinity|Existing operand-stage admission|
|band_only|Whole expert/N-band, all M rows and K segments|Existing stage admission|
|prefetch_only|Existing per-output-row affinity|Independent reserved weight slots|
|combined|Whole expert/N-band|Independent reserved weight slots|

`fabric.frontend` is optional. Omitting it preserves the previous engine's timing,
trace and service hashes. A frontend requires private SRAM, TileCohort control,
and output-block affinity. It is not an uncharged replacement for the old default.

## Band ownership and selection

One band covers all Me rows of one expert and N_tile output columns. Different
bands of the same expert can execute on different cores. Reserve every row's
FP32 accumulator vector and existing 16-byte output record before publishing a
band. All K segments remain on its selected core. Outputs remain allocated until
phase drain; no early eviction or free inter-core partial-sum movement is modeled.

The descriptor generator traverses expert/N bands in input order. A FIFO has no
priority bypass; capacity limits cause backpressure. At most 32 bands are live,
and at most ceil(32/core_count) are assigned to one core. Each resident band has
a next-ready-K index updated by the existing charged local commit actor. Host
containers represent these indices, not a free scan over all unopened bands.

Each admission compares all feasible cores. Define C_c=ceil(Me/M_c), D=ceil(K/Kt).
With configured port rates, s_c is the maximum of II, one full weight-tile read,
one full-M activation read, and one full-M accumulator RMW service. Estimate:

    work_c = D * C_c * s_c
    supply_c = now + past_source_latency_EWMA
                    + pending_weight_slots_c * ceil(weight_tile_bytes/shared_delivery_rate)
    start_c = max(now + queued_estimated_work_c, supply_c,
                  local_weight_write_free, activation_write_free, accumulator_free)
    dependency_span_c = (D-1)*max(MAC_latency+1, C_c*s_c) + C_c*s_c + MAC_latency
    finish_c = max(start_c + work_c + MAC_latency, now + dependency_span_c)

This is an online heuristic, NOT a future-completion oracle or an exact latency
prediction. Pipeline tail latency is added to the candidate finish, NOT accumulated
as serialized work once per queued band. Completed bands remove their throughput
work estimate from queued work. Initial
tile-latency estimate is 500 cycles; completed source transfers update it with
EWMA=(7*old+sample)/8 rounded upward. Equal finish estimates use tail waste then
a rotating cursor. Descriptor installation costs core_count+1+ceil(Me/8) control
cycles; these gate visibility through the SAME serialized control port.

## Prefetch and local execution

Each core's prefetch candidate window is bounded to 32 entries. Resident band
frontiers take precedence in constructing it; row-affinity mode also admits a
bounded prefix of unopened work. This prevents a long global FIFO from hiding a
resident band's next K segment. Slot tags and readiness form a bounded bitmap.
One per-core priority encoder and a shared arbiter select a request, charging
core_count+1 control cycles. The model assumes these bounded comparators; it is
not a synthesized timing/area claim. Raw candidate checks are also reported.

Across cores, request aging activates after 4x the past tile-latency EWMA without
service. Otherwise a core lacking ready pinned weights precedes one with such
weights; a rotating cursor resolves ties. This is per-core service aging, not
knowledge of future HBM completions or fixed 2:1 bandwidth partitioning.

Reserve a private W slot BEFORE submitting the request. No X stage or MAC result
credit is needed to prefetch. Existing native sector credits, shared delivery,
fanout, private write, and charged installation still gate readiness. Outstanding
identical requests can share the existing multicast transfer. Actual byte counts
are measured; changing timing/ownership can change multicast/refetch traffic.

Conservatively pin W through retirement of the whole tile's M cohort (all RMWs),
then release it for reuse. A redundant late prefetch retains its slot until the
response is installed, even if the cohort has already completed. Never cancel
storage reservations while data are in flight. A live source with no admitted
MAC tasks is progress in flight, not a deadlock. Drain includes these requests.

Operand admission looks at reserved weight tags before applying its bounded
lookahead. Otherwise returned prefetches could be hidden behind unrequested
tiles and fill every slot permanently. X uses the existing two stages per core.
Among eligible tags in the finite window, already-returned weights precede in-flight
weights, before applying the old cohort/reuse preference. This prevents prefetch
lookahead from needlessly consuming both X stages ahead of ready data.
Local issue still requires X/W ready, previous-K committed, and result credits;
it may bypass a blocked stage. Ordered per-output FP32 RMWs remain unchanged.

## Storage and control charging

Reserve 4096 B **inside the original aggregate 2 MiB accumulator budget**, split
proportionally to M with 32-byte alignment. Single reserves 4096 B; 3+3 reserves
2048+2048 B; 4+2 reserves 2752+1344 B. This reservation is identical in all four
ablation modes. It is added to per-core accumulator peaks, not to capacity.

Provisioned state: 64 B/band-window entry +32 B/weight-slot DMA tag +96 B/core
counters +64 B global state. With 32 entries and 12 aggregate W slots this is
2592 B single /2688 B dual, both within 4096 B. Old output metadata remains charged
separately at 16 B/output-row/N-tile. No extra W/X/result payload SRAM is added.

Control service is the existing issue/install/completion sum plus band admission
and prefetch selection services. Local commit and operand/MAC actors retain their
old timings. Service totals overlap execution and MUST NOT be summed as wall time.

## Analytical model and native verification

The analytical memory backend transmits 32-byte sectors at configured aggregate
bytes/cycle, returns each sector after configured fixed latency, and respects a
finite outstanding-sector limit. It is live, causal, and gates the same finite
SRAM/compute simulator. Its counters include credit stalls, bus busy cycles, and
drain. It is NOT Ramulator, not a DRAM bank/row/refresh model, and not calibrated
to the prior study's wall times. The primary reduced model uses 256 B/cycle,
28-cycle sector response latency and 256 sectors, with a 1 ns core cycle.
Latency is the upward-rounded 27.57195-cycle mean from OLD B2 single-core native
sector counters only; new policies and B4/B8/B16 wall times are not fit targets.
Bandwidth is the bridge admission ceiling (8 channels x32 B/cycle), assuming
perfect bank availability, not an observed sustainable HBM bandwidth. The original
80-cycle assumption is retained as a complete sensitivity suite. Calibration
provenance is archived in `calibration.json`.

The primary supply ceiling is min(256,256*32/28)=256 B/cycle; the 80-cycle
sensitivity ceiling is 102.4 B/cycle. These are bounds, not attained rates. Slot limits, source admission and downstream
ports can lower it. The native suite separately uses the unchanged HBM2/Ramulator
configuration. Analytical and native results must have separate source columns.

All suites use SHA-verified real DeepSeek-V2-Lite BFCL-derived B2/4/8/16 jobs and
BF16 values; compare six isolated routed/shared gate/up/down GEMMs. Router,
nonlinearity, cross-phase movement, combine and full model are NOT timed. The
sum is not layer/model E2E. Every reported point repeats twice, verifies exact
FP32/BF16 outputs, capacity, ownership/K order, sector accounting and source drain.
