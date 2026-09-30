# Runtime policy and residual-capacity admission experiments

Scope: the standalone Rust analytical runtime. These experiments use captured
routing metadata; they are not native Ramulator, full-model inference, RTL or
silicon measurements. Large timing runs do not evaluate numerical tensors.
Small numerical tests replay the actual DMA addresses, SRAM reads and ascending
K commits against a seeded BF16/FP32 reference.

## Frozen comparison

M×N×K shapes: 6×4×512, two 3×4×512, and 4×4×512 + 2×4×512.
BF16 operands, FP32 accumulation, 12,288 physical multipliers, 1 ns/cycle.
Aggregate HBM: 256 B/ns, 64 ns response, 256 outstanding 32 B transactions.
The credit remains occupied through the actual private-SRAM landing write.
All cores share the same credit pool and HBM throughput budget.

Private W slots: 10 or 5+5 (4 KiB each), plus the existing 8 KiB return budget.
X: two slots/core, 12 KiB aggregate. Arena: 2 MiB aggregate, including a 4 KiB
control reservation, inputs, intermediate/output data and metadata. W/X/arena
bank counts and all arithmetic/transfer/vector timing remain unchanged.

## Three single-variable families (`--suite runtime_policy`)

Every point uses dynamic whole-expert dispatch, group G=4 and bounded
Current/Next contexts. The eight variants are reference; prefetch thresholds
1/2/4; late-binding thresholds 128/512/2048 cycles; Shared pinning. Each variant
runs under both stock and round-robin arbitration, every organization and
B2/B4/B8/B16: 192 points, each repeated twice.

* Reference: bind when a private Next context is available; prefetch its first
  Gate tile as soon as an existing private W slot and the control port permit.
* Tail prefetch: reserve Next only after all Current Gate, Up and Down requests
  have actually been accepted, and the number of fully landed tiles with
  unissued M consumers is strictly below the threshold. This is not merely the
  end of the current projection. Once reserved, a partly sent tile can finish;
  permission is not revoked midway through a DMA transaction.
* Late binding: the FIFO head stays unbound unless a legal core's estimated
  remaining time is strictly below the threshold. Both cores may be ineligible.
  Candidate comparisons use charged descriptor service. A newly eligible core
  cannot join an already paid late-binding comparison for free.
* Shared pinning: Shared uses the largest M core; ties go to core 0. No other
  task rule changes. The same rule applies to single and homogeneous machines.

Stock arbitration is now the runtime default. Its key is nominal remaining
operand-feed cycles from fully landed Current tiles, not the number of tiles.
Consumers already issued are excluded, even if their last SRAM read is still
in progress. The estimate uses existing M/N/K issue cursors and a bounded
reduction over at most ten live slots, not a scan over all expert operations.
Existing aging at 64 cycles and round-robin ties are retained. Stock is an
estimate of available weight-supported work, not proof that X, K dependencies
or SRAM ports will be ready. It never bypasses those execution checks.
Historical suites explicitly select their old arbiter.

Prediction audit freezes absolute completion time at binding; actual completion
is the final expert-output copy into its pre-reserved inbox. Final layer combine
is excluded from this per-expert audit. Error = actual − predicted; positive
means underestimation. Report signed mean, mean absolute error and maximum
positive error. The core distribution counts the legal candidate set admitted
into the charged comparison, revalidated at binding. Zero-candidate waiting
cycles are separate: no binding occurs with zero legal candidates.

## Residual-capacity admission (`--suite surplus`)

This separate cumulative experiment has eleven settings per workload/org:
reference; rule 1; rules 1–2, 1–3, and 1–4 with margins 32/64/128 cycles.
That is 132 points, each repeated twice, at the unchanged 256-credit budget.
The reference uses stock arbitration, so rule 3 is compared against that
new default. Round-robin remains available in the independent family above.

1. Compiler emits eight logarithmic Me bins per core. The lower endpoint
   minimizes the estimated wave count within each bin. The
   nominal full-tile operand service is 16 cycles; useful M tails can consume
   faster. Bandwidth share is capped by both aggregate HBM and the credit
   residence lower bound. The Little-law depth estimate includes one resident
   slot and is never below G=4, or above the physical slot count. Clipping is
   explicit in the table. These values are estimates, not guarantees of service.
   One LUT read is charged on each Current promotion. The live reservation
   ledger uses `free − max(min(D, remaining_stage_tiles) − held, 0)`, saturated
   at zero. End-of-stage protection cannot reserve nonexistent remaining tiles.
2. A core becomes due if idle, if its entire Current request stream has been
   sent, or if remaining estimate ≤ first-tile readiness estimate + margin.
   The FIFO head otherwise waits. Ownership is immutable once committed.
3. Arbitration priority at each new request selection is: Current below low
   threshold; due Next; Current below target; other lookahead. Within a class,
   existing 64-cycle aging overrides inventory order, then round-robin breaks
   ties. Inventory is *accepted bytes retained by Current*, counting in-flight
   and landed bytes once. Arrival does not increment it again. Final operand
   read retires the tile. Allocation checks grant borrowing permission once;
   accepted/partly sent requests retain ownership and may drain. An already
   asserted valid request is never revoked under ready backpressure.
4. One first tile of the next projection may be prefetched in an existing
   private W slot: Gate→Up or Up/activation→Down. Task/phase/tile tags survive
   the transition; the new projection inherits the same slot and any in-flight
   responses. Down compute still waits for actual Z completion. This first
   implementation does not stream an unlimited next projection or remove
   activation, output-copy or K-dependency barriers.

Strict priority only applies to new request choices. Existing lower-priority
traffic still occupies credits, landing ports and slots; this policy therefore
cannot guarantee zero interference or a minimum bandwidth under all workloads.
Known routing eliminates expert-ID prediction error, not timing uncertainty.
Tier 0 is a low-inventory risk indication, not a count of actual array stalls.

## Accounting and observations

The first family adds 32 B of explicitly budgeted policy registers. The residual
implementation reserves another 160 B for a conservatively padded LUT, cached
thresholds, inventory/selection registers and a next-phase cursor. All fit
inside the existing 4 KiB reservation: 2,944 B single / 3,936 B dual total,
leaving 1,152 / 160 B. No extra W, X or accumulator data capacity is added.
Observer histories/histograms are host instrumentation and never guide policy.
There is no synthesis result or claim that the combinational reduction meets
1 GHz timing; clock is a simulation assumption.

`metrics.csv` reports HBM bytes / 256, achieved bytes / complete layer cycles,
and measured credit residence (request acceptance through landing). The latter
includes finite W-bank service. Core no-accept cycles are correlated with the
other core's phase and front wait state. They include rate limiting, full
credits, lack of requests, and arbitration; they do not mean HBM was globally
idle. Core-front states overlap arithmetic and other cores and cannot be summed
as global latency components.

A raw 12,288 MAC/cycle peak is not the model's sustainable arithmetic rate:
a full X operand requires about 16 cycles on the fixed ports. Thus the proposed
Me≈96 crossover from raw MAC count is not a validated runtime boundary. For
fully occupied issues, the X-port roof is about 768 MAC/ns, before dependencies
and conversion/copy costs. This is a derived port limit, not a measured phase
classification; M tails and workload overlap still require simulation.

## Credit diagnostics

`--suite credit_diagnostic` scans 256/512/1024 credits for the twelve baseline
workloads/organizations (36 points). `--suite surplus_credit_diagnostic` repeats
the full 132-point cumulative ablation with 1024 credits, including regeneration
of the Compiler depth table at that credit capacity. All points repeat twice.

Expanded credits require an explicit `diagnostic_credit_expansion` flag. The
model provisions an additional 8/24 KiB of return capacity for 512/1024 credits,
and accounts another 512/1536 B of credit tags. Reports mark these points as
ineligible for equal-budget claims against the 256-credit results. Private
slots, core shapes, banks and the HBM bandwidth/latency are otherwise unchanged.
The formal no-added-SRAM ablation remains at 256 credits. More credits are a
counterfactual resource diagnostic, not a free controller improvement.

`audit_runtime_policies.py --root OUTPUT --expected-points N` checks complete
report equality, conserved work/traffic, drain, ownership and capacity, then
exports bindings, slot histograms, supply observations and arbitration tiers.
Its large timing-point validation does not claim numerical execution of the
pretrained model; that is covered only by the documented small payload tests.

## Larger batches

`check_large_batches.py` constructs B32/64/128/256 from distinct recorded
DeepSeek BFCL decode rows, retaining route IDs and scores. It records the source
hash and explicitly calls this offline rebatching, not a new full-model run.
Compiler capacity checks precede timing. An infeasible plan has no latency
result: raising SRAM silently or copying a small-batch result would invalidate
the experiment. With the current all-results-resident lifetime, these larger
points require a different storage/retirement contract before timing is valid.
