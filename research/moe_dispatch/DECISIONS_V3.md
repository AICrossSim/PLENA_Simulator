# Implementation decisions for supply-first v3

Specification: `v3_reference/TASK_SUPPLY_FIRST_V3.md`, supplied 2026-10-02.
The reference is copied without altering its frozen numerical tables. Changes
to that reference are recorded in its own CHANGELOG.

## D001 — streamed Down dependency

Original: section 3.4 says the first Down issue waits for complete U_d, while
streamed Z schedules early Down segments before U_d is complete.
Implementation: early Down segments issue only the main product. Fused rank
contributions are placed in the final segment, after U_d completion; excess
ranks use charged lane-only issues. All output partial sums remain live.
Impact: streamed mode has additional rank-tail issues and dependencies; these
are reported rather than silently using the full-Z lane capacity.

## D002 — legacy compression-only timing oracle

Original: `weight_bytes_scale=1/3.481` shrinks HBM bytes without changing the
legacy BF16 SRAM path. There is no compressed payload format in joint_v1.
Implementation: only this explicitly nonphysical oracle scales aggregate wire
service and effective wire-credit occupancy by the reciprocal ratio. Logical
32-B BF16 copies, SRAM reads/writes, task ownership and MACs are preserved.
Results report both logical moved bytes and scaled wire bytes. Integer grants
round down and credits round up. This is not a P1/P2 execution or an equal-area
hardware point. Default scale=1 retains the old timing exactly.
Impact: oracle landing traffic intentionally remains BF16 and may become the
bottleneck; transaction alignment is an acknowledged scalar approximation.

## D003 — available precision evidence

Default formats and ranks remain the task's provisional v3 format until Q1
passes its prespecified real-data targets. Synthetic replay checks establish
arithmetic/protocol correctness only and never freeze accuracy claims.
Any missing calibration/evaluation samples or unavailable hardware are recorded
by Q0 and prevent reporting the corresponding numerical milestone as passed.

## D004 — measurement interval

Legacy M0 observers add no events, resource acquisitions or modeled cycles.
HBM/core mutually exclusive states include the final ordered combine tail.
Issue samples are bounded; aggregate stage sums and issue-interval histograms
cover all completed issues. Existing legacy timing fields remain unchanged.

## D005 — initiation interval and result latency

Original: the first-order budget uses one initiation per cycle; it does not
specify a zero-latency dot-product result. Implementation: default dot-product
latency is 20 cycles, matching the preserved PLENA analytical engine. Different
outputs may issue each cycle; the next K segment of the same output waits for
its preceding ordered update. `ideal_onchip` removes supply-port costs but
preserves this latency. `dot_latency=1` is a labelled sensitivity oracle only.
Impact: a throughput bound of ceil(Me/M) cycles is not a completion-time promise.

## D006 — installed capacity versus live occupancy

Original: SRAM budgets depend on maximum T_chunk, not the current batch size.
Implementation: P2/L8 installs T_chunk=128; P0, P1 and L16 install T_chunk=96.
All Current/Next contexts, WOR/XOR capacities, accumulator partitions, decoder
storage and control reserve are charged at this maximum. Actual occupancy is
reported separately, and is checked throughout the simulation. Equal-port
comparisons install 18 aggregate WOR slots and 2,048 B/cycle aggregate P1
decode throughput for both one- and two-core organizations. A P1 decoded
main-weight buffer does not absorb B/U rank operand storage for free.
Impact: b2 cannot obtain an artificial area advantage by shrinking installed
buffers; port equality and capacity equality are separately visible.

## D007 — ordered compensation and atomic combine

Implementation: full-Z Down output groups with excess ranks stay local until
their charged rank tails complete. SiLU and combine cannot consume incomplete
corrections. A shared combine actor holds an atomic read-modify-write lease
from its first read to its last write; its 8-byte owner/valid state comes from
the charged control reserve. Contenders wait in C6. Functional arithmetic
follows the actual issue, vector-completion and combine-completion events.
Impact: two experts targeting the same token cannot silently lose a contribution.

## D008 — causal recovery

Implementation: at most one Next task per layer may be stolen before it produces
any computed outputs. Accepted prefetch is not cancelled for free: old responses
keep their original owner/address, enter ingress, and write the landing banks;
their leases drain before the old reserved addresses are released. Unsent
reservation may be released immediately. The new owner issues its own charged
reads. Old accepted bytes are reported as waste and new reads as retake traffic;
unique useful bytes exclude both. Ownership and the tile plan move atomically;
the decision and dataflow switch are charged. Bind-time predictions remain
immutable in the error report. Helper
offload executes real A/B issues on the second core, waits for that core's
current cold expert, and charges X/Z and per-column FP32 return copies. A
one-core offload configuration is explicitly unsupported rather than deadlocked.

## D009 — numerical qualification

The Q0 calibration and internal numerical-validation captures subdivide only
development requests. Their hashes are disjoint from each other and from the
architecture timing holdout. They are continuous BF16 prompt forwards at layers
2, 13 and 26, not reused token IDs from a different tokenizer. Actual checkpoint
tensors are checked by shard and hash. The default W4 format remains a timing
candidate until all three layers pass the frozen quality targets; neither a
seed replay nor a partial pilot supplies evidence of model-quality preservation.

## D010 — first-order forecast contradicted by M0

Measured legacy compression-only and credit-only oracles are much faster than
the supplied single-core forecast. The empirical 30.4 ns/issue fit is a
throughput correlation, not a fixed serial descriptor charge. Legacy operand
reads overlap each other, and independent outputs overlap arithmetic and
ordered accumulator updates. Therefore neither the fully serial-stage
hypothesis nor a fixed 30-cycle control-only hypothesis describes its code.
The recorded timestamps and counterfactuals are retained, and N1's original
threshold is not relaxed to fit this finding.

## D011 — switchable-core IS legality

Original: switchable cores choose IS whenever Me <= their physical row count.
An arbitrary Me<=4 cap would weaken single-6/8. Implementation: their fixed
accumulator arena is provisioned for full-M routed IS, and IS is selected when
both Me<=M and Me*max(2F,H)*4 bytes fit that arena; otherwise they use WS.
Shared has twice the routed projection width. Provisioning worst-case full-M
Shared IS as a separate fixed arena would exceed T_chunk=128's ceiling by
16,704 B for single-6 and 65,856 B for single-8. The fallback is therefore a
reported footprint check applied to every switchable organization. Specialized
streaming cores retain their task's Me<=4 physical limit.
Impact: routed Me5/6 and Me5--8 can use legitimate IS on single-6 and single-8;
no test point changes SRAM capacity to admit a preferred schedule.

## D012 — operand reservation precedes transfer

The new numerical stress test revealed a finite-buffer deadlock when WOR space
was acquired only after a tile finished reading. A partially read Current tile
could be blocked while later reads filled every WOR slot. WOR ownership now
starts before the first byte is read and lasts through decode and the last M
use. In-read, decoded and in-use tiles all count in the peak occupancy check.
No extra slots, bytes or physical ports are introduced by this correction.

## D013 — physical Down scratch and partial combine

Keeping every Down column live would require an unbudgeted private Me-by-H
FP32 Y. Full-Z WS therefore completes one N band, including residual rank
tails, and drains it before reusing the scratch. Streamed Z drains each
K-segment or rank-tail N band directly into the charged shared FP32 combine
buffer. Its atomic actor charges every read and write. No private full Y is
introduced. The host full-output array is validation-only arithmetic state.
Interexpert partial additions can change FP32 rounding. Hardware gold consumes
the actual ordered sources and valid row ranges, then scatters in the actual
atomic completion order; an FP64 quantized reference separately checks the
algorithm contract. All K/rank contributions and final drains are checked.

## D014 — low-rank backing and two-context execution

Gate/Up A prepasses iterate over bounded rank groups outside ascending K,
then drain and store BF16 U. This avoids silently retaining every rank column
in the smaller local register arena. U_d uses the charged shared activation
arena for FP32 partial RMW and BF16 conversion before B consumes it. Current
and Next may alternate complete bounded groups only when both accumulator
footprints and their nonaliasing Z/U allocations fit the frozen arenas.
Otherwise Next stays prefetched until admission becomes legal. Context and
dataflow switches are charged, and allocated and peak-live bytes are reported
separately. Forced streamed IS uses the segmented production/consumption order
rather than aliasing a full-Z schedule into a two-segment ring.

## D015 — rank operands have physical backing and charged reads

An uncharged U lookup would falsely improve the fused-lane result. BF16 U now
uses a compiler-defined rank permutation: the ranks assigned to a K segment are
contiguous, followed by the residual rank tails. FP32 U_d partial backing keeps
logical rank order; conversion writes the physical BF16 layout. Every external
U read contends on the shared activation banks. Two bounded local rank-input
cache entries per core occupy banked SRAM inside the existing 16 KiB control
reserve; their capacity and local port widths enter the hardware ledger. This
does not add free registers or increase the total storage ceiling. Address tests
check both the physical permutation and the exact written/read bank words.
Pre-correction performance numbers are development diagnostics and must not be
combined with the final campaign.

## D016 — recover the original SWE prompt, rather than substitute another input

The archived routing capture identified the original canonical prepared SWE
file, but that file was missing. Reconstruction from the pinned official source
produced exactly the archived SHA256, including all 2,294 requests. Actual BF16
CPU prompt-prefix forwards at layers 2, 13 and 26 then supplied the development
and held-out continuous mixed windows. The final mixed bundle has all three
datasets; its manifest supersedes the pre-capture availability flag. These are
layer inputs, not full-model inference timing. Layers and adjacent prefixes of
the same prompt are correlated and cannot be treated as independent requests.

## D017 — numerical and runtime rank allocation must be the same policy

Q3 selects one common rank per expert and clips it to each projection's capacity.
The runtime uses the same five common-rank candidates, summing the three
projection error-energy and byte costs before selecting the expert rank. An
independently selected rank per projection is a different policy and cannot
inherit Q3's accuracy evidence. The calibrated energy table is BF16 with
round-to-nearest-even, the byte table is uint32, and all entries and metadata are
charged. Lambda is fitted to the mean actual development-window byte cost, not
to an averaged gate vector; it updates from the completed previous layer and
resets to lambda0 outside the specified range. Shared retains its fixed rank.

## D018 — resume a complete bounded group, rather than a ready first tile

The final multi-K numerical stress test exposed a deadlock: a suspended context
held its byte quota in future weights; the new context could issue its ready
first tile but could not admit the rest of its group. Switching was only legal
at group boundaries, so neither context could release the resources needed by
the other. A resume therefore requires the other context's complete next group
to be ready or already loaded and its required WOR capacity to be reservable.
This legality rule applies to both alternating and starvation-driven switching.
It adds no capacity, cancellation, or free transfer and is checked by the seeded
H=544/F=640/r=16 two-expert streamed-Z regression.

The default policy switches at a complete-group boundary when Current cannot
advance and Other can, with bounded aging. Its default age is
max(configured HBM latency, group tile count * ceil(Me/M)); an explicit age
override is a separately reported sensitivity. This is a fixed runtime rule
using known dimensions, not a hardware configuration tuned for each workload.
The original alternating rule remains a charged diagnostic for all organizations.

## D019 — a suspended context cannot retain an incomplete WOR frame

The captured SWE T128 development window exposed an additional operand-frame
deadlock on the flexible 4+2 comparator. A parked context retained a future
tile while the resumed context changed from a one-tile prepass group to a
two-tile Gate/Up group. The second tile had returned but could not obtain a WOR
slot. Current's complete immediate group now reserves its required slots before
lookahead loading, and switching requires a clean operand frame. With two live
contexts, future groups cannot occupy unreserved WOR slots. DMA and landing
lookahead still overlap; compute contexts do not receive additional operand
frames. The exact failed workload, failure dump and repaired receipt are retained.

## D020 — ablations remove mechanisms, rather than inject approximate penalties

The initial `joint` placement branch aliased earliest-finish, so it could not
support N5. The corrected policy ports the prior bounded critical-anchor,
maximum-cardinality pair and aging algorithm. A charged proposal preserves both
assignments and revalidates legality when each is installed. It sees the same
eight descriptors and finite supply resources as the other policies.

`prefetch_quota=false` now removes future-group and Next request admission while
allowing the complete immediately needed Current group to progress. It does not
replace finite quotas with unbounded capacity. `pipeline_supply=false` serializes
the next operand access/issue behind the preceding MAC and accumulator completion;
it suppresses X lookahead and has no synthetic issue-gap penalty.

Non-inline SiLU uses an actual bounded producer/consumer schedule. WS spills a
group into a second FP32 slice charged to the same accumulator budget; IS reads
its existing FP32 Gate/Up backing. No extra whole-expert array is allocated. This
is a group-local nonfusion control, not the legacy whole-expert protocol. Its
transfers, finite ports and effects on concurrent context admission are reported.
WS performs all three finite transfers: read the completed source RF group,
write the scratch slice, then read that slice for SiLU. It cannot obtain a free
copy of previously completed producer values. IS needs one read of its existing
backing. Copy-source, scratch-write and scratch-read counters remain distinct
from default inline reads.

## D021 — vector inputs and rank-prepass drains are physical reads

The default IS SiLU originally waited for accumulator completion without reading
the retained Gate/Up values through a physical port. The RF/SRAM sources of
Gate/Up vector work and non-U_d BF16 rank stores must now incur finite private
accumulator reads. Completion ordering alone does not supply bytes. U_d retains
its separately charged shared FP32 activation backing and is not read twice.
The non-inline WS path additionally charges its scratch write. The same source
access rules apply to `none`, fused lanes and every legal alternative. All final
timing points are rerun on the corrected binary; earlier timing is diagnostic.

The private port is fixed by the physical core: max(128, 32*M_c) B/cycle,
with one 16-B bank word per lane of that width. This preserves the specified
one-cycle WS RMW (two FP32 accesses for M_c-by-4 outputs) for M6/M8 comparators
as well as D4. Source reads share this finite port and wait prior accumulator
commits. S2 retains its specified 128-B/cycle port. The width is not changed by
the current workload; all port widths enter the resource and area-proxy ledgers.

## D022 — hardware rank-policy validation uses the hardware table format

Per-projection error energies are rounded to BF16 before the three projections
are summed. This changes some rank decisions at restricted byte budgets, so
float-table Q3 results cannot stand in for hardware accuracy. A separate complete
hardware-BF16 Q3 sweep evaluates the actual FFN outputs at its selected ranks.
Lambda0 is calibrated using these same physical energies on calibration windows
and routed-only actual factor bytes. Shared's fixed factor cost is reported
separately. The old float policy and float-calibrated constant remain labelled
software/diagnostic controls. The runtime feedback and numerical policy use the
same exported table and calibration receipt; validation inputs do not fit lambda.

## D023 — soft quotas cannot prevent Current's bounded forward progress

The SWE T128 development stress still deadlocked after source-port accounting:
Next held six of eight prepass tiles, while a stage transition shrank the core's
quota below those held bytes plus Current's first required tile. Neither context
could finish a complete group although 39,424 pool bytes remained physically
unused. Quotas now remain soft partition targets: Current may borrow actually
free physical capacity only for its immediate required group. This never admits
future lookahead, increases the pool or credits, releases a live lease early, or
changes payload bytes. Next remains constrained to its original bounded group.
Borrowed admissions are counted and all physical capacity checks still apply.
The failed dump and both repaired-repeat reports are archived. This is a progress
correction for every organization, not a workload-specific performance rule.

## D024 — the on-chip movement metric includes all physical endpoints

The earlier movement subset omitted ingress endpoints, WOR/XOR destination
writes and array reads, local rank-cache transfers and some RF consumer reads.
The canonical total now sums the complete measured endpoint ledger, including
explicit cross-core wire copies. Its category sum is asserted exactly. Internal
individual-PE wiring is excluded. The old subset remains diagnostic only; N4
uses the corrected total consistently for every organization. U_d deltas,
Down combine and helper returns also incur private source-port service; helper
scratch is leased from the helper's existing physical arena. A capacity lease
alone does not deliver source bytes to a consumer.

## D025 — disable-one controls execute actual alternative data movement

The byte-pool-off branch originally changed only a quota limit and could alias
the enabled allocator. It now reserves fixed 4 KiB slots from static private
partitions of the same total pool capacity. Wire payload bytes are unchanged;
reserved bytes and fragmentation are separately counted. Normal, discarded and
retaken payloads release the actual reservation after their real last consumers
or responses drain. Next admits a whole legal first group only when it fits the
remaining private partition, avoiding two mutually blocked incomplete groups.
Quota-off uses the actual partition capacity for Current's immediate group;
the old five-slot limit belongs only to joint_v1.

Weight-reuse-off previously charged all repeated pool reads before the first M
issue, which did not reproduce their actual dependency timing. Each M block now
reads/fills/decodes its weights, issues, retires its WOR frame and then performs
the next real refill. Pool leases persist through the final copy. P1 decode is
charged on every refill. Future-group loading is suppressed where it would
occupy the required refill frame. These changes do not alter the enabled default.
Finite-slot, per-M-order and all-controls-off progress tests cover P0/P1/P2 and
single, homogeneous and asymmetric organizations.

## D026 — compensation's equal-wire control has native layouts

For equal-wire comparisons, the engine finds the largest native wire byte count
among the legal compensation modes. Smaller plans perform real, at-most-4-KiB
padding transfers at the expert's tail: HBM requests, credits, ingress and pool
bank reads/writes are charged. Padding does not execute arithmetic, reserve WOR,
or decode. Bit widths and factor formats remain fixed, but each compensation
scheme retains its native tile layout and fetch sequence. In particular, `none`
does not fetch unused factors at their original enhanced-tile/A-block positions.
This is an equal-total-wire diagnostic, not a pure arithmetic-only isolation;
tail padding can change completion timing. The six B2/B16 development windows
also receive a separately archived native-wire sensitivity with padding disabled,
without changing any primary N2 threshold or replacing its paired population.

## D027 — accelerate host predicate lookup without changing modeled timing

The fixed-slot path retried context switching during weight wait. Its conceptual
test for resident WOR ownership scanned the host's entire unrolled tile plan on
every retry, although hardware checks only its finite WOR owner tags. A host-only
resident-count cache replaces that scan. It tracks the same in-fill/resident
predicate on allocation and every retirement and is checked against the owner
tags at drain. The new and old executables produced byte-identical full reports
for all 23 stress/control points, each run twice. This reduces host evaluation
time; it adds no modeled physical state, changes no cycle or accounting field,
and is not reported as an architectural speedup.

## D028 — retire consumed helper scratch before admitting a full IS expert

The native-wire P1 offload sensitivity exposed a real circular resource wait.
The dense expert retained its helper's RF lease until whole-expert completion,
even after U or a returned delta was fully stored. A subsequent Me4 cold expert
needed the streaming core's entire private arena; it could not activate while
the dense expert waited for that core to finish and service the next prepass.
Helper scratch now retires only after its real UStore, UAccumulate or delta
return consumer completes. All preceding source reads, stores, RMW and copies
retain their original finite port service. No live result is released early,
no arena grows, and this applies to all offload organizations. The exact failed
BFCL B16 native input and repaired repeats are archived. Because timing changes,
the final campaign restarts with a new binary signature before preregistration.

## D029 — exact carried-lambda JSON transport

The repeated same-budget M5 control trace exposed a one-ULP change when the
default serde_json parser read its previous lambda output back as input.
The dependency now enables float_roundtrip so binary64 control state is
preserved exactly across this serialization boundary. The strict equality
assertion remains unchanged; no approximate tolerance hides the discrepancy.
Cargo manifests and lockfile enter the compiled-source freeze in addition to
Rust sources. All old runtime regression tests are rerun on this dependency
configuration; prior executable signatures remain separate.


## D030 — recover exact experiments after shared scratch exhaustion

An external shared-filesystem capacity exhaustion interrupted CSV publication
and test temporary files. Failed logs and truncated originals are preserved.
The task output tree, immutable captures, historical M0 originals and one model
shard were copied to local storage, individually SHA-verified and exposed through
the same logical paths. Each relocation has a receipt; no tensor, routing trace,
physical configuration, timing byte string or acceptance threshold was changed.
Numerical CSV publication now uses an atomic temporary write and rename. Recovery
reuses only complete Q1 candidate blocks (195 projections +65 experts +1 layer)
and complete Q3 window/policy groups, with unchanged provenance. Incomplete tails
are re-evaluated, and all 2,178 required candidates remain required. Synthetic
timed tests were rerun using a local temporary/cache directory: all56 cases
passed, each executing two identical reports. Their224 original files were
losslessly archived and verified before temporary originals were reclaimed.
Host storage recovery and completed-work reuse are not architectural speedups.

## D031 — protect a complete progress group when weight reuse is disabled

Six real BFCL B8 ablation points (OP0/OP1, forward4/forward5/leave-out-W-reuse)
exposed a hard pool-capacity cycle before heldout evaluation. With W reuse off,
Current retains packed bytes until its final M-block refill. Seven of eight
Current weights could therefore be resident while a partially admitted Next
head consumed the last capacity needed by the eighth. A one-tile quota reserve
was insufficient; unused soft quota cannot create hard storage.

For this refill-disabled path, Current admits only its immediate required
group and receives admission priority. A Next head is admitted atomically only
when the physically free pool can contain that head and all missing immediate
Current group bytes. The test is a bounded check of existing head descriptors
and live leases, with no added SRAM or context. Quotas, partition bounds, reads,
refills, dependencies and lease retirement remain charged and unchanged. No
live bytes are evicted or released early. The default W-reuse-enabled path is
unchanged. The same rule applies to all organizations and affected ablations.

All six original configurations passed twice with byte-identical reports,
including drain and capacity invariants; originals and old deadlocks remain
archived under separate signatures. A regression constructs the exact missing
Current tile versus complete Next-group capacity condition. All 74 Rust tests
pass. Because affected timing changed, v7 restarts the complete timing campaign;
v6 points are not reused as final measurements. Timed numerical and legacy
regressions are repeated on the immutable v7 binary before final authorization.

## D032 — post-heldout correctness amendment: physical frames and resumable contexts

The v7 campaign was frozen in commit `f6c9c5e` before heldout execution. Its
complete matrix then exposed six additional GPQA B16 refill-ablation deadlocks
and two default BFCL B16 M8 heldout deadlocks. All campaign processes were
stopped, the authorization revoked, and every old input/config/result/failure
preserved. This is an explicit correctness amendment after heldout began;
it is not a claim that heldout was never observed. Old measurements cannot
enter the new engine's results, and the entire 55,012-point matrix must restart
with two repetitions. Formats, hardware, inputs, thresholds and the comparator
selection rule remain unchanged.

D031's summed-free-byte test was insufficient. In one captured pool state,
the remaining 4096 B consisted of disjoint 3072-B and 1024-B holes, so Current's
eighth 4096-B tile could not allocate even though the aggregate count fit.
All new groups now use transactional first-fit allocation: the complete
bounded group's actual placements must fit, or no tile receives a lease.
Speculative Current lookahead and Next additionally prove placements for all
missing selected Current heads on that same physical allocator snapshot,
including each private partition. The explicit one-tile Next diagnostic is
the only partial frame and must leave a physically realizable full future
Current frame. Existing Current soft-quota borrowing still obeys hard storage.

The default M8 failures exposed a separate switching omission: a selected
context with no resident WOR tile reached a missing Main head after UStore,
but byte-pool mode did not switch back to an already-ready parked head. The
blocked-head switch now applies in both allocation modes, retaining the
complete-ready-group, real WOR capacity, accumulator and aging checks. Next
prefetch installs finite accumulator/Z/U reservations transactionally after
first protecting all not-yet-activated Current contexts; separate independent
availability checks cannot overcommit an arena. A steal of an unstarted,
action-zero Next returns only its unused context reservations. Accepted DMA
response leases still drain at their original owner, and retakes remain charged.

An offloaded auxiliary group cannot drain while its helper has an independent
Current task. Admission waits at this service-availability barrier rather than
pinning the helper's next frame; offload Next also preserves a complete future
owner frame because offload disables context interleaving. A helper's parked
Next accumulator reservation must additionally leave a physically realizable
worst legal owner helper frame, derived from this batch size and static rank
dimensions. This prevents Next's unpromotable arena from blocking a subsequent
Shared auxiliary result. The extra safe-state edge was derived by independent
review, not observed as a campaign deadlock. No live data is
evicted, compacted, released early or silently copied, and no physical capacity
is added. These rules apply universally across organizations and modes.

Five targeted regressions cover fragmented-frame rollback, shared-pool
blocked-head resume, busy-helper progress, joint context-arena admission and
the parked helper Next's future-owner accumulator headroom.
All 79 Rust tests pass. The original 14 failed configurations (six historical
BFCL, six GPQA and two M8) pass twice under immutable v8, with full JSON identity
and drain/storage invariants. Timed numerical/legacy validation and every
development, native-wire and M5 campaign are repeated before amended PREREG is
committed and new heldout execution is authorized. All failed attempts and
their historical source/binary signatures remain available for audit.

The finalized host implementation searches group descriptors only from the
task's current action, omitting retired prefix descriptors. This changes no
cycle, allocator decision or simulated hardware state; the interim v8a release
and all original raw replays are retained for exact equivalence checking.
