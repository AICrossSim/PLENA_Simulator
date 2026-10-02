# Supply-first v3 experiments

This runner times one complete routed SwiGLU MoE FFN layer in the Rust analytical
engine. Input router execution, attention and full-model inference are excluded.
Latency is cycles at 1 GHz: 1 cycle = 1 ns, 1,000 cycles = 1 µs, 1,000,000 cycles
= 1 ms. Host elapsed time is only execution progress. Native HBM/Ramulator and
physical area/power are not inferred from these measurements.

P2 uses MX low-bit **weights** with BF16 X/U/Z under the supplied arithmetic
contract. `factor_a=mxint4` describes the low-rank A matrix, not activation
quantization; this is not a W4A4 experiment. MX multiplier coefficients are a
prespecified relative proxy. Mixed-input multipliers and exponent reduction
have not been RTL timed or synthesized as INT4×INT4 hardware.

`study.py prepare` freezes route inputs before new timing. It uses all 12 formerly
exposed Joint windows as development data. The 108 held-out windows use 90 new
requests; each fixed B2/B4/B8/B16 group is replayed at layers 2/13/26 and decode
steps 3/7/11. Groups across layers/steps are correlated, rather than 108 independent
request samples. Every prior robust and Joint request is excluded by identity.
No router score, expert ID or routed token is changed in captured workloads.

Original source archives contain per-token decode routing but only aggregate
prefill counts. Subsequent BF16 CPU captures supplied contiguous real prompt
positions: six mixed development windows and 27 mixed heldout windows across
BFCL/GPQA/SWE, three layers and T64/96/128. Those 27 windows use only three
independent prefill prompts and are correlated. Exact SWE canonical requests
were restored and matched to the original prepared-file hash before capture.
`mixed_capture_manifest.json` records the new evidence; historical archive
inventory entries remain unchanged. Constructed offline B32/B64 and Zipf routing
T32/64/96/128 are separate provenance categories and tables. They cannot pass a
claim requiring genuine mixed inference.

## Commands

Run with the NumPy/Matplotlib environment at
`/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python`.

```sh
python study.py prepare --root OUTPUT
python study.py matrix --root OUTPUT --split heldout --suite all
python run_development.py --root OUTPUT --binary BINARY --workers 12
python preregister.py --root OUTPUT --draft
python preregister.py --root OUTPUT --binary BINARY
python run_final.py --root OUTPUT --binary BINARY --workers 20
python study.py run --root OUTPUT --split development --suite m0 --binary BINARY --workers 24
python study.py run --root OUTPUT --split development --suite ablation --binary BINARY --workers 24
python study.py run --root OUTPUT --split heldout --suite main --binary BINARY --workers 24
python study.py run --root OUTPUT --split heldout --suite compensation --binary BINARY --workers 24
python study.py run --root OUTPUT --split heldout --suite policies --binary BINARY --workers 24
python study.py run --root OUTPUT --split heldout --suite sensitivity --binary BINARY --workers 24
python report.py --root OUTPUT
python budget_estimates.py --root OUTPUT
```

`run_development.py` first validates all eight T128 BL2--BL5 OP2 ISO stress
points, then the 84 development comparator points, then all development and
mixed-development mechanisms. `--mode comparator` stops after stress and
comparator selection; `--mode stress` runs only stress. Any failed or unsupported
legal point stops this launcher. The comparator writes
`development_freeze_ready_receipt.json` with the active binary/input/runner
signature, all 84 original JSON hashes, and fixed M6/M8 selections. This receipt
does not authorize heldout or replace the root's preregistration commit.

Every point is durably receipted immediately; aggregate CSV checkpoints occur
every 1,000 points or 120 host seconds to avoid quadratic filesystem work. Small
stages checkpoint every 32 points. Raising worker count on resume only changes
host throughput, never modeled timing. The exact 55,012 legal points require
110,024 individual runs, including both original repeats. A provisional T128
development stress measured 5.9--10.6 host seconds/run while numerical jobs
contended for CPU. A weighted planning estimate is 53--78 CPU-hours total,
approximately 4.5--7 hours with 12 workers or 2.7--4.2 hours with 20, before
additional contention or slow serial-mechanism ablations. This is a host-runtime
estimate, not a simulated-latency result.

The logical output path remains `SIM/outputs/moe_supply_first_v3`. Following
scratch ENOSPC, this entire task-owned output directory is a same-path symlink
to `/tmp`; new raw campaign directories also use task-specific `/tmp` targets.
`whole_output_storage_relocation_receipt.json` records SHA-verified regular
files and preserved symlinks. `storage_resume_audit_v6.json` records intact
metadata and preserved interrupted reports. This changes storage only; inputs,
logical paths, physical-format and timing signatures are unchanged. Completed
raw reports are retained; incomplete attempts are labeled separately. Original
timed-numerical files also have a lossless archive and individual hash receipt.

The root agent must create `OUTPUT/heldout_authorization.json` after committing
`PREREG_V3.md`; non-development timing checks that receipt. A receipt includes
`authorized: true`, the committed prereg hash and `frozen_signature` copied
verbatim from the development freeze-ready receipt, plus the independent final
`quality_receipt_sha256` (null only when no quality file exists). Any changed
executable, runner, actual physical format or input bundle is rejected until its
own preregistration exists. `preregister.py --draft` prepares reviewable
`PREREG_V3.md`/`FROZEN_MANIFEST_V3.json` without simulator execution; final generation
requires all 84 active comparator points and matching original JSON receipts.
The generator neither commits nor writes authorization.
Preparation and closed-form
budget estimates do not run the simulator and may precede the commit.

Every accepted simulation point emits both complete raw reports and requires
byte-for-byte identity before archiving both original byte strings as gzip. Their uncompressed hashes and exact decompression remain checked on resume. Config and workload inputs are content-addressed and shared by symlinks; empty logs are omitted. Config, workload and simulator hashes accompany each
point. Existing point receipts resume without rerunning only when their hashes
and both reports still match. A changed executable or runner creates a new
signature directory. Failures are recorded with stderr and are not included as
valid performance results. Unsupported working-point combinations are explicit
predeclared exclusions, not silently removed rows.
Raw-JSON identity verifies deterministic timing/accounting, not full-model tensor
execution at every large matrix point. Functional issue-stream numeric checks
and real-weight accuracy measurements have separate receipts; their actual
coverage and tolerances are reported independently.

## Matrix and attribution

BL0 keeps Joint v1 behavior. BL1 uses the preserved Joint engine with one 6-row core, the frozen resource layout and FIFO placement.
BL2 single 6, BL3 3+3, BL4 specialized 4+2 and BL5 switchable 4+2 use the same v3
supply implementation. M8 separately compares 8, 4+4, 6+2 and 5+3; M6 versus M8
is not an equal-multiplier comparison. ISO-port experiments fix aggregate pool
read width to 1,024 B/cycle and X width to 640 B/cycle, plus 2,048 B/cycle total P1 decoder throughput. WOR physical reserve must
also be equal and is supplied as 18 slots; active group sizes remain 8 on the
single/dense core and 4 per homogeneous core. Demand-port results separately
report the resulting resource costs.

Main OP0--OP5 combinations, forward and disable-one M1 mechanisms, compensation
schemes, placement policies and bandwidth/latency sensitivity are enumerated
by `matrix`. BL0/BL1 quantized working points are not physical MX modes; their
byte-only counterfactuals belong to M0 and are marked oracle. OP4 W3 results are
conditional diagnostic rows until real-weight numerical quality passes. P1
T128 and capacity-invalid bandwidth512/T128 rows are explicit exclusions.

R_v3 pairs BF16 OP1 and P2 OP2 within the same organization, input and ports.
The N4 comparator is the single fastest BL2/BL3/BL5 development geometric mean
at OP2 ISO ports, selected before held-out timing; it is not picked separately
per test point. Exclusive per-core state fractions are normalized to layer
cycles and never added across cores. `analytical_budget_estimates.csv` contains
fluid closed-form estimates and is never merged into simulated timing tables.

Physical `t_chunk` is fixed per supported working point rather than resized for
each workload: P2/L8 at 128/256 B/cycle reserves T=128; P1, L16 and bandwidth512
reserve T=96. Smaller decoded batches use the same reserved structures. The
physical SRAM ceiling is retained at 2,158,592 B even when live data or reserved
substructures occupy less; an area proxy cannot count unused capacity as free.
BL0/BL1 at OP1 are explicitly overbudget legacy credit-expansion diagnostics,
not equal-budget organization baselines. Their actual frozen Compiler resource
layouts and original timing engine are used rather than an assumed 31-cycle fit.

`inputs/mixed_development.json`, if produced by the capture pipeline, contains
16 actual decoded tokens plus 48/80/112 adjacent positions from one real prompt.
These developmental capture-prefix cases are reported as `captured_mixed`,
separately from Zipf. A separately hash-disjoint mixed held-out bundle is still
required before those cases count as the preregistered N4/N5 test population.

The additional `mixed_capture_manifest.json` freezes the genuine held-out mixed
capture without rewriting the earlier archive-inventory manifest. BFCL/GPQA
capture contributes eighteen windows from only two independent prefill requests,
with matching disjoint B16 decode groups. SWE canonical messages were recovered
from pinned official dataset data and byte-identically verified against the
original prepared-file SHA; `inputs/recovered_swe/recovery_receipt.json` records
the proof. Its CPU capture adds nine SWE windows at layers2/13/26×T64/96/128.
The combined final manifest determines the frozen three-request population;
twenty-seven route windows do not constitute twenty-seven independent prompts.

`report.py` defaults to the active binary hash in `active_result_signature.json`
and excludes stale provisional binaries even if earlier CSVs remain archived.
The campaign signature also includes every frozen input-bundle SHA, the runner
SHA, archive-inventory SHA and a canonical physical-format SHA that is always
present, including defaults. Qualification metadata is separately hashed and
archived. A source/input or actual-format change starts a separate campaign;
identical-format qualification metadata updates can retain their physical timing.
Older signatures without explicit canonical physical proof are not retroactively
merged into the new campaign.
The selected legacy planner's `compiler.py`, loader `frontend.py` and frozen
resource-layout JSON also enter the timing signature and are snapshotted;
external planner changes cannot silently reuse prior raw results. The final
manifest verifies the immutable binary's Rust source manifest and freezes
Compiler-v3 sources plus actual Compiler/Rust numerical correctness receipts.
`--binary-sha` selects an explicit campaign. Every accepted point requires all
eight invariant flags, exclusive state sums, transaction drainage and peak
capacity checks. Timeouts and unsupported points remain explicit records.

Compensation comparisons set `comp_equal_bytes=true`. If a mode needs fewer
physical payload bytes, actual 32-byte padding transfers consume HBM credit,
ingress and pool service; `comp_padding_bytes` is shown separately. Supply
efficiency and byte-saving metrics use unique useful bytes, while matched-mode
compensation comparisons verify equal wire bytes.
This is equal total wire payload through actual expert-tail padding; per-tile
layouts and phase-consumption order are not matched, so it is not pure arithmetic
isolation. `compensation_native.py` supplies six development B2/B16 windows×
five P1/two default P2 modes with `comp_equal_bytes=false`, each repeated twice.
Its84 extra runs are separate sensitivity evidence, retain all installed hardware,
and do not alter N2 thresholds or the main55,012-point inventory. After the main
development compensation rows finish, `--report-only` pairs the native reports
with those original equal-total-wire raw receipts and reports padding/timing shifts.
Bounded work stealing may discard already accepted prefetch payloads; these
requests still complete and consume physical credits and storage. Wire bytes
equal native expert payload plus compensation padding plus work-steal waste.
Unique useful bytes exclude both padding and discarded payloads.
P1 compensation uses the main frozen factor format. P2 compensation separately
sets B to MXINT4 for every matched mode, including the no-compensation timing
control. These rows retain a distinct precision-quality label and cannot inherit
qualification of a BF16-B default; numerical checking of this supplemental
format is reported separately from timing/byte matching.
The primary P2 `none`/`lanes` pair also keeps the frozen default B format with
matched wire payload. All five MXINT4-B modes remain separate `b4_*` points
(K-extension explicitly excluded). N2 primary uses four P1 alternatives and
the default-format P2 lane comparison; supplementary B4 findings do not silently
tighten that primary verdict.

The relative area proxy counts MX main multipliers with coefficient1 and BF16
main/rank multipliers with coefficient2, divided by12,288. SRAM bytes have weight1
and RF bytes weight4, both normalized by2MiB. SRAM charge includes unused capacity
up to the2,158,592-byte ceiling after subtracting RF bytes. Logical read/write
port widths have weight0.1 per2,048B/cycle. Raw component ledgers and individual
coefficient×0.5/1/2 sensitivity are reported; these are not synthesis area estimates.
The two-entry rank operand cache is banked 1R1W SRAM inside the fixed16KiB
control reserve; its installed local read/write widths are additional explicit
port-proxy components, rather than additional storage capacity. The private
accumulator's actual shared 1RW source/RMW width is also reported; the prescribed
logical R+W proxy counts both directions of that physical port. ISO fixes the
task-specified pool/X/WOR/decoder totals, rather than silently claiming identical
private accumulator widths. The default area interpretation assigns a complete
WS-capable arena to RF and a specialized IS arena to SRAM. Alternative hybrid
(`min(arena,2*T_chunk*32*4)` RF per WS-capable core), all-SRAM and all-RF
accumulator interpretations are reported with the same measured timing. These
are physical-type estimates, not an implemented split allocator or RTL mapping.
N4 retains its original nominal criterion using the frozen default proxy and
5% area condition. A separate robustness field checks all four interpretations;
a nominal area/traffic pass that fails this sensitivity is explicitly
type-dependent and cannot support an unconditional hardware-area conclusion.
The latency branch retains its original 5% time criterion.
Default context
switching is `starved`, with age=max(working-point HBM latency, current group
tile count × ceil(Me/M)) cycles. This fixed hardware rule loads a runtime counter
from known shape;64 is only the usual default-latency floor. An unconditional
alternating policy remains an explicit development diagnostic. Dynamic rank selection uses
one common expert rank candidate followed by projection-capacity clipping.

Every timing point starts with fresh runtime state. Latency EWMA and shape-bin
feedback update only from completed requests/tasks in that layer; no held-out
window trains a constant for a later point. The two repeated executions therefore
check a complete reset run. These measurements do not claim predictor convergence
across multiple model iterations or continuous full-model execution.

The optional `--format-config` reads a single frozen JSON `{config: {main_bits,
factor_a, factor_b, rank_lanes, ranks}, quality_status: ...}`. A file named
`frozen_format.json` under the output root is loaded automatically. Its five
normalized actual parameters determine the campaign signature. The complete
file's SHA independently freezes quality/provenance in preregistration and
heldout authorization; all receipt versions are content-addressed in snapshots.
No file and a file specifying the same defaults have identical physical timing
signatures. This format
applies to every non-BF16 working point and organization; it cannot carry
placement, per-batch capacity, or port parameters. W3 OP4 requires its own Q1
validation. Absence of a qualified format keeps all default rows timing candidates.

N2 uses the sum of exclusive busy cycles over both cores as its primary service
metric. Per-core changes are reported as diagnostics because task assignment can
change. The layer-time focus consists of the frozen real mixed T=96
windows with Shared Me>=16. Alternative compensation modes may also demonstrate
a cold-expert throughput penalty: routed Me<=2 rows divided by the span from
the first such binding through their final completion, including queuing. That
ratio requires identical cold descriptor and row populations. Missing evidence
leaves a claim unavailable; complete evidence produces pass/fail without changing
the frozen thresholds, and alternative-mode wins are retained in their own table.

`render_report.py` joins measured timing, exact matrix coverage, milestone
receipts and separate numerical qualification into `REPORT_V3.md`. The final
launcher invokes it after report/assessment and its completion receipt; it can
also render a preliminary report that truthfully marks pending evidence.
N5 requires the authoritative complete hardware-BF16 Q3 qualification summary,
not merely a Q3 computation receipt or selected equal-byte subset.

`table_local_operand_ports.csv` reports worst-case full-row WOR/XOR operand
delivery per cycle, including local register broadcast, separately from the
finite service-port ledger. The original canonical proxy covers pool/HBM/X/
accumulator/Combine/U-cache widths, and does not establish zero cost for omitted
local fanout. `proxy_with_local_operand_broadcast_ports` adds an explicit
hypothetical local-port sensitivity; it does not alter N4's original nominal
criterion or provide RF synthesis evidence. P2 W4/L8 WOR delivery is1152B/core/
cycle; X delivery is4096+2048B/cycle for4+2, before PE-internal fanout.

M5 integration reads the hardware-BF16-RNE LUT with its physical calibration
lambda0. Its common-expert rank selection matches Q3. The uniform-active
routed-only byte budget is computed for each window: budget changes reset
lambda0 and identical budgets carry causal feedback. Two complete actual reset
sequences check both the rule and deterministic original raw reports. Shared
factor bytes remain fixed and excluded from feedback.
The six active-expert counts6/48/11/39/7/21 change the budget every window and
therefore cannot demonstrate cross-window convergence. A separate temporal
replay of the same real BFCL B2 window twice checks actual same-budget lambda
carry with two real reset sequences. It adds zero independent requests and is
excluded from primary six-window or N5 benefit statistics.
