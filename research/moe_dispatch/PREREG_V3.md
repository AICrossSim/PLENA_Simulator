# PLENA-MoE Supply-first v3 preregistration

Status: **Frozen for root commit; heldout timing remains unauthorized**. This file does not authorize evaluation. The root researcher
must commit this file and the adjacent frozen manifest, then record that commit,
the exact frozen timing signature, and the independent quality-receipt SHA in
`heldout_authorization.json`. No heldout timing is used to select the design,
format, comparator, or thresholds. All existing provisional runs remain separate.

## Scope and input populations

Latency times one complete routed SwiGLU MoE FFN layer after routing. Router,
attention and full-model inference are excluded. This is deterministic analytical
Rust event simulation with finite HBM/credit/ingress/pool/ports; it is not native
Ramulator, RTL or silicon measurement. At 1 GHz, 1 cycle = 1 ns and 10⁶ cycles =
1 ms. Host execution time is only campaign progress.

Twelve previously exposed windows are development data. Architectural decode
holdout comprises 108 windows from 90 request identities, at layers 2/13/26,
decode steps 3/7/11 and B2/B4/B8/B16. Identities are SHA-selected and disjoint
from every exposed development request. Reused groups are correlated. Genuine
mixed holdout comprises 27 BFCL/GPQA/SWE windows at those three layers and
T64/96/128, drawn from only three independent BF16-CPU prefill prompt captures
plus distinct B16 decode requests. N4/N5 use the 18 T64/96 windows; T128 is a
separate stress population. SWE canonical input was recovered and byte-matched
to its original prepared-file SHA. Six real mixed development windows and 18
separately labeled constructed/rebatched/Zipf cases are not mixed into holdout
conclusions. Original archive-inventory missing-prefill status is historical;
`mixed_capture_manifest.json` records the subsequent real captures.

## Frozen computation, storage and supply

Physical dimensions are **M×N×K**, with N=4 and K=512. M6 organizations all contain
12,288 main multipliers. M8 organizations share a separate 16,384 multiplier
budget; no M6/M8 comparison is claimed to be equal-compute.

| Design | Main-core dimensions | Dataflow |
|---|---|---|
| BL0 | 4×4×512 + 2×4×512 | legacy / legacy |
| BL1 | 6×4×512 | legacy |
| BL2 | 6×4×512 | switchable |
| BL3 | 3×4×512 + 3×4×512 | switchable / switchable |
| BL4 | 4×4×512 + 2×4×512 | ws_group / is_stream |
| BL5 | 4×4×512 + 2×4×512 | switchable / switchable |
| M8_single | 8×4×512 | switchable |
| M8_homogeneous | 4×4×512 + 4×4×512 | switchable / switchable |
| M8_asym62 | 6×4×512 + 2×4×512 | ws_group / is_stream |
| M8_asym53 | 5×4×512 + 3×4×512 | ws_group / is_stream |
| M8_flex62 | 6×4×512 + 2×4×512 | switchable / switchable |
| M8_flex53 | 5×4×512 + 3×4×512 | switchable / switchable |

Hardware remains fixed across workloads. Physical storage ceiling is 2,158,592 B,
including unused allocated SRAM and 16 KiB control reserve. Physical t_chunk=128
only for P2/L8 at 128/256 B/ns; P1, L16 and bandwidth512 use fixed t_chunk=96.
T128 combinations that violate this allocation are explicit exclusions.
ISO fixes task-specified aggregate pool read=1,024 B/cycle, X=640 B/cycle,
P1 decoder=2,048 B/cycle and installed WOR=18 slots. Group sizes are single/dense8,
homogeneous4 per core, stream1. Demand-port variants report their own costs.
**Private accumulator/source shared 1RW ports are max(128,32×M_c) B/cycle:**
single6=192; homogeneous128+128; asymmetric128+128; single8=256; D6=192.
They are not equal across organizations, and their actual installed width is
included in the component ledger. Vector/UStore reads and RMW share those ports.
U-cache capacity is inside the fixed control reserve; its additional local 1R1W
read/write bandwidth is explicitly costed.

One shared HBM path uses 32 B requests. OP0/OP1: BF16 main weights, 256 B/ns,
64 ns, 256/544 credits. OP2: P2/W4, 256 B/ns, 64 ns,544 credits. OP3: 150 ns,
1,232 credits. OP4: W3 conditional numerical candidate. OP5: P1. Latency/BW
sensitivity also uses the prespecified full inventory. Legacy BL0/BL1 credits544
are labeled overbudget diagnostics; oracle byte shrink retains BF16 local moves.

Default runtime uses window8, two contexts, supply_ipd, bounded prefetch/drop
work stealing, starved-only context switching and dot dependency latency20.
Context age=max(working-point HBM latency, current group tile count×ceil(Me/M));
this is one fixed shape-dependent hardware rule, not per-batch tuning.
Pipeline/quota/byte-pool/reuse/ports/SiLU off-switches must execute actual finite
alternative mechanisms; no fixed issue-gap proxy is accepted.

## Physical numerical format and separate qualification

Frozen actual format: `{"factor_a": "mxint4", "factor_b": "bf16", "main_bits": 4, "rank_lanes": 8, "ranks": {"routed": [32, 32, 24], "shared": [32, 32, 48]}}`.
Physical-format SHA256: `d38b472f954f9dee23881b0c37549f3c1c21736347f3c358bbd466fbef5c54e1`.
Current qualification status: **diagnostic_non_accuracy_preserving**.
Final independent qualification-receipt SHA256: `73050cb77f5bc98bbfe8133d1652c3658ceff5025151f4c925db570ae2bb0c61`.

P2 uses MX low-bit weights with BF16 X/U/Z. `factor_a` describes the low-rank
factor A, not input activation precision; these results are not W4A4. Main input
scaling and mixed-input PE area are not RTL characterized. Numerical quantities
have independent real-weight receipts. Timing signature always hashes the five
normalized actual format fields, including defaults. Changing only qualification
metadata does not invalidate identical physical timing; every bit/factor/L/rank
change creates a new physical campaign and comparator. The final quality file
SHA is separately fixed in preregistration and authorization. No legacy timing
without this explicit physical-format proof is retroactively merged.

Quality selection uses only calibration/development numerical data: ≥32k
calibration and ≥8k request-disjoint evaluation tokens at layer13, plus layers2/26.
When only layer-level results exist, required MoE output error is ≤0.50× W4-RTN
and cosine ≥0.999. If mandatory GPU Q4 is available, perplexity increase is ≤2%.
Choose minimum bytes among qualifying hardware-legal candidates. If none pass,
retain the default only as an explicitly unqualified timing candidate; do not
relax any threshold. W3, B-MXINT4 compensation supplements, and policies whose
rank vectors differ require their own accuracy evidence.

## Comparator frozen from development

M6 comparator: **BL2**. M8 comparator: **M8_flex62**.
Select minimum geometric mean cycles over all 12 development windows at OP2 ISO
from BL2/BL3/BL5 or M8 single/homogeneous/flex62/flex53, respectively. All 84
candidate points require two original identical valid raw JSONs. Never choose a
new best comparator per heldout point or change hardware for a workload.

## Original acceptance criteria (unchanged)

| Claim | Evidence and frozen criterion |
|---|---|
| N1 bottleneck transfer | BL1 R_legacy≤0.40 and BL4 R_v3≥0.85 on decode B2–16 plus real mixedT64/96. R uses equal organization/input/port comparisons, unique useful weight bytes, and simulated time. |
| N2 rank channel | P1 five equal-wire modes; P2 frozen-default none/lanes primary. Lanes sum-core busy overhead≤3%, dense SharedMe≥16/realT96 layer overhead≤3%, installed multipliers≤3.2%. Each of separate/kext/offload must show busy overhead≥10%, or focal layer overhead≥5%, or cold-throughput loss≥10%. Per-core ownership changes are diagnostic. |
| N3 supply mechanisms | All eight forward-add and leave-one-out controls at OP0/OP1/OP2; each leave-one control has time or η loss≥3% in at least one working point. Full OP0 B16 η≥0.95 and OP1 B2–16 η≥0.90. Mechanisms without evidence lose novelty claims. |
| N4 organization | Real mixed BL4 time GM≤0.95× fixed comparator, **or** complete onchip bytes/useful-MAC GM≤0.85 with nominal area-proxy difference≤5%. Decode±3% equivalence reported. ISO/demand and M8 reported separately. |
| N5 runtime | BL4 supply_ipd versus actual v3 joint: real mixed time GM≤0.97 or cross-window p95 ratio≤0.95. All policies share window8/resources/supply path. Physical-BF16 gate_weighted Q3 error≤0.90× uniform at equal bytes, or bytes≤0.80 at equal error. |

N2 B-MXINT4 modes remain supplemental conditional-quality rows, not substitutes
for frozen-default primary comparisons. Equal-wire compensation includes real
32 B padding through credit/ingress/pool; installed rank lanes remain charged
when disabled. Busy=sum exclusive C1+C4+C5+C6+C7 across cores, with matched
work populations. Padding matches total expert-tail wire bytes, not identical
per-tile layouts, A-factor locations or phase consumption order; this is not
pure arithmetic isolation. An additional six development B2/B16 windows×five
P1 modes/two P2 modes×two raw repeats disables wire padding (84 additional
runs), reports native-wire compensation and matching-induced overhead shifts,
and does not substitute or modify any primary N2 gate or main inventory.
For N5 rank integration, common expert-rank choice precedes
projection-cap clipping. Use BF16-RNE energy LUT and its independently calibrated
λ0; routed-only budget changes reset λ0, same budget carries causal feedback.
Two actual reset sequences are repeated; software float-LUT quality is separate.

## Nominal area proxy and additional sensitivity

Default coefficients: main MX1, main/rank BF16 2, divided by12,288; storage SRAM1
and RF4, divided by2MiB; ports0.1 per2,048 logical R+W B/cycle. Full allocated
storage remains charged. WS-capable accumulator arena is interpreted as RF;
specialized IS arena as SRAM, reflecting one-cycle WS RMW. This is a physical-type
proxy interpretation, not mapped RTL. Also report hybrid RF=min(arena,
2×t_chunk×32×4) plus SRAM overflow, all-accumulator-SRAM, and all-accumulator-RF
bounds at **unchanged measured timing**, and coefficient multipliers0.5/1/2.

**N4 nominal criterion remains the original frozen default-proxy criterion.**
`area_type_robust_pass` and coefficient sensitivity are separate diagnostics,
not new hidden thresholds. A nominal area/traffic pass that depends on type is
reported as conditional and cannot support an unconditional hardware-area claim.
No synthesis PPA, power, joules or absolute area is inferred from this proxy.
Canonical port costs cover finite pool/HBM/X/accumulator/Combine/U-cache service
widths. The full-row local WOR/XOR register broadcast demand is separately
reported; a supplementary proxy adds its width with the same port coefficient.
This sensitivity does not silently revise the nominal N4 area gate. Local RF
fanout is not free, and its physical implementation still requires synthesis.

## Full declared matrix and replication

55,012 legal points ×2 = **110,024 raw simulator runs**;
all explicit exclusions remain in the inventory. Historical M0 144×2 is separately
preserved, in addition to full-window M0 extras below. No point is silently removed.

| Split | Suite | Legal points | Declared exclusions |
|---|---|---:|---:|
| development | ablation | 1836 | 0 |
| development | compensation | 132 | 12 |
| development | dev_comparator | 84 | 0 |
| development | m0 | 216 | 0 |
| development | m0_baseline | 48 | 0 |
| development | main | 1536 | 192 |
| development | policies | 96 | 0 |
| development | sensitivity | 192 | 0 |
| heldout | ablation | 16524 | 0 |
| heldout | compensation | 1188 | 108 |
| heldout | m0 | 1944 | 0 |
| heldout | m0_baseline | 432 | 0 |
| heldout | main | 13824 | 1728 |
| heldout | policies | 864 | 0 |
| heldout | sensitivity | 1728 | 0 |
| constructed | ablation | 2448 | 306 |
| constructed | compensation | 183 | 33 |
| constructed | m0 | 270 | 54 |
| constructed | m0_baseline | 60 | 12 |
| constructed | main | 2100 | 492 |
| constructed | policies | 132 | 12 |
| constructed | sensitivity | 276 | 12 |
| mixed_development | ablation | 714 | 204 |
| mixed_development | compensation | 56 | 16 |
| mixed_development | m0 | 72 | 36 |
| mixed_development | m0_baseline | 16 | 8 |
| mixed_development | main | 632 | 232 |
| mixed_development | policies | 40 | 8 |
| mixed_development | sensitivity | 88 | 8 |
| mixed_heldout | ablation | 3213 | 918 |
| mixed_heldout | compensation | 252 | 72 |
| mixed_heldout | m0 | 324 | 162 |
| mixed_heldout | m0_baseline | 72 | 36 |
| mixed_heldout | main | 2844 | 1044 |
| mixed_heldout | policies | 180 | 36 |
| mixed_heldout | sensitivity | 396 | 36 |

Each point archives both original JSON byte strings in gzip, verifies exact
identity and hashes after decompression, and checks all eight invariants,
accepted/landed/drained requests, fixed resource peaks, exclusive core/HBM state
sums, complete onchip movement-category sum and wire-byte decomposition.
Failures/unsupported legal points block completion; a missing claim produces
unavailable, and complete evidence produces pass/fail without changed thresholds.
Aggregate figures separate measured event-model results, closed-form reference
estimates, constructed inputs, and conditional numerical candidates.

## Input hashes

| Artifact | SHA256 |
|---|---|
| `area_coefficients.json` | `336490a7d24b25bb31341144c605305020295e8e07a98730eb6dc42ebbc93285` |
| `declared_matrix_inventory.csv` | `a86a7a8ca8bc878e3c36676ab1484d03b0d424cd0f6d0816a4d1df0fc00d927d` |
| `input_manifest.json` | `234cdbf921b4ce029f8a63abe7f0851d4d729269e225bc5c2b70bff960eb369c` |
| `inputs/constructed.json` | `410ae98d81d3cd87f34a2cae3b59aba7d5855346e3ef80976b46cb4980268c9e` |
| `inputs/development.json` | `168522072cb2fd80e2689af9ad370b25fbad022072f4abc94e4aa512223c95e0` |
| `inputs/heldout.json` | `0628f6e1e34bb936b0bac9be6682263ce53c5f313e3c776b26d5856f9e96191a` |
| `inputs/mixed_development.json` | `c2c4feb51d83abf6b8d224be6bafb9cc9432797f2740eac40ce76503cc8f3f55` |
| `inputs/mixed_heldout.json` | `2910033d86a3346a55ddfdf8781b86039a3726e07515581820bf92f03d233b8f` |
| `mixed_capture_manifest.json` | `b039a34d5e459acb8228908240a00a32d3a337d9a8b5afdbc404b96556591865` |

The adjacent frozen manifest additionally records the final executable, timing
runner, analysis scripts, actual format, final quality receipt, comparator and
84 original development JSON hashes. Root commit and heldout authorization must
match those exact records. Any future substantive correction requires a new
campaign and explicit disclosure; older failed or provisional artifacts remain.
