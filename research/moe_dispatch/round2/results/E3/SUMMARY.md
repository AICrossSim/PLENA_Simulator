# E3 execution and evidence summary

All requested numerical stages completed under engine `943211ec3825b344fa3358dd503f34ca4e452262c4a205b637d52c4b6c05c123`. The frozen input manifest is `bda4dbefd52c997cac8c638581d698b57b6c12d359a004c14726e01b7430df64`; its development set has 18 windows and heldout set has 135. The main selection was frozen only after all three on-chip modes completed. [FROZEN_SELECTION.json](FROZEN_SELECTION.json) SHA-256 is `f461deaba2fce1d86aeaaefe205d8e4ab502ba69ff6a3b890b8dbad7360d0373`. Each accepted concrete physical configuration was evaluated twice with equality assertions. Search completion and matching repeats do not close unvisited regions.

The objective is the executable finite assignment/LPT analytical replay, with BF16 and hypothetical cycle-to-time conversion (1 cycle = 1 ns). A resource-allocation `OPTIMAL` status proves that inner resource optimization; it does not prove an optimal temporal schedule. A `FEASIBLE` status supplies a concrete owner witness and its actual cost, with an open assignment bound. All source-compatible successful executions, exact commands, environment, source hashes and commits are in the linked actual receipts.

## Completed scopes

| Stage | Completed scope | Actual elapsed seconds | Receipt |
|---|---|---:|---|
| Main search | 1,908 seed records (636/mode, 615 valid + 21 rejected/mode), six mode × A/B proofs | 1441.032 | [E3_main_search](../executions/E3_main_search_20261007T135855Z.json) |
| Lower-bound audit | 2,000 concrete feasible designs × 18 development windows = 36,000 checks; all passed, repeated twice | 289.587 | [E3_lb_validity](../executions/E3_lb_validity_20261007T135840Z.json) |
| Scheduling gaps | 3 organizations × 3 modes × 153 windows = 1,377 rows; all inner assignments OPTIMAL, repeats match | 17.821 | [E3_schedule_gaps](../executions/E3_schedule_gaps_20261007T142405Z.json) |
| E3_grid | Full 8 × 5 × 4 × 3 × 3 × 3 grid = 4,320 synthetic workloads | 917.624 | [E3_grid](../executions/E3_grid_20261007T142449Z.json) |
| E3_extreme | 500 actual CMA evaluations, plus final full-domain δ=0 check at requested 120-second traversal cap | 485.158 | [E3_extreme](../executions/E3_extreme_20261007T145104Z.json) |
| E3_robust | 66 candidate measurements; 396 objective rows and 396 stability rows; 36 groups × 200 bootstrap draws | 110.983 | [E3_robust](../executions/E3_robust_20261007T145915Z.json) |
| E3_sobol | All 1,792 Saltelli samples: base N=256, five parameters, no second-order terms | 1220.006 | [E3_sobol](../executions/E3_sobol_20261007T150110Z.json) |
| E3_flip | All 105 samples: five requested axes × 21 values | 62.372 | [E3_flip](../executions/E3_flip_20261007T152135Z.json) |

[single_exhaustion_receipt.json](single_exhaustion_receipt.json) independently accounts for all 44 single-core geometries × three flows = 132 templates per mode: 129 legal repeated points and three physically rejected templates, with all 2,322 legal development assignments OPTIMAL per mode. Its exhaustive claim is restricted to the declared single-core family and finite replay objective. The BnB traversal certificate can retain open single-family regions even when this independent exhaustive single-family receipt is complete.

[inner_assignment_verification_protocol.json](inner_assignment_verification_protocol.json) records the separate higher-effort diagnostic over all 940 formerly unresolved seed/window cases: full-object repeats match, 218 owner assignments changed, and 718 cases remain unproved. This diagnostic preserved the frozen seed/proof/selection files; its changed owners did not retroactively alter headline results.

## Main proof scope

A uses δ=5%; B uses δ=0. Each full-domain traversal was capped at 120 seconds after mandatory repeated incumbent setup. The saved frontier describes every remaining declared geometry × flow × independent private-resource cut.

| Mode | Proof | Incumbent U (ms) | Remaining global LB (ms) | Gap (%) | Global δ proof | All family proofs closed |
|---|---|---:|---:|---:|---|---|
| pipelined | [A](bnb_pipelined_A.json) | 4.013614221 | 4.005723977 | 0.196974 | True | False |
| pipelined | [B](bnb_pipelined_B.json) | 4.013614221 | 4.005723977 | 0.196974 | False | False |
| port_tight | [A](bnb_port_tight_A.json) | 11.243781734 | 11.161047174 | 0.741280 | True | False |
| port_tight | [B](bnb_port_tight_B.json) | 11.243781734 | 11.161047174 | 0.741280 | False | False |
| fixed_issue | [A](bnb_fixed_issue_A.json) | 5.261347663 | 4.953606732 | 6.212462 | False | False |
| fixed_issue | [B](bnb_fixed_issue_B.json) | 5.261347663 | 4.953606732 | 6.212462 | False | False |

Pipelined and port-tight A establish the reported global 5% bound, while their 5+1 family proof remains open. Fixed-issue A does not close the requested 5% bound. Every B proof remains open. A global bound does not certify each family optimum. Per-family covered/declared lattice counts, open counts and bounds are preserved in the compact [bnb_summary.json](bnb_summary.json) index; the six individual proof files preserve the exact frontiers and continuation state. The lower-bound random audit supports implementation confidence and is not itself a mathematical proof.

## Supplemental proof and delta scope

Grid, CMA evaluation, Sobol and Flip points use δ=2% and a requested two-second traversal cap after mandatory repeated witness setup. The table counts closed family search proofs at that tolerance; it does not label their optima exact.

| Stage | Point records | All-family proofs closed at δ=2% | Points still open | Accepted repeated concrete points | Concrete points with unproved inner allocation |
|---|---:|---:|---:|---:|---:|
| grid | 4320 | 1131 | 3189 | 36475 | 306 |
| extreme | 500 | 34 | 466 | 7769 | 177 |
| sobol | 1792 | 5 | 1787 | 5376 | 62 |
| flip | 105 | 4 | 101 | 315 | 0 |

The displayed candidate delta is `100 × (U_H/U_S − 1)` in percentage fields. Certified family bounds imply the true family-optimum delta interval `[100 × (LB_H/U_S − 1), 100 × (U_H/LB_S − 1)]`, and analogously for homogeneous. Sobol/Flip `delta_lower` and `delta_upper` use fractions; `*_pct` uses percentages. Every grid/CMA/Sobol/Flip H-versus-single interval contains zero, so these supplemental data do not prove the candidate ordering of true family optima. The exact point bounds and repeat flags are included in the complete certificate archives.

The best grid candidate is point 3214: B=64, α=0.568926483, shared units=4, E=64, top-k=6, F=512, hypothetical bandwidth=252.061538 GB/s. Its candidate H/S delta is -11.327113%, with certified interval [-14.823192%, 4.104496%].

The CMA final candidate has H/S delta -10.993985% and certified interval [-14.382642%, 3.957909%]. Its final δ=0 check remains open: U=1.744880138 ms, LB=1.678448661 ms, gap=3.957909%. Six rejected CMA concrete attempts were physically invalid because an expert had no physical core; they were not repeat mismatches.

Only grid/CMA use synthetic routing. Both Dirichlet concentration endpoints were fitted exclusively to frozen development data: the highest-entropy non-SWE cohort (three mixed windows) gives α=1.863127473 and loss=2.365878238; all seven SWE development windows give α=0.568926483 and loss=3.022501013. Five geometric levels run from the fitted diffuse endpoint to the fitted SWE endpoint. The diffuse endpoint is an empirical fit, not perfectly uniform routing. Exact cohort IDs, entropy-based selection, seeds, losses and validation metrics are in [synthetic_calibration.json](synthetic_calibration.json) and [synthetic_calibration.csv](synthetic_calibration.csv). Bandwidth variants are hypothetical scenarios and do not add measured hardware points.

Robustness constructs its candidate pool from every concrete evaluated A/B leaf, seed and frozen candidate, deduplicated by the full canonical physical design and classified by its actual physical family. Its development near-set is within 1% of the best evaluated family candidate, not a certified global near-optimum set. Agreement compares flows and every private-resource cut as well as geometry. The three objectives agree for each single and homogeneous pool. They agree for the heterogeneous pool in pipelined/fixed-issue and disagree in port-tight; agreement over the combined pool is false in pipelined/fixed-issue and true in port-tight. Heldout objective selection is explicitly a diagnostic and does not change frozen headline hardware.

Sobol and Flip use all 18 real development windows. τ is `weight_tile_service_cycles`: shared W frontend bandwidth is `min(64 × bank_Bpc, 4096/τ)`, allocated to each core in the proportion `w_banks/64`; the datapath issue interval remains 1. Installed-design assignment and simulator W service use the cap through `p.w_bandwidth`. Universal and regional W lower bounds omit the extra cap and remain conservative but can be looser. The independent timing-helper source SHA is recorded in each parameter payload and protocol.

| Sobol parameter | S1 | S1 CI half-width | ST | ST CI half-width |
|---|---:|---:|---:|---:|
| weight_tile_service_cycles | 0.650410 | 0.260303 | 1.022891 | 0.242410 |
| bank_Bpc | 0.020415 | 0.040286 | 0.263278 | 0.195143 |
| dotstagecycles | 0.002175 | 0.003967 | 0.000170 | 0.000108 |
| credits | 0.055875 | 0.051126 | 0.194503 | 0.104973 |
| vector_scale | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

These are indices of the time-limited best-evaluated candidate score because 1,787/1,792 samples remain open. Their bootstrap confidence half-widths describe estimator uncertainty and exclude hardware-search approximation error. The slightly greater-than-one τ ST is a finite-sample estimator result, with its uncertainty reported; it is not a claim about an exact family-optimum response. Flip records four interpolated candidate zero crossings: τ≈5.463978, bank bandwidth≈12.542976 and≈23.111952 B/cycle, and credits≈257.907438 on the requested grid axis. Actual sample credits are rounded integers. No sampled −5% crossing was found. All interpolated boundaries have `proof_complete=False`; they are not certified sign-change boundaries.

## Independent checks and portable raw evidence

The point-certificate audits [grid](supplemental_audits_grid.json), [extreme](supplemental_audits_extreme.json), [Sobol](supplemental_audits_sobol.json) and [Flip](supplemental_audits_flip.json) passed independent checks of exact row indices/counts, receipt/log/source/input hashes, physical family, complete lattice accounting, every accepted/family repeat flag, workload/parameter hashes, assignment statuses and candidate/bound interval calculations. The [robustness audit](supplemental_audits_robust.json) verified repeated measurements, development/heldout scope, complete objective/group counts, bootstrap accounting and full-design agreement. A separate read-only review checked Sobol/Flip protocol semantics and boundary outputs.

All 6,717 original point certificates, their complete frontiers and saved workloads are preserved in independent tar.gz chunks below 45 MiB (target 30 MiB). Every member passed byte-for-byte SHA-256 roundtrip and saved engine/workload/parameter validation. The archive manifests link the complete-scope audits and record a repeated-witness flag for every point. No numerical continuation was performed during packaging.

| Archive stage | Original point certificates | Raw bytes | Portable compressed bytes | Chunks | Manifest |
|---|---:|---:|---:|---:|---|
| grid | 4320 | 4031680059 | 238182684 | 8 | [manifest](certificate_archives/grid/manifest.json) |
| extreme | 500 | 542393848 | 32807836 | 2 | [manifest](certificate_archives/extreme/manifest.json) |
| sobol | 1792 | 5483704671 | 513788760 | 17 | [manifest](certificate_archives/sobol/manifest.json) |
| flip | 105 | 309960507 | 28163716 | 1 | [manifest](certificate_archives/flip/manifest.json) |

The 143,858,188-byte exact original CMA driver output is preserved in [extreme_final](certificate_archives/extreme_final/README.md), with an independent final δ=0 checkpoint at `cma_verification/final_delta0.json`. The 85,596,013-byte exact original main summary is preserved in [main_summary](certificate_archives/main_summary/README.md). Required `workload_extreme.json` and `bnb_summary.json` are explicitly derived compact summaries; original and derived hashes remain distinct in [original_driver_outputs.csv](certificate_archives/original_driver_outputs.csv). The main report reader loads the six unchanged individual BnB files, retaining their exact frontier/resume fields. The entire original-output archives also passed bytewise roundtrip checks.

From the E3 directory, extract the point certificate sets needed for continuation (each archive restores its original relative paths):

```sh
for stage in grid extreme sobol flip; do
  for archive in certificate_archives/$stage/part_*.tar.gz; do
    tar -xzf "$archive" -C .
  done
done
tar -xzf certificate_archives/extreme_final/part_000.tar.gz -C . cma_verification/final_delta0.json
```

From the repository root, resume a saved full frontier with the unchanged engine and saved routing/parameters:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=20261007 \
  python -m research.moe_dispatch.round2.resume \
  --certificate research/moe_dispatch/round2/results/E3/cma_verification/final_delta0.json --seconds 3600
```

The same interface accepts an extracted grid/Sobol/Flip/CMA evaluation point or an existing `bnb_{mode}_{A|B}.json` file. A new checkout uses portable archive extraction rather than machine-local spill links. Continue the open proofs before claiming global family optima or proven candidate advantages. Actual producer receipts remain authoritative; a later delivery-metadata commit is not substituted for a numerical execution commit.
