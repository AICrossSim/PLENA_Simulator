# Real numerical validation of supply-first v3

These programs use the local DeepSeek-V2-Lite-Chat BF16 checkpoint. Synthetic
replay seeds are correctness tests only. Routing-timing inputs and numerical
calibration data have different roles and are not interchangeable.

`capture_cpu.py` performs real BF16 forwards and saves X, actual gate outputs,
request hashes, adjacent prompt positions, and exact NPZ hashes at layers
2/13/26. It reconstructs canonical archived BFCL/GPQA requests and retokenizes
them with the DeepSeek tokenizer. It does not pretend that the previous
capture's tokenization or generated sequence was reproduced.

The complete capture uses `quant/numeric_selection.json`: 210 previously exposed
development identities are split into request-disjoint numerical calibration and
validation subsets. The folder named `real_captures/heldout` is **internal
numerical validation**, not the untouched architectural timing heldout.
No architectural heldout request is used to choose precision, spectra or lambda0.
Both subsets are now complete: 32,768 calibration tokens from38 requests and
8,192 validation tokens from13 disjoint requests. Check `q0_completed.json` for
the manifest and inventory hashes; completion does not imply an accuracy pass.

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python research/moe_dispatch/quant/capture_cpu.py \
  --model /scratch/shared/mcl123/plena/weights/deepseek-v2-lite-chat \
  --selection outputs/moe_supply_first_v3/quant/numeric_selection.json \
  --archives /scratch/shared/mcl123/plena/paper_artifacts \
  --output outputs/moe_supply_first_v3/quant/real_captures \
  --split both --target-tokens 32768 --eval-tokens 8192 --threads 4

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python research/moe_dispatch/quant/run_after_capture.py \
  --model /scratch/shared/mcl123/plena/weights/deepseek-v2-lite-chat \
  --capture-root outputs/moe_supply_first_v3/quant/real_captures \
  --expected-tensors outputs/moe_supply_first_v3/quant/q0_weight_inventory.json \
  --output outputs/moe_supply_first_v3/quant/full_numerics --rank-lanes 8,16
```

`evaluate.py` rejects a smoke capture, hash corruption, overlap, and fewer than
32,768 calibration plus 8,192 validation tokens at all three layers. It evaluates
RTN, LQER, L2QER, QERA-approx and QERA-exact; W3/W4; the requested ranks and A/B
formats; and MXINT8-RTN as the scale reference. Down calibration uses original
BF16 weights and hardware gate-folded Z. Cold-expert own, pooled and diagonal
statistics are separate measurements, including a no-samples status. The
current full capture has no expert with fewer than32 calibration tokens, so
naturally-cold Q2 comparisons are inapplicable rather than invented. QERA-exact
uses `R=XᵀX/n`, adds `1e−6 tr(R)/K I`, floors covariance eigenvalues to
`1e−6 λmax`, then computes the square root and inverse square root before full
SVD. Some experts have n<K, which bounds the raw covariance rank before
regularization; n≥K does not guarantee full rank. The sample-count audit is
recorded separately. We have not measured condition numbers or established
that rank deficiency causes any observed quality failure.

The combined L8/L16 run reuses exactly the same resident full-matrix SVD bases
and source statistics. Basis construction, candidate FFNs and Q3 expert-output
caches use eight independent expert workers with two BLAS threads each
(`threadpoolctl` is required); ordered collection preserves the FP32 expert
merge order. Serial and parallel candidate outputs/metrics are tested identical.
No randomized SVD or matrix downsampling is used. Duplicate clipped rank points
reuse arithmetic but retain all requested rows. Exact completed Q1 and corrected
Q3 checkpoints can resume with capture/checkpoint/content hash receipts. The
launcher locks its output directory and records the evaluator PID;
`job_status.json` is the authoritative running/completed receipt. A partial CSV
does not establish that the full method/format/layer matrix completed.

The current full campaign is parallelized further by independent layers.
`run_layer_shards.py` leaves the already-running layer13 worker unchanged until
its complete checkpoint boundary, while layer2 and layer26 run the identical
candidate sets in isolated folders with eight expert workers and one BLAS thread
per expert. It prevents the primary worker from duplicating those later-layer
candidates. Samples, full-matrix SVD, formats and per-layer causal lambda
sequences are unchanged. The merger verifies exactly2,178 candidates and each
candidate's65 expert plus195 projection metrics, and exactly196,608 actual-FFN
Q3 rows. Only the verified complete merge receives overall `completed` status.
`test_layer_shards.py` checks missing data, duplicates and crossed provenance.
`finalize_layer_campaign.py` adds the naturally-cold audit and the separate
MXINT4-B quality diagnostics; these supplemental rows never expand or replace
the required Q1 matrix.

For L8, requested rank48 clips to the default routed32/32/24 and Shared32/32/48.
Requested rank32 instead leaves Shared Down32. The whole-expert layer-quality
line preserves expert_ffn_hw. Timing-driven streamed Down verification uses
combine_partial_hw to replay each actual N-group atomic partial drain, including
M-block source ranges and interexpert FP32 event ordering. Those host-side gold
matrices do not create an uncharged private output SRAM.

Q3 exports five energies **for every expert and every projection**. Gate/Up
capacity32 and Down capacity24 remain distinct at L8. Uniform32/32/24 and a
lower-byte uniform16 operating point are both reported there. The frozen Q3
rank-choice set stays0/8/16/24/32 at L16, so its uniform32 reference is32/32/32,
not the full default64/64/48; full default L16 quality remains in Q1. Shared
retains the L-specific full ranks in both studies. Full-maximum rank has no headroom
for error reduction by merely reallocating ranks; a smaller budget tests the
allocation mechanism. Discrete byte differences are explicit. The allocation
oracle is separate from the causal once-per-window lambda update and its first
eight warmup windows. Spectral objectives are labeled proxies; actual FFN output
errors are in `q3_actual_ffn.csv`.

The physical controller stores each projection tail energy as BF16 RNE and
byte costs as uint32. Rounding energies can change a discrete rank decision;
software float tables therefore do not automatically establish hardware quality.
`--q3-bf16-hardware` separately evaluates all four actual-FFN policies at both
budgets and exports complete rank-vector comparisons. Hardware lambda0 is
refitted on the same calibration-only 2,048 actual windows using the rounded
projection energies and routed-only physical factor bytes. Shared rank stays
fixed and its bytes are excluded from the feedback budget. The original float
constant and its physical-table byte discrepancy remain explicit diagnostics.
`export_rank_variants.py --hardware-bf16` produces the matching audited table
and calibration receipt for Rust M5; `merge_hardware_q3.py` verifies all three
layers' 196,608 rows and records all physical table hashes. Old float-calibrated
hardware trials are retained under `obsolete_float_lambda_on_BF16` or
`obsolete_float_calibrated_lambda` and excluded from hardware-quality claims.
Q3 averages are means of actual16-token-window relative errors. They are not
the full8,192-row Frobenius metric measured separately by Q1. Requested uniform
rank budgets do not ensure equal actual factor bytes after dynamic allocation.
`q3_hardware_budget_pair_summary.csv` exposes actual byte gaps and exact-byte
subsets. `summarize_quality.py --output ... --wait` also reports equal-error byte
diagnostics across the two measured budgets. Its quality-selected points are
explicitly post-hoc, with no interpolation or causal-runtime claim.
The legacy CSV label `frequency_static` denotes fixed calibration-frequency
importance with a per-active-set byte-budget bisection. It is a MiLo-style
frequency proxy; its selected ranks can change with the active expert set.
It does not represent a fixed physical rank table or a charged online hardware
algorithm. `gate_weighted_budget_oracle` also has explicit per-window bisection;
only `gate_weighted_causal` carries the implemented feedback state.
`q3_qualification.py` assesses the full512-window population and emits
`q3_qualification_summary.json`. It cannot pass an equal-byte criterion using
only byte-matched subsets. Its equal-error factor-byte ratio includes the fixed
Shared factors (698,496B atL8 or1,396,992B atL16) in both totals. Routed-only
ratios remain diagnostics. The reported N5 quality boolean applies only to the
matching canonical physical-format hash and QERA-approx method; other formats
and methods cannot qualify the current default. Coverage gaps retain null status.
Lambda0 is fitted by bisection over the mean bytes actually allocated in the
16-token development windows. Averaging gate scores before the discrete argmin
does not give the same fit. The superseded mean-weight results are preserved in
`obsolete_q3_mean_weight_lambda` and excluded from final claims. Q1 checkpoint
reuse has a capture/checkpoint/inventory hash receipt and does not reuse Q3 rows.
`validate_outputs.py --output ... --wait` independently verifies all2,178
requested Q1 candidates, including clipped duplicate ranks and three layers;
it writes `q1_coverage.json` and rejects missing or duplicate combinations.

`freeze_candidates.json` can be eligible only when all required captures exist
and the same candidate passes all three layers: layer relative error at most
half of W4-RTN, cosine at least0.999. Passing candidates are sorted by real
physical main-plus-factor bytes. A complete capture alone does not pass Q1.
`q4_fake_quant.py` requires CUDA and a frozen all-layer factor cache; GPU is
currently absent, so Q4 must remain unmeasured.

P2 separate/offload compensation needs MXINT4 B. `--supplemental-b4` measures
default ranks with actual MXINT4 A/B and RTN/lanes at bothL8/L16; this check is
kept outside the mandatory BF16/MXINT8-B Q1 search. Its quality cannot be borrowed
from BF16-B results. `pack_real_example.py` additionally materializes a real
routed and Shared expert's complete default payloads, using real calibration
factors, and replays solely decoded bytes. Exact byte/element/replay checks are
packing evidence; they do not imply that the accuracy target passed.

The separate `mixed_capture_manifest.json` records actual architectural heldout
**inputs**: BFCL and GPQA, three layers and T64/96/128, from only two independent
prompt requests. Eighteen resulting windows are correlated and cannot be counted
as eighteen independent requests. Each combines sixteen previously captured
decode routes with48/80/112 adjacent positions from a new BF16 CPU prompt
forward. SWE was initially unavailable, then the official pinned dataset and
original preparation script reproduced the exact2,294-row canonical-file hash
`c6d11a02b328689daf16806a39d1f68805313f6561a20d7c024d756ec59022d5`.
Two new BF16 CPU128-prefix forwards now provide a SWE architectural heldout
supplement (nine correlated layer/size cases from one prompt) and a development
supplement (three size cases at layer13 from one different prompt). They remain
in separate files, preserving the original18-case input and receipt. Capturing
these inputs is allowed before preregistration; observing their timing is not.

The legacy `request_sha256` hashes the request identity string. Exact content is
bound separately by `canonical_record_sha256` and tokenizer-prefix SHA. Existing
Q0 manifests remain unchanged; `q0_request_content_provenance.json` supplements
all51 captured request records with these content hashes. New captures record
identity and content hashes explicitly.

The host filesystem filled during the original campaign. `recover_campaign.py`
preserves the failed original bytes and SHA receipts, reuses only complete
261-row Q1 candidates and complete4096-row Q3 groups, and evaluates every missing
point on the unchanged captures and full-matrix arithmetic. CSV checkpoints now
publish through a same-directory temporary file, fsync and atomic replacement,
so another host ENOSPC cannot truncate a completed checkpoint. Recovery keeps the
same complete2,178-candidate /196,608-window evidence requirements; interrupted
tails and obsolete jobs never establish completion. Output backing storage and
one model shard were losslessly SHA-verified before same-path symlink relocation
to `/tmp`; recorded payload hashes and logical input paths remain unchanged.
