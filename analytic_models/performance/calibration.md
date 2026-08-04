# Decode calibration

How the analytical decode model is priced, what evidence is required before its
numbers may be used to rank hardware, and what has actually been measured.

## Timing contracts

The model prices matrix issue under one of three contracts:

- `rtl_serialized` — the implemented behaviour, using the matrix issue latency
  `3 * BLEN + 11`.
- `drain_overlapped` — the same issue interval with the accumulate drain hidden
  behind the next issue, which requires a second accumulator bank.
- `ideal_matrix_pipeline` — an architectural oracle that uses `BLEN`.

Evidence artifacts carry a fourth mode, `emulator_serialized`. It prices
identically to `rtl_serialized`; only the measurement reference differs, and it
records that the anchor cycles came from the transactional emulator rather than
from RTL simulation. Its `evidence_tier` is `emulator` instead of `rtl`, and
that tier must be carried into anything the artifact is quoted in.

The serialized label is scoped to the active RTL implementation and is not a
decode-architecture requirement. The source-derived
`plena-rtl-execution-contract` artifact records that scope explicitly and
changes identity whenever the relevant decoder, control, matrix, or vector RTL
changes.

No contract may rank hardware without matched cycle evidence: a compiler
identity plus emulator cycles for the RTL tier's cross-check, and RTL cycles
too when the artifact claims the RTL tier. Each report retains measured compute
time, ideal-array compute
time, memory time, and the serialization gap. `algorithmic_bottleneck`
compares memory with the ideal array. The realized label is
`serialization` when decode is algorithmically memory-bound but the
implemented matrix issue interval is the limiter.

The steady-state step composition is versioned as `max_compute_memory`.
This is the roofline overlap hypothesis, not an assumed RTL fact. The two
required whole-layer anchors must include HBM activity and make the timing
artifact fail when measured execution does not satisfy that composition.

## Evidence

Create a CSV with these columns:

```text
anchor_id,anchor_kind,analytical_cycles,analytical_compute_cycles,analytical_memory_cycles,cache_position,batch,physical_hbm_bytes,emulator_cycles,rtl_cycles,mlen,blen,hlen,vlen,geometry_path,precision_path,asm_path,analytical_trace_path,emulator_trace_path,rtl_trace_path
```

`anchor_kind` must cover `linear`, `qk`, `pv`, `vector`, and `layer`. At least
two `layer` rows must represent consecutive cache appends. Every row must use
the same MLEN/BLEN/HLEN/VLEN geometry, geometry manifest, precision manifest,
compiler identity, and timing mode. Operation rows leave the five whole-layer
fields empty. Layer rows must provide positive compute, memory, and
physical-HBM evidence; `analytical_cycles` must equal their maximum.

Every `*_path` cell must name a real file, resolved relative to the anchor CSV.
The builder hashes those files itself; digest strings supplied in the CSV are
not trusted. `geometry_path` and `precision_path` bind the declared execution
configuration; the geometry file must be JSON containing matching `mlen`,
`blen`, `hlen`, and `vlen` values. `asm_path` binds the compiled program, and
the analytical, emulator, and RTL trace files must be canonical byte-identical
instruction traces for that row. The required compiler provenance file is hashed by the
builder and injected into every anchor. All anchors must share that compiler
digest and it must equal the artifact's `compiler` provenance digest.

Build immutable evidence:

```bash
python -m analytic_models.performance.build_timing_evidence \
  --mode rtl_serialized \
  --anchors evidence/decode_cycle_anchors.csv \
  --provenance compiler=evidence/compiler_trace.json \
  --provenance analytic=evidence/analytic_cycles.json \
  --provenance emulator=evidence/emulator_trace.json \
  --provenance rtl=evidence/rtl_trace.json \
  --out evidence/rtl_serialized_timing.json
```

For an emulator-tier artifact, pass `--mode emulator_serialized` and omit the
`rtl_cycles` column and the `rtl` provenance role; every other requirement is
unchanged.

The gate requires a maximum per-anchor cycle error of at most 5% and an
analytical latency MAPE of at most 10%. The per-anchor error is measured against
RTL in `rtl_serialized` mode and against the emulator in `emulator_serialized`
mode. Both limits are immutable constants: changing either, even while
recomputing the artifact identity, is rejected during loading.

An artifact stays unrankable if any of the following is true — a missing anchor
class; a missing raw provenance role (`compiler`, `analytic`, `emulator`, and
`rtl` for the RTL tier); a missing identity file or field; disagreement between
anchors on geometry, precision or compiler identity; per-row trace disagreement;
fewer than two consecutive layer rows; or either error gate exceeded. Legacy
artifacts without the complete identity contract fail closed.

Use the evidence for hardware search:

```bash
python analytic_models/performance/disagg_decode.py \
  --model qwen3-32b \
  --timing-mode rtl_serialized \
  --timing-evidence evidence/rtl_serialized_timing.json \
  --search
```

PackedKV capacity is evaluated independently of overlap. TPOT gains from the
pipeline oracle require a separate `ideal_matrix_pipeline` evidence artifact
produced after the pipelined implementation is validated.

## Batch and context crossover

A decode workload is not assigned a memory-bound label in advance. Increasing
batch amortizes streamed weights, while increasing context grows KV traffic.
The crossover artifact therefore evaluates both axes and records three times
for every fixed-context `q_len=1` point:

- ideal matrix-issue compute time;
- realized compute time for the selected execution contract;
- physical-byte memory time at the declared HBM operating point.

If memory exceeds the ideal issue time but not the realized serialized time,
the algorithmic label remains `memory` and the realized label is
`serialization`. Capacity-infeasible batches are retained but cannot define a
crossover. Build the artifact with a fixed chip count and explicit HBM point:

```bash
python -m analytic_models.performance.build_decode_crossover \
  --model-json compiler/doc/Model_Lib/qwen3-32b.json \
  --settings plena_settings.toml \
  --isa analytic_models/performance/customISA_lib.json \
  --timing-evidence evidence/rtl_serialized_timing.json \
  --hbm-gen HBM2 \
  --hbm-channels 16 \
  --chips 1 \
  --output-head-location external_bf16_service \
  --out evidence/qwen3_32b_decode_crossover.json
```

The external-head artifact has `decode_body_only` scope. Whole-model TPOT adds
the separately calibrated BF16 output-head service before ranking.

The target `MLEN=1024` configuration uses an MRAM row depth of 4096: four
compiler-addressable 1024×1024 BF16 tiles. SRAM capacity is counted as
`row_depth × MLEN × element_bytes`; treating row depth as a number of complete
tiles overstates capacity by a factor of MLEN and is rejected by the hardware
configuration and physical-ledger gates.

## Output-head boundary

`decode_bf16_unmodeled` preserves the analytical BF16 LM-head and vocabulary
selection sensitivity but cannot establish a native decode-chip realization.
`external_bf16_service` stops the decode ledger after final RMSNorm: LM-head
resident/streamed bytes, HBM traffic, cycles, FLOPs, and power events are
removed together. Embedding lookup and embedding storage remain on the decode
chip.

The external mode is valid only when the caller adds a checksum-bound remote
BF16 service contract. The simulator never imputes link, remote compute,
remote memory, selection, capacity, queueing, or energy costs.

The rankable service is a reserved endpoint on the existing prefill chip, not
a third device. Its artifact must prove one queue-free service instance at
that location. Sharing the endpoint with active prefill work requires separate
queue and interference measurements and remains unrankable without them.

The publication comparison retains both placements. The remote BF16 service is
the deployment path after its link, service, and capacity artifact passes. The
local BF16 head is a decode-chip sensitivity until native BF16 execution and
its resident weight cost are calibrated. A low-precision local head is an
accuracy ablation and does not inherit decoder-stack hardware validity.

Returning the next BF16 embedding with the selected token is a separate
untied-model sensitivity. It may remove the embedding table and row read from
decode HBM, but it needs its own measured response, memory, timing, energy, and
capacity contract before it can enter this boundary or deployment ranking.

## Admission boundary

Prompt K/V crosses in BF16 and is quantized once on decode admission. The
handoff model reports this only as an analytical TTFT sensitivity; it
contributes zero steady-state TPOT. Decode precision changes admitted-cache
bytes, not BF16 wire bytes. Default link and admission coefficients are not
publication-rankable, and a calibrated admission state requires a
content-addressed identity.

## HBM operating points

The main hardware ranking fixes the calibrated HBM2 2 Gb/s operating point.
HBM2E, HBM3, HBM3E, and HBM4 are controlled sensitivities on the final
profiles. Each preset couples rate, interface width, and representative stack
capacity in `disagg_serve/hbm_technology.py`.

The installed Ramulator HBM3 preset runs at 2 Gb/s, not the 6.4 Gb/s product
point. Its DMA measurements therefore cannot calibrate the HBM3 technology
sensitivity. A generation/rate mismatch drops the bandwidth calibration
identity and keeps the result unrankable rather than scaling the measured
bandwidth. HBM2E and HBM3E likewise remain peak-bandwidth sensitivities until
matched emulator or hardware measurements are supplied. HBM4 is represented
at a conservative 11 Gb/s lower bound for Micron's stated greater-than-11 Gb/s,
36 GB, 2048-bit device. It has no emulator calibration and cannot enter
deployment ranking.


## Emulator agreement

Trace calibration above needs matched compiler, emulator and RTL cycles and
stays unranked until it has them. A weaker but earned claim covers the analytic
per-stage terms on their own: `decode_stage_validation.py --emit-calibration`
writes an `emulator-calibrated` artifact when the analytic model agrees with the
transactional emulator stage by stage.

The gate is fail-closed on three limits, none of which was relaxed to pass:
worst stage error <= 5%, total error <= 15%, and uncovered fraction <= 1%,
where uncovered fraction is the share of measured decode-layer cycles that no
analytic term describes. Compiler-trace timing prices the persisted physical
request sidecar with the request-level Ramulator fit. Open HBM rows are carried
through the complete request stream, so the fitted row-conflict term accounts
for descriptor-boundary transitions without using cache length as a feature.
Measured on the `decoder_decode` program at MLEN=64/BLEN=4/HLEN=16, one
artifact per cache length in `calibration/`:

| cache length | measured layer cycles | worst stage | absolute error | coverage | label |
| ---: | ---: | --- | ---: | ---: | --- |
| 128 | 214,535 | KV store | 4.642% | 100.0% | emulator-calibrated |
| 256 | 278,725 | Activation load + RMSNorm | 3.851% | 100.0% | emulator-calibrated |
| 512 | 407,207 | Activation load + RMSNorm | 3.851% | 100.0% | emulator-calibrated |
| 1024 | 663,967 | Activation load + RMSNorm | 3.851% | 100.0% | emulator-calibrated |

Each row is read directly from the corresponding `calibration/decode_kv*.json`
artifact, whose `measured_layer_cycles` and `worst_stage_error` fields are
authoritative. Regenerating the artifacts will move these figures; the table
must be updated from them, never the other way round.

The worst stage is the publication-facing bound. The aggregate error remains an
internal fail gate in the content-addressed artifact, but is not presented as a
more accurate summary than its component stages.

Pass the artifact to a ranking run so the notice states what is and is not
calibrated:

```bash
python analytic_models/performance/disagg_decode.py --model qwen3-32b --search \
  --emulator-calibration analytic_models/performance/calibration/decode_kv1024.json
```

The notice then reports emulator agreement and says plainly that absolute cycle
counts are validated against the emulator but not against RTL. Without the
artifact it says the cycle counts are not validated at all. Neither wording
claims trace calibration.


## Measurement noise floor

The retained calibration set was regenerated after correcting the emulator HBM
wrapper width and rebuilding the release binary. Every artifact binds the new
executable and successful-run receipt by SHA-256; the kv=1024 artifact retains
the resulting 663,967-cycle run. The run-receipt validator recomputes the
SHA-256 digest of the executable named by the receipt, and that digest is also a
mandatory calibration provenance role.

Earlier same-binary repeat blocks differed by four cycles over approximately
664,000 cycles. That historical span motivates the conservative attribution
floor below, but the superseded receipts do not provide provenance for the
current executable.

The retained floor is **0.01% of a stage**. It conservatively covers the observed
cross-block timing variation for HBM attribution and configuration sensitivity;
within-block repeatability alone is not treated as universal determinism.



## State of trace calibration

Timing evidence exists and passes, at the **emulator tier**. The artifact is
`evidence/decode_timing_evidence.json`, built in mode `emulator_serialized` with
`evidence_tier: emulator`.

Every anchor's analytical cycle count is compared against the transactional
emulator running the identical instruction trace:

| anchor | kind | analytical cycles | emulator cycles | error |
| --- | --- | ---: | ---: | ---: |
| `linear-0` | linear | 2,117 | 2,117 | 0.00% |
| `qk-1` | qk | 2,885 | 2,885 | 0.00% |
| `pv-2` | pv | 2,885 | 2,885 | 0.00% |
| `vector-3` | vector | 581 | 581 | 0.00% |
| `layer-p2` | layer | 210,374 | 214,533 | 1.94% |
| `layer-p3` | layer | 242,720 | 246,604 | 1.57% |

Mean absolute percentage error is 0.586% against the 10% limit, and the worst
single anchor is 1.939% against the 5% limit. Neither limit was altered to
achieve this; both are the immutable constants described under *Evidence*
above. The two layer anchors are cache positions 2 and 3 at batch 64 — the
consecutive-append pair the contract requires — and all six anchors share one
geometry, precision manifest and compiler identity.

### What the operation anchors do and do not establish

The four operation anchors are standalone, DMA-free single-operation kernels.
Issue and drain timing is data-independent, so their cycle counts are exact,
which is why they agree to 0.00%. The corollary matters just as much: **they
validate issue timing only and say nothing about DMA pricing.**

That restriction was arrived at by measurement, not preference. The first
attempt used DMA-bearing `linear` and `vector` anchors, which missed the 5%
per-anchor limit at 6.25% and 20.96%. Enlarging the programs made the error
worse rather than better, which ruled out a fixed per-program overhead and
pointed at systematic mispricing of those DMA paths in the analytic model.
Rather than widen the gate, the operation anchors were reduced to pure issue
kernels — the approach already used for `qk` and `pv`.

DMA-inclusive evidence therefore rests entirely on the two whole-layer anchors,
at 1.94% and 1.57%, and on the request-level memory calibration described in
`disagg_serve/CALIBRATION_PROVENANCE.md`.

### Why the tier is emulator and not RTL

An RTL-tier artifact requires RTL cycle counts for the same traces. The anchor
geometry is MLEN=64, and RTL simulation does not currently complete there: the
`linear` workload passes at the committed MLEN=16 geometry but hangs at MLEN=64.
That was root-caused by elimination — the program itself runs to `C_BREAK` under
the emulator, a reverted memory-initialisation change fails identically, the
legacy code generator fails identically, the instruction image and the baked
offset parameter agree, the program is not truncated, the HBM model is large
enough, and a standalone read/write test of the storage passes. The cause is not
yet identified.

Because RTL cycles are unavailable, `emulator_rtl_error` is null in the
artifact and the tier is `emulator`. Anything quoting these numbers must say so.
Promoting the tier requires RTL execution at the anchor geometry, not additional
tooling.

### Historical RTL observations

Three RTL simulations were recorded earlier at the committed MLEN=16 geometry,
with E6M5 scalar, vector and matrix arithmetic and MXINT8 activation, weight and
KV blocks using 8-bit scales and block size 8:

| workload | shape | RTL cycles | numerical validation |
| --- | --- | ---: | --- |
| RMSNorm | batch 8, hidden 128, epsilon 1e-6 | 5,160 | PASS; 1,024 values, 100.00% match |
| linear | `(8 x 128) @ (128 x 256)` | 34,196 | PASS; 2,048 values, 98.54% match |
| fused GQA | sequence 16, KV sequence 16, 2 query heads, 1 KV head, head dimension 8, broadcast 2 | 13,564 | PASS; 256 values, 100.00% match |

These are raw observations at a different geometry from the anchors. **They
calibrate nothing**: none has a matched emulator cycle count and analytical
prediction for the identical instruction trace, and together they do not cover
the required anchor kinds. The fused GQA run in particular is not a substitute
for separate `qk` and `pv` anchors, because neither component has an
independently matched trace. They must not be passed to
`build_timing_evidence.py` as though the missing values existed.

The per-run content hashes that bind these observations to their generated
configuration, assembly, machine code, HBM image and logs are recorded in the
run artifacts themselves rather than reproduced here; the RTL source-tree digest
for any checkout can be recomputed with:

```bash
.venv/bin/python -c 'from analytic_models.power.calibration_workflow import build_rtl_source_manifest as build; print(build("../PLENA_RTL")["source_tree_sha256"])'
```

That census selects `.sv`, `.v`, `.svh` and `.vh` files under the RTL
definitions and RTL directories and under `TileLink_Lib`, excluding `bram.sv`
and `fake_hbm.sv`. It sorts the repository-relative paths, hashes each file, and
then hashes the canonical JSON map of path to hash.

### Builder and loader contract tests

Targeted tests drive `build_timing_evidence.py` with temporary raw files and
then break one requirement at a time. These exercise the tooling; they are
fixtures, not calibration evidence:

| anchor set | result |
| --- | --- |
| all five kinds, two consecutive layer rows, complete identities and raw roles, errors in bounds | `passed=True` |
| any geometry, precision, compiler, assembly or trace identity field removed | rejected as incomplete |
| geometry, precision or compiler changed on one row | `passed=False` |
| any analytical, emulator or RTL canonical trace file differs on one row | `passed=False` |
| any required compiler, analytic, emulator or RTL provenance file omitted | input rejected |
| any anchor CSV, geometry, precision, assembly or provenance byte changed at an existing output path | overwrite rejected |
| either error limit loosened | loader rejects the artifact, including after identity recomputation |
| legacy artifact without execution identities | loads only as `passed=False` |
| evidence used under a different timing mode | `timing_mode_mismatch` |
| no evidence supplied | `missing_timing_evidence` |

The builder refuses to overwrite a different artifact at the same path, so
evidence is immutable once written. The mutation test restores every changed
file afterwards and confirms its digest matches the pre-mutation bytes.
