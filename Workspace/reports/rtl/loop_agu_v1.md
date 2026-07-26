# PLENA Six-Stream Loop AGU v1

## Purpose

The native Qwen3 compiler emits a large number of affine GP-register updates
inside hardware loops. Under the project-wide `ideal-ii1` timing model, each
`S_ADDI_INT` and dynamic `C_LOOP_END` still costs one cycle. The six-stream
loop AGU removes only address updates that can be proven affine; it does not
change Matrix, Vector, HBM, or numerical execution.

The implemented path consists of:

- `C_AGU_BIND gpAddress, stride` to bind up to six affine streams per loop.
- `C_AGU_LOOP_LEN body_instruction_count` to identify the static loop marker.
- `C_LOOP_START_AGU gpCounter, iteration_count` to start a zero-overhead loop.
- Four nested descriptor frames and a 16-entry GP affine-offset sidecar.
- Six parallel stride adders at each accepted loop boundary.
- Architectural GP reads as `regfile value + affine offset`.
- Offset clearing on an ordinary write to the corresponding GP register.

The legacy loop ISA and compiler mode remain available for controlled A/B
comparison.

## Compiler Legality

The optimization runs after immediate legalization. It recognizes a net
`S_ADDI_INT rd, rd, immediate` update only when:

1. the update executes exactly once per loop iteration;
2. it occurs after the last read of that GP register in the loop body;
3. the loop body contains no other conflicting write;
4. the net stride is exactly representable by the signed 17-bit mantissa and
   5-bit shift encoding; and
5. replacing it with AGU setup has positive dynamic instruction savings.

Large-immediate instruction chains are combined before checking the stride.
At most six streams are selected by dynamic benefit. Exact repeated compiler
microkernels may be refolded into an AGU loop without changing operation order.

Updates in the middle of a loop body are intentionally retained. Moving such
an update to the boundary would change later addresses in the same iteration.
This safety condition is why the realized result is slightly lower than the
initial unconstrained counterfactual.

## Reference Configuration

```text
Model:          Qwen3-32B, 64 decoder layers
Sequence:       482
Batch:          16
MLEN/VLEN:      2048/2048
BLEN/HLEN:      1024/128
Compute timing: ideal-ii1
Memory timing:  HBM service V4
Clock:          1 ns reporting assumption
```

## End-to-End Result

| Metric | Legacy addressing | Loop AGU v1 | Delta |
|---|---:|---:|---:|
| Compute work | 2,209,339,175 cycles | 1,729,248,609 cycles | -480,090,566 |
| Stage-roofline latency | 2.20935 s | 1.72927 s | -21.73% |
| One-layer compute work | 34,931,471 cycles | 27,315,390 cycles | -21.80% |
| Scalar work | 1,050,974,721 cycles | 645,173,380 cycles | -38.61% |
| Control work | 84,146,694 cycles | 9,857,469 cycles | -88.29% |
| Matrix work | 639,888,384 cycles | 639,888,384 cycles | unchanged |
| Vector work | 434,329,376 cycles | 434,329,376 cycles | unchanged |

The stage attribution is:

| Stage | Cycles removed |
|---|---:|
| Layer attention | 396,015,488 |
| Layer FFN | 83,958,592 |
| Global input load | 99,926 |
| Global final norm | 16,227 |
| Mask and RoPE setup | 333 |

The AGU trace contains 5,000 transformed loop sites and 1,049 exact
compile-time refolds. Dynamic `C_LOOP_END` execution falls from 83,641,094 to
45,571 instructions, a 99.945% reduction. The remaining fallback census is 54
loops without a recoverable body length and 283 loops for which setup was not
profitable.

The final trace still contains 180,959,227 dynamic `S_ADDI_INT` instructions.
Post-analysis attributes about 145.49M of the residual affine updates to
registers read both before and after the update in the same loop iteration.
They cannot be moved to a v1 loop boundary without changing program semantics.

## Workload Invariants

The controlled A/B reports identical values for:

```text
Matrix compute work:  639,888,384 cycles
HBM V4 work:           70,018,931.498 ns
HBM physical reads:    89,199,280,128 bytes
HBM physical writes:   17,448,304,640 bytes
HBM read requests:      1,393,738,752
HBM write requests:       272,629,760
```

Therefore the latency reduction is attributable to fewer address/control
instructions, not to hidden Matrix, QK/PV, or HBM traffic changes.

## RTL Integration

The RTL adds:

- the two reserved control opcodes at `0x3e` and `0x3f`;
- descriptor and offset state in `loop_agu_state`;
- zero-overhead boundary redirection in the decoder and loop controller;
- GP read-side affine offset addition in the ScalarMachine;
- an assertion against explicit writes to bound streams.

The decoder treats a marker as a boundary only after the final body
instruction is accepted. A marker observed while a pipeline or operand-helper
stall is active remains unconsumed. This prevents offsets from advancing
before the last instruction has resolved its old-iteration address.

## Area and Timing Delta

Small mapped DC measurements use ASAP7 TT 0.7 V, 25 C and a 1 ns constraint.
No large Matrix or full-chip synthesis was required.

| Block | Area | Critical path | WNS |
|---|---:|---:|---:|
| Baseline loop controller | 104.932 um2 | 615.67 ps | +263.39 ps |
| AGU loop controller | 273.623 um2 | 877.99 ps | +1.39 ps |
| Six-stream affine sidecar | 1,722.569 um2 | 879.85 ps | +0.14 ps |
| Combined paired delta | 1,891.259 um2 | 879.85 ps | +0.14 ps |

`area_new` exposes the combined value as `AddressGenerationUnit`. The result
is a pre-layout mapped standard-cell estimate. Both blocks satisfy the 1 ns
constraint, but the sub-2 ps WNS margin is too small to claim robust 1 GHz
physical closure. Decoder integration glue was elaborated in the full PLENA
top but was not isolated as an additional paired area delta.

## Power Status

CostEmitter now emits:

```text
agu_config
agu_loop_setup
agu_loop_boundary
agu_stream_step
agu_offset_read
```

The standalone AGU was calibrated with RTL VCD activity replayed on its mapped
DC netlist. All 13 scenarios completed with 100% sequential SAIF coverage.
The resulting Qwen-like nominal action energies are:

| Action | Energy |
|---|---:|
| Configure one stream | 0.03079 pJ |
| Start one loop frame | 0.43580 pJ |
| Step one active stream | 0.14225 pJ |
| Resolve one GP read | 0.02288 pJ |
| Additional boundary residual | 0 pJ after nonnegative fitting |

The zero boundary residual does not mean that an active boundary is free: its
measured energy is explained by the active stride adders and resolved-read
paths. The six-stream boundary slope across 32/128/512 repetitions has
`R2=0.99999999`. Low-toggle and random activity are respectively 0.966x and
1.027x the Qwen-like energy. The model reports
`rtl_activity_mapped_dc_candidate`; this is still pre-CTS, RTL-activity
evidence rather than gate-level or signoff power.

A one-trial production DSE smoke at `MLEN=VLEN=2048, BLEN=1024` completed
through CostEmitter, V4, area, multi-chip system energy, and the power model.
Its trial record reports `loop_agu_action_energy_v1`, the validation metadata
above, no unknown energy actions, and `power_status=complete`.

## Verification

Completed checks include:

- ISA encoding and signed shifted-stride round trips.
- Zero, one, three, and six selected streams in compiler analysis.
- Safe fallback for non-tail and conflicting GP writes.
- Large-immediate chain folding and exact microkernel refolding.
- Legacy compiler mode byte identity.
- CostTrace dynamic/static opcode reconstruction.
- Emulator positive/negative strides, marker skipping, internal `gp0` counter,
  nested-loop state, and ordinary-write offset clearing.
- Full compiler/area/power/performance focused regression: 165 tests passed.
- Rust AGU tests: 5 passed; all-target `cargo check` passed.
- Full PLENA DC elaboration after integration.
- End-to-end packed-GQA compiler/emulator run:
  - `batch=4`, `seq=7`, `MLEN=VLEN=16`, `BLEN=HLEN=4`;
  - generated assembly contained 335 AGU setup/start instructions;
  - the emulator observed 448 zero-overhead boundaries with old/new affine
    offsets recorded in schema-v4 event traces;
  - numerical comparison passed with a 100% allclose match rate and unchanged
    correctness thresholds; and
  - the AGU and legacy runs produced byte-identical 128 MiB VRAM dumps
    (`SHA-256 77320d357f23258bdd627d6f0328961016c57bc6e81d33be403eb705dead8fad`).

The transactional emulator and compiler tests establish address/control
semantics. They do not make `ideal-ii1` a cycle-exact claim; that mode remains
an explicitly hazard-free architectural timing assumption.

## Acceptance Status

| Requirement | Result |
|---|---|
| Dynamic loop-end elimination at least 99% | Pass: 99.945% |
| End-to-end ideal-II1 reduction at least 20% | Pass: 21.73% |
| Matrix work unchanged | Pass |
| HBM traffic unchanged | Pass |
| Latency at most 1.72 s | Miss: 1.72927 s, 9.27 ms above target |
| Mapped area and WNS reported | Pass |
| Calibrated AGU action energy | Pass: mapped-DC RTL-activity candidate |

The earlier 1.69829 s counterfactual is not used as the final result. It
treated middle-of-body affine updates as boundary updates and was therefore
not compiler-safe. The measured 1.72927 s result is the implementable v1
estimate.
