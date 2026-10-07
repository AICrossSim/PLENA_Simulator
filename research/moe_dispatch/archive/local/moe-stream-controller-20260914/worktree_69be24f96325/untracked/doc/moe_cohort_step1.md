# Step1: charged legacy control and cohort bursts

This revision changes control service granularity. Native weight reads, packed and decoded slot capacities, operand stages, arithmetic order and accumulator service stay on their existing paths. All switches default off; `selection` defaults to `rotating` when event control is enabled.

Configuration:

```json
{
  "diagnostic": { "charge_legacy_control": false, "control_ports": 1 },
  "cores": [{
    "refinement": {
      "stream_ctrl": { "event_ready": true, "cohort_control": true, "selection": "rotating" }
    }
  }]
}
```

This is an excerpt, not a standalone architecture file. Native acceptance uses the archived full JSON files.

## Descriptor port and legacy control

`control_port.rs` provides the same semaphore and service-time implementation to the event pool and the diagnostic N3 path. N3 pays two core cycles per issue, three per operand tile install and two per completed accumulator writeback. Its total descriptor service is `4I + 3T`, for I block issues and T weight tiles. Native N3 with the flag off remains the frozen optimistic control-zero-charge baseline.

One port is the default. The optional second port has the same service width and timing. Operations addressing the same band retain mutual exclusion. A second 64 B descriptor latch is charged to accumulator SRAM, alongside the existing control records; no data port or HBM capability changes. This is a transactional timing/capacity diagnostic, not a physical area estimate. Summed service and the union of busy intervals are reported separately.

## Cohort issue and completion

A cohort contains `C = ceil(Me/Mt)` records; it is admitted whole. Event masks support C <= 32. The issuer selects a ready band and pays two descriptor cycles to start a burst. It then attempts its unissued M blocks in ascending order, spending one sequencer cycle per attempt, outside the descriptor port. Each attempt checks decoded/previous-K readiness. Skipped blocks get exactly one retry pass; any remaining blocks keep ownership of the same weight tile and wait for another burst.

Individual MAC results still complete through the original ordered accumulator port. Only the final issued writeback's completion reference is retained for notification; older references are dropped without cancelling their writebacks. One 16 B event carries the burst's relative cohort mask. Enqueue and completion ordinals are checked as observers. This does not introduce a local partial sum or relax previous-K completion dependencies.

Completion spends two descriptor cycles on burst state, then one cycle per eight cohort positions. Mask width, rather than popcount, determines cost. Pending counters are updated before readiness publication. A `completing` bit prevents band retirement until the final mask byte is published. The 32-bit unissued mask resides in the already budgeted band record; the burst's last-result reference uses an existing pending context record. The W-entry event queue remains bounded.

For B bursts, the complete charged service in core cycles is:

```
D = sum_bands(C + 1) + 8T + I + 4B + sum_bursts ceil(C/8)
```

Terms: band admission; tile load/decoded/bind/release (2+1+2+3); installed-ready fanout; burst start plus completion header (2+2); completion mask bytes. Sequencer cycles equal `I + not_ready_checks` and are reported separately. Old record control pays `sum_bands(C+1) + 8T + 5I`.

The fixed B32 small-core workload has mixed C, so the final term must be summed per expert/projection. With one burst per tile its expected service is 681,984 cycles, or 0.681984 ms at 1 ns/core cycle. Acceptance uses the measured count including any extra partial bursts, not this prediction.

## Selectors and evidence scope

`lowest` selects the lowest ready bit. `rotating` advances the cursor after the selected record (record mode) or selected band (cohort mode). `band_rotating` advances to the next band's first position. The additional selector diagnostic isolates N/M ordering on the record path; cohort mode already sequences all M consumers within a burst. The existing output-switch counter measures transitions between resident band slots, not every change of M record or logical output address.

The runner `scripts/moe_stream_ctrl/cohort_step1.py` executes each native point twice, checks all three numerical outputs (including FP32 bit patterns), identical bytes and MAC work, native drain, source/binary/input hashes, SRAM capacity and service formulas. Fixed B32 cases preserve both frozen expert ownership and order within each core. The validator has negative checks against missing completion/sequence costs, unfunded ports, free legacy service, missing diagnostic opt-ins and corrupted numerical/traffic results.

Default acceptance is one-port rotating cohort O2 versus one-port charged N3 on Me32, and small-core summed descriptor service < 0.7 ms under frozen B32 ownership. Lowest, band-first, O3 and two-port results are diagnostics. Frozen N3's 70.532 us is always reported separately. A failed gate stops Step1.
