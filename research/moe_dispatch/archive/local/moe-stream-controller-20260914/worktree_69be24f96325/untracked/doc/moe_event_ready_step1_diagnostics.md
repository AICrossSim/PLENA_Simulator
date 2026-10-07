**Step1 diagnostics: control cost and fixed-work comparisons**

Step1 remains blocked. The normal rotating Q32/O2 case takes 78.070 us versus
the frozen N3's 70.532 us. At frozen B32 expert ownership/order, the small core
uses 1.078272 ms of scheduler service, above the unchanged 0.7 ms requirement.
No Step2 mechanism or dispatch-policy change is implemented here.

The N3 feedback bit is set only after `refined_port_work` finishes its
accumulator write service. The old pool has the same condition. The event
path additionally publishes and processes a completion event. Consequently
the suggested writeback-completion-to-enqueue semantic relaxation is not
applicable: it would relax both the numerical timing contract and the frozen
comparison rather than align them. All paths retain ascending K accumulation.

The normal controller and storage accounting are unchanged. New observer
counters record event counts, descriptor-update cycles, service, control-port
wait, FIFO residence, producer-to-update latency and actual event/MAC interval
intersection. Kind indices are decoded slot, installed operands, and K done.
Observer interval logs never influence allocation, readiness or timing; they
are simulation instrumentation, not an architectural queue or SRAM resource.

Two disabled-by-default `Architecture.diagnostic` fields support the requested
experiments:

- `allow_three_operand_stages`: permits O3 only on event-ready cores and still
  charges every operand byte and the additional 64-byte stage control record.
  The normal contract continues to reject O3. For the single (P,R,Mt)=(8,512,4)
  core, three combined MX/BF16 slots plus three BF16 operands reserve 62,976 B
  of the existing 65,536 B weight budget. Control records stay in accumulator
  SRAM; O2 uses 4,624 B including the existing 208-byte event supplement, O3
  uses 4,688 B.
- `event_updates_zero_cost`: a causal oracle for D2/D4 only, never an acceptance
  setting. It removes service time of event descriptor updates, retaining
  operation counts, FIFO capacity, the shared control-port mutex, writeback
  completion dependencies, other actors' costs, and memory/data operations.
  Actual timing changes downstream readiness/arbitration, so factorial gains
  are conditional net effects, not independent penalties.

At normal one-cycle descriptor service, the event controller uses
`sum_bands(C+1) + 8*tiles + 5*issues` scheduler cycles. Event handling alone uses
`tiles + 3*issues` cycles. The zero-event oracle removes only the latter term.
For Me32: 7,680 events, 19,200 event-update cycles, 40,320 total scheduler cycles.
8,847 event-service cycles overlap MAC service, measured by interval
intersection, not by subtracting summed actor times. Producer delay due to a
full event FIFO is zero in the rotating Me32 run; ordinary FIFO residence is
reported separately.

The ordered D4 decomposition in us is
`7.538 = 0 (D1) + 1.888 (D2 at O2) + 0.205 (O3 after D2) + 5.445 (residual)`.
O3 alone worsens the normal result by 0.870 us. The order interaction is
1.075 us; the report includes the alternative ordering. The residual is not
uniquely attributed. Two audited structural differences remain: N3 does not
charge separate scheduler descriptor cycles, and its N-group-first round
robin differs from the flat-Q rotating bitmap. Their Me32 output-switch counts
are 3,746 and 765. Neither observation justifies changing the frozen target.

`scripts/moe_stream_ctrl/diagnose_step1.py` creates eleven configurations and
runs each twice with the unchanged native library. The fixed B32 cases use
`diagnostic.fixed_job_order` extracted from the frozen adaptive run, preserving
both job ownership and within-core order. The runner verifies all three golden
output representations (including FP32 bit patterns), provenance, global
traffic, fixed per-core work, resources, native drain, repeatability and frozen
field compatibility. `report_step1_diagnostics.py` emits the tables and keeps
the task stopped at Step1.

The complete Chinese report, inputs, raw runs and reproducibility archive are
at [REPORT.md](/scratch/shared/mcl123/plena/outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7/step1/diagnostics_0ca4cbfb6004438ea1d86a9a5102090e/REPORT.md).
