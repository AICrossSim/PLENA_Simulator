# Hybrid L-Compute executable evidence

This artifact is the functional companion to the official-dimension analytic
campaign. It runs generated machine code in the Rust transactional emulator;
it does not use a GPU, checkpoint, analytic latency substitution, or RTL.

It proves three separate properties:

1. Matrix final writeback can place real values with an affine layout and a
   Vector consumer restores the original lane order.
2. S128 Mamba and KDA prefill state can feed an affine-packet decode update at
   the bank bandwidth floor with zero conflict stalls.
3. B=1/2/4/8/16 requests keep independent recurrent state. The batch tests use
   reduced outer dimensions so every state value can be checked; official
   dimensions and full 52/93-layer timing are covered by the analytic artifact.

| Case | Cycles | ISA lines | Max abs error | Result |
|---|---:|---:|---:|---|
| Matrix affine writeback | 6,789 | 654 | 0 | pass |
| Mamba S128 handoff | 62,723 | 1,363 | 0.002441 | pass |
| KDA S128 handoff | 1,398,237 | 19,132 | 0.001587 | pass |
| Mamba private state B16 | 204,886 | 66,128 | 0.007812 | pass |
| KDA private state B16 | 236,129 | 34,456 | 0.002319 | pass |

Both handoff cases report `packet_service_cycles ==
packet_bandwidth_floor_cycles == 320`, `packet_conflict_stall_cycles == 0`,
and a 100% allclose match rate. Intermediate batch sizes B1/B2/B4/B8 are also
present in `evidence.json`.

Reproduce from the Simulator root:

```bash
PYTHONPATH="$PWD:$PWD/PLENA_Compiler" \
python3 -m analytic_models.performance.hybrid_connected_evidence \
  --compiler-root PLENA_Compiler \
  --json-out artifacts/hybrid_lcompute_connected_v1/evidence.json
```

The prefill chunks themselves use the existing chunked implementation. The
affine packet path starts at the prefill-to-decode state handoff and executes
the following decode update. This artifact therefore proves compatibility and
bank service for the handoff; it does not claim a packetized-prefill speedup.
