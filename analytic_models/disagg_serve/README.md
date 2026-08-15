# Disaggregated serving models

Physical models for one PLENA decode chip in a disaggregated prefill/decode
deployment: HBM request latency, KV-cache traffic, byte-exact resource ledgers,
chip and system area, prefill-to-decode hand-off, and decode power.

Everything here is analytic. Each model names the evidence behind its numbers
and refuses to present a weaker tier as a stronger one. The memory calibration's
provenance — what was measured, with what receipt, and where its error tail lies
— is documented separately in
[`CALIBRATION_PROVENANCE.md`](CALIBRATION_PROVENANCE.md).

## Module map

| Module | Role |
| --- | --- |
| `memory.py` | Request-level HBM latency. Expands compiler DMA descriptors into 64-byte element and MX-scale requests against the pinned Ramulator2 HBM organisation, then prices a busiest-channel transfer floor plus non-negative ridge terms for startup, channel tail, bank serialisation, row misses and conflicts, and read-modify-write. Fitted separately per opcode × generation × channel count. Retains the aggregate effective-bandwidth path for compatibility, but **fails closed** when descriptors are supplied and the structured calibration is absent. |
| `hbm_technology.py` | The five HBM generations: pin rate, interface width, capacity, per-bit read/write energy, background power, and `emulator_rate_matches` — whether the calibration pin rate equals the production rate. `INTERFACE_UNIT_BITS = 64`. |
| `physical_ledger.py` | Aligned HBM and SRAM byte accounting for one configuration: weights, KV, SRAM residency, per-step traffic. Raises rather than rounding when a precision does not encode an integral shared scale. |
| `packed_kv.py` | Physical KV traffic for the four PackedKV ablation modes (padded per-head, dense compiler, dense selector, ideal), plus the architecture-option area for KV head reuse and drain overlap. |
| `area.py` | Bridge to the top-level `analytic_models/area` package. Dispatches `"calibrated"` (precision-aware structural census) or `"proxy"` (multiplier count × 0.237/4096 mm², tier `declared_proxy`), and threads `hbm_interface_units_per_chip` into `estimate_system_area` so chip-side HBM PHY is charged per attached interface unit. |
| `handoff.py` | BF16 prefill→decode KV transfer and admission cost. Canonical owner of `LINK_GENS` and `LINK_ENERGY_PJ_PER_BIT`; `decode_power.py` imports them rather than duplicating. |
| `decode_power.py` | Decode power and energy efficiency — see below. |
| `serve.py` | Thin re-export of the decode evaluator that currently lives in `analytic_models/performance/disagg_decode.py`. |
| `calibration_provenance.py` | The fail-closed audit over every calibration artifact in this directory. |

## `decode_power.py`

Five terms, always reported separately, never folded into one another:

```
P_total = P_memory + P_compute + P_sram + P_leakage + P_link
P_memory = rho_bg * C + e_read * BW_read + e_write * BW_write
```

Which term dominates is a property of the operating point, not a standing claim
about decode being memory bound in *time*; that is why the split is preserved
all the way to `DecodePower.summary()`.

### Component provenance

Each coefficient carries a different quality of evidence. The table is the whole
point of this section: a reader must be able to tell which numbers are
characterised and which are declared.

| Term | Coefficient | Value | Where it comes from |
| --- | --- | --- | --- |
| Memory, dynamic | `read_energy_pj_per_bit`, `write_energy_pj_per_bit` per generation | HBM2 4.2/5.0 … HBM4 2.2/2.4 | Published experimental data (MemExplorer, Table 1), with the pre-HBM3E generations scaled from the reported points by the factors recorded in each entry's `energy_source_label`. |
| Memory, background | `background_power_mw_per_gb` | 75.0 | Midpoint of the reported 50–100 mW/GB range. Identical across generations, and labelled as a midpoint. |
| SRAM read | `SRAM_ACCESS_ENERGY_PJ_PER_BIT` | ≈ 0.0479 pJ/bit | **Median of the vendored ASAP7 macro library internal-power extraction** (`sram_energy_asap7_v1.json`): rise+fall VDD conditional-clock internal power over a 1 ns period, TT / 0.7 V / 25 C, 36 macros, extracted per bit of macro width. Computed at import from the artifact, not hard-coded. |
| Compute | `REFERENCE_MAC_ENERGY_PJ` | 0.203 pJ/MAC at 4-bit operands | Literature-anchored: the residual of a reported 300.09 W whole-chip figure after subtracting this module's own memory prediction at the declared reference assumptions, divided by the reference MAC rate and scaled to 4 bits. `calibrate_reference_mac_energy()` re-derives it; the stored value reproduces the anchor within 0.1%. |
| Leakage | `LOGIC_LEAKAGE_W_PER_MM2` | 0.05 W/mm² | **Declared, and bounded below by measurement.** No complete-chip leakage report exists in this workspace, and the ASAP7 extraction records `sram_leakage_status: "unavailable"`. A gate-level campaign measured 9.21e-04 W/mm² — see below — but at 25 °C on the compute array alone, so it is recorded as a scoped lower bound in `MATRIX_MACHINE_LEAKAGE_MEASUREMENT`, not adopted. |
| Link | `LINK_ENERGY_PJ_PER_BIT` | 1.5–10.0 pJ/bit by generation | Declared inter-package/on-board link sensitivity, owned by `handoff.py`. |

The SRAM coefficient replaced a scaled textbook (Horowitz constant-field)
estimate of 0.0243 pJ/bit, which sat at the optimistic edge of the extracted
0.023–0.114 pJ/bit range. The replacement roughly doubles the SRAM term. Two
consequences are worth stating:

- the coefficient is now a characterised macro figure rather than a scaled
  anchor, but it is still *macro library internal power, not PLENA netlist
  power* — `SRAM_ENERGY_SOURCE.evidence_scope` says exactly that; and
- `analytic_energy_identity()` hashes the coefficient set, so replacing the
  artifact changes the energy identity stamped on every priced row. That is the
  intended behaviour: rows priced under different coefficients are not
  interchangeable.

### Gate-level evidence, and what it did and did not change

A Design Compiler campaign on `matrix_machine` (8 timing-closed points, ASAP7
RVT_TT at `PVT_0P7V_25C`, 1000 ps, MLEN 16–64, MXFP) touches two coefficients
here. Both outcomes are recorded; neither coefficient moved. The measurements and
the derived record live under `analytic_models/area/calibration/`.

**Scope guard.** Those points are one block, in µm², at MLEN 16–64, at 25 °C.
Every coefficient in the table above is charged against a *full chip* in mm² at
MLEN 128–1024. Nothing below licenses comparing the two directly.

*Leakage — a real contradiction, resolved conservatively.* Leakage tracked area
to 1.44% across all eight points at 9.2097e-07 mW/µm² = **9.21e-04 W/mm²**,
roughly **54× below** the declared 0.05 W/mm². The measurement is sound; it is
the wrong corner and the wrong scope to replace the coefficient with:

- **Temperature.** 25 °C is the coldest corner in the library. Subthreshold
  leakage rises steeply with junction temperature and a datacentre part runs at
  85–125 °C. The campaign synthesised no hot corner, so the derating between the
  measured point and an operating point is unmeasured; any factor applied would
  be an assumption, and it is not declared as one here.
- **Scope.** The measured design is 98.3% dense compute array. The coefficient is
  charged against whole-chip non-memory logic, whose cell mix, utilisation and
  threshold-voltage distribution differ from a systolic datapath.

So 0.05 W/mm² stays. It is *conservative with respect to* the measurement rather
than contradicted by it: the 25 °C array density bounds realistic leakage from
below, and the declared value sits above that bound. Adopting the 25 °C figure
would make every design look better on a corner mismatch.

The sensitivity says this costs almost nothing either way. At a representative
decode point — MLEN 1024, BLEN 8, MXINT4, 8 chips on HBM3E at 70% bandwidth
utilisation, 0.93 mm² of logic per chip — the leakage term is **0.20% of total
power** at 0.05 W/mm² and **0.004%** at the measured density; adopting the
measurement would move tokens/joule by **+0.2%**. Even at the most
leakage-favourable point that can be constructed (MLEN 4096, MXINT8, minimal
traffic, 5.2 mm² of logic) the share is 5.4% against 0.10%, a 5.6% swing. The
54× coefficient gap is real and is nowhere near a 54× power gap, because leakage
is a small term at every operating point the decode study reaches.

The upgrade path is explicit in `MATRIX_MACHINE_LEAKAGE_MEASUREMENT`: re-report
the same mapped netlists at a hot operating condition and extend the campaign to
the full chip. That replaces a declared coefficient with a measured one at
matching corner and scope, and needs no API change.

*Compute energy — independent corroboration.* The same campaign priced two
mapped netlists over a declared toggle envelope of 0.05–0.50, giving 0.113–1.126
pJ/MAC. `REFERENCE_MAC_ENERGY_PJ = 0.203` falls inside that envelope at an
implied toggle of **0.0797** (32×4) and **0.0835** (16×4) — consistent across
geometries, 0.073–0.093 across all six swept — which is a genuine independent
check on the module's most load-bearing coefficient. It is recorded as
`COMPUTE_ENERGY_CROSS_CHECK` with `coefficient_changed: False`, because the
toggle rate is *assumed and propagated by the synthesis tool*, not measured from
decode switching. It brackets the anchor; it does not calibrate it, and the tier
stays `analytic_anchored`.

Both records are hashed into `analytic_energy_identity()`, so the evidence a row
was priced under travels with the row.

### Tier ladder

`ENERGY_TIERS` has two rungs:

- **`analytic_anchored`** — what this module emits. `decode_power()` stamps
  `energy_tier=ANALYTIC_ENERGY_TIER` and `energy_id=analytic_energy_identity()`,
  a content hash over the reference configuration and every coefficient above.
  This is the tier every rankable decode-study row carries today.
- **`dc_calibrated`** — declared here but never constructed here. Nothing in
  this package produces a `dc_calibrated` estimate; the rung exists so a
  DC-calibrated coefficient set can be substituted without changing the
  estimator API or any caller.

The DC-calibrated path is a *different engine*, not a different setting of this
one: it is `analytic_models/power` (see that package's
[`README.md`](../power/README.md)), reached from the decode study by supplying a
power-calibration artifact and an area configuration together. The two paths
meet only at the shared tier vocabulary.

A caution for anyone editing the coefficients: the test suite pins the
literature anchor, the quadratic operand-width scaling, chip-count replication
and the link-table identity, but **nothing asserts the SRAM median, the macro
count, or the energy identity digest**. A change to
`sram_energy_asap7_v1.json` will move `SRAM_ACCESS_ENERGY_PJ_PER_BIT` and every
`energy_id` without failing a test.

### Scaling and composition

`decode_power(...)` multiplies the memory, compute, SRAM and leakage terms by
`chip_count` exactly once each; the link term is already system-wide and is not
scaled. `analytic_models/performance/disagg_decode.py` passes system-total rates
divided by chip count, and gates the SRAM and leakage inputs on an explicit
topology — under the legacy aggregate path those two terms are zero, so an
analytic total from that path is not comparable to one from an explicit `TP × KVP`
topology.

## Verifying the calibration artifacts

```bash
python -m analytic_models.disagg_serve.calibration_provenance \
  --output results/calibration/memory-calibration-audit.json
```

The audit re-derives the expected observation count from the declared sweep
axes, re-hashes every retained CSV, harness, receipt and validation file,
re-verifies the historical commits out of git, and replays every recorded
process against its CSV row. Every check raises; there are no warn-and-continue
paths.
