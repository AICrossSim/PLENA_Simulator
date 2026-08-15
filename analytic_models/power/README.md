# Event-level power model (not yet calibrated)

This package is the synthesis-backed power path. It prices a decode run from a
per-event signature, using coefficients fitted to Design Compiler reports and
SAIF switching activity. `calibration/` holds the reproducible workflow that
produces those inputs.

**No power figure reported anywhere comes from this package.** `calibration/`
contains no measurements yet, so `model.py` raises an error on any event
signature it is asked to price: it describes structure, not watts. See
`calibration/README.md` for the workflow that would populate it.

## Where the reported power numbers actually come from

`analytic_models/disagg_serve/decode_power.py` is the live path. It sums five
separately reported terms — memory, compute, SRAM, leakage and link — whose
coefficients carry mixed evidence: published experimental HBM per-bit energies,
a median extracted from the vendored ASAP7 SRAM macro library, a
literature-anchored MAC energy, and a declared leakage density. Its component
provenance table is in
[`../disagg_serve/README.md`](../disagg_serve/README.md).

That makes it an analytic sensitivity — useful for comparing designs against
each other and stamped `analytic_anchored` on every row — and not
measured-silicon or trace-calibrated power. In particular it prices from a
coefficient set and a reference anchor, never from this package's per-event
synthesis signatures.

## The relationship between the two packages

They are different engines that share one tier vocabulary, not two settings of
one engine.

| | `disagg_serve/decode_power.py` | `power/` (this package) |
| --- | --- | --- |
| Input | aggregate traffic, MAC rate, area, chip count | a per-event signature stream from the simulator |
| Coefficients | literature, published memory data, macro-library extraction, declared leakage | fitted to Design Compiler reports and SAIF switching activity |
| Tier emitted | `analytic_anchored` | `dc_calibrated` |
| Status | live; every rankable study row prices here | uncalibrated; raises on any signature |

`decode_power.py` declares `dc_calibrated` as a tier constant but never
constructs one. The switch between the two engines happens on the study side:
`PLENA_Software/decode_dse/hardware/evaluation.py` builds its power engine from
`power.model.PowerCalibration` when — and only when — a power-calibration
artifact and an area configuration are supplied together, and the resulting
rows carry `dc_calibrated`. Neither engine's numbers are edited to resemble the
other's; a row is priced at one tier or the other and says which.

## Why this package must not be deleted

`PLENA_Software/decode_dse/hardware/evaluation.py` imports `power.model`, so the
package is a live dependency even while uncalibrated. It becomes the reported
path once `calibration/` is populated and the resulting fit passes its audit —
see [`calibration/README.md`](calibration/README.md) for the four-stage
library-manifest → schedule → external DC run → ingest-and-fit workflow, every
stage of which is fail-closed.
