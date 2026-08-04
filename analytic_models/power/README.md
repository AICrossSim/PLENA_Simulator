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

`analytic_models/disagg_serve/decode_power.py` is the live path. It is a
two-term model:

- HBM technology coefficients for the memory side;
- a single literature-anchored MAC-energy coefficient for the compute side.

Its reference point reproduces the output of a published analytic and synthesis
model. That makes it an analytic sensitivity — useful for comparing designs
against each other — and not measured-silicon or trace-calibrated power.

## Why this package must not be deleted

`PLENA_Software/decode_dse/hardware/evaluation.py` imports `power.model`, so the
package is a live dependency even while uncalibrated. It becomes the reported
path once `calibration/` is populated and the resulting fit passes its audit.
