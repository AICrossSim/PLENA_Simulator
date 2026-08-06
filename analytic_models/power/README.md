# PLENA Power Models

This package estimates both fixed-corner, pre-layout on-chip energy and an
external-memory system-energy shadow. The on-chip model uses compiler hardware
actions:

```text
E_onchip =
    E_logic_dynamic
  + E_sram_dynamic
  + E_logic_leakage
  + E_sram_background
P_onchip_average = E_onchip / makespan
```

The supported corner is ASAP7 TT, 0.7 V, 25 C, and a 1000 ps clock. The scope
includes MatrixMachine, VectorMachine, ScalarMachine, frontend/control,
on-chip HBM controller logic, and SRAM macros.

The system estimator adds a literature-parameterized HBM3E-equivalent model:

```text
E_external_hbm =
    background_power_per_GB * capacity_GB * makespan
  + read_energy_per_bit * physical_read_bits
  + write_energy_per_bit * physical_write_bits

E_system = E_onchip + E_external_hbm
```

The main DSE fixes capacity and interface bandwidth to an abstract
`80 GB / 2039 GB/s` A100-aligned reference while using HBM3E energy
coefficients. This is deliberately named `abstract_80gb_a100_aligned`; it is
not presented as an integer number of 24 GB HBM3E stacks.

## Fidelity Status

`logic_energy_v2.json` is the active DSE shadow artifact. It passed the v2
RTL-activity/mapped-DC promotion gates with
`rtl_activity_calibrated_candidate_v2` status. `logic_energy_v1.json` remains
available as a compatibility bootstrap and is not used by default. The SRAM
dynamic-energy table is independently derived from all 36 public ASAP7 SRAM
Liberty files. Those files report zero leakage. Capacity-dependent SRAM
background power therefore uses the lower endpoint from MemExplorer Table 1:

```text
E_sram_background = allocated_macro_capacity_GB * 10 W/GB * makespan
```

Allocated capacity includes macro tiling padding and selected physical port
copies. The source reports a 10--50 W/GB experimental range; the active model
deliberately uses 10 W/GB. This is labelled a literature-parameterized lower
endpoint rather than an ASAP7 leakage calibration or statistical P10.

The first RTL-activity fit is retained as a rejected v1 candidate. It proved
that the replay flow is usable, but also showed that a single action coefficient
cannot represent low-toggle, Qwen-like, and random data activity. The v2
candidate therefore uses Qwen-like activity as its nominal P50 estimate,
per-family low/random slopes as an empirical envelope, and component mapped
area for idle clock scaling. The accepted fit has 7.47% grouped-holdout median
error and 20.88% P95 error; the worst Qwen-mix error is 11.41% and the cached
estimator median is 3.21 ms.

The on-chip estimator always returns `calibration_status`, exclusions,
warnings, and P10/P50/P90 uncertainty. Power remains a shadow metric. The v2 promotion level
is deliberately named `rtl_activity_calibrated_candidate_v2`: RTL VCD activity
is replayed on mapped DC netlists, but gate-level simulation, CTS, routing
parasitics, and intrinsic ASAP7 SRAM leakage characterization are not part of
this phase.

## Clock-Gating Semantics

The model exposes two clock-power interpretations:

```text
ungated
    Every mapped logic component is clocked for the full roofline makespan.
    This preserves the original pre-CTS upper bound.

ideal_hierarchical
    Only the subcomponent area required by each compiler action is clocked
    during that action's selected compute-timing/V4 occupancy window.
```

`estimate_onchip_power()` keeps `ungated` as its API-compatible default.
`estimate_system_power()` and DSE use `ideal_hierarchical` by default. The
ideal mode assumes zero idle-clock energy, zero gate overhead, zero wake-up
latency, and perfect inactive-lane/slice gating. It is an architectural lower
bound; no corresponding clock gates have been implemented or validated in the
current RTL.

CostEmitter emits compressed `ClockWork` records for Matrix array/reduction/
conversion/buffer paths, Vector lane and reduction paths, Scalar FP/INT/lane
paths, and V4 HBM-controller service windows. Masked Vector operations must
carry an exact lane mask. Missing lane coverage or a missing DMA service window
makes ideal clock work unavailable instead of silently charging a guessed
fraction. `FullChipTopResidual` remains in logic leakage but is excluded from
ideal clock energy because it mixes wrapper/interconnect area with model
residual.

Every ideal result retains the corresponding ungated energy and power,
component active fractions, subcomponent area-cycles, clock savings, and
unmodeled residual area. P10/P50/P90 continue to represent action-activity and
HBM-background envelopes; they do not assign artificial uncertainty to the
ideal gating assumption.

The default DSE compute timing is `ideal-ii1`. Dynamic action-energy
coefficients are unchanged, but Vector/Scalar/control ClockWork occupancy is
one cycle per action; Matrix occupancy continues to use structural RTL-v3
timing and HBM-controller occupancy uses V4 service windows. Leakage and HBM
background use the resulting stage-roofline makespan. Reports mark both compute
timing and clock gating as architectural ideal assumptions.

External-memory coefficients are kept separately in
`calibration/external_memory_hbm3e_v1.json`. They use 50/75/100 mW/GB
background power and 3.0/3.6 pJ/bit read/write energy. For the default 80 GB
capacity, background power is therefore 4/6/8 W. Dynamic energy is charged
against V4 physical 64-byte line traffic, including read-modify-write reads,
not against logical tensor bytes.

The model does **not** assume full bandwidth utilization. It reports:

```text
achieved_average_bandwidth =
    (physical_read_bytes + physical_write_bytes) / makespan

bandwidth_utilization =
    achieved_average_bandwidth / configured_interface_bandwidth
```

The configured 2039 GB/s remains the existing feasibility reference. It is not
multiplied by runtime to manufacture traffic or dynamic energy.

## API

```python
from analytic_models.power import (
    estimate_external_hbm_power,
    estimate_onchip_power,
    estimate_system_power,
)

onchip = estimate_onchip_power(
    hardware_and_precision_config,
    compiler_cost_trace,
    compiler_cost_report,
    clock_gating_mode="ungated",  # API-compatible upper bound
)

external = estimate_external_hbm_power(
    {
        "HBM_CAPACITY_BYTES": 80_000_000_000,
        "HBM_BANDWIDTH_GBPS": 2039,
    },
    compiler_cost_report,
)

system = estimate_system_power(
    hardware_and_precision_config,
    compiler_cost_trace,
    compiler_cost_report,
    clock_gating_mode="ideal_hierarchical",
    external_memory_config={
        "HBM_CAPACITY_BYTES": 80_000_000_000,
        "HBM_BANDWIDTH_GBPS": 2039,
    },
)
```

`CostTrace.energy_actions` is generated from the same structured lowering that
produces opcode counts and DMA events. Loops remain compressed. SRAM and HBM
actions therefore do not depend on parsing assembly text. The V4 memory report
supplies physical and payload read/write bytes, split by precision role, stage,
and opcode. The system estimator validates each physical-byte breakdown before
using it.

## Calibration

Safely inventory and remove only stale PLENA-owned `/tmp` artifacts first:

```bash
python analytic_models/power/scripts/cleanup_power_tmp.py \
  --manifest Workspace/area_new_power_calibration/tmp_cleanup.json
```

The cleanup tool never removes unknown paths, other users' files,
`/tmp/moe_models_e13`, `/tmp/bc625`, or paths held open by active processes.
Inspect the manifest before adding `--apply`.

Generate the v2 plan without invoking DC or Verilator:

```bash
python analytic_models/power/scripts/run_rtl_activity_power_calibration.py \
  --plan-version v2 \
  --dry-run --component all \
  --run-dir /tmp/plena_power_dry
```

The current v2 plan contains 31 mapped configurations and 395 activity replay
jobs. It reuses all 16 v1 mapped DDCs and maps only 15 new configurations. The
new scenarios isolate Matrix array/reduction/conversion, Vector operation
variants (including single- and multi-segment reduction), Scalar operation
families, DMA directions, and frontend issue work.

The built-in runner has three independent resource pools:

```text
mapped synthesis -> Verilator RTL activity -> DC SAIF power replay
```

It maps each configuration once with normal 1 ns synthesis, retains
DDC/Verilog/SDF/SDC/timing/name-map reports, compiles one Verilator binary per
configuration, and reuses it for all nine scenarios. Every completed VCD is
immediately queued for replay; activity generation does not wait for all map
jobs to finish. Mapping and replay share the DC-license semaphore, while a
separate memory/disk gate throttles heavy Vector/HBM Verilator builds.

Run the full v2 candidate collection with the measured resource limits:

```bash
nix develop --command bash -lc ' \
  source .venv/bin/activate; \
  python analytic_models/power/scripts/run_rtl_activity_power_calibration.py \
  --plan-version v2 \
  --component all \
  --map-workers 8 \
  --activity-workers 6 \
  --heavy-activity-workers 4 \
  --power-workers auto \
  --reserve-licenses 1 \
  --cpu-capacity 60 \
  --verilator-jobs 4 \
  --memory-reserve-gib 16 \
  --tmp-reserve-gib 8 \
  --reuse-mapping-run \
    Workspace/area_new_power_calibration/runs/full_rtl_activity_v1_20260722_193606 \
  --run-dir Workspace/area_new_power_calibration/runs/rtl_activity_v2 \
  --resume \
  --copy-to-calibration'
```

DC feature-checkout failures, including SEC-50 and DCSH-1, retry only the
affected job. The runner records negative slack as `timing_unclosed`, removes
worker-local synthesis products after archiving, and keeps an append-only raw
CSV plus a latest-complete compact CSV. PWR-414 is always fatal. PWR-415 is
accepted only when the quantitative `saif_map -rtl_summary` sequential-cell
coverage meets `--min-sequential-saif-coverage-pct` (90% by default), and the
warning and exact coverage remain in every row.

Fit a candidate after each VCD has a sibling `.vcd.actions.json` containing
the exact structural feature counts exercised by the test. The sidecar keeps
incremental action features separate from continuously clocked hardware:

```json
{
  "dynamic_features": {"vector.lane_add_sub_bit": 4096},
  "clock_features": {"vector_lane": 8192}
}
```

DC's pre-CTS `clock_network` group is recorded separately from register and
combinational activity. Action slopes use the matched active-minus-idle
non-clock residual. The model adds that residual to an idle baseline scaled by
mapped component area. Raw DC total energy and the excluded
activity-dependent clock-group delta remain in the CSV for audit; no negative
residual is clipped.

Train VCDs are removed after successful replay. Qwen, mixed-kernel, holdout,
and failed VCDs are retained (successful validation VCDs are gzip-compressed).
Publishing to `analytic_models/power/calibration/` is opt-in so disposable
smokes cannot overwrite the canonical compact dataset. Even when requested,
publication is skipped unless every selected point/scenario has a latest
successful result; incomplete runs retain only their run-local compact CSV and
can be resumed without partially replacing the canonical dataset.

Fit the v2 candidate with:

```bash
python analytic_models/power/scripts/fit_power_calibration_v2.py \
  --points Workspace/area_new_power_calibration/runs/rtl_activity_v2/power_calibration_points.csv \
  --output Workspace/area_new_power_calibration/runs/rtl_activity_v2/logic_energy_v2_candidate.json \
  --validation-output Workspace/area_new_power_calibration/runs/rtl_activity_v2/power_validation_v2.json \
  --envelope-output Workspace/area_new_power_calibration/runs/rtl_activity_v2/power_activity_envelope_v2.csv \
  --markdown-output Workspace/area_new_power_calibration/runs/rtl_activity_v2/power_validation_v2.md
```

Add `--promote` only after inspecting diagnostics. Promotion is refused unless
action and idle-clock holdout median error is at most 15%, P95 is at most 30%,
mixed/Qwen total dynamic error is at most 20%, every random slope has
R-squared at least 0.95, all CostEmitter action families are covered, and
mapped leakage/area references exist. The canonical run passed every gate; the
same refusal logic protects future refits from partially replacing the active
artifact. Gate-level validation is explicitly `not_run_by_scope`; this remains
a calibrated candidate, not a signoff power model.

## DSE

`run_optuna_dse.py` enables `--power-shadow` by default. It records separate
on-chip, external-HBM, and system energy/power fields, as well as physical to
payload traffic ratio, achieved bandwidth, utilization, uncertainty, scope,
exclusions, and warnings. The on-chip clock/leakage baseline and HBM background
energy are recomputed over the same roofline makespan. No power value is added
to the Optuna objectives or constraints.

The DSE option is:

```text
--clock-gating-mode {ideal-hierarchical,ungated}
```

Its default is `ideal-hierarchical`. Latency, area, accuracy, and HBM traffic
are independent of this option. Trial records retain both selected and ungated
on-chip/system power so the lower/upper-bound sensitivity is always auditable.

The system scope is:

```text
onchip_logic+sram+controller+external_hbm3e_equivalent
```

It still excludes package, cooling, board-regulator loss, KV-link energy, CTS,
routed parasitics, and SRAM leakage. The external coefficient boundary may
include some stack-side I/O energy, while the on-chip controller model excludes
a calibrated PHY; reports preserve this boundary uncertainty rather than
claiming signoff accuracy.

On a production 64-layer trace, the final clock inventory contains 311
compressed records with 100% Matrix/Vector/Scalar/HBM action coverage.
ClockWork construction has a 6.83 ms median cached runtime, and the complete
ideal on-chip estimator has a 19.63 ms median runtime on the development host.
Neither path expands the dynamic ISA.
