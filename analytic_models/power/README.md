# Compiler-Action Power Model

This package estimates energy from the same symbolic final schedule consumed by
the compiler-derived latency model. It groups ISA instructions into calibrated
hardware actions rather than assigning a separate wattage to every opcode.

```text
CostTrace + LatencyReport + ComponentPhysicalProperties
    -> logic action energy
    -> SRAM access energy
    -> ideal/ungated clock energy
    -> logic leakage
    -> external HBM physical-traffic/background energy
```

## API

```python
from analytic_models.power import estimate_action_energy, estimate_power

dynamic = estimate_action_energy(trace, hardware, coefficients)
system = estimate_power(trace, latency_report, physical_properties)
```

`estimate_action_energy` can be used without a chip-area artifact. Full average
power requires versioned `ComponentPhysicalProperties`; missing component area,
leakage, clock density, or SRAM access-energy inputs fail closed.

The primary clock result is `ideal-hierarchical`, an architectural lower bound
with perfect inactive-unit gating and no gate overhead. `ungated` is retained as
an upper bound. Neither result claims that clock gating exists in current RTL.

## Calibration boundary

`logic_energy_main_v1.json` retains only action families exercised by the
unmodified upstream ISA. Its underlying RTL-activity calibration used Qwen-like
VCD activity replayed on mapped ASAP7 DC netlists. The independent retained
holdout contains 50 points: 4 Matrix, 27 Vector, 16 Scalar, and 3 HBM-controller
actions. The median error is 7.47% and P95 is 22.13%.

`power_workload_coverage_main_v1.json` separately checks exact action-family
coverage on Qwen3-32B, Qwen3-235B balanced/skewed routing, LLaMA, and GPT-OSS.
This is a coverage test, not additional physical calibration.

External HBM3E coefficients are literature-parameterized. SRAM dynamic energy
uses compiler-described or main-ISA-implied accesses and caller-selected macro
energies. SRAM leakage, CTS, routed parasitics, package, cooling, board
regulation, multi-chip links, and NVLink are outside this package.
