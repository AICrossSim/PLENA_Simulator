from dataclasses import asdict
import pytest

from .model import Core, Design, Parameters, simulate
from .optimizer import universal_bound
from .sensitivity import SensitivityParameters, parameters_from_dict


def design():
    return Design((Core(4, 4, 512), Core(2, 4, 512)))


def test_tau_is_shared_operand_cap_not_arithmetic_issue_interval():
    d = design()
    p = SensitivityParameters(weight_tile_service_cycles=30.4)
    assert p.issue_interval == 1
    assert sum(p.w_bandwidth(d, c) for c in (0, 1)) == pytest.approx(4096 / 30.4)
    tight = Parameters(onchip_mode="port_tight")
    for c in (0, 1):
        assert p.w_bandwidth(d, c) == pytest.approx(tight.w_bandwidth(d, c))


def test_bank_width_and_tau_are_independent_limits():
    d = design()
    bank_limited = SensitivityParameters(bank_Bpc=8, weight_tile_service_cycles=1)
    tau_limited = SensitivityParameters(bank_Bpc=8, weight_tile_service_cycles=30.4)
    assert sum(bank_limited.w_bandwidth(d, c) for c in (0, 1)) == 512
    assert sum(tau_limited.w_bandwidth(d, c) for c in (0, 1)) == pytest.approx(4096 / 30.4)


@pytest.mark.parametrize("tau", [1, 4, 16, 30.4])
def test_looser_universal_bound_stays_below_executable_schedule(tau):
    w = {"id": "tau_unit", "batch": 4, "top_k": 1, "hidden": 512,
         "experts": [{"id": i, "Me": m, "H": 512, "F": 128, "is_shared": False}
                     for i, m in enumerate((1, 2, 4))]}
    p = SensitivityParameters(weight_tile_service_cycles=tau)
    assert universal_bound(w, p)["lb_cycles"] <= simulate(w, design(), p)["cycles"] + 1e-7


def test_certificate_parameters_roundtrip_and_reject_changed_source():
    p = SensitivityParameters(weight_tile_service_cycles=19, bank_Bpc=12)
    assert parameters_from_dict(asdict(p)) == p
    assert type(parameters_from_dict(asdict(Parameters()))) is Parameters
    values = asdict(p)
    values["timing_model_sha256"] = "stale"
    with pytest.raises(ValueError, match="hash changed"):
        parameters_from_dict(values)


def test_sensitivity_cannot_be_mislabeled_fixed_issue():
    with pytest.raises(ValueError, match="issue II"):
        SensitivityParameters(tile_issue_cycles=30.4)
