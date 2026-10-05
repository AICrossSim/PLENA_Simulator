"""NoC profiles, model configs and the DeepStack import boundary."""

from __future__ import annotations

import copy
import json
import sys

import pytest
from distributed_fixtures import EXAMPLES, GPT_OSS_20B, LLAMA_3_1_8B

from analytic_models.distributed import ModelSpec, load_noc_profile, noc_from_dict
from analytic_models.distributed import _deepstack

EXAMPLE_DEVICES = {
    "illustrative_gpu_node_8": 8,
    "illustrative_gpu_cluster_32": 32,
    "illustrative_ring_switch_32": 32,
    "illustrative_torus_mesh_switch_64": 64,
}


@pytest.mark.parametrize(("name", "devices"), sorted(EXAMPLE_DEVICES.items()))
def test_example_profiles_load(name, devices) -> None:
    profile = load_noc_profile(EXAMPLES / f"{name}.json")
    assert profile.num_devices == devices
    description = profile.describe()
    assert [level["level"] for level in description["layers"]] == ["L3", "L2", "L1"]
    assert description["energy_pj_per_bit"] == {"l1": 1.0, "l2": 2.0, "l3": 3.0}
    assert len(profile.provenance["profile_sha256"]) == 64
    assert "Not a PLENA hardware specification" in profile.provenance["source"]


def test_examples_directory_has_no_unlisted_profiles() -> None:
    assert sorted(path.stem for path in EXAMPLES.glob("*.json")) == sorted(EXAMPLE_DEVICES)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda data: data.pop("layers"), ValueError, "missing"),
        (lambda data: data.update(colour="blue"), ValueError, "unknown"),
        (lambda data: data.update(kind="memory"), ValueError, "noc_hierarchy"),
        (lambda data: data.update(schema_version=2), ValueError, "schema_version"),
        (lambda data: data.update(energy_pj_per_bit={"l1": 1.0}), ValueError, "l1, l2 and l3"),
        (lambda data: data.update(provenance=[]), TypeError, "provenance"),
        (lambda data: data["layers"]["L1"].update(kind="hypercube"), ValueError, "unknown topology kind"),
    ],
)
def test_malformed_profiles_are_rejected(change, error, message) -> None:
    data = json.loads((EXAMPLES / "illustrative_gpu_node_8.json").read_text())
    broken = copy.deepcopy(data)
    change(broken)
    with pytest.raises(error, match=message):
        noc_from_dict(broken)


def test_model_families() -> None:
    llama = ModelSpec.from_hf_config(LLAMA_3_1_8B, name="llama")
    assert (llama.family, llama.has_moe, llama.head_dim) == ("llama", False, 128)
    gpt = ModelSpec.from_hf_config(GPT_OSS_20B, name="gpt-oss")
    assert (gpt.family, gpt.num_experts, gpt.experts_per_token, gpt.moe_intermediate_size) == ("moe", 32, 4, 2880)
    assert gpt.describe()["sliding_layers"] == 12
    routed = ModelSpec.from_hf_config({**LLAMA_3_1_8B, "n_routed_experts": 64, "num_experts_per_tok": 6}, name="r")
    assert (routed.family, routed.num_experts, routed.experts_per_token) == ("moe", 64, 6)
    with pytest.raises(ValueError, match="shared experts"):
        ModelSpec.from_hf_config({**LLAMA_3_1_8B, "n_routed_experts": 64, "n_shared_experts": 1}, name="s")
    with pytest.raises(ValueError, match="layer_types has 2 entries"):
        ModelSpec.from_hf_config({**GPT_OSS_20B, "layer_types": ["full_attention"] * 2}, name="bad")


def test_deepstack_comes_from_the_submodule_without_reference_binaries(monkeypatch, tmp_path) -> None:
    import mosaic

    root = _deepstack.deepstack_root().resolve()
    assert root.name == "DeepStack"
    assert mosaic.__file__.startswith(str(root))
    assert not [name for name in _deepstack._REFERENCE_BINARIES if name in sys.modules]
    monkeypatch.setenv(_deepstack.ENV_VAR, str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="git submodule update --init DeepStack"):
        _deepstack.deepstack_root()
