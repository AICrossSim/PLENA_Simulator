"""The stage comparison must describe the program the dump was generated for.

Every dimension of the comparison used to come from an argparse default. Only
attention scales with the cache length, so a dump generated at one cache length
and compared against a model evaluated at another produced five clean stages and
one that appeared to be off by 8x — which reads as a modelling error in that
stage rather than as a mismatched comparison. These tests pin the shape to the
run manifest the dump is written with.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_stage_validation import (
    ANALYTIC_STAGE_KEYS,
    MANIFEST_DIMENSIONS,
    MANIFEST_NAME,
    assert_op_stats_current,
    calibration_execution_contract,
    compiler_trace_stage_cycles,
    require_complete_calibration_stages,
    run_shape,
    summarize_stage_validation,
)
from perf_model import ffn_decode_auxiliary_histogram
from decode_timing import (
    CANONICAL_ANALYTICAL_MAPE_LIMIT,
    CANONICAL_ANCHOR_MAX_ERROR_LIMIT,
    EMULATOR_SERIALIZED,
    CycleAnchor,
    RTL_SERIALIZED,
    TimingEvidence,
)


class Requested:
    """Stand-in for the parsed arguments, with every dimension unset."""

    def __init__(self, **overrides: int) -> None:
        for name, _location, _flag in MANIFEST_DIMENSIONS:
            setattr(self, name, 0)
        for name, value in overrides.items():
            setattr(self, name, value)


MANIFEST = {
    "kv_size": 1024,
    "inter": 128,
    "kv_head_reuse": False,
    "vocab": 256,
    "geometry": {
        "mlen": 64,
        "blen": 4,
        "hlen": 16,
        "batch": 64,
        "hidden": 64,
        "head_dim": 16,
        "query_heads": 4,
        "kv_heads": 1,
        "fp_sram_depth": 512,
    },
}


def _dynamic_opcode_histogram(assembly: str) -> Counter[str]:
    """Expand constant-count hardware loops into dynamic opcode counts."""
    counts: Counter[str] = Counter()
    loop_counts: list[int] = []
    for raw_line in assembly.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        opcode = line.split()[0]
        multiplier = 1
        for trip_count in loop_counts:
            multiplier *= trip_count
        counts[opcode] += multiplier
        if opcode == "C_LOOP_START":
            loop_counts.append(int(line.rsplit(",", 1)[1].strip()))
        elif opcode == "C_LOOP_END":
            loop_counts.pop()
    assert not loop_counts
    return counts


@pytest.mark.parametrize(
    "mlen,vlen,blen,rows,hidden_size,intermediate_size",
    ((64, 64, 4, 64, 64, 128), (16, 16, 4, 8, 32, 64)),
)
def test_ffn_auxiliary_count_follows_the_looped_emitter(
    mlen: int,
    vlen: int,
    blen: int,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    from compiler.asm_templates.ffn_asm import ffn_asm

    assembly = ffn_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=rows,
        seq_len=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        alive_registers=list(range(1, 12)),
        gate_weight_hbm_offset_reg=1,
        up_weight_hbm_offset_reg=2,
        down_weight_hbm_offset_reg=3,
        const_one_fp_address=5,
        activation_base_address=4096,
        use_loop_instructions=True,
        workspace_base_address=8192,
    )
    emitted = _dynamic_opcode_histogram(assembly)
    derived = ffn_decode_auxiliary_histogram(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rows=rows,
    )
    assert {opcode: emitted[opcode] for opcode in derived} == derived

    if (mlen, rows, hidden_size, intermediate_size) == (64, 64, 64, 128):
        assert derived == {
            "S_ADDI_INT": 6289,
            "S_ADD_INT": 80,
            "S_LD_FP": 1,
            "C_SET_SCALE_REG": 2,
            "C_SET_STRIDE_REG": 2,
            "C_LOOP_START": 89,
            "C_LOOP_END": 1493,
            "H_PREFETCH_M": 6,
        }


@pytest.fixture
def dump(tmp_path: Path) -> Path:
    op_stats = tmp_path / "op_stats.jsonl"
    op_stats.write_text("")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(MANIFEST))
    return op_stats


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_receipt(build: Path, asm: Path, settings: Path) -> None:
    from transactional_emulator.testbench.emulator_runner import (
        _behavior_config_summary,
    )

    op_stats = build / "op_stats.jsonl"
    op_stats.write_text('{"aggregate":true,"total_dt_ps":0}\n', encoding="utf-8")
    artifacts = {}
    for filename, field in (
        ("generated_asm_code.asm", "asm_source_sha256"),
        ("generated_machine_code.mem", "machine_code_sha256"),
        ("hbm_for_behave_sim.bin", "hbm_preload_sha256"),
        ("fp_sram.bin", "fp_sram_preload_sha256"),
        ("int_sram.bin", "int_sram_preload_sha256"),
    ):
        path = build / filename
        if filename == "generated_asm_code.asm":
            path.write_bytes(asm.read_bytes())
        else:
            path.write_bytes(filename.encode("utf-8"))
        artifacts[field] = _sha256(path)
    for filename in ("vram_dump.bin", "comparison_params.json", "golden_result.txt"):
        (build / filename).write_bytes(filename.encode("utf-8"))
    command = [sys.executable]
    receipt = {
        "schema_version": 2,
        "build_dir": str(build.resolve()),
        "command": command,
        "config_path": str(settings.resolve()),
        "config_sha256": _sha256(settings),
        "emulator_binary_sha256": _sha256(Path(sys.executable)),
        "behavior_config": _behavior_config_summary(settings, command),
        "return_code": 0,
        "artifact_complete": True,
        "numerical_validation_passed": True,
        "sim_latency_ns": 0,
        "op_stats_path": str(op_stats.resolve()),
        "op_stats_sha256": _sha256(op_stats),
        "vram_dump_sha256": _sha256(build / "vram_dump.bin"),
        "run_manifest_sha256": _sha256(build / MANIFEST_NAME),
        "comparison_params_sha256": _sha256(build / "comparison_params.json"),
        "golden_result_sha256": _sha256(build / "golden_result.txt"),
        "artifacts": artifacts,
    }
    (build / "rust_emulator_run_stats.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_shape_comes_from_the_manifest(dump: Path) -> None:
    shape = run_shape(dump, Requested())
    assert shape["kv_size"] == 1024
    assert shape["mlen"] == 64
    assert shape["heads"] == 4
    assert set(shape) == {name for name, _, _ in MANIFEST_DIMENSIONS}


def test_calibration_emission_requires_every_decode_stage() -> None:
    names = tuple(stage for stage, _keys in ANALYTIC_STAGE_KEYS)
    modelled = {stage: 100 for stage in names}
    measured = {stage: {"scalar": 100} for stage in names}
    require_complete_calibration_stages(modelled, measured)

    for missing in names:
        mutated = dict(measured)
        mutated.pop(missing)
        with pytest.raises(ValueError, match="measured missing") as raised:
            require_complete_calibration_stages(modelled, mutated)
        assert missing in str(raised.value)


def test_publication_acceptance_uses_worst_stage_and_complete_coverage() -> None:
    names = tuple(stage for stage, _keys in ANALYTIC_STAGE_KEYS)
    measured = {stage: {"scalar": 100} for stage in names}
    passing = summarize_stage_validation(
        {stage: 105 for stage in names},
        measured,
    )
    assert passing.meets_target()

    failing_error = summarize_stage_validation(
        {**{stage: 100 for stage in names}, names[-1]: 106},
        measured,
    )
    assert not failing_error.meets_target()

    measured["Setup"] = {"scalar": 8}
    failing_coverage = summarize_stage_validation(
        {stage: 100 for stage in names},
        measured,
    )
    assert not failing_coverage.meets_target()


def test_compiler_dma_trace_requires_persisted_request_sidecar(
    tmp_path: Path,
) -> None:
    asm = tmp_path / "generated_asm_code.asm"
    asm.write_text(
        "; Load_Batch X -> VRAM\n"
        "H_PREFETCH_V gp1, gp2, a0, 0, 0\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(RuntimeError, match="persisted address-resolved"):
        compiler_trace_stage_cycles(
            asm,
            root / "plena_settings.toml",
            root / "analytic_models/performance/customISA_lib.json",
            mlen=64,
            blen=4,
            hlen=16,
            timing_mode="rtl_serialized",
        )


def test_decoder_traffic_uses_issue_origin_not_waiter_attribution(
    tmp_path: Path,
) -> None:
    from transactional_emulator.testbench.emulator_runner import (
        _decode_hbm_read_ledger,
    )

    (tmp_path / "generated_asm_code.asm").write_text(
        "; Packed K prefetch for cache row\n"
        "H_PREFETCH_M gp1 gp2 ha0 0 KV\n"
        "; Packed V prefetch for cache row\n"
        "H_PREFETCH_M gp1 gp2 ha1 0 KV\n",
        encoding="utf-8",
    )
    (tmp_path / "generated_machine_code.mem").write_text("00000000\n00000000\n", encoding="utf-8")
    op_stats = tmp_path / "op_stats.jsonl"
    records = (
        {
            "pc": 0,
            "op": "H_PREFETCH_M",
            "dt_ps": 1,
            "hbm_rd": 0,
            "hbm_wr": 0,
            "hbm_issue_rd": 128,
            "hbm_issue_wr": 0,
        },
        {
            "pc": 1,
            "op": "H_PREFETCH_M",
            "dt_ps": 1,
            "hbm_rd": 384,
            "hbm_wr": 0,
            "hbm_issue_rd": 256,
            "hbm_issue_wr": 0,
        },
        {
            "aggregate": True,
            "total_hbm_rd": 384,
            "total_hbm_issue_rd": 384,
        },
    )
    op_stats.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    metrics = {
        "hbm_bytes_read": 384,
        "artifacts": {"asm_source_sha256": "a" * 64},
        "config_sha256": "b" * 64,
        "op_stats_sha256": "c" * 64,
        "run_manifest_sha256": "d" * 64,
    }
    ledger = _decode_hbm_read_ledger(tmp_path, metrics)
    assert ledger["key_bytes"] == 128
    assert ledger["value_bytes"] == 256
    assert ledger["issue_origin_bytes"] == ledger["global_bytes"] == 384

    mutated = [dict(record) for record in records]
    mutated[0].pop("hbm_issue_rd")
    op_stats.write_text(
        "".join(json.dumps(record) + "\n" for record in mutated),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks issue-origin"):
        _decode_hbm_read_ledger(tmp_path, metrics)

    inactive = (
        {
            **records[1],
            "hbm_rd": 384,
            "hbm_issue_rd": 384,
        },
        records[2],
    )
    op_stats.write_text(
        "".join(json.dumps(record) + "\n" for record in inactive),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lack positive dynamic issue traffic"):
        _decode_hbm_read_ledger(tmp_path, metrics)


def test_decode_reference_retains_checksum_bound_kv_traffic_evidence() -> None:
    repository = Path(__file__).resolve().parents[2]
    document = (
        repository / "transactional_emulator/doc/decode_reference.md"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"<!-- decode-kv-traffic-evidence:start -->\n```json\n(.*?)\n```\n"
        r"<!-- decode-kv-traffic-evidence:end -->",
        document,
        re.DOTALL,
    )
    assert match is not None
    evidence = json.loads(match.group(1))
    observed_digest = evidence.pop("aggregate_sha256")
    canonical = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == observed_digest
    assert evidence["schema"] == "plena-decode-kv-traffic-evidence-v1"
    assert evidence["settings_sha256"] == _sha256(repository / "plena_settings.toml")
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["emulator_binary_sha256"])
    assert evidence["command_prefix"][1:] == [
        "transactional_emulator/testbench/misc/decoder_decode_test.py",
        "--kv-size",
        "128",
    ]
    assert evidence["environment"]["PLENA_SETTINGS_TOML"].endswith(
        "/PLENA_Simulator/plena_settings.toml"
    )
    assert evidence["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    behavior_contract = evidence["behavior_contract"]
    assert {
        name: behavior_contract[name]
        for name in ("MLEN", "VLEN", "BLEN", "HLEN", "FP_SRAM_DEPTH")
    } == {
        "MLEN": 64,
        "VLEN": 64,
        "BLEN": 4,
        "HLEN": 16,
        "FP_SRAM_DEPTH": 512,
    }
    assert behavior_contract["DRAIN_OVERLAPPED"] is False
    assert behavior_contract["HBM_GEN"] == "HBM2"
    assert behavior_contract["HBM_CHANNELS"] == 8
    assert set(behavior_contract["PRECISION"]) == {
        "MATRIX_SRAM_TYPE",
        "VECTOR_SRAM_TYPE",
        "HBM_M_WEIGHT_TYPE",
        "HBM_M_KV_TYPE",
        "HBM_V_ACT_TYPE",
        "HBM_V_KV_TYPE",
    }
    behavior_digest = hashlib.sha256(
        json.dumps(
            behavior_contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    cells = evidence["cells"]
    assert [cell["cell"] for cell in cells] == [
        "hkv1_default",
        "hkv1_reuse",
        "hkv2_default",
        "hkv2_reuse",
        "hkv4_default",
        "hkv4_reuse",
    ]
    digest_fields = (
        "asm_sha256",
        "machine_code_sha256",
        "hbm_preload_sha256",
        "op_stats_sha256",
        "receipt_sha256",
        "run_manifest_sha256",
        "comparison_params_sha256",
        "golden_result_sha256",
        "vram_sha256",
    )
    for cell in cells:
        assert cell["allclose_pass"] is True
        assert cell["key_bytes"] + cell["value_bytes"] + cell[
            "non_attention_bytes"
        ] == cell["global_bytes"]
        assert cell["key_bytes"] == cell["value_bytes"]
        assert all(re.fullmatch(r"[0-9a-f]{64}", cell[field]) for field in digest_fields)
        arguments = cell["arguments"]
        assert arguments[:4] == [
            "--kv-heads",
            str(cell["kv_heads"]),
            "--softmax-row-tile",
            "4",
        ]
        assert ("--kv-head-reuse" in arguments) is cell["reuse"]
        assert arguments[-1].endswith("/" + cell["cell"])
        assert cell["behavior_contract_sha256"] == behavior_digest

    for default, reuse in zip(cells[::2], cells[1::2], strict=True):
        assert default["kv_heads"] == reuse["kv_heads"]
        assert default["non_attention_bytes"] == reuse["non_attention_bytes"]
        assert default["vram_sha256"] == reuse["vram_sha256"]
        assert default["hbm_preload_sha256"] == reuse["hbm_preload_sha256"]
        assert default["comparison_params_sha256"] == reuse[
            "comparison_params_sha256"
        ]
        assert default["golden_result_sha256"] == reuse["golden_result_sha256"]
        assert reuse["key_bytes"] == 262144
        assert default["key_bytes"] == reuse["key_bytes"] * default["kv_heads"]


def test_every_dimension_is_covered(dump: Path) -> None:
    """A dimension left out of the manifest check can still be defaulted."""
    shape = run_shape(dump, Requested())
    for name in (
        "kv_size",
        "inter",
        "vocab",
        "mlen",
        "blen",
        "hlen",
        "batch",
        "hidden",
        "head_dim",
        "heads",
        "kv_heads",
    ):
        assert name in shape, f"{name} is not pinned to the manifest"


@pytest.mark.parametrize(
    "name,value",
    [("kv_size", 128), ("mlen", 128), ("blen", 8), ("batch", 32), ("inter", 256)],
)
def test_a_contradicting_request_is_refused(dump: Path, name: str, value: int) -> None:
    with pytest.raises(SystemExit) as raised:
        run_shape(dump, Requested(**{name: value}))
    message = str(raised.value)
    assert name in message and str(value) in message
    assert "not a validation" in message


def test_an_agreeing_request_is_accepted(dump: Path) -> None:
    shape = run_shape(dump, Requested(kv_size=1024, mlen=64))
    assert shape["kv_size"] == 1024


def test_a_missing_manifest_is_refused(tmp_path: Path) -> None:
    op_stats = tmp_path / "op_stats.jsonl"
    op_stats.write_text("")
    with pytest.raises(SystemExit) as raised:
        run_shape(op_stats, Requested())
    assert MANIFEST_NAME in str(raised.value)


def test_no_shape_dimension_has_a_nonzero_default() -> None:
    """A non-zero default would let the shape come from anywhere but the dump.

    This is the wiring, not the helper: `run_shape` can be correct while nothing
    calls it, and a restored default is exactly how the mismatch reappeared.
    """
    from decode_stage_validation import build_parser

    defaults = {
        action.dest: action.default
        for action in build_parser()._actions
        if action.dest in {name for name, _, _ in MANIFEST_DIMENSIONS}
    }
    assert set(defaults) == {name for name, _, _ in MANIFEST_DIMENSIONS}, (
        f"a shape dimension has no command-line flag: {defaults}"
    )
    offenders = {name: value for name, value in defaults.items() if value}
    assert not offenders, f"these dimensions can be defaulted, not read: {offenders}"


def test_main_takes_its_shape_from_the_manifest(dump: Path, capsys) -> None:
    """End to end: the header must report the manifest's cache length."""
    import decode_stage_validation as validation

    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("; RMS Norm generation\n")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    _write_run_receipt(dump.parent, asm, settings)
    argv = [
        "decode_stage_validation.py",
        "--asm",
        str(asm),
        "--op-stats",
        str(dump),
        "--settings",
        str(settings),
        "--isa-lib",
        str(Path(__file__).resolve().parent / "customISA_lib.json"),
    ]
    original = sys.argv
    try:
        sys.argv = argv
        validation.main()
    finally:
        sys.argv = original
    header = capsys.readouterr().out.splitlines()[0]
    assert f"kv={MANIFEST['kv_size']}" in header, header
    assert f"MLEN={MANIFEST['geometry']['mlen']}" in header, header


def test_calibration_execution_contract_fails_closed_on_mismatch(dump: Path) -> None:
    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("; RMS Norm generation\n")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    _write_run_receipt(dump.parent, asm, settings)
    receipt = json.loads((dump.parent / "rust_emulator_run_stats.json").read_text(encoding="utf-8"))
    contract = calibration_execution_contract(receipt, MANIFEST, settings, "rtl_serialized")
    assert contract.fp_sram_depth == 512
    assert contract.drain_overlapped is False

    timing_mutation = json.loads(json.dumps(receipt))
    timing_mutation["behavior_config"]["DRAIN_OVERLAPPED"] = True
    with pytest.raises(ValueError, match="drain behavior"):
        calibration_execution_contract(timing_mutation, MANIFEST, settings, "rtl_serialized")

    depth_mutation = json.loads(json.dumps(MANIFEST))
    depth_mutation["geometry"]["fp_sram_depth"] = 1024
    with pytest.raises(ValueError, match="FP SRAM depths"):
        calibration_execution_contract(receipt, depth_mutation, settings, "rtl_serialized")

    precision_mutation = json.loads(json.dumps(receipt))
    precision_mutation["behavior_config"]["PRECISION"]["HBM_M_KV_TYPE"]["block"] = 16
    with pytest.raises(ValueError, match="precision contracts disagree"):
        calibration_execution_contract(precision_mutation, MANIFEST, settings, "rtl_serialized")


def test_stage_validation_requires_a_successful_bound_run_receipt(dump: Path) -> None:
    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("S_ADDI_INT gp1 gp0 1\n", encoding="utf-8")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    with pytest.raises(SystemExit, match="receipt is missing"):
        assert_op_stats_current(dump, asm, settings)


def test_stage_validation_rejects_an_unvalidated_run_receipt(dump: Path) -> None:
    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("S_ADDI_INT gp1 gp0 1\n", encoding="utf-8")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    _write_run_receipt(dump.parent, asm, settings)
    receipt_path = dump.parent / "rust_emulator_run_stats.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["numerical_validation_passed"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SystemExit, match="no successful numerical validation"):
        assert_op_stats_current(dump, asm, settings)


def test_stage_validation_rejects_a_receipt_without_binary_provenance(
    dump: Path,
) -> None:
    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("S_ADDI_INT gp1 gp0 1\n", encoding="utf-8")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    _write_run_receipt(dump.parent, asm, settings)
    receipt_path = dump.parent / "rust_emulator_run_stats.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("emulator_binary_sha256")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SystemExit, match="emulator binary digest"):
        assert_op_stats_current(dump, asm, settings)


def test_stage_validation_rejects_a_receipt_for_a_different_binary(
    dump: Path,
) -> None:
    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("S_ADDI_INT gp1 gp0 1\n", encoding="utf-8")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    _write_run_receipt(dump.parent, asm, settings)
    receipt_path = dump.parent / "rust_emulator_run_stats.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["emulator_binary_sha256"] = "1" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SystemExit, match="binary digest mismatch"):
        assert_op_stats_current(dump, asm, settings)


def test_stage_validation_rejects_tampered_emulator_input(dump: Path) -> None:
    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("S_ADDI_INT gp1 gp0 1\n", encoding="utf-8")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    _write_run_receipt(dump.parent, asm, settings)
    (dump.parent / "fp_sram.bin").write_bytes(b"different preload")
    with pytest.raises(SystemExit, match=r"hash mismatch.*fp_sram.bin"):
        assert_op_stats_current(dump, asm, settings)


def test_stage_validation_rejects_instruction_distribution_tampering(
    dump: Path,
) -> None:
    asm = dump.parent / "generated_asm_code.asm"
    asm.write_text("S_ADDI_INT gp1 gp0 1\n", encoding="utf-8")
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    _write_run_receipt(dump.parent, asm, settings)
    dump.write_text(
        '{"pc":0,"op":"S_ADDI_INT","dt_ps":1000}\n{"aggregate":true,"total_dt_ps":0}\n',
        encoding="utf-8",
    )
    receipt_path = dump.parent / "rust_emulator_run_stats.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["op_stats_sha256"] = _sha256(dump)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SystemExit, match="do not sum"):
        assert_op_stats_current(dump, asm, settings)


def test_receipt_recomputes_issue_origin_ledger_before_accepting(
    tmp_path: Path,
) -> None:
    from transactional_emulator.testbench.emulator_runner import (
        RUN_RECEIPT_SCHEMA,
        _behavior_config_summary,
        _decode_hbm_read_ledger,
        validate_emulator_run_receipt,
    )

    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    asm = tmp_path / "generated_asm_code.asm"
    asm.write_text(
        "; Packed K prefetch for cache row\n"
        "H_PREFETCH_M gp1 gp2 ha0 0 KV\n"
        "; Packed V prefetch for cache row\n"
        "H_PREFETCH_M gp1 gp2 ha1 0 KV\n"
        "H_PREFETCH_V gp1 gp2 ha2 0 KV\n",
        encoding="utf-8",
    )
    machine = tmp_path / "generated_machine_code.mem"
    machine.write_text("00000000\n00000000\n00000000\n", encoding="utf-8")
    op_stats = tmp_path / "op_stats.jsonl"
    records = (
        {
            "pc": 0,
            "op": "H_PREFETCH_M",
            "dt_ps": 1,
            "hbm_rd": 0,
            "hbm_wr": 0,
            "hbm_issue_rd": 128,
            "hbm_issue_wr": 0,
        },
        {
            "pc": 1,
            "op": "H_PREFETCH_M",
            "dt_ps": 1,
            "hbm_rd": 0,
            "hbm_wr": 0,
            "hbm_issue_rd": 256,
            "hbm_issue_wr": 0,
        },
        {
            "pc": 2,
            "op": "H_PREFETCH_V",
            "dt_ps": 1,
            "hbm_rd": 512,
            "hbm_wr": 0,
            "hbm_issue_rd": 128,
            "hbm_issue_wr": 0,
        },
        {
            "aggregate": True,
            "total_hbm_rd": 512,
            "total_hbm_issue_rd": 512,
        },
    )
    op_stats.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    for filename in (
        "hbm_for_behave_sim.bin",
        "fp_sram.bin",
        "int_sram.bin",
        "vram_dump.bin",
        "comparison_params.json",
        "golden_result.txt",
    ):
        (tmp_path / filename).write_bytes(filename.encode("utf-8"))
    manifest = tmp_path / MANIFEST_NAME
    manifest.write_text(json.dumps({"hbm_read_ledger_required": True}), encoding="utf-8")
    command = [sys.executable]
    artifacts = {
        "asm_source_sha256": _sha256(asm),
        "machine_code_sha256": _sha256(machine),
        "hbm_preload_sha256": _sha256(tmp_path / "hbm_for_behave_sim.bin"),
        "fp_sram_preload_sha256": _sha256(tmp_path / "fp_sram.bin"),
        "int_sram_preload_sha256": _sha256(tmp_path / "int_sram.bin"),
    }
    receipt = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "build_dir": str(tmp_path.resolve()),
        "command": command,
        "config_path": str(settings.resolve()),
        "config_sha256": _sha256(settings),
        "emulator_binary_sha256": _sha256(Path(sys.executable)),
        "behavior_config": _behavior_config_summary(settings, command),
        "return_code": 0,
        "artifact_complete": True,
        "numerical_validation_passed": True,
        "sim_latency_ns": 0,
        "hbm_bytes_read": 512,
        "op_stats_path": str(op_stats.resolve()),
        "op_stats_sha256": _sha256(op_stats),
        "vram_dump_sha256": _sha256(tmp_path / "vram_dump.bin"),
        "run_manifest_sha256": _sha256(manifest),
        "comparison_params_sha256": _sha256(tmp_path / "comparison_params.json"),
        "golden_result_sha256": _sha256(tmp_path / "golden_result.txt"),
        "artifacts": artifacts,
    }
    receipt["hbm_read_ledger"] = _decode_hbm_read_ledger(tmp_path, receipt)
    receipt_path = tmp_path / "rust_emulator_run_stats.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    validate_emulator_run_receipt(tmp_path, settings_file=settings)

    receipt["hbm_read_ledger"]["key_bytes"] += 64
    receipt["hbm_read_ledger"]["non_attention_bytes"] -= 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash-bound issue-origin evidence"):
        validate_emulator_run_receipt(tmp_path, settings_file=settings)


def test_build_directory_rejects_an_overlapping_process(tmp_path: Path) -> None:
    """A second process cannot observe or replace an active run's artifacts."""
    root = Path(__file__).resolve().parents[2]
    child = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from transactional_emulator.testbench.emulator_runner import acquire_build_directory",
            "lease = acquire_build_directory(Path(sys.argv[1]))",
            "print('ready', flush=True)",
            "sys.stdin.readline()",
            "lease.release()",
        )
    )
    python_path = os.pathsep.join(path for path in (str(root), os.environ.get("PYTHONPATH", "")) if path)
    environment = {**os.environ, "PYTHONPATH": python_path}
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        from transactional_emulator.testbench.emulator_runner import (
            acquire_build_directory,
        )

        with pytest.raises(RuntimeError, match="Concurrent or interleaved"):
            acquire_build_directory(tmp_path)
    finally:
        assert process.stdin is not None
        process.stdin.write("\n")
        process.stdin.flush()
        _, error = process.communicate(timeout=10)
        assert process.returncode == 0, error

    lease = acquire_build_directory(tmp_path)
    lease.release()


def test_build_directory_rejects_an_overlapping_thread(tmp_path: Path) -> None:
    """Reentrancy belongs to one thread, not every caller sharing its PID."""
    from transactional_emulator.testbench.emulator_runner import (
        acquire_build_directory,
    )

    lease = acquire_build_directory(tmp_path)
    errors: list[BaseException] = []

    def contend() -> None:
        try:
            acquire_build_directory(tmp_path)
        except BaseException as error:  # captured and asserted in the parent
            errors.append(error)

    thread = threading.Thread(target=contend)
    thread.start()
    thread.join(timeout=10)
    lease.release()
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "Concurrent or interleaved" in str(errors[0])


def test_stage_validation_holds_the_lease_while_reading_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command owns the directory until every validation read is complete."""
    import decode_stage_validation as validation

    root = Path(__file__).resolve().parents[2]
    build = tmp_path / "build"
    build.mkdir()
    op_stats = build / "op_stats.jsonl"
    op_stats.write_text("", encoding="utf-8")
    child = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from transactional_emulator.testbench.emulator_runner import acquire_build_directory",
            "lease = acquire_build_directory(Path(sys.argv[1]))",
            "lease.release()",
        )
    )
    python_path = os.pathsep.join(path for path in (str(root), os.environ.get("PYTHONPATH", "")) if path)
    environment = {**os.environ, "PYTHONPATH": python_path}

    def validation_body(_args: object) -> int:
        attempted = subprocess.run(
            [sys.executable, "-c", child, str(build)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert attempted.returncode != 0
        assert "Concurrent or interleaved" in attempted.stderr
        return 0

    monkeypatch.setattr(validation, "_run_validation", validation_body)
    argv = [
        "decode_stage_validation.py",
        "--asm",
        str(tmp_path / "assembly.asm"),
        "--op-stats",
        str(op_stats),
        "--settings",
        str(tmp_path / "settings.toml"),
        "--isa-lib",
        str(tmp_path / "isa.json"),
    ]
    original = sys.argv
    try:
        sys.argv = argv
        assert validation.main() == 0
    finally:
        sys.argv = original

    released = subprocess.run(
        [sys.executable, "-c", child, str(build)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert released.returncode == 0, released.stderr


def test_emulator_execution_serializes_global_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Different build directories cannot race the emulator's global dump."""
    from transactional_emulator.testbench import emulator_runner

    root = Path(__file__).resolve().parents[2]
    child = "\n".join(
        (
            "from transactional_emulator.testbench.emulator_runner import acquire_build_directory, _emulator_execution_directory",
            "lease = acquire_build_directory(_emulator_execution_directory())",
            "lease.release()",
        )
    )
    python_path = os.pathsep.join(path for path in (str(root), os.environ.get("PYTHONPATH", "")) if path)
    environment = {**os.environ, "PYTHONPATH": python_path}

    def execution_body(build_dir: Path, **_kwargs: object) -> dict:
        assert build_dir == tmp_path.resolve()
        attempted = subprocess.run(
            [sys.executable, "-c", child],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert attempted.returncode != 0
        assert "Concurrent or interleaved" in attempted.stderr
        return {}

    monkeypatch.setattr(emulator_runner, "_run_emulator_unlocked", execution_body)
    assert emulator_runner.run_emulator(tmp_path) == {}

    released = subprocess.run(
        [sys.executable, "-c", child],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert released.returncode == 0, released.stderr


_TIMING_ANCHOR_COLUMNS = (
    "anchor_id",
    "anchor_kind",
    "analytical_cycles",
    "analytical_compute_cycles",
    "analytical_memory_cycles",
    "cache_position",
    "batch",
    "physical_hbm_bytes",
    "emulator_cycles",
    "rtl_cycles",
    "mlen",
    "blen",
    "hlen",
    "vlen",
    "geometry_path",
    "precision_path",
    "asm_path",
    "analytical_trace_path",
    "emulator_trace_path",
    "rtl_trace_path",
)
_TIMING_IDENTITY_FIELDS = (
    "mlen",
    "blen",
    "hlen",
    "vlen",
    "geometry_sha256",
    "precision_sha256",
    "compiler_sha256",
    "asm_sha256",
    "analytical_trace_sha256",
    "emulator_trace_sha256",
    "rtl_trace_sha256",
)
_COMPILER_EVIDENCE = b"compiler raw evidence\n"


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _complete_timing_anchors() -> tuple[CycleAnchor, ...]:
    anchors = []
    specifications = (
        ("linear", "linear", None),
        ("qk", "qk", None),
        ("pv", "pv", None),
        ("vector", "vector", None),
        ("layer-128", "layer", 128),
        ("layer-129", "layer", 129),
    )
    for index, (anchor_id, anchor_kind, cache_position) in enumerate(specifications):
        cycles = 1_000 + index
        trace_digest = _digest_bytes(f"trace:{anchor_id}".encode())
        layer = anchor_kind == "layer"
        anchors.append(
            CycleAnchor(
                anchor_id=anchor_id,
                anchor_kind=anchor_kind,
                analytical_cycles=cycles,
                emulator_cycles=cycles,
                rtl_cycles=cycles,
                analytical_compute_cycles=cycles if layer else None,
                analytical_memory_cycles=cycles - 1 if layer else None,
                cache_position=cache_position,
                batch=1 if layer else None,
                physical_hbm_bytes=4_096 + index if layer else None,
                mlen=16,
                blen=4,
                hlen=8,
                vlen=16,
                geometry_sha256=_digest_bytes(b"geometry"),
                precision_sha256=_digest_bytes(b"precision"),
                compiler_sha256=_digest_bytes(_COMPILER_EVIDENCE),
                asm_sha256=_digest_bytes(f"assembly:{anchor_id}".encode()),
                analytical_trace_sha256=trace_digest,
                emulator_trace_sha256=trace_digest,
                rtl_trace_sha256=trace_digest,
            )
        )
    return tuple(anchors)


def _complete_timing_provenance() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "anchors": _digest_bytes(b"anchor CSV"),
                "compiler": _digest_bytes(_COMPILER_EVIDENCE),
                "analytic": _digest_bytes(b"analytic raw evidence\n"),
                "emulator": _digest_bytes(b"emulator raw evidence\n"),
                "rtl": _digest_bytes(b"rtl raw evidence\n"),
            }.items()
        )
    )


def _complete_timing_evidence() -> TimingEvidence:
    return TimingEvidence(
        mode=RTL_SERIALIZED,
        anchors=_complete_timing_anchors(),
        provenance_hashes=_complete_timing_provenance(),
    )


def test_timing_evidence_passes_only_with_complete_matched_identity() -> None:
    evidence = _complete_timing_evidence()
    assert evidence.passed
    assert evidence.execution_identity_matched
    assert evidence.trace_identities_matched
    assert evidence.compiler_provenance_matched
    assert not evidence.missing_provenance_roles
    assert evidence.anchor_max_error_limit == CANONICAL_ANCHOR_MAX_ERROR_LIMIT
    assert evidence.analytical_mape_limit == CANONICAL_ANALYTICAL_MAPE_LIMIT
    assert evidence.evidence_tier == "rtl"


@pytest.mark.parametrize(
    "field,value,failed_gate",
    (
        ("mlen", 32, "shared_geometry"),
        ("geometry_sha256", _digest_bytes(b"other geometry"), "shared_geometry"),
        ("precision_sha256", _digest_bytes(b"other precision"), "shared_precision"),
        ("compiler_sha256", _digest_bytes(b"other compiler"), "shared_compiler"),
    ),
)
def test_timing_evidence_rejects_cross_anchor_identity_mutation(
    field: str,
    value: object,
    failed_gate: str,
) -> None:
    evidence = _complete_timing_evidence()
    anchors = (
        replace(evidence.anchors[0], **{field: value}),
        *evidence.anchors[1:],
    )
    mutated = replace(evidence, anchors=anchors)
    assert not mutated.passed
    assert not getattr(mutated, failed_gate)


def test_timing_evidence_rejects_cross_stack_trace_mutation() -> None:
    evidence = _complete_timing_evidence()
    anchor = replace(
        evidence.anchors[0],
        rtl_trace_sha256=_digest_bytes(b"different RTL trace"),
    )
    mutated = replace(evidence, anchors=(anchor, *evidence.anchors[1:]))
    assert not mutated.trace_identities_matched
    assert not mutated.execution_identity_matched
    assert not mutated.passed


def test_timing_evidence_rejects_missing_or_unbound_provenance() -> None:
    evidence = _complete_timing_evidence()
    without_analytic = replace(
        evidence,
        provenance_hashes=tuple(entry for entry in evidence.provenance_hashes if entry[0] != "analytic"),
    )
    assert without_analytic.missing_provenance_roles == ("analytic",)
    assert not without_analytic.passed

    provenance = dict(evidence.provenance_hashes)
    provenance["compiler"] = _digest_bytes(b"unbound compiler evidence")
    unbound_compiler = replace(
        evidence,
        provenance_hashes=tuple(sorted(provenance.items())),
    )
    assert not unbound_compiler.compiler_provenance_matched
    assert not unbound_compiler.passed


def test_legacy_timing_evidence_loads_fail_closed() -> None:
    payload = _complete_timing_evidence().to_dict()
    payload.pop("evidence_id")
    for anchor in payload["anchors"]:
        for field in _TIMING_IDENTITY_FIELDS:
            anchor.pop(field)
    legacy = TimingEvidence.from_dict(payload)
    assert not legacy.execution_identity_complete
    assert not legacy.passed


@pytest.mark.parametrize("missing_field", _TIMING_IDENTITY_FIELDS)
def test_timing_evidence_rejects_each_partial_identity_field(
    missing_field: str,
) -> None:
    payload = _complete_timing_evidence().to_dict()
    payload.pop("evidence_id")
    payload["anchors"][0].pop(missing_field)
    with pytest.raises(ValueError, match="execution identity must be complete"):
        TimingEvidence.from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    (
        ("anchor_max_error_limit", 0.051),
        ("analytical_mape_limit", 0.101),
        ("anchor_max_error_limit", 1.0),
        ("analytical_mape_limit", 1.0),
    ),
)
def test_timing_evidence_rejects_threshold_mutation(
    field: str,
    value: float,
) -> None:
    payload = _complete_timing_evidence().to_dict()
    payload.pop("evidence_id")
    payload[field] = value
    with pytest.raises(ValueError, match="limits are immutable"):
        TimingEvidence.from_dict(payload)


def _complete_emulator_timing_anchors() -> tuple[CycleAnchor, ...]:
    anchors = []
    specifications = (
        ("linear", "linear", None),
        ("qk", "qk", None),
        ("pv", "pv", None),
        ("vector", "vector", None),
        ("layer-128", "layer", 128),
        ("layer-129", "layer", 129),
    )
    for index, (anchor_id, anchor_kind, cache_position) in enumerate(specifications):
        cycles = 1_000 + index
        trace_digest = _digest_bytes(f"trace:{anchor_id}".encode())
        layer = anchor_kind == "layer"
        anchors.append(
            CycleAnchor(
                anchor_id=anchor_id,
                anchor_kind=anchor_kind,
                analytical_cycles=cycles,
                emulator_cycles=cycles,
                analytical_compute_cycles=cycles if layer else None,
                analytical_memory_cycles=cycles - 1 if layer else None,
                cache_position=cache_position,
                batch=1 if layer else None,
                physical_hbm_bytes=4_096 + index if layer else None,
                mlen=16,
                blen=4,
                hlen=8,
                vlen=16,
                geometry_sha256=_digest_bytes(b"geometry"),
                precision_sha256=_digest_bytes(b"precision"),
                compiler_sha256=_digest_bytes(_COMPILER_EVIDENCE),
                asm_sha256=_digest_bytes(f"assembly:{anchor_id}".encode()),
                analytical_trace_sha256=trace_digest,
                emulator_trace_sha256=trace_digest,
            )
        )
    return tuple(anchors)


def _complete_emulator_timing_provenance() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "anchors": _digest_bytes(b"anchor CSV"),
                "compiler": _digest_bytes(_COMPILER_EVIDENCE),
                "analytic": _digest_bytes(b"analytic raw evidence\n"),
                "emulator": _digest_bytes(b"emulator raw evidence\n"),
            }.items()
        )
    )


def _complete_emulator_timing_evidence() -> TimingEvidence:
    return TimingEvidence(
        mode=EMULATOR_SERIALIZED,
        anchors=_complete_emulator_timing_anchors(),
        provenance_hashes=_complete_emulator_timing_provenance(),
    )


def test_emulator_timing_evidence_passes_with_complete_matched_identity() -> None:
    evidence = _complete_emulator_timing_evidence()
    assert evidence.evidence_tier == "emulator"
    assert evidence.passed
    assert evidence.execution_identity_matched
    assert evidence.trace_identities_matched
    assert not evidence.missing_provenance_roles
    assert evidence.emulator_rtl_error is None
    assert evidence.anchor_max_error == 0.0
    assert evidence.analytical_mape == 0.0


@pytest.mark.parametrize("dropped_kind", ("linear", "qk", "pv", "vector"))
def test_emulator_timing_evidence_requires_every_anchor_kind(
    dropped_kind: str,
) -> None:
    evidence = _complete_emulator_timing_evidence()
    reduced = replace(
        evidence,
        anchors=tuple(anchor for anchor in evidence.anchors if anchor.anchor_kind != dropped_kind),
    )
    assert dropped_kind in reduced.missing_anchor_kinds
    assert not reduced.passed


def test_emulator_timing_evidence_requires_two_layer_anchors() -> None:
    evidence = _complete_emulator_timing_evidence()
    reduced = replace(
        evidence,
        anchors=tuple(anchor for anchor in evidence.anchors if anchor.anchor_id != "layer-129"),
    )
    assert reduced.layer_anchor_count == 1
    assert not reduced.passed


def test_emulator_timing_evidence_rejects_non_consecutive_cache_positions() -> None:
    evidence = _complete_emulator_timing_evidence()
    anchors = tuple(
        replace(anchor, anchor_id="layer-131", cache_position=131)
        if anchor.anchor_id == "layer-129"
        else anchor
        for anchor in evidence.anchors
    )
    with pytest.raises(ValueError, match="consecutive cache appends"):
        TimingEvidence(
            mode=EMULATOR_SERIALIZED,
            anchors=anchors,
            provenance_hashes=_complete_emulator_timing_provenance(),
        )


def test_emulator_timing_evidence_rejects_mixed_layer_batches() -> None:
    evidence = _complete_emulator_timing_evidence()
    anchors = tuple(
        replace(anchor, batch=2) if anchor.anchor_id == "layer-129" else anchor
        for anchor in evidence.anchors
    )
    with pytest.raises(ValueError, match="one batch"):
        TimingEvidence(
            mode=EMULATOR_SERIALIZED,
            anchors=anchors,
            provenance_hashes=_complete_emulator_timing_provenance(),
        )


def test_emulator_timing_evidence_rejects_trace_identity_mismatch() -> None:
    evidence = _complete_emulator_timing_evidence()
    anchor = replace(
        evidence.anchors[0],
        emulator_trace_sha256=_digest_bytes(b"different emulator trace"),
    )
    mutated = replace(evidence, anchors=(anchor, *evidence.anchors[1:]))
    assert not mutated.trace_identities_matched
    assert not mutated.passed


def test_emulator_timing_evidence_rejects_rtl_anchor_evidence() -> None:
    evidence = _complete_emulator_timing_evidence()
    anchor = evidence.anchors[0]
    anchors = (
        replace(
            anchor,
            rtl_cycles=anchor.emulator_cycles,
            rtl_trace_sha256=anchor.emulator_trace_sha256,
        ),
        *evidence.anchors[1:],
    )
    with pytest.raises(ValueError, match="must not carry RTL evidence"):
        TimingEvidence(
            mode=EMULATOR_SERIALIZED,
            anchors=anchors,
            provenance_hashes=_complete_emulator_timing_provenance(),
        )


def test_emulator_timing_evidence_rejects_rtl_provenance_role() -> None:
    provenance = dict(_complete_emulator_timing_provenance())
    provenance["rtl"] = _digest_bytes(b"rtl raw evidence\n")
    with pytest.raises(ValueError, match="RTL provenance role"):
        TimingEvidence(
            mode=EMULATOR_SERIALIZED,
            anchors=_complete_emulator_timing_anchors(),
            provenance_hashes=tuple(sorted(provenance.items())),
        )


@pytest.mark.parametrize("missing_role", ("compiler", "analytic", "emulator"))
def test_emulator_timing_evidence_requires_each_provenance_role(
    missing_role: str,
) -> None:
    evidence = _complete_emulator_timing_evidence()
    reduced = replace(
        evidence,
        provenance_hashes=tuple(entry for entry in evidence.provenance_hashes if entry[0] != missing_role),
    )
    assert reduced.missing_provenance_roles == (missing_role,)
    assert not reduced.passed


def test_emulator_timing_evidence_enforces_anchor_error_limit() -> None:
    evidence = _complete_emulator_timing_evidence()
    anchor = evidence.anchors[0]
    loose = replace(
        anchor,
        analytical_cycles=int(anchor.emulator_cycles * 1.06),
    )
    mutated = replace(evidence, anchors=(loose, *evidence.anchors[1:]))
    assert mutated.anchor_max_error > mutated.anchor_max_error_limit
    assert mutated.analytical_mape <= mutated.analytical_mape_limit
    assert not mutated.passed


def test_emulator_timing_evidence_rejects_threshold_mutation() -> None:
    payload = _complete_emulator_timing_evidence().to_dict()
    payload.pop("evidence_id")
    payload["anchor_max_error_limit"] = 1.0
    with pytest.raises(ValueError, match="limits are immutable"):
        TimingEvidence.from_dict(payload)


def test_emulator_timing_evidence_content_hash_rejects_hand_edits() -> None:
    payload = _complete_emulator_timing_evidence().to_dict()
    assert TimingEvidence.from_dict(payload).passed
    payload["anchors"][0]["emulator_cycles"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        TimingEvidence.from_dict(payload)


def test_emulator_timing_evidence_rejects_mislabeled_tier() -> None:
    payload = _complete_emulator_timing_evidence().to_dict()
    payload.pop("evidence_id")
    payload["evidence_tier"] = "rtl"
    with pytest.raises(ValueError, match="tier does not match its mode"):
        TimingEvidence.from_dict(payload)


def test_timing_evidence_builder_binds_all_raw_roles_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    geometry_path = tmp_path / "geometry.json"
    precision_path = tmp_path / "precision.svh"
    geometry_path.write_text(
        json.dumps({"mlen": 16, "blen": 4, "hlen": 8, "vlen": 16}),
        encoding="utf-8",
    )
    precision_path.write_bytes(b"precision")
    anchors_path = tmp_path / "anchors.csv"
    with anchors_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=_TIMING_ANCHOR_COLUMNS)
        writer.writeheader()
        for anchor in _complete_timing_anchors():
            values = anchor.to_dict()
            asm_path = tmp_path / f"{anchor.anchor_id}.asm"
            asm_path.write_bytes(f"assembly:{anchor.anchor_id}".encode())
            trace_paths = {}
            for role in ("analytical", "emulator", "rtl"):
                trace_path = tmp_path / f"{anchor.anchor_id}.{role}.trace"
                trace_path.write_bytes(f"trace:{anchor.anchor_id}".encode())
                trace_paths[role] = trace_path
            values.update(
                {
                    "geometry_path": geometry_path.name,
                    "precision_path": precision_path.name,
                    "asm_path": asm_path.name,
                    "analytical_trace_path": trace_paths["analytical"].name,
                    "emulator_trace_path": trace_paths["emulator"].name,
                    "rtl_trace_path": trace_paths["rtl"].name,
                }
            )
            writer.writerow({column: values[column] for column in _TIMING_ANCHOR_COLUMNS})

    role_payloads = {
        "compiler": _COMPILER_EVIDENCE,
        "analytic": b"analytic raw evidence\n",
        "emulator": b"emulator raw evidence\n",
        "rtl": b"rtl raw evidence\n",
    }
    role_paths = {}
    for role, payload in role_payloads.items():
        path = tmp_path / f"{role}.json"
        path.write_bytes(payload)
        role_paths[role] = path

    output = tmp_path / "timing.json"
    arguments = [
        "build_timing_evidence.py",
        "--mode",
        RTL_SERIALIZED,
        "--anchors",
        str(anchors_path),
    ]
    for role, path in role_paths.items():
        arguments.extend(("--provenance", f"{role}={path}"))
    arguments.extend(("--out", str(output)))
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "analytic_models.performance.build_timing_evidence",
        *arguments[1:],
    ]
    built_run = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert built_run.returncode == 0, built_run.stderr
    built = TimingEvidence.load(output)
    assert built.passed

    for role in ("analytical", "emulator", "rtl"):
        trace = tmp_path / f"linear.{role}.trace"
        original_trace = trace.read_bytes()
        original_trace_md5 = hashlib.md5(original_trace).hexdigest()
        trace.write_bytes(f"mutated {role} trace".encode())
        mismatched_output = tmp_path / f"{role}-trace-mismatch.json"
        mismatch_command = [*command[:-1], str(mismatched_output)]
        mismatched_run = subprocess.run(
            mismatch_command,
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert mismatched_run.returncode == 2, mismatched_run.stderr
        mismatched = TimingEvidence.load(mismatched_output)
        assert not mismatched.trace_identities_matched
        assert not mismatched.passed
        trace.write_bytes(original_trace)
        assert hashlib.md5(trace.read_bytes()).hexdigest() == original_trace_md5

    original_geometry = geometry_path.read_bytes()
    original_geometry_md5 = hashlib.md5(original_geometry).hexdigest()
    geometry_path.write_text(
        json.dumps({"mlen": 32, "blen": 4, "hlen": 8, "vlen": 16}),
        encoding="utf-8",
    )
    geometry_output = tmp_path / "geometry-mismatch.json"
    geometry_command = [*command[:-1], str(geometry_output)]
    geometry_run = subprocess.run(
        geometry_command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert geometry_run.returncode != 0
    assert "does not match geometry manifest" in geometry_run.stderr
    assert not geometry_output.exists()
    geometry_path.write_bytes(original_geometry)
    assert hashlib.md5(geometry_path.read_bytes()).hexdigest() == original_geometry_md5

    immutable_inputs = (
        (anchors_path, anchors_path.read_bytes() + b"\n"),
        (geometry_path, geometry_path.read_bytes() + b"\n"),
        (precision_path, b"mutated precision"),
        (tmp_path / "linear.asm", b"mutated assembly"),
        *((path, f"mutated {role} provenance".encode()) for role, path in role_paths.items()),
    )
    original_output_md5 = hashlib.md5(output.read_bytes()).hexdigest()
    for raw_path, mutated_bytes in immutable_inputs:
        original_bytes = raw_path.read_bytes()
        original_md5 = hashlib.md5(original_bytes).hexdigest()
        raw_path.write_bytes(mutated_bytes)
        mutation_run = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert mutation_run.returncode != 0
        assert "refusing to replace different timing evidence" in mutation_run.stderr
        assert hashlib.md5(output.read_bytes()).hexdigest() == original_output_md5
        raw_path.write_bytes(original_bytes)
        assert hashlib.md5(raw_path.read_bytes()).hexdigest() == original_md5

        restored_run = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert restored_run.returncode == 0, restored_run.stderr
        assert hashlib.md5(output.read_bytes()).hexdigest() == original_output_md5

    mutated = json.loads(output.read_text(encoding="utf-8"))
    original = output.read_bytes()
    original_md5 = hashlib.md5(original).hexdigest()
    mutated["anchor_max_error_limit"] = 1.0
    output.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="limits are immutable"):
        TimingEvidence.load(output)
    output.write_bytes(original)
    assert hashlib.md5(output.read_bytes()).hexdigest() == original_md5
    assert TimingEvidence.load(output).passed


@pytest.mark.parametrize("missing_role", ("compiler", "analytic", "emulator", "rtl"))
def test_timing_evidence_builder_requires_every_raw_provenance_role(
    tmp_path: Path,
    missing_role: str,
) -> None:
    import build_timing_evidence as builder

    anchors = tmp_path / "anchors.csv"
    anchors.write_text("anchor_id\n", encoding="utf-8")
    values = []
    for role in ("compiler", "analytic", "emulator", "rtl"):
        if role == missing_role:
            continue
        path = tmp_path / f"{role}.json"
        path.write_text(role, encoding="utf-8")
        values.append(f"{role}={path}")
    with pytest.raises(ValueError, match=missing_role):
        builder._provenance(values, anchors, emulator_tier=False)


def test_timing_evidence_builder_rejects_legacy_anchor_columns(
    tmp_path: Path,
) -> None:
    import build_timing_evidence as builder

    anchors = tmp_path / "legacy.csv"
    anchors.write_text(
        "anchor_id,anchor_kind,analytical_cycles,emulator_cycles,rtl_cycles\nlinear,linear,100,100,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"geometry.*trace-identity"):
        builder._load_anchors(
            anchors,
            compiler_sha256=_digest_bytes(_COMPILER_EVIDENCE),
            emulator_tier=False,
        )
