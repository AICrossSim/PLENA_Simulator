"""Delivery metadata must not erase or mislabel the evidence it inventories."""
import csv
import hashlib
import io
import json
import tarfile

import pytest

from .finalize_delivery import finalize, sha


def fixture(tmp_path, producer_header=True):
    root = tmp_path / "round2"
    out = root / "results"
    for directory in (out / "E0", out / "E1", out / "executions"):
        directory.mkdir(parents=True, exist_ok=True)
    (out / "E0/frozen_inputs.json").write_text(json.dumps({"input_sha256": {"trace.json": "original"}}))
    for name in ("model.py", "optimizer.py", "run.py"):
        (root / name).write_text("# numeric source\n")
    payload = out / "E1/numbers.csv"
    payload.write_text("cycle,ms\n1000000,1\n")
    (out / "E1/result.json").write_text('{"cycles":1000000}\n')
    (out / "E1/raw").mkdir()
    (out / "E1/raw/external.bin").write_bytes(b"not bundled")
    (out / "E1/external").symlink_to(out / "E1/raw", target_is_directory=True)
    if producer_header:
        (out / "E1/README.md").write_text("Original producer header: old-commit\n")
        with (out / "E1/PROVENANCE.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("file", "sha256", "execution_commit"))
            writer.writeheader()
            writer.writerow({"file": "numbers.csv", "sha256": sha(payload), "execution_commit": "old-commit"})
    log = out / "executions/E1.log"
    log.write_text("successful actual numerical run\n")
    receipt = {"label": "E1", "returncode": 0, "finished_utc": "2026-10-07T15:00:00Z",
               "execution_commit": "actual-execution-commit", "command": ["python", "-m", "actual.stage", "a b"],
               "environment": {"PYTHONHASHSEED": "20261007"},
               "log": "results/executions/E1.log", "log_sha256": sha(log),
               "source_sha256": {p.name: sha(p) for p in root.glob("*.py")},
               "frozen_input_manifest_sha256": sha(out / "E0/frozen_inputs.json")}
    (out / "executions/E1.json").write_text(json.dumps(receipt))
    return root, out / "E1"


def test_refresh_preserves_numeric_bytes_and_original_producer(tmp_path):
    root, stage = fixture(tmp_path)
    original = {name: (stage / name).read_bytes() for name in ("numbers.csv", "result.json", "README.md", "PROVENANCE.csv")}
    result = finalize(root, ("E1",), metadata_commit="metadata-commit")
    assert (stage / "numbers.csv").read_bytes() == original["numbers.csv"]
    assert (stage / "result.json").read_bytes() == original["result.json"]
    assert (stage / "RUN_README.md").read_bytes() == original["README.md"]
    assert (stage / "PROVENANCE.execution.csv").read_bytes() == original["PROVENANCE.csv"]
    rows = {r["file"]: r for r in csv.DictReader((stage / "PROVENANCE.csv").open())}
    assert rows["numbers.csv"]["execution_commit"] == "old-commit"
    assert rows["numbers.csv"]["metadata_generation_commit"] == "metadata-commit"
    assert rows["result.json"]["execution_commit"] == "actual-execution-commit"
    assert rows["README.md"]["execution_commit"] == ""
    assert rows["README.md"]["evidence_role"] == "metadata_only"
    assert "raw/external.bin" not in rows
    archived = json.loads((stage / "DELIVERY_ARCHIVE_MANIFEST.json").read_text())
    assert {r["kind"] for r in archived["excluded"]} == {"excluded_directory", "symlink_directory"}
    actual = json.loads((stage / "ACTUAL_EXECUTIONS.json").read_text())["selected_current_source_executions"][0]
    assert actual["exact_shell_command"] == "env PYTHONHASHSEED=20261007 python -m actual.stage 'a b'"
    assert result["stages"]["E1"]["provenance_sha256"] == sha(stage / "PROVENANCE.csv")
    for row in rows.values():
        assert row["sha256"] == sha(stage / row["file"])
    finalize(root, ("E1",), metadata_commit="second-metadata-commit")
    assert (stage / "RUN_README.md").read_bytes() == original["README.md"]
    assert (stage / "PROVENANCE.execution.csv").read_bytes() == original["PROVENANCE.csv"]


def test_missing_receipt_rejects_before_any_evidence_mutation(tmp_path):
    root, stage = fixture(tmp_path)
    (root / "results/executions/E1.json").unlink()
    original = (stage / "README.md").read_bytes()
    with pytest.raises(ValueError, match="Missing successful"):
        finalize(root, ("E1",), metadata_commit="not-a-numeric-commit")
    assert (stage / "README.md").read_bytes() == original
    assert not (stage / "RUN_README.md").exists()
    assert not (root / "results/DELIVERY_METADATA.json").exists()


def test_rerun_never_invents_an_absent_producer_header(tmp_path):
    root, stage = fixture(tmp_path, producer_header=False)
    finalize(root, ("E1",), metadata_commit="metadata-only-one")
    finalize(root, ("E1",), metadata_commit="metadata-only-two")
    assert not (stage / "RUN_README.md").exists()
    assert not (stage / "PROVENANCE.execution.csv").exists()


def test_wrong_input_successful_receipt_is_not_assigned_as_producer(tmp_path):
    root, stage = fixture(tmp_path)
    path = root / "results/executions/E1.json"
    record = json.loads(path.read_text())
    record["frozen_input_manifest_sha256"] = "different-input"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Missing successful"):
        finalize(root, ("E1",), metadata_commit="metadata")
    assert not (stage / "ACTUAL_EXECUTIONS.json").exists()


def archive_fixture(tmp_path):
    root, _ = fixture(tmp_path)
    stage = root / "results/E3"
    packaged = stage / "certificate_archives/extreme_final"
    packaged.mkdir(parents=True)
    (stage / "README.md").write_text("Original numeric producer header\n")
    original = b'{"iterations":[1,2,3],"cycles":1234}\n'
    original_sha = hashlib.sha256(original).hexdigest()
    current = stage / "workload_extreme.json"
    current.write_text('{"cycles":1234,"summary_kind":"derived"}\n')
    restored = stage / "cma_verification/final_delta0.json"
    restored.parent.mkdir()
    restored.write_bytes(original)
    container = packaged / "part_000.tar.gz"
    with tarfile.open(container, "w:gz") as archive:
        for name in ("workload_extreme.json", "cma_verification/final_delta0.json"):
            member = tarfile.TarInfo(name)
            member.size = len(original)
            archive.addfile(member, io.BytesIO(original))
    for name in ("search.py", "regions.py", "extreme.py"):
        (root / name).write_text("# actual numeric source\n")
    receipt = json.loads((root / "results/executions/E1.json").read_text())
    receipt.update(label="E3_extreme", execution_commit="original-numeric-commit",
                   source_sha256={p.name: sha(p) for p in root.glob("*.py")})
    receipt_path = root / "results/executions/extreme.json"
    receipt_path.write_text(json.dumps(receipt))
    payload_name = str(container.relative_to(stage))
    manifest = {"roundtrip_verified": True,
                "archive": {"file": payload_name, "sha256": sha(container)},
                "raw_over_50MiB_files": [{"original_relative_path": name,
                                          "sha256": original_sha, "bytes": len(original),
                                          "packaged_in": payload_name}
                                         for name in ("workload_extreme.json", "cma_verification/final_delta0.json")]}
    (packaged / "manifest.json").write_text(json.dumps(manifest))
    row = {"original_relative_path": "workload_extreme.json", "original_bytes": len(original),
           "original_sha256": original_sha,
           "driver_execution_receipt": str(receipt_path.relative_to(root)),
           "driver_execution_receipt_sha256": sha(receipt_path),
           "archived_payload": payload_name, "archived_payload_sha256": sha(container),
           "archive_member": "workload_extreme.json", "derived_summary_sha256": sha(current)}
    index = stage / "certificate_archives/original_driver_outputs.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return root, stage, container, original_sha


def test_derived_summary_never_masquerades_as_original_driver_bytes(tmp_path):
    root, stage, container, original_sha = archive_fixture(tmp_path)
    before = (stage / "workload_extreme.json").read_bytes()
    finalize(root, ("E3",), metadata_commit="metadata-commit", allow_missing=True)
    rows = {r["file"]: r for r in csv.DictReader((stage / "PROVENANCE.csv").open())}
    summary = rows["workload_extreme.json"]
    assert summary["evidence_role"] == "derived_compact_summary"
    assert summary["sha256"] == sha(stage / "workload_extreme.json") != original_sha
    assert summary["original_driver_output_sha256"] == original_sha
    assert summary["execution_commit"] == summary["producer_labels"] == ""
    assert summary["underlying_execution_commit"] == "original-numeric-commit"
    refs = json.loads(summary["archive_reference_json"])
    assert refs[0]["archive_member_sha256"] == original_sha
    archived = rows[str(container.relative_to(stage))]
    assert archived["evidence_role"] == "archived_original_container"
    assert archived["sha256"] == sha(container)
    assert archived["execution_commit"] == ""
    assert archived["underlying_execution_commit"] == "original-numeric-commit"
    restored = rows["cma_verification/final_delta0.json"]
    assert restored["evidence_role"] == "restored_archive_member"
    assert restored["sha256"] == original_sha
    assert (stage / "workload_extreme.json").read_bytes() == before
    actual = json.loads((stage / "ACTUAL_EXECUTIONS.json").read_text())
    assert actual["archived_original_driver_outputs"][0]["original_sha256"] == original_sha
    assert rows["certificate_archives/extreme_final/manifest.json"]["evidence_role"] == "archive_packaging_index"


@pytest.mark.parametrize("corrupt", ("summary", "container", "restored"))
def test_invalid_archive_or_derivative_rejects_before_metadata_writes(tmp_path, corrupt):
    root, stage, container, _ = archive_fixture(tmp_path)
    target = {"summary": stage / "workload_extreme.json", "container": container,
              "restored": stage / "cma_verification/final_delta0.json"}[corrupt]
    target.write_bytes(b"corrupted payload")
    original_header = (stage / "README.md").read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        finalize(root, ("E3",), metadata_commit="metadata", allow_missing=True)
    assert (stage / "README.md").read_bytes() == original_header
    assert not (stage / "ACTUAL_EXECUTIONS.json").exists()
    assert not (stage / "RUN_README.md").exists()


def test_external_restoration_link_is_not_followed_or_claimed_bundled(tmp_path):
    root, stage, _, _ = archive_fixture(tmp_path)
    restored = stage / "cma_verification/final_delta0.json"
    restored.unlink()
    outside = tmp_path / "external-original.json"
    outside.write_bytes(b"external reference is not trusted as bundled bytes")
    restored.symlink_to(outside)
    finalize(root, ("E3",), metadata_commit="metadata", allow_missing=True)
    rows = {r["file"] for r in csv.DictReader((stage / "PROVENANCE.csv").open())}
    assert "cma_verification/final_delta0.json" not in rows
    manifest = json.loads((stage / "DELIVERY_ARCHIVE_MANIFEST.json").read_text())
    assert manifest["external_restoration_paths_not_traversed"] == ["cma_verification/final_delta0.json"]
    assert any(r["kind"] == "symlink_file" for r in manifest["excluded"])
    assert outside.read_bytes() == b"external reference is not trusted as bundled bytes"
