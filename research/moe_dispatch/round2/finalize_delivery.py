"""Refresh delivery metadata without simulating or rewriting numeric evidence.

Producer READMEs and manifests survive verbatim. Execution commits belong to
their actual receipts; the commit running this tool is a separate metadata
commit. Search process success does not imply a closed optimality proof.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
LABELS = {
    "E0": ("round2_final_engine_unit_suite", "round2_final_reader_unit_suite"),
    "E1": ("E1",),
    "E2": ("E2micro_cold_recompute", "E2layer"),
    "E3": ("E3_main_search", "E3_lb_validity", "E3_schedule_gaps", "E3_grid",
           "E3_extreme", "E3_robust", "E3_sobol", "E3_flip",
           "E3_inner_assignment_verification"),
    "E4": ("E4",), "E5": ("E5",), "E6": ("E6",),
}
NUMERIC_SOURCES = {
    "E2micro_cold_recompute": ("model.py",),
    "E3_main_search": ("model.py", "optimizer.py", "search.py", "main_search.py"),
    "E3_lb_validity": ("model.py", "optimizer.py", "main_search.py"),
    "E3_schedule_gaps": ("model.py", "optimizer.py", "main_search.py"),
    "E3_grid": ("model.py", "optimizer.py", "search.py", "regions.py"),
    "E3_extreme": ("model.py", "optimizer.py", "search.py", "regions.py", "extreme.py"),
    "E3_robust": ("model.py", "optimizer.py", "robust.py"),
    "E3_sobol": ("model.py", "optimizer.py", "search.py", "regions.py", "sensitivity.py"),
    "E3_flip": ("model.py", "optimizer.py", "search.py", "regions.py", "sensitivity.py"),
    "E3_inner_assignment_verification": ("model.py", "optimizer.py", "repair_inner.py"),
    "E1": ("model.py", "optimizer.py", "run.py"),
    "E2layer": ("model.py", "run.py"),
    "E4": ("model.py", "optimizer.py", "run.py"),
    "E5": ("model.py", "predictors.py", "oracle_replay.py", "run.py"),
    "E6": ("run.py",),
}
METADATA = {"README.md", "ACTUAL_EXECUTIONS.json", "DELIVERY_ARCHIVE_MANIFEST.json"}
EXCLUDED_DIRS = {"raw", "tmp", "temp", "__pycache__", ".pytest_cache", ".cache",
                 "target", ".git", ".venv", "venv", "build", "checkpoints", "weights"}
SENSITIVE_NAMES = {".env", ".netrc", "credentials.json", "auth.json"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git_commit(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unrecorded"


def receipt_inventory(root):
    """Retain all successful receipts, distinguishing obsolete source versions."""
    result = []
    frozen_sha = sha(root / "results/E0/frozen_inputs.json")
    for path in sorted((root / "results/executions").glob("*.json")):
        value = data(path)
        if value.get("returncode") != 0 or not value.get("finished_utc"):
            continue
        log = root / value.get("log", "missing")
        valid_log = log.is_file() and sha(log) == value.get("log_sha256")
        current = all((root / name).is_file() and
                      value.get("source_sha256", {}).get(name) == sha(root / name)
                      for name in NUMERIC_SOURCES.get(value.get("label"), ()))
        command = value.get("command", [])
        environment = value.get("environment", {})
        exact = shlex.join(["env", *(f"{k}={v}" for k, v in sorted(environment.items())),
                            *command]) if command else None
        result.append({**value, "receipt": str(path.relative_to(root)),
                       "receipt_sha256": sha(path), "log_hash_valid": valid_log,
                       "frozen_input_hash_valid": value.get("frozen_input_manifest_sha256") == frozen_sha,
                       "current_numeric_source_compatible": current,
                       "exact_shell_command": exact})
    return result


def select_receipts(inventory, stage, allow_missing=False):
    selected, missing = [], []
    for label in LABELS[stage]:
        matches = [r for r in inventory if r.get("label") == label and
                   r["log_hash_valid"] and r["frozen_input_hash_valid"] and
                   r["current_numeric_source_compatible"]]
        if not matches:
            missing.append(label)
        else:
            selected.append(max(matches, key=lambda r: r["finished_utc"]))
    if missing and not allow_missing:
        raise ValueError("Missing successful current-source stage receipts: " + ", ".join(missing))
    return selected, missing


def preserve_once(folder):
    # A stage may have had no producer README/manifest. Do not turn our own
    # first metadata refresh into a fictitious producer snapshot on rerun.
    if (folder / "ACTUAL_EXECUTIONS.json").exists():
        return
    for old, saved in (("README.md", "RUN_README.md"),
                       ("PROVENANCE.csv", "PROVENANCE.execution.csv")):
        source, target = folder / old, folder / saved
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)


def original_rows(folder):
    path = folder / "PROVENANCE.execution.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as stream:
        return {r.get("file", r.get("path", "")): r for r in csv.DictReader(stream)}


def local_payload(base, relative):
    """Archive references must stay within their documented delivery root."""
    path = base / relative
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Nonportable archive reference: " + str(relative))
    return path


def archive_evidence(root, folder):
    """Separate compact derivatives, exact archived members and containers.

    Container bytes are rehashed here. Member hashes come from the preserved
    packaging manifest/full-scope roundtrip audit; metadata does not decompress
    or rerun thousands of numerical certificates.
    """
    outputs, members, containers, indexes = {}, {}, {}, []
    prefix = folder / "certificate_archives"
    if not prefix.is_dir():
        return {"outputs": outputs, "members": members, "containers": containers, "indexes": indexes}
    for manifest in sorted(prefix.rglob("manifest.json")):
        value = data(manifest)
        relative = str(manifest.relative_to(folder))
        manifest_sha = sha(manifest)
        chunks = list(value.get("chunks", []))
        if isinstance(value.get("archive"), dict):
            chunks.append(value["archive"])
        for item in chunks:
            name = item["file"]
            payload = local_payload(folder, name)
            if not payload.is_file() or sha(payload) != item["sha256"]:
                raise ValueError("Archive container hash mismatch: " + name)
            containers.setdefault(name, []).append({"archive_index": relative,
                                                    "archive_index_sha256": manifest_sha})
        records = list(value.get("certificates", [])) + list(value.get("raw_over_50MiB_files", []))
        for item in records:
            original = item["original_relative_path"]
            payload = item.get("chunk", item.get("packaged_in"))
            if payload not in containers:
                raise ValueError("Archive member references an unverified container: " + original)
            entry = {"archive_payload": payload, "archive_member": original,
                     "archive_member_sha256": item["sha256"], "archive_member_size_bytes": item["bytes"],
                     "archive_index": relative, "archive_index_sha256": manifest_sha}
            members.setdefault(original, []).append(entry)
        indexes.append({"file": relative, "sha256": manifest_sha,
                        "declared_member_count": len(records),
                        "roundtrip_verified_in_original_manifest": value.get("roundtrip_verified"),
                        "member_hash_boundary": "Preserved packaging manifest and linked full-scope audit; this metadata refresh verifies container bytes without re-extracting members"})
    path = prefix / "original_driver_outputs.csv"
    if path.is_file():
        index_sha = sha(path)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                relative = row["original_relative_path"]
                summary = local_payload(folder, relative)
                if not summary.is_file() or sha(summary) != row["derived_summary_sha256"]:
                    raise ValueError("Derived summary hash mismatch: " + relative)
                payload = local_payload(folder, row["archived_payload"])
                if not payload.is_file() or sha(payload) != row["archived_payload_sha256"]:
                    raise ValueError("Original driver archive hash mismatch: " + relative)
                receipt = local_payload(root, row["driver_execution_receipt"])
                if not receipt.is_file() or sha(receipt) != row["driver_execution_receipt_sha256"]:
                    raise ValueError("Original driver receipt hash mismatch: " + relative)
                execution = data(receipt)
                if execution.get("returncode") != 0 or not execution.get("finished_utc"):
                    raise ValueError("Original driver receipt is not a successful execution: " + relative)
                evidence = {**row, "archive_index": str(path.relative_to(folder)),
                            "archive_index_sha256": index_sha,
                            "underlying_execution_commit": execution.get("execution_commit", "unrecorded"),
                            "underlying_producer_label": execution.get("label"),
                            "archive_member_sha256": row["original_sha256"]}
                outputs[relative] = evidence
                containers.setdefault(row["archived_payload"], []).append(evidence)
        indexes.append({"file": str(path.relative_to(folder)), "sha256": index_sha,
                        "declared_original_driver_outputs": len(outputs),
                        "scope": "Current summaries and original archived driver bytes have distinct hashes and roles"})
    external_restorations = []
    for relative, refs in members.items():
        restored = folder / relative
        if restored.is_symlink() or not restored.resolve().is_relative_to(folder.resolve()):
            external_restorations.append(relative)
            continue
        if restored.is_file() and relative not in outputs and sha(restored) not in {r["archive_member_sha256"] for r in refs}:
            raise ValueError("Restored archive member hash mismatch: " + relative)
    return {"outputs": outputs, "members": members, "containers": containers, "indexes": indexes,
            "external_restoration_paths_not_traversed": external_restorations}


def inventory_files(folder):
    """Never follow directory links or silently inventory external build/raw trees."""
    files, excluded = [], []
    for parent, dirs, names in os.walk(folder, followlinks=False):
        base = Path(parent)
        keep = []
        for name in sorted(dirs):
            path = base / name
            if path.is_symlink() or name in EXCLUDED_DIRS:
                excluded.append({"path": str(path.relative_to(folder)),
                                 "kind": "symlink_directory" if path.is_symlink() else "excluded_directory",
                                 "target": os.readlink(path) if path.is_symlink() else None,
                                 "reason": "external directory not traversed" if path.is_symlink() else "raw/temporary/build/cache payload excluded",
                                 "portable": False if path.is_symlink() else "requires original producer/archive manifest",
                                 "contents_hashed": False})
            else:
                keep.append(name)
        dirs[:] = keep
        for name in sorted(names):
            path = base / name
            if name in SENSITIVE_NAMES or name.startswith(".env."):
                excluded.append({"path": str(path.relative_to(folder)), "kind": "sensitive_file",
                                 "reason": "credentials are never copied or hashed by this metadata tool"})
                continue
            if path.is_symlink():
                target = os.readlink(path)
                excluded.append({"path": str(path.relative_to(folder)), "kind": "symlink_file",
                                 "target": target, "link_text_sha256": hashlib.sha256(target.encode()).hexdigest(),
                                 "reason": "external referenced payload is not copied or claimed bundled",
                                 "portable": False, "contents_hashed": False})
            elif path.is_file():
                files.append(path)
    return sorted(files), excluded


def producer_labels(stage, relative):
    """Assign a producer only where the output-to-stage relationship is known."""
    name = Path(relative).name
    if any(p.startswith(("archive", "retired")) or p == "certificate_archives" for p in Path(relative).parts):
        return ()
    if stage in ("E1", "E4", "E5", "E6"):
        return LABELS[stage]
    if stage == "E2":
        return ("E2micro_cold_recompute",) if name == "micro.csv" else ("E2layer",)
    if stage != "E3":
        return ()
    if relative.startswith("archive/") or "/archive/" in relative:
        return ()
    if "inner_assignment" in name or relative.startswith("inner_assignment"):
        return ("E3_inner_assignment_verification",)
    if name.startswith("lb_validity"):
        return ("E3_lb_validity",)
    if name.startswith("schedule_gaps"):
        return ("E3_schedule_gaps",)
    if "extreme" in name:
        return ("E3_extreme",)
    if "robust" in name or name == "selection_stability.csv":
        return ("E3_robust",)
    if "flip" in relative:
        return ("E3_flip",)
    if "sobol" in relative:
        return ("E3_sobol",)
    if "workload_map" in name or "synthetic_calibration" in name or "search_certificates/grid/" in relative:
        return ("E3_grid",)
    if name.startswith(("bnb_", "seed_", "FROZEN_SELECTION", "single_exhaustion")):
        return ("E3_main_search",)
    return ()


def direct_e0_evidence(folder):
    out = []
    for name in ("reproduction_receipt.json", "PHASE1_GATE.json"):
        p = folder / name
        if not p.is_file():
            continue
        value = data(p)
        out.append({"file": name, "sha256": sha(p),
                    "execution_commit": value.get("commit", value.get("simulator_execution_commit_before_source_commit")),
                    "compiler_sha": value.get("compiler_sha"), "source_sha256": value.get("source_sha256", {}),
                    "input_sha256": value.get("input_sha256", {}), "scope": value.get("scope"),
                    "command": value.get("command"),
                    "command_note": "When absent here, original exact commands remain in RUN_README.md, SOURCES.md and referenced test receipts; no command is invented.",
                    "receipt_sha256": value.get("receipt_sha256", {})})
    return out


def finalize(root, stages=STAGES, metadata_commit=None, allow_missing=False):
    root = Path(root).resolve()
    results = root / "results"
    metadata_commit = metadata_commit or git_commit(root)
    frozen = data(results / "E0/frozen_inputs.json")
    frozen_sha = sha(results / "E0/frozen_inputs.json")
    receipts = receipt_inventory(root)
    selections = {stage: select_receipts(receipts, stage, allow_missing) for stage in stages}
    # All validation precedes the first actual delivery mutation.
    for stage in stages:
        if not (results / stage).is_dir():
            raise ValueError("Missing result directory: " + stage)
    archived = {stage: archive_evidence(root, results / stage) for stage in stages}
    summaries = {}
    for stage in stages:
        folder = results / stage
        preserve_once(folder)
        previous = original_rows(folder)
        selected, missing = selections[stage]
        execution_map = {r["label"]: r for r in selected}
        direct = direct_e0_evidence(folder) if stage == "E0" else []
        test_links = []
        if stage == "E0":
            for path in sorted(folder.rglob("UNIT_CHECKS.json")):
                value = data(path)
                test_links.append({"file": str(path.relative_to(folder)), "sha256": sha(path),
                                   "commit_at_start": value.get("commit_at_start"),
                                   "commit_at_end": value.get("commit_at_end"),
                                   "scope": "Original test attempt; successful and superseded failed attempts are distinguished by the referenced receipt and supported PHASE1_GATE, not relabeled by metadata",
                                   "suites": value.get("suites", [])})
        execution_object = {
            "scope": "Actual successful producer receipts; metadata refresh is not a numerical rerun",
            "metadata_generation_commit": metadata_commit,
            "frozen_input_manifest_sha256": frozen_sha,
            "selected_current_source_executions": selected,
            "missing_stage_receipts": missing,
            "other_successful_stage_executions": [r for r in receipts if r.get("label") in LABELS[stage] and r not in selected],
            "direct_phase1_evidence": direct,
            "test_attempt_receipts": test_links,
            "archived_original_driver_outputs": list(archived[stage]["outputs"].values()),
        }
        write_json(folder / "ACTUAL_EXECUTIONS.json", execution_object)
        _, excluded = inventory_files(folder)
        write_json(folder / "DELIVERY_ARCHIVE_MANIFEST.json", {
            "scope": "Explicit omitted payload/reference manifest; omitted directory contents are not hashed or claimed self-contained",
            "metadata_generation_commit": metadata_commit,
            "excluded": excluded,
            "preserved_archives": "All regular files in non-excluded archive directories are independently hashed in PROVENANCE.csv; retained producer archive manifests remain authoritative for external references.",
            "archive_indexes": archived[stage]["indexes"],
            "external_restoration_paths_not_traversed": archived[stage].get("external_restoration_paths_not_traversed", []),
            "archive_member_hash_boundary": "Container bytes are checked by this metadata refresh. Exact member hashes and roundtrip checks are preserved from original packaging manifests/full-scope audits, not recomputed numerical outputs.",
        })
        lines = [f"# {stage} delivery evidence", "",
                 "This is a metadata refresh, not a simulation or reproduction rerun. Original producer text and hash snapshots are preserved verbatim in [RUN_README.md](RUN_README.md) and [PROVENANCE.execution.csv](PROVENANCE.execution.csv) when originally present.", "",
                 f"Metadata generation commit: `{metadata_commit}`. Frozen input manifest SHA-256: `{frozen_sha}`.", "",
                 "BF16 only; 18 development and 135 held-out windows. New performance results cover post-router MoE Gate/Up, SiLU/Z, Down and combine; one hypothetical cycle = 1 ns, ms = cycles / 1e6. The phase-fluid/shared-credit analytical model is not RTL or native HBM calibration. E0 contains separately identified historical reproduction and test evidence. Search process completion is not proof closure. Complete-model timing remains unavailable without matching non-MoE timings.", "",
                 "Actual execution commits, exact commands, source/input hashes and original receipts are listed in [ACTUAL_EXECUTIONS.json](ACTUAL_EXECUTIONS.json). Old successful source versions are retained there but are not silently assigned as final numerical producers.", ""]
        for row in selected:
            link = os.path.relpath(root / row["receipt"], folder)
            lines += [f"## {row['label']}", "", f"Execution commit: `{row.get('execution_commit', 'unrecorded')}`. Receipt SHA-256: `{row['receipt_sha256']}`.", "",
                      f"Actual successful receipt: [{Path(row['receipt']).name}]({link}); exit code 0; completed `{row['finished_utc']}`. Frozen input SHA-256: `{row.get('frozen_input_manifest_sha256', 'unrecorded')}`.", "",
                      "```sh", row.get("exact_shell_command") or "Command absent from original receipt; see original producer records.", "```", "",
                      "Captured source SHA-256 snapshot:", "```json", json.dumps(row.get("source_sha256", {}), sort_keys=True, indent=2), "```", ""]
        if direct:
            lines += ["## Historical reproduction and supported test gate", "",
                      "The E0 reproduction/test commits remain those in the actual phase-1 receipts; this metadata commit does not replace them. [reproduction_receipt.json](reproduction_receipt.json), [PHASE1_GATE.json](PHASE1_GATE.json), [SOURCES.md](SOURCES.md) and nested test receipts preserve their own source versions, commands, exactness and skip boundaries.", ""]
        if missing:
            lines += ["Incomplete receipt scope: " + ", ".join(missing) + ".", ""]
        lines += ["## Output inventory and portable references", "",
                  "[PROVENANCE.csv](PROVENANCE.csv) hashes every regular delivered file recursively, including audits, preserved producer snapshots and archives. `execution_commit` retains known original producer commits; `metadata_generation_commit` identifies this refresh separately. Metadata/audit files are not claimed to have performed numerical execution. The manifest excludes its own self-referential hash; the outer delivery receipt hashes that manifest.", "",
                  "[DELIVERY_ARCHIVE_MANIFEST.json](DELIVERY_ARCHIVE_MANIFEST.json) explicitly records raw/tmp/build/cache trees and symbolic-link dependencies that are not traversed or bundled. Referenced external payloads remain external; no incomplete bundle is described as self-contained.", "",
                  "Large original driver outputs may be stored as exact archive members while their old paths hold compact derived summaries. `sha256` always hashes the currently delivered file; `original_driver_output_sha256`/`archive_member_sha256` identify the exact original bytes. A derivative or compressed container is not assigned the historical numerical execution commit as its own producer. Linked archive indexes retain original execution receipts and restoration commands.", "",
                  "Frozen input source file hashes:", "```json", json.dumps(frozen.get("input_sha256", {}), indent=2, sort_keys=True), "```", ""]
        (folder / "README.md").write_text("\n".join(lines))
        files, _ = inventory_files(folder)
        rows = []
        for path in files:
            relative = str(path.relative_to(folder))
            if relative == "PROVENANCE.csv":
                continue
            original_key = "README.md" if relative == "RUN_README.md" else relative
            old = previous.get(original_key, {})
            labels = producer_labels(stage, relative)
            matching = [execution_map[label] for label in labels if label in execution_map]
            preserved = relative in ("RUN_README.md", "PROVENANCE.execution.csv")
            metadata = relative in METADATA
            archive = any(p.startswith(("archive", "retired")) for p in Path(relative).parts)
            audit = "audit" in relative.lower() or "independent" in relative.lower()
            role = ("metadata_only" if metadata else "preserved_producer_snapshot" if preserved else
                    "archived_evidence" if archive else "audit_or_source_evidence" if audit else "producer_output")
            if metadata or preserved or archive or audit:
                matching = []
            execution_commit = old.get("execution_commit", old.get("commit", ""))
            if metadata:
                execution_commit = ""
            elif not execution_commit and matching:
                execution_commit = ";".join(sorted({r.get("execution_commit", "unrecorded") for r in matching}))
            if stage == "E0" and not execution_commit and relative in ("reproduce_check.csv", "reproduction_receipt.json", "PHASE1_GATE.json"):
                evidence_name = "PHASE1_GATE.json" if relative == "PHASE1_GATE.json" else "reproduction_receipt.json"
                execution_commit = next((r.get("execution_commit") for r in direct if r["file"] == evidence_name), "")
            derivative = archived[stage]["outputs"].get(relative)
            contained = archived[stage]["containers"].get(relative, [])
            restored = archived[stage]["members"].get(relative, [])
            refs = [derivative] if derivative else contained or restored
            if derivative or contained or relative.startswith("certificate_archives/"):
                matching, execution_commit = [], ""
                role = ("derived_compact_summary" if derivative else
                        "archived_original_container" if contained else "archive_packaging_index")
            elif restored:
                role = "restored_archive_member"
            rows.append({"file": relative, "sha256": sha(path), "size_bytes": path.stat().st_size,
                         "execution_commit": execution_commit, "metadata_generation_commit": metadata_commit,
                         "evidence_role": role, "producer_labels": ";".join(r["label"] for r in matching),
                         "actual_execution_receipts": ";".join(r["receipt"] for r in matching),
                         "actual_execution_receipt_sha256": ";".join(r["receipt_sha256"] for r in matching),
                         "original_manifest_execution_commit": old.get("execution_commit", old.get("commit", "")),
                         "input_manifest_sha256": old.get("input_manifest_sha256", frozen_sha),
                         "underlying_execution_commit": ";".join(sorted({r.get("underlying_execution_commit", "") for r in refs} - {""})),
                         "original_driver_output_sha256": derivative.get("original_sha256", "") if derivative else "",
                         "original_driver_output_size_bytes": derivative.get("original_bytes", "") if derivative else "",
                         "archive_reference_json": json.dumps(refs, sort_keys=True) if refs else ""})
        with (folder / "PROVENANCE.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ("file", "sha256"))
            writer.writeheader()
            writer.writerows(rows)
        summaries[stage] = {"files_hashed": len(rows), "excluded_references": len(excluded),
                            "provenance_sha256": sha(folder / "PROVENANCE.csv"),
                            "README_sha256": sha(folder / "README.md"),
                            "actual_executions_sha256": sha(folder / "ACTUAL_EXECUTIONS.json"),
                            "producer_README_sha256": sha(folder / "RUN_README.md") if (folder / "RUN_README.md").is_file() else None,
                            "producer_manifest_sha256": sha(folder / "PROVENANCE.execution.csv") if (folder / "PROVENANCE.execution.csv").is_file() else None}
    delivery = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "metadata_generation_commit": metadata_commit,
                "metadata_source_sha256": sha(Path(__file__)),
                "frozen_input_manifest_sha256": frozen_sha,
                "scope": "Metadata-only refresh; existing numerical/derived payloads are not modified. Exact original large driver bytes remain in explicitly indexed archives, separate from compact current summaries.",
                "stages": summaries,
                "self_hash_boundary": "This outer JSON excludes its own self hash; the successful caller execution receipt/log records this metadata action."}
    write_json(results / "DELIVERY_METADATA.json", delivery)
    return delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--allow-missing-receipts", action="store_true", help="Explicit partial metadata refresh; never claims missing stages ran")
    args = parser.parse_args()
    result = finalize(args.root, tuple(args.stages), allow_missing=args.allow_missing_receipts)
    print(json.dumps({"scope": result["scope"], "stages": result["stages"]}, sort_keys=True))


if __name__ == "__main__":
    main()
