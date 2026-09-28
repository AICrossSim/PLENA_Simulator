"""Compare a fresh default-dynamic campaign with the frozen analytical reports."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(output: Path) -> dict:
    reference = json.loads((Path(__file__).parent / "results/reference_reports.json").read_text())
    latest = json.loads((output / "latest_matrix.json").read_text())
    manifest = json.loads(Path(latest["manifest"]).read_text())
    if manifest["failures"] or manifest["prepare_only"]:
        raise ValueError("campaign did not complete")
    points = {point["key"]: point for point in manifest["points"]}
    if set(points) != set(reference):
        raise ValueError("expected exactly B2/B4/B8/B16 x 6/33/42, G4 dynamic, no split")
    checks = []
    for key, expected in sorted(reference.items()):
        directory = Path(points[key]["directory"])
        reports = [(directory / f"report_repeat{i}.json").read_bytes() for i in (1, 2)]
        hashes = [hashlib.sha256(report).hexdigest() for report in reports]
        if hashes != [expected["report_sha256"]] * 2:
            raise ValueError(f"historical raw-report mismatch: {key}")
        checks.append({"point": key, **expected, "both_repeats_match": True})
    return {"scope": "fresh analytical raw-report parity, not native HBM",
            "points": len(checks), "runs": 2 * len(checks), "all_match": True,
            "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.output)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))
