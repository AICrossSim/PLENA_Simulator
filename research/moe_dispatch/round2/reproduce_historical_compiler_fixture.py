"""Verify the frozen hybrid compiler evidence without changing old results.

Run from the Simulator root with the regular Compiler test Python. The
historical compiler pin comes from the exact Git tree that wrote the reports.
Current compiler reports are validated separately by the live unit suite.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile


def main():
    repo = Path(__file__).resolve().parents[3]
    fixture = repo / "analytic_models/performance/profiles/historical_hybrid_compiler_evidence.json"
    frozen = json.loads(fixture.read_text())
    pin = subprocess.check_output(
        ["git", "ls-tree", frozen["provenance"]["simulator_artifact_commit"], "PLENA_Compiler"],
        cwd=repo, text=True,
    ).split()[2]
    assert pin == frozen["compiler_sha"]
    with tempfile.TemporaryDirectory(prefix="plena-historical-compiler-", dir="/tmp") as tmp:
        root = Path(tmp)
        archive = root / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", pin], cwd=repo / "PLENA_Compiler",
                           stdout=stream, check=True)
        source = root / "compiler"
        source.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter="data")
        for name, expected in frozen["source_sha256"].items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
        code = """
from pathlib import Path
import json,sys
from analytic_models.performance.hybrid_lcompute_campaign import load_compiler_evidence,paper_2048_hardware_point,_sha256_json
r=Path(sys.argv[1]); f=json.loads(Path(sys.argv[2]).read_text())
for key,hw in [('historical64',None),('paper2048',paper_2048_hardware_point())]:
    now=load_compiler_evidence(r,hw)
    assert _sha256_json(now)==_sha256_json(f[key]),key
    print(key,_sha256_json(now),'exact historical compiler snapshot')
"""
        subprocess.run([sys.executable, "-c", code, str(source), str(fixture)],
                       cwd=repo, check=True)


if __name__ == "__main__":
    main()
