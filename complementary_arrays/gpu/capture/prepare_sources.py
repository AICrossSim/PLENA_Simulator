"""Write the benchmark prompts as <dataset>.jsonl rows {sample_id, messages[, tools]}, one row per distinct sample id.

BFCL and SWE-bench come as prepared JSONL files (optionally gzipped). GPQA comes as the GPQA CSV; its answer options
are shuffled by a hash and the model is asked to answer and explain. Prompts are never printed.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

from .requests import digest, file_sha256, split_requests, write_json


def read_prepared(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_gpqa(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            question = r["Question"]
            options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            options = sorted(options, key=lambda answer: digest([question, answer]))
            listed = "\n".join(f"{chr(65 + i)}. {x}" for i, x in enumerate(options))
            prompt = question + "\n\n" + listed + "\nAnswer the question and explain your reasoning."
            rows.append(dict(sample_id="gpqa_" + digest(question), messages=[dict(role="user", content=prompt)]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bfcl", type=Path, required=True, help="prepared BFCL v3 JSONL")
    p.add_argument("--swe", type=Path, required=True, help="prepared SWE-bench JSONL")
    p.add_argument("--gpqa", type=Path, help="GPQA CSV (e.g. gpqa_extended.csv)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    inputs = {"bfcl": a.bfcl, "swe": a.swe} | ({"gpqa": a.gpqa} if a.gpqa else {})
    manifest = []
    for source, path in inputs.items():
        rows = read_gpqa(path) if source == "gpqa" else read_prepared(path)
        unique = {str(r["sample_id"]): r for r in rows}
        for r in unique.values():
            assert isinstance(r.get("messages"), list) and r["messages"], (source, list(r))
        out = a.out / f"{source}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in unique.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        halves = split_requests(list(unique.values()), source)
        manifest.append(
            dict(
                source=source,
                input=path.name,
                input_sha256=file_sha256(path),
                rows=len(rows),
                unique=len(unique),
                dev=len(halves["dev"]),
                heldout=len(halves["heldout"]),
                sha256=file_sha256(out),
            )
        )
        print(source, "unique", len(unique), "dev", len(halves["dev"]), "heldout", len(halves["heldout"]), flush=True)
    write_json(a.out / "source_manifest.json", manifest)


if __name__ == "__main__":
    main()
