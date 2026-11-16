"""Task files of the generation evaluations (full official sets; prompts as in the routing-capture pool). CPU only.

  gpqa     GPQA Diamond (198 questions, local CSV), options shuffled once per question (seed 37), simple-evals prompt
  math500  HuggingFaceH4/MATH-500 (500), problem + "Please reason step by step, ... \\boxed{}."
  aime     AIME 2024 (HuggingFaceH4/aime_2024) and AIME 2025 (opencompass/AIME2025, I and II), 60 problems, same prompt
  lcb      LiveCodeBench code_generation_lite v6 + v5, the 250 most recent by contest date (the pool's window), its chat
           prompt, with the public and private tests and metadata for execution

usage: python -m complementary_arrays.gpu.accuracy.prepare OUT_DIR --gpqa gpqa_diamond.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..capture import pool as P

GPQA_PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(path.name, len(rows), flush=True)


def gpqa(csv_path):
    rng, rows = np.random.default_rng(37), []
    with Path(csv_path).open() as f:
        for i, r in enumerate(csv.DictReader(f)):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            opts = [o.strip() for o in opts]
            perm = rng.permutation(4)
            a, b, c, d = (opts[j] for j in perm)
            text = GPQA_PROMPT.format(q=r["Question"].strip(), a=a, b=b, c=c, d=d)
            gold = "ABCD"[int(np.where(perm == 0)[0][0])]
            rows.append(
                dict(
                    id=f"gpqa_{i}",
                    messages=[dict(role="user", content=text)],
                    gold=gold,
                    domain=r.get("High-level domain"),
                )
            )
    return rows


def math500():
    rows = [json.loads(line) for line in P.hub_lines("HuggingFaceH4/MATH-500", "test.jsonl")]
    return [
        dict(
            id=r["unique_id"],
            messages=[dict(role="user", content=r["problem"] + P.MATH_SUFFIX)],
            gold=r["answer"],
            level=r.get("level"),
            subject=r.get("subject"),
        )
        for r in rows
    ]


def aime():
    rows = []
    for r in P.hub_parquet("HuggingFaceH4/aime_2024", "data/train-00000-of-00001.parquet"):
        rid = r.get("id", r.get("ID", len(rows)))
        rows.append(
            dict(
                id=f"aime2024_{rid}",
                messages=[dict(role="user", content=r["problem"] + P.MATH_SUFFIX)],
                gold=str(r["answer"]),
            )
        )
    for part in ("aime2025-I.jsonl", "aime2025-II.jsonl"):
        for k, line in enumerate(P.hub_lines("opencompass/AIME2025", part)):
            r = json.loads(line)
            msg = [dict(role="user", content=r["question"] + P.MATH_SUFFIX)]
            rows.append(dict(id=f"aime2025_{part[9:-6]}_{k}", messages=msg, gold=str(r["answer"])))
    return rows


def lcb():
    files = ("test6.jsonl", "test5.jsonl")
    dates = []
    for f in files:
        for line in P.hub_lines("livecodebench/code_generation_lite", f):
            r = json.loads(line)
            dates.append((r["contest_date"], str(r["question_id"]), f))
    keep = {(q, f) for _, q, f in sorted(dates, reverse=True)[:250]}
    rows = []
    for f in files:
        for line in P.hub_lines("livecodebench/code_generation_lite", f):
            r = json.loads(line)
            if (str(r["question_id"]), f) not in keep:
                continue
            item = P.lcb_item(f, r)
            rows.append(
                dict(
                    id=f"lcb_{r['question_id']}",
                    messages=item["record"]["messages"],
                    date=r["contest_date"],
                    platform=r["platform"],
                    difficulty=r["difficulty"],
                    starter_code=r.get("starter_code") or "",
                    public_test_cases=r["public_test_cases"],
                    private_test_cases=r["private_test_cases"],
                    metadata=r["metadata"],
                )
            )
    return sorted(rows, key=lambda x: (x["date"], x["id"]), reverse=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument(
        "--gpqa", type=Path, required=True, help="GPQA Diamond CSV (gpqa_diamond.csv of the official release)"
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "gpqa.jsonl", gpqa(a.gpqa))
    write(a.out / "math500.jsonl", math500())
    write(a.out / "aime.jsonl", aime())
    write(a.out / "lcb.jsonl", lcb())


if __name__ == "__main__":
    main()
