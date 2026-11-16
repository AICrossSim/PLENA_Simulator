"""Request pool of the routing captures: six sources x 250 requests, the same requests for both models. CPU only.

  bfcl, swe, gpqa  the request files of prepare_sources minus every request of the excluded earlier captures, in
                   digest([SEED, 'pool', source, id]) order. SWE is filled up from SWE-bench_bm25_27K dev + validation
                   (same prompt format), GPQA from SuperGPQA (discipline Science, fields Physics / Chemistry / Biology,
                   formatted like the GPQA requests), when too few unseen requests remain.
  wildchat         allenai/WildChat-1M, read by HTTP byte ranges: per parquet shard the first 1,500 conversations that
                   are neither toxic nor redacted and have a non-empty first user turn (repeated first turns skipped);
                   the first user turn; 250 by digest order.
  math             every AIME 2024 and AIME 2025 problem, the rest from MATH-500 in digest order, with the prompt
                   suffix "Please reason step by step, and put your final answer within \\boxed{}."
  lcb              LiveCodeBench code_generation_lite releases v6 then v5, the 250 most recent by contest date, with
                   the benchmark's generic chat prompt.
Split per source by digest([SEED, 'split', source, id]): first 40% dev, next 40% eval, last 20% sealed. Seed of a
request: digest([SEED, 'seed', source, id])[:8]. Greedy subset: the 10% smallest digest([SEED, 'greedy', source, id]).
Tokenization: requests.tokenize_groups with the chat template defaults, function schemas as a leading system message,
Qwen3.5 merging system messages and DeepSeek keeping them, at most 4,096 prompt tokens (longer prompts keep their first
quarter and last three quarters). Only token ids and hashes are written; texts stay in memory.

usage: python -m complementary_arrays.gpu.capture.pool OUT --sources DIR --tokenizer qwen35=DIR --tokenizer deepseek=DIR
       [--exclude MANIFEST ...]   -> OUT/pool_<model>.pkl, OUT/pool_manifest.json
"""

import argparse
import gc
import io
import json
import os
import pickle
import time
from pathlib import Path

from . import requests as R

SEED = 20261113
N = 250
MAX_PROMPT = 4096
MERGE_SYSTEM = {"qwen35": True, "deepseek": False}
SWE_PREFIX = "Analyze the following repository issue and retrieved code. Explain the root cause and propose a concrete patch.\n\n"
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
GPQA_SUFFIX = "\nAnswer the question and explain your reasoning."
LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
    "correct Python program that matches the specification and passes all tests."
)
LCB_STARTER = (
    "### Format: You will use the following starter code to write the solution to the problem and enclose your code "
    "within delimiters.\n```python\n{}\n```\n\n### Answer: (use the provided format with backticks)\n\n"
)
LCB_STDIN = (
    "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
    "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it "
    "reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"
    "### Answer: (use the provided format with backticks)\n\n"
)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def D(seed, *a):
    return R.digest([seed, *a])


def hub_lines(repo, fname):
    """Lines of a dataset file streamed over HTTP (never written to disk)."""
    import requests
    from huggingface_hub import hf_hub_url

    with requests.get(hf_hub_url(repo, fname, repo_type="dataset"), stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1 << 20):
            if line:
                yield line


def hub_parquet(repo, fname, columns=None):
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import hf_hub_url

    data = requests.get(hf_hub_url(repo, fname, repo_type="dataset"), timeout=600).content
    return pq.read_table(io.BytesIO(data), columns=columns).to_pylist()


def hub_parquet_file(path):
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(path, "rb", block_size=8 << 20)


def local_source(sources, name, seen, seed=SEED):
    """Unseen requests of a request file in pool order, with the file's row and distinct-request counts."""
    rows = R.read_jsonl(Path(sources) / f"{name}.jsonl")
    fresh = sorted(R.unseen_requests(rows, name, seen), key=lambda r: D(seed, "pool", name, str(r["sample_id"])))
    items = [dict(dataset=f"request_file/{name}", dataset_id=str(r["sample_id"]), record=r) for r in fresh]
    return items, len(rows), len({str(r["sample_id"]) for r in rows})


def supergpqa_item(r):
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r["options"]))
    record = dict(
        sample_id=f"supergpqa_{r['uuid']}",
        messages=[dict(role="user", content=f"{r['question']}\n\n{opts}{GPQA_SUFFIX}")],
    )
    return dict(
        dataset="m-a-p/SuperGPQA",
        dataset_id=str(r["uuid"]),
        field=r["field"],
        difficulty=r.get("difficulty"),
        record=record,
    )


def supergpqa(n, seed=SEED):
    rows = [json.loads(line) for line in hub_lines("m-a-p/SuperGPQA", "SuperGPQA-all.jsonl")]
    keep = [
        r for r in rows if r.get("discipline") == "Science" and r.get("field") in ("Physics", "Chemistry", "Biology")
    ]
    keep = sorted(keep, key=lambda r: D(seed, "pool", "supergpqa", str(r["uuid"])))[:n]
    return [supergpqa_item(r) for r in keep], len(rows)


def swe_item(split, r):
    record = dict(
        sample_id=f"swebench_{split}_{r['instance_id']}", messages=[dict(role="user", content=SWE_PREFIX + r["text"])]
    )
    return dict(
        dataset=f"princeton-nlp/SWE-bench_bm25_27K/{split}", dataset_id=r["instance_id"], repo=r["repo"], record=record
    )


def swe_rows(fname):
    import pyarrow.parquet as pq

    with hub_parquet_file(f"datasets/princeton-nlp/SWE-bench_bm25_27K/data/{fname}.parquet") as fh:
        return pq.read_table(fh, columns=["instance_id", "text", "repo"]).to_pylist()


def swe_substitute(n, seed=SEED):
    """SWE-bench_bm25_27K dev + validation (the request file holds its test split)."""
    rows = [dict(r, _split=s) for s in ("dev", "validation") for r in swe_rows(f"{s}-00000-of-00001")]
    keep = sorted(rows, key=lambda r: D(seed, "pool", "swebench_devval", r["instance_id"]))[:n]
    return [swe_item(r["_split"], r) for r in keep], len(rows)


def wildchat(per_shard=1500, shards=14):
    """Every eligible first user turn among the first per_shard per parquet shard, as (items, conversations scanned)."""
    import pyarrow.parquet as pq

    cand, seen_text, scanned = [], set(), 0
    columns = ["conversation_hash", "conversation", "toxic", "redacted", "language"]
    for i in range(shards):
        with hub_parquet_file(f"datasets/allenai/WildChat-1M/data/train-{i:05d}-of-{shards:05d}.parquet") as fh:
            f = pq.ParquetFile(fh)  # byte-range reads: the footer and the row groups used
            got = 0
            for rg in range(f.num_row_groups):
                for r in f.read_row_group(rg, columns=columns).to_pylist():
                    scanned += 1
                    if r["toxic"] or r["redacted"] or not r["conversation"]:
                        continue
                    first = next((t for t in r["conversation"] if t.get("role") == "user"), None)
                    text = (first or {}).get("content") or ""
                    h = R.digest([text])
                    if not text.strip() or h in seen_text:
                        continue
                    seen_text.add(h)
                    got += 1
                    record = dict(
                        sample_id=f"wildchat_{r['conversation_hash']}", messages=[dict(role="user", content=text)]
                    )
                    cand.append(
                        dict(
                            dataset="allenai/WildChat-1M",
                            dataset_id=r["conversation_hash"],
                            language=r.get("language"),
                            shard=i,
                            record=record,
                        )
                    )
                    if got >= per_shard:
                        break
                if got >= per_shard:
                    break
            del f
        gc.collect()
        log("wildchat shard", i, "candidates", got)
    return cand, scanned


def math_item(dataset, dataset_id, text, **extra):
    record = dict(
        sample_id=f"math_{R.digest([dataset, dataset_id])[:24]}",
        messages=[dict(role="user", content=text + MATH_SUFFIX)],
    )
    return dict(dataset=dataset, dataset_id=dataset_id, **extra, record=record)


def math_pool(seed=SEED):
    aime = []
    for r in hub_parquet("HuggingFaceH4/aime_2024", "data/train-00000-of-00001.parquet"):
        aime.append(
            (
                "HuggingFaceH4/aime_2024",
                str(r.get("id", r.get("ID", ""))) or R.digest([r["problem"]])[:16],
                r["problem"],
            )
        )
    for part in ("aime2025-I.jsonl", "aime2025-II.jsonl"):
        for k, line in enumerate(hub_lines("opencompass/AIME2025", part)):
            aime.append((f"opencompass/AIME2025/{part}", f"{part}:{k}", json.loads(line)["question"]))
    m500 = [json.loads(line) for line in hub_lines("HuggingFaceH4/MATH-500", "test.jsonl")]
    m500 = sorted(m500, key=lambda r: D(seed, "pool", "math500", r["unique_id"]))[: N - len(aime)]
    items = [math_item(*x) for x in aime]
    items += [math_item("HuggingFaceH4/MATH-500", r["unique_id"], r["problem"], level=r.get("level")) for r in m500]
    return items, len(aime)


def lcb_item(fname, r):
    starter = r.get("starter_code") or ""
    fmt = LCB_STARTER.format(starter) if starter.strip() else LCB_STDIN
    record = dict(
        sample_id=f"lcb_{r['question_id']}",
        messages=[
            dict(role="system", content=LCB_SYSTEM),
            dict(role="user", content=f"### Question:\n{r['question_content']}\n\n{fmt}"),
        ],
    )
    return dict(
        dataset=f"livecodebench/code_generation_lite/{fname}",
        dataset_id=str(r["question_id"]),
        date=r["contest_date"],
        platform=r["platform"],
        difficulty=r["difficulty"],
        record=record,
    )


def lcb_items(files):
    items = []
    for fname in files:
        for line in hub_lines("livecodebench/code_generation_lite", fname):
            items.append(lcb_item(fname, json.loads(line)))
        log("lcb", fname, len(items))
    return items


def lcb():
    items = lcb_items(("test6.jsonl", "test5.jsonl"))
    for x in items:
        x["release"] = "v6" if x["dataset"].endswith("test6.jsonl") else "v5"
    return sorted(items, key=lambda x: (x["date"], x["dataset_id"]), reverse=True)[:N]


def split_40_40_20(k, n):
    return "dev" if k < 0.4 * n else "eval" if k < 0.8 * n else "sealed"


def assign(pools, seed=SEED, split_of=split_40_40_20, greedy=N // 10):
    """Pool entries with split (split_of(rank, n) by split digest), request hash, record hash, seed and greedy-subset
    flag (the `greedy` smallest greedy digests of a source)."""
    entries = []
    for src, items in pools.items():
        n = len(items)
        order = sorted(items, key=lambda x: D(seed, "split", src, x["dataset_id"]))
        chosen = {
            x["dataset_id"] for x in sorted(items, key=lambda x: D(seed, "greedy", src, x["dataset_id"]))[:greedy]
        }
        for k, x in enumerate(order):
            split = split_of(k, n)
            meta = {kk: v for kk, v in x.items() if kk != "record"}
            entries.append(
                dict(
                    meta,
                    source=src,
                    split=split,
                    request=R.request_hash(src, x["record"]["sample_id"]),
                    record_sha256=R.digest(x["record"]),
                    seed=int(D(seed, "seed", src, x["dataset_id"])[:8], 16),
                    greedy_subset=x["dataset_id"] in chosen,
                    _record=x["record"],
                )
            )
    return entries


def tokenize(entries, tokenizers, out, seed, info):
    """Write pool_<model>.pkl (token ids and hashes) for every model and add prompt statistics to info."""
    from transformers import AutoTokenizer

    sources = sorted({e["source"] for e in entries})
    for kind, path in tokenizers.items():
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        group = dict(id="pool", requests=[e["_record"] for e in entries])
        toks = R.tokenize_groups(tok, [group], MAX_PROMPT, MERGE_SYSTEM[kind])[0]["requests"]
        rows = []
        for e, t in zip(entries, toks, strict=True):
            assert t["record_sha256"] == e["record_sha256"]
            rows.append(
                dict(
                    {k: v for k, v in e.items() if k != "_record"},
                    input_ids=t["input_ids"],
                    prompt_tokens=t["prompt_tokens"],
                    original_prompt_tokens=t["original_prompt_tokens"],
                )
            )
        (Path(out) / f"pool_{kind}.pkl").write_bytes(pickle.dumps(dict(model=kind, seed=seed, rows=rows)))
        stats = {}
        for s in sources:
            sel = [r for r in rows if r["source"] == s]
            stats[s] = dict(
                n=len(sel),
                mean=sum(r["prompt_tokens"] for r in sel) / len(sel),
                cut=sum(r["original_prompt_tokens"] > MAX_PROMPT for r in sel),
            )
        info[f"prompt_tokens_{kind}"] = stats
        log(kind, "tokenized", json.dumps(stats))


def parse_args(doc):
    p = argparse.ArgumentParser(description=doc.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--sources", type=Path, required=True, help="request files of prepare_sources")
    p.add_argument("--tokenizer", action="append", required=True, help="model=checkpoint directory (qwen35, deepseek)")
    p.add_argument("--exclude", type=Path, nargs="*", default=[], help="manifests of earlier captures")
    return p


def main():
    a = parse_args(__doc__).parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tokenizers = dict(t.split("=", 1) for t in a.tokenizer)
    manifests = [p for p in a.exclude if "plan" in json.loads(p.read_text())]
    seen = R.seen_requests(manifests)
    info = dict(seed=SEED, n_per_source=N, manifests_excluded=[R.file_sha256(p) for p in manifests], sources={})
    pools = {}
    for name in ("bfcl", "swe", "gpqa"):
        fresh, nrows, distinct = local_source(a.sources, name, seen)
        info["sources"][name] = dict(file_rows=nrows, distinct=distinct, unseen=len(fresh))
        pools[name] = fresh[:N]
        log(name, "rows", nrows, "distinct", distinct, "unseen", len(fresh))
    if len(pools["swe"]) < N:
        sub, total = swe_substitute(N - len(pools["swe"]))
        pools["swe"] += sub
        info["sources"]["swe"].update(
            substitute="princeton-nlp/SWE-bench_bm25_27K dev + validation", substitute_n=len(sub), substitute_pool=total
        )
    if len(pools["gpqa"]) < N:
        sub, total = supergpqa(N - len(pools["gpqa"]))
        pools["gpqa"] += sub
        info["sources"]["gpqa"].update(
            substitute="m-a-p/SuperGPQA, Science: Physics / Chemistry / Biology",
            substitute_n=len(sub),
            supergpqa_rows=total,
        )
    cand, scanned = wildchat()
    pools["wildchat"] = sorted(cand, key=lambda c: D(SEED, "pool", "wildchat", c["dataset_id"]))[:N]
    info["sources"]["wildchat"] = dict(scanned=scanned, candidates=len(cand))
    pools["math"], naime = math_pool()
    info["sources"]["math"] = dict(aime=naime, math500=N - naime)
    pools["lcb"] = lcb()
    info["sources"]["lcb"] = dict(
        date_min=min(x["date"] for x in pools["lcb"]), date_max=max(x["date"] for x in pools["lcb"])
    )
    for src, items in pools.items():
        assert len(items) == N, (src, len(items))
    entries = assign(pools)
    tokenize(entries, tokenizers, a.out, SEED, info)
    info["requests"] = [{k: v for k, v in e.items() if k != "_record"} for e in entries]
    (a.out / "pool_manifest.json").write_text(json.dumps(info, indent=1))
    log("done", len(entries))


if __name__ == "__main__":
    main()
