"""Request pool for batch 256 and 512: 27,000 requests of the six task families of pool.py, disjoint from the excluded
earlier captures and from the pool of pool.py, the same requests for both models. CPU only.

  bfcl and lcb take every remaining request (all LiveCodeBench releases, a problem once); the rest of the budget is
  split equally over swe, gpqa, wildchat and math, each capped by what exists, and what one cannot fill goes to the
  others:
    swe       unseen requests of the SWE request file, then SWE-bench_bm25_27K dev + validation, then its train split;
    gpqa      SuperGPQA, discipline Science, fields Physics / Chemistry / Biology first, then the other Science fields;
    wildchat  WildChat-1M, the first 3,000 eligible conversations per shard (filter of pool.py);
    math      MATH-500, then MATH (EleutherAI/hendrycks_math, train + test) without MATH-500 duplicates.
  The arrival share of a source is its share of the pool. Split per source by digest: 4/27 dev, 23/27 eval; every
  request is sampled (no greedy subset); seed per request = digest([SEED, 'seed', source, id])[:8].

usage: python -m complementary_arrays.gpu.capture.pool_large OUT --sources DIR --tokenizer qwen35=DIR
       --tokenizer deepseek=DIR --previous POOL_MANIFEST [--exclude MANIFEST ...]
"""

import gc
import json

from . import pool as P
from . import requests as R

SEED = 20261116
TOTAL, DEV = 27000, 4000
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def D(*a):
    return P.D(SEED, *a)


def lcb_all(fresh):
    files = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")
    seen, out = set(), []
    for x in P.lcb_items(files):
        if x["dataset_id"] not in seen:
            seen.add(x["dataset_id"])
            out.append(x)
    return fresh("lcb", out)


def swe_all(sources, seen, fresh, need):
    out = fresh("swe", P.local_source(sources, "swe", seen)[0])
    files = [("dev", "dev-00000-of-00001"), ("validation", "validation-00000-of-00001")]
    files += [("train", f"train-{i:05d}-of-00006") for i in range(6)]
    for split, fname in files:
        if len(out) >= need:
            break
        part = sorted(P.swe_rows(fname), key=lambda r: D("pool", "swe", split, r["instance_id"]))
        out += fresh("swe", [P.swe_item(split, r) for r in part])
        del part
        gc.collect()
        P.log("swe", split, len(out))
    return out


def gpqa_all(fresh):
    rows = [json.loads(line) for line in P.hub_lines("m-a-p/SuperGPQA", "SuperGPQA-all.jsonl")]
    sci = [r for r in rows if r.get("discipline") == "Science"]
    core = ("Physics", "Chemistry", "Biology")
    key = lambda r: D("pool", "gpqa", str(r["uuid"]))  # noqa: E731
    first = sorted([r for r in sci if r.get("field") in core], key=key)
    rest = sorted([r for r in sci if r.get("field") not in core], key=key)
    return fresh("gpqa", [P.supergpqa_item(r) for r in first + rest])


def math_all(fresh, previous):
    m500 = [json.loads(line) for line in P.hub_lines("HuggingFaceH4/MATH-500", "test.jsonl")]
    seen_text = {R.digest([r["problem"].strip()]) for r in m500}
    extra = []
    for subj in MATH_SUBJECTS:
        for split in ("train", "test"):
            for r in P.hub_parquet("EleutherAI/hendrycks_math", f"{subj}/{split}-00000-of-00001.parquet"):
                h = R.digest([r["problem"].strip()])
                if h not in seen_text:
                    seen_text.add(h)
                    extra.append(
                        P.math_item(
                            f"EleutherAI/hendrycks_math/{subj}/{split}", h[:24], r["problem"], level=r.get("level")
                        )
                    )
        P.log("math", subj, len(extra))
    m500 = sorted(m500, key=lambda r: D("pool", "math500", r["unique_id"]))
    items = [P.math_item("HuggingFaceH4/MATH-500", r["unique_id"], r["problem"], level=r.get("level")) for r in m500]
    items = [x for x in items if ("math", x["dataset_id"]) not in previous]
    return items + sorted(extra, key=lambda x: D("pool", "math", x["dataset_id"]))


def allocate(avail, total):
    """bfcl and lcb take all they have; the rest of total goes equally to the other sources, capped, the remainder
    redistributed."""
    take = {s: avail[s] for s in ("bfcl", "lcb")}
    rest = [s for s in avail if s not in take]
    left = total - sum(take.values())
    while rest:
        share = left // len(rest)
        capped = [s for s in rest if avail[s] <= share]
        if not capped:
            for i, s in enumerate(sorted(rest)):
                take[s] = share + (1 if i < left - share * len(rest) else 0)
            break
        for s in capped:
            take[s] = avail[s]
            left -= avail[s]
            rest.remove(s)
    return take


def main():
    p = P.parse_args(__doc__)
    p.add_argument("--previous", type=P.Path, required=True, help="pool_manifest.json of pool.py")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tokenizers = dict(t.split("=", 1) for t in a.tokenizer)
    manifests = [m for m in a.exclude if "plan" in json.loads(m.read_text())]
    seen = R.seen_requests(manifests)
    previous = {(r["source"], str(r["dataset_id"])) for r in json.loads(a.previous.read_text())["requests"]}

    def fresh(src, items):
        return [x for x in items if (src, str(x["dataset_id"])) not in previous]

    info = dict(
        seed=SEED,
        total=TOTAL,
        dev=DEV,
        manifests_excluded=[R.file_sha256(m) for m in manifests],
        previous_excluded=len(previous),
        sources={},
    )
    pools = {"bfcl": fresh("bfcl", P.local_source(a.sources, "bfcl", seen)[0]), "lcb": lcb_all(fresh)}
    need = (TOTAL - len(pools["bfcl"]) - len(pools["lcb"])) // 4 + 1
    pools["swe"] = swe_all(a.sources, seen, fresh, need)
    pools["gpqa"] = gpqa_all(fresh)
    cand, scanned = P.wildchat(per_shard=3000)
    pools["wildchat"] = fresh("wildchat", sorted(cand, key=lambda c: D("pool", "wildchat", c["dataset_id"])))
    info["sources"]["wildchat"] = dict(scanned=scanned)
    pools["math"] = math_all(fresh, previous)
    avail = {s: len(v) for s, v in pools.items()}
    take = allocate(avail, TOTAL)
    P.log("available", avail, "taken", take)
    pools = {s: items[: take[s]] for s, items in pools.items()}  # each list is in its digest order
    for s, items in pools.items():
        datasets = {}
        for x in items:
            datasets[x["dataset"]] = datasets.get(x["dataset"], 0) + 1
        info["sources"].setdefault(s, {}).update(available=avail[s], taken=len(items), datasets=datasets)
    entries = P.assign(pools, SEED, split_of=lambda k, n: "dev" if k < round(n * DEV / TOTAL) else "eval", greedy=0)
    del pools
    gc.collect()
    P.tokenize(entries, tokenizers, a.out, SEED, info)
    info["split_counts"] = {
        s: {sp: sum(e["source"] == s and e["split"] == sp for e in entries) for sp in ("dev", "eval")}
        for s in sorted({e["source"] for e in entries})
    }
    info["requests"] = [{k: v for k, v in e.items() if k != "_record"} for e in entries]
    (a.out / "pool_manifest.json").write_text(json.dumps(info, indent=1))
    P.log("done", len(entries), json.dumps(info["split_counts"]))


if __name__ == "__main__":
    main()
