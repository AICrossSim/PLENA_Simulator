"""Benchmark requests: stable hashes, the dev/held-out split, request groups and chat tokenization."""

import hashlib
import json
from pathlib import Path

SEED = 20261024
SEALED_SEED = 304104
DATASETS = ("bfcl", "gpqa", "swe")
BATCHES = (2, 4, 8, 16, 32, 64, 96, 128)
DECODE_STEPS = 32
MAX_PROMPT_TOKENS = 1024


def digest(value, compact=True):
    """SHA-256 of canonical JSON (sorted keys); compact=False keeps JSON's default separators."""
    separators = (",", ":") if compact else None
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=separators)
    return hashlib.sha256(text.encode()).hexdigest()


def request_hash(source, sample_id):
    return digest([source, str(sample_id)], compact=False)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    """Rows of a JSONL file, split at newlines only (JSON strings may contain U+2028)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_requests(rows, source):
    """Halve the distinct requests of a source into dev and held-out by a hash of their sample id."""
    unique = {}
    for r in rows:
        unique.setdefault(str(r["sample_id"]), r)
    ordered = sorted(unique.values(), key=lambda r: digest([SEED, source, str(r["sample_id"])]))
    half = len(ordered) // 2
    return {"dev": ordered[:half], "heldout": ordered[half:]}


def plan_groups(rows, source, batches=BATCHES, groups_small=4, groups_large=2):
    """Request groups of each split and batch size: disjoint within a batch size, at most groups_small of them up to
    batch 32 and groups_large above."""
    groups = []
    for split, items in split_requests(rows, source).items():
        for b in batches:
            ordered = sorted(items, key=lambda r: digest([SEED, "group", source, b, str(r["sample_id"])]))
            for c in range(min(groups_small if b <= 32 else groups_large, len(items) // b)):
                groups.append(
                    dict(
                        id=f"{source}_{split}_b{b}_c{c}",
                        source=source,
                        split=split,
                        batch=b,
                        cohort=c,
                        requests=ordered[c * b : (c + 1) * b],
                    )
                )
    return groups


def seen_requests(manifests):
    """Request hashes and record hashes of every request in the plans of earlier capture manifests. A plan entry may
    list request ids, request hashes or full requests."""
    ids, records = set(), set()
    for path in manifests:
        for g in json.loads(Path(path).read_text())["plan"]:
            for r in g.get("requests", []):
                ids.add(request_hash(g["source"], r["request_id"]))
                records.add(r["record_sha256"])
            ids.update(request_hash(g["source"], i) for i in g.get("request_ids", []))
            ids.update(g.get("request_hashes", []))
            records.update(g.get("record_hashes", []))
    return ids, records


def unseen_requests(rows, source, seen):
    """Requests absent from `seen`, ordered by a hash that is independent of the dev/held-out split."""
    ids, records = seen
    fresh = [r for r in rows if request_hash(source, r["sample_id"]) not in ids and digest(r) not in records]
    return sorted(fresh, key=lambda r: digest([SEALED_SEED, source, str(r["sample_id"])], compact=False))


def plan_sealed(fresh, source, batches=BATCHES, groups=4):
    """Up to `groups` request groups per batch size, taken in order from the unseen requests."""
    return [
        dict(
            id=f"{source}_sealed_b{b}_c{c}",
            source=source,
            split="sealed",
            batch=b,
            cohort=c,
            requests=fresh[c * b : (c + 1) * b],
        )
        for b in batches
        for c in range(min(groups, len(fresh) // b))
    ]


def merge_system_messages(messages):
    """Fold the leading system messages into one and turn later ones into user turns (the Qwen chat templates accept
    a single leading system message). Returns the messages and the number of changes."""
    leading, out, changes = [], [], 0
    for msg in messages:
        msg = dict(msg)
        if msg["role"] == "system" and not out:
            leading.append(str(msg["content"]))
            continue
        if msg["role"] == "system":
            msg["role"] = "user"
            msg["content"] = "[Instruction update for this turn]\n" + str(msg["content"])
            changes += 1
        out.append(msg)
    if leading:
        out.insert(0, dict(role="system", content="\n\n".join(leading)))
    return out, changes + max(0, len(leading) - 1)


def tokenize_groups(tokenizer, groups, max_prompt, merge_system=True):
    """Chat-format and tokenize every request of every group. Function schemas become a leading system message; a
    prompt longer than max_prompt keeps its first quarter and its last three quarters."""
    cache, out = {}, []
    for group in groups:
        requests = []
        for r in group["requests"]:
            messages = list(r["messages"])
            tools = r.get("tools") or r.get("function_schema")
            if tools:
                schema = "Available functions: " + json.dumps(tools, ensure_ascii=False)
                messages = [dict(role="system", content=schema), *messages]
            merged = 0
            if merge_system:
                messages, merged = merge_system_messages(messages)
            key = digest([messages, max_prompt])
            if key not in cache:
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                original = len(ids)
                if original > max_prompt:
                    ids = ids[: max_prompt // 4] + ids[-(max_prompt - max_prompt // 4) :]
                cache[key] = ids, original
            ids, original = cache[key]
            requests.append(
                dict(
                    request_id=str(r["sample_id"]),
                    record_sha256=digest(r),
                    input_ids=ids,
                    prompt_tokens=len(ids),
                    original_prompt_tokens=original,
                    merged_system_messages=merged,
                )
            )
        out.append(dict(group, requests=requests))
    return out


def public_plan(groups):
    """Plan entries for a manifest: request ids, record hashes and prompt lengths, without token ids."""
    return [
        {k: v for k, v in g.items() if k != "requests"}
        | dict(
            request_ids=[r["request_id"] for r in g["requests"]],
            record_hashes=[r["record_sha256"] for r in g["requests"]],
            prompt_lengths=[r["prompt_tokens"] for r in g["requests"]],
            original_prompt_lengths=[r["original_prompt_tokens"] for r in g["requests"]],
            merged_system_messages=sum(r["merged_system_messages"] for r in g["requests"]),
        )
        for g in groups
    ]
