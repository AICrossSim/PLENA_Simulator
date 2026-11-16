"""Generate (and for gpqa, math500 and aime, score) one task against a running serve.py server.

Chat completions with the server's default sampling (the model's generation_config), n samples per item (GPQA and AIME
8, LiveCodeBench 4, MATH-500 1), at most 32,768 new tokens for Qwen3.5 (thinking mode, the default) and 8,192 for
DeepSeek-V2-Lite-Chat unless --max-tokens is given. A sample is truncated when it stops at the length limit. Saved per
sample: the final answer (after the last </think> for Qwen3.5), finish reason and score.
  gpqa            the last 'Answer: X' line (simple-evals); summarize.py re-scores with further answer patterns
  math500, aime   Math-Verify: gold parsed from '$answer$', prediction from the response
  lcb             the last Python code block of the answer, scored in a sandbox by lcb_score.sh

usage: python -m complementary_arrays.gpu.accuracy.generate {qwen35,deepseek} MODE TASK PORT OUT_DIR --data DIR
       [--n N] [--limit K] [--max-tokens N --tag long]
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\(?([A-D])\)?\$?")
CODE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)
SAMPLES = {"gpqa": 8, "aime": 8, "math500": 1, "lcb": 4}


def final_text(kind, text):
    if kind == "qwen35":
        return text.split("</think>")[-1] if "</think>" in text else ""
    return text


def score(task, item, final):
    """(score or None, prediction)."""
    if task == "gpqa":
        m = ANSWER.findall(final)
        return float(bool(m) and m[-1].upper() == item["gold"]), (m[-1].upper() if m else None)
    if task in ("math500", "aime"):
        from math_verify import parse, verify

        try:
            gold, pred = parse("$" + item["gold"] + "$"), parse(final)
            return float(bool(pred) and verify(gold, pred)), (
                str(pred[1]) if len(pred) > 1 else (str(pred[0]) if pred else None)
            )
        except Exception:
            return 0.0, None
    if task == "lcb":
        blocks = CODE.findall(final)
        return None, (blocks[-1] if blocks else "")
    raise ValueError(task)


async def run(a):
    from openai import AsyncOpenAI

    n = a.n or SAMPLES[a.task]
    max_tokens = a.max_tokens or (32768 if a.kind == "qwen35" else 8192)
    items = [json.loads(line) for line in open(a.data / f"{a.task}.jsonl")][: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / (f"{a.task}_{a.tag}.jsonl" if a.tag else f"{a.task}.jsonl")
    done = {json.loads(line)["id"] for line in open(path)} if path.exists() else set()
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x", timeout=7200, max_retries=2)
    sem, t0 = asyncio.Semaphore(a.concurrency), time.time()
    stats = dict(items=0, samples=0, truncated=0, score=0.0, scored=0)

    async def one(item):
        async with sem:
            r = await client.chat.completions.create(model="m", messages=item["messages"], n=n, max_tokens=max_tokens)
        samples = []
        for c in r.choices:
            text = c.message.content or ""
            final = final_text(a.kind, text)
            s, pred = score(a.task, item, final)
            samples.append(
                dict(
                    finish=c.finish_reason,
                    final=final if a.task != "lcb" else None,
                    pred=pred,
                    score=s,
                    think_closed=("</think>" in text) if a.kind == "qwen35" else None,
                )
            )
        rec = dict(
            id=item["id"],
            kind=a.kind,
            mode=a.mode,
            task=a.task,
            tag=a.tag,
            max_tokens=max_tokens,
            n=n,
            completion_tokens=r.usage.completion_tokens if r.usage else None,
            samples=samples,
        )
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        stats["items"] += 1
        stats["samples"] += n
        stats["truncated"] += sum(x["finish"] == "length" for x in samples)
        scored = [x["score"] for x in samples if x["score"] is not None]
        stats["score"] += sum(scored)
        stats["scored"] += len(scored)
        if stats["items"] % 25 == 0:
            print(time.strftime("%H:%M:%S"), a.task, stats, f"{time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(it) for it in items if it["id"] not in done))
    acc = stats["score"] / stats["scored"] if stats["scored"] else None
    print(
        "GENERATE_DONE",
        a.kind,
        a.mode,
        a.task,
        json.dumps(dict(stats, acc=acc, seconds=round(time.time() - t0))),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=("qwen35", "deepseek"))
    p.add_argument("mode")
    p.add_argument("task", choices=tuple(SAMPLES))
    p.add_argument("port")
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path, required=True, help="task files of prepare.py")
    p.add_argument("--n", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--tag", default="")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
