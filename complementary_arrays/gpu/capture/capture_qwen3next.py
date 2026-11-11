"""Record Qwen3-Next-80B-A3B routing on the dev/held-out request groups of capture_routes.

The official FP8 checkpoint runs natively, or with RTN-quantized routed experts. Its hybrid cache cannot drop requests,
so finished requests stay in the batch; every step stores the routes of the requests that have not consumed EOS, and
decoding stops when none is left.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from .decoding import (
    RouterRecorder,
    eos_and_pad,
    finish_manifest,
    left_padded,
    record_live_step,
    runtime,
    save_shard,
    start_manifest,
)
from .requests import (
    BATCHES,
    DATASETS,
    DECODE_STEPS,
    MAX_PROMPT_TOKENS,
    file_sha256,
    plan_groups,
    public_plan,
    read_jsonl,
    tokenize_groups,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True, help="directory of the official FP8 checkpoint")
    p.add_argument("--sources", type=Path, required=True, help="directory with <dataset>.jsonl from prepare_sources")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", choices=("fp8", "w8g128", "w4g32"), default="fp8", help="routed-expert weights")
    p.add_argument("--fp8-kernel", type=Path, help="local snapshot of kernels-community/finegrained-fp8")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--splits", default="dev,heldout")
    p.add_argument("--batches", default=",".join(map(str, BATCHES)))
    p.add_argument("--steps", type=int, default=DECODE_STEPS)
    p.add_argument("--max-prompt", type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    import torch
    from transformers import AutoTokenizer

    from ..models import load_qwen3next, moe_workload, read_config

    datasets, splits = a.datasets.split(","), set(a.splits.split(","))
    batches = [int(b) for b in a.batches.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    groups = [g for s in datasets for g in plan_groups(read_jsonl(a.sources / f"{s}.jsonl"), s, batches)]
    groups = tokenize_groups(tokenizer, [g for g in groups if g["split"] in splits], a.max_prompt)
    workload = moe_workload(read_config(a.model))
    model, layers, quantization = load_qwen3next(a.model, a.format, a.fp8_kernel)
    gate = layers[min(layers)].gate
    assert (gate.top_k, gate.num_experts) == (workload["topk"], workload["experts"])

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        kind="qwen3next",
        model=a.model.name,
        config_sha256=file_sha256(a.model / "config.json"),
        weights="FP8 checkpoint run natively" if a.format == "fp8" else f"routed experts {a.format}, rest FP8",
        format=a.format,
        quantization=quantization,
        experts_class=type(layers[min(layers)].experts).__name__,
        workload=workload,
        sources={s: file_sha256(a.sources / f"{s}.jsonl") for s in datasets},
        steps=a.steps,
        max_prompt=a.max_prompt,
        system_messages="merge",
        eos="finished requests stay in the batch; only requests that have not consumed EOS are stored",
        plan=public_plan(groups),
        layers=list(layers),
        **runtime(),
        shards=[],
        complete=False,
    )
    path = start_manifest(a.out, manifest, a.resume, ("kind", "config_sha256", "format", "steps", "max_prompt", "plan"))

    eos, pad = eos_and_pad(tokenizer)
    recorder = RouterRecorder(layers, workload["topk"])
    done = {s["id"] for s in manifest["shards"]}
    for group in groups:
        if group["id"] in done:
            continue
        start, b = time.monotonic(), group["batch"]
        ids, mask = left_padded(group["requests"], pad)
        finished, arrays, live_counts = np.zeros(b, dtype=bool), {}, []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            position = (mask.cumsum(-1) - 1).clamp_min(0)
            out = model(input_ids=ids, attention_mask=mask, position_ids=position, use_cache=True, logits_to_keep=1)
            for step in range(1, a.steps + 1):
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                finished |= np.isin(token[:, 0].cpu().numpy(), list(eos))
                if finished.all():
                    break
                active = np.flatnonzero(~finished)
                mask = torch.cat((mask, torch.ones((b, 1), dtype=mask.dtype, device="cuda")), 1)
                with recorder.step(b):
                    out = model(
                        input_ids=token,
                        attention_mask=mask,
                        position_ids=mask.sum(-1, keepdim=True) - 1,
                        past_key_values=out.past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                assert torch.isfinite(out.logits).all(), "non-finite logits"
                live_counts.append(len(active))
                record_live_step(arrays, step, active, recorder, rows=active)
        arrays["live_counts"] = np.array(live_counts)
        save_shard(
            path,
            manifest,
            group,
            arrays,
            live_counts=live_counts,
            seconds=time.monotonic() - start,
            peak_gpu_MiB=torch.cuda.max_memory_allocated() / 2**20,
        )
        del out, ids, mask
        gc.collect()
        torch.cuda.empty_cache()
    finish_manifest(path, manifest, groups)


if __name__ == "__main__":
    main()
