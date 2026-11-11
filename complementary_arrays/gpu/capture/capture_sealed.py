"""Record routing on a sealed request group: requests that none of the given earlier captures used.

Decoding is as in capture_routes, except that a request leaves the batch (with its cache rows) once it emits EOS and
decoding stops when no request is left. Every step stores the routes of the remaining requests.
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
    SEED,
    file_sha256,
    plan_sealed,
    public_plan,
    read_jsonl,
    seen_requests,
    tokenize_groups,
    unseen_requests,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-kind", choices=("qwen35", "deepseek"), required=True)
    p.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    p.add_argument("--sources", type=Path, required=True, help="directory with <dataset>.jsonl from prepare_sources")
    p.add_argument("--exclude", type=Path, nargs="+", required=True, help="manifests of earlier captures")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--batches", default=",".join(map(str, BATCHES)))
    p.add_argument("--groups", type=int, default=4, help="request groups per dataset and batch size")
    p.add_argument("--steps", type=int, default=DECODE_STEPS)
    p.add_argument("--max-prompt", type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument("--system-messages", choices=("merge", "keep"), default="merge")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    datasets, batches = a.datasets.split(","), [int(b) for b in a.batches.split(",")]
    seen = seen_requests(a.exclude)
    groups, unseen = [], {}
    for source in datasets:
        fresh = unseen_requests(read_jsonl(a.sources / f"{source}.jsonl"), source, seen)
        unseen[source] = len(fresh)
        groups += plan_sealed(fresh, source, batches, a.groups)
    print("UNSEEN", unseen, "groups", len(groups), flush=True)

    import torch
    from transformers import AutoTokenizer

    from ..models import load_bf16_moe, moe_workload, read_config, use_vllm_experts

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    groups = tokenize_groups(tokenizer, groups, a.max_prompt, a.system_messages == "merge")
    config = read_config(a.model)
    workload = moe_workload(config)
    model, layers = load_bf16_moe(a.model, a.model_kind)
    fused_error = use_vllm_experts(layers)

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        kind=a.model_kind,
        model=a.model.name,
        config_sha256=file_sha256(a.model / "config.json"),
        weights="FP8 checkpoint dequantized to BF16" if "quantization_config" in config else "BF16",
        workload=workload,
        sources={s: file_sha256(a.sources / f"{s}.jsonl") for s in datasets},
        excluded_manifests=[file_sha256(p) for p in a.exclude],
        unseen_requests=unseen,
        steps=a.steps,
        max_prompt=a.max_prompt,
        system_messages=a.system_messages,
        eos="a request leaves the batch once it emits EOS; decoding stops when none is left",
        plan=public_plan(groups),
        layers=list(layers),
        fused_experts_rel_l2=fused_error,
        **runtime(),
        shards=[],
        complete=False,
    )
    path = start_manifest(a.out, manifest, a.resume, ("kind", "config_sha256", "steps", "max_prompt", "plan"))

    eos, pad = eos_and_pad(tokenizer)
    recorder = RouterRecorder(layers, workload["topk"])
    done = {s["id"] for s in manifest["shards"]}
    for group in groups:
        if group["id"] in done:
            continue
        start = time.monotonic()
        ids, mask = left_padded(group["requests"], pad)
        if a.model_kind == "qwen35":
            model.model.rope_deltas = None  # multimodal RoPE offsets cached from the previous batch
        active, arrays, live_counts = np.arange(group["batch"]), {}, []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            position = (mask.cumsum(-1) - 1).clamp_min(0)
            out = model(input_ids=ids, attention_mask=mask, position_ids=position, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            for step in range(1, a.steps + 1):
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                alive = ~np.isin(token[:, 0].cpu().numpy(), list(eos))
                if not alive.any():
                    break
                if not alive.all():
                    keep = torch.tensor(np.flatnonzero(alive), device="cuda")
                    cache.batch_select_indices(keep)
                    mask, token = mask.index_select(0, keep), token.index_select(0, keep)
                    if a.model_kind == "qwen35" and model.model.rope_deltas is not None:
                        model.model.rope_deltas = model.model.rope_deltas.index_select(0, keep)
                    active = active[alive]
                mask = torch.cat((mask, torch.ones((len(active), 1), dtype=mask.dtype, device="cuda")), 1)
                with recorder.step(len(active)):
                    out = model(
                        input_ids=token,
                        attention_mask=mask,
                        position_ids=mask.sum(-1, keepdim=True) - 1,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                assert torch.isfinite(out.logits).all(), "non-finite logits"
                cache = out.past_key_values
                live_counts.append(len(active))
                record_live_step(arrays, step, active, recorder)
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
        del out, cache, ids, mask
        gc.collect()
        torch.cuda.empty_cache()
    finish_manifest(path, manifest, groups)


if __name__ == "__main__":
    main()
