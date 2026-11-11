"""Record the experts that every MoE layer selects for each token while Qwen3.5-35B-A3B or DeepSeek-V2-Lite decodes.

Each request group decodes as one batch: a left-padded prefill, then greedy decode steps. Requests that emit EOS stay
in the batch, and the per-step alive mask marks the requests that have not consumed EOS. Only decode steps are
recorded. The routed experts run through vLLM's fused_experts, checked against the model's own implementation.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from .decoding import RouterRecorder, eos_and_pad, finish_manifest, left_padded, runtime, save_shard, start_manifest
from .requests import (
    BATCHES,
    DATASETS,
    DECODE_STEPS,
    MAX_PROMPT_TOKENS,
    SEED,
    file_sha256,
    plan_groups,
    public_plan,
    read_jsonl,
    tokenize_groups,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-kind", choices=("qwen35", "deepseek"), required=True)
    p.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    p.add_argument("--sources", type=Path, required=True, help="directory with <dataset>.jsonl from prepare_sources")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--batches", default=",".join(map(str, BATCHES)))
    p.add_argument("--steps", type=int, default=DECODE_STEPS)
    p.add_argument("--max-prompt", type=int, default=MAX_PROMPT_TOKENS)
    p.add_argument(
        "--system-messages",
        choices=("merge", "keep"),
        default="merge",
        help="merge: fold all system messages into one leading message; keep: pass messages unchanged",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    import torch
    from transformers import AutoTokenizer

    from ..models import load_bf16_moe, moe_workload, read_config, use_vllm_experts

    torch.manual_seed(SEED)
    datasets, batches = a.datasets.split(","), [int(b) for b in a.batches.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    groups = [g for s in datasets for g in plan_groups(read_jsonl(a.sources / f"{s}.jsonl"), s, batches)]
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
        steps=a.steps,
        max_prompt=a.max_prompt,
        system_messages=a.system_messages,
        eos="finished requests stay in the batch; alive[step, request] is false once the request consumed EOS",
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
        start, b = time.monotonic(), group["batch"]
        ids, mask = left_padded(group["requests"], pad)
        if a.model_kind == "qwen35":
            model.model.rope_deltas = None  # multimodal RoPE offsets cached from the previous batch
        routes, scores = {li: [] for li in layers}, {li: [] for li in layers}
        alive, finished = [], np.zeros(b, dtype=bool)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            position = (mask.cumsum(-1) - 1).clamp_min(0)
            out = model(input_ids=ids, attention_mask=mask, position_ids=position, use_cache=True, logits_to_keep=1)
            for _ in range(a.steps):
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                finished |= np.isin(token[:, 0].cpu().numpy(), list(eos))
                alive.append(~finished)
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
                for li in layers:
                    routes[li].append(recorder.ids[li])
                    scores[li].append(recorder.scores[li])
        alive = np.stack(alive)
        prompt_lengths = np.array([r["prompt_tokens"] for r in group["requests"]])
        arrays = dict(layers=np.array(list(layers)), alive=alive, prompt_lengths=prompt_lengths)
        for li in layers:
            ri, sc = np.stack(routes[li]), np.stack(scores[li])
            assert ri.shape == (a.steps, b, workload["topk"])
            assert np.all((ri >= 0) & (ri < workload["experts"])) and np.all(np.diff(np.sort(ri, axis=-1), axis=-1) > 0)
            assert np.isfinite(sc).all() and np.all(sc >= 0)
            arrays[f"ids_l{li}"], arrays[f"scores_l{li}"] = ri, sc
        save_shard(
            path,
            manifest,
            group,
            arrays,
            live_counts=alive.sum(1).tolist(),
            seconds=time.monotonic() - start,
            peak_gpu_MiB=torch.cuda.max_memory_allocated() / 2**20,
        )
        del out, ids, mask
        gc.collect()
        torch.cuda.empty_cache()
    finish_manifest(path, manifest, groups)


if __name__ == "__main__":
    main()
