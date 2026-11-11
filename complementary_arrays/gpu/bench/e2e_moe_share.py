"""Decode-step time of DeepSeek-V2-Lite in vLLM, with and without its MoE layers, to measure their share by ablation.

Run once with --variant full and once with --variant no_moe, in which every MoE layer (router, shared experts and
routed experts) returns zeros; the generated tokens of that run are meaningless, but their count is fixed. A decode
step is (t(prompt + G tokens) - t(prompt + 1 token)) / (G - 1) for B random prompts, the median of --reps runs, in
vLLM's default execution mode (torch.compile and CUDA graphs).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True, help="DeepSeek-V2-Lite-Chat checkpoint directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=("full", "no_moe"), required=True)
    p.add_argument("--quantization", choices=("none", "fp8"), default="none", help="vLLM online weight quantization")
    p.add_argument("--batches", default="2,4,8,16,32,64,96,128")
    p.add_argument("--prompt", type=int, default=512, help="prompt tokens")
    p.add_argument("--gen", type=int, default=129, help="generated tokens G")
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # the engine must run in this process for the ablation patch
    import torch

    if a.variant == "no_moe":
        import vllm.model_executor.models.deepseek_v2 as deepseek_v2

        def zero_moe(self, hidden_states, *args, **kwargs):
            return torch.zeros_like(hidden_states)

        deepseek_v2.DeepseekV2MoE.forward = zero_moe
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    batches = [int(b) for b in a.batches.split(",")]
    llm = LLM(
        model=str(a.model),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=a.prompt + a.gen + 64,
        gpu_memory_utilization=0.85,
        max_num_seqs=max(batches),
        max_num_batched_tokens=max(batches) * a.prompt + 1024,
        enable_prefix_caching=False,
        quantization=None if a.quantization == "none" else a.quantization,
        seed=37,
    )
    rng = np.random.default_rng(37)
    for b in batches:
        seconds = {}
        for n in (1, a.gen):
            params = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0)
            times = []
            for rep in range(a.reps + 1):  # the first run warms up
                prompts = [
                    TokensPrompt(prompt_token_ids=rng.integers(100, 100000, a.prompt).tolist()) for _ in range(b)
                ]
                start = time.time()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                if rep:
                    times.append(time.time() - start)
                assert all(len(o.outputs[0].token_ids) == n for o in outputs)
            seconds[n] = float(np.median(times))
        step = (seconds[a.gen] - seconds[1]) / (a.gen - 1)
        record = dict(
            model=a.model.name,
            quantization=a.quantization,
            variant=a.variant,
            batch=b,
            prompt=a.prompt,
            gen=a.gen,
            t_1=seconds[1],
            t_gen=seconds[a.gen],
            decode_step_ms=step * 1e3,
        )
        print(json.dumps(record), flush=True)
        with (a.out / "e2e_ablation.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
