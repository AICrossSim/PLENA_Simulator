"""Accuracy of Qwen3.5-35B-A3B or DeepSeek-V2-Lite-Chat with emulated expert-weight formats (transformers, one GPU).

The routed experts, shared experts and shared-expert gate are replaced by their reconstruction in BF16 (models.py),
against the BF16 reference (Qwen3.5: its FP8 checkpoint dequantized to BF16); activations stay BF16 and the experts run
through vLLM's fused-MoE kernel. Results go to <out>/<model-kind>/<format>.json with per-item correctness; each finished
task is saved at once and skipped when the command is repeated. GSM8K is generated greedily up to --max-new tokens.

usage: python -m complementary_arrays.gpu.accuracy.evaluate --model-kind {qwen35,deepseek} --model DIR --data DIR --out DIR
       [--formats bf16,w8g128,fp8b128,w4g32,nvfp4] [--tasks ...] [--max-new 1024]
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

from ..models import FORMATS
from .tasks import TASKS, Scorer, run_task

SEED = 370711


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-kind", choices=("qwen35", "deepseek"), required=True)
    p.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    p.add_argument("--data", type=Path, required=True, help="task data directory (see tasks.py)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--formats", default=",".join(("bf16", *FORMATS)))
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--limit", type=int, default=0, help="items per task, 0 = all")
    p.add_argument("--gen-batch", type=int, default=64, help="GSM8K generation batch")
    p.add_argument("--max-new", type=int, default=1024, help="GSM8K generated-token limit")
    a = p.parse_args()
    a.formats = a.formats.split(",")
    unknown = set(a.formats) - {"bf16", *FORMATS}
    if unknown:
        p.error(f"unknown formats {sorted(unknown)}")
    return a


def save(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    a = parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer

    from ..models import fake_quantize_experts, load_bf16_moe, use_vllm_experts

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    out = a.out / a.model_kind
    out.mkdir(parents=True, exist_ok=True)
    for fmt in a.formats:
        path = out / f"{fmt}.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        todo = [t for t in a.tasks.split(",") if t not in result.get("evals", {})]
        if not todo:
            print("SKIP", a.model_kind, fmt, flush=True)
            continue
        start = time.monotonic()
        model, layers = load_bf16_moe(a.model, a.model_kind)
        use_vllm_experts(layers)
        stats = fake_quantize_experts(model, fmt) if fmt != "bf16" else None
        result = dict(
            result,
            model=a.model_kind,
            format=fmt,
            config_sha256=hashlib.sha256((a.model / "config.json").read_bytes()).hexdigest(),
            quantization=stats,
            load_quantize_seconds=time.monotonic() - start,
            torch=torch.__version__,
            transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(),
            limit=a.limit,
            evals=result.get("evals", {}),
        )
        save(path, result)
        print("READY", a.model_kind, fmt, stats and round(stats["rel_l2"], 5), flush=True)
        scorer = Scorer(model, tok, model.model, model.lm_head)
        for task in todo:
            begin = time.monotonic()
            r = run_task(task, scorer, tok, model, a)
            r["seconds"] = time.monotonic() - begin
            result["evals"][task] = r
            save(path, result)
            summary = {
                k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if not isinstance(v, list | dict)
            }
            print("DONE", a.model_kind, fmt, task, summary, flush=True)
        del model, scorer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
