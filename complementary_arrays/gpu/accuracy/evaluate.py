"""Accuracy of Qwen3.5-35B-A3B, DeepSeek-V2-Lite-Chat or Qwen3-Next-80B-A3B under RTN expert-weight formats.

Qwen3.5 and DeepSeek: the routed experts, shared experts and shared-expert gate are replaced by their RTN
reconstruction in BF16, against the BF16 reference (the Qwen3.5 FP8 checkpoint dequantized to BF16). Qwen3-Next: only
the routed experts are quantized, against its FP8 checkpoint run natively. Results go to <out>/<model-kind>/<format>.json
with per-item correctness; each finished task is saved at once and skipped when the command is repeated.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path


from .tasks import TASKS, Scorer, run_task

FORMATS = {
    "qwen35": ("bf16", "w8g128", "w4g32"),
    "deepseek": ("bf16", "w8g128", "w4g32"),
    "qwen3next": ("fp8", "w8g128", "w4g32"),
}
SEED = 370711


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-kind", choices=tuple(FORMATS), required=True)
    p.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    p.add_argument("--data", type=Path, required=True, help="task data directory (see tasks.py)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--formats", help="comma-separated, default: the reference and both RTN formats")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--limit", type=int, default=0, help="items per task, 0 = all")
    p.add_argument("--gen-batch", type=int, default=64, help="GSM8K generation batch")
    p.add_argument("--max-new", type=int, default=1024, help="GSM8K generated-token limit")
    p.add_argument("--fp8-kernel", type=Path, help="Qwen3-Next: local snapshot of kernels-community/finegrained-fp8")
    a = p.parse_args()
    a.formats = a.formats.split(",") if a.formats else list(FORMATS[a.model_kind])
    unknown = set(a.formats) - set(FORMATS[a.model_kind])
    if unknown:
        p.error(f"formats {sorted(unknown)} are not defined for {a.model_kind}")
    return a


def load(a, fmt):
    """The model in the given format and its quantization error statistics (None for the reference)."""
    from ..models import fake_quantize_experts, load_bf16_moe, load_qwen3next, use_vllm_experts

    if a.model_kind == "qwen3next":
        model, _, stats = load_qwen3next(a.model, fmt, a.fp8_kernel)
        return model, stats
    model, layers = load_bf16_moe(a.model, a.model_kind)
    use_vllm_experts(layers)
    return model, fake_quantize_experts(model, fmt) if fmt != "bf16" else None


def save(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    a = parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    out = a.out / a.model_kind
    out.mkdir(parents=True, exist_ok=True)
    for fmt in a.formats:
        path = out / f"{fmt}.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        todo = [t for t in a.tasks.split(",") if t not in result.get("tasks", {})]
        if not todo:
            print("SKIP", a.model_kind, fmt, flush=True)
            continue
        start = time.monotonic()
        model, stats = load(a, fmt)
        result = dict(
            result,
            model=a.model_kind,
            format=fmt,
            config_sha256=hashlib.sha256((a.model / "config.json").read_bytes()).hexdigest(),
            quantization=stats,
            load_seconds=time.monotonic() - start,
            torch=torch.__version__,
            transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(),
            limit=a.limit,
            tasks=result.get("tasks", {}),
        )
        save(path, result)
        print("READY", a.model_kind, fmt, stats and round(stats["rel_l2"], 5), flush=True)
        scorer = Scorer(model, tok, model.model, model.lm_head)
        for task in todo:
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            r = run_task(task, scorer, tok, model, a)
            r.update(seconds=time.monotonic() - begin, peak_gpu_MiB=torch.cuda.max_memory_allocated() / 2**20)
            result["tasks"][task] = r
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
