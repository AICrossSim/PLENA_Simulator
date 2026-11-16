"""WikiText-2 perplexity in vLLM with a GPU expert-weight format and an emulated KV-cache format (the KV-cache study).

  windows  tokenize the test split into the disjoint 1,024-token windows of tasks.wikitext2 (transformers environment)
  score    sum of the prompt log-probabilities of every window in vLLM (vLLM environment); --eager with --stats
           records the relative quantization error of every cached part

usage: python -m complementary_arrays.gpu.accuracy.perplexity windows --model DIR --data DIR --out WINDOWS.json
       python -m complementary_arrays.gpu.accuracy.perplexity score {qwen35,deepseek} MODE WINDOWS.json OUT.json
              --model DIR --work DIR [--kv-cache mxint4[_k|_v|_c|_pe]] [--eager] [--stats]
"""

import argparse
import json
import math
import time
from pathlib import Path


def windows(a):
    from transformers import AutoTokenizer

    from .tasks import wikitext2_windows

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    w = wikitext2_windows(tok, a.data)
    a.out.write_text(json.dumps(w))
    print("windows", len(w), flush=True)


def score(a):
    from ..bench import formats as f

    f.set_env("vllm", a.mode)
    if a.mode == "nvfp4_w4a4_sh":
        f.register_nvfp4_dense()
    f.patch_vllm_loader(a.kind, a.mode, {})
    if a.kv_cache:
        from .kv_cache import register

        register(a.kv_cache, stats=a.stats)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=f.vllm_model_dir(a.model, a.kind, a.mode, a.work),
        max_model_len=8704,
        gpu_memory_utilization=0.85,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        seed=37,
        trust_remote_code=True,
        enforce_eager=a.eager,
    )
    llm = LLM(**f.vllm_engine_kwargs(a.kind, a.mode, kw))
    wins, t0 = json.loads(a.windows.read_text()), time.time()
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=w) for w in wins],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
        use_tqdm=False,
    )
    nll = -sum(o.prompt_logprobs[i][w[i]].logprob for o, w in zip(outs, wins, strict=True) for i in range(1, len(w)))
    tokens = sum(len(w) - 1 for w in wins)
    res = dict(
        kind=a.kind,
        mode=a.mode,
        kv_cache=a.kv_cache or "bf16",
        perplexity=math.exp(nll / tokens),
        nll=nll,
        tokens=tokens,
        windows=len(wins),
        seconds=time.time() - t0,
    )
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--model", type=Path, required=True)
    w.add_argument("--data", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("kind", choices=("qwen35", "deepseek"))
    s.add_argument("mode")
    s.add_argument("windows", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--kv-cache")
    s.add_argument("--eager", action="store_true")
    s.add_argument("--stats", action="store_true")
    a = p.parse_args()
    (windows if a.cmd == "windows" else score)(a)


if __name__ == "__main__":
    main()
