"""vLLM OpenAI-compatible server of one model with its expert weights in a GPU format (gpu/bench/formats.py), optionally
with an emulated KV-cache format, for the generation, function-calling and agent evaluations. Sampling defaults come
from the model's generation_config (clients may override).

usage: python -m complementary_arrays.gpu.accuracy.serve {qwen35,deepseek} MODE PORT --model DIR --work DIR
       [--max-model-len N] [--seed 37] [--kv-cache mxint8|mxint4[_k|_v|_c|_pe]] [-- extra vLLM server arguments]
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=("qwen35", "deepseek"))
    p.add_argument("mode")
    p.add_argument("port")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True, help="directory for the linked model directory")
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--seed", default="37")
    p.add_argument("--kv-cache", help="emulated KV-cache format (kv_cache.py)")
    a = p.parse_args(argv)
    from ..bench import formats as f

    f.set_env("vllm", a.mode)  # engine core in this process, so that the patches apply
    if a.mode == "nvfp4_w4a4_sh":
        f.register_nvfp4_dense()
    f.patch_vllm_loader(a.kind, a.mode, {})
    if a.kv_cache:
        from .kv_cache import register

        register(a.kv_cache)
    import uvloop
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.entrypoints.launchers.api_server.entry import run_server
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    model_dir = f.vllm_model_dir(a.model, a.kind, a.mode, a.work)
    max_len = a.max_model_len or (65536 if a.kind == "qwen35" else 32768)
    args = [
        "--model",
        model_dir,
        "--served-model-name",
        "m",
        model_dir,
        "--host",
        "127.0.0.1",
        "--port",
        a.port,
        "--max-model-len",
        str(max_len),
        "--gpu-memory-utilization",
        "0.88",
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        "16384",
        "--seed",
        a.seed,
        "--trust-remote-code",
        "--generation-config",
        "auto",
        "--disable-uvicorn-access-log",
    ]
    kw = f.vllm_engine_kwargs(a.kind, a.mode, {})
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", json.dumps(v) if isinstance(v, dict) else str(v)]
    args += extra
    print("SERVE", a.kind, a.mode, " ".join(args), flush=True)
    uvloop.run(run_server(make_arg_parser(FlexibleArgumentParser()).parse_args(args)))


if __name__ == "__main__":
    main()
