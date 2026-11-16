"""Record the experts that every MoE layer selects for every token while vLLM generates the pool's requests (one GPU).

Per request: generation from its prompt token ids with the model's recommended sampling (Qwen3.5: temperature 1.0,
top-k 20, top-p 0.95; DeepSeek-V2-Lite: temperature 0.3, top-p 0.95) and the request's own seed, or greedy (--greedy:
the pool's greedy subset), at most 4,096 new tokens, stopping at the generation-config EOS ids. vLLM's routed-experts
capture returns the top-k expert ids of every MoE layer for every processed token; kept are the rows of the last prompt
token (step 0, it produces generated token 1) and of every generated token fed back (steps 1 .. G-1), G rows per
request; layers without routed experts (DeepSeek layer 0) are dropped. No text is written.

Weights (converted while vLLM loads them, no checkpoint is written):
  ref   DeepSeek BF16; Qwen3.5's official FP8 checkpoint dequantized to BF16 (vLLM would otherwise run it with FP8
        activations, a different model function)
  int4  ref with the routed experts, shared experts and the shared-expert gate replaced by their INT4 RTN
        reconstruction in BF16 (symmetric, FP16 scale per 32 weights along K; models.rtn)

Stored per chunk of requests: request hash, source, split, seed, prompt length, generated length, finish (1 = EOS,
0 = length limit), offsets and routes (uint8 [rows, MoE layers, top-k]). With --keep-steps 0 (default) every row and
the generated token ids are kept (the serving workload of gpu/bench needs them); with --keep-steps K only K rows per
request, drawn uniformly without replacement (numpy rng seeded by the request hash), and their step indices.

usage: python -m complementary_arrays.gpu.capture.capture {qwen35,deepseek} {ref,int4} POOL.pkl MODEL_DIR OUT
       [--greedy] [--ids-file F] [--keep-steps 0] [--max-new 4096] [--chunk 500] [--work DIR]
"""

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # the engine core runs in this process, so the loader patch applies
SAMPLING = {"qwen35": dict(temperature=1.0, top_k=20, top_p=0.95), "deepseek": dict(temperature=0.3, top_p=0.95)}
QUANTIZED = {
    "qwen35": r"^model\.language_model\.layers\.\d+\.mlp\.(experts\.\d+\.(gate|up|down)_proj|"
    r"shared_expert\.(gate|up|down)_proj|shared_expert_gate)\.weight$",
    "deepseek": r"^model\.layers\.\d+\.mlp\.(experts\.\d+|shared_experts)\.(gate|up|down)_proj\.weight$",
}
ROWS = "step 0 = last prompt token (produces generated token 1); step s = generated token s fed back, s = 1 .. G-1"


def patch_loader(kind, version, stats):
    """Dequantize FP8 block weights to BF16 and, for int4, replace the expert weights by their RTN reconstruction."""
    import torch
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from ..bench.formats import dequant_fp8_block, pair_fp8
    from ..models import rtn

    pattern, orig = re.compile(QUANTIZED[kind]), DefaultModelLoader.get_all_weights

    def convert(name, t, scale):
        if scale is not None:
            t = dequant_fp8_block(t.cuda(), scale.cuda())
        if version == "int4" and pattern.match(name):
            w = t.cuda().float()
            q, s = rtn(w, 4, 32)
            rec = (q * s.float()).reshape(w.shape[0], -1)[:, : w.shape[1]]
            stats["quantized"] = stats.get("quantized", 0) + 1
            stats["err"] = stats.get("err", 0.0) + float((rec - w).square().sum())
            stats["norm"] = stats.get("norm", 0.0) + float(w.square().sum())
            t = rec.to(torch.bfloat16)
        yield name, t

    def get_all_weights(self, model_config, model):
        yield from pair_fp8(orig(self, model_config, model), convert, stats)

    DefaultModelLoader.get_all_weights = get_all_weights


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=("qwen35", "deepseek"))
    p.add_argument("version", choices=("ref", "int4"))
    p.add_argument("pool", type=Path)
    p.add_argument("model", type=Path, help="checkpoint directory")
    p.add_argument("out", type=Path)
    p.add_argument("--greedy", action="store_true")
    p.add_argument(
        "--ids-file", type=Path, help="JSON list of request hashes (default: the pool, or its greedy subset)"
    )
    p.add_argument("--keep-steps", type=int, default=0)
    p.add_argument("--max-new", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--tag", default="")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-seqs", type=int, default=256)
    p.add_argument("--work", type=Path, help="directory for the linked model directory (default: OUT/work)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = pickle.loads(a.pool.read_bytes())["rows"]
    if a.ids_file:
        want = set(json.loads(a.ids_file.read_text()))
        rows = [r for r in rows if r["request"] in want]
    elif a.greedy:
        rows = [r for r in rows if r["greedy_subset"]]
    mode = "greedy" if a.greedy else "sample"
    tag = f"{a.kind}_{a.version}_{mode}{a.tag}"

    import torch
    import vllm
    from transformers import GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from ..bench.formats import link_model_dir

    stats = {}
    patch_loader(a.kind, a.version, stats)
    path = link_model_dir(a.model, (a.work or a.out / "work") / f"{a.kind}_bf16", lambda cfg: None)
    gen = GenerationConfig.from_pretrained(a.model)
    eos = gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id]
    t0 = time.time()
    # the routed-experts capture requires prefix caching; decode rows are always computed
    llm = LLM(
        model=path,
        max_model_len=max(r["prompt_tokens"] for r in rows) + a.max_new + 16,
        gpu_memory_utilization=a.gpu_mem,
        max_num_seqs=a.max_seqs,
        max_num_batched_tokens=16384,
        enable_prefix_caching=True,
        seed=0,
        trust_remote_code=True,
        enable_return_routed_experts=True,
    )
    meta = dict(
        kind=a.kind,
        version=a.version,
        mode=mode,
        vllm=vllm.__version__,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        load_s=time.time() - t0,
        load_stats=stats,
        eos=eos,
        sampling="greedy" if a.greedy else dict(SAMPLING[a.kind], seed="per request"),
        max_new_tokens=a.max_new,
        generation_config={k: getattr(gen, k, None) for k in ("temperature", "top_k", "top_p", "do_sample")},
        rows_definition=ROWS,
    )
    if a.keep_steps:
        meta["stored_steps_per_request"] = a.keep_steps
    if stats.get("norm"):
        meta["int4_rel_l2"] = (stats["err"] / stats["norm"]) ** 0.5
    print("META", json.dumps(meta), flush=True)
    moe_layers = None
    done = {f.name for f in a.out.glob(f"{tag}_chunk*.npz")}
    for c0 in range(0, len(rows), a.chunk):
        name = f"{tag}_chunk{c0 // a.chunk:02d}.npz"
        if name in done:
            continue
        part, t1 = rows[c0 : c0 + a.chunk], time.time()
        sps = [
            SamplingParams(max_tokens=a.max_new, stop_token_ids=eos, temperature=0.0)
            if a.greedy
            else SamplingParams(max_tokens=a.max_new, stop_token_ids=eos, seed=int(r["seed"]), **SAMPLING[a.kind])
            for r in part
        ]
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["input_ids"]) for r in part], sps, use_tqdm=False)
        P, G, F, toks, routes, shapes, steps, kept = [], [], [], [], [], [], [], []
        for r, o in zip(part, outs, strict=True):
            c = o.outputs[0]
            ids, p_len = list(c.token_ids), len(r["input_ids"])
            g = len(ids)
            re_ = np.asarray(c.routed_experts)
            shapes.append(list(re_.shape))
            if moe_layers is None:
                nonzero = [layer for layer in range(re_.shape[1]) if np.any(re_[:p_len, layer] != 0)]
                moe_layers = nonzero if a.kind == "deepseek" else list(range(re_.shape[1]))
                meta["moe_layers"] = moe_layers
            assert re_.shape[0] >= p_len + g - 1, (re_.shape, p_len, g)
            rows_g = re_[p_len - 1 : p_len - 1 + g][:, moe_layers].astype(np.uint8)
            if a.keep_steps:
                rng = np.random.default_rng(int(r["request"][:15], 16))
                k = min(a.keep_steps, g)
                st = np.sort(rng.choice(g, size=k, replace=False)) if g > 0 else np.zeros(0, np.int64)
                rows_g = rows_g[st]
                steps.append(st.astype(np.int32))
            else:
                toks.append(np.asarray(ids, np.int32))
            routes.append(rows_g)
            kept.append(len(rows_g))
            P.append(p_len)
            G.append(g)
            F.append(int(c.finish_reason == "stop"))
        arrays = dict(
            request=np.array([r["request"] for r in part]),
            source=np.array([r["source"] for r in part]),
            split=np.array([r["split"] for r in part]),
            seed=np.array([r["seed"] for r in part], np.int64),
            prompt_tokens=np.array(P, np.int32),
            gen_tokens=np.array(G, np.int32),
            finish_eos=np.array(F, np.uint8),
            offsets=np.concatenate([[0], np.cumsum(kept)]).astype(np.int64),
            routes=np.concatenate(routes),
            routed_shape=np.array(shapes, np.int64),
            moe_layers=np.array(moe_layers, np.int16),
            meta=json.dumps(meta),
        )
        if a.keep_steps:
            arrays["steps"] = np.concatenate(steps)
        else:
            arrays["out_tokens"] = np.concatenate(toks)
        tmp = a.out / (name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, a.out / name)
        print(
            "CHUNK",
            name,
            len(part),
            "requests",
            int(sum(G)),
            "tokens",
            f"{time.time() - t1:.0f}s",
            "eos",
            sum(F),
            "routed rows - (p+g)",
            sorted({s[0] - (p + g) for s, p, g in zip(shapes, P, G, strict=True)}),
            flush=True,
        )
    print("ALL_DONE", tag, flush=True)


if __name__ == "__main__":
    main()
