"""Engine check: vLLM's routed-experts capture (capture.py) against the transformers capture path (capture_routes.py with
vLLM's fused experts, left-padded group batches), on the same requests and prompt tokens (held-out batch-8 groups,
1,024-token cut), reference weights, greedy decoding and 32 decode steps. Compared while every request of a group is
alive: per (request, layer, step) whether the top-k expert sets are identical, their overlap, and the first step at
which a request's sets differ.

usage: python -m complementary_arrays.gpu.capture.verify {qwen35,deepseek} MODEL_DIR SOURCES_DIR TRACES_DIR OUT.json
       [--groups bfcl_heldout_b8_c0,gpqa_heldout_b8_c0,swe_heldout_b8_c0]
  TRACES_DIR holds <model>_b8_heldout.npz from convert_traces.py.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import requests as R
from .capture import patch_loader

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
TRACE_MODEL = {"qwen35": "qwen", "deepseek": "deepseek"}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=("qwen35", "deepseek"))
    p.add_argument("model", type=Path)
    p.add_argument("sources", type=Path)
    p.add_argument("traces", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--groups", default="bfcl_heldout_b8_c0,gpqa_heldout_b8_c0,swe_heldout_b8_c0")
    p.add_argument("--work", type=Path, help="directory for the linked model directory (default: next to OUT)")
    a = p.parse_args()
    import vllm
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from ..bench.formats import link_model_dir

    gids = a.groups.split(",")
    tok = AutoTokenizer.from_pretrained(a.model)
    groups = [
        g for s in R.DATASETS for g in R.plan_groups(R.read_jsonl(a.sources / f"{s}.jsonl"), s, (8,)) if g["id"] in gids
    ]
    groups = R.tokenize_groups(tok, groups, 1024, a.kind == "qwen35")
    with np.load(a.traces / f"{TRACE_MODEL[a.kind]}_b8_heldout.npz") as z:
        meta, routes, live = json.loads(str(z["metadata"])), z["routes"], z["live"]
    index = {(m["group"], m["layer"], m["step"]): i for i, m in enumerate(meta)}
    layers = sorted({m["layer"] for m in meta})
    patch_loader(a.kind, "ref", {})
    path = link_model_dir(a.model, (a.work or a.out.parent / "work") / f"{a.kind}_bf16", lambda cfg: None)
    gen = GenerationConfig.from_pretrained(a.model)
    eos = gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id]
    llm = LLM(
        model=path,
        max_model_len=1024 + 64,
        gpu_memory_utilization=0.85,
        max_num_seqs=64,
        enable_prefix_caching=True,
        seed=0,
        trust_remote_code=True,
        enable_return_routed_experts=True,
    )
    res = dict(kind=a.kind, vllm=vllm.__version__, groups=gids, steps=32, per_request=[])
    total = same = 0
    overlap_sum = 0.0
    by_layer, by_step = {}, {}
    for g in groups:
        reqs = g["requests"]
        sp = SamplingParams(max_tokens=33, temperature=0.0, stop_token_ids=eos)
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["input_ids"]) for r in reqs], sp, use_tqdm=False)
        for j, (r, o) in enumerate(zip(reqs, outs, strict=True)):
            c = o.outputs[0]
            got, p_len = np.asarray(c.routed_experts), len(r["input_ids"])
            n = n_same = 0
            first = None
            for s in range(1, min(33, len(c.token_ids))):
                for layer in layers:
                    k = (g["id"], layer, s)
                    if k not in index or live[index[k]] != len(reqs):
                        continue
                    x, y = np.sort(routes[index[k], j]), np.sort(got[p_len - 1 + s, layer])
                    ok = bool(np.array_equal(x, y))
                    ov = len(set(x.tolist()) & set(y.tolist())) / len(x)
                    n += 1
                    n_same += ok
                    overlap_sum += ov
                    by_layer.setdefault(int(layer), []).append((ok, ov))
                    by_step.setdefault(s, []).append((ok, ov))
                    if not ok and first is None:
                        first = s
            res["per_request"].append(
                dict(
                    group=g["id"],
                    j=j,
                    prompt=p_len,
                    generated=len(c.token_ids),
                    compared=n,
                    identical=n_same,
                    first_diff_step=first,
                )
            )
            total += n
            same += n_same
    res.update(compared=total, identical_share=same / max(total, 1), mean_overlap=overlap_sum / max(total, 1))
    mean = lambda v, i: float(np.mean([x[i] for x in v]))  # noqa: E731
    res["by_layer"] = {k: [mean(v, 0), mean(v, 1)] for k, v in sorted(by_layer.items())}
    res["by_step"] = {k: [mean(v, 0), mean(v, 1)] for k, v in sorted(by_step.items())}
    res["median_request_identical_share"] = float(
        np.median([x["identical"] / max(x["compared"], 1) for x in res["per_request"]])
    )
    res["requests_identical_all_steps"] = sum(x["first_diff_step"] is None for x in res["per_request"])
    a.out.write_text(json.dumps(res, indent=1))
    print(
        "VERIFY",
        a.kind,
        "identical share",
        round(res["identical_share"], 4),
        "mean overlap",
        round(res["mean_overlap"], 4),
        "requests without any difference",
        res["requests_identical_all_steps"],
        "of",
        len(res["per_request"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
