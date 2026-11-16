"""Serving engines of the B200 measurements: one vLLM or SGLang engine on one GPU with streaming generation.

The weight-format and ablation patches must also run in SGLang's scheduler process, which re-imports the main module
of the run; a run therefore stores its setup in the MOE_BENCH environment variable and every entry point calls
setup_from_env() at import time.
"""

import json
import os

ENV = "MOE_BENCH"
MAX_LEN = 4096 + 4096 + 64
_applied = {}


def configure(stack, kind, mode, model, work, moe="full", no_shared=False, profile=False):
    """Store the run setup for this process and its children and apply it here."""
    os.environ[ENV] = json.dumps(
        dict(
            stack=stack,
            kind=kind,
            mode=mode,
            model=str(model),
            work=str(work),
            moe=moe,
            no_shared=no_shared,
            profile=profile,
        )
    )
    return setup_from_env()


def setup_from_env():
    """Apply the stored setup once per process: environment switches, weight-loader patch, MoE or shared-expert zeroing,
    profiler ranges."""
    if ENV not in os.environ:
        return None
    s = json.loads(os.environ[ENV])
    if _applied:
        assert _applied["setup"] == s, "setup changed within one process"
        return _applied["setup"]
    from . import formats as f

    f.set_env(s["stack"], s["mode"])
    stats = {}
    if s["stack"] == "vllm":
        if s["mode"] == "nvfp4_w4a4_sh":
            f.register_nvfp4_dense()
        f.patch_vllm_loader(s["kind"], s["mode"], stats)
    else:
        f.patch_sglang_loader(s["kind"], s["mode"])
    if s["moe"] == "no_moe":
        f.zero_moe(s["stack"], s["kind"])
    if s["no_shared"]:
        f.zero_shared(s["stack"], s["kind"])
    if s.get("profile"):
        f.annotate_moe(s["stack"], s["kind"])
    _applied.update(setup=s, load_stats=stats)
    return s


def tag(s, suffix=""):
    return f"{s['stack']}_{s['kind']}_{s['mode']}_{s['moe']}" + ("_noshared" if s["no_shared"] else "") + suffix


def engine(s, max_batch, max_len=MAX_LEN):
    """(stream, abort_all, shutdown, meta). stream(ids, n) is an async generator of new-token counts for a request that
    produces exactly n tokens (greedy, EOS ignored); abort_all() cancels every running request."""
    from . import formats as f

    if s["stack"] == "sglang":
        import sglang as sgl

        kw = dict(
            model_path=f.sglang_model_dir(s["model"], s["kind"], s["mode"], s["work"]),
            tp_size=1,
            mem_fraction_static=0.80,
            disable_radix_cache=True,
            max_running_requests=max_batch,
            context_length=max_len,
            chunked_prefill_size=16384,
            cuda_graph_max_bs_decode=max_batch,
            random_seed=37,
            trust_remote_code=True,
            log_level="warning",
        )
        f.sglang_engine_kwargs(s["kind"], s["mode"], kw)
        llm = sgl.Engine(**kw)

        async def stream(ids, n):
            sp = dict(max_new_tokens=n, min_new_tokens=n, ignore_eos=True, temperature=0.0)
            prev = 0
            async for chunk in await llm.async_generate(input_ids=ids, sampling_params=sp, stream=True):
                done = chunk["meta_info"]["completion_tokens"]
                if done > prev:
                    yield done - prev
                prev = done

        async def abort_all():
            llm.tokenizer_manager.abort_request(abort_all=True)

        meta = dict(stack="sglang", version=sgl.__version__, kwargs={k: str(v) for k, v in kw.items()})
        return stream, abort_all, llm, meta

    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    kw = dict(
        model=f.vllm_model_dir(s["model"], s["kind"], s["mode"], s["work"]),
        max_model_len=max_len,
        gpu_memory_utilization=0.85,
        max_num_seqs=max_batch,
        max_num_batched_tokens=16384,
        enable_prefix_caching=False,
        seed=37,
        trust_remote_code=True,
    )
    f.vllm_engine_kwargs(s["kind"], s["mode"], kw)
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
    live, counter = set(), [0]

    async def stream(ids, n):
        counter[0] += 1
        rid = f"q{counter[0]}"
        live.add(rid)
        sp = SamplingParams(
            max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0, output_kind=RequestOutputKind.DELTA
        )
        try:
            async for o in llm.generate(TokensPrompt(prompt_token_ids=ids), sp, rid):
                c = len(o.outputs[0].token_ids)
                if c > 0:
                    yield c
        finally:
            live.discard(rid)

    async def abort_all():
        if live:
            await llm.abort(list(live))

    meta = dict(stack="vllm", version=vllm.__version__, kwargs={k: str(v) for k, v in kw.items()})
    meta["load_stats"] = _applied.get("load_stats")
    return stream, abort_all, llm, meta


def run_batches(s, llm, measure_all):
    """Run the measurement coroutine on the loop the engine expects (SGLang: the loop it was created on)."""
    import asyncio

    if s["stack"] == "sglang":
        asyncio.get_event_loop().run_until_complete(measure_all())
    else:
        asyncio.run(measure_all())


def stop(llm):
    """Stop the engine and exit without waiting for engine threads."""
    import time

    try:
        llm.shutdown()
    except Exception as e:
        print("shutdown error", repr(e), flush=True)
    time.sleep(3)
    os._exit(0)
