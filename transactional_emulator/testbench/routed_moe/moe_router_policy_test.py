# ruff: noqa: E402
"""V_TOPK routing at arbitrary ``(num_experts, top_k)``.

``gpt_oss_topk_test`` pins the two hardwired ``rmask`` policies, 32/top-4 and
128/top-8. Those cover GPT-OSS and Qwen3-30B-A3B and *nothing else* -- every
other production MoE shape falls outside the table:

    ==================  ============  ======  ==================
    model               num_experts   top_k   encoding
    ==================  ============  ======  ==================
    GPT-OSS                       32       4   rmask=0 (fixed)
    Qwen3-30B-A3B                128       8   rmask=1 (fixed)
    Llama-4 Scout                 16       1   C_SET_TOPK_REG
    Qwen2-MoE                     60       4   C_SET_TOPK_REG
    DeepSeek-V2-Lite              64       6   C_SET_TOPK_REG
    DeepSeek-V3 / Kimi K2        256       8   C_SET_TOPK_REG
    ==================  ============  ======  ==================

This exercises the ``C_SET_TOPK_REG`` escape end to end: compiler packs
``(num_experts << 8) | top_k``, assembler encodes the new opcode, emulator
unpacks it and routes. Nothing else covers that path, and a packing/unpacking
disagreement between the two repos would not be caught by either side's unit
tests -- each would keep agreeing with itself.

Each case asserts the encoding it actually took, so a policy silently falling
back to a fixed ``rmask`` (or to the wrong one) fails rather than passing on a
path it was not meant to test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Make the `compiler` package importable when this script is run directly.
# Prefer the pinned in-repo submodule over a sibling workspace checkout, which
# may be on a different branch and would silently shadow it on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _compiler_root in (_REPO_ROOT / "PLENA_Compiler", _REPO_ROOT.parent / "PLENA_Compiler"):
    if (_compiler_root / "aten" / "plena" / "compiler.py").exists():
        sys.path.insert(0, str(_compiler_root))
        break

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.emulator_runner import run_emulator
from transactional_emulator.testbench.gpt_oss_testkit import _decode_bf16_dump, _decode_u32_dump
from transactional_emulator.testbench.layout_utils import prestage_bf16_vram_matrix
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env

#: name -> (num_experts, top_k, expected encoding)
#:
#: "fixed" means the compiler must reuse a hardwired rmask so pre-existing
#: programs keep emitting byte-identical ASM; "control_register" means it must
#: take the C_SET_TOPK_REG escape.
POLICIES: dict[str, tuple[int, int, str]] = {
    "gpt_oss": (32, 4, "fixed"),
    "qwen3_moe": (128, 8, "fixed"),
    "llama4_scout": (16, 1, "control_register"),
    "qwen2_moe": (60, 4, "control_register"),
    "deepseek_v2_lite": (64, 6, "control_register"),
    "deepseek_v3": (256, 8, "control_register"),
}

#: BF16 softmax over a handful of terms; the emulator computes it in f32 and
#: rounds to BF16, so exact equality is not available. Same bound the legacy
#: top-k test uses.
WEIGHT_ATOL = 0.003


def _hot_experts(num_experts: int, top_k: int) -> list[tuple[int, float]]:
    """Deterministic logits with a known ranking and one exact tie inside top-k.

    Values descend by 0.5 from 8.0 and are spread across the whole expert range
    rather than clustered low, so a policy that reads too few experts (say, falls
    back to 128 when 256 were requested) drops a genuine winner instead of
    silently agreeing. The tie pins the low-index-wins contract at every width.
    """
    count = min(top_k + 2, num_experts)
    # Spread across [0, num_experts) with the last entry at the very top index.
    stride = max(1, (num_experts - 1) // max(1, count - 1)) if count > 1 else 1
    indices = [min(i * stride, num_experts - 1) for i in range(count)]
    indices[-1] = num_experts - 1
    indices = sorted(set(indices))
    while len(indices) < count:  # stride collisions at tiny widths
        candidate = max(indices) - 1
        if candidate < 0:
            break
        indices.append(candidate)
        indices = sorted(set(indices))

    entries = [(idx, 8.0 - 0.5 * rank) for rank, idx in enumerate(sorted(indices, reverse=True))]
    if top_k >= 2 and len(entries) >= 2:
        # Force an exact tie between the first two, so the winner is decided by
        # expert index rather than by logit value.
        entries[1] = (entries[1][0], entries[0][1])
    return entries


def run_router_policy(args: argparse.Namespace) -> dict:
    build_dir = args.build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)
    mlen = hw.mlen
    blen = hw.blen

    num_experts, top_k, expected_encoding = POLICIES[args.policy]
    expert_blocks = math.ceil(num_experts / mlen)

    logits = torch.full((expert_blocks, mlen), -100.0, dtype=torch.bfloat16)
    entries = _hot_experts(num_experts, top_k)
    for expert_idx, value in entries:
        logits[expert_idx // mlen, expert_idx % mlen] = torch.tensor(value, dtype=torch.bfloat16)

    ranked = sorted(entries, key=lambda item: (-item[1], item[0]))[:top_k]
    expected_indices = torch.tensor([idx for idx, _ in ranked], dtype=torch.int64)
    expected_weights = torch.softmax(torch.tensor([value for _, value in ranked], dtype=torch.float32), dim=0).to(
        torch.bfloat16
    )

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    vram_preload = torch.zeros(expert_blocks * mlen, dtype=torch.bfloat16)
    logits_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="RouterLogits",
        tensor=logits,
        vram_addr=0,
        physical_shape=(expert_blocks, mlen),
        vram_preload=vram_preload,
    )

    weights_fp_base = 32
    indices_int_base = 64
    prog.moe_router_select_v0(
        logits_vram,
        token_idx=0,
        weights_fp_base=weights_fp_base,
        indices_int_base=indices_int_base,
        num_experts=num_experts,
        top_k=top_k,
        policy_name=args.policy,
        name=f"router_policy_{args.policy}",
    )
    isa = prog.compile()

    # What the compiler actually emitted, not what it was asked for. Without this
    # a case could quietly take a fixed rmask and still produce correct routing,
    # leaving the escape path untested while appearing covered.
    emitted_control_register = "C_SET_TOPK_REG" in isa
    got_encoding = "control_register" if emitted_control_register else "fixed"

    create_sim_env(
        {},
        isa,
        {"original_output": torch.zeros(1, dtype=torch.bfloat16)},
        fp_preload=torch.zeros(128, dtype=torch.float16),
        int_preload=torch.zeros(128, dtype=torch.int32),
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts={},
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="moe_router_policy",
        data=None,
        specified_data_order=[],
        build_path=build_dir,
        input_tensors={},
        tensor_layouts={},
        hbm_addrs={},
    )
    hbm_path = build_dir / "hbm_for_behave_sim.bin"
    if not hbm_path.exists():
        hbm_path.write_bytes(bytes(64))

    metrics = run_emulator(build_dir, hbm_size=64, threads=args.emu_threads)
    got_weights = _decode_bf16_dump(build_dir / "fpsram_dump.bin")[weights_fp_base : weights_fp_base + top_k]
    got_indices = _decode_u32_dump(build_dir / "intsram_dump.bin")[indices_int_base : indices_int_base + top_k]

    indices_match = torch.equal(got_indices, expected_indices)
    weights_close = torch.allclose(got_weights.float(), expected_weights.float(), atol=WEIGHT_ATOL, rtol=0.0)
    encoding_match = got_encoding == expected_encoding

    summary = {
        "build_dir": str(build_dir),
        "policy": args.policy,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_blocks": expert_blocks,
        "expected_encoding": expected_encoding,
        "got_encoding": got_encoding,
        "packed_policy": (num_experts << 8) | top_k if expected_encoding == "control_register" else None,
        "expected_indices": expected_indices.tolist(),
        "got_indices": got_indices.tolist(),
        "expected_weights": [float(v) for v in expected_weights.float().tolist()],
        "got_weights": [float(v) for v in got_weights.float().tolist()],
        "indices_match": bool(indices_match),
        "weights_close": bool(weights_close),
        "encoding_match": encoding_match,
        "run_metrics": metrics,
        "passed": bool(indices_match and weights_close and encoding_match),
    }
    (build_dir / "router_policy_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if not encoding_match:
        raise AssertionError(
            f"policy {args.policy} ({num_experts}/top-{top_k}) took the {got_encoding!r} encoding, "
            f"expected {expected_encoding!r}. A shape outside the hardwired rmask table must go "
            "through C_SET_TOPK_REG; one inside it must not, or previously-working programs stop "
            "emitting byte-identical ASM."
        )
    if not summary["passed"]:
        raise AssertionError(f"V_TOPK routing check failed for {args.policy}: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="deepseek_v2_lite")
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--emu-threads", type=int, default=None)
    args = parser.parse_args()
    if args.build_dir is None:
        args.build_dir = Path(__file__).parent / "build" / f"moe_router_policy_{args.policy}"
    run_router_policy(args)


if __name__ == "__main__":
    main()
