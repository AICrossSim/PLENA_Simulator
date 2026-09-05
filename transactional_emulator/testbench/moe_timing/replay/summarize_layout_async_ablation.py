#!/usr/bin/env python3
"""Combine paired row-major/tile-major MoE ablations into one audited report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_FILES = (
    "X.pt",
    "W_shared_gate_proj.pt",
    "W_shared_up.pt",
    "W_shared_down.pt",
    "golden_output.pt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant(document: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in document["variants"] if row["variant"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} variant, found {len(matches)}")
    return matches[0]


def load_pair(label: str, row_path: Path, tile_path: Path) -> dict[str, Any]:
    row = json.loads(row_path.read_text(encoding="utf-8"))
    tile = json.loads(tile_path.read_text(encoding="utf-8"))
    if row["weight_layout"] != "row_major" or tile["weight_layout"] != "tile_major":
        raise ValueError(f"{label}: summaries are not a row-major/tile-major pair")
    if row["repeats"] != 3 or tile["repeats"] != 3:
        raise ValueError(f"{label}: publication ablation requires exactly three repeats")
    if not all(row["gates"].values()) or not all(tile["gates"].values()):
        raise ValueError(f"{label}: one or more per-layout validation gates failed")
    if row["binary_sha256"] != tile["binary_sha256"]:
        raise ValueError(f"{label}: row and tile runs used different emulator binaries")

    row_build = Path(row["build_dir"])
    tile_build = Path(tile["build_dir"])
    source_hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        row_file = row_build / name
        tile_file = tile_build / name
        if not row_file.exists() or not tile_file.exists():
            raise FileNotFoundError(f"{label}: missing paired source file {name}")
        row_hash = sha256_file(row_file)
        tile_hash = sha256_file(tile_file)
        if row_hash != tile_hash:
            raise ValueError(f"{label}: row/tile source differs for {name}")
        source_hashes[name] = row_hash

    row_legacy = variant(row, "legacy_serial")
    row_coalesced = variant(row, "coalesced_serial")
    tile_legacy = variant(tile, "legacy_serial")
    tile_coalesced = variant(tile, "coalesced_serial")
    tile_serialized = variant(tile, "coalesced_scoreboard_serialized")
    tile_async = variant(tile, "coalesced_scoreboard")

    hashes = {
        entry["vram_sha256"]
        for document in (row, tile)
        for entry in document["variants"]
    }
    if len(hashes) != 1:
        raise ValueError(f"{label}: row/tile variants produced different VRAM outputs")
    if tile_serialized["total_cycles"] != tile_coalesced["total_cycles"]:
        raise ValueError(f"{label}: serialized scoreboard does not reproduce serial cycles")
    if tile_serialized["hbm_bytes_read"] != tile_coalesced["hbm_bytes_read"]:
        raise ValueError(f"{label}: serialized scoreboard changes HBM bytes")

    baseline_cycles = int(row_legacy["total_cycles"])
    baseline_bytes = int(row_legacy["hbm_bytes_read"])
    final_cycles = int(tile_async["total_cycles"])
    final_bytes = int(tile_async["hbm_bytes_read"])
    return {
        "case": label,
        "row_summary": str(row_path.resolve()),
        "tile_summary": str(tile_path.resolve()),
        "binary_sha256": row["binary_sha256"],
        "source_sha256": source_hashes,
        "output_vram_sha256": hashes.pop(),
        "functional_rel_rms": float(tile_async["functional_rel_rms"]),
        "functional_exact_match_pct": float(tile_async["functional_exact_match_rate"]),
        "baseline_cycles": baseline_cycles,
        "baseline_hbm_bytes": baseline_bytes,
        "row_coalesced_cycles": int(row_coalesced["total_cycles"]),
        "row_coalesced_hbm_bytes": int(row_coalesced["hbm_bytes_read"]),
        "tile_packing_only_cycles": int(tile_legacy["total_cycles"]),
        "tile_packing_only_hbm_bytes": int(tile_legacy["hbm_bytes_read"]),
        "tile_coalesced_serial_cycles": int(tile_coalesced["total_cycles"]),
        "tile_coalesced_serial_hbm_bytes": int(tile_coalesced["hbm_bytes_read"]),
        "tile_async_cycles": final_cycles,
        "tile_async_hbm_bytes": final_bytes,
        "packing_only_speedup": baseline_cycles / int(tile_legacy["total_cycles"]),
        "coalescing_after_packing_speedup": int(tile_legacy["total_cycles"])
        / int(tile_coalesced["total_cycles"]),
        "memory_package_speedup": baseline_cycles / int(tile_coalesced["total_cycles"]),
        "async_incremental_speedup": int(tile_coalesced["total_cycles"]) / final_cycles,
        "combined_speedup": baseline_cycles / final_cycles,
        "physical_hbm_byte_reduction_pct": 100.0
        * (baseline_bytes - final_bytes)
        / baseline_bytes,
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "case",
        "baseline_cycles",
        "baseline_hbm_bytes",
        "row_coalesced_cycles",
        "row_coalesced_hbm_bytes",
        "tile_packing_only_cycles",
        "tile_packing_only_hbm_bytes",
        "tile_coalesced_serial_cycles",
        "tile_coalesced_serial_hbm_bytes",
        "tile_async_cycles",
        "tile_async_hbm_bytes",
        "packing_only_speedup",
        "coalescing_after_packing_speedup",
        "memory_package_speedup",
        "async_incremental_speedup",
        "combined_speedup",
        "physical_hbm_byte_reduction_pct",
        "functional_rel_rms",
        "functional_exact_match_pct",
        "output_vram_sha256",
    ]
    with (output_dir / "layout_async_ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": "plena.moe_layout_async_ablation.v1",
        "cases": rows,
        "gates": {
            "three_repeats_per_variant": True,
            "same_binary_within_each_pair": True,
            "same_nonzero_source_tensors_within_each_pair": True,
            "same_vram_output_across_all_variants_within_each_pair": True,
            "scoreboard_serialization_control_passed": True,
        },
    }
    (output_dir / "layout_async_ablation_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    table = [
        "| 测试点 | 原路径周期 | 仅改布局 | 布局+Burst 合并 | 再加异步 | 内存包加速 | 异步额外加速 | 总加速 | HBM 物理字节减少 | rel-RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['case']} | {row['baseline_cycles']:,} | "
            f"{row['tile_packing_only_cycles']:,} | "
            f"{row['tile_coalesced_serial_cycles']:,} | {row['tile_async_cycles']:,} | "
            f"{row['memory_package_speedup']:.3f}x | "
            f"{row['async_incremental_speedup']:.3f}x | "
            f"{row['combined_speedup']:.3f}x | "
            f"{row['physical_hbm_byte_reduction_pct']:.2f}% | "
            f"{row['functional_rel_rms']:.3e} |"
        )

    report = """# PLENA Shared-MoE 权重布局与异步预取消融

## 实测结果

{table}

## 每一列真正代表什么

- **原路径**：row-major 权重、每个逻辑 scale slice 各发一个 64B 请求、opcode 串行等待。
- **仅改布局**：把每个 64x64 权重 tile 在 HBM 中连续存放；请求数不合并。
- **布局 + Burst 合并**：同一条物理 64B cache line 只发一次请求，再把其中的多个 MX scale slice 分发到目标位置。
- **再加异步**：使用依赖/资源 scoreboard，让无依赖的 DMA 与计算重叠；强制串行 scoreboard 已证明会恢复到上一列的周期数。

## 可以下的结论

1. 布局和 64B 请求合并有效，但当前实测不支持 `1.78x`。M=4、H=I=512 的总内存路径收益是 `1.127x`，再加异步后的总收益是 `1.237x`。
2. 收益随 M 增大明显下降。M=64 时总收益只有 `1.059x`，说明这组优化主要帮助小 M，而不是所有 Shared Expert 场景。
3. HBM 字节下降来自消除重复的 **物理 64B scale burst**，不是减少模型的逻辑权重参数。真正“少搬权重”仍需要同专家任务合并、驻留/复用，或经过精度验证的冷专家低比特方案。

## 验证门

- 每个状态重复 3 次，cycle、HBM bytes、VRAM SHA256 一致。
- 同一测试点的 row-major 与 tile-major 使用完全相同的非零 X/weight/golden tensor。
- 所有状态产生相同 VRAM SHA256，并通过非零 PyTorch golden 的 `rel_rms <= 0.01`。
- 实验开关默认关闭，因此原串行 ABI 和默认 timing 不变。

## 结论边界

这些是 Shared-FFN 微基准，不是完整 Transformer、真实 SWE trajectory 或 RTL sign-off。DeepSeek-style 两点使用 H=I=512；Qwen 点使用 H=I=64，作用是覆盖 sigmoid shared-gate 语义，单 tile 不能代表真实 Qwen 宽度。scoreboard 结果是 Rust simulation cycles，仍需 RTL primitive 校准后才能声称绝对硬件周期准确。
""".format(table="\n".join(table))
    (output_dir / "LAYOUT_ASYNC_ABLATION_REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "ROW_SUMMARY", "TILE_SUMMARY"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        load_pair(label, Path(row_path), Path(tile_path))
        for label, row_path, tile_path in args.case
    ]
    write_outputs(rows, args.output_dir.resolve())
    print(f"Wrote {args.output_dir.resolve() / 'LAYOUT_ASYNC_ABLATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
