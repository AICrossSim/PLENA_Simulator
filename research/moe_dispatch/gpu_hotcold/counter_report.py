"""Summarize separate Nsight Compute and Kineto characterization runs.

This module never substitutes replayed kernel durations for the benchmark's
unprofiled CUDA-event E2E timings, and never invents a binary bottleneck label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

BATCHES = (2, 4, 8, 16, 64, 96, 128)
PATHS = ("vllm_grouped", "vllm_split")
METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
    "dram_util_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_pipe_util_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm_util_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}
STAGES = ("shared", "routed", "hot", "cold", "combine")
TIME_TO_NS = {"ns": 1.0, "us": 1000.0, "µs": 1000.0, "ms": 1e6, "s": 1e9}
BYTES = {"byte": 1.0, "bytes": 1.0, "B": 1.0, "Kbyte": 1000.0,
         "Mbyte": 1e6, "Gbyte": 1e9, "KiB": 1024.0, "MiB": 1024.0 ** 2}
UTIL = ("dram_util_pct", "tensor_pipe_util_pct", "sm_util_pct", "occupancy_pct")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(text, *, metric):
    try:
        value = float(str(text).replace(",", "").strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Missing or invalid required counter {metric}: {text!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Nonfinite/negative required counter {metric}: {text!r}")
    return value


def parse_ncu(path):
    """Read NCU --page raw wide CSV; require the explicit units row."""
    path = Path(path)
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        raise ValueError(f"No NCU raw wide CSV header: {path}")
    records = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    if len(records) < 3:
        raise ValueError(f"Missing units or kernel rows in {path}")
    columns = records[0]
    units = dict(zip(columns, records[1]))
    if len(records[1]) != len(columns) or records[1][0] != "":
        raise ValueError(f"Invalid NCU units row in {path}")
    missing = [col for col in METRICS.values() if col not in columns]
    if missing:
        raise ValueError(f"Required counters absent from {path}: {missing}")
    nvtx = next((col for col in columns if col.startswith("thread Domain:")), None)
    if nvtx is None:
        raise ValueError("NVTX thread domain column is required for exact stage attribution")
    if "Kernel Name" not in columns:
        raise ValueError("Missing kernel name")
    factors = {}
    for key, metric in METRICS.items():
        unit = units[metric].strip()
        if key == "duration_ns":
            if unit not in TIME_TO_NS:
                raise ValueError(f"Unsupported duration unit {unit!r}")
            factors[key] = TIME_TO_NS[unit]
        elif key.endswith("bytes"):
            if unit not in BYTES:
                raise ValueError(f"Unsupported byte unit {unit!r}")
            factors[key] = BYTES[unit]
        else:
            if unit != "%":
                raise ValueError(f"Counter {metric} must carry percent units, got {unit!r}")
            factors[key] = 1.0
    result = []
    for raw in records[2:]:
        if not raw or raw[0].startswith("=="):
            continue
        if len(raw) != len(columns):
            raise ValueError(f"Malformed CSV kernel row in {path}: {raw[:2]}")
        row = dict(zip(columns, raw))
        if not row["ID"].isdigit():
            raise ValueError(f"Invalid kernel ID in {path}: {row['ID']!r}")
        stages = re.findall(r"<default domain>:(shared|routed|hot|cold|combine):", row[nvtx])
        if not stages:
            raise ValueError(f"Kernel {row['ID']} has no recognized NVTX stage")
        parsed = {"kernel_id": int(row["ID"]), "kernel_name": row["Kernel Name"],
                  "stage": stages[-1]}
        for key, metric in METRICS.items():
            parsed[key] = number(row[metric], metric=metric) * factors[key]
        if parsed["duration_ns"] <= 0:
            raise ValueError("Kernel duration must be positive")
        result.append(parsed)
    if not result:
        raise ValueError(f"No measured kernels in {path}")
    ids = [r["kernel_id"] for r in result]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate kernel IDs in {path}")
    return result


def aggregate_kernels(kernels, batch, path, window_id=None):
    groups = defaultdict(list)
    for row in kernels:
        groups[row["stage"]].append(row)
        groups["all_kernel_sum"].append(row)
    result = []
    for stage in (*STAGES, "all_kernel_sum"):
        rows = groups.get(stage)
        if not rows:
            continue
        duration = sum(r["duration_ns"] for r in rows)
        read = sum(r["dram_read_bytes"] for r in rows)
        write = sum(r["dram_write_bytes"] for r in rows)
        row = {"batch": batch, "path": path, "window_id": window_id or "not_recorded",
               "stage": stage, "kernel_count": len(rows),
               "serialized_kernel_sum_us": duration / 1000,
               "dram_read_MiB": read / 1024 ** 2, "dram_write_MiB": write / 1024 ** 2,
               "dram_read_write_GBps": (read + write) / duration}
        for key in UTIL:
            row[key] = sum(r[key] * r["duration_ns"] for r in rows) / duration
        row["interpretation"] = "descriptive sampled replay counters; no binary bottleneck verdict"
        result.append(row)
    return result



def routed_gemm_rows(kernels, batch, path, window_id=None):
    """For this unquantized vLLM implementation GU then Down are two launches."""
    result = []
    for stage in ("routed", "hot", "cold"):
        relevant = [r for r in kernels if r["stage"] == stage]
        if not relevant:
            continue
        gemms = [r for r in relevant if r["kernel_name"].startswith("fused_moe_kernel")]
        if len(gemms) != 2:
            raise ValueError(f"Expected two unquantized vLLM expert GEMMs in {stage}; got {len(gemms)}")
        gemms.sort(key=lambda r: r["kernel_id"])
        for phase, kernel in zip(("Gate_Up", "Down"), gemms):
            duration = kernel["duration_ns"]
            row = {"batch": batch, "path": path, "window_id": window_id or "not_recorded",
                   "stage": stage, "phase": phase, "kernel_id": kernel["kernel_id"],
                   "kernel_name": kernel["kernel_name"], "replay_duration_us": duration / 1000,
                   "dram_read_MiB": kernel["dram_read_bytes"] / 1024 ** 2,
                   "dram_write_MiB": kernel["dram_write_bytes"] / 1024 ** 2,
                   "dram_read_write_GBps": (kernel["dram_read_bytes"] + kernel["dram_write_bytes"]) / duration}
            row.update({key: kernel[key] for key in UTIL})
            result.append(row)
    return result


def parse_kineto(path, iterations=3):
    """Keep runtime and driver launch APIs separate to avoid double counting."""
    data = json.loads(Path(path).read_text())
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    events = data.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise ValueError(f"No Kineto traceEvents in {path}")
    grouped = defaultdict(list)
    api_names = defaultdict(set)
    for event in events:
        category = event.get("cat", "")
        if category not in ("cuda_runtime", "cuda_driver") or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        if "launchkernel" not in name.lower() and "graphlaunch" not in name.lower():
            continue
        value = number(event.get("dur"), metric=f"Kineto duration {name}")
        grouped[category].append(value)
        api_names[category].add(name)
    if not grouped:
        raise ValueError(f"No CUDA launch API events in {path}")
    result = []
    for category in ("cuda_runtime", "cuda_driver"):
        if category not in grouped:
            continue
        values = grouped[category]
        result.append({"api_category": category, "launch_calls_total": len(values),
                       "launch_calls_per_layer": len(values) / iterations,
                       "launch_api_sum_us_per_layer": sum(values) / iterations,
                       "launch_api_mean_us": statistics.fmean(values),
                       "launch_api_median_us": statistics.median(values),
                       "iterations": iterations, "api_names": ";".join(sorted(api_names[category])),
                       "preferred_nonduplicated_category": category ==
                       ("cuda_runtime" if "cuda_runtime" in grouped else "cuda_driver"),
                       "interpretation": "host launch API duration; not GPU gap or E2E overhead"})
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(input_dir, out, allow_partial=False):
    input_dir, out = Path(input_dir), Path(out)
    ncu_dir = input_dir / "ncu"
    files = list(ncu_dir.glob("B*_vllm_*.csv")) if ncu_dir.exists() else []
    if not files:
        files = list(input_dir.glob("B*_vllm_*.csv"))
    counters, sources, coverage, kernel_rows, gemm_rows = [], {}, set(), [], []
    representatives = {}
    for file in sorted(files):
        match = re.fullmatch(r"B(\d+)_(vllm_grouped|vllm_split)\.csv", file.name)
        if not match:
            continue
        batch, path = int(match[1]), match[2]
        if (batch, path) in coverage:
            raise ValueError(f"Duplicate counter sample {(batch, path)}")
        coverage.add((batch, path))
        metadata_file = file.with_suffix(".json")
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        window_id = metadata.get("window_id")
        if not window_id and not allow_partial:
            raise ValueError(f"Counter representative-window provenance missing: {metadata_file}")
        if metadata:
            if metadata.get("batch") != batch or metadata.get("path") != path:
                raise ValueError(f"Counter provenance mismatch in {metadata_file}")
            representatives[f"B{batch}/{path}"] = metadata
            sources[str(metadata_file)] = digest(metadata_file)
        kernels = parse_ncu(file)
        if metadata and set(metadata.get("collected_stages", [])) != set(r["stage"] for r in kernels):
            raise ValueError(f"NVTX stages disagree with counter collection metadata: {metadata_file}")
        counters.extend(aggregate_kernels(kernels, batch, path, window_id))
        gemm_rows.extend(routed_gemm_rows(kernels, batch, path, window_id))
        kernel_rows.extend({"batch": batch, "path": path, "window_id": window_id or "not_recorded", **r}
                           for r in kernels)
        sources[str(file)] = digest(file)
    for batch in BATCHES:
        pair = [representatives.get(f"B{batch}/{p}", {}).get("window_id") for p in PATHS]
        if all(pair) and pair[0] != pair[1]:
            raise ValueError(f"Grouped/split counter runs use different representative windows at B{batch}")
    expected = {(b, p) for b in BATCHES for p in PATHS}
    missing = sorted(expected - coverage)
    if (missing or not coverage) and not allow_partial:
        raise ValueError(f"Counter campaign incomplete; missing {missing}")
    launches, kineto_coverage = [], set()
    kineto_dir = input_dir / "kineto"
    for file in sorted(kineto_dir.glob("B*_vllm_grouped_kineto.json")):
        match = re.fullmatch(r"B(\d+)_vllm_grouped_kineto\.json", file.name)
        if not match:
            continue
        batch = int(match[1])
        summary_file = file.with_suffix(".summary.json")
        iterations = 3
        if summary_file.exists():
            summary_data = json.loads(summary_file.read_text())
            iterations = int(summary_data["iterations"])
            sources[str(summary_file)] = digest(summary_file)
        for row in parse_kineto(file, iterations):
            launches.append({"batch": batch, "path": "vllm_grouped", **row})
        kineto_coverage.add(batch)
        sources[str(file)] = digest(file)
    kineto_missing = sorted(set(BATCHES) - kineto_coverage)
    if kineto_missing and not allow_partial:
        raise ValueError(f"Kineto campaign incomplete; missing batches {kineto_missing}")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "counter_summary.csv", counters)
    write_csv(out / "counter_kernels.csv", kernel_rows)
    write_csv(out / "routed_gemm_summary.csv", gemm_rows)
    write_csv(out / "kineto_launch_summary.csv", launches)
    lines = ["# GPU 硬件计数器与调用开销补测", "",
             "这些是独立于 CUDA Event 主测试的表征实验：每个 Batch 选取一个真实留出窗口，"
             "分别采集普通 vLLM grouped 与冷热拆分 vLLM split。Nsight Compute 采用串行 kernel replay "
             "及默认 cache-control=all（每次 replay 清缓存）；clock-control=none。下列 kernel 时间之和不是正常执行的整层延迟。", "",
             "利用率按各 kernel 的 replay duration 加权。不同计数器分母不同：DRAM/Tensor/SM "
             "使用 elapsed-cycle 比例，Occupancy 使用 active-cycle 比例；加权值仅作为描述性汇总，"
             "不是从整层时间线直接测得的占空比。冷热拆分还会重复分派、补齐和合并开销。", "",
             "| Batch | 路径 | 阶段 | Kernel 数 | 串行 kernel 和（μs） | DRAM 读＋写（GB/s） | DRAM 利用率 | Tensor pipe | SM 利用率 | Occupancy |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(counters, key=lambda r: (r["batch"], r["path"],
                                              (*STAGES, "all_kernel_sum").index(r["stage"]))):
        lines.append(f"| B{row['batch']} | {row['path']} | {row['stage']} | {row['kernel_count']} | "
                     f"{row['serialized_kernel_sum_us']:.3f} | {row['dram_read_write_GBps']:.2f} | "
                     f"{row['dram_util_pct']:.2f}% | {row['tensor_pipe_util_pct']:.2f}% | "
                     f"{row['sm_util_pct']:.2f}% | {row['occupancy_pct']:.2f}% |")
    lines += ["", "## 普通 grouped 路径的专家 GEMM", "",
              "同一次代表窗口中，vLLM 的两次专家矩阵计算按执行顺序分别为 Gate/Up 和 Down。"
              "下表直接列出该 kernel 的计数器，没有与分派、激活或合并平均。", "",
              "| Batch | GEMM | Replay 时间（μs） | DRAM 读＋写（GB/s） | DRAM 利用率 | Tensor pipe | SM 利用率 | Occupancy |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted((r for r in gemm_rows if r["path"] == "vllm_grouped"),
                      key=lambda r: (r["batch"], r["kernel_id"])):
        lines.append(f"| B{row['batch']} | {row['phase']} | {row['replay_duration_us']:.3f} | "
                     f"{row['dram_read_write_GBps']:.2f} | {row['dram_util_pct']:.2f}% | "
                     f"{row['tensor_pipe_util_pct']:.2f}% | {row['sm_util_pct']:.2f}% | "
                     f"{row['occupancy_pct']:.2f}% |")
    lines += ["", "未出现的类别表示该次采样没有相应 kernel，不能据此填写缺失硬件计数器为零。"
              "Tensor pipe 指定指标也不等同于有效 BF16 FLOP/s。", "",
              "## 单独采集的 CPU Launch API 开销", "",
              "每个窗口采集三次完整 eager 执行。下表统计 cuda_runtime 或 cuda_driver "
              "launch API 的主机端调用耗时，**不能当作 GPU 空闲间隙或加到 CUDA Event 延迟上**。"
              "runtime 和 driver 包装层可能重叠，两类不相加。", "",
              "| Batch | API 类别 | 每层调用数 | 每层 API 时间和（μs） | 单次调用均值（μs） | 单次调用中位数（μs） |",
              "|---|---|---:|---:|---:|---:|"]
    for row in sorted(launches, key=lambda r: (r["batch"], r["api_category"])):
        lines.append(f"| B{row['batch']} | {row['api_category']} | {row['launch_calls_per_layer']:.1f} | "
                     f"{row['launch_api_sum_us_per_layer']:.3f} | {row['launch_api_mean_us']:.3f} | "
                     f"{row['launch_api_median_us']:.3f} |")
    lines += ["", "## 可支持的结论", "",
              "低 Tensor 利用率加上中等 DRAM 利用率，说明该采样 kernel 没有打满这两项指标；"
              "它不能单独证明整层是纯 Memory-bound 或纯 Compute-bound。小矩阵并行度、补齐、"
              "调用与同步也可能限制性能。判断瓶颈翻转还需要正常执行时间线和带宽／计算敏感性实验，"
              "本报告不凭计数器自动宣布翻转 Batch。", "",
              "原始逐 kernel 数据保存在 `counter_kernels.csv`；阶段汇总为 `counter_summary.csv`；"
              "API 统计为 `kineto_launch_summary.csv`。所有输入 SHA256 和覆盖范围见 `COUNTER_AUDIT.json`。", ""]
    (out / "GPU_COUNTERS.md").write_text("\n".join(lines))
    audit = {"status": "PASS" if not missing and not kineto_missing else "PARTIAL",
             "ncu_points": len(coverage), "ncu_kernel_rows": len(kernel_rows),
             "ncu_expected_points": 14, "ncu_missing": missing,
             "kineto_batches": sorted(kineto_coverage), "kineto_missing": kineto_missing,
             "all_required_counters_numeric_and_units_checked": True,
             "duration_weighted_not_e2e_utilization": True,
             "ncu_cache_control": "all (default; replay cache flushing)",
             "ncu_clock_control": "none",
             "routed_gemm_rows": len(gemm_rows),
             "no_automatic_binary_bottleneck_verdict": True,
             "input_sha256": sources, "representative_windows": representatives}
    (out / "COUNTER_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    audit = report(args.input, args.out, args.allow_partial)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("input_sha256", "representative_windows")}))


if __name__ == "__main__":
    main()
