#!/usr/bin/env python3
"""Generate small BFCL/GPQA true-router traces with top-k weights for P2.

This is intentionally a pilot helper, not a full benchmark runner.  It runs the
local HF model on CPU for a small selected subset and writes router top-k expert
ids plus softmax route weights, which P1's route trace schema requires.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transactional_emulator.testbench.moe_timing.qwen.utils import (
    INPUT_FILES,
    MODEL_CONFIGS,
    OUT_ROOT,
    append_jsonl,
    ensure_paths,
    iter_jsonl,
)


def _input_records(path: Path, model_key: str) -> list[dict[str, Any]]:
    return [row for row in iter_jsonl(path) if row.get("model_key") == model_key]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _route_key(row: dict[str, Any]) -> tuple[str, str, int, int | None]:
    decode_step = row.get("decode_step")
    return (
        str(row["sample_id"]),
        str(row["phase"]),
        int(row["layer"]),
        int(decode_step) if decode_step is not None else None,
    )


def _existing_route_keys(path: Path) -> set[tuple[str, str, int, int | None]]:
    if not path.exists():
        return set()
    keys = set()
    for row in iter_jsonl(path):
        try:
            keys.add(_route_key(row))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def _existing_route_keys_from_paths(paths: list[Path]) -> set[tuple[str, str, int, int | None]]:
    keys: set[tuple[str, str, int, int | None]] = set()
    for path in paths:
        keys.update(_existing_route_keys(path))
    return keys


def _resume_scan_paths(
    *,
    output_path: Path,
    resume_from: list[Path],
    benchmark: str,
    model_key: str,
) -> list[Path]:
    paths = [output_path, *resume_from]
    prefix = f"{benchmark}_{model_key}"
    if output_path.parent.exists():
        paths.extend(path for path in output_path.parent.glob("*.jsonl") if path.name.startswith(prefix))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _expected_route_keys(
    *,
    sample_id: str,
    layers: list[int],
    write_phases: str,
    decode_steps: int,
) -> set[tuple[str, str, int, int | None]]:
    keys: set[tuple[str, str, int, int | None]] = set()
    if write_phases in ("prefill", "both"):
        keys.update((sample_id, "prefill", layer, None) for layer in layers)
    if write_phases in ("decode", "both"):
        for step in range(1, decode_steps + 1):
            keys.update((sample_id, "decode", layer, step) for layer in layers)
    return keys


def _batch_log_row(
    *,
    args: argparse.Namespace,
    batch: list[dict[str, Any]],
    status: str,
    seconds: float,
    processed_samples: int = 0,
    processed_tokens: int = 0,
    error: BaseException | None = None,
) -> dict[str, Any]:
    row = {
        "benchmark": args.benchmark,
        "model_key": args.model_key,
        "sample_ids": [item["sample_id"] for item in batch],
        "sample_indices": [item["sample_index"] for item in batch],
        "input_tokens": [len(item["input_ids"]) for item in batch],
        "status": status,
        "seconds": seconds,
        "processed_samples": processed_samples,
        "processed_prefill_tokens": processed_tokens,
        "layers": args.layers,
        "decode_steps": args.decode_steps,
        "write_phases": args.write_phases,
    }
    if error is not None:
        row["error_type"] = type(error).__name__
        row["error"] = str(error)
    return row


def _gini(values: list[int]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    weighted = sum((idx + 1) * value for idx, value in enumerate(sorted_values))
    return (2 * weighted) / (n * total) - (n + 1) / n


def _safe_layers(requested: list[int], layer_count: int) -> list[int]:
    out = []
    for layer in requested:
        if layer < 0:
            layer = layer_count + layer
        if 0 <= layer < layer_count:
            out.append(layer)
    return sorted(set(out))


def _make_padded_batch(batch: list[dict[str, Any]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(item["input_ids"]) for item in batch]
    padded = max(lengths)
    input_ids = torch.full((len(batch), padded), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), padded), dtype=torch.long)
    for row, item in enumerate(batch):
        values = torch.tensor(item["input_ids"], dtype=torch.long)
        input_ids[row, : values.numel()] = values
        attention_mask[row, : values.numel()] = 1
    return input_ids, attention_mask, lengths


def _base_forward(model, **kwargs):
    base = getattr(model, "model", None)
    if base is None:
        return model(**kwargs)
    return base(**kwargs)


def _next_token_from_lengths(model, hidden_states: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("model has no lm_head for greedy decode token selection")
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    last_idx = torch.tensor([length - 1 for length in lengths], device=hidden_states.device)
    with torch.inference_mode():
        logits = lm_head(hidden_states[batch_idx, last_idx, :].unsqueeze(1))
    return logits.argmax(dim=-1)


def _next_token_from_hidden(model, hidden_states: torch.Tensor) -> torch.Tensor:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("model has no lm_head for greedy decode token selection")
    with torch.inference_mode():
        logits = lm_head(hidden_states[:, -1:, :])
    return logits.argmax(dim=-1)


def _split_router_logits(router_logits: torch.Tensor, lengths: list[int], padded_length: int) -> list[torch.Tensor]:
    logits = router_logits.detach()
    batch_size = len(lengths)
    if logits.ndim == 3:
        if logits.shape[0] != batch_size:
            raise ValueError(f"unexpected 3D router logits shape: {tuple(logits.shape)}")
        return [logits[idx, : lengths[idx], :] for idx in range(batch_size)]
    if logits.ndim != 2:
        raise ValueError(f"unexpected router logits shape: {tuple(logits.shape)}")
    if logits.shape[0] == batch_size * padded_length:
        reshaped = logits.reshape(batch_size, padded_length, logits.shape[-1])
        return [reshaped[idx, : lengths[idx], :] for idx in range(batch_size)]
    if logits.shape[0] == sum(lengths):
        chunks = []
        start = 0
        for length in lengths:
            chunks.append(logits[start : start + length, :])
            start += length
        return chunks
    raise ValueError(
        f"cannot split router logits shape={tuple(logits.shape)} "
        f"batch={batch_size} padded={padded_length} sum_lengths={sum(lengths)}"
    )


def _routing_payload(logits: torch.Tensor, top_k: int, num_experts: int) -> dict[str, Any]:
    logits_f = logits.float()
    top_values, top_indices = torch.topk(logits_f, k=top_k, dim=-1)
    top_weights = torch.softmax(top_values, dim=-1)
    counts = torch.bincount(top_indices.reshape(-1).cpu(), minlength=num_experts).tolist()
    tokens = int(logits_f.shape[0])
    pair_count = tokens * top_k
    return {
        "tokens": tokens,
        "pair_count": pair_count,
        "counts": counts,
        "unique_experts": sum(1 for value in counts if value > 0),
        "zero_experts": sum(1 for value in counts if value == 0),
        "tail_lt4_experts": sum(1 for value in counts if 0 < value < 4),
        "hot_ge16_experts": sum(1 for value in counts if value >= 16),
        "gini": _gini(counts),
        "routes": top_indices.cpu().tolist(),
        "route_weights": [[float(x) for x in row] for row in top_weights.cpu().tolist()],
    }


def _write_phase_rows(
    *,
    out_path: Path,
    batch: list[dict[str, Any]],
    router_logits: tuple[torch.Tensor, ...],
    layers: list[int],
    lengths: list[int],
    padded_length: int,
    top_k: int,
    num_experts: int,
    phase: str,
    routing_source: str,
    decode_step: int | None = None,
) -> None:
    for layer in layers:
        split = _split_router_logits(router_logits[layer], lengths, padded_length)
        for item, logits in zip(batch, split, strict=True):
            row = {
                "benchmark": item["benchmark"],
                "model_key": item["model_key"],
                "model": MODEL_CONFIGS[item["model_key"]]["name"],
                "sample_id": item["sample_id"],
                "sample_index": item["sample_index"],
                "category": item.get("category", ""),
                "phase": phase,
                "layer": layer,
                "input_tokens": len(item["input_ids"]),
                "routing_source": routing_source,
                **_routing_payload(logits, top_k, num_experts),
            }
            if decode_step is not None:
                row["decode_step"] = decode_step
            append_jsonl(out_path, row)


def _process_batch(
    *,
    model,
    batch: list[dict[str, Any]],
    layers: list[int],
    top_k: int,
    num_experts: int,
    pad_token_id: int,
    decode_steps: int,
    write_phases: str,
    out_path: Path,
) -> tuple[int, int]:
    # Only run the decode forward passes when decode routing is actually requested;
    # otherwise (e.g. --write-phases prefill) they are wasted work whose router
    # logits are discarded, and the batched-KV path can crash for batch-size > 1.
    run_decode = decode_steps > 0 and write_phases in ("decode", "both")
    if run_decode and len(batch) > 1:
        raise ValueError("decode routing is only supported with --batch-size 1")
    input_ids, attention_mask, lengths = _make_padded_batch(batch, pad_token_id)
    need_prefill_router_logits = write_phases in ("prefill", "both")
    with torch.inference_mode():
        prefill = _base_forward(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=need_prefill_router_logits,
            use_cache=run_decode,
            return_dict=True,
        )
    safe = layers
    if write_phases in ("prefill", "both"):
        _write_phase_rows(
            out_path=out_path,
            batch=batch,
            router_logits=tuple(prefill.router_logits),
            layers=safe,
            lengths=lengths,
            padded_length=input_ids.shape[1],
            top_k=top_k,
            num_experts=num_experts,
            phase="prefill",
            routing_source="true_hidden_state_router_logits_topk_softmax",
        )

    past_key_values = getattr(prefill, "past_key_values", None)
    next_token = _next_token_from_lengths(model, prefill.last_hidden_state, lengths) if run_decode else None
    decode_attention_mask = attention_mask
    for step in range(decode_steps if run_decode else 0):
        decode_attention_mask = torch.cat(
            [decode_attention_mask, torch.ones((decode_attention_mask.shape[0], 1), dtype=decode_attention_mask.dtype)],
            dim=1,
        )
        with torch.inference_mode():
            decoded = _base_forward(
                model,
                input_ids=next_token,
                attention_mask=decode_attention_mask,
                past_key_values=past_key_values,
                output_router_logits=True,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = decoded.past_key_values
        if write_phases in ("decode", "both"):
            _write_phase_rows(
                out_path=out_path,
                batch=batch,
                router_logits=tuple(decoded.router_logits),
                layers=safe,
                lengths=[1] * len(batch),
                padded_length=1,
                top_k=top_k,
                num_experts=num_experts,
                phase="decode",
                routing_source="true_hidden_state_router_logits_topk_softmax_greedy_decode",
                decode_step=step + 1,
            )
        next_token = _next_token_from_hidden(model, decoded.last_hidden_state)
    return len(batch), sum(lengths)


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_paths()
    torch.set_num_threads(args.threads)
    model_cfg = MODEL_CONFIGS[args.model_key]
    output_path = args.output or (OUT_ROOT / "true_routing" / f"{args.benchmark}_{args.model_key}_pilot.jsonl")
    if output_path.exists() and not args.resume:
        output_path.unlink()

    config = AutoConfig.from_pretrained(model_cfg["path"], local_files_only=True)
    # _routing_payload uses softmax over the top-k logits. That equals the model's
    # true routing weights only when norm_topk_prob is True (softmax-over-all then
    # renormalize-over-topk is identical to softmax-over-topk). Fail loudly rather
    # than silently emit wrong weights for a model that does not renormalize.
    if not getattr(config, "norm_topk_prob", True):
        raise ValueError(
            "generate_true_routing assumes norm_topk_prob=True; this model has it False, "
            "so top-k softmax would not reproduce its routing weights"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], local_files_only=True)
    top_k = int(getattr(config, "num_experts_per_tok", None) or getattr(config, "experts_per_token"))
    num_experts = int(getattr(config, "num_experts", None) or getattr(config, "num_local_experts"))
    sample_filter = _parse_csv_strings(args.sample_ids)
    safe_layers = _safe_layers(args.layers, int(model_cfg["layers"]))
    resume_from = [Path(path) for path in (getattr(args, "resume_from", None) or [])]
    resume_paths = _resume_scan_paths(
        output_path=output_path,
        resume_from=resume_from,
        benchmark=args.benchmark,
        model_key=args.model_key,
    )
    existing_keys = _existing_route_keys_from_paths(resume_paths) if args.resume else set()

    records = []
    selected_total = 0
    skipped_existing = 0
    for sample_index, record in enumerate(_input_records(INPUT_FILES[args.benchmark], args.model_key)):
        sample_id = str(record["sample_id"])
        if sample_filter is not None and sample_id not in sample_filter:
            continue
        input_ids = list(record["input_ids"])
        if args.max_input_tokens is not None:
            input_ids = input_ids[: args.max_input_tokens]
        if not input_ids:
            continue
        selected_total += 1
        if args.limit is not None and selected_total > args.limit:
            break
        if args.resume:
            expected = _expected_route_keys(
                sample_id=sample_id,
                layers=safe_layers,
                write_phases=args.write_phases,
                decode_steps=args.decode_steps,
            )
            if expected and expected.issubset(existing_keys):
                skipped_existing += 1
                continue
        records.append(
            {
                "benchmark": args.benchmark,
                "model_key": args.model_key,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "category": record.get("category", ""),
                "input_ids": input_ids,
            }
        )

    if args.sort_by_length:
        records.sort(key=lambda item: (len(item["input_ids"]), item["sample_index"]))

    print(f"model_path={model_cfg['path']}")
    print(f"benchmark={args.benchmark} records={len(records)} output={output_path}")
    if not records:
        return {
            "schema_version": 1,
            "output": str(output_path),
            "benchmark": args.benchmark,
            "model_key": args.model_key,
            "selected_samples_total": selected_total,
            "selected_samples_after_resume": 0,
            "skipped_existing_samples": skipped_existing,
            "processed_samples": 0,
            "processed_prefill_tokens": 0,
            "run_seconds": 0.0,
            "layers": args.layers,
            "decode_steps": args.decode_steps,
            "write_phases": args.write_phases,
            "contains_route_weights": True,
            "failed_batches": 0,
            "failure_log": str(args.failure_log) if args.failure_log else None,
            "sample_log": str(args.sample_log) if args.sample_log else None,
            "resume_from": [str(path) for path in resume_from],
        }
    print("loading model", flush=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        local_files_only=True,
        dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"model_loaded_sec={time.time() - t_load:.2f}", flush=True)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    processed = 0
    tokens = 0
    failures: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    t_run = time.time()
    for item in records:
        if args.resume:
            expected = _expected_route_keys(
                sample_id=str(item["sample_id"]),
                layers=safe_layers,
                write_phases=args.write_phases,
                decode_steps=args.decode_steps,
            )
            live_resume_paths = _resume_scan_paths(
                output_path=output_path,
                resume_from=resume_from,
                benchmark=args.benchmark,
                model_key=args.model_key,
            )
            live_existing_keys = _existing_route_keys_from_paths(live_resume_paths)
            if expected and expected.issubset(live_existing_keys):
                skipped_existing += 1
                print(f"skip_existing_dynamic id={item['sample_id']}", flush=True)
                continue
        print(f"sample={processed + len(batch) + 1} id={item['sample_id']} tokens={len(item['input_ids'])}", flush=True)
        batch.append(item)
        if len(batch) >= args.batch_size:
            batch_start = time.time()
            try:
                done, seen = _process_batch(
                    model=model,
                    batch=batch,
                    layers=safe_layers,
                    top_k=top_k,
                    num_experts=num_experts,
                    pad_token_id=pad_token_id,
                    decode_steps=args.decode_steps,
                    write_phases=args.write_phases,
                    out_path=output_path,
                )
                processed += done
                tokens += seen
                if args.sample_log:
                    append_jsonl(
                        args.sample_log,
                        _batch_log_row(
                            args=args,
                            batch=batch,
                            status="passed",
                            seconds=time.time() - batch_start,
                            processed_samples=done,
                            processed_tokens=seen,
                        ),
                    )
            except Exception as exc:
                batch_seconds = time.time() - batch_start
                failure = {
                    "benchmark": args.benchmark,
                    "model_key": args.model_key,
                    "sample_ids": [item["sample_id"] for item in batch],
                    "sample_indices": [item["sample_index"] for item in batch],
                    "seconds": batch_seconds,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                if args.failure_log:
                    append_jsonl(args.failure_log, failure)
                if args.sample_log:
                    append_jsonl(
                        args.sample_log,
                        _batch_log_row(
                            args=args,
                            batch=batch,
                            status="failed",
                            seconds=batch_seconds,
                            error=exc,
                        ),
                    )
                print(f"failed batch sample_ids={failure['sample_ids']}: {exc}", flush=True)
                if not args.keep_going:
                    raise
            batch = []
    if batch:
        batch_start = time.time()
        try:
            done, seen = _process_batch(
                model=model,
                batch=batch,
                layers=safe_layers,
                top_k=top_k,
                num_experts=num_experts,
                pad_token_id=pad_token_id,
                decode_steps=args.decode_steps,
                write_phases=args.write_phases,
                out_path=output_path,
            )
            processed += done
            tokens += seen
            if args.sample_log:
                append_jsonl(
                    args.sample_log,
                    _batch_log_row(
                        args=args,
                        batch=batch,
                        status="passed",
                        seconds=time.time() - batch_start,
                        processed_samples=done,
                        processed_tokens=seen,
                    ),
                )
        except Exception as exc:
            batch_seconds = time.time() - batch_start
            failure = {
                "benchmark": args.benchmark,
                "model_key": args.model_key,
                "sample_ids": [item["sample_id"] for item in batch],
                "sample_indices": [item["sample_index"] for item in batch],
                "seconds": batch_seconds,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            if args.failure_log:
                append_jsonl(args.failure_log, failure)
            if args.sample_log:
                append_jsonl(
                    args.sample_log,
                    _batch_log_row(
                        args=args,
                        batch=batch,
                        status="failed",
                        seconds=batch_seconds,
                        error=exc,
                    ),
                )
            print(f"failed batch sample_ids={failure['sample_ids']}: {exc}", flush=True)
            if not args.keep_going:
                raise
    summary = {
        "schema_version": 1,
        "output": str(output_path),
        "benchmark": args.benchmark,
        "model_key": args.model_key,
        "selected_samples_total": selected_total,
        "selected_samples_after_resume": len(records),
        "skipped_existing_samples": skipped_existing,
        "processed_samples": processed,
        "processed_prefill_tokens": tokens,
        "run_seconds": time.time() - t_run,
        "layers": args.layers,
        "decode_steps": args.decode_steps,
        "write_phases": args.write_phases,
        "contains_route_weights": True,
        "failed_batches": len(failures),
        "failure_log": str(args.failure_log) if args.failure_log else None,
        "sample_log": str(args.sample_log) if args.sample_log else None,
        "resume_from": [str(path) for path in resume_from],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(INPUT_FILES), required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), default="qwen3")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids")
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--layers", type=_parse_csv_ints, default=_parse_csv_ints("0,12,23"))
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--write-phases", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--sample-log", type=Path)
    parser.add_argument("--resume-from", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = run(args)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
