import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


SAFETENSORS_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def human_bytes(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 B"

    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def human_count(value: int) -> str:
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1_000:.2f}K"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1_000_000_000:.2f}B"


def shape_to_str(shape) -> str:
    if not shape:
        return "[]"
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def safe_numel(shape) -> int:
    if not shape:
        return 1
    return math.prod(shape)


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for inspecting .pt/.bin/.pth checkpoints.")
    return torch


def normalize_dtype(dtype) -> str:
    if torch is not None and isinstance(dtype, torch.dtype):
        return str(dtype).replace("torch.", "").upper()
    return str(dtype).replace("torch.", "").upper()


def dtype_size_bytes(dtype) -> int | None:
    dtype_str = normalize_dtype(dtype)
    if dtype_str in SAFETENSORS_DTYPE_SIZES:
        return SAFETENSORS_DTYPE_SIZES[dtype_str]

    if torch is None:
        return None

    torch_dtype = getattr(torch, dtype_str.lower(), None)
    if isinstance(torch_dtype, torch.dtype):
        return torch.tensor([], dtype=torch_dtype).element_size()

    return None


def print_pt_file(file_path, max_depth=3, max_items=100):
    """
    Print contents of a PyTorch checkpoint recursively.

    Args:
        file_path: Path to the .pt file
        max_depth: Maximum depth for nested structures
        max_items: Maximum number of items to print for large collections
    """
    torch_mod = require_torch()
    print(f"\n=== Contents of {file_path} ===\n")
    data = torch_mod.load(file_path, map_location="cpu")
    print_recursive(data, depth=0, max_depth=max_depth, max_items=max_items)


def print_recursive(obj, depth=0, max_depth=3, max_items=100):
    torch_mod = require_torch()
    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}... (max depth reached)")
        return

    if isinstance(obj, dict):
        print(f"{indent}Dictionary with {len(obj)} keys:")
        for i, (key, value) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more items)")
                break
            print(f"{indent}  Key: '{key}'")
            print_recursive(value, depth + 2, max_depth, max_items)

    elif isinstance(obj, torch_mod.Tensor):
        print(f"{indent}Tensor:")
        print(f"{indent}  Shape: {tuple(obj.shape)}")
        print(f"{indent}  Dtype: {obj.dtype}")
        print(f"{indent}  Device: {obj.device}")
        if obj.numel() > 0 and torch_mod.is_floating_point(obj):
            print(f"{indent}  Min: {obj.min().item():.6f}")
            print(f"{indent}  Max: {obj.max().item():.6f}")
            print(f"{indent}  Mean: {obj.mean().item():.6f}")
        if obj.numel() <= 20:
            print(f"{indent}  Values: {obj.tolist()}")
        else:
            print(f"{indent}  First few values: {obj.flatten()[:10].tolist()}")

    elif isinstance(obj, (list, tuple)):
        type_name = "List" if isinstance(obj, list) else "Tuple"
        print(f"{indent}{type_name} with {len(obj)} items:")
        for i, item in enumerate(obj):
            if i >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more items)")
                break
            print(f"{indent}  [{i}]:")
            print_recursive(item, depth + 2, max_depth, max_items)

    elif isinstance(obj, torch_mod.nn.Module):
        print(f"{indent}PyTorch Model: {obj.__class__.__name__}")
        print(f"{indent}{obj}")

    elif isinstance(obj, (int, float, str, bool)):
        print(f"{indent}Value ({type(obj).__name__}): {obj}")

    else:
        print(f"{indent}Type: {type(obj)}")
        print(f"{indent}Value: {obj}")


def resolve_hf_model_path(model_ref: str, local_files_only: bool) -> Path:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required when model_ref is not a local path."
        ) from exc

    downloaded_path = snapshot_download(model_ref, local_files_only=local_files_only)
    return Path(downloaded_path).resolve()


def load_model_config_summary(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}

    with config_path.open() as f:
        config = json.load(f)

    summary_keys = [
        "model_type",
        "architectures",
        "torch_dtype",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    ]

    return {key: config[key] for key in summary_keys if key in config}


def read_safetensors_header(file_path: Path) -> dict:
    with file_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header


def inspect_safetensors_file(file_path: Path) -> list[dict]:
    header = read_safetensors_header(file_path)
    tensors = []

    for name, info in header.items():
        if name == "__metadata__":
            continue

        shape = tuple(info["shape"])
        dtype = normalize_dtype(info["dtype"])
        numel = safe_numel(shape)
        bytes_per_elem = dtype_size_bytes(dtype)

        tensors.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "numel": numel,
                "nbytes": None if bytes_per_elem is None else numel * bytes_per_elem,
                "source": file_path.name,
            }
        )

    return tensors


def inspect_safetensors_index(index_path: Path) -> list[dict]:
    with index_path.open() as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_headers = {}

    for shard_name in sorted(set(weight_map.values())):
        shard_path = index_path.parent / shard_name
        shard_tensors = inspect_safetensors_file(shard_path)
        shard_headers[shard_name] = {tensor["name"]: tensor for tensor in shard_tensors}

    tensors = []
    for tensor_name, shard_name in weight_map.items():
        tensor_info = dict(shard_headers[shard_name][tensor_name])
        tensor_info["source"] = shard_name
        tensors.append(tensor_info)

    return tensors


def load_torch_object_metadata(file_path: Path):
    torch_mod = require_torch()
    last_error = None
    load_attempts = [
        ("meta/weights_only", dict(map_location="meta", weights_only=True)),
        ("meta/full", dict(map_location="meta")),
        ("cpu/weights_only", dict(map_location="cpu", weights_only=True)),
        ("cpu/full", dict(map_location="cpu")),
    ]

    for mode, kwargs in load_attempts:
        try:
            obj = torch_mod.load(file_path, **kwargs)
            return obj, mode
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to inspect checkpoint {file_path}: {last_error}") from last_error


def flatten_tensors(obj, prefix="") -> list[dict]:
    torch_mod = require_torch()
    tensors = []

    if isinstance(obj, torch_mod.Tensor):
        shape = tuple(obj.shape)
        dtype = normalize_dtype(obj.dtype)
        numel = obj.numel()
        tensors.append(
            {
                "name": prefix or "<root>",
                "shape": shape,
                "dtype": dtype,
                "numel": numel,
                "nbytes": numel * obj.element_size(),
                "source": None,
            }
        )
        return tensors

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            tensors.extend(flatten_tensors(value, child_prefix))
        return tensors

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            tensors.extend(flatten_tensors(value, child_prefix))

    return tensors


def inspect_torch_checkpoint(file_path: Path) -> tuple[list[dict], str]:
    obj, mode = load_torch_object_metadata(file_path)
    tensors = flatten_tensors(obj)
    return tensors, mode


def inspect_hf_weights(model_ref: str, local_files_only: bool) -> tuple[Path, dict, list[dict], str]:
    model_path = resolve_hf_model_path(model_ref, local_files_only=local_files_only)

    if model_path.is_file():
        suffix = model_path.suffix.lower()
        if suffix == ".safetensors":
            return model_path, {}, inspect_safetensors_file(model_path), "single safetensors file"
        if suffix in {".pt", ".bin", ".pth"}:
            tensors, mode = inspect_torch_checkpoint(model_path)
            return model_path, {}, tensors, f"single torch checkpoint ({mode})"
        raise ValueError(f"Unsupported weight file: {model_path}")

    config_summary = load_model_config_summary(model_path)

    safetensors_index = model_path / "model.safetensors.index.json"
    if safetensors_index.exists():
        tensors = inspect_safetensors_index(safetensors_index)
        return model_path, config_summary, tensors, "sharded safetensors index"

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if safetensors_files:
        tensors = []
        for file_path in safetensors_files:
            tensors.extend(inspect_safetensors_file(file_path))
        source_desc = "single safetensors file" if len(safetensors_files) == 1 else f"{len(safetensors_files)} safetensors files"
        return model_path, config_summary, tensors, source_desc

    pytorch_index = model_path / "pytorch_model.bin.index.json"
    if pytorch_index.exists():
        with pytorch_index.open() as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_cache = {}
        tensors = []

        for tensor_name, shard_name in weight_map.items():
            if shard_name not in shard_cache:
                shard_tensors, mode = inspect_torch_checkpoint(model_path / shard_name)
                shard_cache[shard_name] = {
                    tensor["name"]: tensor for tensor in shard_tensors
                }
                shard_cache[shard_name]["__load_mode__"] = mode

            tensor_info = dict(shard_cache[shard_name][tensor_name])
            tensor_info["source"] = shard_name
            tensors.append(tensor_info)

        load_modes = sorted(
            {
                shard_cache[shard_name]["__load_mode__"]
                for shard_name in weight_map.values()
            }
        )
        return model_path, config_summary, tensors, f"sharded torch index ({', '.join(load_modes)})"

    pytorch_candidates = sorted(model_path.glob("*.bin")) + sorted(model_path.glob("*.pth")) + sorted(model_path.glob("*.pt"))
    if pytorch_candidates:
        tensors = []
        load_modes = []
        for file_path in pytorch_candidates:
            file_tensors, load_mode = inspect_torch_checkpoint(file_path)
            for tensor in file_tensors:
                tensor["source"] = file_path.name
            tensors.extend(file_tensors)
            load_modes.append(load_mode)
        return model_path, config_summary, tensors, f"{len(pytorch_candidates)} torch checkpoint file(s) ({', '.join(sorted(set(load_modes)))})"

    raise FileNotFoundError(f"No supported weight files found under {model_path}")


def group_tensors_by_prefix(tensors: list[dict], depth: int) -> list[dict]:
    depth = max(depth, 1)
    grouped = defaultdict(lambda: {"tensor_count": 0, "numel": 0, "nbytes": 0})

    for tensor in tensors:
        parts = tensor["name"].split(".")
        prefix = ".".join(parts[:depth]) if len(parts) >= depth else tensor["name"]
        grouped[prefix]["tensor_count"] += 1
        grouped[prefix]["numel"] += tensor["numel"]
        grouped[prefix]["nbytes"] += tensor["nbytes"] or 0

    grouped_rows = []
    for name, stats in grouped.items():
        grouped_rows.append(
            {
                "name": name,
                "tensor_count": stats["tensor_count"],
                "numel": stats["numel"],
                "nbytes": stats["nbytes"],
            }
        )

    return sorted(grouped_rows, key=lambda row: row["nbytes"], reverse=True)


def build_dtype_summary(tensors: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: {"tensor_count": 0, "numel": 0, "nbytes": 0})

    for tensor in tensors:
        grouped[tensor["dtype"]]["tensor_count"] += 1
        grouped[tensor["dtype"]]["numel"] += tensor["numel"]
        grouped[tensor["dtype"]]["nbytes"] += tensor["nbytes"] or 0

    rows = []
    for dtype, stats in grouped.items():
        rows.append(
            {
                "dtype": dtype,
                "tensor_count": stats["tensor_count"],
                "numel": stats["numel"],
                "nbytes": stats["nbytes"],
            }
        )
    return sorted(rows, key=lambda row: row["nbytes"], reverse=True)


def print_tensor_table(
    tensors: list[dict],
    limit: int,
    sort_by: str,
    name_filter: str,
    title: str,
):
    rows = tensors
    if name_filter:
        rows = [tensor for tensor in rows if name_filter in tensor["name"]]

    if sort_by == "size":
        rows = sorted(rows, key=lambda tensor: tensor["nbytes"] or 0, reverse=True)
    elif sort_by == "name":
        rows = sorted(rows, key=lambda tensor: tensor["name"])

    if limit >= 0:
        rows = rows[:limit]

    print(f"\n{title}")
    if not rows:
        print("  (no tensors matched)")
        return

    for tensor in rows:
        source = f" | source={tensor['source']}" if tensor.get("source") else ""
        print(
            "  "
            f"{tensor['name']} | shape={shape_to_str(tensor['shape'])} | dtype={tensor['dtype']} | "
            f"numel={human_count(tensor['numel'])} | size={human_bytes(tensor['nbytes'] or 0)}{source}"
        )


def print_hf_weight_summary(
    model_ref: str,
    local_files_only: bool,
    tensor_limit: int,
    largest_limit: int,
    group_depth: int,
    group_limit: int,
    sort_by: str,
    name_filter: str,
):
    model_path, config_summary, tensors, source_desc = inspect_hf_weights(
        model_ref,
        local_files_only=local_files_only,
    )

    total_numel = sum(tensor["numel"] for tensor in tensors)
    total_nbytes = sum((tensor["nbytes"] or 0) for tensor in tensors)
    dtype_rows = build_dtype_summary(tensors)
    grouped_rows = group_tensors_by_prefix(tensors, depth=group_depth)
    largest_tensors = sorted(tensors, key=lambda tensor: tensor["nbytes"] or 0, reverse=True)

    print("\n=== Hugging Face Weight Summary ===\n")
    print(f"Model ref      : {model_ref}")
    print(f"Resolved path  : {model_path}")
    print(f"Weight source  : {source_desc}")
    print(f"Tensor count   : {len(tensors)}")
    print(f"Total params   : {human_count(total_numel)} ({total_numel})")
    print(f"Total size     : {human_bytes(total_nbytes)} ({total_nbytes} bytes)")

    if config_summary:
        print("\nConfig summary:")
        for key, value in config_summary.items():
            print(f"  {key}: {value}")

    print("\nDtype breakdown:")
    for row in dtype_rows:
        print(
            "  "
            f"{row['dtype']}: tensors={row['tensor_count']}, params={human_count(row['numel'])} ({row['numel']}), "
            f"size={human_bytes(row['nbytes'])}"
        )

    print(f"\nTop module groups (prefix depth={group_depth}):")
    for row in grouped_rows[:group_limit]:
        print(
            "  "
            f"{row['name']}: tensors={row['tensor_count']}, params={human_count(row['numel'])} ({row['numel']}), "
            f"size={human_bytes(row['nbytes'])}"
        )

    print_tensor_table(
        largest_tensors,
        limit=largest_limit,
        sort_by="size",
        name_filter=name_filter,
        title=f"Top {largest_limit} largest tensors:",
    )
    print_tensor_table(
        tensors,
        limit=tensor_limit,
        sort_by=sort_by,
        name_filter=name_filter,
        title=f"Tensor list (limit={tensor_limit}, sort={sort_by}):",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch checkpoints and Hugging Face model weights."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pt_parser = subparsers.add_parser(
        "print-pt",
        help="Recursively print the contents of a .pt/.pth checkpoint.",
    )
    pt_parser.add_argument("file_path", help="Path to the checkpoint file.")
    pt_parser.add_argument("--max-depth", type=int, default=3)
    pt_parser.add_argument("--max-items", type=int, default=100)

    hf_parser = subparsers.add_parser(
        "hf-summary",
        help="Print Hugging Face weight summary from a local path or repo id.",
    )
    hf_parser.add_argument(
        "model_ref",
        help="Local model directory / weight file, or a Hugging Face repo id.",
    )
    hf_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local Hugging Face cache; do not try to download.",
    )
    hf_parser.add_argument(
        "--tensor-limit",
        type=int,
        default=80,
        help="Number of tensors to print in the main tensor list. Use -1 for all.",
    )
    hf_parser.add_argument(
        "--largest-limit",
        type=int,
        default=20,
        help="Number of largest tensors to print.",
    )
    hf_parser.add_argument(
        "--group-depth",
        type=int,
        default=2,
        help="How many name segments to use when grouping tensors by module prefix.",
    )
    hf_parser.add_argument(
        "--group-limit",
        type=int,
        default=20,
        help="How many grouped module rows to print.",
    )
    hf_parser.add_argument(
        "--sort-by",
        choices=["order", "size", "name"],
        default="order",
        help="Sort order for the main tensor list.",
    )
    hf_parser.add_argument(
        "--name-filter",
        default="",
        help="Only print tensors whose names contain this substring.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "print-pt":
        print_pt_file(
            args.file_path,
            max_depth=args.max_depth,
            max_items=args.max_items,
        )
        return

    if args.command == "hf-summary":
        print_hf_weight_summary(
            model_ref=args.model_ref,
            local_files_only=args.local_files_only,
            tensor_limit=args.tensor_limit,
            largest_limit=args.largest_limit,
            group_depth=args.group_depth,
            group_limit=args.group_limit,
            sort_by=args.sort_by,
            name_filter=args.name_filter,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
