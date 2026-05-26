import os
import torch
import numpy as np
import json


def np_array_to_str_2f(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return "[" + " ".join([f"{v:.2f}" for v in arr]) + "]"
    elif arr.ndim == 2:
        rows = ["  " + " ".join([f"{v:.2f}" for v in row]) for row in arr]
        return "[\n" + "\n".join(rows) + "\n]"
    else:
        # For higher dimensions, default to numpy's print (rare for this context).
        # Force full output (threshold=sys.maxsize) — default threshold=1000
        # summarizes large arrays with "..." which silently truncates goldens.
        import sys as _sys

        return np.array2string(
            arr,
            formatter={"float_kind": lambda x: f"{x:.2f}"},
            threshold=_sys.maxsize,
        )


def np_array_to_str_precise(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return "[" + " ".join([f"{v:.9g}" for v in arr]) + "]"
    if arr.ndim == 2:
        rows = ["  " + " ".join([f"{v:.9g}" for v in row]) for row in arr]
        return "[\n" + "\n".join(rows) + "\n]"

    import sys as _sys

    return np.array2string(
        arr,
        formatter={"float_kind": lambda x: f"{x:.9g}"},
        threshold=_sys.maxsize,
    )


def create_sim_env(
    input_tensor,
    generated_code,
    golden_result,
    fp_preload=None,
    int_preload=None,
    build_dir=None,
    vram_preload=None,
    tensor_layouts=None,
):
    if build_dir is None:
        build_dir = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_dir, exist_ok=True)
    if isinstance(input_tensor, dict):
        for key, value in input_tensor.items():
            with open(os.path.join(build_dir, f"{key}.pt"), "wb") as f:
                torch.save(_materialize_tensor(value), f)
    else:
        with open(os.path.join(build_dir, "input_tensor.pt"), "wb") as f:
            torch.save(_materialize_tensor(input_tensor), f)
    with open(os.path.join(build_dir, "generated_asm_code.asm"), "w") as f:
        f.write(generated_code)

    if tensor_layouts is None and isinstance(golden_result, dict):
        tensor_layouts = golden_result.get("tensor_layouts")
    if tensor_layouts is not None:
        with open(os.path.join(build_dir, "tensor_layouts.json"), "w") as f:
            json.dump(tensor_layouts, f, indent=2, sort_keys=True)

    if isinstance(golden_result, dict):
        _sidecar_map = {
            "compile_info": "compile_info.json",
            "stage_checkpoints": "stage_checkpoints.json",
            "data_order": "data_order.json",
        }
        for key, filename in _sidecar_map.items():
            if key in golden_result and golden_result[key] is not None:
                with open(os.path.join(build_dir, filename), "w") as f:
                    json.dump(golden_result[key], f, indent=2, sort_keys=True)

    output_tensor = golden_result["original_output"].detach().cpu().float()
    torch.save(output_tensor, os.path.join(build_dir, "golden_output.pt"))

    # Store golden_result in a readable format for small tests. Large padded
    # model builds otherwise spend minutes writing hundreds of MiB of text that
    # the emulator never needs; check_mem.py reads golden_output.pt when present.
    text_dump_max_values = int(os.environ.get("PLENA_TEXT_DUMP_MAX_VALUES", "200000"))
    force_full_text = os.environ.get("PLENA_FULL_TEXT_GOLDEN", "0") == "1"
    input_value_count = _count_tensor_values(input_tensor)
    output_value_count = int(output_tensor.numel())
    write_full_text = force_full_text or (input_value_count + output_value_count <= text_dump_max_values)

    if fp_preload is not None:
        fp_to_load = fp_preload
    else:
        fp_to_load = torch.zeros(10, dtype=torch.float16)
    with open(os.path.join(build_dir, "fp_sram.bin"), "wb") as f:
        _fp_data = fp_to_load.numpy() if hasattr(fp_to_load, "numpy") else fp_to_load
        fp16_array = np.array(_fp_data, dtype=np.float16)
        f.write(fp16_array.tobytes())

    if int_preload is not None:
        int_to_load = int_preload
    else:
        int_to_load = torch.zeros(10, dtype=torch.int32)
    with open(os.path.join(build_dir, "int_sram.bin"), "wb") as f:
        _int_data = int_to_load.numpy() if hasattr(int_to_load, "numpy") else int_to_load
        int_array = np.array(_int_data, dtype=np.uint32)
        f.write(int_array.tobytes())

    with open(os.path.join(build_dir, "golden_result.txt"), "w") as f:
        if write_full_text:
            f.write("Input Tensor:\n")
            if isinstance(input_tensor, dict):
                for key, value in input_tensor.items():
                    value = _materialize_tensor(value)
                    value_np = value.detach().cpu().float().numpy()
                    f.write(f"{key}:\n{np_array_to_str_2f(value_np)}\n")
            else:
                input_tensor = _materialize_tensor(input_tensor)
                value_np = input_tensor.detach().cpu().float().numpy()
                f.write(np_array_to_str_2f(value_np))
            f.write("\n\nOriginal Output:\n")
            f.write(np_array_to_str_precise(output_tensor.numpy()))
        else:
            summary = {
                "format": "compact",
                "reason": "large tensor dump suppressed",
                "input_values": input_value_count,
                "output_values": output_value_count,
                "golden_output_file": "golden_output.pt",
                "input_tensors": _tensor_summary(input_tensor),
                "original_output": _tensor_summary(output_tensor),
            }
            f.write("Compact Golden Result\n")
            f.write(json.dumps(summary, indent=2, sort_keys=True))

    if vram_preload is not None:
        # vram_preload: a flat tensor or numpy array of fp16 values representing
        # the initial VRAM contents.  Written as raw fp16 bytes so the emulator
        # can load it via --vram vram_preload.bin.
        with open(os.path.join(build_dir, "vram_preload.bin"), "wb") as f:
            _vram_data = vram_preload.numpy() if hasattr(vram_preload, "numpy") else vram_preload
            vram_fp16 = np.array(_vram_data, dtype=np.float16)
            f.write(vram_fp16.tobytes())


def _count_tensor_values(value):
    if isinstance(value, dict):
        return sum(_count_tensor_values(v) for v in value.values())
    shape = getattr(value, "shape", None)
    if shape is not None:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total
    if hasattr(value, "numel"):
        return int(value.numel())
    return int(np.asarray(value).size)


def _tensor_summary(value):
    if isinstance(value, dict):
        return {key: _tensor_summary(tensor) for key, tensor in value.items()}
    shape = getattr(value, "shape", None)
    if shape is not None and not hasattr(value, "detach"):
        return {
            "shape": [int(dim) for dim in shape],
            "dtype": str(getattr(getattr(value, "vector", None), "dtype", "lazy")),
            "numel": _count_tensor_values(value),
            "storage": "lazy",
        }
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
    }


def _materialize_tensor(value):
    materialize = getattr(value, "materialize", None)
    if callable(materialize):
        return materialize()
    return value
