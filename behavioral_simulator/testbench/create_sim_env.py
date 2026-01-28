import os
import torch
import numpy as np


def np_array_to_str_2f(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return "[" + " ".join([f"{v:.2f}" for v in arr]) + "]"
    elif arr.ndim == 2:
        rows = ["  " + " ".join([f"{v:.2f}" for v in row]) for row in arr]
        return "[\n" + "\n".join(rows) + "\n]"
    else:
        # For higher dimensions, default to numpy's print (rare for this context)
        return np.array2string(arr, formatter={'float_kind':lambda x: "%.2f" % x})

def create_sim_env(input_tensor, generated_code, golden_result, fp_preload = None, int_preload = None, build_dir = None):
    if build_dir is None:
        build_dir = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_dir, exist_ok=True)
    if isinstance(input_tensor, dict):
        for key, value in input_tensor.items():
            with open(os.path.join(build_dir, f"{key}.pt"), "wb") as f:
                torch.save(value, f)
    else:
        with open(os.path.join(build_dir, "input_tensor.pt"), "wb") as f:
            torch.save(input_tensor, f)
    with open(os.path.join(build_dir, "generated_asm_code.asm"), "w") as f:
        f.write(generated_code)

    # Store golden_result in a readable format, including tensor contents.
    if fp_preload is not None:
        fp_to_load = fp_preload
    else:
        fp_to_load = torch.zeros(10, dtype=torch.float16)
    with open(os.path.join(build_dir, "fp_sram.bin"), "wb") as f:
        fp16_array = np.array(fp_to_load, dtype=np.float16)
        f.write(fp16_array.tobytes())

    if int_preload is not None:
        int_to_load = int_preload
    else:   
        int_to_load = torch.zeros(10, dtype=torch.int32)
    with open(os.path.join(build_dir, "int_sram.bin"), "wb") as f:
        int_array = np.array(int_to_load, dtype=np.uint32)
        f.write(int_array.tobytes())
    
    with open(os.path.join(build_dir, "golden_result.txt"), "w") as f:
        f.write("Input Tensor:\n")
        if isinstance(input_tensor, dict):
            for key, value in input_tensor.items():
                value_np = value.detach().cpu().float().numpy()
                f.write(f"{key}:\n{np_array_to_str_2f(value_np)}\n")
        else:
            value_np = input_tensor.detach().cpu().float().numpy()
            f.write(np_array_to_str_2f(value_np))
        f.write("\n\nOriginal Output:\n")
        # Convert BFloat16 to float32 before converting to numpy
        output_np = golden_result["original_output"].detach().cpu().float().numpy()
        f.write(np_array_to_str_2f(output_np))