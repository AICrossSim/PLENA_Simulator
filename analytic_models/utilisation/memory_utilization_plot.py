import json
import re

import os
from pathlib import Path

def compute_hbm_storage(batch_size, model_param_path, kv_size, kv_precision=2, act_precision=2, wt_precision=2):
    """
    Compute HBM storage requirements for model weights, kv_cache, and other elements.

    Args:
        batch_size (int): The batch size.
        model_param_path (str or Path): Path to the model parameter JSON file.
        kv_size (int): The kv cache sequence length.
        kv_precision (int, optional): Byte size of KV cache data type (default: 2 for fp16).
        act_precision (int, optional): Byte size of activation data type (unused; default: 2).
        wt_precision (int, optional): Byte size of weight data type (default: 2 for fp16).

    Returns:
        dict: Dictionary with keys 'weights', 'kv_cache', and 'other' with size in GB.
    """
    model_param = json.load(open(model_param_path))
    hidden_size = model_param["hidden_size"]
    num_attention_heads = model_param["num_attention_heads"]
    num_hidden_layers = model_param["num_hidden_layers"]
    intermediate_size = model_param["intermediate_size"]
    num_key_value_heads = model_param["num_key_value_heads"]
    repeat_layer = model_param["num_hidden_layers"]
    vocab_size = model_param["vocab_size"]
    head_dim = hidden_size // num_attention_heads

    # Extract model size in Billion parameters from the file name (e.g., ".../llama-3.1-8b.json" -> 8)
    model_filename = str(model_param_path).split('/')[-1]
    match = re.search(r'(\d+)[bB]', model_filename)
    if match:
        model_size_b = int(match.group(1))
    else:
        model_size_b = None  # Optionally raise error

    if model_size_b is not None:
        weight_size_gb = model_size_b * 1e9 * wt_precision / (1024 ** 3)
    else:
        weight_size_gb = 0  # Or raise error

    kv_cache_gb = (2 * head_dim * num_key_value_heads * kv_size * batch_size * kv_precision * repeat_layer) / 1024 / 1024 / 1024

    other_gb = (hidden_size * 4 + intermediate_size * 4 + vocab_size * 4) / 1024 / 1024 / 1024

    hbm_storage = {
        "weights": weight_size_gb,
        "kv_cache": kv_cache_gb,
        "other": other_gb
    }
    return hbm_storage



if __name__ == "__main__":
    # Project root is 2 levels up from analytic_models/utilisation/
    current_dir = Path(__file__).resolve().parents[2]
    model_param_path = os.path.join(current_dir, "doc/Model_Lib/llama-3.1-8b.json")
    batch_size = 1
    kv_size = 8024
    kv_precision = 2
    act_precision = 2
    wt_precision = 2
    hbm_storage = compute_hbm_storage(batch_size, model_param_path, kv_size, kv_precision, act_precision, wt_precision)
    print(hbm_storage)

    import matplotlib.pyplot as plt
    import numpy as np

    # kv_size values
    kv_sizes = [1000, 10000, 50000, 100000]  # [1k, 10k, 50k, 100k]
    kv_size_labels = ['1k', '10k', '50k', '100k']

    colors = [
        (240/255, 249/255, 232/255),  # light green
        (186/255, 228/255, 188/255),  # green
        (123/255, 204/255, 196/255),  # teal
        (67/255, 162/255, 202/255),   # blue
        (8/255, 104/255, 172/255)     # dark blue
    ]


    # Define the four precision settings (kv, act, wt)
    # Format: (kv_precision, act_precision, wt_precision)
    precision_settings = [
        (2, 2, 2),
        (1.125, 1.125, 1.125),
        (1.125, 1.125, 0.625),
        (0.625, 1.125, 0.625),
    ]
    precision_labels = [
        "KV=BF16 ,ACT=BF16, Weight=BF16",
        "KV=MXINT E4M3 ,ACT=MXINT E4M3, Weight= MXINT E4M3",
        "KV=MXINT E4M3 ,ACT=MXINT E4M3, Weight= MXINT E2M1",
        "KV=MXINT E2M1 ,ACT=MXINT E4M3, Weight= MXINT E2M1",
    ]
    num_settings = len(precision_settings)

    # Colors for each memory type
    color_dict = {
        "weights": colors[2],
        "kv_cache": colors[4],
    }
    mem_keys = ["weights", "kv_cache"]

    # Collect data: result[setting_idx][kv_idx][mem_type]
    results = []  # (settings, kv_size, mem_type)
    for setting in precision_settings:
        setting_results = []
        for kv in kv_sizes:
            kv_precision, act_precision, wt_precision = setting
            hbm_storage = compute_hbm_storage(batch_size, model_param_path, kv, kv_precision, act_precision, wt_precision)
            vals = [hbm_storage[k] for k in mem_keys]
            setting_results.append(vals)
        results.append(setting_results)
    results = np.array(results)  # shape: (num_settings, num_kv, 3)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.1  # Thinner bars
    spacing = 0.02  # Space between each precision group
    group_width = num_settings * bar_width + (num_settings - 1) * spacing
    x = np.arange(len(kv_sizes)) * 0.6  # Reduce spacing between bar groups

    # For bar placement (center group at each kv_size)
    for sidx, setting_label in enumerate(precision_labels):
        for kidx in range(len(kv_sizes)):
            left = x[kidx] - group_width/2 + sidx*(bar_width + spacing) + bar_width/2
            bottom = 0
            prev_bottom = 0
            # Plot stacked: weights, kv_cache, other
            for midx, mem in enumerate(mem_keys):
                val = results[sidx, kidx, midx]
                color = color_dict[mem]
                bar = ax.bar(left, val, bar_width, bottom=prev_bottom, color=color, edgecolor="black" if sidx==0 else None)
                prev_bottom += val
        # Put legend entries only for the "weights" bar for nice stacking
        if sidx == 0:
            # Just for the legend
            handles = [plt.Rectangle((0,0),1,1, color=color_dict[m]) for m in mem_keys]
            ax.legend(handles, mem_keys, title="Memory Type", loc='upper right', fontsize=13, title_fontsize=14)

    # X-axis
    ax.set_xticks(x)
    xtick_labels = []
    for k in kv_size_labels:
        label = f"{k}"
        xtick_labels.append(label)
    ax.set_xticklabels(xtick_labels, fontsize=12)

    # Put precision labels (P1,P2,...) centered on each bar
    for kidx, xpos in enumerate(x):
        for sidx in range(num_settings):
            left = xpos - group_width/2 + sidx*(bar_width + spacing) + bar_width/2
            height = results[sidx, kidx].sum()
            # Center P-label on top part of each stacked bar
            ax.text(
                left, height + 0.03*max(results.flatten()),
                f"P{sidx+1}",
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=0
            )

    # Additional legend for precision index -> actual precision setting (top left)
    precision_legend = [f"P{i+1}: {lbl}" for i, lbl in enumerate(precision_labels)]
    ax.text(0.02, 0.98, "Precisions:\n" + "\n".join(precision_legend), 
            transform=ax.transAxes, fontsize=13, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    ax.set_xlabel("KV Size", fontsize=15)
    ax.set_ylabel("Memory Usage (GB, Stacked)", fontsize=15)
    ax.set_title("Single Batch Memory Distribution vs KV Size", fontsize=17)
    ax.set_ylim(bottom=0, top=40)
    plt.tight_layout()
    plt.savefig("memory_utilization.png", dpi=300, bbox_inches='tight')
