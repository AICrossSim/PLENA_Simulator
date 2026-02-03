from attainable import attn_model_config

import os
from pathlib import Path
# Project root is 2 levels up from analytic_models/utilisation/
current_dir = Path(__file__).resolve().parents[2]
model_param_path = os.path.join(current_dir, "doc/Model_Lib/llama-3.1-8b.json")

import matplotlib.pyplot as plt
import numpy as np

# Define MLEN and BLEN configurations and their labels
configs = [
    {"MLEN": 64, "BLEN": 64, "label": "MLEN=64, BLEN=64"},
    {"MLEN": 128, "BLEN": 32, "label": "MLEN=128, BLEN=32"},
    {"MLEN": 256, "BLEN": 16, "label": "MLEN=256, BLEN=16"},
    {"MLEN": 512, "BLEN": 8, "label": "MLEN=512, BLEN=8"},
    {"MLEN": 1024, "BLEN": 4, "label": "MLEN=1024, BLEN=4"}
]

batch_sizes = [1, 2, 4, 8, 16]

bar_width = 0.06  # Further reduced bar width
bar_gap = 0.02  # Gap between bars in same group
group_gap = 0  # Gap between groups of bars

# Create x positions with spacing between groups (reduced distance)
x = np.arange(len(batch_sizes)) * (0.5 + group_gap)

# Create two subplots side by side: utilization (left) and execution time (right)
fig, (ax1_ffn, ax2_time) = plt.subplots(1, 2, figsize=(16, 4))

colors = [
    (240/255, 249/255, 232/255),  # light green
    (186/255, 228/255, 188/255),  # green
    (123/255, 204/255, 196/255),  # teal
    (67/255, 162/255, 202/255),   # blue
    (8/255, 104/255, 172/255)     # dark blue
]

# Process each configuration
for idx, cfg in enumerate(configs):
    # FFN data
    ffn_utilizations = []
    execution_times = []  # Execution time in microseconds (cycles / 1GHz)
    
    for bs in batch_sizes:
        # reload model with required settings
        model = attn_model_config(
            MLEN=cfg["MLEN"], BLEN=cfg["BLEN"], VLEN=1024, partitioned_matrix=True,
            model_param_path=model_param_path, batch_size=bs, seq_len=8024, output_token=100, device_num=1
        )
        
        # FFN stage
        ffn_attainable, ffn_theoretical = model._report_ffn_utilization("decode")[0:2]
        ffn_utilization = ffn_attainable / ffn_theoretical if ffn_theoretical != 0 else 0
        ffn_utilizations.append(ffn_utilization)
        
        # Execution time: theoretical cycles / 1GHz = time in seconds, convert to microseconds
        execution_time_us = (ffn_theoretical / 1e9) * 1e6  # Convert to microseconds
        execution_times.append(execution_time_us)

    # Bar positions for configs - with spacing between groups
    pos = x + (idx - len(configs)/2) * (bar_width + bar_gap) + bar_width/2
    
    # Plot FFN utilization (left)
    rects_ffn = ax1_ffn.bar(pos, ffn_utilizations, width=bar_width, label=cfg["label"], color=colors[idx], alpha=0.7)
    
    # Plot execution time (right)
    rects_time = ax2_time.bar(pos, execution_times, width=bar_width, label=cfg["label"], color=colors[idx], alpha=0.7)

# Configure FFN utilization plot (left)
ax1_ffn.set_xlabel("Batch Size", fontsize=18)
ax1_ffn.set_ylabel("Utilization", fontsize=18)
ax1_ffn.tick_params(axis='both', which='major', labelsize=14)
ax1_ffn.set_title("FFN Decode Systolic Array Utilization vs. Batch Size", fontsize=20)
ax1_ffn.set_xticks(x)
ax1_ffn.set_xticklabels([str(bs) for bs in batch_sizes])
ax1_ffn.set_ylim(0, 1)

# Set x-axis limits to show all bars with spacing
total_config_width = len(configs) * (bar_width + bar_gap) - bar_gap
ax1_ffn.set_xlim(x[0] - total_config_width/2 - group_gap/2, x[-1] + total_config_width/2 + group_gap/2)

# Configure execution time plot (right)
ax2_time.set_xlabel("Batch Size", fontsize=18)
ax2_time.set_ylabel("Execution Time (μs)", fontsize=18)
ax2_time.tick_params(axis='both', which='major', labelsize=14)
ax2_time.set_title("FFN Decode Execution Time vs. Batch Size (1GHz)", fontsize=20)
ax2_time.set_xticks(x)
ax2_time.set_xticklabels([str(bs) for bs in batch_sizes])

# Set x-axis limits to show all bars with spacing (same as left plot)
ax2_time.set_xlim(x[0] - total_config_width/2 - group_gap/2, x[-1] + total_config_width/2 + group_gap/2)

# Create shared legend at the bottom in two rows (placed lower)
handles1, labels1 = ax1_ffn.get_legend_handles_labels()
# Create legend with all config labels - transparent with no frame, arranged in 2 rows
# With 5 configs, use ncol=3 to get 2 rows (3 in first row, 2 in second row)
# Push legend lower using bbox_to_anchor
fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=5, fontsize=15, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 0.98])
plt.savefig("decode_FFN_GEMM_Utilization_and_Latency.png", dpi=300, bbox_inches='tight')
